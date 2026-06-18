/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/dwio/dwrf/reader/DwrfData.h"

#include "velox/dwio/common/BufferUtil.h"

namespace facebook::velox::dwrf {

DwrfData::DwrfData(
    std::shared_ptr<const dwio::common::TypeWithId> fileType,
    StripeStreams& stripe,
    const StreamLabels& streamLabels,
    FlatMapContext flatMapContext)
    : memoryPool_(stripe.getMemoryPool()),
      fileType_(std::move(fileType)),
      flatMapContext_(std::move(flatMapContext)),
      stripeRows_{stripe.stripeRows()},
      rowsPerRowGroup_{stripe.rowsPerRowGroup()} {
  EncodingKey encodingKey{fileType_->id(), flatMapContext_.sequence};
  std::unique_ptr<dwio::common::SeekableInputStream> stream = stripe.getStream(
      StripeStreamsUtil::getStreamForKind(
          stripe,
          encodingKey,
          proto::Stream_Kind_PRESENT,
          proto::orc::Stream_Kind_PRESENT),
      streamLabels.label(),
      false /* 允许不存在 */);
  if (stream) {
    notNullDecoder_ = createBooleanRleDecoder(std::move(stream), encodingKey);
  }

  // We always initialize indexStream_ because indices are needed as soon as
  // there is a single filter that can trigger row group skips anywhere in the
  // reader tree. This is not known at construct time because the first filter
  // can come from a hash join or other run time push-down.
  indexStream_ = stripe.getStream(
      StripeStreamsUtil::getStreamForKind(
          stripe,
          encodingKey,
          proto::Stream_Kind_ROW_INDEX,
          proto::orc::Stream_Kind_ROW_INDEX),
      streamLabels.label(),
      false /* 允许不存在 */);
}

uint64_t DwrfData::skipNulls(uint64_t numValues, bool /*nullsOnly*/) {
  if (!notNullDecoder_ && !flatMapContext_.inMapDecoder) {
    return numValues;
  }

  // Page through the values that we want to skip and count how many are
  // non-null.
  constexpr uint64_t kBufferBytes = 1024;
  constexpr auto kBitCount = kBufferBytes * 8;
  const auto countNulls = [](ByteRleDecoder& decoder, size_t size) {
    char buffer[kBufferBytes];
    decoder.next(buffer, size, nullptr);
    return bits::countNulls(reinterpret_cast<const uint64_t*>(buffer), 0, size);
  };

  uint64_t remaining = numValues;
  while (remaining > 0) {
    const uint64_t chunkSize = std::min(remaining, kBitCount);
    uint64_t nullCount{0};
    if (flatMapContext_.inMapDecoder != nullptr) {
      nullCount = countNulls(*flatMapContext_.inMapDecoder, chunkSize);
      if (notNullDecoder_) {
        if (const auto inMapSize = chunkSize - nullCount; inMapSize > 0) {
          nullCount += countNulls(*notNullDecoder_, inMapSize);
        }
      }
    } else {
      nullCount = countNulls(*notNullDecoder_, chunkSize);
    }
    remaining -= chunkSize;
    numValues -= nullCount;
  }
  return numValues;
}

void DwrfData::ensureRowGroupIndex() {
  VELOX_CHECK(index_ || indexStream_, "Reader needs to have an index stream");
  if (indexStream_) {
    index_ = ProtoUtils::readProto<proto::RowIndex>(std::move(indexStream_));
  }
}

dwio::common::PositionProvider DwrfData::seekToRowGroup(int64_t index) {
  ensureRowGroupIndex();
  VELOX_CHECK_LT(index, index_->entry_size(), "RowGroup index is corrupted");

  positionsHolder_ = toPositionsInner(index_->entry(index));
  dwio::common::PositionProvider positionProvider(positionsHolder_);
  if (flatMapContext_.inMapDecoder) {
    flatMapContext_.inMapDecoder->seekToRowGroup(positionProvider);
  }

  if (notNullDecoder_) {
    notNullDecoder_->seekToRowGroup(positionProvider);
  }

  return positionProvider;
}

///
/// incomingNulls可以理解为来自parent column(比如struct)的nulls. 而child column
/// 只会处理parent nulls中bits为1的行(即不为null的行). 比如incomingNulls为[0010 1100]
/// 且 numValues 为8, 那么child只需要读取next 3 rows的nulls情况, 并对incomingNulls中
/// 为1的bits进行替换, 替换后的nulls作为child column的null bits.
///
void DwrfData::readNulls(
    vector_size_t numValues,
    const uint64_t* incomingNulls,
    BufferPtr& nulls,
    bool /*nullsOnly*/) {
  if (!notNullDecoder_ && !flatMapContext_.inMapDecoder && !incomingNulls) {
    nulls = nullptr;
    return;
  }

  //
  // 这里的numValues对应的并不是要读取的rows数量, 比如要读取的rows为[0, 10, 100, 200], 则
  // numValues对应的值为 200+1, 即需要维护读取范围内的所有bits. child column需要从PRESENT
  // stream读取的bits数量由incomingNulls决定, 即只需要为incomingNulls中bit为1的row读取对
  // 应的bits, 并将这些bits scatter到incomingNulls中为1的位置来作为child column的全局nulls. 
  // 之所以需要这么做, 是因为orc的parent column的值为null时, 它的null信息不回反馈到child
  // column的PRESENT stream中. 有了全局nulls信息后, child column读取完row#100对应的值后, 
  // 再读取row#200时, 就知道需要从DATA stream中跳过多少个值了 (bit为1的row才会有值).
  //
  const auto numBytes = bits::nbytes(numValues);
  if (!nulls || nulls->capacity() < numBytes) {
    nulls = AlignedBuffer::allocate<char>(numBytes, &memoryPool_);
  }
  nulls->setSize(numBytes);

  auto* nullsPtr = nulls->asMutable<char>();
  if (flatMapContext_.inMapDecoder) {
    if (notNullDecoder_) {
      dwio::common::ensureCapacity<char>(inMap_, numBytes, &memoryPool_);
      flatMapContext_.inMapDecoder->next(
          inMap_->asMutable<char>(), numValues, incomingNulls);
      notNullDecoder_->next(nullsPtr, numValues, inMap_->as<uint64_t>());
    } else {
      flatMapContext_.inMapDecoder->next(nullsPtr, numValues, incomingNulls);
      inMap_ = nulls;
    }
  } else if (notNullDecoder_) {
    notNullDecoder_->next(nullsPtr, numValues, incomingNulls);
  } else {
    ::memcpy(nullsPtr, incomingNulls, numBytes);
  }
}

void DwrfData::filterRowGroups(
    const common::ScanSpec& scanSpec,
    uint64_t rowGroupSize,
    const dwio::common::StatsContext& writerContext,
    FilterRowGroupsResult& result) {
  // 并不是所有的column都会定义index stream, 比如struct类型的column,
  // 就不会定义index stream.
  if (!index_ && !indexStream_) {
    return;
  }

  // 如果index_不存在, 则基于indexStream_来实例话RowIndex
  ensureRowGroupIndex();

  auto* filter = scanSpec.filter();
  auto* dwrfContext = reinterpret_cast<const StatsContext*>(&writerContext);
  result.totalCount = std::max(result.totalCount, index_->entry_size());
  const auto nwords = bits::nwords(result.totalCount);
  if (result.filterResult.size() < nwords) {
    result.filterResult.resize(nwords);
  }

  const auto metadataFiltersStartIndex = result.metadataFilterResults.size();
  for (int i = 0; i < scanSpec.numMetadataFilters(); ++i) {
    result.metadataFilterResults.emplace_back(
        scanSpec.metadataFilterNodeAt(i), std::vector<uint64_t>(nwords));
  }

  for (auto i = 0; i < index_->entry_size(); ++i) {
    const auto& entry = index_->entry(i);
    const auto columnStats = buildColumnStatisticsFromProto(
        ColumnStatisticsWrapper(&entry.statistics()), *dwrfContext);
    if (filter &&
        !testFilter(
            filter, columnStats.get(), rowGroupSize, fileType_->type())) {
      VLOG(1) << "Drop stride " << i << " on " << scanSpec.toString();
      bits::setBit(result.filterResult.data(), i);
      continue;
    }

    for (int j = 0; j < scanSpec.numMetadataFilters(); ++j) {
      auto* metadataFilter = scanSpec.metadataFilterAt(j);
      if (!testFilter(
              metadataFilter,
              columnStats.get(),
              rowGroupSize,
              fileType_->type())) {
        bits::setBit(
            result.metadataFilterResults[metadataFiltersStartIndex + j]
                .second.data(),
            i);
      }
    }
  }
}

} // namespace facebook::velox::dwrf
