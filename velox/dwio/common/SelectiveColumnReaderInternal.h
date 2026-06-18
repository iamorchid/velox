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

#pragma once

#include "velox/common/base/Portability.h"
#include "velox/dwio/common/ColumnVisitors.h"
#include "velox/dwio/common/DirectDecoder.h"
#include "velox/dwio/common/SelectiveColumnReader.h"
#include "velox/dwio/common/TypeUtils.h"
#include "velox/type/Timestamp.h"
#include "velox/vector/AggregationHook.h"
#include "velox/vector/ConstantVector.h"
#include "velox/vector/DictionaryVector.h"
#include "velox/vector/FlatVector.h"

#include <numeric>
#include <type_traits>

namespace facebook::velox::dwio::common {

/// True for arithmetic types and extended integer types (int128_t, uint128_t).
template <typename T>
inline constexpr bool kIsNumericScalar = std::is_arithmetic_v<T> ||
    std::is_same_v<T, int128_t> || std::is_same_v<T, uint128_t>;

template <typename T>
void SelectiveColumnReader::ensureValuesCapacity(
    vector_size_t numRows,
    bool preserveData /* false */) {
  if (values_ && (isFlatMapValue_ || values_->unique()) &&
      values_->capacity() >=
          BaseVector::byteSize<T>(numRows) + simd::kPadding) {
    return;
  }
  // AlignedBuffer::allocate本身就会考虑simd::kPadding, 这里额外处理
  // simd::kPadding, 感觉是多余的
  auto newValues =
      AlignedBuffer::allocate<T>(numRows + simd::kPadding / sizeof(T), pool_);
  if (preserveData) {
    std::memcpy(
        newValues->template asMutable<char>(), rawValues_, values_->capacity());
  }
  values_ = std::move(newValues);
  rawValues_ = values_->asMutable<char>();
}

template <typename T>
void SelectiveColumnReader::prepareRead(
    int64_t offset,
    const RowSet& rows,
    // incomingNulls包含了父column中rows为nulls的信息. 输入参数rows可以是dense, 
    // 也可以不是; 可以对应null, 也可以不是. 比如rows为[0, 20-30, 50, 200-1000],
    // 则incomingNulls至少覆盖 1000+1 个bits. 
    // 对于orc的情况, 只有incomingNulls中bit为1的row, child column才会对应的值,
    // child column的值可以是null, 也可以不是。
    const uint64_t* incomingNulls) {
  const vector_size_t numRows = rows.back() + 1;

  // 当前读取范围内的所有rows的nulls情况会放入nullsInReadRange_
  readNulls(offset, numRows, incomingNulls);

  // We check for all nulls and no nulls. We expect both calls to
  // bits::isAllSet to fail early in the common case. We could do a
  // single traversal of null bits counting the bits and then compare
  // this to 0 and the total number of rows but this would end up
  // reading more in the mixed case and would not be better in the all
  // (non)-null case.
  if (nullsInReadRange_) {
    const uint64_t* rawNulls = nullsInReadRange_->as<uint64_t>();
    allNull_ = bits::isAllSet(rawNulls, 0, numRows, bits::kNull);
    if (bits::isAllSet(rawNulls, 0, numRows, bits::kNotNull)) {
      nullsInReadRange_ = nullptr;
    }
  }

  innerNonNullRows_.clear();
  outerNonNullRows_.clear();
  outputRows_.clear();
  // Is part of read() and after read returns getValues may be called.
  mayGetValues_ = true;
  numValues_ = 0;
  valueSize_ = sizeof(T);
  inputRows_ = rows;

  // 如果column没有定义filter, 则输出的values个数和inputRows_保持一致.
  // 否则, 输出值对应行号保存到outputRows_中.
  if (scanSpec_->filter() || hasDeletion()) {
    outputRows_.reserve(rows.size());
  }

  ensureValuesCapacity<T>(rows.size());

  if (scanSpec_->keepValues() && !scanSpec_->valueHook()) {
    valueRows_.clear();
    prepareNulls(rows, nullsInReadRange_ != nullptr);
  }
}

template <typename T, typename TVector>
void SelectiveColumnReader::getFlatValues(
    const RowSet& rows,
    VectorPtr* result,
    const TypePtr& type,
    bool isFinal) {
  static_assert(
      std::is_trivially_copyable_v<T> && std::is_trivially_copyable_v<TVector>,
      "T and TVector must be trivially copyable types");

  // When T and TVector differ, both must be numeric scalars. This prevents
  // accidental cross-domain copies such as Timestamp<->int64_t or
  // StringView<->int128_t. Same-size cross-domain conversions (e.g.,
  // int32_t -> float) would be strict-aliasing violations in
  // compactScalarValues; schema validation must reject them before reaching
  // here.
  if constexpr (!std::is_same_v<T, TVector>) {
    static_assert(
        kIsNumericScalar<T> && kIsNumericScalar<TVector>,
        "Cross-type getFlatValues requires both T and TVector to be numeric");
    static_assert(
        sizeof(T) != sizeof(TVector) ||
            std::is_floating_point_v<T> == std::is_floating_point_v<TVector>,
        "Same-size cross-domain conversions (e.g., int32 -> float) are not "
        "supported");
  }
  VELOX_CHECK_NE(valueSize_, kNoValueSize);
  VELOX_CHECK(mayGetValues_);
  if (isFinal) {
    mayGetValues_ = false;
  }

  if (allNull_) {
    if (isFlatMapValue_) {
      if (flatMapValueConstantNullValues_) {
        flatMapValueConstantNullValues_->resize(rows.size());
      } else {
        flatMapValueConstantNullValues_ =
            std::make_shared<ConstantVector<TVector>>(
                pool_, rows.size(), true, type, TVector());
      }
      *result = flatMapValueConstantNullValues_;
    } else {
      *result = std::make_shared<ConstantVector<TVector>>(
          pool_, rows.size(), true, type, TVector());
    }
    return;
  }

  if (valueSize_ == sizeof(TVector)) {
    // 这里之所以需要进行compact, 是因为struct column允许对child column逐级进行filter. 
    // 比如一开始rows为[0, 999], child#1执行filter后, 只有范围为[100, 399]的rows满足需要, 
    // 因此child#1的output rows为[100, 399], 且为这个range的rows准备好了values. 但当
    // 执行child#2执行它的filter后, [100, 399]范围内的rows只有[200, 299]满足, 此时对整体
    // 输出而言, 只应该输出[200, 299]范围内的100个rows. 因此, child#1之前多余的结果需要裁剪
    // 掉, 即对应这里的compact.
    compactScalarValues<TVector, TVector>(rows, isFinal);
  } else if (sizeof(T) >= sizeof(TVector)) {
    compactScalarValues<T, TVector>(rows, isFinal);
  } else {
    upcastScalarValues<T, TVector>(rows);
  }
  valueSize_ = sizeof(TVector);
  if (isFlatMapValue_) {
    if (flatMapValueFlatValues_) {
      auto* flat = flatMapValueFlatValues_->asUnchecked<FlatVector<TVector>>();
      flat->unsafeSetSize(numValues_);
      flat->setNulls(resultNulls());
      flat->unsafeSetValues(values_);
      flat->setStringBuffers(std::move(stringBuffers_));
    } else {
      flatMapValueFlatValues_ = std::make_shared<FlatVector<TVector>>(
          pool_,
          type,
          resultNulls(),
          numValues_,
          values_,
          std::move(stringBuffers_));
    }
    *result = flatMapValueFlatValues_;
  } else {
    *result = std::make_shared<FlatVector<TVector>>(
        pool_,
        type,
        resultNulls(),
        numValues_,
        values_,
        std::move(stringBuffers_));
  }
}

template <>
void SelectiveColumnReader::getFlatValues<int8_t, bool>(
    const RowSet& rows,
    VectorPtr* result,
    const TypePtr& type,
    bool isFinal);

template <typename T, typename TVector>
void SelectiveColumnReader::upcastScalarValues(const RowSet& rows) {
  VELOX_CHECK_LE(rows.size(), numValues_);
  VELOX_CHECK(!rows.empty());
  if (!values_) {
    return;
  }
  VELOX_CHECK_GT(sizeof(TVector), sizeof(T));
  // Since upcast is not going to be a common path, allocate buffer to copy
  // upcasted values to and then copy back to the values buffer.
  std::vector<TVector> buf;
  buf.resize(rows.size());
  T* typedSourceValues = reinterpret_cast<T*>(rawValues_);
  RowSet sourceRows;
  // The row numbers corresponding to elements in 'values_' are in
  // 'valueRows_' if values have been accessed before. Otherwise
  // they are in 'outputRows_' if these are non-empty (there is a
  // filter) and in 'inputRows_' otherwise.
  if (!valueRows_.empty()) {
    sourceRows = valueRows_;
  } else if (!outputRows_.empty()) {
    sourceRows = outputRows_;
  } else {
    sourceRows = inputRows_;
  }
  if (valueRows_.empty()) {
    valueRows_.resize(rows.size());
  }
  vector_size_t rowIndex = 0;
  auto nextRow = rows[rowIndex];
  auto* moveNullsFrom = shouldMoveNulls(rows);
  for (size_t i = 0; i < numValues_; i++) {
    if (sourceRows[i] < nextRow) {
      continue;
    }

    VELOX_DCHECK(sourceRows[i] == nextRow);
    buf[rowIndex] = typedSourceValues[i];
    if (moveNullsFrom && rowIndex != i) {
      bits::setBit(rawResultNulls_, rowIndex, bits::isBitSet(moveNullsFrom, i));
    }
    valueRows_[rowIndex] = nextRow;
    rowIndex++;
    if (rowIndex >= rows.size()) {
      break;
    }
    nextRow = rows[rowIndex];
  }
  ensureValuesCapacity<TVector>(rows.size());
  std::memcpy(rawValues_, buf.data(), rows.size() * sizeof(TVector));
  numValues_ = rows.size();
  valueRows_.resize(numValues_);
  values_->setSize(numValues_ * sizeof(TVector));
}

template <typename T, typename TVector>
void SelectiveColumnReader::compactScalarValues(
    const RowSet& rows,
    bool isFinal) {
  VELOX_CHECK_LE(rows.size(), numValues_);
  VELOX_CHECK(!rows.empty());
  if (!values_ || (rows.size() == numValues_ && sizeof(T) == sizeof(TVector))) {
    if (values_) {
      values_->setSize(numValues_ * sizeof(T));
    }
    return;
  }

  VELOX_CHECK_LE(sizeof(TVector), sizeof(T));
  T* typedSourceValues = reinterpret_cast<T*>(rawValues_);
  TVector* typedDestValues = reinterpret_cast<TVector*>(rawValues_);
  RowSet sourceRows;
  // The row numbers corresponding to elements in 'values_' are in
  // 'valueRows_' if values have been accessed before. Otherwise
  // they are in 'outputRows_' if these are non-empty (there is a
  // filter) and in 'inputRows_' otherwise.
  if (!valueRows_.empty()) {
    sourceRows = valueRows_;
  } else if (!outputRows_.empty()) {
    sourceRows = outputRows_;
  } else {
    sourceRows = inputRows_;
  }
  if (valueRows_.empty()) {
    valueRows_.resize(rows.size());
  }

  vector_size_t rowIndex = 0;
  auto nextRow = rows[rowIndex];
  const auto* moveNullsFrom = shouldMoveNulls(rows);
  for (size_t i = 0; i < numValues_; ++i) {
    if (sourceRows[i] < nextRow) {
      continue;
    }

    VELOX_DCHECK_EQ(sourceRows[i], nextRow);
    typedDestValues[rowIndex] = typedSourceValues[i];
    if (moveNullsFrom && rowIndex != i) {
      bits::setBit(rawResultNulls_, rowIndex, bits::isBitSet(moveNullsFrom, i));
    }
    if (!isFinal) {
      valueRows_[rowIndex] = nextRow;
    }
    ++rowIndex;
    if (rowIndex >= rows.size()) {
      break;
    }
    nextRow = rows[rowIndex];
  }

  numValues_ = rows.size();
  valueRows_.resize(numValues_);
  values_->setSize(numValues_ * sizeof(TVector));
}

template <>
void SelectiveColumnReader::compactScalarValues<bool, bool>(
    const RowSet& rows,
    bool isFinal);

inline int32_t sizeOfIntKind(TypeKind kind) {
  switch (kind) {
    case TypeKind::SMALLINT:
      return 2;
    case TypeKind::INTEGER:
      return 4;
    case TypeKind::BIGINT:
      return 8;
    default:
      VELOX_FAIL("Not an integer TypeKind: {}", static_cast<int>(kind));
  }
}

template <typename T>
void SelectiveColumnReader::filterNulls(
    const RowSet& rows,
    bool isNull,
    bool extractValues) {
  const bool isDense = rows.back() == rows.size() - 1;
  // We decide is (not) null based on 'nullsInReadRange_'. This may be
  // set due to nulls in enclosing structs even if the column itself
  // does not add nulls.
  auto* rawNulls =
      nullsInReadRange_ ? nullsInReadRange_->as<uint64_t>() : nullptr;
  if (isNull) {
    if (!rawNulls) {
      // The stripe has nulls but the current range does not. Nothing matches.
    } else if (isDense) {
      bits::forEachUnsetBit(
          rawNulls, 0, rows.back() + 1, [&](vector_size_t row) {
            addOutputRow(row);
            if (extractValues) {
              addNull<T>();
            }
          });
    } else {
      for (auto row : rows) {
        if (bits::isBitNull(rawNulls, row)) {
          addOutputRow(row);
          if (extractValues) {
            addNull<T>();
          }
        }
      }
    }
    return;
  }

  VELOX_CHECK(
      !extractValues,
      "filterNulls for not null only applies to filter-only case");
  if (!rawNulls) {
    // All pass.
    for (auto row : rows) {
      addOutputRow(row);
    }
  } else if (isDense) {
    bits::forEachSetBit(rawNulls, 0, rows.back() + 1, [&](vector_size_t row) {
      addOutputRow(row);
    });
  } else {
    for (auto row : rows) {
      if (!bits::isBitNull(rawNulls, row)) {
        addOutputRow(row);
      }
    }
  }
}

} // namespace facebook::velox::dwio::common
