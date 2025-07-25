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

#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/tpch/gen/TpchGen.h"

namespace facebook::velox::connector::tpch {

using facebook::velox::tpch::Table;

namespace {

RowVectorPtr getTpchData(
    Table table,
    size_t maxRows,
    size_t offset,
    double scaleFactor,
    memory::MemoryPool* pool) {
  switch (table) {
    case Table::TBL_PART:
      return velox::tpch::genTpchPart(pool, maxRows, offset, scaleFactor);
    case Table::TBL_SUPPLIER:
      return velox::tpch::genTpchSupplier(pool, maxRows, offset, scaleFactor);
    case Table::TBL_PARTSUPP:
      return velox::tpch::genTpchPartSupp(pool, maxRows, offset, scaleFactor);
    case Table::TBL_CUSTOMER:
      return velox::tpch::genTpchCustomer(pool, maxRows, offset, scaleFactor);
    case Table::TBL_ORDERS:
      return velox::tpch::genTpchOrders(pool, maxRows, offset, scaleFactor);
    case Table::TBL_LINEITEM:
      return velox::tpch::genTpchLineItem(pool, maxRows, offset, scaleFactor);
    case Table::TBL_NATION:
      return velox::tpch::genTpchNation(pool, maxRows, offset, scaleFactor);
    case Table::TBL_REGION:
      return velox::tpch::genTpchRegion(pool, maxRows, offset, scaleFactor);
  }
  return nullptr;
}

} // namespace

std::string TpchTableHandle::toString() const {
  return fmt::format(
      "table: {}, scale factor: {}", toTableName(table_), scaleFactor_);
}

TpchDataSource::TpchDataSource(
    const RowTypePtr& outputType,
    const connector::ConnectorTableHandlePtr& tableHandle,
    const connector::ColumnHandleMap& columnHandles,
    velox::memory::MemoryPool* pool)
    : pool_(pool) {
  auto tpchTableHandle =
      std::dynamic_pointer_cast<const TpchTableHandle>(tableHandle);
  VELOX_CHECK_NOT_NULL(
      tpchTableHandle, "TableHandle must be an instance of TpchTableHandle");
  tpchTable_ = tpchTableHandle->getTable();
  scaleFactor_ = tpchTableHandle->getScaleFactor();
  tpchTableRowCount_ = getRowCount(tpchTable_, scaleFactor_);

  auto tpchTableSchema = getTableSchema(tpchTableHandle->getTable());
  VELOX_CHECK_NOT_NULL(tpchTableSchema, "TpchSchema can't be null.");

  outputColumnMappings_.reserve(outputType->size());

  for (const auto& outputName : outputType->names()) {
    auto it = columnHandles.find(outputName);
    VELOX_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column '{}' on table '{}'",
        outputName,
        toTableName(tpchTable_));

    auto handle = std::dynamic_pointer_cast<const TpchColumnHandle>(it->second);
    VELOX_CHECK_NOT_NULL(
        handle,
        "ColumnHandle must be an instance of TpchColumnHandle "
        "for '{}' on table '{}'",
        it->second->name(),
        toTableName(tpchTable_));

    auto idx = tpchTableSchema->getChildIdxIfExists(handle->name());
    VELOX_CHECK(
        idx != std::nullopt,
        "Column '{}' not found on TPC-H table '{}'.",
        handle->name(),
        toTableName(tpchTable_));
    outputColumnMappings_.emplace_back(*idx);
  }
  outputType_ = outputType;
}

RowVectorPtr TpchDataSource::projectOutputColumns(RowVectorPtr inputVector) {
  std::vector<VectorPtr> children;
  children.reserve(outputColumnMappings_.size());

  for (const auto channel : outputColumnMappings_) {
    children.emplace_back(inputVector->childAt(channel));
  }

  return std::make_shared<RowVector>(
      pool_,
      outputType_,
      BufferPtr(),
      inputVector->size(),
      std::move(children));
}

void TpchDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  VELOX_CHECK_EQ(
      currentSplit_,
      nullptr,
      "Previous split has not been processed yet. Call next() to process the split.");
  currentSplit_ = std::dynamic_pointer_cast<TpchConnectorSplit>(split);
  VELOX_CHECK(currentSplit_, "Wrong type of split for TpchDataSource.");

  // Lineitems is generated based on the row ids of the orders table, so
  // splitOffset_ and splitEnd_ will refer to orders.
  size_t effectiveRowCount = isLineItem()
      ? getRowCount(Table::TBL_ORDERS, scaleFactor_)
      : tpchTableRowCount_;

  size_t partSize = std::ceil(
      static_cast<double>(effectiveRowCount) /
      static_cast<double>(currentSplit_->totalParts));

  splitOffset_ = partSize * currentSplit_->partNumber;
  splitEnd_ = splitOffset_ + partSize;
}

std::optional<RowVectorPtr> TpchDataSource::next(
    uint64_t size,
    velox::ContinueFuture& /*future*/) {
  VELOX_CHECK_NOT_NULL(
      currentSplit_, "No split to process. Call addSplit() first.");

  // LineItems generates records based on orders, so it will generate on
  // average 4 times more records than what is requested. Dividing by 4 so it
  // generates about the right amount of records. Note that the exact amount of
  // lineitems for an order is random (from 1 to 7), so this function may
  // generate slightly more or fewer records than `size`.
  if (isLineItem()) {
    size /= 4;
  }

  size_t maxRows = std::min(size, (splitEnd_ - splitOffset_));
  auto outputVector =
      getTpchData(tpchTable_, maxRows, splitOffset_, scaleFactor_, pool_);

  // If the split is exhausted.
  if (!outputVector || outputVector->size() == 0) {
    currentSplit_ = nullptr;
    return nullptr;
  }

  // splitOffset needs to advance based on maxRows passed to getTpchData(), and
  // not the actual number of returned rows in the output vector, as they are
  // not the same for lineitem.
  splitOffset_ += maxRows;
  completedRows_ += outputVector->size();
  completedBytes_ += outputVector->retainedSize();

  return projectOutputColumns(outputVector);
}

bool TpchDataSource::isLineItem() const {
  return tpchTable_ == Table::TBL_LINEITEM;
}

} // namespace facebook::velox::connector::tpch
