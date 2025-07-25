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

#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateInfo.h"
#include "velox/exec/AggregationMasks.h"
#include "velox/exec/DistinctAggregations.h"
#include "velox/exec/Operator.h"
#include "velox/exec/SortedAggregations.h"

namespace facebook::velox::exec {

class RowContainer;

class StreamingAggregation : public Operator {
 public:
  StreamingAggregation(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::AggregationNode>& aggregationNode);

  void initialize() override;

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  bool needsInput() const override {
    // We don't need input if the first group is ready to output which has mixed
    // input sources across streaming input batches.
    return true;
  }

  bool startDrain() override;

  BlockingReason isBlocked(ContinueFuture* /* unused */) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

  void close() override;

 private:
  void maybeFinishDrain();

  // Returns the rows to aggregate with masking applied if applicable.
  const SelectivityVector& getSelectivityVector(size_t aggregateIndex) const;

  // Allocate new group or re-use previously allocated group that has been fully
  // calculated and included in the output.
  char* startNewGroup(vector_size_t index);

  // Write grouping keys from the specified input row into specified group.
  void storeKeys(char* group, vector_size_t index);

  // Populate output_ vector using specified number of groups from the beginning
  // of the groups_ vector.
  RowVectorPtr createOutput(size_t numGroups);

  // Assign input rows to groups based on values of the grouping keys. Store the
  // assignments in inputGroups_. Returns true if there is input rows have been
  // assigned to the previously last group.
  bool assignGroups();

  // Add input data to accumulators.
  void evaluateAggregates();

  // Initialize the new groups calculated through current and previous groups.
  void initializeNewGroups(size_t numPrevGroups);

  // Create accumulators and RowContainer for aggregations.
  std::unique_ptr<RowContainer> makeRowContainer(
      const std::vector<TypePtr>& groupingKeyTypes);

  // Initialize the aggregations setting allocator and offsets.
  void initializeAggregates(uint32_t numKeys);

  // Maximum number of rows in the output batch.
  const vector_size_t maxOutputBatchSize_;

  // Maximum number of rows in the output batch.
  const vector_size_t minOutputBatchSize_;

  // Used at initialize() and gets reset() afterward.
  std::shared_ptr<const core::AggregationNode> aggregationNode_;

  const core::AggregationNode::Step step_;

  std::vector<column_index_t> groupingKeys_;
  std::vector<AggregateInfo> aggregates_;
  std::unique_ptr<SortedAggregations> sortedAggregations_;
  std::vector<std::unique_ptr<DistinctAggregations>> distinctAggregations_;
  std::unique_ptr<AggregationMasks> masks_;
  std::vector<DecodedVector> decodedKeys_;

  // Storage of grouping keys and accumulators.
  std::unique_ptr<RowContainer> rows_;

  // Previous input vector. Used to compare grouping keys for groups which span
  // batches.
  RowVectorPtr prevInput_;

  // Unique groups.
  std::vector<char*> groups_;

  // Number of active entries at the beginning of the groups_ vector. The
  // remaining entries are re-usable.
  size_t numGroups_{0};

  // If true, we want to output the first group which has inputs across
  // different batches. Hence the next output could only contain the input from
  // a single streaming input batch. This is used to help avoid data copy in
  // streaming aggregation function processing which is only applicable if all
  // the sources are from the same input batch.
  //
  // NOTE: the streaming aggregation operator must have at-least more than one
  // groups in this case. Also we only enable this optimization if
  // 'minOutputBatchSize_' is set to one for eagerly streaming output producing.
  bool outputFirstGroup_{false};

  // Reusable memory.

  // Pointers to groups for all input rows.
  std::vector<char*> inputGroups_;

  // Indices into `groups` indicating the row after last row of each group.  The
  // last element of this is the total size of input.
  std::vector<vector_size_t> groupBoundaries_;

  // A subset of input rows to evaluate the aggregate function on. Rows
  // where aggregation mask is false are excluded.
  SelectivityVector inputRows_;
};

} // namespace facebook::velox::exec
