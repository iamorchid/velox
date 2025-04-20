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

#include "velox/common/memory/AllocationPool.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/MemoryAllocator.h"

namespace facebook::velox::memory {

folly::Range<char*> AllocationPool::rangeAt(int32_t index) const {
  if (index < allocations_.size()) {
    auto run = allocations_[index].runAt(0);
    return folly::Range<char*>(
        run.data<char>(),
        run.data<char>() == startOfRun_ ? currentOffset_ : run.numBytes());
  }
  const auto largeIndex = index - allocations_.size();
  if (largeIndex < largeAllocations_.size()) {
    auto range = largeAllocations_[largeIndex].hugePageRange().value();
    if (range.data() == startOfRun_) {
      return folly::Range<char*>(range.data(), currentOffset_);
    }
    return range;
  }
  VELOX_FAIL("Out of range index for rangeAt(): {}", index);
}

void AllocationPool::clear() {
  allocations_.clear();
  largeAllocations_.clear();
  startOfRun_ = nullptr;
  bytesInRun_ = 0;
  currentOffset_ = 0;
  usedBytes_ = 0;
}

char* AllocationPool::allocateFixed(uint64_t bytes, int32_t alignment) {
  VELOX_CHECK_GT(bytes, 0, "Cannot allocate zero bytes");
  if (freeAddressableBytes() >= bytes && alignment == 1) {
    auto* result = startOfRun_ + currentOffset_;
    maybeGrowLastAllocation(bytes);
    return result;
  }
  VELOX_CHECK_EQ(
      __builtin_popcount(alignment), 1, "Alignment can only be power of 2");

  auto numPages = AllocationTraits::numPages(bytes + alignment - 1);

  if (freeAddressableBytes() == 0) {
    newRunImpl(numPages);
  } else {
    auto alignedBytes = bytes + alignmentPadding(firstFreeInRun(), alignment);
    if (freeAddressableBytes() < alignedBytes) {
      newRunImpl(numPages);
    }
  }
  currentOffset_ += alignmentPadding(firstFreeInRun(), alignment);
  VELOX_CHECK_LE(bytes + currentOffset_, bytesInRun_);
  auto* result = startOfRun_ + currentOffset_;
  VELOX_CHECK_EQ(reinterpret_cast<uintptr_t>(result) % alignment, 0);
  maybeGrowLastAllocation(bytes);
  return result;
}

void AllocationPool::maybeGrowLastAllocation(uint64_t bytesRequested) {
  const auto updateOffset = currentOffset_ + bytesRequested;
  if (updateOffset > endOfReservedRun()) {
    VELOX_CHECK_GT(bytesInRun_, AllocationTraits::kHugePageSize);
    const auto bytesToReserve = bits::roundUp(
        updateOffset - endOfReservedRun(), AllocationTraits::kHugePageSize);
    // 这里会增加ContiguousAllocation的size_大小, 同时为新增的大小进行reserve操作
    largeAllocations_.back().grow(AllocationTraits::numPages(bytesToReserve));
    usedBytes_ += bytesToReserve;
  }
  // Only update currentOffset_ once it points to valid data.
  currentOffset_ = updateOffset;
}

void AllocationPool::newRunImpl(MachinePageCount numPages) {
  if (usedBytes_ >= hugePageThreshold_ ||
      numPages > pool_->sizeClasses().back()) {
    // At least 16 huge pages, no more than kMaxMmapBytes. The next is
    // double the previous. Because the previous is a hair under the
    // power of two because of fractional pages at ends of allocation,
    // add an extra huge page size.
    int64_t nextSize = std::min(
        kMaxMmapBytes,
        std::max<int64_t>(
            16 * AllocationTraits::kHugePageSize,
            bits::nextPowerOfTwo(
                usedBytes_ + AllocationTraits::kHugePageSize)));
    // Round 'numPages' to no of pages in huge page. Allocating this plus an
    // extra huge page guarantees that 'numPages' worth of contiguous aligned
    // huge pages will be found in the allocation.
    numPages = bits::roundUp(numPages, AllocationTraits::numPagesInHugePage());
    if (AllocationTraits::pageBytes(numPages) +
            AllocationTraits::kHugePageSize >
        nextSize) {
      // Extra large single request.
      nextSize = AllocationTraits::pageBytes(numPages) +
          AllocationTraits::kHugePageSize;
    }

    ContiguousAllocation largeAlloc;
    const MachinePageCount pagesToAlloc =
        AllocationTraits::numPagesInHugePage();
    pool_->allocateContiguous(
        pagesToAlloc, largeAlloc, AllocationTraits::numPages(nextSize));

    //
    // 对于ContiguousAllocation(即对应大块内存), 虽然请求的内存大小为pagesToAlloc, 但
    // 实际上会向os分配更大的虚拟内存, 此时os并没有为这些内存分配物理内存, 同时MemoryPool
    // 仅仅为pagesToAlloc大小的内存进行过reserve操作, 调用方只允许访问这块reserve过的内存
    // 区域. 
    //
    // 当调用方需下次要继续分配内存时, 此时可以不需要再次向os申请虚拟内存了, 而可以直接使用
    // 之前预分配好的虚拟内存, 但调用方在真正使用这些内存之前, 也必须进行reserve操作, 避
    // 免超过它的内存quota (见maybeGrowLastAllocation).
    // 
    auto range = largeAlloc.hugePageRange().value(); // 虚拟内存的实际range
    startOfRun_ = range.data();                      // 虚拟内存的start
    bytesInRun_ = range.size();                      // 虚拟内存的实际大小(比如pagesToAlloc大)
    currentOffset_ = 0;

    // ContiguousAllocation中的size对应的是pagesToAlloc(即512个page或2MB字节), 
    // maxSize对应的是虚拟内存的实际大小.
    largeAllocations_.emplace_back(std::move(largeAlloc));

    // 此时, memory pool仅仅为numPages的内存进行过reserve操作
    usedBytes_ += AllocationTraits::pageBytes(pagesToAlloc);
    return;
  }

  Allocation allocation;
  auto roundedPages = std::max<int32_t>(kMinPages, numPages);
  pool_->allocateNonContiguous(roundedPages, allocation, roundedPages);

  // 上面allocateNonContiguous是怎么保证PageRun只有一个的? 通过第三个参数
  // 限定了分配的最小PageRun包好的pages个数 (只需要一个PageRun).
  VELOX_CHECK_EQ(allocation.numRuns(), 1);

  startOfRun_ = allocation.runAt(0).data<char>();
  bytesInRun_ = allocation.runAt(0).numBytes();
  currentOffset_ = 0;
  allocations_.push_back(std::move(allocation));
  usedBytes_ += bytesInRun_;
}

void AllocationPool::newRun(int64_t preferredSize) {
  newRunImpl(AllocationTraits::numPages(preferredSize));
}

} // namespace facebook::velox::memory
