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

#include <string>

#include "velox/functions/UDFOutputString.h"
#include "velox/type/StringView.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {
class StringWriter : public UDFOutputString {
 public:
  // Used to initialize top-level strings and allow zero-copy writes.
  StringWriter(FlatVector<StringView>* vector, int32_t offset)
      : vector_(vector), offset_(offset) {}

  // If not initialized for zero-copy write, returns a string to copy into the
  // target vector on commit.
  const std::string& value() const {
    return value_;
  }

  /// Reserve a space for the output string with size of at least newCapacity
  void reserve(size_t newCapacity) override {
    if (newCapacity <= capacity()) {
      return;
    }

    auto* newDataBuffer = vector_->getBufferWithSpace(newCapacity);
    auto actualCapacity = newDataBuffer->capacity() - newDataBuffer->size();

    // Impossible to be the same due to the way the capacity is computed.
    VELOX_DCHECK(dataBuffer_ != newDataBuffer);

    auto newStartAddress =
        newDataBuffer->asMutable<char>() + newDataBuffer->size();

    if (size() != 0) {
      std::memcpy(newStartAddress, data(), size());
    }

    setCapacity(actualCapacity);
    setData(newStartAddress);
    dataBuffer_ = newDataBuffer;
  }

  /// Not called by the UDF Implementation. Should be called at the end to
  /// finalize the allocation and the string writing.
  void finalize() {
    if (!finalized_) {
      VELOX_DCHECK(size() == 0 || data());
      VELOX_USER_CHECK_LE(size(), INT32_MAX);
      if LIKELY (size()) {
        VELOX_CHECK_NOT_NULL(dataBuffer_);
        dataBuffer_->setSize(dataBuffer_->size() + size());
      }
      vector_->setNoCopy(offset_, StringView(data(), size()));
    }
  }

  void prepareForReuse(bool isSet) {
    if (isSet) {
      setCapacity(capacity() - size());
      setData(data() + size());
    }
    resize(0);
    finalized_ = false;
  }

  void finalizeNull() {}

  void setEmpty() {
    static const StringView kEmpty("");
    vector_->setNoCopy(offset_, kEmpty);
    finalized_ = true;
  }

  void setNoCopy(const StringView& value) {
    vector_->setNoCopy(offset_, value);
    finalized_ = true;
  }

  template <typename T>
  void operator+=(const T& input) {
    append(input);
  }

  void operator+=(const char* input) {
    append(std::string_view(input));
  }

  template <typename T>
  void operator=(const T& input) {
    resize(0);
    append(input);
  }

  template <typename T>
  void append(const T& input) {
    VELOX_DCHECK(!finalized_);
    auto oldSize = size();
    resize(this->size() + input.size());
    if (input.size() != 0) {
      VELOX_DCHECK(data());
      VELOX_DCHECK(input.data());
      std::memcpy(data() + oldSize, input.data(), input.size());
    }
  }

  void append(const char* input) {
    append(std::string_view(input));
  }

  template <typename T>
  void copy_from(const T& input) {
    append(input);
  }

  void copy_from(const char* input) {
    append(std::string_view(input));
  }

 private:
  StringWriter() = default;

  bool finalized_{false};

  /// The buffer that the output string uses for its allocation set during
  /// reserve() call
  Buffer* dataBuffer_ = nullptr;

  FlatVector<StringView>* vector_;

  int32_t offset_;

  std::string value_;

  template <typename A, typename B>
  friend struct VectorWriter;
};
} // namespace facebook::velox::exec
