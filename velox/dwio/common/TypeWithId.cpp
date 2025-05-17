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

#include "velox/dwio/common/TypeWithId.h"

#include "velox/dwio/common/exception/Exception.h"

namespace facebook::velox::dwio::common {

using velox::Type;
using velox::TypeKind;

namespace {
std::vector<std::shared_ptr<const TypeWithId>> toShared(
    std::vector<std::unique_ptr<TypeWithId>> nodes) {
  std::vector<std::shared_ptr<const TypeWithId>> result;
  result.reserve(nodes.size());
  for (auto&& node : nodes) {
    result.emplace_back(std::move(node));
  }
  return result;
}
} // namespace

TypeWithId::TypeWithId(
    std::shared_ptr<const Type> type,
    std::vector<std::unique_ptr<TypeWithId>>&& children,
    uint32_t id,
    uint32_t maxId,
    uint32_t column)
    : type_{std::move(type)},
      parent_{nullptr},
      id_{id},
      maxId_{maxId},
      column_{column},
      children_{toShared(std::move(children))} {
  for (auto& child : children_) {
    if (child) {
      const_cast<const TypeWithId*&>(child->parent_) = this;
    }
  }
}

std::unique_ptr<TypeWithId> TypeWithId::create(
    const std::shared_ptr<const Type>& root,
    uint32_t next) {
  return create(root, next, 0);
}

namespace {

int countNodes(const TypePtr& type) {
  int count = 1;
  for (auto& child : *type) {
    count += countNodes(child);
  }
  return count;
}

} // namespace

std::unique_ptr<TypeWithId> TypeWithId::create(
    const RowTypePtr& type,
    const velox::common::ScanSpec& spec) {
  uint32_t next = 1;
  std::vector<std::unique_ptr<TypeWithId>> children(type->size());
  for (int i = 0, size = type->size(); i < size; ++i) {
    //
    // type对应底层文件数据本身的schema, spec包含了上层需要读取那些列的信息.
    // 比如type为{f1:int,f2:{f2_1:int,f2_2:bigint},f3:{f3_1:int,f3_2:{f3_2_1:int,f3_2_2:varchar}}}, 
    // spec为{f2.f2_1, f3.f3_2.f3_2_1}, 则这里可以直接跳过f1, 但f2和f3还是会加载完整child定义, 而不仅仅是
    // spec中定义的child. 因为只有加载完整定义, 才能确定spec中请求字段的column ID (根据orc type介绍可以知道, 
    // column ID是按照前序遍历顺序递增的).
    //
    // 另外, 加载orc中的column stream数据时, 则会按照spec的具体要求加载, 具体可以参考:
    // SelectiveStructColumnReader::SelectiveStructColumnReader(...)
    //
    auto* childSpec = spec.childByName(type->nameOf(i));
    if (childSpec && !childSpec->isConstant()) {
      children[i] = create(type->childAt(i), next, i);
    } else {
      // next对应的就是type定义中, 下一个column的node ID.
      next += countNodes(type->childAt(i));
    }
  }
  return std::make_unique<TypeWithId>(
      type, std::move(children), 0, next - 1, 0);
}

uint32_t TypeWithId::size() const {
  return children_.size();
}

const std::shared_ptr<const TypeWithId>& TypeWithId::childAt(
    uint32_t idx) const {
  return children_.at(idx);
}

std::unique_ptr<TypeWithId> TypeWithId::create(
    const std::shared_ptr<const Type>& type,
    uint32_t& next,
    uint32_t column) {
  DWIO_ENSURE_NOT_NULL(type);
  const uint32_t myId = next++;
  std::vector<std::unique_ptr<TypeWithId>> children;
  children.reserve(type->size());
  auto offset = 0;
  for (const auto& child : *type) {
    children.emplace_back(create(
        child,
        next,
        (myId == 0 && type->kind() == TypeKind::ROW) ? offset++ : column));
  }
  const uint32_t maxId = next - 1;
  return std::make_unique<TypeWithId>(
      type, std::move(children), myId, maxId, column);
}

std::string TypeWithId::fullName() const {
  std::vector<std::string> path;
  auto* child = this;
  while (auto* parent = child->parent_) {
    switch (parent->type()->kind()) {
      case TypeKind::ROW: {
        auto& siblings = parent->children_;
        bool found = false;
        for (int i = 0; i < siblings.size(); ++i) {
          if (siblings[i].get() == child) {
            path.push_back('.' + parent->type()->asRow().nameOf(i));
            found = true;
            break;
          }
        }
        if (!found) {
          VELOX_FAIL(
              "Child {} not found in parent {}",
              child->type()->toString(),
              parent->type()->toString());
        }
        break;
      }
      case TypeKind::ARRAY:
        break;
      case TypeKind::MAP:
        if (child == parent->children_.at(0).get()) {
          path.push_back(".<keys>");
        } else {
          VELOX_CHECK(child == parent->children_.at(1).get());
          path.push_back(".<values>");
        }
        break;
      default:
        VELOX_UNREACHABLE();
    }
    child = parent;
  }
  std::string ans;
  for (int i = path.size() - 1; i >= 0; --i) {
    if (i == path.size() - 1) {
      VELOX_CHECK_EQ(path[i][0], '.');
      ans += path[i].substr(1);
    } else {
      ans += path[i];
    }
  }
  return ans;
}

} // namespace facebook::velox::dwio::common
