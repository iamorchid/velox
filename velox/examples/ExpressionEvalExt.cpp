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

#include "velox/common/memory/Memory.h"
#include "velox/functions/Udf.h"
#include "velox/type/Type.h"
#include "velox/type/Variant.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook::velox;

/// This file contains a step-by-step usage example of Velox's expression
/// evaluation engine.
///
/// It shows how to register a simple function and describes all the steps
/// required to create the appropriate query and expression structures, a simple
/// expression tree, an input batch of data, and execute the expression over it.

// First, define a toy function that multiplies the input argument by two.
//
// Check `velox/docs/develop/scalar-functions.rst` for more documentation on how
// to build scalar functions.
template <typename T>
struct TimesTwoFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(int64_t& out, const TInput& a) {
    out = a * 2;
  }
};

template <typename T>
struct ArraySumFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      int64_t& output,
      const arg_type<Array<int64_t>>& array) {
    output = 0;
    for (const auto& element : array) {
      if (element.has_value()) {
        output += element.value();
      }
    }
  }
};

template <typename T>
struct VariadicSumFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& output,
      const arg_type<Variadic<int64_t>>& inputs) {
    for (const auto& input : inputs) {
      if (input.has_value()) {
        output += input.value();
      }
    }

    return true;
  }
};

// 参考velox/vector/tests/utils/VectorMaker.h中的arrayVectorNullableImpl
template <typename T>
ArrayVectorPtr createArrayVector(
    memory::MemoryPool* pool,
    const TypePtr& type,
    const std::vector<std::vector<T>>& data) {
  VELOX_CHECK(type->isArray(), "Type must be an array: {}", type->toString());

  vector_size_t size = data.size();
  BufferPtr offsets = AlignedBuffer::allocate<vector_size_t>(size, pool);
  BufferPtr sizes = AlignedBuffer::allocate<vector_size_t>(size, pool);
  BufferPtr nulls = AlignedBuffer::allocate<uint64_t>(size, pool);

  auto rawOffsets = offsets->asMutable<vector_size_t>();
  auto rawSizes = sizes->asMutable<vector_size_t>();
  auto rawNulls = nulls->asMutable<uint64_t>();
  bits::fillBits(rawNulls, 0, size, pool);

  // Count number of elements.
  vector_size_t numElements = 0;
  vector_size_t indexPtr = 0;
  for (const auto& array : data) {
    numElements += array.size();
    indexPtr++;
  }

  using V = typename CppToType<T>::NativeType;

  // Create the underlying flat vector.
  auto flatVector =
      BaseVector::create<FlatVector<V>>(type->childAt(0), numElements, pool);
  vector_size_t currentIdx = 0;

  for (const auto& array : data) {
    *rawOffsets++ = currentIdx;
    *rawSizes++ += array.size();

    for (auto element : array) {
      flatVector->set(currentIdx, V(element));
      ++currentIdx;
    }
  }

  return std::make_shared<ArrayVector>(
      pool, type, nulls, size, offsets, sizes, flatVector);
}

template <TypeKind kind>
static VectorPtr createScalaVector(
    const TypePtr& type,
    vector_size_t size,
    memory::MemoryPool* pool,
    int baseValue) {
  using T = typename TypeTraits<kind>::NativeType;

  auto flatVector = BaseVector::create<FlatVector<T>>(type, size, pool);

  if constexpr (std::is_integral_v<T>) {
    auto rawValues = flatVector->mutableRawValues();
    std::iota(rawValues, rawValues + size, baseValue);
  }

  return flatVector;
}

int main(int argc, char** argv) {
  // Register the function defined above. The first template parameter is the
  // class that implements the `call()` function (or one of its variations), the
  // second template parameter is the function return type, followed by the list
  // of function input parameters.
  //
  // This function takes as an argument a list of aliases for the function being
  // registered.
  registerFunction<TimesTwoFunction, int64_t, int64_t>({"times_two"});
  registerFunction<TimesTwoFunction, int64_t, int32_t>({"times_two"});
  registerFunction<TimesTwoFunction, int64_t, int16_t>({"times_two"});
  registerFunction<ArraySumFunction, int64_t, Array<int64_t>>({"array_sum"});
  registerFunction<VariadicSumFunction, int64_t, Variadic<int64_t>>(
      {"variadic_sum"});

  memory::MemoryManager::initialize({});

  // First of all, executing an expression in Velox will require us to create a
  // query context, a memory pool, and an execution context.
  //
  // QueryCtx holds the metadata and configuration associated with a
  // particular query. This is shared between all threads of execution
  // for the same query (one object per query).
  auto queryCtx = std::make_shared<core::QueryCtx>();

  // ExecCtx holds structures associated with a single thread of execution
  // (one per thread). Each thread of execution requires a scoped memory pool,
  // which is where allocations from this thread will be made. When required, a
  // pointer to this pool can be obtained using execCtx.pool().
  //
  // Optionally, one can control the per-thread memory cap by passing it as an
  // argument to add() - no limit by default.
  auto pool = memory::memoryManager()->addLeafPool();
  core::ExecCtx execCtx{pool.get(), queryCtx.get()};

  auto inputRowType = ROW({
      {"my_col_int16", SMALLINT()},
      {"my_col_int64", BIGINT()},
      {"my_array", ARRAY(BIGINT())},
  });
  const size_t vectorSize = 10;

  std::vector<core::TypedExprPtr> exprs;
  std::vector<VectorPtr> columns;

  {
    for (auto index : std::vector<int>{0, 1}) {
      auto type = inputRowType->childAt(index);
      auto flatVector = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createScalaVector,
          type->kind(),
          type,
          vectorSize,
          execCtx.pool(),
          index * vectorSize);
      columns.emplace_back(std::move(flatVector));

      auto fieldExpr = std::make_shared<core::FieldAccessTypedExpr>(
          type, inputRowType->nameOf(index));

      // velox会根据input的类型自动选择合适的事例化的模版函数
      auto exprPtr = std::make_shared<core::CallTypedExpr>(
          BIGINT(), std::vector<core::TypedExprPtr>{fieldExpr}, "times_two");

      exprs.emplace_back(std::move(exprPtr));
    }
  }

  {
    auto columnType = inputRowType->childAt(2);
    std::vector<std::vector<int64_t>> arrayDataBigInt = {
        {},
        {10},
        {20, 1},
        {},
        {0, 1, 2, 4},
        {99, 98},
        {101},
        {100, 200, 300},
        {},
        {},
    };
    auto arrayVector =
        createArrayVector(pool.get(), columnType, arrayDataBigInt);
    columns.emplace_back(std::move(arrayVector));

    auto fieldExpr =
        std::make_shared<core::FieldAccessTypedExpr>(columnType, "my_array");
    auto exprPtr = std::make_shared<core::CallTypedExpr>(
        BIGINT(), std::vector<core::TypedExprPtr>{fieldExpr}, "array_sum");
    exprs.emplace_back(std::move(exprPtr));
  }

  {
    std::vector<core::TypedExprPtr> inputExprs = exprs;
    auto constExpr = std::make_shared<core::ConstantTypedExpr>(
        BIGINT(), variant(static_cast<int64_t>(100)));
    inputExprs.emplace_back(std::move(constExpr));

    auto exprPtr = std::make_shared<core::CallTypedExpr>(
        BIGINT(), inputExprs, "variadic_sum");
    exprs.emplace_back(std::move(exprPtr));
  }

  // Lastly, ExprSet contains the main expression evaluation logic. It takes a
  // vector of expression trees (if there are multiple expressions to be
  // evaluated). ExprSet will output one column per input exprTree. It also
  // takes the execution context associated with the current thread of
  // execution.
  exec::ExprSet exprSet(exprs, &execCtx);

  // Then, let's wrap the generated flatVector in a RowVector:
  auto rowVector = std::make_shared<RowVector>(
      execCtx.pool(), // pool where allocations will be made.
      inputRowType, // input row type (defined above).
      BufferPtr(nullptr), // no nulls for this example.
      vectorSize, // length of the vectors.
      std::move(columns)); // the input vector data.

  // Now we move to the actual execution.
  //
  // We first create a vector of VectorPtrs to hold the expression results.
  // (ExprSet outputs one vector per input expression - in this case, 1). The
  // output vector will be allocated internally by ExprSet, so we just need to
  // have a single null VectorPtr in this std::vector.
  std::vector<VectorPtr> result{nullptr};

  // Next, we create an input selectivity vector that controls the visibility
  // of records from the input RowVector. In this case we don't want to filter
  // out any rows, so just create a selectivity vector with all bits set.
  SelectivityVector rows{vectorSize};

  // Before execution we need to create one last structure - EvalCtx - which
  // holds context about the expression evaluation of this particular batch.
  // ExprSets can be reused by the same expression over multiple batches, but we
  // need one EvalCtx per RowVector.
  exec::EvalCtx evalCtx(&execCtx, &exprSet, rowVector.get());

  // Voila! Here we do the actual evaluation. When this function returns, the
  // output vectors will be available in the results vector. Note that ExprSet's
  // logic is synchronous and single threaded.
  exprSet.eval(rows, evalCtx, result);

  // Print the output vector, just for fun:
  for (auto& outputVector : result) {
    LOG(INFO) << "------------------------";
    for (vector_size_t i = 0; i < outputVector->size(); ++i) {
      LOG(INFO) << outputVector->toString(i);
    }
  }

  // Lastly, remember that all allocations are associated with the scoped pool
  // created in the beginning, and moved to ExecCtx. Once ExecCtx dies, it
  // destructs the pool which will deallocate all memory associated with it, so
  // be mindful about the object lifetime!
  //
  // (in this example this is safe since ExecCtx will be destructed last).
  return 0;
}
