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
#include "velox/expression/DecodedArgs.h"
#include "velox/expression/EvalCtx.h"
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions {
namespace {

template <bool IsNotNULL>
class IsNullFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /*outputType*/,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    auto* arg = args[0].get();
    auto* pool = context.pool();
    if (arg->isConstantEncoding()) {
      bool isNull = arg->isNullAt(rows.begin());
      auto localResult = BaseVector::createConstant(
          BOOLEAN(), IsNotNULL ? !isNull : isNull, rows.end(), pool);
      context.moveOrCopyResult(localResult, rows, result);
      return;
    }

    if (!arg->mayHaveNulls()) {
      // No nulls.
      auto localResult = BaseVector::createConstant(
          BOOLEAN(), IsNotNULL ? true : false, rows.end(), pool);
      context.moveOrCopyResult(localResult, rows, result);
      return;
    }

    BufferPtr isNull;
    if (arg->isFlatEncoding()) {
      if constexpr (IsNotNULL) {
        isNull = arg->nulls();
      } else {
        isNull = AlignedBuffer::allocate<bool>(rows.end(), pool);
        memcpy(
            isNull->asMutable<int64_t>(),
            arg->rawNulls(),
            bits::nbytes(rows.end()));
        bits::negate(isNull->asMutable<uint64_t>(), rows.end());
      }
    } else {
      exec::DecodedArgs decodedArgs(rows, args, context);

      isNull = AlignedBuffer::allocate<bool>(rows.end(), pool);
      memcpy(
          isNull->asMutable<int64_t>(),
          decodedArgs.at(0)->nulls(&rows),
          bits::nbytes(rows.end()));

      if (!IsNotNULL) {
        bits::negate(isNull->asMutable<uint64_t>(), rows.end());
      }
    }

    //
    // 这里的nulls为nullptr, 表明所有的rows都不为null. 参数result为nullptr时, 下面的
    // moveOrCopyResult会直接使用localResult作为result. 如果参数rows并没有覆盖input
    // 的所有rows, 这种不会有问题吗?
    //
    // 首先, VectorFunction默认采用的defaultNullBehavior为true, 即velox框架保证参数
    // rows指定的行中, 不会包含input为null的情况, 同时我们保证result中这些行的处理结果也
    // 一定不为null. 虽然, 对于rows没有指定的行, 我们返回了非null, Expr::evalWithNulls
    // 会自动将那些rows中没有覆盖的rows置为null.
    //
    // 其次, 对于case/when等场景, 如果此时result为nullptr, 虽然当前分支会将其他分支的结
    // 果暂时设置为非null (真实结果可能为null). 但等到执行其他分支时, moveOrCopyResult
    // 会为相关的行设置正确的结果以及nulls (copy不会有问题).
    //
    // 再次, 对于包含filter的情况(见ProjectFilter.cpp), 虽然这里filter之外的行也设置了
    // 非null(注意, Expr::evalWithNulls不会自动将这些行置为null), 但ProjectFilter算子
    // 返回output时, 会自动过滤掉filter之外的行. 这种情况下, 可能存在的问题是: 如果这里的
    // 返回的vector result的nulls设置为了nullptr, 在case/when的场景下, 其他分支通过
    // EvalCtx::moveOrCopyResult对已执行分支的result进行复制时, 也会对filter排除的rows
    // 进行复制 (见BaseVector::ensureWritable). 对于FlatVector而言, 对未初始化的rows
    // 复制不会有什么问题, 即顶多是一些不必要的复制开销; 但对于DictionaryVector来说, 对未
    // 初始化的rows进行复制, 将导致系统crash, 因为这些rows对应的dict index是不确定的.
    //
    auto localResult = std::make_shared<FlatVector<bool>>(
        pool, BOOLEAN(), nullptr, rows.end(), isNull, std::vector<BufferPtr>{});
    
    context.moveOrCopyResult(localResult, rows, result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // T -> boolean
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("boolean")
                .argumentType("T")
                .build()};
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION_WITH_METADATA(
    udf_is_null,
    IsNullFunction<false>::signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    std::make_unique<IsNullFunction</*IsNotNUll=*/false>>());

void registerIsNullFunction(const std::string& name) {
  VELOX_REGISTER_VECTOR_FUNCTION(udf_is_null, name);
}

VELOX_DECLARE_VECTOR_FUNCTION_WITH_METADATA(
    udf_is_not_null,
    IsNullFunction<true>::signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    std::make_unique<IsNullFunction</*IsNotNUll=*/true>>());

void registerIsNotNullFunction(const std::string& name) {
  VELOX_REGISTER_VECTOR_FUNCTION(udf_is_not_null, name);
}

} // namespace facebook::velox::functions
