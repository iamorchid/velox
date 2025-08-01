# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_subdirectory(specialforms)
velox_add_library(
  velox_functions_spark_impl
  ArrayGetFunction.cpp
  ArraySort.cpp
  CharVarcharUtils.cpp
  Comparisons.cpp
  ConcatWs.cpp
  DecimalArithmetic.cpp
  DecimalCeil.cpp
  DecimalCompare.cpp
  Hash.cpp
  In.cpp
  LeastGreatest.cpp
  MakeTimestamp.cpp
  Map.cpp
  RegexFunctions.cpp
  Size.cpp
  String.cpp
  UnscaledValueFunction.cpp)

velox_link_libraries(
  velox_functions_spark_impl
  velox_functions_lib
  velox_functions_prestosql_impl
  velox_functions_spark_specialforms
  velox_functions_util
  Folly::folly
  simdjson::simdjson)

if(NOT VELOX_MONO_LIBRARY)
  set_property(TARGET velox_functions_spark_impl PROPERTY JOB_POOL_COMPILE
                                                          high_memory_pool)
endif()

add_subdirectory(window)

if(${VELOX_ENABLE_AGGREGATES})
  add_subdirectory(aggregates)
endif()

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
  add_subdirectory(coverage)
  add_subdirectory(fuzzer)
endif()

if(${VELOX_ENABLE_BENCHMARKS})
  add_subdirectory(benchmarks)
endif()

add_subdirectory(registration)
