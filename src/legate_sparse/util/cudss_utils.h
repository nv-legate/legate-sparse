/* Copyright 2022-2024 NVIDIA Corporation
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
 *
 */

#pragma once

#include "legate_sparse/sparse.h"
#include "legate_sparse/util/cuda_help.h"
#include "legate_sparse/util/legate_utils.h"

namespace sparse {

using namespace legate;

// Template dispatch for value type.
// Note: cuDSS only supports floating-point and complex types.
// Integer and boolean types are not supported by cuDSS.
template <typename VAL_TY>
cudaDataType_t cudssDataType();

template <>
inline cudaDataType_t cudssDataType<float>()
{
  return CUDA_R_32F;
}

template <>
inline cudaDataType_t cudssDataType<double>()
{
  return CUDA_R_64F;
}

template <>
inline cudaDataType_t cudssDataType<legate::Complex<float>>()
{
  return CUDA_C_32F;
}

template <>
inline cudaDataType_t cudssDataType<legate::Complex<double>>()
{
  return CUDA_C_64F;
}

// Template dispatch for the index type.
template <typename INDEX_TY>
cudaDataType_t cudssIndexType();

template <>
inline cudaDataType_t cudssIndexType<int32_t>()
{
  return CUDA_R_32I;
}

template <>
inline cudaDataType_t cudssIndexType<int64_t>()
{
  return CUDA_R_64I;
}

}  // namespace sparse
