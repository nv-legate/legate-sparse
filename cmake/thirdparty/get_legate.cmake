#=============================================================================
# Copyright 2024 NVIDIA Corporation
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
#=============================================================================

function(find_or_configure_legate)
  set(oneValueArgs VERSION REPOSITORY BRANCH EXCLUDE_FROM_ALL)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(legate version git_repo git_branch shallow exclude_from_all)

  # Normalize version to match conda pkg naming (e.g., 26.01.00 -> 26.01.0)
  string(REPLACE "00" "0" version "${version}")

  set(FIND_PKG_ARGS
      GLOBAL_TARGETS     legate::legate
      BUILD_EXPORT_SET   legate-sparse-exports
      INSTALL_EXPORT_SET legate-sparse-exports)

  # Require legate to be pre-installed; do not fall back to cloning.
  rapids_find_package(legate ${version} EXACT CONFIG REQUIRED ${FIND_PKG_ARGS})

  set(Legion_USE_CUDA ${Legion_USE_CUDA} PARENT_SCOPE)
  set(Legion_USE_OpenMP ${Legion_USE_OpenMP} PARENT_SCOPE)
  set(Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS} PARENT_SCOPE)

  message(VERBOSE "Legion_USE_CUDA=${Legion_USE_CUDA}")
  message(VERBOSE "Legion_USE_OpenMP=${Legion_USE_OpenMP}")
  message(VERBOSE "Legion_BOUNDS_CHECKS=${Legion_BOUNDS_CHECKS}")
endfunction()

foreach(_var IN ITEMS "legate_sparse_LEGATE_VERSION"
                      "legate_sparse_LEGATE_BRANCH"
                      "legate_sparse_LEGATE_REPOSITORY"
                      "legate_sparse_EXCLUDE_LEGATE_FROM_ALL")
  if(DEFINED ${_var})
    # Create a legate_sparse_LEGATE_BRANCH variable in the current scope either from the existing
    # current-scope variable, or the cache variable.
    set(${_var} "${${_var}}")
    # Remove legate_sparse_LEGATE_BRANCH from the CMakeCache.txt. This ensures reconfiguring the same
    # build dir without passing `-Dlegate_sparse_LEGATE_BRANCH=` reverts to the value in versions.json
    # instead of reusing the previous `-Dlegate_sparse_LEGATE_BRANCH=` value.
    unset(${_var} CACHE)
  endif()
endforeach()

if(NOT DEFINED legate_sparse_LEGATE_VERSION)
  set(legate_sparse_LEGATE_VERSION "${legate_sparse_VERSION}")
endif()

find_or_configure_legate(VERSION          ${legate_sparse_LEGATE_VERSION}
                         REPOSITORY       ${legate_sparse_LEGATE_REPOSITORY}
                         BRANCH           ${legate_sparse_LEGATE_BRANCH}
                         EXCLUDE_FROM_ALL ${legate_sparse_EXCLUDE_LEGATE_FROM_ALL}
)
