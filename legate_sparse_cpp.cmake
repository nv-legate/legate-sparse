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

##############################################################################
# - User Options  ------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build legate sparse shared libraries" ON)
option(legate_sparse_EXCLUDE_LEGATE_FROM_ALL "Exclude legate targets from Legate Sparse's 'all' target" OFF)

##############################################################################
# - Project definition -------------------------------------------------------

# TODO (rohany): Do we need something like this for Legate Sparse?
# Write the version header
# rapids_cmake_write_version_file(include/cupynumeric/version_config.hpp)

# Needed to integrate with LLVM/clang tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - Build Type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# ############################################################################
# * conda environment --------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# We're building python extension libraries, which must always be installed
# under lib/, even if the system normally uses lib64/. Rapids-cmake currently
# doesn't realize this when we're going through scikit-build, see
# https://github.com/rapidsai/rapids-cmake/issues/426
if(TARGET conda_env)
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()

##############################################################################
# - Dependencies -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init(OVERRIDE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/versions.json)

find_package(OpenMP REQUIRED)
#find_package(legate REQUIRED)
#TODO: we don't need cupynumeric? we assume it's there?
#find_package(legate_cupynumeric REQUIRED)

option(Legion_USE_CUDA "Use CUDA" ON)
option(Legion_USE_OpenMP "Use OpenMP" ${OpenMP_FOUND})
option(Legion_BOUNDS_CHECKS "Build legate.sparse with bounds checks (expensive)" OFF)


#################
# From cupynumeric:

###
# If we find legate already configured on the system, it will report
# whether it was compiled with bounds checking (Legion_BOUNDS_CHECKS),
# CUDA (Legion_USE_CUDA), and OpenMP (Legion_USE_OpenMP).
#
# We use the same variables as legate because we want to enable/disable
# each of these features based on how legate was configured (it doesn't
# make sense to build cupynumeric's CUDA bindings if legate wasn't built
# with CUDA support).
###
include(cmake/thirdparty/get_legate.cmake)

if(Legion_USE_CUDA)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/cuda_arch_helpers.cmake)
  # Needs to run before `rapids_cuda_init_architectures`
  set_cuda_arch_from_names()
  # Needs to run before `enable_language(CUDA)`
  rapids_cuda_init_architectures(legate_sparse)
  enable_language(CUDA)
  # Since legate_sparse only enables CUDA optionally we need to manually include
  # the file that rapids_cuda_init_architectures relies on `project` calling
  if(CMAKE_PROJECT_legate_sparse_INCLUDE)
    include("${CMAKE_PROJECT_legate_sparse_INCLUDE}")
  endif()

  # Must come after enable_language(CUDA)
  # Use `-isystem <path>` instead of `-isystem=<path>`
  # because the former works with clangd intellisense
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "-isystem ")

  rapids_find_package(
    CUDAToolkit REQUIRED
    BUILD_EXPORT_SET legate-sparse-exports
    INSTALL_EXPORT_SET legate-sparse-exports
  )

  include(cmake/thirdparty/get_nccl.cmake)
endif()

# End From cupynumeric
#################

#################
# Sources

set(legate_sparse_SOURCES "")
set(legate_sparse_CXX_DEFS "")
set(legate_sparse_CUDA_DEFS "")
set(legate_sparse_CXX_OPTIONS "")
set(legate_sparse_CUDA_OPTIONS "")

include(cmake/Modules/set_cpu_arch_flags.cmake)
set_cpu_arch_flags(legate_sparse_CXX_OPTIONS)


list(APPEND legate_sparse_SOURCES
  src/sparse/projections.cc

  src/sparse/mapper/mapper.cc

  src/sparse/array/conv/dense_to_csr.cc
  src/sparse/array/conv/csr_to_dense.cc
  src/sparse/array/conv/pos_to_coordinates.cc

  src/sparse/array/csr/get_diagonal.cc
  src/sparse/array/csr/spmv.cc
  src/sparse/array/csr/spgemm_csr_csr_csr.cc
  
  src/sparse/array/util/unzip_rect.cc
  src/sparse/array/util/zip_to_rect.cc

  src/sparse/partition/fast_image_partition.cc

  src/sparse/io/mtx_to_coo.cc
  src/sparse/linalg/axpby.cc
)

if(Legion_USE_OpenMP)
  list(APPEND legate_sparse_SOURCES
    src/sparse/array/conv/dense_to_csr_omp.cc
    src/sparse/array/conv/csr_to_dense_omp.cc
    src/sparse/array/conv/pos_to_coordinates_omp.cc

    src/sparse/array/csr/get_diagonal_omp.cc
    src/sparse/array/csr/spmv_omp.cc
    src/sparse/array/csr/spgemm_csr_csr_csr_omp.cc

    src/sparse/array/util/unzip_rect_omp.cc
    src/sparse/array/util/zip_to_rect_omp.cc

    src/sparse/linalg/axpby_omp.cc
  )
endif()

if(Legion_USE_CUDA)
  list(APPEND legate_sparse_SOURCES
    src/sparse/cudalibs.cu 

    src/sparse/array/conv/dense_to_csr.cu
    src/sparse/array/conv/csr_to_dense.cu
    src/sparse/array/conv/pos_to_coordinates.cu

    src/sparse/array/csr/get_diagonal.cu
    src/sparse/array/csr/spmv.cu
    src/sparse/array/csr/spgemm_csr_csr_csr.cu

    src/sparse/array/util/unzip_rect.cu
    src/sparse/array/util/zip_to_rect.cu
    
    src/sparse/partition/fast_image_partition.cu

    src/sparse/linalg/axpby.cu
  )
endif()


list(APPEND legate_sparse_SOURCES
  
  # This must always be the last file!
  # It guarantees we do our registration callback
  # only after all task variants are recorded
  src/sparse/sparse.cc
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND legate_sparse_CXX_DEFS DEBUG_LEGATE_SPARSE)
  list(APPEND legate_sparse_CUDA_DEFS DEBUG_LEGATE_SPARSE)
endif()

if(Legion_BOUNDS_CHECKS)
  list(APPEND legate_sparse_CXX_DEFS BOUNDS_CHECKS)
  list(APPEND legate_sparse_CUDA_DEFS BOUNDS_CHECKS)
endif()

list(APPEND legate_sparse_CUDA_OPTIONS -Xfatbin=-compress-all)
list(APPEND legate_sparse_CUDA_OPTIONS --expt-extended-lambda)
list(APPEND legate_sparse_CUDA_OPTIONS --expt-relaxed-constexpr)
list(APPEND legate_sparse_CXX_OPTIONS -Wno-deprecated-declarations)
list(APPEND legate_sparse_CUDA_OPTIONS -Wno-deprecated-declarations)

add_library(legate_sparse ${legate_sparse_SOURCES})
if(NOT TARGET legate_sparse::legate_sparse)
  add_library(legate_sparse::legate_sparse ALIAS legate_sparse)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(platform_rpath_origin "\$ORIGIN")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(platform_rpath_origin "@loader_path")
endif ()

set_target_properties(legate_sparse
           PROPERTIES BUILD_RPATH                         "${platform_rpath_origin}"
                      INSTALL_RPATH                       "${platform_rpath_origin}"
                      CXX_STANDARD                        17
                      CXX_STANDARD_REQUIRED               ON
                      POSITION_INDEPENDENT_CODE           ON
                      INTERFACE_POSITION_INDEPENDENT_CODE ON
                      CUDA_STANDARD                       17
                      CUDA_STANDARD_REQUIRED              ON
                      LIBRARY_OUTPUT_DIRECTORY            lib)

target_link_libraries(legate_sparse
   PUBLIC legate::legate
          $<TARGET_NAME_IF_EXISTS:NCCL::NCCL>
          # do we need to put this dependency here?
          # what is the correct target?
          # cupynumeric::cupynumeric
  PRIVATE 
          # Add Conda library and include paths
          $<TARGET_NAME_IF_EXISTS:conda_env>
          $<TARGET_NAME_IF_EXISTS:CUDA::cublas>
          $<TARGET_NAME_IF_EXISTS:CUDA::cusparse>
          $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>)


# Change THRUST_DEVICE_SYSTEM for `.cpp` files
if(Legion_USE_OpenMP)
  list(APPEND legate_sparse_CXX_OPTIONS -UTHRUST_DEVICE_SYSTEM)
  list(APPEND legate_sparse_CXX_OPTIONS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
elseif(NOT Legion_USE_CUDA)
  list(APPEND legate_sparse_CXX_OPTIONS -UTHRUST_DEVICE_SYSTEM)
  list(APPEND legate_sparse_CXX_OPTIONS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
endif()

target_compile_options(legate_sparse
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${legate_sparse_CXX_OPTIONS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${legate_sparse_CUDA_OPTIONS}>")

target_compile_definitions(legate_sparse
  PUBLIC  "$<$<COMPILE_LANGUAGE:CXX>:${legate_sparse_CXX_DEFS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${legate_sparse_CUDA_DEFS}>")


target_include_directories(legate_sparse
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  INTERFACE
    $<INSTALL_INTERFACE:include/legate_sparse>
)

if(Legion_USE_CUDA)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
[=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=])
  # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
  target_link_options(legate_sparse PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")
endif()

##############################################################################
# - install targets-----------------------------------------------------------

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate_sparse
        DESTINATION ${lib_dir}
        EXPORT legate-sparse-exports)

install(
  FILES src/sparse/sparse_c.h
        #TODO: ?
        #${CMAKE_CURRENT_BINARY_DIR}/include/cupynumeric/version_config.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/legate_sprase)


##############################################################################
# - install export -----------------------------------------------------------

set(doc_string
        [=[
Provide targets for Legate Sparse, an aspiring drop-in replacement for Scipy.Sparse at scale.

Imported Targets:
  - legate_sparse::legate_sparse

]=])

string(JOIN "\n" code_string
  "set(Legion_USE_CUDA ${Legion_USE_CUDA})"
  "set(Legion_USE_OpenMP ${Legion_USE_OpenMP})"
  "set(Legion_BOUNDS_CHECKS ${Legion_BOUNDS_CHECKS})"
)

if(DEFINED Legion_USE_Python)
  string(APPEND code_string "\nset(Legion_USE_Python ${Legion_USE_Python})")
endif()

if(DEFINED Legion_NETWORKS)
  string(APPEND code_string "\nset(Legion_NETWORKS ${Legion_NETWORKS})")
endif()

rapids_export(
  INSTALL legate_sparse
  EXPORT_SET legate-sparse-exports
  GLOBAL_TARGETS legate_sparse
  NAMESPACE legate_sparse::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(
  BUILD legate_sparse
  EXPORT_SET legate-sparse-exports
  GLOBAL_TARGETS legate_sparse
  NAMESPACE legate-sparse::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)
