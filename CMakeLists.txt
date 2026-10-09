cmake_minimum_required(VERSION 3.28)

# Build for the GPUs in this machine unless architectures are given
# (-DCMAKE_CUDA_ARCHITECTURES=89, for example, to build without a GPU present).
# CUDA 13 supports compute capability 7.5 (Turing) and newer. Must be set before project().
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(ChessGPU LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build types: Debug, Release, and Profile (Release + source line info for Nsight).
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(is_multi_config)
  set(CMAKE_CONFIGURATION_TYPES Debug Release Profile)
elseif(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Debug, Release or Profile" FORCE)
endif()
set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CUDA_FLAGS_PROFILE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo")
set(CMAKE_EXE_LINKER_FLAGS_PROFILE "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")

# Static MSVC runtime in every config, matching the static CUDA runtime
# (cudart_static). Avoids mixed-runtime link warnings and gives a self-contained .exe.
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)

# windows.h (pulled in by NVTX) must not define min/max macros: they break std::min/std::max.
if(WIN32)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
# The standard-conforming MSVC preprocessor, which libcu++ (cuda/atomic) requires.
# /bigobj: the variant dispatchers (e.g. the CPU search's, every board x eval
# x ordering x feature set x layout) exceed the default object's sections.
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
                      $<$<COMPILE_LANGUAGE:CXX>:/bigobj> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
endif()

# Warnings for our own code. Calling a host-only function from host/device
# code is an error, not a warning: it would silently break the GPU build.
add_library(chessgpu_warnings INTERFACE)
target_compile_options(chessgpu_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:/W4>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4 --Werror cross-execution-space-call>)

# Header-only core (board, move generation, perft), compiled for both CPU and GPU.
add_library(chessgpu_core INTERFACE)
target_include_directories(chessgpu_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(chessgpu_core INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  # Profile builds report registers and spills per kernel.
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Profile>>:--ptxas-options=-v>)

# Host-only code: FEN, validation, UCI move strings, perft suite files.
add_library(chessgpu_host STATIC
  src/host/debug_print.cpp
  src/host/epd.cpp
  src/host/fen.cpp
  src/host/game_history.cpp
  src/host/host_tables.cpp
  src/host/multipv.cpp
  src/host/position.cpp
  src/host/position_moves.cpp
  src/host/search_backend.cpp
  src/host/search_cpu.cpp
  src/host/split_search.cpp
  src/host/tt_storage.cpp
  src/host/uci_move.cpp
  src/host/nnue_default.cpp
  src/host/nnue_train.cpp
  src/host/selfplay.cpp
  src/host/tune.cpp
  src/host/variants.cpp
  src/host/ybwc.cpp)
target_link_libraries(chessgpu_host PUBLIC chessgpu_core CUDA::nvtx3 PRIVATE chessgpu_warnings)

# Variants compiled into the GPU kernels. A disabled variant isn't
# instantiated (shorter builds) but stays in the source, so it can be switched
# back on and re-tested under other combinations. Losing variants are kept as
# toggles, not deleted. Example: cmake --preset default -DCHESSGPU_BOARDS=quad
set(CHESSGPU_BOARDS "std;quad" CACHE STRING "Board representations compiled into the GPU kernels: std, quad")
set(CHESSGPU_MODELS "thread;warp" CACHE STRING "GPU execution models compiled in: thread, warp")
set(CHESSGPU_EVALS "material;pesto;terms;nnue" CACHE STRING
    "Evaluations compiled into the GPU search kernels: material, pesto, terms, nnue")
set(CHESSGPU_ORDERINGS "staged;list" CACHE STRING "Thread-model move ordering implementations compiled in: staged, list")
set(CHESSGPU_STACKS "local;shared" CACHE STRING "Thread-model stack placements compiled in: local, shared")
# Feature sets: listed in src/core/feature_list.h ("all": every listed set).
# (The list's lines end in backslashes, which would escape CMake's list
# separators, so the whole file is matched at once; entries are indented,
# unlike the example in the comment.)
file(READ ${PROJECT_SOURCE_DIR}/src/core/feature_list.h _cb_feature_text)
string(REGEX MATCHALL "\n +X\\([A-Za-z0-9]+, [A-Za-z0-9]+, \"[a-z0-9-]+\"" _cb_feature_entries "${_cb_feature_text}")
set(_cb_all_features "")
set(_cb_features_types "")
set(_cb_features_kinds "")
foreach(line IN LISTS _cb_feature_entries)
  if(line MATCHES "X\\(([A-Za-z0-9]+), ([A-Za-z0-9]+), \"([a-z0-9-]+)\"")
    list(APPEND _cb_all_features ${CMAKE_MATCH_3})
    list(APPEND _cb_features_types "${CMAKE_MATCH_3}=${CMAKE_MATCH_2}")
    list(APPEND _cb_features_kinds "${CMAKE_MATCH_3}=${CMAKE_MATCH_1}")
  endif()
endforeach()
list(LENGTH _cb_all_features _cb_feature_count)
if(_cb_feature_count LESS 2)
  message(FATAL_ERROR "src/core/feature_list.h: could not read the feature sets")
endif()
set(CHESSGPU_FEATURES "all" CACHE STRING
    "Search feature sets compiled into the GPU kernels: all, or names from src/core/feature_list.h")
set(CHESSGPU_TT_LAYOUTS "packed64;wide128" CACHE STRING
    "Transposition table entry layouts compiled into the GPU kernels: packed64, wide128")
set(CHESSGPU_ATTACKS "hq;magic;kogge" CACHE STRING
    "Sliding-attack variants compiled into the GPU perft kernels: hq, magic, kogge")
set(CHESSGPU_SEARCH_ATTACKS "hq;magic" CACHE STRING
    "Sliding-attack variants compiled into the GPU search kernels: hq, magic")
set(CHESSGPU_MOVES "16;32" CACHE STRING
    "Move encodings compiled into the GPU perft kernels: 16, 32 (bits; the search uses 16)")
set(CHESSGPU_DEVICE_SEARCHES "ybwc;abdada" CACHE STRING
    "Parallel searches run entirely on the GPU compiled in: ybwc, abdada")
function(chessgpu_variant_flags list_name allowed suffix)
  if(NOT ${list_name})
    message(FATAL_ERROR "${list_name} must name at least one of: ${allowed}")
  endif()
  foreach(value IN LISTS ${list_name})
    if(NOT value IN_LIST allowed)
      message(FATAL_ERROR "${list_name}: unknown value '${value}' (allowed: ${allowed})")
    endif()
  endforeach()
  foreach(value IN LISTS allowed)
    string(TOUPPER "${value}" upper)
    string(REPLACE "-" "_" upper "${upper}")
    if(value IN_LIST ${list_name})
      set(CHESSGPU_WITH_${upper}_${suffix} 1 PARENT_SCOPE)
    else()
      set(CHESSGPU_WITH_${upper}_${suffix} 0 PARENT_SCOPE)
    endif()
  endforeach()
endfunction()
chessgpu_variant_flags(CHESSGPU_BOARDS "std;quad" BOARD)
chessgpu_variant_flags(CHESSGPU_MODELS "thread;warp" MODEL)
chessgpu_variant_flags(CHESSGPU_EVALS "material;pesto;terms;nnue" EVAL)
chessgpu_variant_flags(CHESSGPU_ORDERINGS "staged;list" ORDERING)
chessgpu_variant_flags(CHESSGPU_STACKS "local;shared" STACK)
if(CHESSGPU_FEATURES STREQUAL "all")
  set(_cb_compiled_features ${_cb_all_features})
else()
  set(_cb_compiled_features ${CHESSGPU_FEATURES})
endif()
foreach(value IN LISTS _cb_compiled_features)
  if(NOT value IN_LIST _cb_all_features)
    message(FATAL_ERROR "CHESSGPU_FEATURES: unknown value '${value}' (allowed: all, ${_cb_all_features})")
  endif()
endforeach()
set(CB_FEATURE_DEFINES "")
foreach(pair IN LISTS _cb_features_kinds)
  string(REGEX MATCH "^([a-z0-9-]+)=(.+)$" _cb_unused "${pair}")
  if(CMAKE_MATCH_1 IN_LIST _cb_compiled_features)
    string(APPEND CB_FEATURE_DEFINES "#define CHESSGPU_WITH_FEATURES_${CMAKE_MATCH_2} 1\n")
  else()
    string(APPEND CB_FEATURE_DEFINES "#define CHESSGPU_WITH_FEATURES_${CMAKE_MATCH_2} 0\n")
  endif()
endforeach()
chessgpu_variant_flags(CHESSGPU_TT_LAYOUTS "packed64;wide128" LAYOUT)
chessgpu_variant_flags(CHESSGPU_ATTACKS "hq;magic;kogge" ATTACKS)
chessgpu_variant_flags(CHESSGPU_SEARCH_ATTACKS "hq;magic" SEARCH_ATTACKS)
chessgpu_variant_flags(CHESSGPU_DEVICE_SEARCHES "ybwc;abdada" DEVICE_SEARCH)
chessgpu_variant_flags(CHESSGPU_MOVES "16;32" MOVES)
configure_file(src/gpu/variants.h.in ${CMAKE_BINARY_DIR}/generated/gpu/variants.h @ONLY)
message(STATUS "GPU kernel variants: boards ${CHESSGPU_BOARDS}; models ${CHESSGPU_MODELS}; evals ${CHESSGPU_EVALS}; "
               "orderings ${CHESSGPU_ORDERINGS}; stacks ${CHESSGPU_STACKS}; features ${_cb_compiled_features}; "
               "table layouts ${CHESSGPU_TT_LAYOUTS}; perft attacks ${CHESSGPU_ATTACKS}; "
               "search attacks ${CHESSGPU_SEARCH_ATTACKS}; device searches ${CHESSGPU_DEVICE_SEARCHES}; "
               "perft moves ${CHESSGPU_MOVES}")

# One generated unit of search kernels per feature set and table layout
# (gpu/kernels.cu.in), so the variant matrix compiles in parallel (the build
# presets run 16 jobs). Feature sets without a table have one unit.
set(CHESSGPU_KERNEL_UNITS "")
set(_cb_layout_types "packed64=TTPacked64;wide128=TTWide128")
foreach(features IN LISTS _cb_compiled_features)
  foreach(pair IN LISTS _cb_features_types)
    if(pair MATCHES "^${features}=(.+)$")
      set(CB_FEATURES_TYPE ${CMAKE_MATCH_1})
    endif()
  endforeach()
  if(NOT features MATCHES "^tt")  # sets without a table: one unit
    set(layouts "none")
  else()
    set(layouts ${CHESSGPU_TT_LAYOUTS})
  endif()
  foreach(layout IN LISTS layouts)
    set(CB_FEATURES_NAME ${features})
    set(CB_LAYOUT_NAME ${layout})
    set(CB_LAYOUT_TYPE TTNone)
    foreach(pair IN LISTS _cb_layout_types)
      if(pair MATCHES "^${layout}=(.+)$")
        set(CB_LAYOUT_TYPE ${CMAKE_MATCH_1})
      endif()
    endforeach()
    string(REPLACE "-" "_" unit "kernels_${features}_${layout}")
    configure_file(src/gpu/kernels.cu.in ${CMAKE_BINARY_DIR}/generated/gpu/${unit}.cu @ONLY)
    list(APPEND CHESSGPU_KERNEL_UNITS ${CMAKE_BINARY_DIR}/generated/gpu/${unit}.cu)
  endforeach()
endforeach()

# GPU code: device setup, perft and search kernels.
add_library(chessgpu_gpu STATIC
  src/gpu/device_context.cu
  src/gpu/device_tt.cu
  src/gpu/perft_gpu.cu
  src/gpu/search_gpu.cu
  src/gpu/ybwc_device.cu
  src/gpu/abdada_device.cu
  ${CHESSGPU_KERNEL_UNITS})
target_include_directories(chessgpu_gpu PUBLIC ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(chessgpu_gpu PUBLIC chessgpu_host CUDA::cudart_static PRIVATE chessgpu_warnings)

# Engine: UCI protocol and search backends, shared by the executable and the tests.
add_library(chessgpu_engine STATIC
  src/engine/backends.cpp
  src/engine/uci.cpp)
target_link_libraries(chessgpu_engine PUBLIC chessgpu_gpu PRIVATE chessgpu_warnings)

add_executable(chessgpu src/engine/main.cpp)
target_link_libraries(chessgpu PRIVATE chessgpu_engine chessgpu_warnings)
target_compile_definitions(chessgpu PRIVATE CHESSGPU_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

enable_testing()
add_subdirectory(tests)
