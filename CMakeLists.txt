# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.14)

# Subproject of efika
project(efika-core)

# Include build configurations (compiler standards, options, etc.)
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND EFIKA_BUILDCONFIG_FILE)
  include(${EFIKA_BUILDCONFIG_FILE})
endif()

# Include required CMake modules
include(GenerateExportHeader)

#-------------------------------------------------------------------------------
# COMPONENT configuration
#-------------------------------------------------------------------------------
add_library(${PROJECT_NAME}     OBJECT)
add_library(${PROJECT_NAME}-sdk INTERFACE)

generate_export_header(${PROJECT_NAME}
  EXPORT_FILE_NAME efika/core/export.h
  BASE_NAME        core
  PREFIX_NAME      EFIKA_)

target_sources(${PROJECT_NAME}
  PRIVATE src/comp.c src/free.c src/iidx.c src/init.c src/norm.c src/sort.c
          src/test.c src/blas/csrcsc.c src/blas/vnrm2.c src/blas/vscal.c
          src/gc/cleanup.c src/gc/free.c src/gc/realloc.c)

target_include_directories(${PROJECT_NAME}
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include
          ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(${PROJECT_NAME}-sdk
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)

add_library(Efika::core    ALIAS ${PROJECT_NAME})
add_library(EfikaSDK::core ALIAS ${PROJECT_NAME}-sdk)

#-------------------------------------------------------------------------------
# Unit test configuration
#-------------------------------------------------------------------------------
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
  if(BUILD_TESTING)
    add_subdirectory(test)
  endif()
endif()
