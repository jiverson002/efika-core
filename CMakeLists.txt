cmake_minimum_required(VERSION 3.10)

#-------------------------------------------------------------------------------
# BOOTSTRAP w/ latest bootstrap/cmake scripts
#-------------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(cmake
  GIT_REPOSITORY git@github.com:jiverson/bootstrap-cmake.git
)
FetchContent_MakeAvailable(cmake)

#-------------------------------------------------------------------------------
# PROJECT configuration
#-------------------------------------------------------------------------------
project(efika LANGUAGES C CXX)

#-------------------------------------------------------------------------------
# BOOTSTRAP configuration
#-------------------------------------------------------------------------------
set(BS_PROJECT_NAME ${PROJECT_NAME} CACHE STRING "Bootstrap project name")

#-------------------------------------------------------------------------------
# Require C99 support from compiler
#-------------------------------------------------------------------------------
c_99()

#-------------------------------------------------------------------------------
# LIBRARY configuration -- this is the genesis of the efika library target
#-------------------------------------------------------------------------------
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
