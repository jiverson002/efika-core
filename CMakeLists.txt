cmake_minimum_required(VERSION 3.13)

#-------------------------------------------------------------------------------
# BOOTSTRAP configuration -- this is the genesis of the bootstrap project
#-------------------------------------------------------------------------------
set(BS_PROJECT_NAME efika CACHE STRING "Bootstrap project name")

#-------------------------------------------------------------------------------
# SUBPROJECT configuration
#-------------------------------------------------------------------------------
project(${BS_PROJECT_NAME}-core LANGUAGES C CXX)

cmake_policy(SET CMP0076 NEW)
cmake_policy(SET CMP0079 NEW)

#-------------------------------------------------------------------------------
# Require C99/C++17 support from compiler
#-------------------------------------------------------------------------------
c_99()
cxx_17()

#-------------------------------------------------------------------------------
# LIBRARY configuration
#-------------------------------------------------------------------------------
add_library(${BS_PROJECT_NAME} INTERFACE)

target_include_directories(${BS_PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

#-------------------------------------------------------------------------------
# SUBLIBRARY configuration
#-------------------------------------------------------------------------------
add_library(${PROJECT_NAME})

target_link_libraries(${BS_PROJECT_NAME} INTERFACE ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_subdirectory(src)
