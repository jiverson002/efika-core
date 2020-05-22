# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.14)

#-------------------------------------------------------------------------------
# PROJECT configuration
#-------------------------------------------------------------------------------
set(Library_NAME Efika CACHE STRING "The name of the main library.")
set(component_NAME core)

string(TOLOWER ${Library_NAME} library_NAME)
string(TOUPPER ${Library_NAME} LIBRARY_NAME)

project(${library_NAME}-${component_NAME})

#-------------------------------------------------------------------------------
# COMPONENT configuration
#-------------------------------------------------------------------------------
add_library(${PROJECT_NAME} OBJECT)
add_library(${Library_NAME}::${component_NAME} ALIAS ${PROJECT_NAME})

target_sources(${PROJECT_NAME}
  PRIVATE src/comp.c src/conv.c src/copy.c src/cord.c src/free.c src/globals.c
          src/iidx.c src/init.c src/norm.c src/perm.c src/rord.c src/sort.c
          src/test.c src/blas/csrcsc.c src/blas/spgemm.c src/blas/vdoti.c
          src/blas/vgthrz.c src/blas/vnrm2.c src/blas/vscal.c src/blas/vsctr.c
          src/blas/vsctrz.c src/gc/cleanup.c src/gc/free.c src/gc/realloc.c
          src/rsb/spgemm.c)

target_include_directories(${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:include>)

#-------------------------------------------------------------------------------
# EXPORT configuration
#-------------------------------------------------------------------------------
include(GenerateExportHeader)

generate_export_header(${PROJECT_NAME}
  BASE_NAME        ${LIBRARY_NAME}
  EXPORT_FILE_NAME include/${library_NAME}/${component_NAME}/export.h)

target_compile_definitions(${PROJECT_NAME}
  PUBLIC $<IF:$<BOOL:${BUILD_SHARED_LIBS}>,$<BUILD_INTERFACE:${library_NAME}_${component_NAME}_EXPORTS>,${LIBRARY_NAME}_STATIC_DEFINE>)

#-------------------------------------------------------------------------------
# TEST configuration
#-------------------------------------------------------------------------------
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
  if(BUILD_TESTING)
    add_subdirectory(test)
  endif()
endif()

#-------------------------------------------------------------------------------
# BENCHMARK configuration
#-------------------------------------------------------------------------------
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  add_subdirectory(bench)
endif()

#-------------------------------------------------------------------------------
# INSTALL configuration
#-------------------------------------------------------------------------------
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

configure_package_config_file(${component_NAME}-config.cmake.in
                      cmake/${component_NAME}-config.cmake
  INSTALL_DESTINATION lib/cmake/${library_NAME})

install(TARGETS ${PROJECT_NAME}
  EXPORT   ${PROJECT_NAME}Targets
  RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
           COMPONENT   ${Library_NAME}_Runtime
  LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
           COMPONENT          ${Library_NAME}_Runtime
           NAMELINK_COMPONENT ${Library_NAME}_Development
  ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
           COMPONENT   ${Library_NAME}_Development
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT ${PROJECT_NAME}Targets
  FILE        ${component_NAME}-targets.cmake
  NAMESPACE   ${Library_NAME}::
  DESTINATION lib/cmake/${library_NAME}
  COMPONENT   ${Library_NAME}_Development)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
                  ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT   ${Library_NAME}_Development)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cmake/
  DESTINATION lib/cmake/${library_NAME}
  COMPONENT   ${Library_NAME}_Development)
