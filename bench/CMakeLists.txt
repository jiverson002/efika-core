# Include required CMake modules
include(FetchContent)

#-------------------------------------------------------------------------------
# Celero configuration
#-------------------------------------------------------------------------------
FetchContent_Declare(celero
  GIT_REPOSITORY https://github.com/jiverson002/Celero.git
  GIT_TAG        develop)

FetchContent_MakeAvailable(celero)

#-------------------------------------------------------------------------------
# EXECUTABLE configuration
#-------------------------------------------------------------------------------
add_executable(${PROJECT_NAME}-benchmarks)

target_sources(${PROJECT_NAME}-benchmarks
  PRIVATE main.cc spgemm.cc)

target_link_libraries(${PROJECT_NAME}-benchmarks
  PRIVATE celero)
