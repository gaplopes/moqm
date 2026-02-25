cmake_minimum_required(VERSION 3.10)

project(moqm-tests)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Print the build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Warning flags
set(MOQM_CXX_WARN_FLAGS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND MOQM_CXX_WARN_FLAGS
    -Wall
    -Wextra
    -Wpedantic
  )
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    list(APPEND MOQM_CXX_WARN_FLAGS
      -Wvector-operation-performance
      -Wduplicated-cond
    )
  endif()
else()
  message(WARNING "Unknown compiler, not setting any warning flags.")
endif()

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Header-only quality metrics library (moqm)
add_library(moqm INTERFACE)
target_include_directories(moqm INTERFACE ${CMAKE_SOURCE_DIR}/include)

# Optional: link GLPK if found (needed only for classification.hpp)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(GLPK QUIET glpk)
endif()
if(NOT GLPK_FOUND)
  find_library(GLPK_LIBRARY NAMES glpk)
  if(GLPK_LIBRARY)
    set(GLPK_FOUND TRUE)
  endif()
endif()
if(GLPK_FOUND)
  target_link_libraries(moqm INTERFACE glpk)
  message(STATUS "GLPK found — classification.hpp available")
else()
  message(STATUS "GLPK not found — classification.hpp will not be usable")
endif()

# ---- moqm library tests (dependency-free, no GLPK needed) ----
enable_testing()

add_executable(test_moqm ${CMAKE_SOURCE_DIR}/tests/test_moqm.cpp)
target_include_directories(test_moqm PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(test_moqm PRIVATE cxx_std_17)
target_compile_options(test_moqm PRIVATE ${MOQM_CXX_WARN_FLAGS})

add_test(NAME moqm_tests COMMAND test_moqm)
