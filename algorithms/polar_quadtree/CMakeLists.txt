cmake_minimum_required(VERSION 3.14)
project(polar_quadtree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(polar_quadtree
    src/quadtree.cpp
    src/merge.cpp
    src/framework.cpp
    src/solvers.cpp
)
target_include_directories(polar_quadtree PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(pqt_demo src/main.cpp)
target_link_libraries(pqt_demo PRIVATE polar_quadtree)

add_executable(test_divide_merge tests/test_divide_merge.cpp)
target_link_libraries(test_divide_merge PRIVATE polar_quadtree)

add_executable(dump_viz src/dump_viz.cpp)
target_link_libraries(dump_viz PRIVATE polar_quadtree)

enable_testing()
add_test(NAME divide_merge COMMAND test_divide_merge)
