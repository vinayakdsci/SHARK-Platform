project(Shortfin2)
cmake_minimum_required(VERSION 3.29)
add_executable(sfin-serve)
target_link_libraries(sfin-serve PUBLIC shortfin)
target_sources(sfin-serve PRIVATE sfin_driver.cpp mobilenet.cpp mobilenet.h)

set_target_properties(
    sfin-serve
    PROPERTIES
    CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
