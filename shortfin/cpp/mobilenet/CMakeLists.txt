add_executable(mobilenet)
target_link_libraries(mobilenet
  PUBLIC
    iree_base_base
    PUBLIC
    ${SHORTFIN_LINK_LIBRARY_NAME}
)
target_sources(mobilenet PUBLIC driver.cc mobilenet.cc mobilenet.h)
add_library(mobilenet_cpp mobilenet.h mobilenet.cc)
target_link_libraries(mobilenet_cpp
  PUBLIC
    iree_base_base
    PUBLIC
    ${SHORTFIN_LINK_LIBRARY_NAME}
)

set_target_properties(mobilenet PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_subdirectory(bindings)