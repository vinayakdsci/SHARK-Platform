FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        0f9ce749b257fdfe701edb3cf6f7027ba029434a # v2.4.0
)
FetchContent_MakeAvailable(nanobind)

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

nanobind_add_module(
  _mobilenet_cpp_binding_impl
  NB_STATIC LTO FREE_THREADED
  lib_ext.cc
)

target_link_libraries(_mobilenet_cpp_binding_impl
  PUBLIC
    iree_base_base
    mobilenet_cpp
  PRIVATE
    ${SHORTFIN_LINK_LIBRARY_NAME}
)

nanobind_add_stub(
  shortfin_cpp_mobilenet_stub
  MODULE _mobilenet_cpp_binding_impl
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/__init__.pyi"
  DEPENDS _mobilenet_cpp_binding_impl
)