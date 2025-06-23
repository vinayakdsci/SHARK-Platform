#include <nanobind/nanobind.h>

NB_MODULE(_mobilenet_cpp_binding_impl, m) {
    m.def("hello", []() { return "Hello world!"; });
}