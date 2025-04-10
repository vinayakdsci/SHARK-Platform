#include "mobilenet.h"

namespace shortfin::cpp {
MobileNetImpl::MobileNetImpl() {
  test_toggle_ = true;
  if (test_toggle_) {
    std::cerr << "Working\n";
  }
}
}  // namespace shortfin::cpp
