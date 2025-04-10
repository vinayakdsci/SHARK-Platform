#include <iostream>

namespace shortfin::cpp {
class MobileNetImpl {
 public:
  explicit MobileNetImpl();
  ~MobileNetImpl() { std::cerr << "Exec Done\n"; }

 private:
  bool test_toggle_;
};

}  // namespace shortfin::cpp
