#include <memory>

#include "shortfin/local/fiber.h"
#include "shortfin/local/system.h"

namespace shortfin::cpp {
class MobileNetImpl {
 public:
  explicit MobileNetImpl() = default;
  ~MobileNetImpl() {
    system_->Shutdown();
    system_.reset();
  }

  std::string ReadFile(const char *path);
  /// Load a module from the given path.
  local::ProgramModule loadProgramModule(const char *path);

 private:
  local::SystemPtr system_;
  std::shared_ptr<local::Fiber> fiber_;
  local::ScopedDevice device_;
};

}  // namespace shortfin::cpp
