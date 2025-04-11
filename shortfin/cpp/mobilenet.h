#pragma once

#include <memory>

#include "shortfin/local/fiber.h"
#include "shortfin/local/program.h"
#include "shortfin/local/system.h"

namespace shortfin::cpp {
class MobileNetImpl {
 public:
  MobileNetImpl();
  ~MobileNetImpl() {
    system_->Shutdown();
    system_.reset();
  }

  std::string ReadFile(const char *path);
  /// Load a module from the given path.
  local::ProgramModule loadProgramModule(const char *path);
  void loadMobileNetProgram(const char *filepath, const char *param_path);

  local::System &system() { return *system_.get(); }

 private:
  local::SystemPtr system_;
  std::shared_ptr<local::Fiber> fiber_;
  local::ScopedDevice device_;
  local::StaticProgramParameters model_params_;
};

}  // namespace shortfin::cpp
