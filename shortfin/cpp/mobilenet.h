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

  void loadMobileNetProgram(const char *filepath, const char *param_path);
  local::System &system() { return *system_.get(); }
  local::ScopedDevice &device() { return device_; }

 private:
  local::ProgramModule loadProgramModule(const char *path);
  local::SystemPtr system_;
  std::shared_ptr<local::Fiber> fiber_;
  local::ScopedDevice device_;
  local::StaticProgramParameters model_params_;
};

}  // namespace shortfin::cpp
