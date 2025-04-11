#include "mobilenet.h"
namespace fs = std::filesystem;
namespace shortfin::cpp {
local::ProgramModule MobileNetImpl::loadProgramModule(const char *path) {
  const auto model_fpath = fs::path(path);
  system_ = local::System::Create(iree_allocator_system(), "amdgpu");

  local::ProgramModule prog_module =
      local::ProgramModule::Load(*system_.get(), model_fpath);

  return prog_module;
}

}  // namespace shortfin::cpp
