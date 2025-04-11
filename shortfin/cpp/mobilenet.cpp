#include "mobilenet.h"

#include <filesystem>

#include "shortfin/local/program.h"
#include "shortfin/local/systems/amdgpu.h"

namespace fs = std::filesystem;

namespace shortfin::cpp {

MobileNetImpl::MobileNetImpl()
    : system_(local::systems::AMDGPUSystemBuilder().CreateSystem()),
      model_params_(local::StaticProgramParameters(*system_.get(), "model")) {
  fiber_ = system_->CreateFiber(system_->init_worker(), system_->devices());
  device_ = fiber_->device(0);
}

local::ProgramModule MobileNetImpl::loadProgramModule(const char *path) {
  const auto model_fpath = fs::path(path);
  local::ProgramModule prog_module =
      local::ProgramModule::Load(*system_.get(), model_fpath);

  return prog_module;
}

void MobileNetImpl::loadMobileNetProgram(const char *filepath,
                                         const char *param_path) {
  auto fb_module = loadProgramModule(filepath);
  model_params_.Load(fs::path(param_path));

  std::array<local::BaseProgramParameters *, 1> param_arr = {&model_params_};
  auto param_module = fb_module.ParameterProvider(
      *system_, std::span<local::BaseProgramParameters *>{param_arr});

  local::Program::Options options = local::Program::Options();
  std::vector<const local::Device *> placeholder;
  placeholder.reserve(system_->devices().size());

  for (auto device : system_->devices()) {
    placeholder.push_back(device);
  }

  options.devices = std::span<const local::Device *>{placeholder};
  options.isolation = local::ProgramIsolation::PER_FIBER;
  options.trace_execution = false;

  auto mod_arr = std::array<local::ProgramModule, 2>{param_module, fb_module};
  auto mobilenet_prog = local::Program::Load(
      std::span<local::ProgramModule>{mod_arr}, std::move(options));

  mobilenet_prog.PrepareIsolate(*fiber_);
}

}  // namespace shortfin::cpp
