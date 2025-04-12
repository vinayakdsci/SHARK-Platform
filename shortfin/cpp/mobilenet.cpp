#include "mobilenet.h"

#include <filesystem>
#include <memory>

#include "shortfin/array/array.h"
#include "shortfin/array/dtype.h"
#include "shortfin/local/async.h"
#include "shortfin/local/fiber.h"
#include "shortfin/local/program.h"
#include "shortfin/local/systems/amdgpu.h"

namespace fs = std::filesystem;

namespace shortfin::cpp {

////////////////////////////////////////
/// Buffer Object
////////////////////////////////////////
MobileNetBufferObject::MobileNetBufferObject(std::span<size_t> dims,
                                             local::ScopedDevice &device)
    : input_(array::device_array::for_device(device, dims,
                                             array::DType::float32())) {}

void MobileNetBufferObject::fillBufferAndTransferToDevice(
    std::vector<float> input_data) {
  auto host_arr = input_.for_transfer();
  {
    auto map = host_arr.typed_data_w<float>();
    std::copy(input_data.begin(), input_data.end(), map.begin());
  }
  input_.copy_from(host_arr);
}

////////////////////////////////////////
/// InferenceExecProcess
////////////////////////////////////////
void InferenceExecProcess::RunInference(InferenceExecProcess *process) {
  if (!process->exec_req_) {
    std::cerr << "No request attached to the Process for inference\n";
    exit(1);
  }

  auto &bo = process->exec_req_->dataBuffer();
  auto prog_function = process->program_->LookupFunction("module.main_graph");

  if (!prog_function.has_value()) {
    std::cerr
        << "Failed to lookup function: VM function 'main_graph' not found\n";
    return;
  }
  std::cerr << prog_function->calling_convention() << "\n";
  auto invocation = prog_function->CreateInvocation(
      process->invocation_fiber_->shared_from_this(),
      local::ProgramIsolation::PER_FIBER);
  auto &device_arr = bo.device_array();

  auto result_fut = local::ProgramInvocation::Invoke(std::move(invocation));
  result_fut.AddCallback(
      [process](local::Future &future) { process->Terminate(); });

  auto fut = process->invocation_fiber_->device().OnSync();
  while (!fut.is_done());
}

////////////////////////////////////////
/// MobileNet Service
////////////////////////////////////////
MobileNetService::MobileNetService()
    : system_(local::systems::AMDGPUSystemBuilder().CreateSystem()),
      model_params_(local::StaticProgramParameters(*system_, "model")) {
  fiber_ = system_->CreateFiber(system_->init_worker(), system_->devices());
  device_ = fiber_->device(0);
}

local::ProgramModule MobileNetService::loadProgramModule(const char *path) {
  const auto model_fpath = fs::path(path);
  local::ProgramModule prog_module =
      local::ProgramModule::Load(*system_.get(), model_fpath);

  return prog_module;
}

void MobileNetService::Run(std::vector<float> input_data, const char *filepath,
                           const char *param_path) {
  auto dims = std::array<size_t, 4>{1, 3, 224, 224};
  auto bo = MobileNetBufferObject(std::span<size_t>{dims}, device_);

  bo.fillBufferAndTransferToDevice(input_data);

  InferenceRequest req = InferenceRequest(std::move(bo));
  local::Program program = loadMobileNetProgram(filepath, param_path);
  InferenceExecProcess exec_proc = InferenceExecProcess(fiber_, &program);

  exec_proc.attachInferenceRequest(&req);
  exec_proc.Launch();

  // We should have the result in BufferObject now.
  auto host_arr = array::device_array::for_host(
      device_, std::span<size_t>{dims}, array::DType::float32());

  auto &device_arr = bo.device_array();
  host_arr.copy_from(device_arr);

  auto raw_data = host_arr.contents_to_s();
  std::cout << raw_data.value() << std::endl;
}

local::Program MobileNetService::loadMobileNetProgram(const char *filepath,
                                                      const char *param_path) {
  auto fb_module = loadProgramModule(filepath);

  auto param_load_opts = local::StaticProgramParameters::LoadOptions();
  param_load_opts.format = "irpa";

  model_params_.Load(fs::path(param_path), param_load_opts);

  std::array<local::BaseProgramParameters *, 1> param_arr = {&model_params_};
  auto param_module = fb_module.ParameterProvider(
      *system_, std::span<local::BaseProgramParameters *>{param_arr});

  local::Program::Options options = local::Program::Options();

  auto device_span = [this]() -> std::vector<const local::Device *> {
    std::vector<const local::Device *> devices;
    for (auto &[str_v, device] : fiber_->raw_devices()) {
      devices.push_back(device);
    }
    return devices;
  }();

  options.devices = std::span<const local::Device *>{device_span};
  options.isolation = local::ProgramIsolation::PER_FIBER;
  options.trace_execution = true;

  auto mod_arr = std::array<local::ProgramModule, 2>{param_module, fb_module};
  auto mobilenet_prog = local::Program::Load(
      std::span<local::ProgramModule>{mod_arr}, std::move(options));

  mobilenet_prog.PrepareIsolate(*fiber_);

  return mobilenet_prog;
}

}  // namespace shortfin::cpp
