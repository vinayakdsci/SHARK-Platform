#include "mobilenet.h"

#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <stdexcept>

#include "fmt/format.h"
#include "iree/base/allocator.h"
#include "iree/base/time.h"
#include "iree/vm/ref.h"
#include "iree/vm/ref_cc.h"
#include "shortfin/array/array.h"
#include "shortfin/array/dtype.h"
#include "shortfin/local/async.h"
#include "shortfin/local/fiber.h"
#include "shortfin/local/program.h"
#include "shortfin/local/program_interfaces.h"
#include "shortfin/local/system.h"
#include "shortfin/local/systems/amdgpu.h"
#include "shortfin/local/worker.h"
#include "shortfin/support/iree_helpers.h"

namespace fs = std::filesystem;

namespace shortfin::cpp {

void WriteBufferToDisk(unsigned char *buffer, const char *path, size_t size) {
  float *data = reinterpret_cast<float *>(buffer);
  std::ofstream out(fs::path(path), std::ios::out | std::ios::binary);

  out.write((const char *)buffer, size);
  out.flush();
  out.close();
}

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
    exit(1);
  }

  auto &bo = process->exec_req_->dataBuffer();
  auto prog_function = process->program_->LookupFunction("module.main_graph");

  if (!prog_function.has_value()) {
    std::cerr
        << "Failed to lookup function: VM function 'main_graph' not found\n";
    return;
  }
  auto invocation = prog_function->CreateInvocation(
      process->fiber()->shared_from_this(), local::ProgramIsolation::PER_FIBER);
  auto &device_arr = bo.device_array();

  device_arr.AddAsInvocationArgument(invocation.get(),
                                     local::ProgramResourceBarrier::DEFAULT);

  auto result_fut = local::ProgramInvocation::Invoke(std::move(invocation));

  auto fut = process->fiber()->device(0).OnSync();
  auto &invocation_ptr = result_fut;
  process->invocation_ptr_ = invocation_ptr;
  auto Worker = local::Worker::GetCurrent();
  Worker->Kill();
}

std::variant<array::device_array, array::storage>
InferenceExecProcess::GetResultFromInvocationRef(
    local::ProgramInvocation::Ptr &invocation, ::iree::vm::opaque_ref ref,
    local::CoarseInvocationTimelineImporter *timeline_importer) {
  auto type = ref.get()->type;
  if (local::ProgramInvocationMarshalableFactory::invocation_marshalable_type<
          array::device_array>() == type) {
    return local::ProgramInvocationMarshalableFactory::
        CreateFromInvocationResultRef<array::device_array>(
            invocation.get(), timeline_importer, std::move(ref));
  } else if (local::ProgramInvocationMarshalableFactory::
                 invocation_marshalable_type<array::storage>() == type) {
    // storage
    return local::ProgramInvocationMarshalableFactory::
        CreateFromInvocationResultRef<array::storage>(
            invocation.get(), timeline_importer, std::move(ref));
  }

  throw std::invalid_argument(
      fmt::format("Could not marshal ref type {}",
                  to_string_view(iree_vm_ref_type_name(type))));
}

////////////////////////////////////////
/// MobileNet Service
////////////////////////////////////////

void MobileNetService::InitializeService() {
  system_ = local::systems::AMDGPUSystemBuilder().CreateSystem();
  local::Worker::Options options(iree_allocator_system(), "pump_runner_main");
  auto &worker = system_->CreateWorker(options);
  fiber_ = system_->CreateFiber(worker, system_->devices());
  device_ = fiber_->device(0);
}

void MobileNetService::RunMain(std::vector<float> input_data,
                               const char *filepath, const char *parampath) {
  local::Worker::Options options(iree_allocator_system(), "pump_runner_tmp");
  auto &worker = system_->CreateWorker(options);
  worker.CallThreadsafe([&]() { this->Run(input_data, filepath, parampath); });
  // worker.RunOnCurrentThread();
  worker.WaitForShutdown();
}

local::ProgramModule MobileNetService::loadProgramModule(const char *path) {
  const auto model_fpath = fs::path(path);
  local::ProgramModule prog_module =
      local::ProgramModule::Load(*system_, model_fpath);

  return prog_module;
}

struct Data {
  MobileNetService *service;
  std::vector<float> input;
  const char *fp;
  const char *ip;
};

void MobileNetService::Run(std::vector<float> input_data, const char *filepath,
                           const char *param_path) {
  // TODO(vinayakdsci): Move device array creation to InferenceRequest
  auto dims = std::array<size_t, 4>{1, 3, 224, 224};
  auto input_bo = MobileNetBufferObject(std::span<size_t>{dims}, device_);

  input_bo.fillBufferAndTransferToDevice(input_data);

  InferenceRequest req = InferenceRequest(std::move(input_bo));
  local::Program program = loadMobileNetProgram(filepath, param_path);

  InferenceExecProcess exec_proc = InferenceExecProcess(fiber_, &program);

  exec_proc.attachInferenceRequest(&req);
  exec_proc.Launch();
  auto nfut = exec_proc.fiber()->device(0).OnSync();

  nfut.AddCallback([&](local::Future &future) {
    std::cerr << "Terminating process" << std::endl;
  });

  std::future<bool> future = std::async(std::launch::async, [&] {
    while (!nfut.is_done()) {
    }
    return nfut.is_done();
  });

  std::future<bool> future2 = std::async(std::launch::async, [&] {
    while (!exec_proc.future().is_done()) {
    }
    return exec_proc.future().is_done();
  });

  future.wait();
  future2.wait();

  if (future.get() && future2.get()) {
    exec_proc.Terminate();
  }

  auto completion_event = exec_proc.OnTermination();
  completion_event.BlockingWait(iree_infinite_timeout());

  local::CoarseInvocationTimelineImporter::Options options;
  options.assume_no_alias = true;

  auto &invocation_ptr = exec_proc.future().result();
  local::CoarseInvocationTimelineImporter timeline_importer(
      invocation_ptr.get(), options);
  ::iree::vm::opaque_ref ref = invocation_ptr->result_ref(0);
  if (!ref) {
    throw std::logic_error("Program returned NULL ref\n");
  }

  auto arr = std::get<array::device_array>(exec_proc.GetResultFromInvocationRef(
      invocation_ptr, std::move(ref), &timeline_importer));

  auto output_dims_arr = std::array<size_t, 2>{1, 1000};
  auto output_dims = std::span<size_t>{output_dims_arr};
  auto output_array = array::device_array::for_host(device_, output_dims,
                                                    array::DType::float32());
  output_array.copy_from(arr);

  WriteBufferToDisk(output_array.data().data(), "sfin_mobilenet_output.bin",
                    output_array.data().size());

  std::cerr << output_array.contents_to_s().value() << std::endl;
  auto worker = local::Worker::GetCurrent();
  worker->Kill();
}

local::Program MobileNetService::loadMobileNetProgram(const char *filepath,
                                                      const char *param_path) {
  auto fb_module = loadProgramModule(filepath);

  auto param_load_opts = local::StaticProgramParameters::LoadOptions();
  param_load_opts.format = "irpa";

  auto model_params = local::StaticProgramParameters(*system_, "model");

  model_params.Load(fs::path(param_path), param_load_opts);

  std::array<local::BaseProgramParameters *, 1> param_arr = {&model_params};
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
  options.trace_execution = false;

  auto mod_arr = std::array<local::ProgramModule, 2>{param_module, fb_module};
  auto mobilenet_prog = local::Program::Load(
      std::span<local::ProgramModule>{mod_arr}, std::move(options));

  mobilenet_prog.PrepareIsolate(*fiber_);

  return mobilenet_prog;
}

}  // namespace shortfin::cpp
