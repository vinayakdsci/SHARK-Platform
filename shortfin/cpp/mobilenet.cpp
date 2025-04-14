#include "mobilenet.h"

#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "fmt/format.h"
#include "iree/base/allocator.h"
#include "iree/base/attributes.h"
#include "iree/base/status.h"
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

  std::cerr << "THIS THREAD" << std::endl;

  auto &bo = process->exec_req_->dataBuffer();
  auto prog_function = process->program_->LookupFunction("module.main_graph");

  if (!prog_function.has_value()) {
    std::cerr
        << "Failed to lookup function: VM function 'main_graph' not found\n";
    return;
  }
  std::cerr << prog_function->calling_convention() << "\n";
  auto invocation = prog_function->CreateInvocation(
      process->fiber()->shared_from_this(), local::ProgramIsolation::PER_FIBER);
  auto &device_arr = bo.device_array();

  device_arr.AddAsInvocationArgument(invocation.get(),
                                     local::ProgramResourceBarrier::DEFAULT);

  auto result_fut = local::ProgramInvocation::Invoke(std::move(invocation));
  // result_fut.AddCallback(
  //     [&process](local::Future &future) { process->Terminate(); });

  std::cerr << __LINE__ << std::endl;
  auto fut = process->fiber()->device(0).OnSync();
  std::cerr << __LINE__ << std::endl;
  process->done_ = std::make_unique<local::VoidFuture>(fut);
  // fut.AddCallback([&process](local::Future &future) { process->Terminate();
  // });

  std::cerr << __LINE__ << std::endl;
  // process->OnTermination().BlockingWait();

  auto &invocation_ptr = result_fut;

  std::cerr << __LINE__ << std::endl;
  process->invocation_ptr_ = invocation_ptr;

  std::cerr << __LINE__ << std::endl;
  auto Worker = local::Worker::GetCurrent();
  Worker->Kill();
  std::cerr << __LINE__ << std::endl;
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
MobileNetService::MobileNetService(local::Fiber &fiber, local::Worker &worker,
                                   array::device_array *output)
    : model_params_(local::StaticProgramParameters(fiber.system(), "model")),
      output_(output) {
  // auto options =
  //     local::Worker::Options(iree_allocator_system(),
  //     "internal_pump_runner");
  fiber_ = fiber.shared_ptr();
  device_ = fiber_->device(0);
}

local::ProgramModule MobileNetService::loadProgramModule(const char *path) {
  const auto model_fpath = fs::path(path);
  local::ProgramModule prog_module =
      local::ProgramModule::Load(fiber().system(), model_fpath);

  return prog_module;
}

struct Data {
  MobileNetService *service;
  std::vector<float> input;
  const char *fp;
  const char *ip;
};

namespace mobilenet {
void Run(std::vector<float> input_data, const char *filepath,
         const char *param_path) {
  local::SystemPtr system =
      local::System::Create(iree_allocator_system(), "amdgpu");

  local::Worker::Options options(iree_allocator_system(), "pump_runner");
  auto &worker = system->CreateWorker(options);

  auto fiber = system->CreateFiber(worker, system->devices());
  auto output_dims_arr = std::array<size_t, 2>{1, 1000};
  auto output_dims = std::span<size_t>{output_dims_arr};
  auto device = fiber->device(0);
  auto output_array = array::device_array::for_host(device, output_dims,
                                                    array::DType::float32());

  auto service = MobileNetService(*fiber, worker, &output_array);

  Data data = {
      .service = &service,
      .input = input_data,
      .fp = filepath,
      .ip = param_path,
  };

  // worker.RunOnCurrentThread();

  std::mutex mutex;
  std::condition_variable cvar;
  bool done;

  worker.CallThreadsafe([&]() {
    data.service->Run(data.input, data.fp, data.ip, done, cvar, mutex);
  });

  // worker.CallLowLevel(
  //     [](void *d, iree_loop_t, iree_status_t) noexcept -> iree_status_t {
  //       Data data = *(Data *)d;
  //       data.service->Run(data.input, data.fp, data.ip);
  //       return iree_ok_status();
  //     },
  //     &data);

  {
    std::unique_lock<std::mutex> lock(mutex);
    while (!done) {
      cvar.wait(lock);
    }
  }
  std::cerr << *output_array.contents_to_s() << std::endl;
  // system->Shutdown();
}
}  // namespace mobilenet

void MobileNetService::Run(std::vector<float> input_data, const char *filepath,
                           const char *param_path, bool &done,
                           std::condition_variable &cvar, std::mutex &mutex) {
  // TODO(vinayakdsci): Move device array creation to InferenceRequest
  auto fut = fiber().device(0).OnSync(false);

  fut.AddCallback(
      [](local::Future &future) { std::cerr << "REALITY CHECK" << std::endl; });

  auto dims = std::array<size_t, 4>{1, 3, 224, 224};
  auto device = fiber().device(0);
  auto input_bo = MobileNetBufferObject(std::span<size_t>{dims}, device);

  input_bo.fillBufferAndTransferToDevice(input_data);

  InferenceRequest req = InferenceRequest(std::move(input_bo));
  local::Program program = loadMobileNetProgram(filepath, param_path);

  InferenceExecProcess exec_proc = InferenceExecProcess(fiber_, &program);

  exec_proc.attachInferenceRequest(&req);
  exec_proc.Launch();

  sleep(15);

  std::cerr << __LINE__ << std::endl;
  exec_proc.future().AddCallback(
      [&exec_proc](local::Future &future) { exec_proc.Terminate(); });

  std::cerr << __LINE__ << std::endl;
  auto nfut = exec_proc.fiber()->device(0).OnSync();
  if (nfut.is_failure()) {
    std::cerr << "FAILURE" << " " << __LINE__ << std::endl;
  }

  if (nfut.is_done()) {
    std::cerr << "DONE" << std::endl;
  }  // else {
  //   nfut.AddCallback([&exec_proc](local::Future &future) {
  //     exec_proc.Terminate();
  //     std::cerr << "Terminated" << std::endl;
  //   });
  // }

  std::cerr << __LINE__ << std::endl;

  auto completion_event = exec_proc.OnTermination();

  auto worker = local::Worker::GetCurrent();
  // completion_event.BlockingWait(iree_infinite_timeout());
  std::cerr << __LINE__ << std::endl;

  // auto err = worker.WaitOneLowLevel(
  //     completion_event, iree_infinite_timeout(),
  //     +[](void *ptr, iree_loop_t loop, iree_status_t status) noexcept {
  //       return iree_ok_status();
  //     },
  //     nullptr);

  local::CoarseInvocationTimelineImporter::Options options;
  options.assume_no_alias = true;

  auto &invocation_ptr = exec_proc.future().result();

  std::cerr << invocation_ptr->to_s();

  local::CoarseInvocationTimelineImporter timeline_importer(
      invocation_ptr.get(), options);
  std::cerr << __LINE__ << std::endl;
  for (size_t i = 0; i < invocation_ptr->results_size(); ++i) {
    std::cerr << "RESULT\n";
  }
  ::iree::vm::opaque_ref ref = invocation_ptr->result_ref(0);
  if (!ref) {
    throw std::logic_error("Program returned NULL ref\n");
  }

  std::cerr << __LINE__ << std::endl;
  auto arr = std::get<array::device_array>(exec_proc.GetResultFromInvocationRef(
      invocation_ptr, std::move(ref), &timeline_importer));

  // exec_proc.output().copy_from(arr);
  // while (!t) {
  //   auto host_array_xtensor = host_array.data().data();
  //   auto check = [&host_array_xtensor, &host_array]() -> bool {
  //     auto tmp = 1;
  //     for (size_t i = 0; i < host_array.data().size(); ++i) {
  //       if (host_array_xtensor[i] == 0) {
  //         // std::cerr << "0\n";
  //         tmp = 0;
  //       } else {
  //         std::cerr << host_array_xtensor[i] << "\n";
  //       }
  //     }
  //     return tmp == 1;
  //   };
  //   t = check();
  // }

  bool t = false;

  std::cerr << __LINE__ << std::endl;
  output_->copy_from(arr);

  std::cerr << __LINE__ << std::endl;
  std::scoped_lock<std::mutex> lock(mutex);
  done = true;
  cvar.notify_all();
  std::cerr << __LINE__ << std::endl;
  // std::cerr << host_array.contents_to_s().value();
}

local::Program MobileNetService::loadMobileNetProgram(const char *filepath,
                                                      const char *param_path) {
  auto fb_module = loadProgramModule(filepath);

  auto param_load_opts = local::StaticProgramParameters::LoadOptions();
  param_load_opts.format = "irpa";

  model_params_.Load(fs::path(param_path), param_load_opts);

  std::array<local::BaseProgramParameters *, 1> param_arr = {&model_params_};
  auto param_module = fb_module.ParameterProvider(
      fiber().system(), std::span<local::BaseProgramParameters *>{param_arr});

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
