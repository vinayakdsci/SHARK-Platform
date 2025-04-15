#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <utility>
#include <variant>

#include "iree/base/status.h"
#include "shortfin/array/array.h"
#include "shortfin/local/async.h"
#include "shortfin/local/fiber.h"
#include "shortfin/local/messaging.h"
#include "shortfin/local/process.h"
#include "shortfin/local/program.h"
#include "shortfin/local/scheduler.h"
#include "shortfin/local/system.h"
#include "shortfin/local/worker.h"
#include "shortfin/support/iree_concurrency.h"

namespace shortfin::cpp {

namespace mobilenet {
void Run(std::vector<float> input_data, const char *filepath,
         const char *param_path);
}

class MobileNetBufferObject {
 public:
  MobileNetBufferObject(std::span<size_t> dims, local::ScopedDevice &device);
  MobileNetBufferObject(const MobileNetBufferObject &) = delete;
  MobileNetBufferObject(MobileNetBufferObject &&other) = default;

  MobileNetBufferObject(array::device_array array)
      : input_(std::move(array)) {};

  void fillBufferAndTransferToDevice(std::vector<float> input_data);

  array::device_array &device_array() { return input_; }

 private:
  array::device_array input_;
};

class InferenceRequest : public local::Message {
 public:
  InferenceRequest(MobileNetBufferObject input_buffer)
      : local::Message(), input_bo_(std::move(input_buffer)) {}
  // output_bo_(std::move(output_buffer)) {};
  InferenceRequest(const InferenceRequest &) = delete;

  void fillDeviceBuffer(std::vector<float> input_data) {
    input_bo_.fillBufferAndTransferToDevice(input_data);
  }

  MobileNetBufferObject &dataBuffer() { return input_bo_; }

 private:
  MobileNetBufferObject input_bo_;
  // MobileNetBufferObject output_bo_;
};

class InferenceExecProcess : public local::detail::BaseProcess {
 public:
  using local::detail::BaseProcess::Initialize;
  using local::detail::BaseProcess::is_initialized;
  using local::detail::BaseProcess::Launch;
  using local::detail::BaseProcess::Terminate;

  InferenceExecProcess(const InferenceExecProcess &) = delete;
  InferenceExecProcess(std::shared_ptr<local::Fiber> fiber,
                       local::Program *program)
      : local::detail::BaseProcess(), program_(program) {
    Initialize(fiber);
    if (!is_initialized()) {
      std::cerr << "Process should be initialized before launch\n";
      exit(1);
    }
  }

  static std::variant<array::device_array, array::storage>
  GetResultFromInvocationRef(
      local::ProgramInvocation::Ptr &invocation, ::iree::vm::opaque_ref ref,
      local::CoarseInvocationTimelineImporter *timeline_importer);

  void attachInferenceRequest(InferenceRequest *request) {
    exec_req_ = request;
  }

  void awaitCompletionEvent(local::CompletionEvent &event);

  local::ProgramInvocation::Future &future() { return invocation_ptr_; }

  void Launch() { BaseProcess::Launch(); }

  void ScheduleOnWorker() override {
    // TODO(vinayakdsci): The result of this process could be carrying
    // and exception. Handle that here.
    local::Worker::Options options(iree_allocator_system(), "internal_pump");
    auto &internal_pump = fiber()->system().CreateWorker(options);
    internal_pump.CallThreadsafe(
        std::bind(&InferenceExecProcess::RunInference, this));
    internal_pump.WaitForShutdown();
  }

  local::VoidFuture &done() { return *done_; }

 private:
  static void RunInference(InferenceExecProcess *process);
  // The list of requests to execute on this process.
  InferenceRequest *exec_req_;
  local::Program *program_;
  // array::device_array output_arr_;
  local::ProgramInvocation::Future invocation_ptr_;
  std::unique_ptr<local::VoidFuture> done_;
};

class MobileNetService {
 public:
  MobileNetService() { InitializeService(); }
  ~MobileNetService() {
    system_->Shutdown();
    system_.reset();
  }

  void RunMain(std::vector<float> input_data, const char *filepath,
               const char *parampath);

  local::Program loadMobileNetProgram(const char *filepath,
                                      const char *param_path);
  local::System &system() { return *system_; }
  local::ScopedDevice &device() { return device_; }

  local::Fiber &fiber() { return *fiber_; }

  void InitializeService();

  void Run(std::vector<float> input_data, const char *filepath,
           const char *param_path);

 private:
  local::ProgramModule loadProgramModule(const char *path);
  local::SystemPtr system_;
  std::shared_ptr<local::Fiber> fiber_;
  local::ScopedDevice device_;
  local::QueuePtr queue_;
  void *data_;
  size_t data_len_;
};

}  // namespace shortfin::cpp
