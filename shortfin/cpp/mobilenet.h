#pragma once

#include <functional>
#include <memory>

#include "shortfin/array/array.h"
#include "shortfin/local/fiber.h"
#include "shortfin/local/messaging.h"
#include "shortfin/local/process.h"
#include "shortfin/local/program.h"
#include "shortfin/local/scheduler.h"
#include "shortfin/local/system.h"

namespace shortfin::cpp {

class MobileNetBufferObject {
 public:
  MobileNetBufferObject(std::span<size_t> dims, local::ScopedDevice &device);
  MobileNetBufferObject(const MobileNetBufferObject &) = delete;
  MobileNetBufferObject(MobileNetBufferObject &&other) = default;

  void fillBufferAndTransferToDevice(std::vector<float> input_data);

  array::device_array &device_array() { return input_; }

 private:
  array::device_array input_;
};

class InferenceRequest : public local::Message {
 public:
  InferenceRequest(MobileNetBufferObject buffer)
      : local::Message(), bo_(std::move(buffer)) {};
  InferenceRequest(const InferenceRequest &) = delete;

  void fillDeviceBuffer(std::vector<float> input_data) {
    bo_.fillBufferAndTransferToDevice(input_data);
  }

  MobileNetBufferObject &dataBuffer() { return bo_; }

 private:
  MobileNetBufferObject bo_;
};

class InferenceExecProcess : public local::detail::BaseProcess {
 public:
  InferenceExecProcess(const InferenceExecProcess &) = delete;
  InferenceExecProcess(std::shared_ptr<local::Fiber> fiber,
                       local::Program *program)
      : local::detail::BaseProcess(),
        invocation_fiber_(fiber),
        program_(program) {
    Initialize(fiber);
    if (!is_initialized()) {
      std::cerr << "Process should be initialized before launch\n";
      exit(1);
    }
  }

  void attachInferenceRequest(InferenceRequest *request) {
    exec_req_ = request;
  }

  using local::detail::BaseProcess::Launch;

 private:
  static void RunInference(InferenceExecProcess *process);

  void ScheduleOnWorker() override {
    // TODO(vinayakdsci): The result of this process could be carrying
    // and exception. Handle that here.
    invocation_fiber_->worker().CallThreadsafe(
        std::bind(InferenceExecProcess::RunInference, this));
  }

  // The list of requests to execute on this process.
  InferenceRequest *exec_req_;
  std::shared_ptr<local::Fiber> invocation_fiber_;
  local::Program *program_;
};

class MobileNetService {
 public:
  MobileNetService();
  ~MobileNetService() {
    system_->Shutdown();
    system_.reset();
  }

  local::Program loadMobileNetProgram(const char *filepath,
                                      const char *param_path);
  local::System &system() { return *system_; }
  local::ScopedDevice &device() { return device_; }

  void Run(std::vector<float> input_data, const char *filepath,
           const char *param_path);

 private:
  local::ProgramModule loadProgramModule(const char *path);
  local::SystemPtr system_;
  std::shared_ptr<local::Fiber> fiber_;
  local::ScopedDevice device_;
  local::StaticProgramParameters model_params_;
};

}  // namespace shortfin::cpp
