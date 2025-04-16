#include "async.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <mutex>

#include "shortfin/local/fiber.h"
#include "shortfin/local/systems/amdgpu.h"
#include "systems/host.h"

using namespace shortfin::local;

class CrossWorkerFutCallbackTest : public testing::Test {
 protected:
  CrossWorkerFutCallbackTest() {}

  void SetUp() override {
    system = systems::AMDGPUSystemBuilder().CreateSystem();
    auto &worker = system->CreateWorker(
        Worker::Options(iree_allocator_system(), "pump_runner"));
    fiber = system->CreateFiber(worker, system->devices());
    device = fiber->device(0);
  }
  void TearDown() override {
    system->Shutdown();
    system.reset();
  }

  SystemPtr system;
  std::shared_ptr<Fiber> fiber;
  ScopedDevice device;
};

TEST_F(CrossWorkerFutCallbackTest, add_device_onsync_callback) {
  // No work on the device, should callback immediately.
  auto onsync_fut = device.OnSync();
  std::condition_variable cv;
  std::mutex mutex;
  bool done = false;

  onsync_fut.AddCallback([&](Future &) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      done = true;
    }
    cv.notify_one();
  });

  std::unique_lock<std::mutex> g(mutex);
  bool success = cv.wait_for(g, std::chrono::seconds(10), [&] { return done; });

  ASSERT_TRUE(success);
}
