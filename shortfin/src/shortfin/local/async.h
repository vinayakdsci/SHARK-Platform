// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_ASYNC_H
#define SHORTFIN_LOCAL_ASYNC_H

#include <any>
#include <coroutine>
#include <exception>
#include <functional>
#include <iostream>
#include <utility>

#include "iree/base/status.h"
#include "shortfin/support/api.h"
#include "shortfin/support/iree_concurrency.h"
#include "shortfin/support/iree_helpers.h"

namespace shortfin::local {

class Worker;

// CompletionEvents are the most basic form of awaitable object. They
// encapsulate a native iree_wait_source_t (which multiplexes any supported
// system level wait primitive) with a resource baton which keeps any needed
// references alive for the duration of all copies.
//
// Depending on the system wait source used, there may be a limited exception
// side-band (i.e. a way to signal that the wait handle has failed and have
// that propagate to consumers). However, in general, this is a very coarse
// mechanism. For rich result and error propagation, see the higher level
// Promise/Future types, which can be signalled with either a result or
// exception.
class SHORTFIN_API CompletionEvent {
 public:
  CompletionEvent();
  CompletionEvent(iree::shared_event::ref event);
  CompletionEvent(iree::hal_semaphore_ptr sem, uint64_t payload);
  CompletionEvent(CompletionEvent &&other)
      : wait_source_(other.wait_source_),
        resource_baton_(std::move(other.resource_baton_)) {
    other.wait_source_ = iree_wait_source_immediate();
  }
  CompletionEvent(const CompletionEvent &other)
      : wait_source_(other.wait_source_),
        resource_baton_(other.resource_baton_) {}
  CompletionEvent &operator=(const CompletionEvent &other) {
    wait_source_ = other.wait_source_;
    resource_baton_ = other.resource_baton_;
    return *this;
  }
  ~CompletionEvent();

  // Returns true if this CompletionEvent is ready.
  bool is_ready();
  // Block the current thread for up to |timeout|. If a non-infinite timeout
  // was given and the timeout expires while waiting, returns false. In all
  // other cases, returns true.
  // This should not be used in worker loops.
  bool BlockingWait(iree_timeout_t timeout = iree_infinite_timeout());

  // Access the raw wait source.
  operator const iree_wait_source_t &() { return wait_source_; }

 private:
  iree_wait_source_t wait_source_;
  // A baton used to keep any needed backing resource alive.
  std::any resource_baton_;
};

// Object that will eventually be set to some completion state, either a result
// value or an exception status. Like CompletionEvents, Futures are copyable,
// and all such copies share the same state. Future objects are bound to the
// worker on which they are created. When signaled from the same worker,
// they use a fast path, but when signaled from elsewhere, cross-worker
// signaling is used (which has more overhead).
class SHORTFIN_API Future {
 public:
  using FutureCallback = std::function<void(Future &)>;

  Future(const Future &other) = delete;
  Future(Future &&other) = delete;
  Future &operator=(const Future &other) = delete;
  virtual ~Future();

  void set_failure(iree_status_t failure_status);

  // Returns whether this future is done.
  bool is_done() {
    iree::slim_mutex_lock_guard g(state_->lock_);
    return state_->done_;
  }
  bool is_failure() {
    iree::slim_mutex_lock_guard g(state_->lock_);
    return !iree_status_is_ok(state_->failure_status_.status());
  }
  void ThrowFailure() {
    iree::slim_mutex_lock_guard g(state_->lock_);
    ThrowFailureWithLockHeld();
  }

  // Adds a callback that will be made when the future is satisfied (either
  // with a value or a failure). If the future is already satisfied, they
  // will be queued for delivery on a future cycle of the event loop. If
  // running on the same worker as owns this Future, then the callback will
  // never be executed within the scope of this call. If adding a callback
  // from another thread, then it is possible that the callback runs concurrent
  // with returning from this function.
  void AddCallback(FutureCallback callback);

  // Blocking wait that stops the thread until the Future completes.
  void Wait();

 protected:
  struct SHORTFIN_API BaseState {
    BaseState(Worker *worker) : worker_(worker) {}
    virtual ~BaseState();
    iree::slim_mutex lock_;
    Worker *worker_;
    int ref_count_ SHORTFIN_GUARDED_BY(lock_) = 1;
    iree::ignorable_status failure_status_ SHORTFIN_GUARDED_BY(lock_);
    bool done_ SHORTFIN_GUARDED_BY(lock_) = false;
    std::vector<FutureCallback> callbacks_ SHORTFIN_GUARDED_BY(lock_);
  };

  Future(BaseState *state) : state_(state) {}
  void Retain() const;
  void Release() const;
  static Worker *GetRequiredWorker();
  void SetSuccessWithLockHeld() SHORTFIN_REQUIRES_LOCK(state_->lock_) {
    state_->done_ = true;
  }
  // Posts a message to the worker to issue callbacks. Lock must be held.
  void IssueCallbacksWithLockHeld() SHORTFIN_REQUIRES_LOCK(state_->lock_);
  static iree_status_t RawHandleWorkerCallback(void *state_vp, iree_loop_t loop,
                                               iree_status_t status) noexcept;
  void HandleWorkerCallback();
  void ThrowFailureWithLockHeld() SHORTFIN_REQUIRES_LOCK(state_->lock_);

  mutable BaseState *state_;
};

// Future that has no result type. It can be done without result or have
// a failure set.
class SHORTFIN_API VoidFuture : public Future {
 public:
  VoidFuture() : Future(new BaseState(GetRequiredWorker())) {}
  VoidFuture(Worker *worker) : Future(new BaseState(worker)) {}
  ~VoidFuture() override = default;
  VoidFuture(const VoidFuture &other) : Future(other.state_) { Retain(); }
  VoidFuture &operator=(const VoidFuture &other) {
    other.Retain();
    Release();
    state_ = other.state_;
    return *this;
  }

  void set_success() {
    iree::slim_mutex_lock_guard g(state_->lock_);
    SetSuccessWithLockHeld();
    IssueCallbacksWithLockHeld();
  }

  // Coro support.
  bool await_ready() const noexcept {
    iree::slim_mutex_lock_guard guard(state_->lock_);
    return state_->done_;
  }

  void await_suspend(std::coroutine_handle<> handle) {
    // Add a callback that will resume the handle when the coro
    // completes.
    AddCallback([handle](Future &) { handle.resume(); });
  }

  void await_resume() {
    // Propagate any exceptions if they occur.
    ThrowFailure();
  }
};

// Value containing Future.
template <typename ResultTy>
class SHORTFIN_API TypedFuture : public Future {
 public:
  TypedFuture() : Future(new TypedState(GetRequiredWorker())) {}
  TypedFuture(Worker *worker) : Future(new TypedState(worker)) {}
  ~TypedFuture() override = default;
  TypedFuture(const TypedFuture &other) : Future(other.state_) { Retain(); }
  TypedFuture &operator=(const TypedFuture &other) {
    other.Retain();
    Release();
    state_ = other.state_;
    return *this;
  }

  // Futures are non-nullable, so construct/assign from an rvalue reference
  // is just a copy and does not clear the original.
  TypedFuture(TypedFuture &&other) : Future(other.state_) { Retain(); }
  TypedFuture &operator=(TypedFuture &&other) {
    other.Retain();
    Release();
    state_ = other.state_;
    return *this;
  }

  void set_result(ResultTy result) {
    iree::slim_mutex_lock_guard g(state_->lock_);
    if (state_->done_) {
      throw std::logic_error(
          "Cannot 'set_failure' on a Future that is already done");
    }
    static_cast<TypedState *>(state_)->result_ = std::move(result);
    SetSuccessWithLockHeld();
    IssueCallbacksWithLockHeld();
  }

  ResultTy &result() {
    iree::slim_mutex_lock_guard g(state_->lock_);
    ThrowFailureWithLockHeld();
    return static_cast<TypedState *>(state_)->result_;
  }

  // Coro support.
  bool await_ready() const noexcept {
    iree::slim_mutex_lock_guard guard(state_->lock_);
    return state_->done_;
  }

  void await_suspend(std::coroutine_handle<> handle) {
    AddCallback([handle](Future &) { handle.resume(); });
  }

  ResultTy await_resume() {
    // In case of an exception, propagate. Else return the result.
    iree::slim_mutex_lock_guard guard(state_->lock_);
    ThrowFailureWithLockHeld();
    return static_cast<TypedState *>(state_)->result_;
  }

 private:
  struct SHORTFIN_API TypedState : public BaseState {
    using BaseState::BaseState;
    ResultTy result_;
  };
};

// Wraps awaitable Futures.
template <typename T = void>
class Promise {
 public:
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;

  Promise(handle_type handle) : handle_(handle) {}
  Promise(Promise &&other) : handle_(std::exchange(other.handle_, nullptr)) {}
  Promise(const Promise &other) = delete;
  Promise &operator=(const Promise &) = delete;
  Promise &operator=(Promise &&other) noexcept {
    if (this != &other) {
      if (handle_) {
        handle_.destroy();
        handle_ = std::exchange(other.handle_, nullptr);
      }
    }
    return *this;
  }
  ~Promise() {
    if (handle_) {
      handle_.destroy();
    }
  }

  struct promise_type {
    // promise_type for a TypedFuture.
    // We'll specialise for a VoidFuture.
    TypedFuture<T> future_;
    std::exception_ptr curr_exception_;

    promise_type() : future_() {}

    Promise get_return_object() {
      return Promise(handle_type::from_promise(*this));
    }

    // Do not suspend the coroutine when it begins.
    std::suspend_never initial_suspend() { return {}; }

    // Instead of using std::suspend_* for the final_suspend,
    // we define our own suspend structure that will handle the
    // result of the future.
    struct awaiter {
      // Similar to suspend_always.
      constexpr bool await_ready() noexcept { return false; }
      constexpr void await_resume() noexcept {}
      // The await_suspend method here returns a coro handle
      // that could be used to resume the coro elsewhere.
      std::coroutine_handle<> await_suspend(
          std::coroutine_handle<promise_type> h) noexcept {
        promise_type &promise = h.promise();
        // In case an exception occurred.
        if (promise.curr_exception_) {
          // Catch the exception immediately and convert to iree_status_t.
          // This is done this way because the await function is not allowed
          // to throw an exception.
          try {
            std::rethrow_exception(promise.curr_exception_);
          } catch (std::exception &e) {
            promise.future_.set_failure(iree::exception_to_status(e));
          }
        }
        // The future is now satisfied, with an error or successfully.
        // Therefore, simply return.
        return std::noop_coroutine();
      }
    };

    // Return the custom awaiter when final_suspend is called.
    auto final_suspend() noexcept { return awaiter{}; }

    void unhandled_exception() {
      // If the coro encounters an exception during execution,
      // it will call this method. We store the exception so our
      // awaiter can use it to set the future's status.
      curr_exception_ = std::current_exception();
    }

    template <typename U>
    void return_value(U &&value) {
      future_.set_result(std::forward<U>(value));
    }

    TypedFuture<T> get_future() {
      // Return the future from this promise_type.
      // This function will be called when the caller
      // needs the Promise wrapper's future.
      return future_;
    }
  };

  auto get_future() { return handle_.promise().get_future(); }

  void wait() {
    auto future = get_future();
    future.Wait();
  }

  T get() {
    auto future = get_future();
    future.Wait();
    return future.result();
  }

 private:
  handle_type handle_;
};

// Specialization of Promise for a VoidFuture.
template <>
class Promise<void> {
 public:
  struct promise_type;
  using handle_type = std::coroutine_handle<promise_type>;

  Promise(handle_type handle) : handle_(handle) {}
  Promise(Promise &&other) : handle_(std::exchange(other.handle_, nullptr)) {}
  Promise(const Promise &other) = delete;
  Promise &operator=(const Promise &) = delete;
  Promise &operator=(Promise &&other) noexcept {
    if (this != &other) {
      if (handle_) {
        handle_.destroy();
        handle_ = std::exchange(other.handle_, nullptr);
      }
    }
    return *this;
  }
  ~Promise() {
    if (handle_) {
      handle_.destroy();
    }
  }

  struct promise_type {
    // promise_type for a TypedFuture.
    // We'll specialise for a VoidFuture.
    VoidFuture future_;
    std::exception_ptr curr_exception_;

    promise_type() : future_() {}

    Promise get_return_object() {
      return Promise(handle_type::from_promise(*this));
    }

    // Do not suspend the coroutine when it begins.
    std::suspend_never initial_suspend() noexcept { return {}; }

    // Return the custom awaiter when final_suspend is called.
    auto final_suspend() noexcept { return awaiter{}; }

    void unhandled_exception() {
      // If the coro encounters an exception during execution,
      // it will call this method. We store the exception so our
      // awaiter can use it to set the future's status.
      curr_exception_ = std::current_exception();
    }

    void return_void() {
      // Do nothing here. Wait for final_suspend to execute.
    }

    VoidFuture get_future() {
      // Return the future from this promise_type.
      // This function will be called when the caller
      // needs the Promise wrapper's future.
      return future_;
    }
  };

  auto get_future() { return handle_.promise().get_future(); }

  void wait() {
    auto future = get_future();
    future.Wait();
    // If a failure occurs, throw.
    future.ThrowFailure();
  }

 private:
  handle_type handle_;
  // Instead of using std::suspend_* for the final_suspend,
  // we define our own suspend structure that will handle the
  // result of the future.
  struct awaiter {
    // Similar to suspend_always.
    constexpr bool await_ready() noexcept { return false; }
    constexpr void await_resume() noexcept {}
    // The await_suspend method here returns a coro handle
    // that could be used to resume the coro elsewhere.
    std::coroutine_handle<> await_suspend(
        std::coroutine_handle<promise_type> h) noexcept {
      promise_type &promise = h.promise();
      // In case an exception occurred.
      if (promise.curr_exception_) {
        // Catch the exception immediately and convert to iree_status_t.
        // This is done this way because the await functio is not allowed
        // to throw an exception.
        try {
          std::rethrow_exception(promise.curr_exception_);
        } catch (std::exception &e) {
          promise.future_.set_failure(iree::exception_to_status(e));
        }
      } else {
        promise.future_.set_success();
      }

      // The future is now satisfied, with an error or successfully.
      // Therefore, simply return.
      return std::noop_coroutine();
    }
  };
};

// Helpers to construct futures.
template <typename T>
static Promise<T> makePromise(TypedFuture<T> future) {
  co_return co_await future;
}

static inline Promise<void> makePromise(VoidFuture future) noexcept {
  co_await future;
}

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_ASYNC_H
