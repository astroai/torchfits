#include "core/parallel.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace torchfits {
namespace core {
namespace {

// One parallel_for call's worth of work. Chunks are claimed dynamically
// through `next`: pool workers that picked up a ticket for this batch and the
// calling thread itself all take the next unclaimed chunk until none is left.
// The caller therefore never idles behind a worker that has not woken up yet
// (for a range of a few grains it usually runs every chunk itself), and a
// slow core (an efficiency core, a preempted worker) simply claims fewer
// chunks. The caller waits only for chunks a worker has already claimed, then
// rethrows the first captured exception on its own stack, so callers keep the
// "throws out of parallel_for" contract at::parallel_for provides.
struct Batch {
    detail::ParallelThunk thunk = nullptr;
    void* ctx = nullptr;
    std::int64_t begin = 0;
    std::int64_t end = 0;
    std::int64_t grain = 1;
    std::int64_t chunks = 0;
    std::atomic<std::int64_t> next{0};
    std::atomic<std::int64_t> done{0};
    std::mutex mutex;
    std::condition_variable cv;
    std::exception_ptr error;

    // Run unclaimed chunks until none remain. Every chunk runs even after one
    // throws (as before); the first exception is kept.
    void drain() {
        for (;;) {
            const std::int64_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= chunks) return;
            const std::int64_t lo = begin + i * grain;
            const std::int64_t hi = (lo + grain < end) ? lo + grain : end;
            std::exception_ptr err;
            try {
                thunk(ctx, lo, hi);
            } catch (...) {
                err = std::current_exception();
            }
            if (err) {
                std::lock_guard<std::mutex> lock(mutex);
                if (!error) error = err;
            }
            if (done.fetch_add(1, std::memory_order_acq_rel) + 1 == chunks) {
                // Taking the mutex orders this notify after the waiter's
                // predicate check, so the wake-up cannot be lost.
                std::lock_guard<std::mutex> lock(mutex);
                cv.notify_all();
            }
        }
    }
};

// A worker's handle on a batch. After the batch is fully claimed a late
// ticket finds nothing to do; it never touches `ctx` (the caller's stack).
struct Task {
    std::shared_ptr<Batch> batch;
};

// True while this thread is executing user code on behalf of the pool.
//
// A worker must never submit a parallel_for and then wait for it. The workers
// that would pick the work up may themselves all be inside user code, so the
// wait can never be satisfied: measured deadlock at every pool size >= 2. A
// nested parallel_for is therefore run inline on the worker that asked for it.
thread_local bool t_is_pool_worker = false;

// Fixed-size worker pool. Created on first use and deliberately never
// destroyed: workers touch process-lifetime state, and a static destructor
// joining them would run after the CFITSIO/nanobind teardown that a late
// parallel_for could otherwise still reach.
class Pool {
public:
    explicit Pool(int worker_count) {
        workers_.reserve(static_cast<size_t>(worker_count));
        for (int i = 0; i < worker_count; ++i) {
            // A thread that cannot be started (EAGAIN under a thread limit)
            // must not unwind the constructor: destroying the joinable workers
            // already started calls std::terminate. Keep those; with none,
            // run_chunks runs every batch on the calling thread.
            try {
                workers_.emplace_back([this] { worker_loop(); });
            } catch (const std::system_error&) {
                break;
            }
        }
    }

    int worker_count() const { return static_cast<int>(workers_.size()); }

    // Queue `tickets` handles on `batch`. Workers still spinning after their
    // last batch pick them up without a wake-up; only the shortfall is woken,
    // one notify per sleeping worker needed (a broadcast for a four-chunk
    // batch woke every idle worker to contend for the queue lock).
    void submit(const std::shared_ptr<Batch>& batch, int tickets) {
        if (tickets <= 0) return;
        int wake = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (int i = 0; i < tickets; ++i) queue_.push_back(Task{batch});
            pending_.fetch_add(tickets, std::memory_order_release);
            const int spinning = spinners_.load(std::memory_order_acquire);
            wake = std::min(tickets - std::min(tickets, spinning), sleepers_);
        }
        for (int i = 0; i < wake; ++i) work_cv_.notify_one();
    }

private:
    // How long an idle worker keeps polling for the next batch before it
    // sleeps on the condition variable. Table decodes issue one parallel_for
    // per column back to back; a futex wake per worker per column cost about
    // 8 us per call (measured: 10-column, 10k-row window 0.31 ms vs 0.23 ms
    // serial) while the work per chunk is a few microseconds. ATen's OpenMP
    // pool, which the decoders used before, spins for 200 ms by default; a
    // short spin keeps the back-to-back case fast without holding idle cores.
    static constexpr auto kSpin = std::chrono::microseconds(100);

    bool try_pop(Task& task) {
        if (pending_.load(std::memory_order_acquire) == 0) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        task = std::move(queue_.front());
        queue_.pop_front();
        pending_.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }

    void worker_loop() {
        for (;;) {
            Task task;
            bool got = false;
            spinners_.fetch_add(1, std::memory_order_acq_rel);
            const auto deadline = std::chrono::steady_clock::now() + kSpin;
            for (unsigned i = 0; !got; ++i) {
                got = try_pop(task);
                if (got) break;
                if ((i & 63u) == 63u && std::chrono::steady_clock::now() >= deadline) break;
                std::this_thread::yield();
            }
            spinners_.fetch_sub(1, std::memory_order_acq_rel);
            if (!got) {
                std::unique_lock<std::mutex> lock(mutex_);
                ++sleepers_;
                work_cv_.wait(lock, [this] { return !queue_.empty(); });
                --sleepers_;
                task = std::move(queue_.front());
                queue_.pop_front();
                pending_.fetch_sub(1, std::memory_order_relaxed);
            }
            // drain() catches everything the user body throws.
            t_is_pool_worker = true;
            task.batch->drain();
            t_is_pool_worker = false;
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::deque<Task> queue_;
    std::atomic<int> pending_{0};   // queue_.size(), readable without the lock
    std::atomic<int> spinners_{0};  // workers polling, not yet asleep
    int sleepers_ = 0;              // workers in work_cv_.wait (under mutex_)
};

// Resolve TORCHFITS_NUM_THREADS once (PyTorch's own thread settings do not
// size this pool).
int resolve_thread_count() {
    int threads = 0;
    if (const char* raw = std::getenv("TORCHFITS_NUM_THREADS")) {
        try {
            threads = std::stoi(std::string(raw));
        } catch (const std::exception&) {
            threads = 0;
        }
    }
    if (threads <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        threads = hw > 0 ? static_cast<int>(hw) : 1;
    }
    if (threads > 64) threads = 64;
    if (threads < 1) threads = 1;
    return threads;
}

// The pool pointer lives at namespace scope rather than inside a function-local
// static so the fork handlers below can drop it. Creation is double-checked
// under g_pool_mutex: two threads making their first parallel_for call at once
// each used to see null and each build a pool, leaking the loser's workers.
std::atomic<Pool*> g_pool{nullptr};
std::mutex g_pool_mutex;

// fork() hands the child a copy of this address space: the std::thread objects
// the pool holds name threads that do not exist in the child, so a batch queued
// there is never worked on and the caller waits on its condition variable
// forever. Measured: a child that had not forked before the pool was warm hung
// until SIGALRM (10s) with no output. `tests/cpp/test_fork_after_pool.cpp`
// guards it. The mutex is held across fork() so the child never inherits it
// locked by a thread that does not exist there; the forking thread owns it on
// both sides and releases it after the child has dropped the pool.
void lock_pool_before_fork() { g_pool_mutex.lock(); }
void unlock_pool_in_parent() { g_pool_mutex.unlock(); }
void drop_pool_in_forked_child() {
    g_pool.store(nullptr, std::memory_order_relaxed);
    g_pool_mutex.unlock();
}

Pool& pool() {
    static const bool registered = [] {
        pthread_atfork(lock_pool_before_fork, unlock_pool_in_parent, drop_pool_in_forked_child);
        return true;
    }();
    (void)registered;
    Pool* current = g_pool.load(std::memory_order_acquire);
    if (current) return *current;
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    current = g_pool.load(std::memory_order_relaxed);
    if (!current) {
        current = new Pool(resolve_thread_count() - 1);
        g_pool.store(current, std::memory_order_release);
    }
    return *current;
}

}  // namespace

int thread_count() {
    static const int count = resolve_thread_count();
    return count;
}

namespace detail {

void run_chunks(
    std::int64_t begin,
    std::int64_t end,
    std::int64_t grain_size,
    ParallelThunk thunk,
    void* ctx
) {
    if (end <= begin) return;
    if (grain_size < 1) grain_size = 1;

    // Nested inside a parallel_for: run the chunks serially here rather than
    // queueing them for workers we would then have to wait on. The work still
    // gets done and the throw contract is preserved; it just does not fan out
    // again, which is the correct trade against a guaranteed deadlock.
    if (t_is_pool_worker) {
        std::exception_ptr nested_error;
        for (std::int64_t lo = begin; lo < end; lo += grain_size) {
            const std::int64_t hi = (lo + grain_size < end) ? lo + grain_size : end;
            try {
                thunk(ctx, lo, hi);
            } catch (...) {
                if (!nested_error) nested_error = std::current_exception();
            }
        }
        if (nested_error) std::rethrow_exception(nested_error);
        return;
    }

    Pool& workers = pool();
    const std::int64_t total = end - begin;
    const std::int64_t chunks = (total + grain_size - 1) / grain_size;

    // One chunk, or no workers to hand it to: stay on the calling thread.
    // Metadata scans are routinely smaller than one grain, and waking the
    // pool for those would cost more than the work itself.
    if (chunks < 2 || workers.worker_count() == 0) {
        thunk(ctx, begin, end);
        return;
    }

    // The caller drains the batch alongside the workers, so a parallel_for
    // nested on the *calling* thread cannot deadlock waiting on workers the
    // outer call owns: it can always finish every unclaimed chunk itself, and
    // a chunk a worker claimed runs to completion. Worker-side nesting never
    // reaches here: it is handled by the t_is_pool_worker branch above, which
    // never waits.
    auto batch = std::make_shared<Batch>();
    batch->thunk = thunk;
    batch->ctx = ctx;
    batch->begin = begin;
    batch->end = end;
    batch->grain = grain_size;
    batch->chunks = chunks;
    const std::int64_t helpers = std::min<std::int64_t>(
        chunks - 1, static_cast<std::int64_t>(workers.worker_count()));
    workers.submit(batch, static_cast<int>(helpers));

    batch->drain();

    std::exception_ptr error;
    {
        std::unique_lock<std::mutex> lock(batch->mutex);
        batch->cv.wait(lock, [&batch] {
            return batch->done.load(std::memory_order_acquire) == batch->chunks;
        });
        error = batch->error;
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

}  // namespace detail
}  // namespace core
}  // namespace torchfits
