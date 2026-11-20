#pragma once

// Torch-free replacement for at::parallel_for, and the pool every native
// parallel decode in torchfits runs on (no native target links libtorch).
// The API mirrors the at::parallel_for contract the codebase relies on --
// inclusive/exclusive range [begin, end), a grain size in work units, and a
// body invoked as body(begin, end) that must be safe to run concurrently.
//
// Thread count: TORCHFITS_NUM_THREADS when set, otherwise the hardware
// concurrency, clamped to [1, 64].  A single caller thread plus the pool
// workers means at most `thread_count()` threads are ever running user code.
//
// The calling thread claims chunks alongside the workers, so a call never
// waits on a worker that has not started; idle workers poll briefly before
// sleeping, so back-to-back calls (one per table column) skip the wake-up.
//
// Nesting a core parallel_for inside another one is safe but does not fan out
// a second time: a call made from inside a pool worker runs its chunks inline
// on that worker.  Submitting from a worker and waiting for the results is a
// measured deadlock at every pool size >= 2, because the workers that would
// take the work are themselves inside user code.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "core_export.h"

namespace torchfits {
namespace core {

// Effective worker count (>= 1). Resolved once per process.
TORCHFITS_CORE_API int thread_count();

namespace detail {

// Type-erased chunk runner. The template wrapper below adapts a lambda to this
// signature without a std::function allocation on the hot path.
using ParallelThunk = void (*)(void* ctx, std::int64_t begin, std::int64_t end);

TORCHFITS_CORE_API void run_chunks(
    std::int64_t begin,
    std::int64_t end,
    std::int64_t grain_size,
    ParallelThunk thunk,
    void* ctx
);

}  // namespace detail

// Split [begin, end) into ceil((end - begin) / grain_size) chunks and run them
// across the pool. Work smaller than one grain runs inline on the calling
// thread: spawning helpers for a 4 KiB range costs far more than the range.
template <typename F>
inline void parallel_for(std::int64_t begin, std::int64_t end, std::int64_t grain_size, F&& body) {
    using Body = typename std::remove_reference<F>::type;
    auto thunk = [](void* ctx, std::int64_t b, std::int64_t e) {
        (*static_cast<Body*>(ctx))(b, e);
    };
    detail::run_chunks(begin, end, grain_size, thunk, &body);
}

}  // namespace core
}  // namespace torchfits
