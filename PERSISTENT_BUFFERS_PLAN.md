# Persistent device-resident buffers — closing the 99.9% transfer-overhead gap

**Status:** plan, April 2026.
**Predecessor:** the elementwise-binary shader is fast (0.07 ms at
1M elements on RTX 3060 Ti) but `bench_gpu_add_with_transfer`
takes 50 ms for the same workload. The 49.93 ms is alloc + upload
+ download + free overhead. **This plan kills 99.9% of that.**
**Effort:** ~3–4 days, four sub-iterations that ship independently.

---

## What's wasteful today

The current `bench_gpu_add_with_transfer` shape is what most
naive callers do:

```c
for (each operation) {
    buf_alloc(&a);
    buf_alloc(&b);
    buf_alloc(&c);          // 3 × vkAllocateMemory
    upload(&a, ...);
    upload(&b, ...);        // 2 × CPU→GPU transfer
    dispatch(...);          // ~70 µs of actual compute
    download(&c, ...);      // 1 × GPU→CPU transfer
    buf_free(&a);
    buf_free(&b);
    buf_free(&c);           // 3 × vkFreeMemory
}
```

Per-iteration cost on the 3060 Ti at N=1M elements (4 MB
buffers):

| Phase | Time | Notes |
|---|---|---|
| `buf_alloc` × 3 | ~10 ms | `vkAllocateMemory` + `vkBindBufferMemory`; driver-side, mostly synchronous |
| `upload` × 2 | ~20 ms | host-staging buffer + `vkCmdCopyBuffer` + fence wait |
| `dispatch` | ~0.07 ms | the actual compute |
| `download` × 1 | ~10 ms | inverse of upload |
| `buf_free` × 3 | ~10 ms | `vkFreeMemory` |
| **Total** | **~50 ms** | dispatch is **0.14% of wall time** |

The fix: keep buffers on the device across operations. Calling
code that does N operations on the same tensors pays the alloc +
upload cost **once**, the dispatch cost **N times**, and the
download cost **once**. For a Spirit simulation that does 10,000
elementwise ops on the same spin field, that's 50,000× less
overhead per op.

---

## Architecture decisions to lock

| # | Decision | Default lean |
|---|---|---|
| 1 | Lifetime model | **Caller-owned `VkBuf`.** Caller alloc, caller free. The backend doesn't track buffer lifetimes — too much policy. Caller may build a pool on top if desired. |
| 2 | New API surface | One-call helpers (`vk_alloc_device`, `vk_alloc_host_visible`, `vk_free`) and a "scratch pool" for transient intermediates inside the backend (e.g., reductions). |
| 3 | Allocator strategy | **Direct `vkAllocateMemory` per buffer for now.** Sub-allocation from a single big `VkDeviceMemory` is faster but harder to get right; defer to iteration 5+. |
| 4 | Memory type selection | Default device-local. Host-visible variant (`vk_alloc_host_visible`) for buffers the CPU mmap-reads or writes directly — saves one stage copy. |
| 5 | Suballocation budget | If iteration 5 lands: 256 MB pool per buffer-size class, sub-allocate via free-list. Up to 1 GB total. Spills to direct allocation when full. |
| 6 | Transfer batching | Multi-buffer uploads coalesce into one `vkCmdCopyBuffer` chain in a single command buffer. Fence once. (Iteration 3.) |
| 7 | Persistent staging buffer | One host-visible staging buffer, reused for every transfer. Sized to the largest active buffer. (Iteration 2.) |
| 8 | Fence reuse | One reusable fence per worker thread, reset between submits. (Iteration 4.) |
| 9 | Out of scope | Async transfers (overlapped with compute), separate transfer queue, GPU-direct PCIe peer-to-peer, virtual-memory tricks. All useful, none on the critical path. |

---

## Iteration 1 — Persistent device buffers (caller-owned)

The minimal change. Today's API already supports this — the
caller can hold a `VkBuf` across operations. What's missing is
the **bench evidence** and a documented usage pattern.

### Deliverables

- New benchmark `bench_gpu_add_persistent` that allocates buffers
  once, runs `iters` dispatches against them, downloads once.
  Reports the steady-state per-op cost.
- Update `RESULTS_*.md` with the persistent vs naïve comparison.
- Document the pattern in `Backend_par_vulkan.hpp` near the
  `VkBuf` definition.

### Expected numbers

| N | Naïve (alloc + xfer per op) | Persistent (alloc once) | Ratio |
|---|---|---|---|
| 64 K | ~5 ms | 0.04 ms | ~125× |
| 1 M | ~50 ms | 0.07 ms | ~700× |
| 4 M | ~200 ms | 0.22 ms | ~900× |

The bigger N gets, the bigger the win — large buffers hurt more
under the alloc-each-time pattern.

### Acceptance

Persistent benchmark numbers documented. Caller usage pattern
written down. No backend code changes (pure additions).

---

## Iteration 2 — Persistent staging buffer

The current `upload` and `download` both allocate a host-visible
staging buffer per call, copy via `vkCmdCopyBuffer`, fence-wait,
free. The staging allocation is the cost.

### Strategy

One host-visible staging buffer per thread, sized to the largest
upload/download in flight. Grown geometrically (×2) when an op
exceeds capacity. Lives in `g_vk_ctx`, freed at `vk_destroy`.

### API change

`upload`/`download` signatures unchanged. Internals route through
the persistent staging buffer instead of allocating a fresh one.

### Test

Existing `test_vulkan_init` TEST 2 (round-trip) still passes —
the contract is unchanged.

### Expected numbers

Cuts upload + download cost roughly in half — the
`vkAllocateMemory` step disappears, the `vkCmdCopyBuffer` +
fence stay.

| N | Today (per upload) | After (per upload) |
|---|---|---|
| 1 M | ~10 ms | ~5 ms |
| 4 M | ~40 ms | ~20 ms |

### Risk

Two threads calling `upload` concurrently against the same
staging buffer is a data race. **Lock the staging buffer with a
per-thread copy** if multi-threaded callers materialise. For
single-threaded usage (current bench, current Spirit), one
staging buffer is fine.

---

## Iteration 3 — Transfer batching

Today `upload(a)`, `upload(b)` is two command-buffer submits, two
fences. Most call patterns upload several buffers at once
(matmul = A + B + C; conv = input + kernel + output).

### Strategy

New `upload_batch(VkBuf** bufs, void** datas, size_t* sizes,
int n)` builds one command buffer with N `vkCmdCopyBuffer` calls,
submits once, fences once. For N=2 this is ~2× faster than two
individual uploads; for N=3 ~3×.

### Implementation

Stage all N source datas into the persistent staging buffer at
distinct offsets. Issue N copy commands. Single submit, single
fence wait.

### Acceptance

`upload_batch` matches `upload` × N for correctness on a 4-buffer
test fixture. Benchmark shows a ~2–3× speedup vs sequential
uploads.

---

## Iteration 4 — Fence reuse

`submit_and_wait` (current) creates a fence per submit, waits,
destroys. `vkCreateFence` + `vkDestroyFence` are non-trivial
driver calls.

### Strategy

One reusable fence per worker thread, stashed in `g_vk_ctx`.
Created at `vk_init`, destroyed at `vk_destroy`. Reset
(`vkResetFences`) between submits. Saves ~50 µs per dispatch.

### Acceptance

Per-dispatch overhead drops by ~50 µs at N ≤ 64K (where dispatch
overhead is the floor). No effect at N = 1M+ (compute dominates).

---

## Iteration 5 — Sub-allocator (optional, deferred)

Only worth it if profiling shows `vkAllocateMemory` is still
the hot spot after iters 1–4. Most workloads never re-allocate
buffers — they reuse the persistent ones — so this is the
optimisation that may turn out to be unnecessary.

### Strategy

One large `VkDeviceMemory` arena per memory type. Free-list
allocator with size-class buckets. `buf_alloc` carves chunks from
the arena instead of calling `vkAllocateMemory`. `buf_free`
returns the chunk to the free list.

### Risk

Memory fragmentation under non-uniform buffer sizes. Bucket
sizes (rounded to powers of 2) help but don't eliminate. If
the workload's allocation pattern is regular (Nx tensors of
predictable shapes), bucketing handles it. If irregular,
fragmentation accumulates.

### Acceptance

`buf_alloc` is sub-microsecond on a warm pool. Memory waste
under typical workloads stays under 25%. Spirit's simulation
loop and a representative Nx graph both run without OOM.

---

## What this plan DOES NOT cover

- **Async transfers** — copies overlap with compute. Requires a
  dedicated transfer queue and semaphore-based sync. ~1 day of
  work; not on the critical path because synchronous transfers
  are already 99% of the time on transfers, not on stalls.
- **Pinned host memory** — `VK_MEMORY_PROPERTY_HOST_CACHED_BIT`
  for faster CPU access of the staging buffer. Marginal; skip.
- **GPU-direct PCIe** — peer-to-peer transfers between two GPUs.
  Multi-GPU concern; out of scope.
- **Streaming uploads** — chunked overlapping `vkCmdCopyBuffer`
  with compute on adjacent chunks. Useful for very large
  datasets that don't fit in VRAM. Not relevant to current
  Spirit/Nx workloads.

---

## Order of operations

| Step | Effort | Depends on | Outcome |
|---|---|---|---|
| 1. Persistent device buffers + bench | 0.5 d | — | 100–900× speedup, documented pattern |
| 2. Persistent staging buffer | 1 d | — | ~2× upload/download speedup |
| 3. Transfer batching | 1 d | step 2 | ~3× speedup on multi-buffer uploads |
| 4. Fence reuse | 0.5 d | — | ~50 µs/dispatch saved at small N |
| 5. Sub-allocator | 1 d (optional) | profile shows need | sub-µs `buf_alloc` |

Critical path: **steps 1 + 2 = 1.5 days** to claw back the
biggest wins. Steps 3 + 4 are smaller wins; step 5 is on-demand.

---

## Acceptance criteria (across all iterations)

- [ ] `bench_gpu_add_persistent` shows ≥100× speedup over
      `bench_gpu_add_with_transfer` at N=1M.
- [ ] `bench_gpu_add` (dispatch-only) drops by ≥30 µs at N=1K
      after fence reuse.
- [ ] `RESULTS_*.md` updated with new numbers; old naïve numbers
      retained for the comparison.
- [ ] No regression in correctness — existing 3-test suite passes
      unchanged.
- [ ] Memory leak-free under repeated alloc/free cycles
      (verified via `valgrind --tool=memcheck` on Linux, by
      `pkg install valgrind`).

---

## What this opens up

The persistent-buffer story is the gating optimisation for **any
real Nx-style workload**. With it:

- **Spirit**: simulation loops keep the spin field on-device for
  thousands of iterations. The current ~50 ms per-op overhead
  drops to ~0.07 ms. A 10,000-iteration simulation runs in 0.7
  seconds instead of ~500 seconds.
- **Nx.Vulkan**: tensors created by `Nx.tensor/2` allocate device
  memory once. Subsequent operations chain dispatches without
  ever round-tripping data to host. Matches the EXLA / EMLX
  contract that Nx callers expect.
- **Inference benchmarks**: model weights load to device once at
  inference-server boot. Per-request work is dispatch-only.
  Latency budget gets the 99.9% back.

The blog headline number changes from "8.8× at 1M" to "the
persistent-buffer story changes the dispatch overhead from 99.9%
of wall time to 0.1%." That's the post-optimisation pitch and the
real story for FreeBSD GPU compute.

---

## Cross-references

- [SHADERS_PLAN.md](SHADERS_PLAN.md) — companion plan for what
  shaders consume the persistent buffers.
- [RESULTS_RTX_3060_TI.md](RESULTS_RTX_3060_TI.md) — current
  numbers; updated after each iteration.
- `core/include/engine/Backend_par_vulkan.hpp` — the API this
  plan extends.
- `bench_vulkan.cpp` — where the new `bench_gpu_add_persistent`
  lands.
