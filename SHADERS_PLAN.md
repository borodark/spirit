# Shader plan — what to build next on the Vulkan compute backend

**Status:** plan, April 2026.
**Predecessors:** the elementwise binary shader is done; tests + bench
green on FreeBSD GT 750M and Linux RTX 3060 Ti
(`RESULTS_RTX_3060_TI.md`). The 540-line backend handles dispatch,
allocation, and pipeline lifecycle; everything below adds a shader
and a test, no backend changes unless explicitly noted.
**Effort:** ~2–3 weeks total across seven iterations. Each iteration
is independently shippable — pause anywhere and the previous
iteration's shader still works.

---

## Priorities

The downstream consumer is **Nx.Vulkan** (Elixir tensor backend).
Nx's hot path is well-known: matmul dominates ML workloads;
elementwise + unary covers the kernel zoo; reductions pin the
loss/gradient codepaths; conv 2D unlocks vision. We build in the
order that maximises early Nx coverage, with Spirit-specific ops
slotted last (they're niche and the team has CUDA fallbacks).

| # | Shader | What it unlocks | Effort | Risk |
|---|---|---|---|---|
| 1 | **Reductions (sum/mean/min/max)** | Loss functions, statistics, normalisation | 2 d | low |
| 2 | **Unary fused** (parametric, many ops) | exp/log/sigmoid/tanh/relu/abs/neg/sqrt | 1 d | low |
| 3 | **Matmul (naive)** | Linear layers, embeddings, attention | 1 d | low |
| 4 | **Matmul (tiled)** | Same surface, 5–20× faster | 2 d | medium |
| 5 | **Softmax** (reduction + elementwise + reduction) | Attention, classification heads | 1 d | low |
| 6 | **Conv 2D (direct)** | CNN inference, simple vision | 2 d | medium |
| 7 | **Spirit physics ops** (spin rotate, Heisenberg energy) | Spirit's own simulation loop | 2 d | low |

Total: ~11 person-days. Single-host work; no cross-platform retest
needed beyond the existing two-machine workflow.

---

## Architecture decisions to lock

| # | Decision | Default lean |
|---|---|---|
| 1 | Shader source layout | `shaders/<op>.comp` per op family. Pre-compile to `.spv`, check both into the repo. Build script regenerates `.spv` from `.comp`. |
| 2 | Workgroup size | **256 by default** (good occupancy on Ampere/RDNA/Kepler/MoltenVK). Per-op overrides for matmul (16×16 tiles) and conv (8×8 tiles). |
| 3 | Push-constant size | Cap at 128 bytes (the FreeBSD GT 750M's limit). Larger configs go through a uniform buffer; not needed for any shader in this plan. |
| 4 | Buffer layout | Row-major, contiguous, f32 default; f64 supported via spec constant when `has_float64` is true. |
| 5 | Dynamic vs static shapes | **Dynamic.** Push constants carry N (1D) or {M, N, K} (matmul) or {N, C, H, W} (conv). One pipeline per op family, not per-shape. Matches Nx's runtime-shape model. |
| 6 | Reduction strategy | **Two-pass**: workgroup-local reduce in shared memory (256→1), then a second dispatch reduces the intermediate. Avoids non-deterministic atomic ordering. |
| 7 | Specialisation constants | Used for ops within a family (add/sub/mul/div, exp/log/...). `vkSpecializationInfo` at pipeline-create time means one shader file, many pipelines. |
| 8 | Test framework | Each shader gets a `test_<op>.cpp` with hand-computed truth values. Fixtures small (≤256 elements) so failures are inspectable. |
| 9 | Benchmark integration | Extend `bench_vulkan.cpp` with one row per op. Same N range (1K → 4M). Same dispatch-only + GPU+xfer columns. |
| 10 | Out-of-scope | No autograd, no training, no fp16, no bfloat, no quantization, no kernel fusion. Each shader is one `vkCmdDispatch`. Composition happens in Nx, not in the backend. |

---

## Iteration 1 — Reductions

Two-pass tree reduction. The first pass takes N elements and
produces ⌈N/256⌉ partials; the second pass takes those partials
and produces 1 final value (or runs again if needed).

### Shader contract

```glsl
#version 450
#extension GL_EXT_shader_atomic_float : enable

layout(local_size_x = 256) in;

layout(binding = 0) buffer In  { float in_data[];  };
layout(binding = 1) buffer Out { float out_data[]; };

layout(push_constant) uniform PC {
    uint  n;       // input length
    uint  op;      // 0=sum, 1=min, 2=max, 3=mean (sum + final-divide)
} pc;

shared float sdata[256];

// ... tree reduction ...
```

### Backend additions

- New helper `reduce(VkBuf* in, VkBuf* out, int N, ReductionOp op)`
  in `Backend_par_vulkan.hpp` — handles the two-pass dispatch
  internally, allocates the intermediate buffer if N > 256.
- Allocator gains a small "scratch" pool (one buffer reused
  across reductions of the same size class) to avoid per-call
  alloc/free.

### Test cases

- Sum of `[1..1000]` = 500500
- Min of `[3, 1, 4, 1, 5, 9, 2, 6]` = 1
- Max of same = 9
- Mean of `[2, 4, 6, 8]` = 5.0
- Sum of `[1.0]` = 1.0 (single-element edge case)
- Sum of empty buffer = 0.0 (zero-length edge case; may need
  early return on host side)

### Acceptance

- All four ops (sum/min/max/mean) correct on f32
- Two-pass logic kicks in correctly for N > 256
- Performance comparable to a single elementwise dispatch at
  matched memory traffic (~50% of peak bandwidth target)

---

## Iteration 2 — Unary fused

One parametric shader that covers many unary ops, dispatched via
spec constant. Saves on pipeline-create overhead and shader-cache
size.

### Shader contract

```glsl
#version 450
layout(local_size_x = 256) in;
layout(constant_id = 0) const uint OP = 0u;

layout(binding = 0) buffer In  { float in_data[];  };
layout(binding = 1) buffer Out { float out_data[]; };

layout(push_constant) uniform PC { uint n; } pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    float x = in_data[i];
    float y;
    switch (OP) {
        case 0u: y = exp(x);    break;
        case 1u: y = log(x);    break;
        case 2u: y = sqrt(x);   break;
        case 3u: y = abs(x);    break;
        case 4u: y = -x;        break;
        case 5u: y = 1.0 / (1.0 + exp(-x));  break;  // sigmoid
        case 6u: y = tanh(x);   break;
        case 7u: y = max(x, 0.0); break;             // relu
    }
    out_data[i] = y;
}
```

### Tests

Per op, 8-element input with hand-computed truth. The expected
values for sigmoid/tanh use `expf()` from libm so the comparison
is GPU-vs-CPU on the same math primitive (allowable epsilon
1e-5 for f32).

### Acceptance

All eight ops correct. Per-op pipeline create takes <10 ms.
Dispatch identical to elementwise binary in shape.

---

## Iteration 3 — Matmul (naive)

Each thread computes one element of C. C[m,n] = Σ A[m,k] * B[k,n].
Bandwidth-inefficient (every thread re-reads K elements) but
correct, and the baseline for the tiled version.

### Shader contract

```glsl
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) buffer A_buf { float A[]; };
layout(binding = 1) buffer B_buf { float B[]; };
layout(binding = 2) buffer C_buf { float C[]; };

layout(push_constant) uniform PC {
    uint M;
    uint N;
    uint K;
} pc;

void main() {
    uint m = gl_GlobalInvocationID.y;
    uint n = gl_GlobalInvocationID.x;
    if (m >= pc.M || n >= pc.N) return;

    float acc = 0.0;
    for (uint k = 0u; k < pc.K; k++) {
        acc += A[m * pc.K + k] * B[k * pc.N + n];
    }
    C[m * pc.N + n] = acc;
}
```

### Backend additions

- New helper `matmul(VkBuf* A, VkBuf* B, VkBuf* C, int M, int N, int K)`
  that dispatches with 2D workgroup grid `((N+15)/16, (M+15)/16)`.

### Tests

Identity matmul (A·I = A). 4×4 hand-computed example. Mismatched
dimensions error on the host side before dispatch.

### Acceptance

Correct on small fixtures. Naive matmul should hit ~50–100 GFLOPS
on the 3060 Ti — well below the 16 TFLOPS peak; the tiled version
fixes that.

---

## Iteration 4 — Matmul (tiled)

Workgroup of 16×16 = 256 threads cooperatively load tiles of A and
B into shared memory, compute a 16×16 tile of C. Each thread does
K/16 inner iterations across tiles. Memory traffic drops from
O(MNK) reads to O(MN(K/16)) per tile.

### Notes

- Shared memory budget on most cards: 48 KB. Two 16×16 f32 tiles
  = 2 KB. Plenty of room for 32×32 tiles or 16×32 rectangular
  tiles if benchmarks suggest gains.
- Boundary handling: pad with zeros for non-multiple-of-16 sizes.
  Simpler than a separate edge-case kernel.
- f64 doubles shared-memory cost — keep tile size at 16×16 for
  the f64 pipeline.

### Risk

Tiled matmul on Vulkan takes longer to land than expected because
shared-memory access patterns interact with bank conflicts in
non-obvious ways. The naive version (iter 3) ships first so Nx
can run on something while we tune the tiled version. Target:
the tiled matmul does ≥1 TFLOPS on the 3060 Ti at M=N=K=1024.

### Acceptance

Same correctness suite as iter 3. Bench shows ≥5× speedup over
the naive version at M=N=K=1024.

---

## Iteration 5 — Softmax

Three-stage: max-reduce, elementwise `exp(x - max)`, sum-reduce,
elementwise `÷sum`. Composes the existing reduction + unary +
elementwise pipelines; **no new shader** if iter 1 + 2 are clean.

### Backend additions

- New helper `softmax(VkBuf* in, VkBuf* out, int N)` that
  orchestrates the four-dispatch sequence with a small scratch
  buffer.

### Tests

Softmax of `[1.0, 2.0, 3.0]` matches `[0.090, 0.244, 0.665]`
(epsilon 1e-4).

### Acceptance

Matches `Nx.softmax/1` to 1e-4 on a few hand-curated inputs.

---

## Iteration 6 — Conv 2D (direct)

Single-channel-in, single-channel-out, stride 1, padding 0. Each
output pixel sums the elementwise product of a kernel-sized
neighbourhood from the input. Direct convolution; no im2col, no
Winograd.

### Shader contract

```glsl
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) buffer In_buf  { float in_data[];  };
layout(binding = 1) buffer K_buf   { float kernel[];   };
layout(binding = 2) buffer Out_buf { float out_data[]; };

layout(push_constant) uniform PC {
    uint H;     // input height
    uint W;     // input width
    uint kH;    // kernel height
    uint kW;    // kernel width
} pc;

void main() {
    uint y = gl_GlobalInvocationID.y;
    uint x = gl_GlobalInvocationID.x;
    uint outH = pc.H - pc.kH + 1u;
    uint outW = pc.W - pc.kW + 1u;
    if (y >= outH || x >= outW) return;

    float acc = 0.0;
    for (uint ky = 0u; ky < pc.kH; ky++) {
        for (uint kx = 0u; kx < pc.kW; kx++) {
            acc += in_data[(y + ky) * pc.W + (x + kx)] *
                   kernel[ky * pc.kW + kx];
        }
    }
    out_data[y * outW + x] = acc;
}
```

### Tests

3×3 identity kernel returns the centre crop. 3×3 averaging kernel
on a constant input returns the constant. 3×3 Sobel-X on a
gradient returns ones.

### Acceptance

Correct. Performance is whatever direct conv gets (slower than
im2col+matmul, but enough for inference of small models). Tuning
+ multi-channel support are deferred to a later iteration.

### Out of scope (for this iteration)

Multi-channel input/output, stride ≠ 1, padding ≠ 0, dilation,
group conv. Each adds a push-constant field and a few lines of
GLSL but blows up the test matrix.

---

## Iteration 7 — Spirit physics ops

These are Spirit-specific kernels that the simulator's hot loop
already calls into the CUDA backend. We translate them to Vulkan
so Spirit gets GPU compute on FreeBSD without reaching for CUDA.

### Inventory (from `Backend_par_cuda.hpp`)

- `apply_spin_rotation(spins, axis, angle)` — Rodrigues' formula,
  one rotation per spin.
- `heisenberg_energy(spins, neighbours, exchange_J)` — sum over
  pairs of `J * (s_i · s_j)`. Reduction.
- `dmi_energy(spins, neighbours, dm_vec)` — sum over pairs of
  `D · (s_i × s_j)`. Reduction.
- `effective_field(spins, ...)` — gradient of energy w.r.t.
  spins. Per-spin output.
- `gradient_descent_step(spins, field, eta)` — elementwise; reuses
  the binary shader with a small wrapper.

### Strategy

Each becomes a `.comp` file. Heisenberg + DMI energies use the
reduction primitive from iter 1. Effective field is a custom
kernel; a few hours each.

### Acceptance

Spirit's existing CUDA test suite (energies match analytic values
for known configurations) passes against the Vulkan backend on
FreeBSD. Numerical tolerance same as the CUDA path.

---

## Test fixture pattern (applies to every shader)

```cpp
// test_<op>.cpp — same shape as test_vulkan_init.cpp
//
//   ./test_<op>           runs all sub-tests
//   ./test_<op> --bench   runs the benchmark variants

#include <engine/Backend_par_vulkan.hpp>
#include "test_helpers.hpp"

int test_<op>_correctness() {
    /* small fixed inputs, hand-computed truth, epsilon compare */
}

int test_<op>_edge_cases() {
    /* N=0, N=1, N=workgroup_size-1, N=workgroup_size+1 */
}

int main() {
    if (vk_init(0) != 0) return 1;
    if (test_<op>_correctness() != 0) return 1;
    if (test_<op>_edge_cases() != 0) return 1;
    vk_destroy();
    printf("=== ALL TESTS PASSED ===\n");
    return 0;
}
```

A small `test_helpers.hpp` collects assertion macros (`ASSERT_EQ`,
`ASSERT_NEAR`, `ASSERT_VEC_NEAR`) so each test file is short.

---

## Benchmark integration

`bench_vulkan.cpp` grows one section per shader family. Shape:

```
=== reductions (sum) ===
N           CPU (ms)    GPU (ms)    speedup
65536        0.020       0.041       0.49x
1048576      0.323       0.075       4.31x
...

=== matmul ===
M×N×K               CPU (ms)    GPU naive   GPU tiled  speedup_tiled
128×128×128          0.50        0.04        0.02        25.0x
512×512×512         32.10        2.10        0.21       152.9x
...
```

The CPU baselines are single-core unoptimised loops, same as the
elementwise bench. The Linux RTX 3060 Ti and FreeBSD GT 750M
results both go in `RESULTS_*.md` files.

---

## What this DOES NOT cover

- **Autograd.** No backward kernels. Forward-only inference.
- **Training.** No gradient computation, no optimiser, no
  parameter updates as a backend concern.
- **Mixed precision.** f32 is the default; f64 where supported.
  No fp16, no bf16. Adding them later is a per-shader spec
  constant.
- **Sparse tensors.** Dense only.
- **Distributed compute.** Single-GPU. Multi-GPU is a separate
  iteration after persistent device buffers land.
- **Kernel fusion.** Each `vkCmdDispatch` is one shader. Fusion
  is a Nx-side concern (the Nx graph optimiser may emit fused
  pipelines later).
- **Dynamic dispatch shapes.** Workgroup size is fixed at
  pipeline creation. Re-creating pipelines for different
  workgroup sizes is not free; the default 256 is good enough
  for everything in this plan.

---

## Cross-references

- [HANDOFF_LINUX_VULKAN.md](HANDOFF_LINUX_VULKAN.md) — the run
  that got us to a 3-test-pass baseline.
- [RESULTS_RTX_3060_TI.md](RESULTS_RTX_3060_TI.md) — what the
  current shader achieves on real hardware.
- `core/include/engine/Backend_par_vulkan.hpp` — the public API
  every new shader plugs into.
- `core/src/engine/Backend_par_vulkan.cpp` — the dispatch
  primitive every new shader uses unchanged.
- `shaders/elementwise_binary.comp` — the reference shader
  shape; new shaders follow the same layout (binding 0/1/2, push
  constants, spec constants, workgroup-aligned launch).
