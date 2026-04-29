# Handoff: Vulkan compute shaders — Mac (free-macpro-gpu, .247)

You wrote the original `elementwise_binary.comp` and have the
`glslangValidator` toolchain. The Linux box (.249) doesn't, and
sudo isn't available there. Clean split:

**Mac side** — write `.comp` files, compile to `.spv`, push both.
**Linux side** — integrate into bench + test, run on RTX 3060 Ti,
report numbers, update `RESULTS_RTX_3060_TI.md`.

---

## Briefing (paste verbatim)

> You are continuing work on the **Spirit** project's Vulkan
> compute backend. Branch: `feature/vulkan-backend`. Read first:
>
> 1. `git fetch origin && git checkout feature/vulkan-backend && git pull --ff-only`
> 2. `SHADERS_PLAN.md` — seven-iteration roadmap; you're working
>    on iteration 1 (reductions).
> 3. `RESULTS_RTX_3060_TI.md` — current bench numbers, including
>    the just-landed persistent-buffer iter-1 results.
> 4. `shaders/elementwise_binary.comp` — the reference shape:
>    spec constant for op family, push constants for shape,
>    standard binding 0/1/2 layout.
>
> ### Toolchain
>
> You should already have `glslangValidator` installed (you used
> it for the elementwise shader). Confirm:
>
> ```sh
> glslangValidator --version
> ```
>
> If missing on this Mac:
>
> ```sh
> doas pkg install glslang   # FreeBSD
> # or
> brew install glslang        # if on macOS
> ```
>
> ### Your scope: SHADERS_PLAN.md iterations 1–4
>
> Write each shader, compile to `.spv`, commit both source AND the
> compiled binary. The Linux session integrates each one as it
> lands. Order matters — finish iter 1 before starting iter 2,
> push between each so Linux can run on real hardware while you
> work on the next.
>
> #### Iter 1: reductions
>
> File: `shaders/reduce.comp` + `shaders/reduce.spv`
>
> Two-pass tree reduction. Workgroup-local reduce in shared
> memory (256→1), then a second dispatch reduces the
> intermediate buffer. See SHADERS_PLAN.md §"Iteration 1".
>
> Spec constant `OP`: 0=sum, 1=min, 2=max, 3=mean
> (mean = sum + final-divide on host side).
>
> Bindings:
>
>   binding 0: input  (storage buffer, float[])
>   binding 1: output (storage buffer, float[])
>
> Push constants:
>
>   uint n;   // input length
>
> Workgroup size: 256.
>
> Build:
>
> ```sh
> glslangValidator -V shaders/reduce.comp -o shaders/reduce.spv
> ```
>
> Commit: "shaders: reduce.comp — sum/min/max via spec constant".
> Push immediately.
>
> #### Iter 2: unary fused
>
> File: `shaders/unary.comp` + `shaders/unary.spv`
>
> Per SHADERS_PLAN.md §"Iteration 2". Eight unary ops via spec
> constant OP=0..7. Single binding 0 (in), binding 1 (out).
> Push constants: `uint n`.
>
> #### Iter 3: matmul (naive)
>
> File: `shaders/matmul_naive.comp` + `shaders/matmul_naive.spv`
>
> Per SHADERS_PLAN.md §"Iteration 3". 2D workgroup 16×16.
> Bindings 0/1/2 = A/B/C. Push constants: `uint M, N, K`.
>
> #### Iter 4: matmul (tiled)
>
> File: `shaders/matmul_tiled.comp` + `shaders/matmul_tiled.spv`
>
> Per SHADERS_PLAN.md §"Iteration 4". 16×16 shared-memory tiles.
> Same bindings + push constants as naive.
>
> ### What NOT to do
>
> - **Don't write tests on the Mac side.** Linux session writes
>   `test_reduce.cpp`, `test_unary.cpp`, etc. and benches.
> - **Don't modify** `Backend_par_vulkan.{hpp,cpp}` for these
>   shaders. The dispatch helper is already generic — it takes a
>   `VkPipe`, buffer array, group count, and push constants.
>   New shaders just need a new `VkPipe` (created with
>   `create_pipeline(shader, n_bindings, push_size, spec_const)`).
> - **Don't try to integrate matmul** into Spirit's CUDA-equivalent
>   API surface. That's a later iteration after Nx wrapping.
> - **Don't push to upstream `spirit-code/spirit`** — only push to
>   `192.168.0.33:/mnt/jeff/home/git/repos/spirit.git` (origin)
>   and the GitHub fork at `borodark/spirit`.
>
> ### Coordination
>
> Linux session is on `feature/vulkan-backend`, currently working
> on bench scaffolding for the new shaders. **Rebase before each
> commit:**
>
> ```sh
> git fetch origin && git rebase origin/feat/demo-cluster
> # ... actually: rebase origin/feature/vulkan-backend
> git fetch origin && git rebase origin/feature/vulkan-backend
> ```
>
> Files you own (don't touch in Linux side):
>   shaders/*.comp
>   shaders/*.spv
>
> Files Linux owns (don't touch on Mac):
>   bench_vulkan.cpp
>   test_*.cpp
>   RESULTS_*.md
>
> Confirm you've read SHADERS_PLAN.md before starting.

---

## Out-of-band notes (don't paste)

### Why two-pass reduction over atomic-float

GLSL's `GL_EXT_shader_atomic_float` works on most modern GPUs but
the order of atomic add operations is non-deterministic. Spirit
will eventually want determinism for reproducible energy
calculations. Two-pass tree reduction is deterministic by
construction — the same inputs always produce the same output
regardless of how warps are scheduled.

### Why ship the .spv binaries with source

Compiling on the consumer side adds a build dep (glslangValidator)
and a build step. Pre-compiling and shipping both source and
binary means the Linux box can run the test/bench immediately.
Same pattern Spirit follows for the existing
`elementwise_binary.spv`.

### What Linux is doing in parallel

Setting up `test_reduce.cpp`, `test_unary.cpp`, `test_matmul.cpp`
scaffolding so when each `.spv` lands, integration is one git
pull + one rebuild + one run. Plus extending `bench_vulkan.cpp`
with section headers per shader family.

### Sync point

After iter 4 lands (matmul tiled), the next iteration is
softmax (composes existing pieces — no new shader file needed).
At that point we converge: Linux writes the softmax helper that
chains reduce → unary → reduce → elementwise. No more Mac shader
work until iter 6 (conv 2D).
