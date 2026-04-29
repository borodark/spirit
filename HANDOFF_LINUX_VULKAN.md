# Handoff: Vulkan Compute Backend — Linux Host (.249, RTX 3070 8GB)

Paste this as the first message in a Claude Code session on the Linux
box (192.168.0.249).

---

## Briefing (paste verbatim)

> You are continuing work on the **Spirit** project — a spin simulation
> framework. We're adding a **Vulkan compute backend** as a third
> backend alongside CUDA and OpenMP.
>
> The FreeBSD Mac (192.168.0.248) just proved the concept: Vulkan
> init + GPU memory round-trip + elementwise add shader dispatch —
> all passing on a GeForce GT 750M (Kepler, 384 cores). Now we need
> to build and benchmark the same code on a real GPU.
>
> **This machine**: Linux host with an **RTX 3070 8GB** (Ampere, 5888
> CUDA cores). Expected 30-50x faster than the GT 750M results.
>
> ### Step 1: Clone + checkout
>
> ```sh
> git clone git@192.168.0.33:/mnt/jeff/home/git/repos/spirit.git ~/spirit
> cd ~/spirit
> git checkout feature/vulkan-backend
> ```
>
> If the repo doesn't exist on 192.168.0.33 yet, clone from GitHub
> and push:
> ```sh
> git clone https://github.com/spirit-code/spirit.git ~/spirit
> cd ~/spirit
> git checkout -b feature/vulkan-backend
> # Apply the patches from 192.168.0.248 (4 commits):
> git remote add mac ssh://io@192.168.0.248/home/io/spirit
> git fetch mac feature/vulkan-backend
> git reset --hard mac/feature/vulkan-backend
> ```
>
> ### Step 2: Install deps
>
> ```sh
> # Vulkan SDK + tools
> sudo apt install vulkan-tools libvulkan-dev vulkan-validationlayers-dev
> sudo apt install glslang-tools spirv-tools
>
> # Verify
> vulkaninfo --summary   # should show RTX 3070
> glslangValidator --version
> ```
>
> ### Step 3: Build the test + benchmark
>
> ```sh
> mkdir -p build-vulkan && cd build-vulkan
>
> # Build the standalone test (no CMake needed — just the Vulkan backend):
> c++ -std=c++14 -O2 \
>   -I../core/include -I/usr/include \
>   -DSPIRIT_USE_VULKAN \
>   ../test_vulkan_init.cpp \
>   ../core/src/engine/Backend_par_vulkan.cpp \
>   -lvulkan -o test_vulkan_init
>
> # Build benchmark
> c++ -std=c++14 -O2 \
>   -I../core/include -I/usr/include \
>   -DSPIRIT_USE_VULKAN \
>   ../bench_vulkan.cpp \
>   ../core/src/engine/Backend_par_vulkan.cpp \
>   -lvulkan -o bench_vulkan
> ```
>
> ### Step 4: Compile shader (if .spv not in repo)
>
> ```sh
> cd ~/spirit
> glslangValidator -V shaders/elementwise_binary.comp -o shaders/elementwise_binary.spv
> ```
>
> ### Step 5: Run tests
>
> ```sh
> cd build-vulkan
> ./test_vulkan_init ../shaders/elementwise_binary.spv
> ```
>
> Expected output:
> ```
> === TEST 1: Vulkan init ===
> spirit-vulkan: NVIDIA GeForce RTX 3070 (f64=yes)
>   PASS
> === TEST 2: tensor round-trip ===
>   PASS
> === TEST 3: GPU compute — elementwise add ===
>   result: [11, 22, 33, 44, 55, 66, 77, 88]
>   PASS
> === ALL TESTS PASSED ===
> ```
>
> ### Step 6: Run benchmark
>
> ```sh
> ./bench_vulkan ../shaders/elementwise_binary.spv
> ```
>
> We expect the RTX 3070 to show:
> - **10-50x speedup** at 1M+ elements (dispatch-only)
> - Sub-millisecond dispatch at all sizes
> - Transfer overhead much lower (PCIe gen 4 vs the Mac's internal bus)
>
> ### Step 7: Report results
>
> Post the benchmark table. Compare with FreeBSD GT 750M results:
>
> ```
> FreeBSD GT 750M (384 cores):
>   1M elements: 0.30ms GPU, 1.66x vs CPU
>   4M elements: 1.41ms GPU, 1.76x vs CPU
> ```
>
> ### Step 8: Additional tests (if time)
>
> Test the other specialization constants (multiply, subtract, etc.):
>
> ```cpp
> // In test_vulkan_init.cpp, create_pipeline with spec_constant=1 for multiply
> // Verify: [1*10, 2*20, 3*30, ...] = [10, 40, 90, ...]
> ```
>
> Or try larger dispatches (16M, 64M elements) to see where the
> 3070 saturates.
>
> ### What NOT to do
>
> - Don't modify the backend code yet — just build and benchmark
> - Don't try to integrate with CMake full Spirit build (Eigen
>   dependency issues on some distros)
> - Don't push to the GitHub upstream (spirit-code/spirit)
>
> ### Context: what this is for
>
> This Vulkan backend will become **Nx.Vulkan** — a GPU tensor backend
> for Elixir's Nx library that works on FreeBSD (where CUDA doesn't).
> The benchmark numbers go in a blog post announcing GPU compute on
> FreeBSD via Vulkan. The RTX 3070 numbers are the "real hardware"
> benchmark; the GT 750M numbers prove the FreeBSD path works.
>
> Confirm the tests pass before doing anything else.

---

## Out-of-band notes (don't paste)

### What's on feature/vulkan-backend (4 commits)

```
f9dd0649 feature: Vulkan compute backend — context, buffers, pipeline, dispatch
d54ac765 fix: PUBLIC Vulkan link + C++14 compat in Backend_par_vulkan
df7974cc test: Vulkan init + round-trip + elementwise add shader — ALL PASS
95e7c65e bench: elementwise add — GPU vs CPU at varying tensor sizes
```

### Files that matter

```
core/include/engine/Backend_par_vulkan.hpp  — header (context, buffer, pipeline, dispatch API)
core/src/engine/Backend_par_vulkan.cpp      — implementation (~540 lines)
test_vulkan_init.cpp                        — 3 tests: init, round-trip, shader dispatch
bench_vulkan.cpp                            — perf: GPU vs CPU at various sizes
shaders/elementwise_binary.comp             — parametric GLSL compute shader
shaders/elementwise_binary.spv              — pre-compiled SPIR-V
```

### Getting the code to .249

Option A (if 192.168.0.33 bare repo exists):
```sh
# On .248 (this mac):
cd ~/spirit && git remote add shared git@192.168.0.33:/mnt/jeff/home/git/repos/spirit.git
git push shared feature/vulkan-backend
# On .249:
git clone git@192.168.0.33:... && git checkout feature/vulkan-backend
```

Option B (direct scp):
```sh
# On .249:
scp -r io@192.168.0.248:/home/io/spirit ~/spirit
cd ~/spirit && git checkout feature/vulkan-backend
```

Option C (patch files):
```sh
# On .248:
cd ~/spirit && git format-patch v2.2.0..feature/vulkan-backend -o /tmp/vk-patches
scp /tmp/vk-patches/* io@192.168.0.249:/tmp/
# On .249:
cd ~/spirit && git am /tmp/*.patch
```
