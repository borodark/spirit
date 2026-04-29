/* bench_vulkan.cpp — Performance benchmarks for Vulkan compute on FreeBSD.
 *
 * Measures: elementwise add, scale, throughput at various tensor sizes.
 * Compares GPU dispatch vs CPU baseline.
 *
 * Build (from build-vulkan/):
 *   c++ -std=c++14 -O2 -I../core/include -I/usr/local/include \
 *       -DSPIRIT_USE_VULKAN ../bench_vulkan.cpp \
 *       ../core/src/engine/Backend_par_vulkan.cpp \
 *       -L/usr/local/lib -lvulkan -o bench_vulkan
 */

#include <engine/Backend_par_vulkan.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

using namespace Engine::Backend::vulkan;
using Clock = std::chrono::high_resolution_clock;

/* ---------------------------------------------------------------- */

static double bench_cpu_add(int N, int iters) {
    std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N);
    auto t0 = Clock::now();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < N; i++)
            c[i] = a[i] + b[i];
    }
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    /* Prevent optimization */
    volatile float sink = c[N/2];
    (void)sink;
    return ms / iters;
}

static double bench_gpu_add(int N, int iters, VkPipe* pipe) {
    VkDeviceSize sz = N * sizeof(float);
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkBuf buf_a{}, buf_b{}, buf_c{};
    buf_alloc(&buf_a, sz, usage, mem);
    buf_alloc(&buf_b, sz, usage, mem);
    buf_alloc(&buf_c, sz, usage, mem);

    /* Fill with data */
    std::vector<float> a(N, 1.0f), b(N, 2.0f);
    upload(&buf_a, a.data(), sz);
    upload(&buf_b, b.data(), sz);

    VkBuffer bufs[3] = {buf_a.buffer, buf_b.buffer, buf_c.buffer};
    uint32_t n = (uint32_t)N;
    uint32_t groups = (N + 255) / 256;

    /* Warmup */
    for (int i = 0; i < 3; i++)
        dispatch(pipe, bufs, 3, groups, sizeof(uint32_t), &n);

    /* Benchmark */
    auto t0 = Clock::now();
    for (int it = 0; it < iters; it++) {
        dispatch(pipe, bufs, 3, groups, sizeof(uint32_t), &n);
    }
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    buf_free(&buf_a);
    buf_free(&buf_b);
    buf_free(&buf_c);

    return ms / iters;
}

static double bench_gpu_add_with_transfer(int N, int iters, VkPipe* pipe) {
    VkDeviceSize sz = N * sizeof(float);
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N);

    /* Warmup */
    VkBuf buf_a{}, buf_b{}, buf_c{};
    buf_alloc(&buf_a, sz, usage, mem);
    buf_alloc(&buf_b, sz, usage, mem);
    buf_alloc(&buf_c, sz, usage, mem);

    VkBuffer bufs[3] = {buf_a.buffer, buf_b.buffer, buf_c.buffer};
    uint32_t n = (uint32_t)N;
    uint32_t groups = (N + 255) / 256;

    upload(&buf_a, a.data(), sz);
    upload(&buf_b, b.data(), sz);
    dispatch(pipe, bufs, 3, groups, sizeof(uint32_t), &n);
    download(&buf_c, c.data(), sz);

    buf_free(&buf_a);
    buf_free(&buf_b);
    buf_free(&buf_c);

    /* Benchmark: full round-trip (upload + compute + download) */
    auto t0 = Clock::now();
    for (int it = 0; it < iters; it++) {
        VkBuf ba{}, bb{}, bc{};
        buf_alloc(&ba, sz, usage, mem);
        buf_alloc(&bb, sz, usage, mem);
        buf_alloc(&bc, sz, usage, mem);
        upload(&ba, a.data(), sz);
        upload(&bb, b.data(), sz);
        VkBuffer bs[3] = {ba.buffer, bb.buffer, bc.buffer};
        dispatch(pipe, bs, 3, groups, sizeof(uint32_t), &n);
        download(&bc, c.data(), sz);
        buf_free(&ba);
        buf_free(&bb);
        buf_free(&bc);
    }
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ms / iters;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <elementwise_binary.spv>\n", argv[0]);
        return 1;
    }

    if (vk_init(0) != 0) return 1;
    printf("GPU: %s\n", g_vk_ctx.device_props.deviceName);
    printf("f64: %s\n\n", g_vk_ctx.has_float64 ? "yes" : "no");

    VkShaderModule shader = load_shader(argv[1]);
    if (!shader) return 1;

    VkPipe pipe{};
    create_pipeline(&pipe, shader, 3, sizeof(uint32_t), 0 /* OP=add */);

    printf("%-12s  %10s  %10s  %10s  %8s\n",
           "N", "CPU (ms)", "GPU (ms)", "GPU+xfer", "speedup");
    printf("%-12s  %10s  %10s  %10s  %8s\n",
           "---", "---", "---", "---", "---");

    int sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        int iters = N < 65536 ? 1000 : (N < 1048576 ? 200 : 50);

        double cpu_ms = bench_cpu_add(N, iters);
        double gpu_ms = bench_gpu_add(N, iters, &pipe);
        double gpu_xfer_ms = bench_gpu_add_with_transfer(N, iters > 50 ? 50 : iters, &pipe);
        double speedup = cpu_ms / gpu_ms;

        printf("%-12d  %10.4f  %10.4f  %10.4f  %7.2fx\n",
               N, cpu_ms, gpu_ms, gpu_xfer_ms, speedup);
    }

    printf("\n");
    printf("CPU:       single-core loop (no SIMD, no OpenMP)\n");
    printf("GPU:       dispatch only (data already on device)\n");
    printf("GPU+xfer:  full round-trip (alloc + upload + dispatch + download + free)\n");
    printf("speedup:   CPU / GPU (dispatch-only)\n");

    destroy_pipeline(&pipe);
    vk_destroy();
    return 0;
}
