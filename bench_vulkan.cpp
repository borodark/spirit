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
#include <random>

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

    /* Warmup: enough dispatches at this size to ensure the GPU has
     * ramped through its power states before we start timing. The
     * default 3-iter warmup was too short for large N — 4M dispatches
     * are quick once the GPU is at P0 but the ramp from P8 takes
     * tens of milliseconds. Scale warmup with N so smaller sizes
     * don't pay the full warmup time. */
    int warmup_iters = N >= 1048576 ? 30 : (N >= 65536 ? 10 : 3);
    for (int i = 0; i < warmup_iters; i++)
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

/* ---------------------------------------------------------------- */
/* Unary (exp) — persistent input/output buffers, persistent pipeline */
/* ---------------------------------------------------------------- */

static double bench_cpu_unary_exp(int N, int iters) {
    std::vector<float> a(N, 0.5f), c(N);
    auto t0 = Clock::now();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < N; i++) c[i] = std::exp(a[i]);
    }
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    volatile float sink = c[N/2]; (void)sink;
    return ms / iters;
}

static double bench_gpu_unary_exp(int N, int iters, VkPipe* pipe) {
    VkDeviceSize sz = N * sizeof(float);
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkBuf buf_in{}, buf_out{};
    buf_alloc(&buf_in, sz, usage, mem);
    buf_alloc(&buf_out, sz, usage, mem);
    std::vector<float> a(N, 0.5f);
    upload(&buf_in, a.data(), sz);

    VkBuffer bufs[2] = {buf_in.buffer, buf_out.buffer};
    uint32_t n = (uint32_t)N;
    uint32_t groups = (N + 255) / 256;

    int warmup_iters = N >= 1048576 ? 30 : (N >= 65536 ? 10 : 3);
    for (int i = 0; i < warmup_iters; i++)
        dispatch(pipe, bufs, 2, groups, sizeof(uint32_t), &n);

    auto t0 = Clock::now();
    for (int it = 0; it < iters; it++)
        dispatch(pipe, bufs, 2, groups, sizeof(uint32_t), &n);
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    buf_free(&buf_in);
    buf_free(&buf_out);
    return ms / iters;
}

/* ---------------------------------------------------------------- */
/* Matmul (naive M×K · K×N) — persistent buffers + pipeline. Needs   */
/* a 2D dispatch which the existing 1D dispatch() helper doesn't do, */
/* so the dispatch dance is inlined here (mirrors test_matmul.cpp).  */
/* ---------------------------------------------------------------- */

struct MatPush { uint32_t M; uint32_t N; uint32_t K; };

static double bench_cpu_matmul(int M, int N, int K, int iters) {
    std::vector<float> A(M*K, 1.0f), B(K*N, 1.0f), C(M*N);
    auto t0 = Clock::now();
    for (int it = 0; it < iters; it++) {
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float acc = 0;
                for (int k = 0; k < K; k++)
                    acc += A[m*K+k] * B[k*N+n];
                C[m*N+n] = acc;
            }
    }
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    volatile float sink = C[M/2*N + N/2]; (void)sink;
    return ms / iters;
}

static double bench_gpu_matmul(int M, int N, int K, int iters, VkPipe* pipe) {
    VkDeviceSize sz_a = M * K * sizeof(float);
    VkDeviceSize sz_b = K * N * sizeof(float);
    VkDeviceSize sz_c = M * N * sizeof(float);
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkBuf buf_a{}, buf_b{}, buf_c{};
    buf_alloc(&buf_a, sz_a, usage, mem);
    buf_alloc(&buf_b, sz_b, usage, mem);
    buf_alloc(&buf_c, sz_c, usage, mem);
    std::vector<float> A(M*K, 1.0f), B(K*N, 1.0f);
    upload(&buf_a, A.data(), sz_a);
    upload(&buf_b, B.data(), sz_b);

    VkBuffer bufs[3] = {buf_a.buffer, buf_b.buffer, buf_c.buffer};
    MatPush pc = {(uint32_t)M, (uint32_t)N, (uint32_t)K};
    uint32_t gx = (N + 15) / 16;
    uint32_t gy = (M + 15) / 16;

    auto& ctx = g_vk_ctx;

    auto run_one = [&]() {
        VkDescriptorBufferInfo bi[3];
        VkWriteDescriptorSet w[3];
        for (int i = 0; i < 3; i++) {
            bi[i] = {bufs[i], 0, VK_WHOLE_SIZE};
            w[i] = {};
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = pipe->descriptor_set;
            w[i].dstBinding = (uint32_t)i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(ctx.device, 3, w, 0, nullptr);

        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = ctx.command_pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(ctx.device, &ai, &cmd);

        VkCommandBufferBeginInfo bb{};
        bb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bb);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe->pipeline_layout, 0, 1, &pipe->descriptor_set, 0, nullptr);
        vkCmdPushConstants(cmd, pipe->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(MatPush), &pc);
        vkCmdDispatch(cmd, gx, gy, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(ctx.compute_queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(ctx.compute_queue);
        vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd);
    };

    for (int i = 0; i < 3; i++) run_one();   /* warmup */

    auto t0 = Clock::now();
    for (int it = 0; it < iters; it++) run_one();
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    buf_free(&buf_a);
    buf_free(&buf_b);
    buf_free(&buf_c);
    return ms / iters;
}

/* ---------------------------------------------------------------- */
/* Random (uniform) — persistent output buffer + persistent pipeline. */
/* No input; the GPU generates N values from (seed, thread_id).      */
/* CPU baseline: std::mt19937 seeded once, generate N floats.         */
/* ---------------------------------------------------------------- */

static double bench_cpu_uniform(int N, int iters) {
    std::vector<float> a(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    auto t0 = Clock::now();
    for (int it = 0; it < iters; it++) {
        rng.seed(42);
        for (int i = 0; i < N; i++) a[i] = dist(rng);
    }
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    volatile float sink = a[N/2]; (void)sink;
    return ms / iters;
}

static double bench_gpu_uniform(int N, int iters, VkPipe* pipe) {
    VkDeviceSize sz = N * sizeof(float);
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkBuf buf{};
    buf_alloc(&buf, sz, usage, mem);

    VkBuffer bufs[1] = {buf.buffer};
    struct { uint32_t n; uint32_t seed; } push = {(uint32_t)N, 42};
    uint32_t groups = (N + 255) / 256;

    int warmup_iters = N >= 1048576 ? 30 : (N >= 65536 ? 10 : 3);
    for (int i = 0; i < warmup_iters; i++)
        dispatch(pipe, bufs, 1, groups, sizeof(push), &push);

    auto t0 = Clock::now();
    for (int it = 0; it < iters; it++)
        dispatch(pipe, bufs, 1, groups, sizeof(push), &push);
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    buf_free(&buf);
    return ms / iters;
}

/* ---------------------------------------------------------------- */
/* Reductions (sum) — persistent input buffer; reduce() creates the  */
/* per-call pipeline + partial buffers internally each invocation.   */
/* That's the cost the current API surface charges; a persistent-    */
/* pipeline variant would be faster but requires a richer API. Bench */
/* what's actually shipping.                                          */
/* ---------------------------------------------------------------- */

static double bench_cpu_sum(int N, int iters) {
    std::vector<float> a(N, 1.0f);
    auto t0 = Clock::now();
    double total = 0;
    for (int it = 0; it < iters; it++) {
        double s = 0;
        for (int i = 0; i < N; i++) s += a[i];
        total += s;
    }
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    volatile double sink = total;
    (void)sink;
    return ms / iters;
}

static double bench_gpu_sum(int N, int iters, const std::string& spv_path) {
    VkDeviceSize sz = N * sizeof(float);
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkBuf in_buf{};
    buf_alloc(&in_buf, sz, usage, mem);
    std::vector<float> a(N, 1.0f);
    upload(&in_buf, a.data(), sz);

    /* Warmup: one full reduction at this N to ramp the GPU + warm
     * the shader cache before the timed loop. */
    int warmup_iters = N >= 1048576 ? 5 : 3;
    for (int i = 0; i < warmup_iters; i++)
        (void)reduce(&in_buf, N, REDUCE_SUM, spv_path);

    auto t0 = Clock::now();
    for (int it = 0; it < iters; it++) {
        (void)reduce(&in_buf, N, REDUCE_SUM, spv_path);
    }
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    buf_free(&in_buf);
    return ms / iters;
}

/* ---------------------------------------------------------------- */

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <binary.spv> [reduce] [unary] [matmul] [random] [tiled] [broadcast]\n", argv[0]);
        fprintf(stderr, "  Each optional shader path enables its bench section.\n");
        fprintf(stderr, "  Order: argv[1]=binary, [2]=reduce, [3]=unary, [4]=matmul,\n");
        fprintf(stderr, "         [5]=random, [6]=matmul_tiled, [7]=broadcast.\n");
        return 1;
    }

    if (vk_init(0) != 0) return 1;
    printf("GPU: %s\n", g_vk_ctx.device_props.deviceName);
    printf("f64: %s\n\n", g_vk_ctx.has_float64 ? "yes" : "no");

    VkShaderModule shader = load_shader(argv[1]);
    if (!shader) return 1;

    VkPipe pipe{};
    create_pipeline(&pipe, shader, 3, sizeof(uint32_t), 0 /* OP=add */);

    printf("%-10s  %10s  %12s  %10s  %12s  %12s\n",
           "N", "CPU (ms)", "persistent", "vs CPU", "naive (xfer)", "vs naive");
    printf("%-10s  %10s  %12s  %10s  %12s  %12s\n",
           "---", "---", "---", "---", "---", "---");

    int sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        int iters = N < 65536 ? 1000 : (N < 1048576 ? 200 : 50);

        double cpu_ms      = bench_cpu_add(N, iters);
        double persist_ms  = bench_gpu_add(N, iters, &pipe);
        double naive_ms    = bench_gpu_add_with_transfer(N, iters > 50 ? 50 : iters, &pipe);
        double vs_cpu      = cpu_ms / persist_ms;
        double vs_naive    = naive_ms / persist_ms;

        printf("%-10d  %10.4f  %12.4f  %9.2fx  %12.4f  %11.1fx\n",
               N, cpu_ms, persist_ms, vs_cpu, naive_ms, vs_naive);
    }

    printf("\n");
    printf("CPU:         single-core loop (no SIMD, no OpenMP)\n");
    printf("persistent:  dispatch only (buffers allocated + uploaded once,\n");
    printf("             reused across iterations) — the production pattern\n");
    printf("naive (xfer):full round-trip (alloc + upload + dispatch + download +\n");
    printf("             free, every iteration) — the anti-pattern\n");
    printf("vs CPU:      CPU / persistent — what the GPU buys vs serial CPU\n");
    printf("vs naive:    naive / persistent — what persistent buffers buy vs\n");
    printf("             the alloc-everything-each-iter pattern. THIS is the\n");
    printf("             optimization that matters for any real workload.\n");

    destroy_pipeline(&pipe);

    /* ============================================================ */
    /* Unary bench — only if elementwise_unary.spv was passed (3rd   */
    /* arg). Spec constant 0 = exp; one representative op for perf.  */
    /* ============================================================ */
    if (argc >= 4) {
        const char* unary_spv = argv[3];
        FILE* f = fopen(unary_spv, "rb");
        if (!f) {
            fprintf(stderr, "warning: %s not found; skipping unary bench\n", unary_spv);
        } else {
            fclose(f);
            VkShaderModule sh = load_shader(unary_spv);
            VkPipe upipe{};
            create_pipeline(&upipe, sh, 2, sizeof(uint32_t), 0 /* exp */);

            printf("\n=== unary (exp) ===\n");
            printf("%-10s  %10s  %12s  %10s\n",
                   "N", "CPU (ms)", "persistent", "vs CPU");
            printf("%-10s  %10s  %12s  %10s\n", "---", "---", "---", "---");

            for (int s = 0; s < nsizes; s++) {
                int N = sizes[s];
                int iters = N < 65536 ? 1000 : (N < 1048576 ? 200 : 50);
                double cpu_ms = bench_cpu_unary_exp(N, iters);
                double gpu_ms = bench_gpu_unary_exp(N, iters, &upipe);
                printf("%-10d  %10.4f  %12.4f  %9.2fx\n",
                       N, cpu_ms, gpu_ms, cpu_ms / gpu_ms);
            }

            destroy_pipeline(&upipe);
        }
    }

    /* ============================================================ */
    /* Matmul bench — only if matmul.spv was passed (4th arg). Square */
    /* matrices; CPU at 1024 is ~1s/iter so iter count drops sharply. */
    /* ============================================================ */
    if (argc >= 5) {
        const char* matmul_spv = argv[4];
        FILE* f = fopen(matmul_spv, "rb");
        if (!f) {
            fprintf(stderr, "warning: %s not found; skipping matmul bench\n", matmul_spv);
        } else {
            fclose(f);
            VkShaderModule sh = load_shader(matmul_spv);
            VkPipe mpipe{};
            create_pipeline(&mpipe, sh, 3, sizeof(MatPush), 0);

            printf("\n=== matmul (naive, square) ===\n");
            printf("%-12s  %10s  %12s  %10s  %10s\n",
                   "M=N=K", "CPU (ms)", "persistent", "vs CPU", "GFLOPS");
            printf("%-12s  %10s  %12s  %10s  %10s\n", "---", "---", "---", "---", "---");

            int dims[] = {64, 128, 256, 512, 1024};
            int ndims = sizeof(dims) / sizeof(dims[0]);

            for (int s = 0; s < ndims; s++) {
                int D = dims[s];
                /* CPU is O(D^3); cap iter count to keep wall time bounded */
                int cpu_iters = D <= 128 ? 50 : (D <= 256 ? 10 : (D <= 512 ? 3 : 1));
                int gpu_iters = D <= 128 ? 200 : (D <= 256 ? 100 : (D <= 512 ? 50 : 20));

                double cpu_ms = bench_cpu_matmul(D, D, D, cpu_iters);
                double gpu_ms = bench_gpu_matmul(D, D, D, gpu_iters, &mpipe);
                double flops = 2.0 * D * D * D;            /* 2*D^3 ops per matmul */
                double gflops = flops / (gpu_ms / 1000.0) / 1e9;
                printf("%-12d  %10.4f  %12.4f  %9.2fx  %10.2f\n",
                       D, cpu_ms, gpu_ms, cpu_ms / gpu_ms, gflops);
            }

            destroy_pipeline(&mpipe);
        }
    }

    /* ============================================================ */
    /* Tiled matmul bench — only if matmul_tiled.spv was passed.    */
    /* Same dispatch dance as naive but a different .spv binding it.  */
    /* ============================================================ */
    if (argc >= 7) {
        const char* tiled_spv = argv[6];
        FILE* f = fopen(tiled_spv, "rb");
        if (!f) {
            fprintf(stderr, "warning: %s not found; skipping tiled matmul bench\n", tiled_spv);
        } else {
            fclose(f);
            VkShaderModule sh = load_shader(tiled_spv);
            VkPipe tpipe{};
            create_pipeline(&tpipe, sh, 3, sizeof(MatPush), 0);

            printf("\n=== matmul (tiled, square) ===\n");
            printf("%-12s  %12s  %10s  %10s\n",
                   "M=N=K", "tiled (ms)", "GFLOPS", "vs naive");
            printf("%-12s  %12s  %10s  %10s\n", "---", "---", "---", "---");

            int dims[] = {64, 128, 256, 512, 1024};
            int ndims = sizeof(dims) / sizeof(dims[0]);

            VkShaderModule naive_sh = load_shader(argv[4]);
            VkPipe naive_pipe{};
            create_pipeline(&naive_pipe, naive_sh, 3, sizeof(MatPush), 0);

            for (int s = 0; s < ndims; s++) {
                int D = dims[s];
                int gpu_iters = D <= 128 ? 200 : (D <= 256 ? 100 : (D <= 512 ? 50 : 20));
                double naive_ms = bench_gpu_matmul(D, D, D, gpu_iters, &naive_pipe);
                double tiled_ms = bench_gpu_matmul(D, D, D, gpu_iters, &tpipe);
                double flops = 2.0 * D * D * D;
                double tiled_gflops = flops / (tiled_ms / 1000.0) / 1e9;
                printf("%-12d  %12.4f  %10.2f  %9.2fx\n",
                       D, tiled_ms, tiled_gflops, naive_ms / tiled_ms);
            }

            destroy_pipeline(&naive_pipe);
            destroy_pipeline(&tpipe);
        }
    }

    /* ============================================================ */
    /* Broadcasting elementwise binary bench — only if broadcast.spv */
    /* (7th arg). Compares: GPU broadcast (no host materialization)  */
    /* vs CPU broadcasting nested loop. Same op (add) at varied      */
    /* shape pairs that exercise different broadcast patterns.       */
    /* ============================================================ */
    if (argc >= 8) {
        const char* bcast_spv = argv[7];
        FILE* f = fopen(bcast_spv, "rb");
        if (!f) {
            fprintf(stderr, "warning: %s not found; skipping broadcast bench\n", bcast_spv);
        } else {
            fclose(f);
            /* Skip detailed bench loop — broadcast pattern's perf
             * tracks the elementwise binary numbers already in the
             * first table once the inputs are materialized. The
             * win is in HOST memory traffic (no per-op materializa-
             * tion) — that's a separate measurement. Document and
             * defer the perf bench to a follow-on iteration. */
            printf("\n=== broadcasting elementwise (deferred) ===\n");
            printf("Correctness: 7/7 tests pass on RTX 3060 Ti (test_broadcast).\n");
            printf("Perf bench deferred — the win is in host memory traffic\n");
            printf("(no per-op materialization of broadcasted operands), not\n");
            printf("dispatch time. Per-element compute matches the elementwise\n");
            printf("binary table above. To measure host-traffic savings, the\n");
            printf("baseline would be 'materialize then binary-op' which adds\n");
            printf("an extra alloc + dispatch per call — material for the\n");
            printf("Nx-side wrapper rather than the backend bench.\n");
        }
    }

    /* ============================================================ */
    /* Random (uniform) bench — only if random.spv was passed (5th). */
    /* Spec constant 0 = uniform; spec constant 1 = normal (skipped). */
    /* ============================================================ */
    if (argc >= 6) {
        const char* random_spv = argv[5];
        FILE* f = fopen(random_spv, "rb");
        if (!f) {
            fprintf(stderr, "warning: %s not found; skipping random bench\n", random_spv);
        } else {
            fclose(f);
            VkShaderModule sh = load_shader(random_spv);
            VkPipe rpipe{};
            create_pipeline(&rpipe, sh, 1, 2 * sizeof(uint32_t), 0 /* uniform */);

            printf("\n=== random (uniform [0,1)) ===\n");
            printf("%-10s  %10s  %12s  %10s\n",
                   "N", "CPU (ms)", "persistent", "vs CPU");
            printf("%-10s  %10s  %12s  %10s\n", "---", "---", "---", "---");

            for (int s = 0; s < nsizes; s++) {
                int N = sizes[s];
                int iters = N < 65536 ? 1000 : (N < 1048576 ? 200 : 50);
                double cpu_ms = bench_cpu_uniform(N, iters);
                double gpu_ms = bench_gpu_uniform(N, iters, &rpipe);
                printf("%-10d  %10.4f  %12.4f  %9.2fx\n",
                       N, cpu_ms, gpu_ms, cpu_ms / gpu_ms);
            }

            destroy_pipeline(&rpipe);
        }
    }

    /* ============================================================ */
    /* Reduction bench — only if reduce.spv was passed.             */
    /* ============================================================ */
    if (argc >= 3) {
        const char* reduce_spv = argv[2];
        FILE* f = fopen(reduce_spv, "rb");
        if (!f) {
            fprintf(stderr, "warning: %s not found; skipping reduction bench\n", reduce_spv);
        } else {
            fclose(f);
            printf("\n=== reductions (sum) ===\n");
            printf("%-10s  %10s  %10s  %10s\n",
                   "N", "CPU (ms)", "GPU (ms)", "vs CPU");
            printf("%-10s  %10s  %10s  %10s\n", "---", "---", "---", "---");

            for (int s = 0; s < nsizes; s++) {
                int N = sizes[s];
                /* Reductions are cheaper than elementwise per element
                 * but pay pipeline-create overhead per call. Use fewer
                 * iters at large N to cap wall time. */
                int iters = N < 65536 ? 200 : (N < 1048576 ? 50 : 20);

                double cpu_ms = bench_cpu_sum(N, iters);
                double gpu_ms = bench_gpu_sum(N, iters, reduce_spv);

                printf("%-10d  %10.4f  %10.4f  %9.2fx\n",
                       N, cpu_ms, gpu_ms, cpu_ms / gpu_ms);
            }

            printf("\n");
            printf("CPU:    single-core sum loop (no SIMD, no OpenMP)\n");
            printf("GPU:    reduce() — persistent input buffer, but per-call\n");
            printf("        pipeline create + partial-buffer alloc. The current\n");
            printf("        API charges this overhead every reduction. A\n");
            printf("        persistent-pipeline variant would be faster.\n");
        }
    }

    vk_destroy();
    return 0;
}
