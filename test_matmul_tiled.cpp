/* test_matmul_tiled.cpp — Correctness + perf comparison: naive vs tiled matmul.
 *
 * Build:
 *   c++ -std=c++14 -O2 -I../core/include -I/usr/local/include \
 *       -DSPIRIT_USE_VULKAN ../test_matmul_tiled.cpp \
 *       ../core/src/engine/Backend_par_vulkan.cpp \
 *       -L/usr/local/lib -lvulkan -lm -o test_matmul_tiled
 *
 * Run:
 *   ./test_matmul_tiled ../shaders/matmul.spv ../shaders/matmul_tiled.spv
 */

#include <engine/Backend_par_vulkan.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

using namespace Engine::Backend::vulkan;
using Clock = std::chrono::high_resolution_clock;

static int g_fail = 0;

struct MatDims { uint32_t M, N, K; };

static void dispatch_matmul(VkPipe* pipe, VkBuffer* bufs, MatDims d)
{
    auto& ctx = g_vk_ctx;
    uint32_t gx = (d.N + 15) / 16;
    uint32_t gy = (d.M + 15) / 16;

    VkDescriptorBufferInfo buf_infos[3];
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        buf_infos[i] = {bufs[i], 0, VK_WHOLE_SIZE};
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = pipe->descriptor_set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);

    VkCommandBufferAllocateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ci.commandPool = ctx.command_pool;
    ci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ci.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(ctx.device, &ci, &cmd);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe->pipeline_layout, 0, 1, &pipe->descriptor_set, 0, nullptr);
    vkCmdPushConstants(cmd, pipe->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(MatDims), &d);
    vkCmdDispatch(cmd, gx, gy, 1);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkResetFences(ctx.device, 1, &ctx.sync_fence);
    vkQueueSubmit(ctx.compute_queue, 1, &submit, ctx.sync_fence);
    vkWaitForFences(ctx.device, 1, &ctx.sync_fence, VK_TRUE, UINT64_MAX);
    vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd);
}

static int test_correctness(const char* spv, const char* label,
                            const float* A, const float* B, const float* expected,
                            MatDims d, float eps = 1e-2f)
{
    printf("=== %s [%u x %u] * [%u x %u] ===\n", label, d.M, d.K, d.K, d.N);

    VkDeviceSize sz_a = d.M * d.K * sizeof(float);
    VkDeviceSize sz_b = d.K * d.N * sizeof(float);
    VkDeviceSize sz_c = d.M * d.N * sizeof(float);

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkBuf buf_a{}, buf_b{}, buf_c{};
    buf_alloc(&buf_a, sz_a, usage, mem);
    buf_alloc(&buf_b, sz_b, usage, mem);
    buf_alloc(&buf_c, sz_c, usage, mem);
    upload(&buf_a, A, sz_a);
    upload(&buf_b, B, sz_b);

    VkShaderModule shader = load_shader(spv);
    VkPipe pipe{};
    create_pipeline(&pipe, shader, 3, sizeof(MatDims), 0);

    VkBuffer bufs[3] = {buf_a.buffer, buf_b.buffer, buf_c.buffer};
    dispatch_matmul(&pipe, bufs, d);

    std::vector<float> result(d.M * d.N);
    download(&buf_c, result.data(), sz_c);

    int ok = 1;
    for (uint32_t i = 0; i < d.M * d.N; i++) {
        float diff = fabsf(result[i] - expected[i]);
        float scale = fmaxf(1.0f, fabsf(expected[i]));
        if (diff / scale > eps) {
            uint32_t row = i / d.N, col = i % d.N;
            printf("  FAIL C[%u,%u]: got %f, expected %f\n", row, col, result[i], expected[i]);
            ok = 0;
            if (--ok < -5) break;  /* don't flood */
        }
    }
    if (ok == 1) printf("  PASS\n");
    else g_fail = 1;

    destroy_pipeline(&pipe);
    buf_free(&buf_a);
    buf_free(&buf_b);
    buf_free(&buf_c);
    return ok == 1 ? 0 : 1;
}

static void bench_matmul(const char* naive_spv, const char* tiled_spv, int S, int iters)
{
    MatDims d = {(uint32_t)S, (uint32_t)S, (uint32_t)S};
    VkDeviceSize sz = S * S * sizeof(float);

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkBuf buf_a{}, buf_b{}, buf_c{};
    buf_alloc(&buf_a, sz, usage, mem);
    buf_alloc(&buf_b, sz, usage, mem);
    buf_alloc(&buf_c, sz, usage, mem);

    /* Fill A with row index, B with identity */
    std::vector<float> data(S * S);
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++)
            data[i * S + j] = (i == j) ? 1.0f : 0.0f;
    upload(&buf_a, data.data(), sz);
    upload(&buf_b, data.data(), sz);

    VkBuffer bufs[3] = {buf_a.buffer, buf_b.buffer, buf_c.buffer};

    /* Benchmark naive */
    VkShaderModule sh_naive = load_shader(naive_spv);
    VkPipe pipe_naive{};
    create_pipeline(&pipe_naive, sh_naive, 3, sizeof(MatDims), 0);

    for (int i = 0; i < 3; i++) dispatch_matmul(&pipe_naive, bufs, d);
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) dispatch_matmul(&pipe_naive, bufs, d);
    auto t1 = Clock::now();
    double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    /* Benchmark tiled */
    VkShaderModule sh_tiled = load_shader(tiled_spv);
    VkPipe pipe_tiled{};
    create_pipeline(&pipe_tiled, sh_tiled, 3, sizeof(MatDims), 0);

    for (int i = 0; i < 3; i++) dispatch_matmul(&pipe_tiled, bufs, d);
    auto t2 = Clock::now();
    for (int i = 0; i < iters; i++) dispatch_matmul(&pipe_tiled, bufs, d);
    auto t3 = Clock::now();
    double tiled_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / iters;

    double flops = 2.0 * S * S * S;
    double naive_gflops = flops / (naive_ms * 1e6);
    double tiled_gflops = flops / (tiled_ms * 1e6);
    double speedup = naive_ms / tiled_ms;

    printf("  %4dx%d: naive %7.2f ms (%5.1f GFLOPS)  tiled %7.2f ms (%5.1f GFLOPS)  %.1fx\n",
           S, S, naive_ms, naive_gflops, tiled_ms, tiled_gflops, speedup);

    destroy_pipeline(&pipe_naive);
    destroy_pipeline(&pipe_tiled);
    buf_free(&buf_a);
    buf_free(&buf_b);
    buf_free(&buf_c);
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <matmul.spv> <matmul_tiled.spv>\n", argv[0]);
        return 1;
    }
    const char* naive_spv = argv[1];
    const char* tiled_spv = argv[2];

    if (vk_init(0) != 0) return 1;
    printf("GPU: %s\n\n", g_vk_ctx.device_props.deviceName);

    /* Correctness (tiled) */
    {
        float A[] = {1,2,3,4};
        float B[] = {5,6,7,8};
        float C[] = {19,22,43,50};
        test_correctness(tiled_spv, "tiled 2x2", A, B, C, {2,2,2});
    }
    {
        float A[] = {1,2,3, 4,5,6};
        float B[] = {7,8, 9,10, 11,12};
        float C[] = {58,64, 139,154};
        test_correctness(tiled_spv, "tiled 2x3*3x2", A, B, C, {2,2,3});
    }
    {
        float A[] = {1,2,3, 4,5,6, 7,8,9};
        float I[] = {1,0,0, 0,1,0, 0,0,1};
        test_correctness(tiled_spv, "tiled 3x3*I", A, I, A, {3,3,3});
    }
    /* Non-tile-aligned: 17x19 * 19x13 */
    {
        const int M=17, N=13, K=19;
        std::vector<float> A(M*K), B(K*N), C_expected(M*N, 0);
        for (int i = 0; i < M*K; i++) A[i] = (float)(i % 7) - 3.0f;
        for (int i = 0; i < K*N; i++) B[i] = (float)(i % 5) - 2.0f;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
                for (int k = 0; k < K; k++)
                    C_expected[m*N+n] += A[m*K+k] * B[k*N+n];
        test_correctness(tiled_spv, "tiled 17x19*19x13 (non-aligned)",
                         A.data(), B.data(), C_expected.data(), {M,N,K});
    }

    /* Performance comparison */
    printf("\n=== Naive vs Tiled matmul performance ===\n");
    bench_matmul(naive_spv, tiled_spv, 64, 100);
    bench_matmul(naive_spv, tiled_spv, 128, 50);
    bench_matmul(naive_spv, tiled_spv, 256, 20);
    bench_matmul(naive_spv, tiled_spv, 512, 10);
    bench_matmul(naive_spv, tiled_spv, 1024, 5);

    vk_destroy();
    printf("\n=== %s ===\n", g_fail ? "SOME TESTS FAILED" : "ALL TILED MATMUL TESTS PASSED");
    return g_fail;
}
