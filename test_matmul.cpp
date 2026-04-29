/* test_matmul.cpp — Test GPU matrix multiply shader.
 *
 * Build:
 *   c++ -std=c++14 -O2 -I../core/include -I/usr/local/include \
 *       -DSPIRIT_USE_VULKAN ../test_matmul.cpp \
 *       ../core/src/engine/Backend_par_vulkan.cpp \
 *       -L/usr/local/lib -lvulkan -lm -o test_matmul
 *
 * Run:
 *   ./test_matmul ../shaders/matmul.spv
 */

#include <engine/Backend_par_vulkan.hpp>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace Engine::Backend::vulkan;

static int g_fail = 0;

struct MatDims { uint32_t M, N, K; };

static void run_matmul(const char* label, const char* spv_path,
                       const float* A, const float* B, const float* expected,
                       MatDims d, float eps = 1e-3f)
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

    VkShaderModule shader = load_shader(spv_path);
    VkPipe pipe{};
    /* matmul uses 3 push constant uint32s (M, N, K) = 12 bytes, no spec constant */
    create_pipeline(&pipe, shader, 3, sizeof(MatDims), 0);

    VkBuffer bufs[3] = {buf_a.buffer, buf_b.buffer, buf_c.buffer};
    uint32_t gx = (d.N + 15) / 16;
    uint32_t gy = (d.M + 15) / 16;

    /* dispatch uses 2D grid — need to patch dispatch call.
     * Our dispatch() only does 1D. Use raw Vulkan for 2D. */
    {
        auto& ctx = g_vk_ctx;

        /* Update descriptors */
        VkDescriptorBufferInfo buf_infos[3];
        VkWriteDescriptorSet writes[3];
        for (int i = 0; i < 3; i++) {
            buf_infos[i] = {bufs[i], 0, VK_WHOLE_SIZE};
            writes[i] = {};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = pipe.descriptor_set;
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
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe.pipeline_layout, 0, 1, &pipe.descriptor_set, 0, nullptr);
        vkCmdPushConstants(cmd, pipe.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(MatDims), &d);
        vkCmdDispatch(cmd, gx, gy, 1);  /* 2D dispatch */
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
        }
    }
    if (ok) printf("  PASS\n");
    else g_fail = 1;

    destroy_pipeline(&pipe);
    buf_free(&buf_a);
    buf_free(&buf_b);
    buf_free(&buf_c);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matmul.spv>\n", argv[0]);
        return 1;
    }
    const char* spv = argv[1];

    if (vk_init(0) != 0) return 1;
    printf("GPU: %s\n\n", g_vk_ctx.device_props.deviceName);

    /* 2x2 * 2x2 */
    {
        float A[] = {1, 2, 3, 4};
        float B[] = {5, 6, 7, 8};
        float C[] = {19, 22, 43, 50};
        run_matmul("2x2 * 2x2", spv, A, B, C, {2, 2, 2});
    }

    /* 2x3 * 3x2 */
    {
        float A[] = {1, 2, 3, 4, 5, 6};
        float B[] = {7, 8, 9, 10, 11, 12};
        float C[] = {58, 64, 139, 154};
        run_matmul("2x3 * 3x2", spv, A, B, C, {2, 2, 3});
    }

    /* Identity: 3x3 * I */
    {
        float A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        float I[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        run_matmul("3x3 * I", spv, A, I, A, {3, 3, 3});
    }

    /* 1x4 * 4x1 (dot product as matmul) */
    {
        float A[] = {1, 2, 3, 4};
        float B[] = {5, 6, 7, 8};
        float C[] = {70};  /* 1*5 + 2*6 + 3*7 + 4*8 = 70 */
        run_matmul("1x4 * 4x1 (dot)", spv, A, B, C, {1, 1, 4});
    }

    /* Non-square: 4x3 * 3x5 */
    {
        float A[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
        float B[] = {1,0,0,1,0, 0,1,0,0,1, 0,0,1,0,0};
        /* C = A*B where B selects/permutes columns:
         * row0: 1*1+2*0+3*0, 1*0+2*1+3*0, 1*0+2*0+3*1, 1*1+2*0+3*0, 1*0+2*1+3*0
         * = 1, 2, 3, 1, 2 */
        float C[] = {1,2,3,1,2, 4,5,6,4,5, 7,8,9,7,8, 10,11,12,10,11};
        run_matmul("4x3 * 3x5", spv, A, B, C, {4, 5, 3});
    }

    /* Larger: 32x32 * 32x32 (tests multi-workgroup) */
    {
        const int S = 32;
        std::vector<float> A(S*S), B(S*S), C(S*S);
        /* A = row index, B = identity → C = A */
        for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++) {
                A[i*S+j] = (float)(i+1);
                B[i*S+j] = (i == j) ? 1.0f : 0.0f;
            }
        run_matmul("32x32 * I", spv, A.data(), B.data(), A.data(), {S, S, S});
    }

    vk_destroy();
    printf("\n=== %s ===\n", g_fail ? "SOME TESTS FAILED" : "ALL 6 MATMUL TESTS PASSED");
    return g_fail;
}
