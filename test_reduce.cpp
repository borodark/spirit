/* test_reduce.cpp — correctness tests for shaders/reduce.spv
 *
 * Builds against Backend_par_vulkan.cpp; expects shaders/reduce.spv
 * to be present (Mac side compiles it). Pattern mirrors
 * test_vulkan_init.cpp.
 *
 * Build (from build-vulkan/):
 *   c++ -std=c++14 -O2 -I../core/include -I/usr/include \
 *       -DSPIRIT_USE_VULKAN ../test_reduce.cpp \
 *       ../core/src/engine/Backend_par_vulkan.cpp \
 *       -lvulkan -o test_reduce
 *
 * Run:
 *   ./test_reduce ../shaders/reduce.spv
 *
 * Skips with a diagnostic if the .spv isn't present yet — lets the
 * Linux session keep building the harness while the Mac session
 * writes the shader.
 */

#include <engine/Backend_par_vulkan.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace Engine::Backend::vulkan;

/* Op values must match the shader's spec constant OP layout:
 *   0 = sum
 *   1 = min
 *   2 = max
 *   3 = mean
 */
enum ReduceOp { OP_SUM = 0, OP_MIN = 1, OP_MAX = 2, OP_MEAN = 3 };

/* Pre-allocate buffers + dispatch a reduction. Two-pass: first
 * dispatch reduces N → ceil(N/256) partials; second reduces the
 * partials to 1. For N <= 256 we skip the second dispatch.
 *
 * Allocator hygiene: per the persistent-buffers usage pattern in
 * Backend_par_vulkan.hpp, we hold input + intermediate + output
 * buffers and reuse them across pipeline op variants.
 */
static float run_reduce(VkPipe* pipe, const std::vector<float>& input, ReduceOp op) {
    int N = (int)input.size();
    VkDeviceSize sz = N * sizeof(float);
    VkDeviceSize psz = ((N + 255) / 256) * sizeof(float);

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkBuf in_buf{}, partial_buf{}, out_buf{};
    buf_alloc(&in_buf, sz, usage, mem);
    buf_alloc(&partial_buf, psz, usage, mem);
    buf_alloc(&out_buf, sizeof(float), usage, mem);

    upload(&in_buf, input.data(), sz);

    /* Pass 1: in → partials */
    VkBuffer bufs1[2] = {in_buf.buffer, partial_buf.buffer};
    uint32_t n = (uint32_t)N;
    uint32_t groups = (N + 255) / 256;
    dispatch(pipe, bufs1, 2, groups, sizeof(uint32_t), &n);

    float result;
    if (groups <= 1) {
        /* Single workgroup output is the answer */
        download(&partial_buf, &result, sizeof(float));
    } else {
        /* Pass 2: partials → out */
        VkBuffer bufs2[2] = {partial_buf.buffer, out_buf.buffer};
        uint32_t np = groups;
        dispatch(pipe, bufs2, 2, 1, sizeof(uint32_t), &np);
        download(&out_buf, &result, sizeof(float));
    }

    if (op == OP_MEAN) result /= (float)N;

    buf_free(&in_buf);
    buf_free(&partial_buf);
    buf_free(&out_buf);
    return result;
}

static int near(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps * std::max(1.0f, std::fabs(b));
}

#define CHECK(cond, label) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", label); return 1; } \
    printf("  PASS: %s\n", label); \
} while (0)

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <reduce.spv>\n", argv[0]);
        return 1;
    }

    /* If the .spv isn't there yet, exit cleanly so CI doesn't fail
     * while the shader is being written on the Mac side. */
    FILE* f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "SKIP: %s not present yet (waiting for shader)\n", argv[1]);
        return 0;
    }
    fclose(f);

    if (vk_init(0) != 0) return 1;

    VkShaderModule shader = load_shader(argv[1]);
    if (!shader) return 1;

    /* One pipeline per op; spec constant 0 = sum, 1 = min, 2 = max,
     * 3 = mean (same dispatch as sum, host-side divide). */
    VkPipe pipe_sum{}, pipe_min{}, pipe_max{};
    create_pipeline(&pipe_sum, shader, 2, sizeof(uint32_t), OP_SUM);
    create_pipeline(&pipe_min, shader, 2, sizeof(uint32_t), OP_MIN);
    create_pipeline(&pipe_max, shader, 2, sizeof(uint32_t), OP_MAX);

    printf("=== TEST 1: sum ===\n");
    {
        std::vector<float> v(1000);
        std::iota(v.begin(), v.end(), 1.0f);   /* 1..1000 */
        float got = run_reduce(&pipe_sum, v, OP_SUM);
        CHECK(near(got, 500500.0f), "sum([1..1000]) == 500500");
    }
    {
        std::vector<float> v = {1.0f};
        float got = run_reduce(&pipe_sum, v, OP_SUM);
        CHECK(near(got, 1.0f), "sum([1]) == 1 (single-element edge)");
    }

    printf("\n=== TEST 2: min ===\n");
    {
        std::vector<float> v = {3, 1, 4, 1, 5, 9, 2, 6};
        float got = run_reduce(&pipe_min, v, OP_MIN);
        CHECK(near(got, 1.0f), "min([3,1,4,1,5,9,2,6]) == 1");
    }

    printf("\n=== TEST 3: max ===\n");
    {
        std::vector<float> v = {3, 1, 4, 1, 5, 9, 2, 6};
        float got = run_reduce(&pipe_max, v, OP_MAX);
        CHECK(near(got, 9.0f), "max([3,1,4,1,5,9,2,6]) == 9");
    }

    printf("\n=== TEST 4: mean (sum + host divide) ===\n");
    {
        std::vector<float> v = {2, 4, 6, 8};
        float got = run_reduce(&pipe_sum, v, OP_MEAN);
        CHECK(near(got, 5.0f), "mean([2,4,6,8]) == 5.0");
    }

    printf("\n=== TEST 5: two-pass kicks in for N > 256 ===\n");
    {
        std::vector<float> v(1024, 1.0f);
        float got = run_reduce(&pipe_sum, v, OP_SUM);
        CHECK(near(got, 1024.0f), "sum([1.0]*1024) == 1024 (two-pass)");
    }
    {
        std::vector<float> v(100000, 1.0f);
        float got = run_reduce(&pipe_sum, v, OP_SUM);
        CHECK(near(got, 100000.0f), "sum([1.0]*100000) == 100000 (two-pass)");
    }

    destroy_pipeline(&pipe_sum);
    destroy_pipeline(&pipe_min);
    destroy_pipeline(&pipe_max);
    vk_destroy();

    printf("\n=== ALL REDUCE TESTS PASSED ===\n");
    return 0;
}
