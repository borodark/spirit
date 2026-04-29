/* test_reduce.cpp — Test the GPU reduction shader (sum, min, max).
 *
 * Build:
 *   c++ -std=c++14 -O2 -I../core/include -I/usr/local/include \
 *       -DSPIRIT_USE_VULKAN ../test_reduce.cpp \
 *       ../core/src/engine/Backend_par_vulkan.cpp \
 *       -L/usr/local/lib -lvulkan -o test_reduce
 *
 * Run:
 *   ./test_reduce ../shaders/reduce.spv
 */

#include <engine/Backend_par_vulkan.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace Engine::Backend::vulkan;

static const char* g_spv_path;

static VkBuf make_buf(const float* data, int N)
{
    VkDeviceSize sz = N * sizeof(float);
    VkBuf buf{};
    buf_alloc(&buf, sz,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    upload(&buf, data, sz);
    return buf;
}

static int check(const char* label, float got, float expected, float eps = 1e-3f)
{
    float diff = fabsf(got - expected);
    if (diff > eps) {
        printf("  FAIL %s: got %f, expected %f (diff %f)\n", label, got, expected, diff);
        return 1;
    }
    printf("  PASS %s: %f (expected %f)\n", label, got, expected);
    return 0;
}

static int test_sum_small()
{
    printf("=== TEST: sum [1, 2, 3, 4, 5] ===\n");
    float data[] = {1, 2, 3, 4, 5};
    VkBuf buf = make_buf(data, 5);
    float result = reduce(&buf, 5, REDUCE_SUM, g_spv_path);
    buf_free(&buf);
    return check("sum(1..5)", result, 15.0f);
}

static int test_sum_1000()
{
    printf("=== TEST: sum [1..1000] ===\n");
    std::vector<float> data(1000);
    for (int i = 0; i < 1000; i++) data[i] = (float)(i + 1);
    VkBuf buf = make_buf(data.data(), 1000);
    float result = reduce(&buf, 1000, REDUCE_SUM, g_spv_path);
    buf_free(&buf);
    return check("sum(1..1000)", result, 500500.0f, 1.0f);
}

static int test_sum_large()
{
    printf("=== TEST: sum 100000 ones ===\n");
    std::vector<float> data(100000, 1.0f);
    VkBuf buf = make_buf(data.data(), 100000);
    float result = reduce(&buf, 100000, REDUCE_SUM, g_spv_path);
    buf_free(&buf);
    /* f32 sum of 100k ones should be exact */
    return check("sum(100k ones)", result, 100000.0f, 1.0f);
}

static int test_sum_single()
{
    printf("=== TEST: sum [42] ===\n");
    float data[] = {42.0f};
    VkBuf buf = make_buf(data, 1);
    float result = reduce(&buf, 1, REDUCE_SUM, g_spv_path);
    buf_free(&buf);
    return check("sum([42])", result, 42.0f);
}

static int test_min()
{
    printf("=== TEST: min [3, 1, 4, 1, 5, 9, 2, 6] ===\n");
    float data[] = {3, 1, 4, 1, 5, 9, 2, 6};
    VkBuf buf = make_buf(data, 8);
    float result = reduce(&buf, 8, REDUCE_MIN, g_spv_path);
    buf_free(&buf);
    return check("min", result, 1.0f);
}

static int test_max()
{
    printf("=== TEST: max [3, 1, 4, 1, 5, 9, 2, 6] ===\n");
    float data[] = {3, 1, 4, 1, 5, 9, 2, 6};
    VkBuf buf = make_buf(data, 8);
    float result = reduce(&buf, 8, REDUCE_MAX, g_spv_path);
    buf_free(&buf);
    return check("max", result, 9.0f);
}

static int test_min_large()
{
    printf("=== TEST: min 10000 elements (min at index 7777) ===\n");
    std::vector<float> data(10000);
    for (int i = 0; i < 10000; i++) data[i] = (float)(i + 100);
    data[7777] = -42.0f;
    VkBuf buf = make_buf(data.data(), 10000);
    float result = reduce(&buf, 10000, REDUCE_MIN, g_spv_path);
    buf_free(&buf);
    return check("min(10k, needle at 7777)", result, -42.0f);
}

static int test_max_large()
{
    printf("=== TEST: max 10000 elements (max at index 3333) ===\n");
    std::vector<float> data(10000, 1.0f);
    data[3333] = 99999.0f;
    VkBuf buf = make_buf(data.data(), 10000);
    float result = reduce(&buf, 10000, REDUCE_MAX, g_spv_path);
    buf_free(&buf);
    return check("max(10k, needle at 3333)", result, 99999.0f);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <reduce.spv>\n", argv[0]);
        return 1;
    }
    g_spv_path = argv[1];

    if (vk_init(0) != 0) return 1;
    printf("GPU: %s\n\n", g_vk_ctx.device_props.deviceName);

    int rc = 0;
    rc |= test_sum_small();
    rc |= test_sum_single();
    rc |= test_sum_1000();
    rc |= test_sum_large();
    rc |= test_min();
    rc |= test_max();
    rc |= test_min_large();
    rc |= test_max_large();

    vk_destroy();
    printf("\n=== %s ===\n", rc ? "SOME TESTS FAILED" : "ALL 8 TESTS PASSED");
    return rc;
}
