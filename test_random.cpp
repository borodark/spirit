/* test_random.cpp — Test GPU random number generation.
 *
 * Build:
 *   c++ -std=c++14 -O2 -I../core/include -I/usr/local/include \
 *       -DSPIRIT_USE_VULKAN ../test_random.cpp \
 *       ../core/src/engine/Backend_par_vulkan.cpp \
 *       -L/usr/local/lib -lvulkan -lm -o test_random
 *
 * Run:
 *   ./test_random ../shaders/random_philox.spv
 */

#include <engine/Backend_par_vulkan.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace Engine::Backend::vulkan;

static int g_fail = 0;

static std::vector<float> gpu_random(const char* spv, int N, uint32_t seed, int dist)
{
    VkDeviceSize sz = N * sizeof(float);
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VkBuf buf{};
    buf_alloc(&buf, sz, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkShaderModule shader = load_shader(spv);
    VkPipe pipe{};
    create_pipeline(&pipe, shader, 1, 2 * sizeof(uint32_t), dist);

    VkBuffer bufs[1] = {buf.buffer};
    uint32_t groups = (N + 255) / 256;

    struct { uint32_t n; uint32_t seed; } push = {(uint32_t)N, seed};
    dispatch(&pipe, bufs, 1, groups, sizeof(push), &push);

    std::vector<float> result(N);
    download(&buf, result.data(), sz);

    destroy_pipeline(&pipe);
    buf_free(&buf);
    return result;
}

static int test_uniform(const char* spv)
{
    printf("=== Uniform [0,1) — 10000 samples ===\n");
    auto r = gpu_random(spv, 10000, 42, 0);

    float mn = *std::min_element(r.begin(), r.end());
    float mx = *std::max_element(r.begin(), r.end());
    float sum = std::accumulate(r.begin(), r.end(), 0.0f);
    float mean = sum / r.size();

    printf("  min=%.6f max=%.6f mean=%.6f\n", mn, mx, mean);

    int ok = 1;
    if (mn < 0.0f) { printf("  FAIL: min < 0\n"); ok = 0; }
    if (mx >= 1.0f) { printf("  FAIL: max >= 1\n"); ok = 0; }
    if (fabsf(mean - 0.5f) > 0.02f) { printf("  FAIL: mean not ~0.5\n"); ok = 0; }

    if (ok) printf("  PASS\n");
    else g_fail = 1;
    return ok ? 0 : 1;
}

static int test_uniform_deterministic(const char* spv)
{
    printf("=== Uniform determinism — same seed → same output ===\n");
    auto r1 = gpu_random(spv, 1000, 123, 0);
    auto r2 = gpu_random(spv, 1000, 123, 0);

    int ok = (r1 == r2);
    if (ok) printf("  PASS (1000 values identical)\n");
    else { printf("  FAIL: different outputs for same seed\n"); g_fail = 1; }
    return ok ? 0 : 1;
}

static int test_uniform_different_seeds(const char* spv)
{
    printf("=== Uniform different seeds → different output ===\n");
    auto r1 = gpu_random(spv, 100, 1, 0);
    auto r2 = gpu_random(spv, 100, 2, 0);

    int same = 0;
    for (int i = 0; i < 100; i++)
        if (r1[i] == r2[i]) same++;

    int ok = (same < 5);  /* allowing tiny collision rate */
    if (ok) printf("  PASS (%d/100 collisions)\n", same);
    else { printf("  FAIL: %d/100 same values\n", same); g_fail = 1; }
    return ok ? 0 : 1;
}

static int test_normal(const char* spv)
{
    printf("=== Normal (Box-Muller) — 10000 samples ===\n");
    auto r = gpu_random(spv, 10000, 77, 1);

    float sum = std::accumulate(r.begin(), r.end(), 0.0f);
    float mean = sum / r.size();

    float var_sum = 0;
    for (auto x : r) var_sum += (x - mean) * (x - mean);
    float stddev = sqrtf(var_sum / r.size());

    float mn = *std::min_element(r.begin(), r.end());
    float mx = *std::max_element(r.begin(), r.end());

    printf("  mean=%.4f stddev=%.4f min=%.4f max=%.4f\n", mean, stddev, mn, mx);

    int ok = 1;
    if (fabsf(mean) > 0.05f) { printf("  FAIL: mean not ~0\n"); ok = 0; }
    if (fabsf(stddev - 1.0f) > 0.1f) { printf("  FAIL: stddev not ~1\n"); ok = 0; }
    if (mx < 2.0f) { printf("  FAIL: max too small for normal\n"); ok = 0; }
    if (mn > -2.0f) { printf("  FAIL: min too large for normal\n"); ok = 0; }

    if (ok) printf("  PASS\n");
    else g_fail = 1;
    return ok ? 0 : 1;
}

static int test_normal_large(const char* spv)
{
    printf("=== Normal — 100000 samples, tighter stats ===\n");
    auto r = gpu_random(spv, 100000, 999, 1);

    float sum = std::accumulate(r.begin(), r.end(), 0.0f);
    float mean = sum / r.size();

    float var_sum = 0;
    for (auto x : r) var_sum += (x - mean) * (x - mean);
    float stddev = sqrtf(var_sum / r.size());

    printf("  mean=%.5f stddev=%.5f\n", mean, stddev);

    int ok = 1;
    if (fabsf(mean) > 0.02f) { printf("  FAIL: mean not ~0\n"); ok = 0; }
    if (fabsf(stddev - 1.0f) > 0.05f) { printf("  FAIL: stddev not ~1\n"); ok = 0; }

    if (ok) printf("  PASS\n");
    else g_fail = 1;
    return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <random_philox.spv>\n", argv[0]);
        return 1;
    }

    if (vk_init(0) != 0) return 1;
    printf("GPU: %s\n\n", g_vk_ctx.device_props.deviceName);

    test_uniform(argv[1]);
    test_uniform_deterministic(argv[1]);
    test_uniform_different_seeds(argv[1]);
    test_normal(argv[1]);
    test_normal_large(argv[1]);

    vk_destroy();
    printf("\n=== %s ===\n", g_fail ? "SOME TESTS FAILED" : "ALL 5 RANDOM TESTS PASSED");
    return g_fail;
}
