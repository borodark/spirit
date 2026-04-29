/* test_vulkan_init.cpp — Smoke test for the Vulkan compute backend.
 *
 * 1. Init Vulkan context, print GPU info
 * 2. Allocate tensor, round-trip data through GPU
 * 3. Load a compute shader, dispatch elementwise add, verify result
 *
 * Build (from build-vulkan/):
 *   c++ -std=c++14 -I../core/include -I/usr/local/include \
 *       ../test_vulkan_init.cpp \
 *       -Lcore -lSpirit -L/usr/local/lib -lvulkan \
 *       -o test_vulkan_init
 *
 * Run:
 *   ./test_vulkan_init [path/to/elementwise_binary.spv]
 */

#include <engine/Backend_par_vulkan.hpp>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace Engine::Backend::vulkan;

static int test_init()
{
    printf("=== TEST 1: Vulkan init ===\n");
    if (vk_init(0) != 0) {
        printf("FAIL: vk_init\n");
        return 1;
    }
    printf("  device: %s\n", g_vk_ctx.device_props.deviceName);
    printf("  f64:    %s\n", g_vk_ctx.has_float64 ? "yes" : "no");
    printf("  PASS\n\n");
    return 0;
}

static int test_roundtrip()
{
    printf("=== TEST 2: tensor round-trip (host → GPU → host) ===\n");

    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {};

    VkBuf buf{};
    if (buf_alloc(&buf, sizeof(input),
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
        printf("FAIL: buf_alloc\n");
        return 1;
    }

    if (upload(&buf, input, sizeof(input)) != 0) {
        printf("FAIL: upload\n");
        return 1;
    }

    if (download(&buf, output, sizeof(output)) != 0) {
        printf("FAIL: download\n");
        return 1;
    }

    int ok = 1;
    for (int i = 0; i < 4; i++) {
        if (fabsf(input[i] - output[i]) > 1e-6f) {
            printf("  MISMATCH [%d]: sent %f got %f\n", i, input[i], output[i]);
            ok = 0;
        }
    }

    buf_free(&buf);

    if (ok) printf("  [%.0f, %.0f, %.0f, %.0f] round-tripped\n  PASS\n\n",
                   output[0], output[1], output[2], output[3]);
    return ok ? 0 : 1;
}

static int test_shader_add(const char* spv_path)
{
    printf("=== TEST 3: GPU compute — elementwise add ===\n");
    printf("  shader: %s\n", spv_path);

    /* Load shader */
    VkShaderModule shader = load_shader(spv_path);
    if (shader == VK_NULL_HANDLE) {
        printf("FAIL: load_shader\n");
        return 1;
    }

    /* Create pipeline: 3 buffers (A, B, C), push constant uint n,
     * specialization constant 0 = add */
    VkPipe pipe{};
    if (create_pipeline(&pipe, shader, 3, sizeof(uint32_t), 0) != 0) {
        printf("FAIL: create_pipeline\n");
        return 1;
    }

    /* Allocate GPU buffers */
    const int N = 8;
    VkDeviceSize sz = N * sizeof(float);
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkBuf buf_a{}, buf_b{}, buf_c{};
    buf_alloc(&buf_a, sz, usage, mem);
    buf_alloc(&buf_b, sz, usage, mem);
    buf_alloc(&buf_c, sz, usage, mem);

    /* Upload input data */
    float a[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[N] = {10, 20, 30, 40, 50, 60, 70, 80};
    upload(&buf_a, a, sz);
    upload(&buf_b, b, sz);

    /* Dispatch: C = A + B */
    VkBuffer bufs[3] = {buf_a.buffer, buf_b.buffer, buf_c.buffer};
    uint32_t n = N;
    uint32_t groups = (N + 255) / 256;

    if (dispatch(&pipe, bufs, 3, groups, sizeof(uint32_t), &n) != 0) {
        printf("FAIL: dispatch\n");
        return 1;
    }

    /* Read back result */
    float c[N] = {};
    download(&buf_c, c, sz);

    int ok = 1;
    printf("  result: [");
    for (int i = 0; i < N; i++) {
        printf("%.0f", c[i]);
        if (i < N - 1) printf(", ");
        float expected = a[i] + b[i];
        if (fabsf(c[i] - expected) > 1e-6f) {
            ok = 0;
        }
    }
    printf("]\n");

    if (ok)
        printf("  expected: [11, 22, 33, 44, 55, 66, 77, 88]\n  PASS\n\n");
    else
        printf("  FAIL: result mismatch\n\n");

    /* Cleanup */
    buf_free(&buf_a);
    buf_free(&buf_b);
    buf_free(&buf_c);
    destroy_pipeline(&pipe);

    return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    int rc = 0;

    rc |= test_init();
    if (rc) return rc;

    rc |= test_roundtrip();

    if (argc > 1) {
        rc |= test_shader_add(argv[1]);
    } else {
        printf("=== TEST 3: skipped (pass .spv path as arg) ===\n\n");
    }

    vk_destroy();
    printf("=== %s ===\n", rc ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return rc;
}
