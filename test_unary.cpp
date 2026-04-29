/* test_unary.cpp — Test elementwise unary ops on GPU.
 *
 * Build:
 *   c++ -std=c++14 -O2 -I../core/include -I/usr/local/include \
 *       -DSPIRIT_USE_VULKAN ../test_unary.cpp \
 *       ../core/src/engine/Backend_par_vulkan.cpp \
 *       -L/usr/local/lib -lvulkan -lm -o test_unary
 *
 * Run:
 *   ./test_unary ../shaders/elementwise_unary.spv
 */

#include <engine/Backend_par_vulkan.hpp>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace Engine::Backend::vulkan;

enum UnaryOp {
    OP_EXP=0, OP_LOG=1, OP_SQRT=2, OP_ABS=3, OP_NEG=4,
    OP_SIGMOID=5, OP_TANH=6, OP_RELU=7, OP_CEIL=8, OP_FLOOR=9,
    OP_SIGN=10, OP_RECIP=11, OP_SQUARE=12
};

static int g_fail = 0;

static void run_test(const char* label, const char* spv_path,
                     UnaryOp op, const float* input, const float* expected, int N,
                     float eps = 1e-4f)
{
    printf("=== %s ===\n", label);

    VkDeviceSize sz = N * sizeof(float);
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VkBuf buf_in{}, buf_out{};
    buf_alloc(&buf_in, sz, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    buf_alloc(&buf_out, sz, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    upload(&buf_in, input, sz);

    VkShaderModule shader = load_shader(spv_path);
    VkPipe pipe{};
    create_pipeline(&pipe, shader, 2, sizeof(uint32_t), (int32_t)op);

    VkBuffer bufs[2] = {buf_in.buffer, buf_out.buffer};
    uint32_t n = (uint32_t)N;
    uint32_t groups = (N + 255) / 256;
    dispatch(&pipe, bufs, 2, groups, sizeof(uint32_t), &n);

    std::vector<float> result(N);
    download(&buf_out, result.data(), sz);

    int ok = 1;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(result[i] - expected[i]);
        float scale = fmaxf(1.0f, fabsf(expected[i]));
        if (diff / scale > eps) {
            printf("  FAIL [%d]: got %f, expected %f\n", i, result[i], expected[i]);
            ok = 0;
        }
    }
    if (ok) printf("  PASS\n");
    else g_fail = 1;

    destroy_pipeline(&pipe);
    buf_free(&buf_in);
    buf_free(&buf_out);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <elementwise_unary.spv>\n", argv[0]);
        return 1;
    }
    const char* spv = argv[1];

    if (vk_init(0) != 0) return 1;
    printf("GPU: %s\n\n", g_vk_ctx.device_props.deviceName);

    {
        float in[] = {0, 1, 2, -1};
        float ex[] = {1.0f, 2.71828f, 7.38906f, 0.36788f};
        run_test("exp", spv, OP_EXP, in, ex, 4);
    }
    {
        float in[] = {1, 2.71828f, 10, 0.5f};
        float ex[] = {0, 1.0f, 2.30259f, -0.69315f};
        run_test("log", spv, OP_LOG, in, ex, 4);
    }
    {
        float in[] = {0, 1, 4, 9, 16, 25};
        float ex[] = {0, 1, 2, 3, 4, 5};
        run_test("sqrt", spv, OP_SQRT, in, ex, 6);
    }
    {
        float in[] = {-3, -1, 0, 1, 3};
        float ex[] = {3, 1, 0, 1, 3};
        run_test("abs", spv, OP_ABS, in, ex, 5);
    }
    {
        float in[] = {-3, -1, 0, 1, 3};
        float ex[] = {3, 1, 0, -1, -3};
        run_test("neg", spv, OP_NEG, in, ex, 5);
    }
    {
        float in[] = {0, 1, -1, 5, -5};
        float ex[] = {0.5f, 0.73106f, 0.26894f, 0.99331f, 0.00669f};
        run_test("sigmoid", spv, OP_SIGMOID, in, ex, 5, 1e-3f);
    }
    {
        float in[] = {0, 1, -1, 2};
        float ex[] = {0, 0.76159f, -0.76159f, 0.96403f};
        run_test("tanh", spv, OP_TANH, in, ex, 4, 1e-3f);
    }
    {
        float in[] = {-2, -1, 0, 1, 2};
        float ex[] = {0, 0, 0, 1, 2};
        run_test("relu", spv, OP_RELU, in, ex, 5);
    }
    {
        float in[] = {1.2f, 1.8f, -1.2f, -1.8f, 2.0f};
        float ex[] = {2, 2, -1, -1, 2};
        run_test("ceil", spv, OP_CEIL, in, ex, 5);
    }
    {
        float in[] = {1.2f, 1.8f, -1.2f, -1.8f, 2.0f};
        float ex[] = {1, 1, -2, -2, 2};
        run_test("floor", spv, OP_FLOOR, in, ex, 5);
    }
    {
        float in[] = {-3, 0, 5};
        float ex[] = {-1, 0, 1};
        run_test("sign", spv, OP_SIGN, in, ex, 3);
    }
    {
        float in[] = {2, 4, 0.5f, 10};
        float ex[] = {0.5f, 0.25f, 2.0f, 0.1f};
        run_test("reciprocal", spv, OP_RECIP, in, ex, 4);
    }
    {
        float in[] = {2, 3, -4, 0.5f};
        float ex[] = {4, 9, 16, 0.25f};
        run_test("square", spv, OP_SQUARE, in, ex, 4);
    }

    vk_destroy();
    printf("\n=== %s ===\n", g_fail ? "SOME TESTS FAILED" : "ALL 13 UNARY TESTS PASSED");
    return g_fail;
}
