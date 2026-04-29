/* test_broadcast.cpp — Test broadcasting elementwise ops on GPU.
 *
 * Build:
 *   c++ -std=c++14 -O2 -I../core/include -I/usr/local/include \
 *       -DSPIRIT_USE_VULKAN ../test_broadcast.cpp \
 *       ../core/src/engine/Backend_par_vulkan.cpp \
 *       -L/usr/local/lib -lvulkan -lm -o test_broadcast
 *
 * Run:
 *   ./test_broadcast ../shaders/elementwise_binary_broadcast.spv
 */

#include <engine/Backend_par_vulkan.hpp>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace Engine::Backend::vulkan;

static int g_fail = 0;

struct BroadcastPush {
    uint32_t n;
    uint32_t ndim;
    uint32_t out_shape[4];
    uint32_t a_strides[4];
    uint32_t b_strides[4];
};

/* Compute strides for a shape, with 0 for broadcast dims.
 * If the tensor's dim size is 1 (broadcast) → stride=0.
 * Otherwise stride = product of subsequent dims. */
static void compute_strides(const uint32_t* tensor_shape, const uint32_t* out_shape,
                            uint32_t ndim, uint32_t* strides)
{
    uint32_t stride = 1;
    for (int d = (int)ndim - 1; d >= 0; d--) {
        if (tensor_shape[d] == 1)
            strides[d] = 0;  /* broadcast */
        else
            strides[d] = stride;
        stride *= tensor_shape[d];
    }
}

static uint32_t numel(const uint32_t* shape, uint32_t ndim) {
    uint32_t n = 1;
    for (uint32_t i = 0; i < ndim; i++) n *= shape[i];
    return n;
}

static int run_test(const char* label, const char* spv,
                    const float* a_data, const uint32_t* a_shape,
                    const float* b_data, const uint32_t* b_shape,
                    const float* expected, const uint32_t* out_shape,
                    uint32_t ndim, int op, float eps = 1e-4f)
{
    printf("=== %s ===\n", label);

    uint32_t a_n = numel(a_shape, ndim);
    uint32_t b_n = numel(b_shape, ndim);
    uint32_t c_n = numel(out_shape, ndim);

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkBuf buf_a{}, buf_b{}, buf_c{};
    buf_alloc(&buf_a, a_n * sizeof(float), usage, mem);
    buf_alloc(&buf_b, b_n * sizeof(float), usage, mem);
    buf_alloc(&buf_c, c_n * sizeof(float), usage, mem);
    upload(&buf_a, a_data, a_n * sizeof(float));
    upload(&buf_b, b_data, b_n * sizeof(float));

    BroadcastPush push = {};
    push.n = c_n;
    push.ndim = ndim;
    for (uint32_t i = 0; i < ndim; i++) push.out_shape[i] = out_shape[i];
    compute_strides(a_shape, out_shape, ndim, push.a_strides);
    compute_strides(b_shape, out_shape, ndim, push.b_strides);

    VkShaderModule shader = load_shader(spv);
    VkPipe pipe{};
    create_pipeline(&pipe, shader, 3, sizeof(BroadcastPush), op);

    VkBuffer bufs[3] = {buf_a.buffer, buf_b.buffer, buf_c.buffer};
    uint32_t groups = (c_n + 255) / 256;
    dispatch(&pipe, bufs, 3, groups, sizeof(BroadcastPush), &push);

    std::vector<float> result(c_n);
    download(&buf_c, result.data(), c_n * sizeof(float));

    int ok = 1;
    for (uint32_t i = 0; i < c_n; i++) {
        float diff = fabsf(result[i] - expected[i]);
        float scale = fmaxf(1.0f, fabsf(expected[i]));
        if (diff / scale > eps) {
            printf("  FAIL [%u]: got %f, expected %f\n", i, result[i], expected[i]);
            ok = 0;
        }
    }
    if (ok) printf("  PASS\n");
    else g_fail = 1;

    destroy_pipeline(&pipe);
    buf_free(&buf_a);
    buf_free(&buf_b);
    buf_free(&buf_c);
    return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <elementwise_binary_broadcast.spv>\n", argv[0]);
        return 1;
    }
    const char* spv = argv[1];

    if (vk_init(0) != 0) return 1;
    printf("GPU: %s\n\n", g_vk_ctx.device_props.deviceName);

    /* 1D: [3] + [3] — no broadcast, same as non-broadcast version */
    {
        float a[] = {1, 2, 3};
        float b[] = {10, 20, 30};
        float c[] = {11, 22, 33};
        uint32_t sa[] = {3}, sb[] = {3}, so[] = {3};
        run_test("[3] + [3] (no broadcast)", spv, a, sa, b, sb, c, so, 1, 0);
    }

    /* 1D: [3] + [1] — scalar broadcast */
    {
        float a[] = {1, 2, 3};
        float b[] = {10};
        float c[] = {11, 12, 13};
        uint32_t sa[] = {3}, sb[] = {1}, so[] = {3};
        run_test("[3] + [1] (scalar broadcast)", spv, a, sa, b, sb, c, so, 1, 0);
    }

    /* 2D: [2,3] + [1,3] — broadcast first dim */
    {
        float a[] = {1,2,3, 4,5,6};
        float b[] = {10, 20, 30};
        float c[] = {11,22,33, 14,25,36};
        uint32_t sa[] = {2,3}, sb[] = {1,3}, so[] = {2,3};
        run_test("[2,3] + [1,3] (row broadcast)", spv, a, sa, b, sb, c, so, 2, 0);
    }

    /* 2D: [2,3] + [2,1] — broadcast second dim */
    {
        float a[] = {1,2,3, 4,5,6};
        float b[] = {10, 20};
        float c[] = {11,12,13, 24,25,26};
        uint32_t sa[] = {2,3}, sb[] = {2,1}, so[] = {2,3};
        run_test("[2,3] + [2,1] (col broadcast)", spv, a, sa, b, sb, c, so, 2, 0);
    }

    /* 2D: [3,1] + [1,4] — outer product shape broadcast */
    {
        float a[] = {1, 2, 3};
        float b[] = {10, 20, 30, 40};
        /* result is [3,4]:
         * 1+10 1+20 1+30 1+40
         * 2+10 2+20 2+30 2+40
         * 3+10 3+20 3+30 3+40 */
        float c[] = {11,21,31,41, 12,22,32,42, 13,23,33,43};
        uint32_t sa[] = {3,1}, sb[] = {1,4}, so[] = {3,4};
        run_test("[3,1] + [1,4] (outer broadcast)", spv, a, sa, b, sb, c, so, 2, 0);
    }

    /* 2D: [3,1] * [1,4] — outer product via multiply */
    {
        float a[] = {1, 2, 3};
        float b[] = {10, 20, 30, 40};
        float c[] = {10,20,30,40, 20,40,60,80, 30,60,90,120};
        uint32_t sa[] = {3,1}, sb[] = {1,4}, so[] = {3,4};
        run_test("[3,1] * [1,4] (outer product)", spv, a, sa, b, sb, c, so, 2, 1);
    }

    /* 3D: [2,1,3] + [1,4,1] */
    {
        float a[] = {1,2,3, 4,5,6};  /* shape [2,1,3] */
        float b[] = {10, 20, 30, 40}; /* shape [1,4,1] */
        /* output shape [2,4,3]:
         * batch 0: a[0,:] + b[j] for j in 0..3
         * [1+10,2+10,3+10, 1+20,2+20,3+20, 1+30,2+30,3+30, 1+40,2+40,3+40]
         * batch 1: same with a[1,:] = [4,5,6] */
        float c[] = {
            11,12,13, 21,22,23, 31,32,33, 41,42,43,
            14,15,16, 24,25,26, 34,35,36, 44,45,46
        };
        uint32_t sa[] = {2,1,3}, sb[] = {1,4,1}, so[] = {2,4,3};
        run_test("[2,1,3] + [1,4,1] (3D broadcast)", spv, a, sa, b, sb, c, so, 3, 0);
    }

    vk_destroy();
    printf("\n=== %s ===\n", g_fail ? "SOME TESTS FAILED" : "ALL 7 BROADCAST TESTS PASSED");
    return g_fail;
}
