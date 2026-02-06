#include <random>

#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

namespace {

template <typename T>
class MusaRandomOp : public MusaOpKernel {
 public:
  explicit MusaRandomOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // 跟踪日志，确认到底执行了哪个算子
    fprintf(stderr, ">>> [MUSA_TRACE_AUTO] %s - Start\n", name().c_str());

    const Tensor& shape_tensor = ctx->input(0);
    TensorShape out_shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_tensor, &out_shape));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

    if (output->NumElements() == 0) return;

    // 1. 分配 Host 临时内存
    Tensor tmp_host_out;
    AllocatorAttributes host_attr;
    host_attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(output->dtype(), out_shape,
                                           &tmp_host_out, host_attr));

    // 2. 生成随机数 (CPU)
    T* cpu_ptr = tmp_host_out.flat<T>().data();
    std::random_device rd;
    std::mt19937 gen(rd());

    // 根据算子名称决定分布逻辑
    if (name().find("Normal") != std::string::npos) {
      // 正态分布 (Standard Normal)
      std::normal_distribution<float> dist(0.0f, 1.0f);
      for (int i = 0; i < output->NumElements(); ++i) {
        cpu_ptr[i] = static_cast<T>(dist(gen));
      }
    } else {
      // 均匀分布 (Uniform)
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      for (int i = 0; i < output->NumElements(); ++i) {
        cpu_ptr[i] = static_cast<T>(dist(gen));
      }
    }

    // 3. 搬运到 MUSA
    size_t total_bytes = output->NumElements() * sizeof(T);
    mStatus status =
        MusaMemcpyH2D(output->data(), tmp_host_out.data(), total_bytes);

    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("MUSA Random H2D copy failed for ", name()));

    musaDeviceSynchronize();
    fprintf(stderr, ">>> [MUSA_TRACE_AUTO] %s - Success (%s)\n", name().c_str(),
            DataTypeString(output->dtype()).c_str());
  }
};

}  // namespace

// ============================================================
// 完整注册宏：涵盖有状态、无状态、正态及均匀分布
// ============================================================
#define REGISTER_MUSA_RANDOM_KERNELS(TYPE)                          \
  /* 1. 有状态均匀分布 (tf.random.uniform) */                \
  REGISTER_KERNEL_BUILDER(Name("RandomUniform")                     \
                              .Device("MUSA")                       \
                              .HostMemory("shape")                  \
                              .TypeConstraint<TYPE>("dtype"),       \
                          MusaRandomOp<TYPE>);                      \
                                                                    \
  /* 2. 有状态标准正态分布 (tf.random.normal) */           \
  REGISTER_KERNEL_BUILDER(Name("RandomStandardNormal")              \
                              .Device("MUSA")                       \
                              .HostMemory("shape")                  \
                              .TypeConstraint<TYPE>("dtype"),       \
                          MusaRandomOp<TYPE>);                      \
                                                                    \
  /* 3. 有状态截断正态分布 (tf.random.truncated_normal) */ \
  REGISTER_KERNEL_BUILDER(Name("TruncatedNormal")                   \
                              .Device("MUSA")                       \
                              .HostMemory("shape")                  \
                              .TypeConstraint<TYPE>("dtype"),       \
                          MusaRandomOp<TYPE>);                      \
                                                                    \
  /* 4. 无状态均匀分布 V1 (tf.get_static_value 常用) */    \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomUniform")            \
                              .Device("MUSA")                       \
                              .HostMemory("shape")                  \
                              .HostMemory("seed")                   \
                              .TypeConstraint<TYPE>("dtype"),       \
                          MusaRandomOp<TYPE>);                      \
                                                                    \
  /* 5. 无状态均匀分布 V2 (Keras 默认常用) */            \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomUniformV2")          \
                              .Device("MUSA")                       \
                              .HostMemory("shape")                  \
                              .HostMemory("key")                    \
                              .HostMemory("counter")                \
                              .HostMemory("alg")                    \
                              .TypeConstraint<TYPE>("dtype"),       \
                          MusaRandomOp<TYPE>);                      \
                                                                    \
  /* 6. 无状态标准正态分布 V2 */                           \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomNormalV2")           \
                              .Device("MUSA")                       \
                              .HostMemory("shape")                  \
                              .HostMemory("key")                    \
                              .HostMemory("counter")                \
                              .HostMemory("alg")                    \
                              .TypeConstraint<TYPE>("dtype"),       \
                          MusaRandomOp<TYPE>);                      \
                                                                    \
  /* 7. 无状态截断正态分布 V2 */                           \
  REGISTER_KERNEL_BUILDER(Name("StatelessTruncatedNormalV2")        \
                              .Device("MUSA")                       \
                              .HostMemory("shape")                  \
                              .HostMemory("key")                    \
                              .HostMemory("counter")                \
                              .HostMemory("alg")                    \
                              .TypeConstraint<TYPE>("dtype"),       \
                          MusaRandomOp<TYPE>);

// 执行批量注册
REGISTER_MUSA_RANDOM_KERNELS(float);
REGISTER_MUSA_RANDOM_KERNELS(double);
REGISTER_MUSA_RANDOM_KERNELS(Eigen::half);
REGISTER_MUSA_RANDOM_KERNELS(bfloat16);

#undef REGISTER_MUSA_RANDOM_KERNELS

}  // namespace musa
}  // namespace tensorflow
