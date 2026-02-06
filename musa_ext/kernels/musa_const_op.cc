/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved.
 */
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/version.h"
#include "utils_op.h"  // 假设这里有 GetHandleByCtx 或类似获取 stream 的工具

namespace tensorflow {
namespace musa {

class MusaConstOp : public OpKernel {
 public:
  explicit MusaConstOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    const TensorProto* proto = nullptr;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));

    // 1. 先将数据解析到 Host (CPU) Tensor 中
    OP_REQUIRES(ctx, cpu_tensor_.FromProto(*proto),
                errors::InvalidArgument("Unparseable tensor proto"));

    OP_REQUIRES(
        ctx, cpu_tensor_.dtype() == ctx->output_type(0),
        errors::InvalidArgument("Type mismatch between value and output"));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, cpu_tensor_.shape(), &output));
    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    void* dst_ptr = const_cast<char*>(output->tensor_data().data());
    const void* src_ptr = cpu_tensor_.tensor_data().data();
    size_t total_bytes = cpu_tensor_.TotalBytes();

    // --- 修改这里 ---
    // 使用 musaError_t 接收 Runtime API 的结果
    musaError_t err =
        musaMemcpyAsync(dst_ptr, src_ptr, total_bytes, musaMemcpyHostToDevice,
                        (musaStream_t)handle.GetStream());

    // 检查是否成功 (musaSuccess 是 musaError_t 的成功标志)
    OP_REQUIRES(ctx, err == musaSuccess,
                errors::Internal("MUSA Const H2D Memcpy failed: ",
                                 musaGetErrorString(err)));
  }

 private:
  Tensor cpu_tensor_;  // 存储在算子对象中的 CPU 数据副本
};

// --- 算子注册 ---
// 注意：这里去掉了 .HostMemory("output")，因为我们要数据进显存
#define REGISTER_MUSA_CONST(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Const").Device(DEVICE_MTGPU).TypeConstraint<type>("dtype"), \
      MusaConstOp);

REGISTER_MUSA_CONST(float);
REGISTER_MUSA_CONST(double);
REGISTER_MUSA_CONST(int32);
REGISTER_MUSA_CONST(int64);
REGISTER_MUSA_CONST(bool);
REGISTER_MUSA_CONST(Eigen::half);
REGISTER_MUSA_CONST(bfloat16);  // 补全 BF16 支持

}  // namespace musa
}  // namespace tensorflow
