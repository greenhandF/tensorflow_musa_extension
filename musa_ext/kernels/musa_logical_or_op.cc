/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved.
 */
#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

class MusaLogicalOrOp : public MusaOpKernel {
 public:
  explicit MusaLogicalOrOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    // 1. 处理广播逻辑
    BCast bcast(BCast::Vec(in0.shape().dim_sizes()),
                BCast::Vec(in1.shape().dim_sizes()));
    OP_REQUIRES(
        ctx, bcast.IsValid(),
        errors::InvalidArgument(
            "Incompatible shapes for LogicalOr op: ", in0.shape().DebugString(),
            " vs ", in1.shape().DebugString()));

    // 2. 分配输出张量
    // LogicalOr 的输入和输出在 TensorFlow 中通常强制为 DT_BOOL
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0, BCast::ToShape(bcast.output_shape()), &out));

    if (out->NumElements() == 0) return;

    // 3. 准备 muDNN 句柄和张量描述符
    // 此时 GetType(DT_BOOL) 必须返回 mType::BOOL，否则会再次导致内存越界
    auto& handle = GetHandleByCtx(ctx);
    mTensor t0 = CreateMTensor(in0);
    mTensor t1 = CreateMTensor(in1);
    mTensor t_out = CreateMTensor(*out);

    // 4. 配置 muDNN Binary 算子
    ::musa::dnn::Binary op;

    // 设置模式为 LOGICAL_OR
    // 如果你在之前的 grep 中看到了这个枚举值，它就可以被正确设置
    auto status = op.SetMode(::musa::dnn::Binary::Mode::LOGICAL_OR);
    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("muDNN Binary SetMode(LOGICAL_OR) failed"));

    // 5. 执行算子
    status = op.Run(handle, t_out, t0, t1);

    if (status != mStatus::SUCCESS) {
      LOG(ERROR) << "muDNN LogicalOr Run failed, status: " << (int)status;
    }

    OP_REQUIRES(
        ctx, status == mStatus::SUCCESS,
        errors::Internal(
            "muDNN LogicalOr Run failed. "
            "If this still crashes with out_of_range, "
            "it confirms muDNN missing internal kernel for BOOL logic."));
  }
};

// --- 算子注册 ---
// LogicalOr 在 TensorFlow 中只支持 bool 类型输入
REGISTER_KERNEL_BUILDER(Name("LogicalOr").Device(DEVICE_MTGPU),
                        MusaLogicalOrOp);

}  // namespace musa
}  // namespace tensorflow
