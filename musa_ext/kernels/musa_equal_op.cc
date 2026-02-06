/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved.
 */
#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

class MusaEqualOp : public MusaOpKernel {
 public:
  explicit MusaEqualOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    // 1. 处理广播逻辑
    // TensorFlow 的 BCast 用于计算两个不一致 Shape 之间的对齐方式
    BCast bcast(BCast::Vec(in0.shape().dim_sizes()),
                BCast::Vec(in1.shape().dim_sizes()));
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument("Incompatible shapes for Equal op: ",
                                        in0.shape().DebugString(), " vs ",
                                        in1.shape().DebugString()));

    // 2. 分配输出张量
    // 关键点：Equal 算子的输出 dtype 必须是 DT_BOOL
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0, BCast::ToShape(bcast.output_shape()), &out));

    // 如果输出为空，直接返回
    if (out->NumElements() == 0) return;

    // 3. 准备 muDNN 句柄和张量描述符
    // CreateMTensor 会调用你补齐了 DT_BOOL 的 GetType 函数
    auto& handle = GetHandleByCtx(ctx);
    mTensor t0 = CreateMTensor(in0);
    mTensor t1 = CreateMTensor(in1);
    mTensor t_out = CreateMTensor(*out);

    // 4. 配置 muDNN Binary 算子
    ::musa::dnn::Binary op;
    // 设置模式为 EQ (Equality)
    auto status = op.SetMode(::musa::dnn::Binary::Mode::EQ);
    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("muDNN Binary SetMode(EQ) failed"));

    // 5. 执行算子
    // 由于 GetType(BOOL) 已经正确，muDNN 现在能识别 1-byte 的输出步长
    status = op.Run(handle, t_out, t0, t1);

    // 如果执行失败，打印状态码以便调试
    if (status != mStatus::SUCCESS) {
      LOG(ERROR) << "muDNN Equal Run failed, status: " << (int)status
                 << ". Input0 shape: " << in0.shape().DebugString()
                 << ", Input1 shape: " << in1.shape().DebugString();
    }

    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("muDNN Equal Run failed. Check if muDNN "
                                 "supports this specific broadcast pattern."));
  }
};

// --- 算子注册与多类型支持 ---

// 定义支持的输入类型宏
// 注意：对于半精度，TensorFlow 使用 Eigen::half，MUSA 使用 bfloat16
#define REGISTER_MUSA_EQUAL(type)                                   \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Equal").Device(DEVICE_MTGPU).TypeConstraint<type>("T"), \
      MusaEqualOp);

// 注册常用数据类型
REGISTER_MUSA_EQUAL(float);        // FP32
REGISTER_MUSA_EQUAL(double);       // FP64
REGISTER_MUSA_EQUAL(int32);        // INT32
REGISTER_MUSA_EQUAL(int64);        // INT64
REGISTER_MUSA_EQUAL(Eigen::half);  // FP16
REGISTER_MUSA_EQUAL(bfloat16);     // BF16

}  // namespace musa
}  // namespace tensorflow
