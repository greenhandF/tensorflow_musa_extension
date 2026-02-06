#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T>
class MusaAssignOp : public MusaOpKernel {
 public:
  explicit MusaAssignOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("validate_shape", &validate_shape_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_locking_));
  }

  void Compute(OpKernelContext* ctx) override {
    // 获取 ref 输入（要被赋值的变量）
    Tensor ref = ctx->mutable_input(0, use_locking_);
    const Tensor& value = ctx->input(1);

    // 验证形状（如果需要）
    if (validate_shape_) {
      OP_REQUIRES(ctx, ref.shape() == value.shape(),
                  errors::InvalidArgument(
                      "Assign requires shapes of both tensors to match. "
                      "ref shape: ",
                      ref.shape().DebugString(),
                      ", value shape: ", value.shape().DebugString()));
    } else {
      // 如果不验证形状，需要检查是否可以重新分配
      OP_REQUIRES(
          ctx, ref.shape().num_elements() == value.shape().num_elements(),
          errors::InvalidArgument(
              "Assign requires tensors to have the same number of elements. "
              "ref elements: ",
              ref.shape().num_elements(),
              ", value elements: ", value.shape().num_elements()));
    }

    const int64_t size = value.NumElements();

    // 空张量直接返回
    if (size == 0) {
      ctx->forward_ref_input_to_ref_output(0, 0);
      return;
    }

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = (musaStream_t)handle.GetStream();

    // 执行内存复制
    auto status =
        musaMemcpyAsync(ref.flat<T>().data(), value.flat<T>().data(),
                        size * sizeof(T), musaMemcpyDeviceToDevice, stream);
    OP_REQUIRES(ctx, status == musaSuccess,
                errors::Internal("MUSA memcpy failed in Assign: ",
                                 musaGetErrorString(status)));

    // 转发引用输入到输出
    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool validate_shape_ = true;
  bool use_locking_ = true;
};

// AssignVariableOp - 用于 ResourceVariable

// 注册 Assign Op
#define REGISTER_MUSA_ASSIGN(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("Assign").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaAssignOp<TYPE>)

REGISTER_MUSA_ASSIGN(float);
REGISTER_MUSA_ASSIGN(double);
REGISTER_MUSA_ASSIGN(int32);
REGISTER_MUSA_ASSIGN(int64);
REGISTER_MUSA_ASSIGN(bfloat16);
REGISTER_MUSA_ASSIGN(Eigen::half);
REGISTER_MUSA_ASSIGN(bool);
REGISTER_MUSA_ASSIGN(uint8);
REGISTER_MUSA_ASSIGN(int8);
REGISTER_MUSA_ASSIGN(int16);
REGISTER_MUSA_ASSIGN(complex64);
REGISTER_MUSA_ASSIGN(complex128);
#undef REGISTER_MUSA_ASSIGN

// 注册 AssignVariableOp

}  // namespace musa
}  // namespace tensorflow