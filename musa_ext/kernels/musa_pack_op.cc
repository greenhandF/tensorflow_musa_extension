#include <mudnn.h>  // For musa::dnn::Tensor, musa::dnn::Concat, musa::dnn::Status
#include <musa_runtime.h>  // For musaStream_t

#include <vector>

#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "utils_op.h"  // 假设 CreateMTensor 和 GetHandleByCtx 在这里

namespace tensorflow {
namespace musa {

template <typename T>
class MusaPackOp : public MusaOpKernel {
 public:
  explicit MusaPackOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* ctx) override {
    const int N = ctx->num_inputs();
    const Tensor& first_input = ctx->input(0);
    const int dims = first_input.dims();

    int normalized_axis = axis_ < 0 ? axis_ + dims + 1 : axis_;

    TensorShape out_shape = first_input.shape();
    out_shape.InsertDim(normalized_axis, N);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));

    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    if (N == 1) {
      musaMemcpyAsync(const_cast<char*>(output->tensor_data().data()),
                      first_input.tensor_data().data(),
                      first_input.TotalBytes(), musaMemcpyDeviceToDevice,
                      stream);
      return;
    }

    // 【核心修正1】：移除所有关于数据类型的硬编码和转换。
    // musa::dnn::Tensor 的数据类型应该由 CreateMTensor 自动处理。
    // SetNdInfo 仅用于设置维度。

    // 【核心修正2】：将维度信息存储为 int64_t 类型，以匹配 SetNdInfo 的要求。
    std::vector<int64_t> expanded_dims_int64;
    for (int i = 0; i < dims; ++i) {
      expanded_dims_int64.push_back(
          static_cast<int64_t>(first_input.dim_size(i)));
    }
    // 在指定轴插入一个大小为 1 的维度，表示 Pack 的效果
    expanded_dims_int64.insert(expanded_dims_int64.begin() + normalized_axis,
                               1);

    std::vector<::musa::dnn::Tensor> mudnn_ins;
    mudnn_ins.reserve(N);

    for (int i = 0; i < N; ++i) {
      const Tensor& input_tensor = ctx->input(i);
      // CreateMTensor 应该负责将 TensorFlow Tensor 转换为 MUSA Tensor，
      // 并正确设置其数据指针和数据类型。
      ::musa::dnn::Tensor mt = CreateMTensor(input_tensor);

      // 【核心修正3】：调用 SetNdInfo 时，只传递维度数量和维度数组指针。
      // 使用 int64_t 作为 ndims 的类型，与 dim 数组的类型保持一致。
      auto status =
          mt.SetNdInfo(static_cast<int64_t>(expanded_dims_int64.size()),
                       expanded_dims_int64.data());

      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("muDNN SetNdInfo failed for input ", i));

      mudnn_ins.push_back(mt);
    }

    ::musa::dnn::Tensor mudnn_out = CreateMTensor(*output);

    ::musa::dnn::Concat concat_op;
    concat_op.SetAxis(normalized_axis);

    auto status =
        concat_op.Run(handle, mudnn_out, static_cast<int>(mudnn_ins.size()),
                      mudnn_ins.data());

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Pack Run failed."));
  }

 private:
  int axis_;
};

#define REGISTER_MUSA_PACK(TYPE) \
  REGISTER_KERNEL_BUILDER(       \
      Name("Pack").Device("MUSA").TypeConstraint<TYPE>("T"), MusaPackOp<TYPE>)

REGISTER_MUSA_PACK(float);
REGISTER_MUSA_PACK(int32);
REGISTER_MUSA_PACK(int64);
REGISTER_MUSA_PACK(Eigen::half);

#undef REGISTER_MUSA_PACK

}  // namespace musa
}  // namespace tensorflow
