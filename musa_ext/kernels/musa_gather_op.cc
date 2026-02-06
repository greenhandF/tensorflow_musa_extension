#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename T, typename IndexT>
class MusaGatherOp : public MusaOpKernel {
 public:
  explicit MusaGatherOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // fprintf(stderr, ">>> [MUSA_TRACE_AUTO] %s\n", name().c_str());

    const Tensor& params = ctx->input(0);
    const Tensor& indices = ctx->input(1);
    const Tensor& axis_tensor = ctx->input(2);

    // LOG(INFO) << "Params dtype: " << params.dtype();
    // LOG(INFO) << "Indices dtype: " << indices.dtype();
    //  LOG(INFO) << "Axis dtype: " << axis_tensor.dtype();

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(axis_tensor.shape()),
                errors::InvalidArgument("axis must be a scalar"));

    int64_t axis = 0;
    if (axis_tensor.dtype() == DT_INT32) {
      axis = static_cast<int64_t>(axis_tensor.scalar<int32>()());
    } else if (axis_tensor.dtype() == DT_INT64) {
      axis = axis_tensor.scalar<int64>()();
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("axis must be int32 or int64"));
    }

    const int64_t params_dims = params.dims();
    if (axis < 0) {
      axis += params_dims;
    }

    OP_REQUIRES(
        ctx, axis >= 0 && axis < params_dims,
        errors::InvalidArgument("Expected axis in the range [", -params_dims,
                                ", ", params_dims, "), but got ", axis));

    OP_REQUIRES(ctx, indices.dtype() == DT_INT32 || indices.dtype() == DT_INT64,
                errors::InvalidArgument("indices must be int32 or int64"));

    TensorShape output_shape;

    for (int64_t i = 0; i < axis; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }

    for (int64_t i = 0; i < indices.dims(); ++i) {
      output_shape.AddDim(indices.dim_size(i));
    }

    for (int64_t i = axis + 1; i < params_dims; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mTensor t_params = CreateMTensor(params, format_);
    mTensor t_indices = CreateMTensor(indices, format_);
    mTensor t_output = CreateMTensor(*output, format_);

    mGatherX op;

    OP_REQUIRES(ctx, axis <= std::numeric_limits<int>::max(),
                errors::InvalidArgument("Axis value too large"));
    op.SetAxis(static_cast<int>(axis));

    auto status = op.Run(handle, t_output, t_indices, t_params);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA muDNN Gather execution failed. Status: ",
                                 static_cast<int>(status)));
  }
};

// 注册 GatherV2 算子（支持 Taxis 类型）
REGISTER_KERNEL_BUILDER(Name("GatherV2")
                            .Device("MUSA")
                            .TypeConstraint<float>("Tparams")
                            .TypeConstraint<int32>("Tindices")
                            .TypeConstraint<int32>("Taxis")
                            .HostMemory("axis"),
                        MusaGatherOp<float, int32>);

REGISTER_KERNEL_BUILDER(Name("GatherV2")
                            .Device("MUSA")
                            .TypeConstraint<float>("Tparams")
                            .TypeConstraint<int64>("Tindices")
                            .TypeConstraint<int64>("Taxis")
                            .HostMemory("axis"),
                        MusaGatherOp<float, int64>);

REGISTER_KERNEL_BUILDER(Name("GatherV2")
                            .Device("MUSA")
                            .TypeConstraint<Eigen::half>("Tparams")
                            .TypeConstraint<int32>("Tindices")
                            .TypeConstraint<int32>("Taxis")
                            .HostMemory("axis"),
                        MusaGatherOp<Eigen::half, int32>);

REGISTER_KERNEL_BUILDER(Name("GatherV2")
                            .Device("MUSA")
                            .TypeConstraint<Eigen::half>("Tparams")
                            .TypeConstraint<int64>("Tindices")
                            .TypeConstraint<int64>("Taxis")
                            .HostMemory("axis"),
                        MusaGatherOp<Eigen::half, int64>);

REGISTER_KERNEL_BUILDER(Name("GatherV2")
                            .Device("MUSA")
                            .TypeConstraint<bfloat16>("Tparams")
                            .TypeConstraint<int32>("Tindices")
                            .TypeConstraint<int32>("Taxis")
                            .HostMemory("axis"),
                        MusaGatherOp<bfloat16, int32>);

REGISTER_KERNEL_BUILDER(Name("GatherV2")
                            .Device("MUSA")
                            .TypeConstraint<bfloat16>("Tparams")
                            .TypeConstraint<int64>("Tindices")
                            .TypeConstraint<int64>("Taxis")
                            .HostMemory("axis"),
                        MusaGatherOp<bfloat16, int64>);

REGISTER_KERNEL_BUILDER(Name("GatherV2")
                            .Device("MUSA")
                            .TypeConstraint<double>("Tparams")
                            .TypeConstraint<int32>("Tindices")
                            .TypeConstraint<int32>("Taxis")
                            .HostMemory("axis"),
                        MusaGatherOp<double, int32>);

REGISTER_KERNEL_BUILDER(Name("GatherV2")
                            .Device("MUSA")
                            .TypeConstraint<double>("Tparams")
                            .TypeConstraint<int64>("Tindices")
                            .TypeConstraint<int64>("Taxis")
                            .HostMemory("axis"),
                        MusaGatherOp<double, int64>);

REGISTER_KERNEL_BUILDER(Name("GatherV2")
                            .Device("MUSA")
                            .TypeConstraint<int32>("Tparams")
                            .TypeConstraint<int32>("Tindices")
                            .TypeConstraint<int32>("Taxis")
                            .HostMemory("axis"),
                        MusaGatherOp<int32, int32>);

REGISTER_KERNEL_BUILDER(Name("GatherV2")
                            .Device("MUSA")
                            .TypeConstraint<int32>("Tparams")
                            .TypeConstraint<int64>("Tindices")
                            .TypeConstraint<int64>("Taxis")
                            .HostMemory("axis"),
                        MusaGatherOp<int32, int64>);

REGISTER_KERNEL_BUILDER(Name("GatherV2")
                            .Device("MUSA")
                            .TypeConstraint<int64>("Tparams")
                            .TypeConstraint<int32>("Tindices")
                            .TypeConstraint<int32>("Taxis")
                            .HostMemory("axis"),
                        MusaGatherOp<int64, int32>);

REGISTER_KERNEL_BUILDER(Name("GatherV2")
                            .Device("MUSA")
                            .TypeConstraint<int64>("Tparams")
                            .TypeConstraint<int64>("Tindices")
                            .TypeConstraint<int64>("Taxis")
                            .HostMemory("axis"),
                        MusaGatherOp<int64, int64>);

// 注册 Gather 算子（无 Taxis 类型）
REGISTER_KERNEL_BUILDER(Name("Gather")
                            .Device("MUSA")
                            .TypeConstraint<float>("Tparams")
                            .TypeConstraint<int32>("Tindices"),
                        MusaGatherOp<float, int32>);

REGISTER_KERNEL_BUILDER(Name("Gather")
                            .Device("MUSA")
                            .TypeConstraint<float>("Tparams")
                            .TypeConstraint<int64>("Tindices"),
                        MusaGatherOp<float, int64>);

REGISTER_KERNEL_BUILDER(Name("Gather")
                            .Device("MUSA")
                            .TypeConstraint<Eigen::half>("Tparams")
                            .TypeConstraint<int32>("Tindices"),
                        MusaGatherOp<Eigen::half, int32>);

REGISTER_KERNEL_BUILDER(Name("Gather")
                            .Device("MUSA")
                            .TypeConstraint<Eigen::half>("Tparams")
                            .TypeConstraint<int64>("Tindices"),
                        MusaGatherOp<Eigen::half, int64>);

REGISTER_KERNEL_BUILDER(Name("Gather")
                            .Device("MUSA")
                            .TypeConstraint<bfloat16>("Tparams")
                            .TypeConstraint<int32>("Tindices"),
                        MusaGatherOp<bfloat16, int32>);

REGISTER_KERNEL_BUILDER(Name("Gather")
                            .Device("MUSA")
                            .TypeConstraint<bfloat16>("Tparams")
                            .TypeConstraint<int64>("Tindices"),
                        MusaGatherOp<bfloat16, int64>);

REGISTER_KERNEL_BUILDER(Name("Gather")
                            .Device("MUSA")
                            .TypeConstraint<double>("Tparams")
                            .TypeConstraint<int32>("Tindices"),
                        MusaGatherOp<double, int32>);

REGISTER_KERNEL_BUILDER(Name("Gather")
                            .Device("MUSA")
                            .TypeConstraint<double>("Tparams")
                            .TypeConstraint<int64>("Tindices"),
                        MusaGatherOp<double, int64>);

REGISTER_KERNEL_BUILDER(Name("Gather")
                            .Device("MUSA")
                            .TypeConstraint<int32>("Tparams")
                            .TypeConstraint<int32>("Tindices"),
                        MusaGatherOp<int32, int32>);

REGISTER_KERNEL_BUILDER(Name("Gather")
                            .Device("MUSA")
                            .TypeConstraint<int32>("Tparams")
                            .TypeConstraint<int64>("Tindices"),
                        MusaGatherOp<int32, int64>);

REGISTER_KERNEL_BUILDER(Name("Gather")
                            .Device("MUSA")
                            .TypeConstraint<int64>("Tparams")
                            .TypeConstraint<int32>("Tindices"),
                        MusaGatherOp<int64, int32>);

REGISTER_KERNEL_BUILDER(Name("Gather")
                            .Device("MUSA")
                            .TypeConstraint<int64>("Tparams")
                            .TypeConstraint<int64>("Tindices"),
                        MusaGatherOp<int64, int64>);

}  // namespace musa
}  // namespace tensorflow
