#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

namespace {

mType GetMusaTypeLocal(DataType t) {
  switch (t) {
    case DataType::DT_FLOAT:
      return mType::FLOAT;
    case DataType::DT_HALF:
      return mType::HALF;
    case DataType::DT_BFLOAT16:
      return mType::BFLOAT16;
    case DataType::DT_INT32:
      return mType::INT32;
    case DataType::DT_INT64:
      return mType::INT64;
    case DataType::DT_DOUBLE:
      return mType::DOUBLE;
    case DataType::DT_BOOL:
      return mType::BOOL;
    case DataType::DT_INT8:
      return mType::INT8;
    case DataType::DT_UINT8:
      return mType::UINT8;
    default:
      return mType::FLOAT;
  }
}

mTensor CreateBroadcastMTensor(const Tensor& input,
                               const TensorShape& target_shape,
                               mFormat format) {
  mTensor rst;
  rst.SetAddr(
      const_cast<void*>(static_cast<const void*>(input.tensor_data().data())));
  rst.SetType(GetMusaTypeLocal(input.dtype()));
  rst.SetFormat(format);

  int target_rank = target_shape.dims();
  std::vector<int64_t> target_dims(target_rank);
  for (int i = 0; i < target_rank; ++i)
    target_dims[i] = target_shape.dim_size(i);

  std::vector<int64_t> input_strides(target_rank, 0);
  int input_rank = input.dims();
  std::vector<int64_t> dense_strides(input_rank, 1);
  if (input_rank > 0) {
    for (int i = input_rank - 2; i >= 0; --i) {
      dense_strides[i] = dense_strides[i + 1] * input.dim_size(i + 1);
    }
  }

  for (int i = 1; i <= target_rank; ++i) {
    int target_idx = target_rank - i;
    int input_idx = input_rank - i;

    if (input_idx >= 0) {
      int64_t in_dim = input.dim_size(input_idx);
      int64_t out_dim = target_dims[target_idx];

      if (in_dim == out_dim) {
        input_strides[target_idx] = dense_strides[input_idx];
      } else if (in_dim == 1) {
        input_strides[target_idx] = 0;
      }
    } else {
      input_strides[target_idx] = 0;
    }
  }

  rst.SetNdInfo(static_cast<int>(target_rank), target_dims.data(),
                input_strides.data());
  return rst;
}

}  // namespace

template <typename T>
class MusaSelectOp : public MusaOpKernel {
 public:
  explicit MusaSelectOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& cond = ctx->input(0);
    const Tensor& then_t = ctx->input(1);
    const Tensor& else_t = ctx->input(2);

    BCast bcast_te(BCast::FromShape(then_t.shape()),
                   BCast::FromShape(else_t.shape()));
    if (!bcast_te.IsValid()) {
      ctx->SetStatus(
          errors::InvalidArgument("Incompatible shapes: then vs else"));
      return;
    }
    TensorShape te_shape = BCast::ToShape(bcast_te.output_shape());

    BCast bcast_final(BCast::FromShape(cond.shape()),
                      BCast::FromShape(te_shape));
    if (!bcast_final.IsValid()) {
      ctx->SetStatus(
          errors::InvalidArgument("Incompatible shapes: cond vs (then/else)"));
      return;
    }
    TensorShape output_shape = BCast::ToShape(bcast_final.output_shape());

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    auto cond_mt = CreateBroadcastMTensor(cond, output_shape, format_);
    auto then_mt = CreateBroadcastMTensor(then_t, output_shape, format_);
    auto else_mt = CreateBroadcastMTensor(else_t, output_shape, format_);
    auto out_mt = CreateMTensor(*output, format_);

    ::musa::dnn::Ternary op;
    MTOP_CHECK_OK(op.SetMode(::musa::dnn::Ternary::Mode::SELECT),
                  "SetMode SELECT", ctx);
    MTOP_CHECK_OK_RUN(op.Run(handle, out_mt, cond_mt, then_mt, else_mt),
                      "Ternary Run", ctx);
  }
};

#define REGISTER_SELECT(T) \
  REGISTER_KERNEL_BUILDER( \
      Name("SelectV2").Device("MUSA").TypeConstraint<T>("T"), MusaSelectOp<T>)

REGISTER_SELECT(float);
REGISTER_SELECT(double);
REGISTER_SELECT(int32);
REGISTER_SELECT(int64);
REGISTER_SELECT(Eigen::half);
REGISTER_SELECT(Eigen::bfloat16);
REGISTER_SELECT(bool);

#undef REGISTER_SELECT

#define REGISTER_SELECT_V1(T) \
  REGISTER_KERNEL_BUILDER(    \
      Name("Select").Device("MUSA").TypeConstraint<T>("T"), MusaSelectOp<T>)

REGISTER_SELECT_V1(float);
REGISTER_SELECT_V1(double);
REGISTER_SELECT_V1(int32);
REGISTER_SELECT_V1(int64);
REGISTER_SELECT_V1(Eigen::half);
REGISTER_SELECT_V1(Eigen::bfloat16);
REGISTER_SELECT_V1(bool);

#undef REGISTER_SELECT_V1

}  // namespace musa
}  // namespace tensorflow
