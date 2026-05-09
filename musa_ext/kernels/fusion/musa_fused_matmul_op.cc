#include <mudnn.h>

#include <cstdlib>

#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/matmul_bcast.h"

#define ENABLE_MUSA_DEBUG 0

namespace tensorflow {
namespace musa {

namespace {

inline bool ResolveTF32Enabled() {
  const char* tf32_env = std::getenv("MUSA_ENABLE_TF32");
  if (tf32_env == nullptr) {
    return true;
  }
  return std::atoi(tf32_env) != 0;
}

}  // namespace

REGISTER_OP("MusaFusedMatMul")
    .Input("a: T")
    .Input("b: T")
    .Input("bias: T")
    .Output("product: T")
    .Attr("T: {float, half, double, bfloat16}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("fused_ops: list(string) = []")
    .Attr("num_args: int >= 0 = 0")
    .Attr("epsilon: float = 0.0001")
    .SetShapeFn(::tensorflow::shape_inference::MatMulShape);

template <typename T>
class MusaFusedMatMulOp : public MusaOpKernel {
 public:
  explicit MusaFusedMatMulOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    if (ctx->HasAttr("transpose_a")) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &trans_x_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &trans_y_));
    } else {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_x", &trans_x_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("adj_y", &trans_y_));
    }

    std::vector<string> fused_ops;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops));

    if (fused_ops.size() == 1 && fused_ops[0] == "BiasAdd") {
      fusion_type_ = FusionType::BIAS_ADD;
      fusion_name_ = "BiasAdd";
    } else if (fused_ops.size() == 2 && fused_ops[0] == "BiasAdd" &&
               fused_ops[1] == "Relu") {
      fusion_type_ = FusionType::BIAS_ADD_RELU;
      fusion_name_ = "BiasAdd+Relu";
    } else {
      fusion_type_ = FusionType::BIAS_ADD;
      fusion_name_ = "Unknown(Default=BiasAdd)";
    }

    static const bool tf32_enabled_global = ResolveTF32Enabled();
    tf32_enabled_ = tf32_enabled_global;
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    const int dims0 = in0.dims();
    const int dims1 = in1.dims();
    OP_REQUIRES(ctx, dims0 >= 2 && dims1 >= 2,
                errors::InvalidArgument("Input tensors must have rank >= 2"));

    MatMulBCast bcast(in0.shape().dim_sizes(), in1.shape().dim_sizes());
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes: ", in0.shape().DebugString(), " vs ",
                    in1.shape().DebugString()));

    const int64 d0 = in0.dim_size(dims0 - 2);
    const int64 d1 = in0.dim_size(dims0 - 1);
    const int64 d2 = in1.dim_size(dims1 - 2);
    const int64 d3 = in1.dim_size(dims1 - 1);

    const int64 m = trans_x_ ? d1 : d0;
    const int64 k = trans_x_ ? d0 : d1;
    const int64 n = trans_y_ ? d2 : d3;
    const int64 k_check = trans_y_ ? d3 : d2;
    OP_REQUIRES(
        ctx, k == k_check,
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0] shape ", in0.shape().DebugString(),
            ", In[1] shape ", in1.shape().DebugString(),
            ", transpose_a=", trans_x_, ", transpose_b=", trans_y_));

    TensorShape out_shape = bcast.output_batch_shape();
    out_shape.AddDim(m);
    out_shape.AddDim(n);

    OP_REQUIRES(ctx, ctx->num_inputs() >= 3,
                errors::InvalidArgument("FusedMatMul requires Bias input"));
    const Tensor& bias = ctx->input(2);
    OP_REQUIRES(ctx, bias.dims() == 1,
                errors::InvalidArgument("Bias must be 1D, got shape ",
                                        bias.shape().DebugString()));
    OP_REQUIRES(ctx, bias.dim_size(0) == n,
                errors::InvalidArgument("Bias dim mismatch: bias shape ",
                                        bias.shape().DebugString(),
                                        ", expected [", n, "]"));

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    handle.SetAllowTF32(tf32_enabled_);

    mTensor mt_out = CreateMTensor(*out, format_);
    mTensor mt_bias = CreateMTensor(bias, format_);

    const int64_t out_batch = bcast.output_batch_shape().num_elements();
    mt_out.SetNdInfo({out_batch, m, n}, {m * n, n, 1});

    if (in0.NumElements() == 0 || in1.NumElements() == 0) {
      musaStream_t stream = handle.GetStream();
      musaError_t err =
          musaMemsetAsync(out->data(), 0, out->TotalBytes(), stream);
      OP_REQUIRES(ctx, err == musaSuccess,
                  errors::Internal("musaMemsetAsync failed: ",
                                   musaGetErrorString(err)));
      mBinary binary_op;
      auto status = binary_op.SetMode(::musa::dnn::Binary::Mode::ADD);
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("muDNN Binary SetMode(ADD) failed. Status: ",
                                   (int)status));
      status = binary_op.Run(handle, mt_out, mt_out, mt_bias);
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("BiasAdd failed. Status: ", (int)status));

      if (fusion_type_ == FusionType::BIAS_ADD_RELU) {
        mUnary unary_op;
        status = unary_op.SetMode(::musa::dnn::Unary::Mode::RELU);
        OP_REQUIRES(
            ctx, status == ::musa::dnn::Status::SUCCESS,
            errors::Internal("muDNN Unary SetMode(RELU) failed. Status: ",
                             (int)status));
        status = unary_op.Run(handle, mt_out, mt_out);
        OP_REQUIRES(
            ctx, status == ::musa::dnn::Status::SUCCESS,
            errors::Internal("Fused ReLU failed. Status: ", (int)status));
      }
      return;
    }

    mTensor mt0 = CreateMTensor(in0, format_);
    mTensor mt1 = CreateMTensor(in1, format_);

    auto ReshapeTo3D = [out_batch](mTensor& mt, const Tensor& t) {
      const int64_t dims = t.dims();
      const int64_t rows = t.dim_size(dims - 2);
      const int64_t cols = t.dim_size(dims - 1);
      const int64_t batch = t.NumElements() / (rows * cols);
      if (dims != 3 || (batch == 1 && out_batch > 1)) {
        mt.SetNdInfo(
            {batch == 1 && out_batch > 1 ? out_batch : batch, rows, cols},
            {batch == 1 && out_batch > 1 ? 0 : rows * cols, cols, 1});
      }
    };

    ReshapeTo3D(mt0, in0);
    ReshapeTo3D(mt1, in1);

    mBatchMatMul matmul_op;
    const auto compute_mode = (!tf32_enabled_ && (in0.dtype() == DT_FLOAT ||
                                                  in0.dtype() == DT_DOUBLE))
                                  ? mBatchMatMul::ComputeMode::SCALAR
                                  : mBatchMatMul::ComputeMode::TENSOR;
    auto status = matmul_op.SetComputeMode(compute_mode);
    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("muDNN BatchMatMul SetComputeMode failed. Status: ",
                         (int)status));
    status = matmul_op.SetTranspose(trans_x_, trans_y_);
    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("muDNN BatchMatMul SetTranspose failed. Status: ",
                         (int)status));
    status = matmul_op.SetAlpha(1.0);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN BatchMatMul SetAlpha failed. Status: ",
                                 (int)status));
    status = matmul_op.SetBeta(0.0);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN BatchMatMul SetBeta failed. Status: ",
                                 (int)status));
    status = matmul_op.SetGamma(1.0);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("muDNN BatchMatMul SetGamma failed. Status: ",
                                 (int)status));

    if (fusion_type_ == FusionType::BIAS_ADD_RELU) {
      if (tf32_enabled_) {
        ::musa::dnn::MatMulLtParam param;
        status =
            param.SetEpilogue(::musa::dnn::MatMulLtParam::MatMulLtEpilogueMode::
                                  MATMULLT_EPILOGUE_RELU_BIAS);
        OP_REQUIRES(
            ctx, status == ::musa::dnn::Status::SUCCESS,
            errors::Internal("muDNN MatMulLtParam SetEpilogue failed. Status: ",
                             (int)status));

        status =
            matmul_op.RunLt(handle, mt_out, mt0, mt1, mt_out, mt_bias, param);
        if (status == ::musa::dnn::Status::SUCCESS) {
          return;
        }
      }

      status = matmul_op.RunWithBiasAdd(handle, mt_out, mt0, mt1, mt_bias);
      OP_REQUIRES(
          ctx, status == ::musa::dnn::Status::SUCCESS,
          errors::Internal("MatMul+BiasAdd failed. Status: ", (int)status));

      mUnary unary_op;
      status = unary_op.SetMode(::musa::dnn::Unary::Mode::RELU);
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("muDNN Unary SetMode(RELU) failed. Status: ",
                                   (int)status));
      status = unary_op.Run(handle, mt_out, mt_out);
      OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                  errors::Internal("Fused ReLU failed. Status: ", (int)status));
      return;
    }

    status = matmul_op.RunWithBiasAdd(handle, mt_out, mt0, mt1, mt_bias);
    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MatMul+BiasAdd failed. Status: ", (int)status));
  }

  bool IsExpensive() override { return true; }

 private:
  bool trans_x_ = false;
  bool trans_y_ = false;
  bool tf32_enabled_ = false;
  std::string fusion_name_;
  enum class FusionType { BIAS_ADD, BIAS_ADD_RELU };
  FusionType fusion_type_ = FusionType::BIAS_ADD;
};

#define REGISTER_MUSA_FUSED_MATMUL(TYPE)                                \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("MusaFusedMatMul").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaFusedMatMulOp<TYPE>);

REGISTER_MUSA_FUSED_MATMUL(float);
REGISTER_MUSA_FUSED_MATMUL(double);
REGISTER_MUSA_FUSED_MATMUL(Eigen::half);
REGISTER_MUSA_FUSED_MATMUL(bfloat16);

#undef REGISTER_MUSA_FUSED_MATMUL

}  // namespace musa
}  // namespace tensorflow
