#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

REGISTER_OP("MusaSigmoidGrad")
    .Input("y: T")
    .Input("dy: T")
    .Output("z: T")
    .Attr("T: {float, double, half}")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

template <typename T>
class MusaSigmoidGradOp : public MusaOpKernel {
 public:
  explicit MusaSigmoidGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    fprintf(stderr, ">>> [MUSA_TRACE_AUTO] %s\n", name().c_str());
    const Tensor& y = ctx->input(0);
    const Tensor& dy = ctx->input(1);

    Tensor* dz = nullptr;

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, y.shape(), &dz));

    if (y.NumElements() == 0) return;
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SigmoidGrad").Device("MUSA").TypeConstraint<float>("T"),
    MusaSigmoidGradOp<float>);

}  // namespace musa
}  // namespace tensorflow
