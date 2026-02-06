#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
namespace tensorflow {
namespace musa {

/*		   REGISTER_OP("Identity")
                       .Input("input: T")
                .Output("output: T")
     .Attr("T: type")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);
*/
class MusaIdentityOp : public OpKernel {
 public:
  explicit MusaIdentityOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //    fprintf(stderr, ">>> [MUSA_TRACE_AUTO] %s\n", name().c_str());

    if (context->num_inputs() > 0) {
      const Tensor& in0 = context->input(0);

      name().c_str(),

          context->set_output(0, in0);
    }
  }
};

#define REGISTER_MUSA_IDENTITY(type)                             \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Identity").Device("MUSA").TypeConstraint<type>("T"), \
      MusaIdentityOp);

REGISTER_MUSA_IDENTITY(float);
REGISTER_MUSA_IDENTITY(double);
REGISTER_MUSA_IDENTITY(Eigen::half);
REGISTER_MUSA_IDENTITY(int32);
REGISTER_MUSA_IDENTITY(int64);
REGISTER_MUSA_IDENTITY(bool);

}  // namespace musa
}  // namespace tensorflow
