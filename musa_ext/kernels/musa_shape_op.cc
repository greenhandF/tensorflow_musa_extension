#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

template <typename OutType>
class MusaShapeOp : public MusaOpKernel {
 public:
  explicit MusaShapeOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const TensorShape& shape = input.shape();
    const int rank = shape.dims();

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({rank}), &output));

    auto flat_output = output->flat<OutType>();
    for (int i = 0; i < rank; ++i) {
      flat_output(i) = static_cast<OutType>(shape.dim_size(i));
    }
  }

  bool IsExpensive() override { return false; }
};

#define REGISTER_MUSA_SHAPE(out_type)                                \
  REGISTER_KERNEL_BUILDER(Name("Shape")                              \
                              .Device("MUSA")                        \
                              .HostMemory("output")                  \
                              .TypeConstraint<out_type>("out_type"), \
                          MusaShapeOp<out_type>)

REGISTER_MUSA_SHAPE(int32);
REGISTER_MUSA_SHAPE(int64);

#undef REGISTER_MUSA_SHAPE

}  // namespace musa
}  // namespace tensorflow
