/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved.
 */

#include <iostream>

#include "mu/device/musa_device.h"
#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

using Var = ::tensorflow::Var;

namespace {

/**
 * 1. MusaVarHandleOp (代理类)
 */
class MusaVarHandleOp : public OpKernel {
 public:
  explicit MusaVarHandleOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("container", &container_));
    OP_REQUIRES_OK(c, c->GetAttr("shared_name", &shared_name_));
  }
  void Compute(OpKernelContext* context) override {
    std::cerr << ">>> [MUSA_DEBUG] Step 1: VarHandleOp" << std::endl;
    Tensor* handle;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &handle));
    ResourceHandle r =
        MakeResourceHandle<Var>(context, container_, shared_name_);
    handle->flat<ResourceHandle>()(0) = r;
  }

 private:
  string container_, shared_name_;
};

/**
 * 2. MusaAssignVariableOp
 */
template <typename T>
class MusaAssignVariableOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;
  void Compute(OpKernelContext* context) override {
    std::cerr << ">>> [MUSA_DEBUG] Step 2: AssignVariable" << std::endl;
    const Tensor& value = context->input(1);
    core::RefCountPtr<Var> variable;
    OP_REQUIRES_OK(context, LookupOrCreateResource<Var>(
                                context, HandleFromInput(context, 0), &variable,
                                [this, &value](Var** ptr) {
                                  *ptr = new Var(value.dtype());
                                  *(*ptr)->tensor() = value;
                                  (*ptr)->is_initialized = true;
                                  return Status::OK();
                                }));
    mutex_lock ml(*variable->mu());
    *variable->tensor() = value;
    variable->is_initialized = true;
  }
};

/**
 * 3. MusaReadVariableOp
 */
template <typename T>
class MusaReadVariableOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;
  void Compute(OpKernelContext* context) override {
    std::cerr << ">>> [MUSA_DEBUG] Step 3: ReadVariable" << std::endl;
    core::RefCountPtr<Var> variable;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &variable));
    tf_shared_lock ml(*variable->mu());
    context->set_output(0, *variable->tensor());
  }
};

/**
 * 4. MusaAssignUpdateVariableOp (Add/Sub)
 */
template <typename T, mBinary::Mode BMODE>
class MusaAssignUpdateVariableOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;
  void Compute(OpKernelContext* context) override {
    std::cerr << ">>> [MUSA_DEBUG] Step 4: AssignUpdate" << std::endl;
    core::RefCountPtr<Var> variable;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &variable));
    const Tensor& value = context->input(1);
    mutex_lock ml(*variable->mu());
    Tensor* var_tensor = variable->tensor();

    if (!var_tensor->RefCountIsOne()) {
      Tensor tmp;
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      OP_REQUIRES_OK(context,
                     context->allocate_temp(var_tensor->dtype(),
                                            var_tensor->shape(), &tmp, attr));
      MusaMemcpyD2D(tmp.data(), var_tensor->data(), var_tensor->TotalBytes());
      *var_tensor = tmp;
    }

    auto in = CreateMTensor(value, format_);
    auto out = CreateMTensor(*var_tensor, format_);
    auto& h = GetHandleByCtx(context);
    mBinary op;
    op.SetMode(BMODE);
    MTOP_CHECK_OK_RUN(op.Run(h, out, out, in), "RunBinaryUpdate", context);
  }
};

/**
 * 5. ResourceGatherOp (Embedding Lookup)
 */
template <typename T, typename Index>
class MusaResourceGatherOp : public MusaOpKernel {
 public:
  explicit MusaResourceGatherOp(OpKernelConstruction* c) : MusaOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("batch_dims", &batch_dims_));
  }
  void Compute(OpKernelContext* c) override {
    std::cerr << ">>> [MUSA_DEBUG] Step 6: ResourceGather" << std::endl;
    core::RefCountPtr<Var> v;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
    tf_shared_lock ml(*v->mu());
    const Tensor& params = *v->tensor();
    const Tensor& indices = c->input(1);

    TensorShape res_shape;
    for (int i = 0; i < indices.dims(); ++i)
      res_shape.AddDim(indices.dim_size(i));
    for (int i = batch_dims_ + 1; i < params.dims(); ++i)
      res_shape.AddDim(params.dim_size(i));

    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, res_shape, &out));

    if (indices.NumElements() > 0) {
      auto out_mt = CreateMTensor(*out, format_);
      auto params_mt = CreateMTensor(params, format_);
      auto indices_mt = CreateMTensor(indices, format_);
      auto& h = GetHandleByCtx(c);
      mGatherX op;
      op.SetMode(mGatherX::Mode::GATHER);
      op.SetAxis(batch_dims_);
      MTOP_CHECK_OK_RUN(op.Run(h, out_mt, indices_mt, params_mt), "RunGatherX",
                        c);
    }
  }

 private:
  int32 batch_dims_;
};

/**
 * 6. ResourceScatterAddOp (Gradient Update)
 */
template <typename T, typename Index>
class MusaResourceScatterAddOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;
  void Compute(OpKernelContext* c) override {
    std::cerr << ">>> [MUSA_DEBUG] Step 7: ResourceScatterAdd" << std::endl;
    core::RefCountPtr<Var> v;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
    mutex_lock ml(*v->mu());
    Tensor* params = v->tensor();
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);

    if (indices.NumElements() > 0) {
      auto params_mt = CreateMTensor(*params, format_);
      auto indices_mt = CreateMTensor(indices, format_);
      auto updates_mt = CreateMTensor(updates, format_);
      auto& h = GetHandleByCtx(c);

      auto* device = static_cast<MusaDevice*>(c->device());
      auto maintainer = device->GetMemMaintainer(
          [](size_t size) { return ::musa::dnn::MemoryHandler(); });

      mScatter op;
      op.SetMode(mScatter::Mode::ADD);
      MTOP_CHECK_OK_RUN(
          op.Run(h, params_mt, indices_mt, updates_mt, 0, maintainer),
          "RunScatterAdd", c);
    }
  }
};

/**
 * 7. MusaDestroyResourceOp
 */
class MusaDestroyResourceOp : public OpKernel {
 public:
  explicit MusaDestroyResourceOp(OpKernelConstruction* c) : OpKernel(c) {}
  void Compute(OpKernelContext* context) override {
    std::cerr << ">>> [MUSA_DEBUG] Step 5: DestroyResource" << std::endl;
    const ResourceHandle& handle = HandleFromInput(context, 0);
    DeleteResource(context, handle);
  }
};

}  // namespace

// --- 精简注册逻辑 ---

// 全局唯一的算子注册
REGISTER_KERNEL_BUILDER(
    Name("DestroyResourceOp").Device("MUSA").HostMemory("resource"),
    MusaDestroyResourceOp);

// 按数据类型注册所有相关算子
#define REGISTER_MUSA_RESOURCE_KERNELS(type)                      \
  REGISTER_KERNEL_BUILDER(Name("VarHandleOp")                     \
                              .Device("MUSA")                     \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype"),     \
                          MusaVarHandleOp);                       \
  REGISTER_KERNEL_BUILDER(Name("AssignVariableOp")                \
                              .Device("MUSA")                     \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype"),     \
                          MusaAssignVariableOp<type>);            \
  REGISTER_KERNEL_BUILDER(Name("ReadVariableOp")                  \
                              .Device("MUSA")                     \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype"),     \
                          MusaReadVariableOp<type>);              \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("AssignAddVariableOp")                                 \
          .Device("MUSA")                                         \
          .HostMemory("resource")                                 \
          .TypeConstraint<type>("dtype"),                         \
      MusaAssignUpdateVariableOp<type, mBinary::Mode::ADD>);      \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("AssignSubVariableOp")                                 \
          .Device("MUSA")                                         \
          .HostMemory("resource")                                 \
          .TypeConstraint<type>("dtype"),                         \
      MusaAssignUpdateVariableOp<type, mBinary::Mode::SUB>);      \
  REGISTER_KERNEL_BUILDER(Name("ResourceGather")                  \
                              .Device("MUSA")                     \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int32>("Tindices"), \
                          MusaResourceGatherOp<type, int32>);     \
  REGISTER_KERNEL_BUILDER(Name("ResourceGather")                  \
                              .Device("MUSA")                     \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int64>("Tindices"), \
                          MusaResourceGatherOp<type, int64>);     \
  REGISTER_KERNEL_BUILDER(Name("ResourceScatterAdd")              \
                              .Device("MUSA")                     \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int32>("Tindices"), \
                          MusaResourceScatterAddOp<type, int32>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceScatterAdd")              \
                              .Device("MUSA")                     \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int64>("Tindices"), \
                          MusaResourceScatterAddOp<type, int64>);

// 为浮点类型注册完整算子
REGISTER_MUSA_RESOURCE_KERNELS(float);
REGISTER_MUSA_RESOURCE_KERNELS(Eigen::half);
REGISTER_MUSA_RESOURCE_KERNELS(bfloat16);

// 为 int32 注册基础变量操作（去掉了 Gather/Scatter）
REGISTER_KERNEL_BUILDER(Name("VarHandleOp")
                            .Device("MUSA")
                            .HostMemory("resource")
                            .TypeConstraint<int32>("dtype"),
                        MusaVarHandleOp);
REGISTER_KERNEL_BUILDER(Name("AssignVariableOp")
                            .Device("MUSA")
                            .HostMemory("resource")
                            .TypeConstraint<int32>("dtype"),
                        MusaAssignVariableOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ReadVariableOp")
                            .Device("MUSA")
                            .HostMemory("resource")
                            .TypeConstraint<int32>("dtype"),
                        MusaReadVariableOp<int32>);

}  // namespace musa
}  // namespace tensorflow
