#include <musa_runtime.h>  // 必须包含这个头文件以使用 musaGetDeviceCount
#include <stdio.h>

#include <vector>

#include "device/musa_device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
void ForceMusaOptimizationPassRegistration();
}

namespace tensorflow {
namespace musa {

class MusaDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override {
    int count = 0;
    // 【核心修复】先询问驱动有几张卡
    musaError_t err = musaGetDeviceCount(&count);
    if (err != musaSuccess) {
      fprintf(stderr, ">>>> [MUSA] ERROR: musaGetDeviceCount failed: %d\n",
              err);
      return Status::OK();  // 返回空列表
    }

    fprintf(stderr,
            ">>>> [MUSA] DeviceFactory detected %d physical devices <<<<\n",
            count);

    // 只循环实际存在的数量
    for (int i = 0; i < count; ++i) {
      devices->push_back(strings::StrCat("/physical_device:MUSA:", i));
    }
    return Status::OK();
  }

  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    int count = 0;
    // 【核心修复】再次确认数量（防止竞争条件）
    musaError_t err = musaGetDeviceCount(&count);
    if (err != musaSuccess) {
      return errors::Internal("Failed to get MUSA device count");
    }

    fprintf(stderr,
            ">>>> [MUSA] DeviceFactory creating %d logical instances <<<<\n",
            count);

    for (int i = 0; i < count; ++i) {
      DeviceAttributes attr;
      // 这里的 i 是逻辑 ID (0, 1, 2...)，永远是合法的
      string name = strings::StrCat(name_prefix, "/device:MUSA:", i);
      attr.set_name(name);
      attr.set_device_type("MUSA");

      // 建议：实际显存大小也应该从 musaDeviceProp 查询，这里先保留你的硬编码
      attr.set_memory_limit(16ULL * 1024 * 1024 * 1024);

      attr.mutable_locality()->set_bus_id(i);
      attr.set_physical_device_desc(strings::StrCat("device: MUSA device ", i));

      // 传入的 i 是逻辑 ID，直接传给 MusaDevice
      devices->push_back(
          std::unique_ptr<Device>(new MusaDevice(Env::Default(), attr, i)));

      fprintf(stderr,
              ">>>> [MUSA] Logical Device /device:MUSA:%d created. <<<<\n", i);
    }
    return Status::OK();
  }
};

// 注册优先级建议设高一点，或者保持 210
REGISTER_LOCAL_DEVICE_FACTORY("MUSA", MusaDeviceFactory, 210);

}  // namespace musa
}  // namespace tensorflow

// ... (后面的 Global Constructor 保持不变) ...

extern "C" {
void __attribute__((constructor)) OnMusaPluginLoad() {
  fprintf(stderr,
          "\n>>>> [MUSA] SUCCESS: MUSA Factory Object Registered via Global "
          "Constructor! <<<<\n");

  tensorflow::ForceMusaOptimizationPassRegistration();

  fprintf(stderr,
          ">>>> [MUSA] SUCCESS: MUSA Optimization Passes Activated! <<<<\n\n");
}
}
extern "C" void ForceLinkMusaAmpOptimizer();
