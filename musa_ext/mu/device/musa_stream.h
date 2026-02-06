#ifndef TENSORFLOW_MUSA_MU1_DEVICE_MUSA_STREAM_H_
#define TENSORFLOW_MUSA_MU1_DEVICE_MUSA_STREAM_H_

#include <musa_runtime.h>

#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace musa {

class MusaStream : public internal::StreamInterface {
 public:
  explicit MusaStream(musaStream_t stream) : musa_stream_(stream) {}
  ~MusaStream() override {}

  void* GpuStreamHack() override { return (void*)musa_stream_; }
  void** GpuStreamMemberHack() override {
    return reinterpret_cast<void**>(&musa_stream_);
  }

 private:
  musaStream_t musa_stream_;
};

}  // namespace musa
}  // namespace stream_executor

#endif
