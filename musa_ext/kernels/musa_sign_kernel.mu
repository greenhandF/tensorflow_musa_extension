#include <musa_runtime.h>
#include <musa_fp16.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#pragma GCC diagnostic pop

namespace tensorflow
{
  namespace musa
  {

    // ============ 辅助转换函数 ============

    __device__ __forceinline__ float HalfToFloat(const Eigen::half &h)
    {
      __half tmp;
      memcpy(&tmp, &h, sizeof(__half));
      return __half2float(tmp);
    }

    __device__ __forceinline__ Eigen::half FloatToHalf(float f)
    {
      __half tmp = __float2half(f);
      Eigen::half result;
      memcpy(&result, &tmp, sizeof(__half));
      return result;
    }

    __device__ __forceinline__ float Bfloat16ToFloat(const tensorflow::bfloat16 &bf)
    {
      uint16_t raw;
      memcpy(&raw, &bf, sizeof(uint16_t));
      uint32_t bits = static_cast<uint32_t>(raw) << 16;
      float result;
      memcpy(&result, &bits, sizeof(float));
      return result;
    }

    __device__ __forceinline__ tensorflow::bfloat16 FloatToBfloat16(float f)
    {
      uint32_t bits;
      memcpy(&bits, &f, sizeof(float));
      uint16_t raw = static_cast<uint16_t>(bits >> 16);
      tensorflow::bfloat16 result;
      memcpy(&result, &raw, sizeof(uint16_t));
      return result;
    }

    // ============ Kernel 实现 ============

    // 通用类型 Sign kernel
    template <typename T>
    __global__ void MusaSignKernel(const T *input, T *output, int64_t size)
    {
      int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

      if (idx < size)
      {
        T val = input[idx];
        T zero = static_cast<T>(0);
        T one = static_cast<T>(1);
        T neg_one = static_cast<T>(-1);

        if (val > zero)
        {
          output[idx] = one;
        }
        else if (val < zero)
        {
          output[idx] = neg_one;
        }
        else
        {
          output[idx] = zero;
        }
      }
    }

    //Eigen::half 特化 kernel
    __global__ void MusaSignKernelHalf(const Eigen::half *input,
                                       Eigen::half *output,
                                       int64_t size)
    {
      int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

      if (idx < size)
      {
        float val = HalfToFloat(input[idx]);

        float result;
        if (val > 0.0f)
        {
          result = 1.0f;
        }
        else if (val < 0.0f)
        {
          result = -1.0f;
        }
        else
        {
          result = 0.0f;
        }

        output[idx] = FloatToHalf(result);
      }
    }

    // bfloat16 特化 kernel
    __global__ void MusaSignKernelBfloat16(const tensorflow::bfloat16 *input,
                                           tensorflow::bfloat16 *output,
                                           int64_t size)
    {
      int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

      if (idx < size)
      {
        float val = Bfloat16ToFloat(input[idx]);

        float result;
        if (val > 0.0f)
        {
          result = 1.0f;
        }
        else if (val < 0.0f)
        {
          result = -1.0f;
        }
        else
        {
          result = 0.0f;
        }

        output[idx] = FloatToBfloat16(result);
      }
    }

    // ============ Launcher 实现 ============

    // 通用类型 launcher 模板
    template <typename T>
    void MusaSignKernelLauncher(const T *input, T *output, int64_t size)
    {
      if (size == 0)
        return;

      const int block_size = 256;
      const int num_blocks = (size + block_size - 1) / block_size;

      MusaSignKernel<T><<<num_blocks, block_size, 0>>>(input, output, size);
    }

    //Eigen::half 特化 launcher
    template <>
    void MusaSignKernelLauncher<Eigen::half>(const Eigen::half *input,
                                             Eigen::half *output,
                                             int64_t size)
    {
      if (size == 0)
        return;

      const int block_size = 256;
      const int num_blocks = (size + block_size - 1) / block_size;

      MusaSignKernelHalf<<<num_blocks, block_size, 0>>>(input, output, size);
    }

    //bfloat16 特化 launcher
    template <>
    void MusaSignKernelLauncher<tensorflow::bfloat16>(const tensorflow::bfloat16 *input,
                                                      tensorflow::bfloat16 *output,
                                                      int64_t size)
    {
      if (size == 0)
        return;

      const int block_size = 256;
      const int num_blocks = (size + block_size - 1) / block_size;

      MusaSignKernelBfloat16<<<num_blocks, block_size, 0>>>(input, output, size);
    }

    // ============ 显式实例化（仅通用类型）============
    template void MusaSignKernelLauncher<float>(const float *, float *, int64_t);
    template void MusaSignKernelLauncher<int>(const int *, int *, int64_t);
    template void MusaSignKernelLauncher<long long>(const long long *, long long *, int64_t);

  } // namespace musa
} // namespace tensorflow