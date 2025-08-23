#define HAS_CUDA 1
#define __CUDACC__ 1

#define __CUDA_INTERNAL_COMPILATION__
#include "KTrigger_GPU.h"
#include <device_launch_parameters.h>
#include "PrGPU/KernelSupport/KernelCore.h"
#include "PrGPU/KernelSupport/KernelMemory.h"
#undef __CUDA_INTERNAL_COMPILATION__

#if HAS_CUDA
    #include <cuda_runtime.h>
#endif

#if HAS_CUDA

void ClearToTransparentBlack_CUDA(float* output, int width, int height, int pitch)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    // clear_to_transparent_black_kernel<<<gridSize, blockSize>>>(
    //     (float4*)output, width, height, pitch);
    
    cudaDeviceSynchronize();
}

// 图层合成内核
GF_KERNEL_FUNCTION(
    composite_layer_kernel,
    // buffers
    ((const GF_PTR(float4))(src))
    ((GF_PTR(float4))(dst)),
    // Values
    ((int)(src_width))
    ((int)(src_height))
    ((int)(src_pitch))
    ((int)(dst_width))
    ((int)(dst_height))
    ((int)(dst_pitch))
    ((GF_PTR(PF_FloatMatrix))(matrix))  // 变换矩阵
    ((int)(blend_mode))          // 混合模式
    ((float)(opacity)),
    // Position
    ((uint2)(inXY)(KERNEL_XY)))           // 不透明度 0-255
{
    // 计算目标像素位置
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    // 应用变换矩阵（逆变换，从目标到源）
    PF_FpLong src_x_f = (dst_x * matrix->mat[0][0] + dst_y * matrix->mat[0][1] + matrix->mat[0][2]) / 65536.0;
    PF_FpLong src_y_f = (dst_x * matrix->mat[1][0] + dst_y * matrix->mat[1][1] + matrix->mat[1][2]) / 65536.0;
    
    int src_x = (int)src_x_f;
    int src_y = (int)src_y_f;
    
    // 检查源坐标是否在范围内
    if (src_x < 0 || src_x >= src_width || src_y < 0 || src_y >= src_height) {
        return; // 超出边界，不处理
    }
    
    int src_idx = src_y * src_pitch + src_x;
    int dst_idx = dst_y * dst_pitch + dst_x;
    
    float4 src_pixel = src[src_idx];
    float4 dst_pixel = dst[dst_idx];
    
    // 应用不透明度
    float alpha = src_pixel.w * (opacity / 255.0f);
    
    // 根据混合模式进行合成
    float4 result;
    switch (blend_mode) {
        case PF_Xfer_IN_FRONT: // 正常混合
            result.x = src_pixel.x * alpha + dst_pixel.x * (1.0f - alpha);
            result.y = src_pixel.y * alpha + dst_pixel.y * (1.0f - alpha);
            result.z = src_pixel.z * alpha + dst_pixel.z * (1.0f - alpha);
            result.w = alpha + dst_pixel.w * (1.0f - alpha);
            break;
            
        case PF_Xfer_ADD: // 相加模式
            result.x = min(src_pixel.x * alpha + dst_pixel.x, 1.0f);
            result.y = min(src_pixel.y * alpha + dst_pixel.y, 1.0f);
            result.z = min(src_pixel.z * alpha + dst_pixel.z, 1.0f);
            result.w = min(alpha + dst_pixel.w, 1.0f);
            break;
            
        // 可以添加更多混合模式...
            
        default: // 默认正常混合
            result.x = src_pixel.x * alpha + dst_pixel.x * (1.0f - alpha);
            result.y = src_pixel.y * alpha + dst_pixel.y * (1.0f - alpha);
            result.z = src_pixel.z * alpha + dst_pixel.z * (1.0f - alpha);
            result.w = alpha + dst_pixel.w * (1.0f - alpha);
    }
    
    dst[dst_idx] = result;
}

PF_Err CompositeLayer_CUDA(
    const float* src,
    float* dst,
    int src_width,
    int src_height,
    int src_pitch,
    int dst_width,
    int dst_height,
    int dst_pitch,
    PF_FloatMatrix* matrix,
    int blend_mode,
    int opacity,
    bool premultiplied)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((dst_width + blockSize.x - 1) / blockSize.x, 
                  (dst_height + blockSize.y - 1) / blockSize.y);
    
    composite_layer_kernel<<<gridSize, blockSize>>>(
        (const float4*)src,
        (float4*)dst,
        src_width,
        src_height,
        src_pitch,
        dst_width,
        dst_height,
        dst_pitch,
        matrix,
        blend_mode,
        (float)opacity);
    
    if (cudaPeekAtLastError() != cudaSuccess) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    
    cudaDeviceSynchronize();

    return PF_Err_NONE;
}
#endif // HAS_CUDA