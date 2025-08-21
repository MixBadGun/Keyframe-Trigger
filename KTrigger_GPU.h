#define HAS_CUDA 0

#pragma once
#ifndef KTGPU_H
#define KTGPU_H

#include <stdio.h>
#include "KTrigger.h"

#if _WIN32
#include <CL/cl.h>
#define HAS_METAL 0
#else
#include <OpenCL/cl.h>
#define HAS_METAL 1
#include <Metal/Metal.h>
#include "KTrigger_GPU_Kernel.metal.h"
#endif
#include <driver_types.h>

#if HAS_METAL
/*
 ** Plugins must not rely on a host autorelease pool.
 ** Create a pool if autorelease is used, or Cocoa convention calls, such as
 *Metal, might internally autorelease.
 */
struct ScopedAutoreleasePool {
  ScopedAutoreleasePool() : mPool([[NSAutoreleasePool alloc] init]) {}

  ~ScopedAutoreleasePool() { [mPool release]; }

  NSAutoreleasePool *mPool;
};
#endif

inline PF_Err CL2Err(cl_int cl_result) {
  if (cl_result == CL_SUCCESS) {
    return PF_Err_NONE;
  } else {
    // set a breakpoint here to pick up OpenCL errors.
    return PF_Err_INTERNAL_STRUCT_DAMAGED;
  }
}

#define CL_ERR(FUNC) ERR(CL2Err(FUNC))

#define CUDA_CHECK(FUNC)                                                                                          \
  do {                                                                                                            \
    if (!err) {                                                                                                   \
      cudaError_t cerr = (FUNC);                                                                                  \
      if (cerr != cudaSuccess) {                                                                                  \
        (*in_dataP->utils->ansi.sprintf)(out_dataP->return_msg, "GPU Assert:\r\n%s\n", cudaGetErrorString(cerr)); \
        out_dataP->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;                                                 \
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;                                                                     \
      }                                                                                                           \
    }                                                                                                             \
  } while (0)

extern void Main_CUDA(float const* src,
                      float* dst,
                      cudaArray* d_envArray, cudaChannelFormatDesc channelDesc,
                      unsigned int srcPitch,
                      unsigned int dstPitch,
                      int is16f,
                      unsigned int width,
                      unsigned int height, float parameter, float time);

// GPU data initialized at GPU setup and used during render.
struct OpenCLGPUData {
  cl_kernel composite_layer_kernel;
};

void ClearToTransparentBlack_CUDA(float* output, int width, int height, int pitch);
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
    bool premultiplied);
#if HAS_METAL
struct MetalGPUData {
  id<MTLComputePipelineState> composite_layer_kernel;
};
#endif

typedef struct {
  int   mSrcPitch;
  int   mDstPitch;
  int   m16f;
  int   mWidth;
  int   mHeight;
} InputKernelParams;

PF_Err GPUDeviceSetup(PF_InData *in_dataP, PF_OutData *out_dataP, PF_GPUDeviceSetupExtra *extraP);

PF_Err GPUDeviceSetdown(PF_InData *in_dataP, PF_OutData *out_dataP, PF_GPUDeviceSetdownExtra *extraP);

#endif