// #include "KTrigger_GPU.h"
// #include <AE_EffectGPUSuites.h>
// #include <AEFX_SuiteHandlerTemplate.h>

// // static BOOL didSaveShaderFile = NO;

// #if HAS_METAL
// PF_Err NSError2PFErr(NSError *inError) {
//   if (inError) {
//     return PF_Err_INTERNAL_STRUCT_DAMAGED; // For debugging, uncomment above
//                                            // line and set breakpoint here
//   }
//   return PF_Err_NONE;
// }
// #endif // HAS_METAL

// static size_t RoundUp(size_t inValue, size_t inMultiple) { return inValue ? ((inValue + inMultiple - 1) / inMultiple) * inMultiple : 0; }

// static size_t DivideRoundUp(size_t inValue, size_t inMultiple) { return inValue ? (inValue + inMultiple - 1) / inMultiple : 0; }

// PF_Err GPUDeviceSetup(PF_InData *in_dataP, PF_OutData *out_dataP, PF_GPUDeviceSetupExtra *extraP) {
//   PF_Err            err = PF_Err_NONE;
//   AEGP_SuiteHandler suites(in_dataP->pica_basicP);
// //   auto              *globalData = static_cast<GlobalData *>(suites.HandleSuite1()->host_lock_handle(in_dataP->global_data));

//   PF_GPUDeviceInfo device_info;
//   AEFX_CLR_STRUCT(device_info);

//   AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite, kPFHandleSuiteVersion1,
//                                                                                      out_dataP);

//   AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpuDeviceSuite = AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP, kPFGPUDeviceSuite,
//                                                                                              kPFGPUDeviceSuiteVersion1, out_dataP);

//   gpuDeviceSuite->GetDeviceInfo(in_dataP->effect_ref, extraP->input->device_index, &device_info);

//   // Load and compile the kernel - a real plugin would cache binaries to disk

//   if (extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
//     // Nothing to do here. CUDA Kernel statically linked
//     out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
//   } else if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL) {
//     PF_Handle      gpu_dataH = handle_suite->host_new_handle(sizeof(OpenCLGPUData));
//     OpenCLGPUData *cl_gpu_data = reinterpret_cast<OpenCLGPUData *>(*gpu_dataH);

//     cl_int result = CL_SUCCESS;

//     char const *k16fString = "#define GF_OPENCL_SUPPORTS_16F 0\n";

//     auto kGPU_Skeleton_Kernel_OpenCLString = "";

//     const size_t sizes[] = {strlen(k16fString),
//                             strlen(kGPU_Skeleton_Kernel_OpenCLString)};
//     char const * strings[] = { k16fString, kGPU_Skeleton_Kernel_OpenCLString };
//     cl_context   context = (cl_context)device_info.contextPV;
//     cl_device_id device = (cl_device_id)device_info.devicePV;

//     cl_program program;
//     if (!err) {
//       program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
//       CL_ERR(result);
//     }

//     CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

//     size_t logSize;
//     clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

//     if (logSize > 1) {
//       char *log = (char *)malloc(logSize);
//       clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
//       MessageBox(NULL, log, "OpenCL Error", MB_ICONERROR | MB_OK);
//       free(log);
//     }

//     if (!err) {
//       cl_gpu_data->composite_layer_kernel = clCreateKernel(program, "MainKernel", &result);
//       CL_ERR(result);
//     }

//     extraP->output->gpu_data = gpu_dataH;

//     out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
//   }
// #if HAS_METAL
//   else if (extraP->input->what_gpu == PF_GPU_Framework_METAL) {
//     ScopedAutoreleasePool pool;

//     // Create a library from source
//     NSString *    source = [NSString stringWithCString:kGPU_Skeleton_Kernel_MetalString encoding:NSUTF8StringEncoding];
//     id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;

//     NSError *      error = nil;
//     id<MTLLibrary> library = [[device newLibraryWithSource:source options:nil error:&error] autorelease];

//     // An error code is set for Metal compile warnings, so use nil library as
//     // the error signal
//     if (!err && !library) { err = NSError2PFErr(error); }

//     // For debugging only. This will contain Metal compile warnings and erorrs.
//     NSString *getError = error.localizedDescription;
//     if (error) {
//       globalData->sceneInfo->status = "Compiling Error";
//       globalData->sceneInfo->errorLog = "";

//       string input = getError.UTF8String;
//       regex  re(R"((\d+:\d+).*(error): (.*)\n(?:\s*)(.*)\n)");
//       smatch m;

//       while (regex_search(input, m, re)) {
//         string number = m[1].str();
//         string type = m[2].str();
//         string message = m[3].str();
//         string code = m[4].str();
//         globalData->sceneInfo->errorLog += type + " [" + number + "] " + message + ": " + code + "\n";
//         input = m.suffix().str();
//       }
//       cout << globalData->sceneInfo->errorLog << endl;

//       (*in_dataP->utils->ansi.sprintf)(out_dataP->return_msg, globalData->sceneInfo->errorLog.c_str());
//       out_dataP->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;
//     }

//     PF_Handle     metal_handle = handle_suite->host_new_handle(sizeof(MetalGPUData));
//     MetalGPUData *metal_data = reinterpret_cast<MetalGPUData *>(*metal_handle);

//     // Create pipeline state from function extracted from library
//     if (err == PF_Err_NONE) {
//       id<MTLFunction> main_function = nil;

//       NSString *func_name = [NSString stringWithCString:"MainKernel" encoding:NSUTF8StringEncoding];

//       main_function = [[library newFunctionWithName:func_name] autorelease];

//       if (!main_function) { err = PF_Err_INTERNAL_STRUCT_DAMAGED; }

//       if (!err) {
//         metal_data->composite_layer_kernel = [device newComputePipelineStateWithFunction:main_function error:&error];
//         err = NSError2PFErr(error);
//       }

//       if (!err) {
//         extraP->output->gpu_data = metal_handle;
//         out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;

//         globalData->sceneInfo->status = "Compiled Successfully";
//         FX_LOG(globalData->sceneInfo->status);
//       }
//     }
//   }
// #endif

//   suites.HandleSuite1()->host_unlock_handle(in_dataP->global_data);
//   return err;
// }

// PF_Err GPUDeviceSetdown(PF_InData *in_dataP, PF_OutData *out_dataP, PF_GPUDeviceSetdownExtra *extraP) {
//   PF_Err err = PF_Err_NONE;

//   if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL) {
//     PF_Handle      gpu_dataH = (PF_Handle)extraP->input->gpu_data;
//     OpenCLGPUData *cl_gpu_dataP = reinterpret_cast<OpenCLGPUData *>(*gpu_dataH);

//     (void)clReleaseKernel(cl_gpu_dataP->composite_layer_kernel);

//     AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite, kPFHandleSuiteVersion1,
//                                                                                        out_dataP);

//     handle_suite->host_dispose_handle(gpu_dataH);
//   }

//   return err;
// }

// PF_Err SmartRenderGPU(
//     PF_InData *in_data,
//     PF_OutData *out_data,
//     PF_SmartRenderExtra *extra
// )
// {
//     A_long bytes_per_pixel = 16;

// 	PF_EffectWorld* input_worldP = NULL,
// 	* output_worldP = NULL;

//     PF_Err err = PF_Err_NONE;
    
//     AEGP_SuiteHandler suites(in_data->pica_basicP);
//     AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);

//     LayerPack* infoP = reinterpret_cast<LayerPack*>(suites.HandleSuite1()->host_lock_handle(
//         reinterpret_cast<PF_Handle>(extra->input->pre_render_data)));

//     if (!infoP) {
//         return PF_Err_BAD_CALLBACK_PARAM;
//     }

// 	ERR(extra->cb->checkout_output(in_data->effect_ref, &output_worldP));

//     // 获取GPU信息
//     PF_GPUDeviceInfo device_info;
//     ERR(gpu_suite->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &device_info));

//     // 获取输出的内存地址
//     void* output_mem = NULL;
//     ERR(gpu_suite->GetGPUWorldData(in_data->effect_ref, output_worldP, &output_mem));

//     const A_long dst_row_bytes = output_worldP->rowbytes;

//     // 读取每一层，开始渲染
//     for (LayerInfo& layer : infoP->pack) {
//         PF_EffectWorld* layer_worldP = NULL;
//         ERR((extra->cb->checkout_layer_pixels(in_data->effect_ref, layer.idL, &layer_worldP)));

//         // 读取当前层的内存地址
//         void* layer_mem = NULL;
//         if (!err && layer_worldP) {
//             ERR(gpu_suite->GetGPUWorldData(in_data->effect_ref, layer_worldP, &layer_mem));
//         } else {
//             ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, layer.idL));
//             continue; // Skip layers that cannot be checked out
//         }
// #if HAS_CUDA
//         // CUDA 加速
//         if (!err && extra->input->what_gpu == PF_GPU_Framework_CUDA) {   
//             CompositeLayer_CUDA(
//                     (const float*)layer_mem,
//                     (float*)output_mem,
//                     layer_worldP->width,
//                     layer_worldP->height,
//                     layer_worldP->rowbytes / bytes_per_pixel,
//                     output_worldP->width,
//                     output_worldP->height,
//                     output_worldP->rowbytes / bytes_per_pixel,
//                     &layer.matrix,
//                     infoP->xfer,
//                     255,
//                     false // premultiplied alpha
//             );
//         }
// #endif
//         ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, layer.idL));
//     }
//     suites.HandleSuite1()->host_unlock_handle(reinterpret_cast<PF_Handle>(extra->input->pre_render_data));
//     return err;
// }