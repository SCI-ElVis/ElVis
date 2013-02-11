///////////////////////////////////////////////////////////////////////////////
//
// The MIT License
//
// Copyright (c) 2006 Scientific Computing and Imaging Institute,
// University of Utah (USA)
//
// License for the specific language governing rights and limitations under
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
///////////////////////////////////////////////////////////////////////////////

#include "Util.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <ElVis/Core/Float.cu>
#include <stdio.h>

namespace ElVis
{
    ElVisFloat3 MakeFloat3(const WorldPoint& p)
    {
        return ::MakeFloat3((ElVisFloat)p.x(), (ElVisFloat)p.y(), (ElVisFloat)p.z());
    }

    ElVisFloat3 MakeFloat3(const WorldVector& v)
    {
        return ::MakeFloat3((ElVisFloat)v.x(), (ElVisFloat)v.y(), (ElVisFloat)v.z());
    }

    ElVisFloat4 MakeFloat4(const WorldPoint& p)
    {
        return ::MakeFloat4((ElVisFloat)p.x(), (ElVisFloat)p.y(), (ElVisFloat)p.z(), 1.0);
    }

    ElVisFloat4 MakeFloat4(const WorldVector& v)
    {
        return ::MakeFloat4((ElVisFloat)v.x(), (ElVisFloat)v.y(), (ElVisFloat)v.z(), 0.0);
    }



    unsigned int GetFreeCudaMemory()
    {
        size_t free, total;
        int gpuCount, i;
        CUresult res;
        CUdevice dev;
        CUcontext ctx;

        gpuCount = 0;

        cuInit(0);
        //CheckCudaError("");

        cuDeviceGetCount(&gpuCount);
        //CheckCudaError("");

        printf("Detected %d GPU\n",gpuCount);

        for (i=0; i<gpuCount; i++)
        {
            cuDeviceGet(&dev,i);
            cuCtxCreate(&ctx, 0, dev);
            res = cuMemGetInfo(&free, &total);
            if(res != CUDA_SUCCESS)
                printf("!!!! cuMemGetInfo failed! (status = %x)", res);
            printf("^^^^ Device: %d\n",i);
            printf("%zu Total Memory.\n", total);
            printf("%zu Memory Available.\n", free);
            cuCtxDetach(ctx);
        }
        return 0;
    }

	void Min(WorldPoint& lhs, const WorldPoint& rhs)
    {
        for(int i = 0; i < 3; ++i)
        {
            if( rhs[i] < lhs[i] )
            {
                lhs.SetValue(i, rhs[i]);
            }
        }
    }

    void Max(WorldPoint& lhs, const WorldPoint& rhs)
    {
        for(int i = 0; i < 3; ++i)
        {
            if( rhs[i] > lhs[i] )
            {
                lhs.SetValue(i, rhs[i]);
            }
        }
    }

    void DeviceProperties()
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        //CheckCudaError("");
        if (deviceCount == 0)
            printf("There is no device supporting CUDA\n");
        int dev;
        for (dev = 0; dev < deviceCount; ++dev) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, dev);
            if (dev == 0) {
                if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                    printf("There is no device supporting CUDA.\n");
                else if (deviceCount == 1)
                    printf("There is 1 device supporting CUDA\n");
                else
                    printf("There are %d devices supporting CUDA\n", deviceCount);
            }
            printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
            printf("  Major revision number:                         %d\n",
                   deviceProp.major);
            printf("  Minor revision number:                         %d\n",
                   deviceProp.minor);
            printf("  Total amount of global memory:                 %zu bytes\n",
                   deviceProp.totalGlobalMem);
        #if CUDART_VERSION >= 2000
            printf("  Number of multiprocessors:                     %d\n",
                   deviceProp.multiProcessorCount);
            printf("  Number of cores:                               %d\n",
                   8 * deviceProp.multiProcessorCount);
        #endif
            printf("  Total amount of constant memory:               %zu bytes\n",
                   deviceProp.totalConstMem);
            printf("  Total amount of shared memory per block:       %zu bytes\n",
                   deviceProp.sharedMemPerBlock);
            printf("  Total number of registers available per block: %d\n",
                   deviceProp.regsPerBlock);
            printf("  Warp size:                                     %d\n",
                   deviceProp.warpSize);
            printf("  Maximum number of threads per block:           %d\n",
                   deviceProp.maxThreadsPerBlock);
            printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
                   deviceProp.maxThreadsDim[0],
                   deviceProp.maxThreadsDim[1],
                   deviceProp.maxThreadsDim[2]);
            printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
                   deviceProp.maxGridSize[0],
                   deviceProp.maxGridSize[1],
                   deviceProp.maxGridSize[2]);
            printf("  Maximum memory pitch:                          %zu bytes\n",
                   deviceProp.memPitch);
            printf("  Texture alignment:                             %zu bytes\n",
                   deviceProp.textureAlignment);
            printf("  Clock rate:                                    %.2f GHz\n",
                   deviceProp.clockRate * 1e-6f);
            printf("  Timeout Enabled:                               %s\n",
                   deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        #if CUDART_VERSION >= 2000
            printf("  Concurrent copy and execution:                 %s\n",
                   deviceProp.deviceOverlap ? "Yes" : "No");
        #endif
        }
        printf("\nTest PASSED\n");
    }
}
