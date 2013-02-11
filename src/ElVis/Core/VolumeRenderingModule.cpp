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


#include <ElVis/Core/VolumeRenderingModule.h>
#include <ElVis/Core/VolumeRenderingIntegrationCategory.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/Util.hpp>
#include <ElVis/Core/Float.h>

#include <boost/timer.hpp>
#include <iostream>
#include <algorithm>
#include <boost/bind.hpp>

#include <stdio.h>

#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL

#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/png_io.hpp>

namespace ElVis
{
    VolumeRenderingModule::VolumeRenderingModule() :
        ElementTraversalModule(),
        m_segmentIntegrationType(eRiemann_SingleThreadPerRay),
        m_integrateSegmentSingleThreadPerRayRiemann(0),
        m_integrateFull(0),
        m_integrateFullSingleSegmentPerWarp(0),
        m_integrateSegmentSingleThreadPerRay(0),
        m_gkOnly(0),
        m_Trapezoidal_SingleThreadPerRay(0),
        m_mappedSegmentIndex(0),
        m_pixelCategoryBuf(0),
        m_accumulatedOpacityBuf(0),
        m_accumulatedColorBuf(0),
        m_numSamples(0),
        m_enableSampleTracking(false),
        m_clearAccumlatorBuffers(0),
        m_populateColorBuffer(0),
        m_transferFunction(new HostTransferFunction()),
        m_compositingStepSize(.01),
        m_epsilon(.001),
        m_renderIntegrationType(false),
        m_enableEmptySpaceSkipping(true),
        m_initializationComplete(false)
    {
        m_transferFunction->OnTransferFunctionChanged.connect(boost::bind(&VolumeRenderingModule::HandleTransferFunctionChanged, this));
    }
  
    void VolumeRenderingModule::DoRender(SceneView* view)
    {
        std::cout << "Volume render." << std::endl;
        try
        {
            {
                int bufSize = view->GetWidth()*view->GetHeight();

                dim3 gridDim;
                gridDim.x = bufSize/1024 + 1;
                gridDim.y = 1;
                gridDim.z = 1;

                dim3 blockDim;
                blockDim.x = 1024;
                blockDim.y = 1;
                blockDim.z = 1;
                // Clear the accumulator buffers.
                
                void* args[] = {&m_accumulatedOpacityBuf, &m_accumulatedColorBuf, &bufSize};
                CUresult r = cuLaunchKernel(m_clearAccumlatorBuffers, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, 0, args, 0);
                r = cuCtxSynchronize();
                if( r != CUDA_SUCCESS )
                {
                    //std::cout << "Error running ClearAccumulatorBuffers " << getCudaDrvErrorString(r) << std::endl;
                }
            }
            optixu::Context context = view->GetContext();


            ResetSomeSegmentsNeedToBeIntegrated();
            ResetSampleCount();
            std::cout << "Program index: " << GetInitTraversalProgram().Index << std::endl;
            context->launch(GetInitTraversalProgram().Index, view->GetWidth(), view->GetHeight());

            uint2 imageDimensions;
            imageDimensions.x = view->GetWidth();
            imageDimensions.y = view->GetHeight();

            ElVisFloat3 eye = MakeFloat3(view->GetEye());
            ElVisFloat3 U = MakeFloat3(view->GetViewSettings()->GetU());
            ElVisFloat3 V = MakeFloat3(view->GetViewSettings()->GetV());
            ElVisFloat3 W = MakeFloat3(view->GetViewSettings()->GetW());

            dim3 gridDim;
            gridDim.x = view->GetWidth()/4;
            gridDim.y = view->GetHeight();
            gridDim.z = 1;

            dim3 blockDim;
            blockDim.x = 32;
            blockDim.y = 4;
            blockDim.z = 1;

            int someSegmentsNeedToBeIntegrated = GetSomeSegmentsNeedToBeIntegrated();
            std::cout << "Some segments need to be integrated: " <<someSegmentsNeedToBeIntegrated << std::endl;
            int infiniteLoopGuard = 0;
            while(someSegmentsNeedToBeIntegrated == 1 && infiniteLoopGuard < 200)
            {
                IntegrateSegment(view);
                
                ResetSomeSegmentsNeedToBeIntegrated();
                context->launch(GetTraveralProgram().Index, view->GetWidth(), view->GetHeight());
                someSegmentsNeedToBeIntegrated = GetSomeSegmentsNeedToBeIntegrated();
                ++infiniteLoopGuard;
            }
           
            {
                int bufSize = view->GetWidth()*view->GetHeight();

                dim3 gridDim;
                gridDim.x = bufSize/1024 + 1;
                gridDim.y = 1;
                gridDim.z = 1;

                dim3 blockDim;
                blockDim.x = 1024;
                blockDim.y = 1;
                blockDim.z = 1;
                // Clear the accumulator buffers.
                
                Color c  = view->GetBackgroundColor();
                ElVisFloat3 color;
                color.x = c.Red();
                color.y = c.Green();
                color.z = c.Blue();

                CUdeviceptr colorBuffer = view->GetColorBuffer().GetMappedCudaPtr();//view->GetDeviceColorBuffer();
                void* args[] = {&m_accumulatedColorBuf, &m_accumulatedOpacityBuf, &colorBuffer, &bufSize, &color};
                CUresult r = cuLaunchKernel(m_populateColorBuffer, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, 0, args, 0);
                r = cuCtxSynchronize();
                if( r != CUDA_SUCCESS )
                {
                    //std::cout << "Error running PopulateColorBuffer " << getCudaDrvErrorString(r) << std::endl;
                }
                view->GetColorBuffer().UnmapCudaPtr();
            }

            if( m_numSamples )
            {
                int numSampleBuffer[] = {-1};
                if( m_numSamples )
                {
                    cuMemcpyDtoH(numSampleBuffer, m_numSamples, sizeof(int));
                }

                std::cout << "Total number of samples taken: " << numSampleBuffer[0] << std::endl;
            }
        }
        catch(optixu::Exception& e)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
            std::cerr << e.getErrorString() << std::endl;
            std::cout << e.getErrorString().c_str() << std::endl;
        }
//         catch(optix::CudaError& e)
//         {
//             std::cout << "Exception encountered rendering primary rays." << std::endl;
//             std::cout << e.getErrorString().c_str() << std::endl;
//         }
        catch(std::exception& e)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
            std::cout << e.what() << std::endl;
        }
        catch(...)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
        }
    }

    void VolumeRenderingModule::SetIntegrationType(VolumeRenderingIntegrationType type)
    {
        if( m_segmentIntegrationType == type ) return;

        m_segmentIntegrationType = type;
        OnModuleChanged(*this);
        OnIntegrationTypeChanged(*this, type);
        SetRenderRequired();
    }

    void VolumeRenderingModule::DoEvaluateSegment(SceneView* view)
    {
    }

    void VolumeRenderingModule::WriteAccumulatedDensityBuffer(const std::string& fileName, SceneView* view)
    {
        unsigned int numEntries = view->GetWidth()*view->GetHeight();
        uchar3* imageData = new uchar3[numEntries];
        ElVisFloat* data = new ElVisFloat[numEntries];

        cuMemcpyDtoH(data, m_accumulatedOpacityBuf, sizeof(ElVisFloat)*view->GetWidth()*view->GetHeight());

        ElVisFloat* max = std::max_element(data, data+numEntries);

        std::cout << "Max element = " << *max << std::endl;
        for(unsigned int i = 0; i < numEntries; ++i)
        {
            imageData[i].x = data[i]*255.0/(*max);
            imageData[i].y = data[i]*255.0/(*max);
            imageData[i].z = data[i]*255.0/(*max);
        }


        boost::gil::rgb8_image_t forPng(view->GetWidth(), view->GetHeight());
        boost::gil::copy_pixels( boost::gil::interleaved_view(view->GetWidth(), view->GetHeight(), (boost::gil::rgb8_pixel_t*)imageData, 3*view->GetWidth()), boost::gil::view(forPng));
        boost::gil::png_write_view(fileName + "_density.png", boost::gil::const_view(forPng));



        int numSampleBuffer[] = {-1};
        if( m_numSamples )
        {
            cuMemcpyDtoH(numSampleBuffer, m_numSamples, sizeof(int));
        }

        std::cout << "Total number of samples taken: " << numSampleBuffer[0] << std::endl;

        std::cout << "Writing density file" << std::endl;
        std::string floatingPointFileName = fileName + "_density.bin";
        FILE* outFile = fopen(floatingPointFileName.c_str(), "wb");
        unsigned int w = view->GetWidth();
        unsigned int h = view->GetHeight();
        fwrite(&w, sizeof(unsigned int), 1, outFile);
        fwrite(&h, sizeof(unsigned int), 1, outFile);
        fwrite(numSampleBuffer, sizeof(int), 1, outFile);
        fwrite(data, sizeof(ElVisFloat), view->GetWidth()*view->GetHeight(), outFile);
        fclose(outFile);

        ElVisFloat3* colorData = new ElVisFloat3[numEntries];
        cuMemcpyDtoH(colorData, m_accumulatedColorBuf, sizeof(ElVisFloat3)*view->GetWidth()*view->GetHeight());

        std::cout << "Writing color file" << std::endl;
        std::string colorFileName = fileName + "_color.bin";
        FILE* colorFile = fopen(colorFileName.c_str(), "wb");
        fwrite(&w, sizeof(unsigned int), 1, colorFile);
        fwrite(&h, sizeof(unsigned int), 1, colorFile);
        fwrite(numSampleBuffer, sizeof(int), 1, colorFile);
        fwrite(colorData, sizeof(ElVisFloat3), view->GetWidth()*view->GetHeight(), colorFile);
        fclose(colorFile);

        delete [] colorData;
        delete [] data;
        delete [] imageData;
        std::cout << "Done writing files." << std::endl;
    }


    void VolumeRenderingModule::IntegrateSegment(SceneView* view)
    {
//        cudaError_t cacheResult = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
//        if( cacheResult != cudaSuccess )
//        {
//            std::cout << "Error setting cache: " << cudaGetErrorString(cacheResult) << std::endl;
//        }
//        cudaFuncCache cacheType;

//        cudaDeviceGetCacheConfig(&cacheType);
//        std::cout << "Cache Type: " << cacheType << std::endl;

        view->GetScene()->GetModel()->MapInteropBuffersForCuda();

        switch( m_segmentIntegrationType )
        {
            case eRiemann_SingleThreadPerRay:
            {
                IntegrateSegmentSingleThreadPerRayRiemann(view);
                break;
            }

            case eTrapezoidal_SingleThreadPerRay:
            {
                IntegrateTrapezoidal_SingleThreadPerRay(view);
                break;
            }


            case eIntegrateSegment_FullAlgorithm:
            {
                IntegrateSegmentSingleThreadPerRay(view);
                break;
            }

            default:
                std::cout << "Unknown integration type." << std::endl;
        };

        view->GetScene()->GetModel()->UnMapInteropBuffersForCuda();

    }




    void VolumeRenderingModule::IntegrateTrapezoidal_SingleThreadPerRay(SceneView* view)
    {
        try
        {
            ElVisFloat3 eye = MakeFloat3(view->GetEye());
            dim3 gridDim;
            gridDim.x = static_cast<int>(ceil(static_cast<float>(view->GetWidth()/8)));
            gridDim.y = static_cast<int>(ceil(static_cast<float>(view->GetHeight()/4)));
            gridDim.z = 1;

            dim3 blockDim;
            blockDim.x = 8;
            blockDim.y = 4;
            blockDim.z = 1;


            CUdeviceptr idBuffer = GetSegmentElementIdBuffer().GetMappedCudaPtr();
            CUdeviceptr typeBuffer = GetSegmentElementTypeBuffer().GetMappedCudaPtr();
            CUdeviceptr directionBuffer = GetSegmentRayDirectionBuffer().GetMappedCudaPtr();
            CUdeviceptr segmentStartBuffer = GetSegmentStartBuffer().GetMappedCudaPtr();
            CUdeviceptr segmentEndBuffer = GetSegmentEndBuffer().GetMappedCudaPtr();
            CUdeviceptr transferFunction = m_transferFunction->GetDeviceObject();

            bool m_enableTrace = view->GetScene()->GetEnableOptixTrace();
            int m_tracex = view->GetScene()->GetOptixTracePixelIndex().x();
            int m_tracey = view->GetScene()->GetOptixTracePixelIndex().y();
            int fieldId = view->GetScalarFieldIndex();
            unsigned int screenx = view->GetWidth();
            unsigned int screeny = view->GetHeight();
            void* args[] = {&eye, &idBuffer, &typeBuffer, &directionBuffer, &segmentStartBuffer, &segmentEndBuffer,
                            &fieldId,
                            &transferFunction, &m_epsilon, &m_compositingStepSize,
                            &screenx, &screeny,
                            &m_enableTrace, &m_tracex, &m_tracey,
                            &m_numSamples,
                            &m_accumulatedOpacityBuf, &m_accumulatedColorBuf};
            cuLaunchKernel(m_Trapezoidal_SingleThreadPerRay, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, 0, args, 0);
            cuCtxSynchronize();
            GetSegmentElementIdBuffer().UnmapCudaPtr();
            GetSegmentElementTypeBuffer().UnmapCudaPtr();
            GetSegmentRayDirectionBuffer().UnmapCudaPtr();
            GetSegmentStartBuffer().UnmapCudaPtr();
            GetSegmentEndBuffer().UnmapCudaPtr();
        }
        catch(std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    void VolumeRenderingModule::IntegrateSegmentSingleThreadPerRayRiemann(SceneView* view)
    {
        try
        {
            ElVisFloat3 eye = MakeFloat3(view->GetEye());
            dim3 blockDim;
            blockDim.x = 16;
            blockDim.y = 8;
            blockDim.z = 1;

            dim3 gridDim;
            gridDim.x = view->GetWidth()/blockDim.x;
            gridDim.y = view->GetHeight()/blockDim.y;
            gridDim.z = 1;

            CUdeviceptr idBuffer = GetSegmentElementIdBuffer().GetMappedCudaPtr();
            CUdeviceptr typeBuffer = GetSegmentElementTypeBuffer().GetMappedCudaPtr();
            CUdeviceptr directionBuffer = GetSegmentRayDirectionBuffer().GetMappedCudaPtr();
            CUdeviceptr segmentStartBuffer = GetSegmentStartBuffer().GetMappedCudaPtr();
            CUdeviceptr segmentEndBuffer = GetSegmentEndBuffer().GetMappedCudaPtr();
            CUdeviceptr transferFunction = m_transferFunction->GetDeviceObject();

            bool m_enableTrace = view->GetScene()->GetEnableOptixTrace();
            int m_tracex = view->GetScene()->GetOptixTracePixelIndex().x();
            int m_tracey = view->GetScene()->GetOptixTracePixelIndex().y();
            int fieldId = view->GetScalarFieldIndex();
            unsigned int screenx = view->GetWidth();
            unsigned int screeny = view->GetHeight();
            void* args[] = {&eye, &idBuffer, &typeBuffer, &directionBuffer, &segmentStartBuffer, &segmentEndBuffer,
                            &fieldId,
                            &transferFunction, &m_epsilon, &m_compositingStepSize,
                            &screenx, &screeny,
                            &m_enableTrace, &m_tracex, &m_tracey,
                            &m_numSamples,
                            &m_accumulatedOpacityBuf, &m_accumulatedColorBuf};
            cuLaunchKernel(m_integrateSegmentSingleThreadPerRayRiemann, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, 0, args, 0);
            cuCtxSynchronize();
            GetSegmentElementIdBuffer().UnmapCudaPtr();
            GetSegmentElementTypeBuffer().UnmapCudaPtr();
            GetSegmentRayDirectionBuffer().UnmapCudaPtr();
            GetSegmentStartBuffer().UnmapCudaPtr();
            GetSegmentEndBuffer().UnmapCudaPtr();
        }
        catch(std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    void VolumeRenderingModule::IntegrateSingleThreadPerRayWithSpacing(SceneView* view, CUfunction f)
    {
        try
        {
            ElVisFloat3 eye = MakeFloat3(view->GetEye());
            dim3 gridDim;
            gridDim.x = view->GetWidth()/8;
            gridDim.y = view->GetHeight()/8;
            gridDim.z = 1;

            dim3 blockDim;
            blockDim.x = 8;
            blockDim.y = 8;
            blockDim.z = 1;


            CUdeviceptr idBuffer = GetSegmentElementIdBuffer().GetMappedCudaPtr();
            CUdeviceptr typeBuffer = GetSegmentElementTypeBuffer().GetMappedCudaPtr();
            CUdeviceptr directionBuffer = GetSegmentRayDirectionBuffer().GetMappedCudaPtr();
            CUdeviceptr segmentStartBuffer = GetSegmentStartBuffer().GetMappedCudaPtr();
            CUdeviceptr segmentEndBuffer = GetSegmentEndBuffer().GetMappedCudaPtr();
            CUdeviceptr transferFunction = m_transferFunction->GetDeviceObject();

            bool m_enableTrace = view->GetScene()->GetEnableOptixTrace();
            int m_tracex = view->GetScene()->GetOptixTracePixelIndex().x();
            int m_tracey = view->GetScene()->GetOptixTracePixelIndex().y();
            int fieldId = view->GetScalarFieldIndex();

            void* args[] = {&eye, &idBuffer, &typeBuffer, &directionBuffer, &segmentStartBuffer, &segmentEndBuffer,
                            &fieldId,
                            &m_compositingStepSize, &transferFunction,
                            &m_accumulatedOpacityBuf, &m_accumulatedColorBuf};
            cuLaunchKernel(f, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, 0, args, 0);
            cuCtxSynchronize();
            GetSegmentElementIdBuffer().UnmapCudaPtr();
            GetSegmentElementTypeBuffer().UnmapCudaPtr();
            GetSegmentRayDirectionBuffer().UnmapCudaPtr();
            GetSegmentStartBuffer().UnmapCudaPtr();
            GetSegmentEndBuffer().UnmapCudaPtr();
        }
        catch(std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }


    void VolumeRenderingModule::IntegrateSingleThreadPerRayFull(SceneView* view)
    {
        try
        {
            ElVisFloat3 eye = MakeFloat3(view->GetEye());
            dim3 gridDim;
            gridDim.x = view->GetWidth()/8;
            gridDim.y = view->GetHeight()/8;
            gridDim.z = 1;

            dim3 blockDim;
            blockDim.x = 8;
            blockDim.y = 8;
            blockDim.z = 1;


            CUdeviceptr idBuffer = GetSegmentElementIdBuffer().GetMappedCudaPtr();
            CUdeviceptr typeBuffer = GetSegmentElementTypeBuffer().GetMappedCudaPtr();
            CUdeviceptr directionBuffer = GetSegmentRayDirectionBuffer().GetMappedCudaPtr();
            CUdeviceptr segmentStartBuffer = GetSegmentStartBuffer().GetMappedCudaPtr();
            CUdeviceptr segmentEndBuffer = GetSegmentEndBuffer().GetMappedCudaPtr();
            CUdeviceptr transferFunction = m_transferFunction->GetDeviceObject();

            bool m_enableTrace = view->GetScene()->GetEnableOptixTrace();
            int m_tracex = view->GetScene()->GetOptixTracePixelIndex().x();
            int m_tracey = view->GetScene()->GetOptixTracePixelIndex().y();
            int fieldId = view->GetScalarFieldIndex();

            void* args[] = {&eye, &idBuffer, &typeBuffer, &directionBuffer, &segmentStartBuffer, &segmentEndBuffer,
                            &fieldId,
                            &transferFunction, &m_epsilon, &m_enableTrace,
                            &m_accumulatedOpacityBuf, &m_accumulatedColorBuf};
            cuLaunchKernel(m_integrateFull, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, 0, args, 0);
            cuCtxSynchronize();
            GetSegmentElementIdBuffer().UnmapCudaPtr();
            GetSegmentElementTypeBuffer().UnmapCudaPtr();
            GetSegmentRayDirectionBuffer().UnmapCudaPtr();
            GetSegmentStartBuffer().UnmapCudaPtr();
            GetSegmentEndBuffer().UnmapCudaPtr();
        }
        catch(std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    void VolumeRenderingModule::IntegrateSingleWarpPerSegmentFull(SceneView* view)
    {
        try
        {
            ElVisFloat3 eye = MakeFloat3(view->GetEye());
            dim3 gridDim;
            gridDim.x = view->GetWidth();
            gridDim.y = view->GetHeight();
            gridDim.z = 1;

            dim3 blockDim;
            blockDim.x = 32;
            blockDim.y = 1;
            blockDim.z = 1;


            CUdeviceptr idBuffer = GetSegmentElementIdBuffer().GetMappedCudaPtr();
            CUdeviceptr typeBuffer = GetSegmentElementTypeBuffer().GetMappedCudaPtr();
            CUdeviceptr directionBuffer = GetSegmentRayDirectionBuffer().GetMappedCudaPtr();
            CUdeviceptr segmentStartBuffer = GetSegmentStartBuffer().GetMappedCudaPtr();
            CUdeviceptr segmentEndBuffer = GetSegmentEndBuffer().GetMappedCudaPtr();
            CUdeviceptr transferFunction = m_transferFunction->GetDeviceObject();

            bool m_enableTrace = view->GetScene()->GetEnableOptixTrace();
            int m_tracex = view->GetScene()->GetOptixTracePixelIndex().x();
            int m_tracey = view->GetScene()->GetOptixTracePixelIndex().y();
            int fieldId = view->GetScalarFieldIndex();

            void* args[] = {&eye, &idBuffer, &typeBuffer, &directionBuffer, &segmentStartBuffer, &segmentEndBuffer,
                            &fieldId,
                            &transferFunction, &m_epsilon, &m_enableTrace,
                            &m_accumulatedOpacityBuf, &m_accumulatedColorBuf};
            cuLaunchKernel(m_integrateFullSingleSegmentPerWarp, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, 0, args, 0);
            cuCtxSynchronize();
            GetSegmentElementIdBuffer().UnmapCudaPtr();
            GetSegmentElementTypeBuffer().UnmapCudaPtr();
            GetSegmentRayDirectionBuffer().UnmapCudaPtr();
            GetSegmentStartBuffer().UnmapCudaPtr();
            GetSegmentEndBuffer().UnmapCudaPtr();
        }
        catch(std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    void VolumeRenderingModule::IntegrateGKOnly(SceneView* view)
    {
        try
        {
            ElVisFloat3 eye = MakeFloat3(view->GetEye());
            dim3 gridDim;
            gridDim.x = view->GetWidth()/8;
            gridDim.y = view->GetHeight()/4;
            gridDim.z = 1;

            dim3 blockDim;
            blockDim.x = 8;
            blockDim.y = 4;
            blockDim.z = 1;


            CUdeviceptr idBuffer = GetSegmentElementIdBuffer().GetMappedCudaPtr();
            CUdeviceptr typeBuffer = GetSegmentElementTypeBuffer().GetMappedCudaPtr();
            CUdeviceptr directionBuffer = GetSegmentRayDirectionBuffer().GetMappedCudaPtr();
            CUdeviceptr segmentStartBuffer = GetSegmentStartBuffer().GetMappedCudaPtr();
            CUdeviceptr segmentEndBuffer = GetSegmentEndBuffer().GetMappedCudaPtr();
            CUdeviceptr transferFunction = m_transferFunction->GetDeviceObject();

            bool m_enableTrace = view->GetScene()->GetEnableOptixTrace();
            int m_tracex = view->GetScene()->GetOptixTracePixelIndex().x();
            int m_tracey = view->GetScene()->GetOptixTracePixelIndex().y();
            int fieldId = view->GetScalarFieldIndex();

            void* args[] = {&eye, &idBuffer, &typeBuffer, &directionBuffer, &segmentStartBuffer, &segmentEndBuffer,
                            &fieldId,
                            &transferFunction, &m_epsilon, &m_compositingStepSize, &m_enableTrace, &m_tracex, &m_tracey,
                            &m_numSamples,
                            &m_renderIntegrationType,
                            &m_enableEmptySpaceSkipping,
                            &m_accumulatedOpacityBuf, &m_accumulatedColorBuf};
            cuLaunchKernel(m_gkOnly, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, 0, args, 0);
            cuCtxSynchronize();
            GetSegmentElementIdBuffer().UnmapCudaPtr();
            GetSegmentElementTypeBuffer().UnmapCudaPtr();
            GetSegmentRayDirectionBuffer().UnmapCudaPtr();
            GetSegmentStartBuffer().UnmapCudaPtr();
            GetSegmentEndBuffer().UnmapCudaPtr();
        }
        catch(std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    void VolumeRenderingModule::IntegrateSegmentSingleThreadPerRay(SceneView* view)
    {
        try
        {
            ElVisFloat3 eye = MakeFloat3(view->GetEye());
            dim3 blockDim;
            blockDim.x = 16;
            blockDim.y = 8;
            blockDim.z = 1;

            dim3 gridDim;
            gridDim.x = view->GetWidth()/blockDim.x;
            gridDim.y = view->GetHeight()/blockDim.y;
            gridDim.z = 1;




            CUdeviceptr idBuffer = GetSegmentElementIdBuffer().GetMappedCudaPtr();
            CUdeviceptr typeBuffer = GetSegmentElementTypeBuffer().GetMappedCudaPtr();
            CUdeviceptr directionBuffer = GetSegmentRayDirectionBuffer().GetMappedCudaPtr();
            CUdeviceptr segmentStartBuffer = GetSegmentStartBuffer().GetMappedCudaPtr();
            CUdeviceptr segmentEndBuffer = GetSegmentEndBuffer().GetMappedCudaPtr();
            CUdeviceptr transferFunction = m_transferFunction->GetDeviceObject();

            bool m_enableTrace = view->GetScene()->GetEnableOptixTrace();
            int m_tracex = view->GetScene()->GetOptixTracePixelIndex().x();
            int m_tracey = view->GetScene()->GetOptixTracePixelIndex().y();
            int fieldId = view->GetScalarFieldIndex();

            void* args[] = {&eye, &idBuffer, &typeBuffer, &directionBuffer, &segmentStartBuffer, &segmentEndBuffer,
                            &fieldId,
                            &transferFunction, &m_epsilon, &m_compositingStepSize, &m_enableTrace, &m_tracex, &m_tracey,
                            &m_numSamples,
                            &m_renderIntegrationType,
                            &m_enableEmptySpaceSkipping,
                            &m_accumulatedOpacityBuf, &m_accumulatedColorBuf};
            cuLaunchKernel(m_integrateSegmentSingleThreadPerRay, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, 0, args, 0);
            cuCtxSynchronize();
            GetSegmentElementIdBuffer().UnmapCudaPtr();
            GetSegmentElementTypeBuffer().UnmapCudaPtr();
            GetSegmentRayDirectionBuffer().UnmapCudaPtr();
            GetSegmentStartBuffer().UnmapCudaPtr();
            GetSegmentEndBuffer().UnmapCudaPtr();
        }
        catch(std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    void VolumeRenderingModule::IntegrateDensity_AdaptiveTrapezoidal_SingleBlockPerRay(SceneView* view)
    {
        // Primarily a test of the warp-based adaptive quadrature approach
        try
        {
            //
        }
        catch(std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    void VolumeRenderingModule::IntegrateDensity_NonAdaptiveTrapezoidal_SingleBlockPerRay(SceneView* view)
    {
        // Global adaptive, where the error estimate
        try
        {
            //
        }
        catch(std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }


    void VolumeRenderingModule::DoSetupAfterInteropModule(SceneView* view)
    {
        optixu::Context context = view->GetContext();
        CUmodule module = view->GetScene()->GetCudaModule();
        m_transferFunction->GetDeviceObject();

        if( !m_initializationComplete )
        {
            cuModuleGetFunction(&m_clearAccumlatorBuffers, module, "ClearAccumulatorBuffers");
            cuModuleGetFunction(&m_populateColorBuffer, module, "PopulateColorBuffer");
            cuModuleGetFunction(&m_integrateSegmentSingleThreadPerRayRiemann, module, "IntegrateSegmentSingleThreadPerRayRiemann");
            cuModuleGetFunction(&m_integrateFull, module, "IntegrateSegmentSingleThreadPerRayFullVersion");
            cuModuleGetFunction(&m_integrateFullSingleSegmentPerWarp, module, "IntegrateSegmentWarpPerSegment");
            cuModuleGetFunction(&m_integrateSegmentSingleThreadPerRay, module, "IntegrateSegmentSingleThreadPerRay");
            cuModuleGetFunction(&m_gkOnly, module, "GKOnly");
            cuModuleGetFunction(&m_Trapezoidal_SingleThreadPerRay, module, "Trapezoidal_SingleThreadPerRay");

            cuMemAlloc(&m_mappedSegmentIndex, sizeof(int)*view->GetWidth()*view->GetHeight());
            cuMemAlloc(&m_pixelCategoryBuf, sizeof(VolumeRenderingIntegrationCategory)*view->GetWidth()*view->GetHeight());
            cuMemAlloc(&m_accumulatedOpacityBuf, sizeof(ElVisFloat)*view->GetWidth()*view->GetHeight());
            cuMemAlloc(&m_accumulatedColorBuf, sizeof(ElVisFloat3)*view->GetWidth()*view->GetHeight());        

            m_initializationComplete = true;
        }
    }

    void VolumeRenderingModule::SetTrackNumberOfSamples(bool value)
    {
        m_enableSampleTracking = value;
        ResetSampleCount();
        SetRenderRequired();
        OnModuleChanged(*this);
    }

    void VolumeRenderingModule::ResetSampleCount()
    {
        if( m_enableSampleTracking )
        {
            if( !m_numSamples )
            {
                cuMemAlloc(&m_numSamples, sizeof(int));
            }
            int data[] = {0};
            cuMemcpyHtoD(m_numSamples, &data[0], sizeof(int));
        }
        else
        {
            if( m_numSamples )
            {
                cuMemFree(m_numSamples);
                m_numSamples = 0;
            }
        }

    }

    void VolumeRenderingModule::SetCompositingStepSize(ElVisFloat value)
    {
        if( value > 0.0 && value != m_compositingStepSize)
        {
            m_compositingStepSize = value;
            SetRenderRequired();
            OnModuleChanged(*this);
        }
    }

    void VolumeRenderingModule::SetEpsilon(ElVisFloat value)
    {
        if( value > 0.0 && value != m_epsilon )
        {
            m_epsilon = value;
            SetRenderRequired();
            OnModuleChanged(*this);
        }
    }

    void VolumeRenderingModule::HandleTransferFunctionChanged()
    {
        SetSyncAndRenderRequired();
        OnModuleChanged(*this);
    }
}

