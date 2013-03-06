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

#include <ElVis/Core/ElementTraversalModule.h>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/Cuda.h>

#include <boost/bind.hpp>

namespace ElVis
{
    RayGeneratorProgram ElementTraversalModule::m_ElementByElementVolumeTraversalInitProgram;
    RayGeneratorProgram ElementTraversalModule::m_ElementByElementVolumeTraversalProgram;
    bool ElementTraversalModule::Initialized = ElementTraversalModule::InitializeStatic();

    InteropBuffer<ElementId> ElementTraversalModule::m_IdSortBuffer("IdSortBuffer");
    InteropBuffer<int> ElementTraversalModule::m_SegmentIdBuffer("SegmentIdBuffer");
    InteropBuffer<int> ElementTraversalModule::m_SegmentElementIdBuffer("SegmentElementIdBuffer");
    InteropBuffer<int> ElementTraversalModule::m_SegmentElementTypeBuffer("SegmentElementTypeBuffer");
    InteropBuffer<ElVisFloat> ElementTraversalModule::m_SegmentStart("SegmentStart");
    InteropBuffer<ElVisFloat> ElementTraversalModule::m_SegmentEnd("SegmentEnd");
    InteropBuffer<ElVisFloat3> ElementTraversalModule::m_SegmentRayDirection("SegmentRayDirection");
    CUfunction ElementTraversalModule::m_copyElementIdKeyData;

    optixu::Buffer ElementTraversalModule::m_SomeSegmentsNeedToBeIntegrated;

    ElementTraversalModule::ElementTraversalModule() :
        RenderModule()
    {
    }


    void ElementTraversalModule::DoSetup(SceneView* view)
    {
        std::cout << "ElementTraversalModule::DoSetup." << std::endl;
        optixu::Context context = view->GetContext();
        CUmodule module = view->GetScene()->GetCudaModule();

        if( m_ElementByElementVolumeTraversalInitProgram.Index == -1 )
        {
            m_ElementByElementVolumeTraversalInitProgram = view->AddRayGenerationProgram("ElementByElementVolumeTraversalInit");
        }

        if( m_ElementByElementVolumeTraversalProgram.Index == -1 )
        {
            m_ElementByElementVolumeTraversalProgram = view->AddRayGenerationProgram("ElementByElementVolumeTraversal");
        }

        if( !m_SegmentElementIdBuffer.Initialized() )
        {
            m_SegmentElementIdBuffer.SetContextInfo(context, module);
            m_SegmentElementTypeBuffer.SetContextInfo(context, module);
            m_SegmentStart.SetContextInfo(context, module);
            m_SegmentEnd.SetContextInfo(context, module);
            m_SegmentRayDirection.SetContextInfo(context, module);
            m_SegmentIdBuffer.SetContextInfo(context, module);
            m_IdSortBuffer.SetContextInfo(context, module);

            m_SegmentElementIdBuffer.SetDimensions(view->GetWidth()*view->GetHeight());
            m_SegmentElementTypeBuffer.SetDimensions(view->GetWidth()*view->GetHeight());
            m_SegmentStart.SetDimensions(view->GetWidth()*view->GetHeight());
            m_SegmentEnd.SetDimensions(view->GetWidth()*view->GetHeight());
            m_SegmentRayDirection.SetDimensions(view->GetWidth()*view->GetHeight());
            m_SegmentIdBuffer.SetDimensions(view->GetWidth()*view->GetHeight());
            m_IdSortBuffer.SetDimensions(view->GetWidth()*view->GetHeight());
            m_SomeSegmentsNeedToBeIntegrated = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, 1);
            context["SomeSegmentsNeedToBeIntegrated"]->set(m_SomeSegmentsNeedToBeIntegrated);

            checkedCudaCall(cuModuleGetFunction(&m_copyElementIdKeyData, module, "CopyToElementId"));
        }

        DoSetupAfterInteropModule(view);
    }

    void ElementTraversalModule::DoRender(SceneView* view)
    {
        if( !HasWork() ) return;

        try
        {
            //std::cout << "Element Traversal." << std::endl;
            optixu::Context context = view->GetContext();

            ResetSomeSegmentsNeedToBeIntegrated();
            //std::cout << "Program index: " << GetInitTraversalProgram().Index << std::endl;
            context->launch(GetInitTraversalProgram().Index, view->GetWidth(), view->GetHeight());

            int someSegmentsNeedToBeIntegrated = GetSomeSegmentsNeedToBeIntegrated();
            //std::cout << "Some segments need to be integrated: " <<someSegmentsNeedToBeIntegrated << std::endl;
            int infiniteLoopGuard = 0;
            while(someSegmentsNeedToBeIntegrated == 1 && infiniteLoopGuard < 200)
            {
                DoEvaluateSegment(view);
                ResetSomeSegmentsNeedToBeIntegrated();
                context->launch(GetTraveralProgram().Index, view->GetWidth(), view->GetHeight());
                someSegmentsNeedToBeIntegrated = GetSomeSegmentsNeedToBeIntegrated();
                ++infiniteLoopGuard;
            }

        }
        catch(optixu::Exception& e)
        {
            std::cout << "Exception encountered rendering isosurface." << std::endl;
            std::cerr << e.getErrorString() << std::endl;
            std::cout << e.getErrorString().c_str() << std::endl;
        }
        catch(std::exception& e)
        {
            std::cout << "Exception encountered rendering isosurface." << std::endl;
            std::cout << e.what() << std::endl;
        }
        catch(...)
        {
            std::cout << "Exception encountered rendering isosurface." << std::endl;
        }
    }



    void ElementTraversalModule::RunCopyElementIdKeyData(SceneView* view)
    {
        try
        {
            dim3 gridDim;
            gridDim.x = view->GetWidth()*view->GetHeight()/512 + 1;
            gridDim.y = 1;
            gridDim.z = 1;

            dim3 blockDim;
            blockDim.x = 512;
            blockDim.y = 1;
            blockDim.z = 1;

            CUdeviceptr keyBuffer = m_IdSortBuffer.GetMappedCudaPtr();
            CUdeviceptr idBuffer = GetSegmentElementIdBuffer().GetMappedCudaPtr();
            CUdeviceptr typeBuffer = GetSegmentElementTypeBuffer().GetMappedCudaPtr();

            bool enableTrace = view->GetScene()->GetEnableOptixTrace();
            int tracex = view->GetScene()->GetOptixTracePixelIndex().x();
            int tracey = view->GetHeight() - view->GetScene()->GetOptixTracePixelIndex().y() - 1;
            int n = view->GetWidth() * view->GetHeight();

            void* args[] = {&idBuffer, &typeBuffer, &keyBuffer,
                            &enableTrace, &tracex, &tracey, &n};
            checkedCudaCall(cuLaunchKernel(m_copyElementIdKeyData, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, 0, args, 0));
            checkedCudaCall(cuCtxSynchronize());
            GetSegmentElementIdBuffer().UnmapCudaPtr();
            GetSegmentElementTypeBuffer().UnmapCudaPtr();
            m_IdSortBuffer.UnmapCudaPtr();
        }
        catch(std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    int ElementTraversalModule::GetSomeSegmentsNeedToBeIntegrated()
    {
        int* data = static_cast<int*>(GetSegmentsNeedToBeIntegratedBuffer()->map());
        int result = data[0];
        GetSegmentsNeedToBeIntegratedBuffer()->unmap();
        return result;
    }

    void ElementTraversalModule::ResetSomeSegmentsNeedToBeIntegrated()
    {
        int* data = static_cast<int*>(GetSegmentsNeedToBeIntegratedBuffer()->map());
        data[0] = 0;
        GetSegmentsNeedToBeIntegratedBuffer()->unmap();
    }

    bool ElementTraversalModule::InitializeStatic()
    {
        PtxManager::GetOnPtxLoaded().connect(boost::bind(&ElementTraversalModule::LoadPrograms, _1, _2));
        return true;
    }

    void ElementTraversalModule::LoadPrograms(const std::string& prefix, optixu::Context context)
    {
//        m_ElementByElementVolumeTraversalInitProgram.Program = PtxManager::LoadProgram(prefix, "ElementByElementVolumeTraversalInit");
//        m_ElementByElementVolumeTraversalProgram.Program = PtxManager::LoadProgram(prefix, "ElementByElementVolumeTraversal");
    }

}

