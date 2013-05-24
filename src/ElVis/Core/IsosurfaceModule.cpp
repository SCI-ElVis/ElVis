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


#include <ElVis/Core/IsosurfaceModule.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/Util.hpp>

#include <fstream>

#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

namespace ElVis
{
  RayGeneratorProgram IsosurfaceModule::m_ElementByElementVolumeTraversalInitProgramForIsosurface;
  RayGeneratorProgram IsosurfaceModule::m_ElementByElementVolumeTraversalProgramForIsosurface;
  RayGeneratorProgram IsosurfaceModule::m_FindIsosurface;

  bool IsosurfaceModule::Initialized = IsosurfaceModule::InitializeStaticForIsosuface();

  bool IsosurfaceModule::InitializeStaticForIsosuface()
  {
    return true;
  }

  IsosurfaceModule::IsosurfaceModule() :
  ElementTraversalModule(),
    OnIsovalueAdded(),
    OnIsovalueChanged(),
    m_isovalues(),
    m_isovalueBufferSize(),
    m_gaussLegendreNodesBuffer("Nodes"),
    m_gaussLegendreWeightsBuffer("Weights"),
    m_monomialConversionTableBuffer("MonomialConversionTable"),
    m_isovalueBuffer("SurfaceIsovalues"),
    m_findIsosurfaceFunction(0)
  {
  }

  void IsosurfaceModule::AddIsovalue(const ElVisFloat& value)
  {
    if( m_isovalues.find(value) == m_isovalues.end() )
    {
      m_isovalues.insert(value);
      SetSyncAndRenderRequired();
      OnIsovalueAdded(value);
      OnModuleChanged(*this);
    }
  }

  void IsosurfaceModule::RemoveIsovalue(const ElVisFloat& value)
  {
    std::set<ElVisFloat>::iterator found = m_isovalues.find(value);
    if( found != m_isovalues.end() )
    {
      m_isovalues.erase(found);
      SetSyncAndRenderRequired();
      OnIsovalueRemoved(value);
      OnModuleChanged(*this);
    }
  }

  void IsosurfaceModule::DoRender(SceneView* view)
  {
    if( !HasWork() ) return;

    try
    {
      std::cout << "Isosurface Element Traversal." << std::endl;
      optixu::Context context = view->GetContext();

      context->launch(m_FindIsosurface.Index, view->GetWidth(), view->GetHeight());

      //ResetSomeSegmentsNeedToBeIntegrated();
      ////std::cout << "Program index: " << GetInitTraversalProgram().Index << std::endl;
      //context->launch(m_ElementByElementVolumeTraversalInitProgramForIsosurface.Index, view->GetWidth(), view->GetHeight());

      //int someSegmentsNeedToBeIntegrated = GetSomeSegmentsNeedToBeIntegrated();
      ////std::cout << "Some segments need to be integrated: " <<someSegmentsNeedToBeIntegrated << std::endl;
      //int infiniteLoopGuard = 0;
      //while(someSegmentsNeedToBeIntegrated == 1 && infiniteLoopGuard < 200)
      //{
      //  ResetSomeSegmentsNeedToBeIntegrated();
      //  context->launch(m_ElementByElementVolumeTraversalProgramForIsosurface.Index, view->GetWidth(), view->GetHeight());
      //  someSegmentsNeedToBeIntegrated = GetSomeSegmentsNeedToBeIntegrated();
      //  ++infiniteLoopGuard;
      //}

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

  void IsosurfaceModule::DoEvaluateSegment(SceneView* view)
  {
    //        try
    //        {
    //
    //
    ////            // Sort data first.
    ////            {
    ////                RunCopyElementIdKeyData(view);
    ////                CUdeviceptr keyBuffer = GetElementSortBuffer().GetMappedCudaPtr();
    ////                CUdeviceptr idBuffer = GetSegmentIdBuffer().GetMappedCudaPtr();
    //
    ////                thrust::device_ptr<ElementId> keyPtr = thrust::device_ptr<ElementId>(reinterpret_cast<ElementId*>(keyBuffer));
    ////                thrust::device_ptr<int> valuePtr = thrust::device_ptr<int>(reinterpret_cast<int*>(idBuffer));
    ////                int n = view->GetWidth() * view->GetHeight();
    ////                SortByElementIdAndType(keyPtr, valuePtr, n);
    ////                GetElementSortBuffer().UnmapCudaPtr();
    ////                GetSegmentIdBuffer().UnmapCudaPtr();
    ////            }
    //
    //
    //            if( m_isovalues.size() == 0 )
    //            {
    //                return;
    //            }
    //
    //            ElVisFloat3 eye = MakeFloat3(view->GetEye());
    //            dim3 gridDim;
    //            gridDim.x = view->GetWidth()/8+1;
    //            gridDim.y = view->GetHeight()/4+1;
    //            gridDim.z = 1;
    //
    //            dim3 blockDim;
    //            blockDim.x = 8;
    //            blockDim.y = 4;
    //            blockDim.z = 1;
    //
    //
    //            view->GetScene()->GetModel()->MapInteropBuffersForCuda();
    //            CUdeviceptr idBuffer = GetSegmentElementIdBuffer().GetMappedCudaPtr();
    //            CUdeviceptr typeBuffer = GetSegmentElementTypeBuffer().GetMappedCudaPtr();
    //            CUdeviceptr directionBuffer = GetSegmentRayDirectionBuffer().GetMappedCudaPtr();
    //            CUdeviceptr segmentStartBuffer = GetSegmentStartBuffer().GetMappedCudaPtr();
    //            CUdeviceptr segmentEndBuffer = GetSegmentEndBuffer().GetMappedCudaPtr();
    //            CUdeviceptr sampleBuffer = view->GetSampleBuffer().GetMappedCudaPtr();
    //            CUdeviceptr intersectionBuffer = view->GetIntersectionPointBuffer().GetMappedCudaPtr();
    //            CUdeviceptr segmentIdBuffer = GetSegmentIdBuffer().GetMappedCudaPtr();
    //
    //
    //            bool enableTrace = view->GetScene()->GetEnableOptixTrace();
    //            int tracex = view->GetScene()->GetOptixTracePixelIndex().x();
    //            int tracey = view->GetHeight() - view->GetScene()->GetOptixTracePixelIndex().y() - 1;
    //            int fieldId = view->GetScalarFieldIndex();
    //            int numIsosurfaces = m_isovalues.size();
    //            int screen_x = view->GetWidth();
    //            int screen_y = view->GetHeight();
    //
    //            void* args[] = {&eye, &idBuffer, &typeBuffer, &segmentIdBuffer,
    //                            &directionBuffer, &segmentStartBuffer, &segmentEndBuffer, &fieldId,
    //                            &numIsosurfaces, &m_isovalueBuffer,
    //                            &enableTrace, &tracex, &tracey,
    //                            &screen_x, &screen_y,
    //                            &m_gaussLegendreNodesBuffer, &m_gaussLegendreWeightsBuffer, &m_monomialConversionTableBuffer, &sampleBuffer, &intersectionBuffer};
    //            checkedCudaCall(cuLaunchKernel(m_findIsosurfaceFunction, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, 0, args, 0));
    //            checkedCudaCall(cuCtxSynchronize());
    //            GetSegmentElementIdBuffer().UnmapCudaPtr();
    //            GetSegmentElementTypeBuffer().UnmapCudaPtr();
    //            GetSegmentRayDirectionBuffer().UnmapCudaPtr();
    //            GetSegmentStartBuffer().UnmapCudaPtr();
    //            GetSegmentEndBuffer().UnmapCudaPtr();
    //            view->GetSampleBuffer().UnmapCudaPtr();
    //            view->GetIntersectionPointBuffer().UnmapCudaPtr();
    //            GetSegmentIdBuffer().UnmapCudaPtr();
    //            view->GetScene()->GetModel()->UnMapInteropBuffersForCuda();
    //        }
    //        catch(std::exception& e)
    //        {
    //            std::cerr << e.what() << std::endl;
    //        }

  }


  void IsosurfaceModule::DoResize(unsigned int newWidth, unsigned int newHeight)
  {
  }

  void IsosurfaceModule::DoSetupAfterInteropModule(SceneView* view)
  {
    optixu::Context context = view->GetContext();
    CUmodule module = view->GetScene()->GetCudaModule();
    try
    {
      std::cout << "Isourface setup." << std::endl;
      optixu::Context context = view->GetContext();
      CUmodule module = view->GetScene()->GetCudaModule();

      if( m_ElementByElementVolumeTraversalInitProgramForIsosurface.Index == -1 )
      {
        //m_ElementByElementVolumeTraversalInitProgramForIsosurface = view->AddRayGenerationProgram("ElementByElementVolumeTraversalInitForIsosurface");
      }

      if( m_ElementByElementVolumeTraversalProgramForIsosurface.Index == -1 )
      {
        //m_ElementByElementVolumeTraversalProgramForIsosurface = view->AddRayGenerationProgram("ElementByElementVolumeTraversalForIsosurface");
      }

      if( m_FindIsosurface.Index == -1 )
      {
        m_FindIsosurface = view->AddRayGenerationProgram("FindIsosurface");
      }

      //if( !m_findIsosurfaceFunction )
      //{
      //  checkedCudaCall(cuModuleGetFunction(&m_findIsosurfaceFunction, module, "FindIsosurfaceInSegment"));
      //}



      if( !m_gaussLegendreNodesBuffer.Initialized() )
      {
        std::vector<ElVisFloat> nodes;
        ReadFloatVector("Nodes.txt", nodes);
        m_gaussLegendreNodesBuffer.SetContextInfo(context, module);
        m_gaussLegendreNodesBuffer.SetDimensions(nodes.size());
        ElVisFloat* nodeData = m_gaussLegendreNodesBuffer.MapOptiXPointer();
        std::copy(nodes.begin(), nodes.end(), nodeData);
        m_gaussLegendreNodesBuffer.UnmapOptiXPointer();
      }

      if( !m_gaussLegendreWeightsBuffer.Initialized() )
      {
        std::vector<ElVisFloat> weights;
        ReadFloatVector("Weights.txt", weights);
        m_gaussLegendreWeightsBuffer.SetContextInfo(context, module);
        m_gaussLegendreWeightsBuffer.SetDimensions(weights.size());
        ElVisFloat* data = m_gaussLegendreWeightsBuffer.MapOptiXPointer();
        std::copy(weights.begin(), weights.end(), data);
        m_gaussLegendreWeightsBuffer.UnmapOptiXPointer();
      }

      if( !m_monomialConversionTableBuffer.Initialized() )
      {
        std::vector<ElVisFloat> monomialCoversionData;
        ReadFloatVector("MonomialConversionTables.txt", monomialCoversionData);
        m_monomialConversionTableBuffer.SetContextInfo(context, module);
        m_monomialConversionTableBuffer.SetDimensions(monomialCoversionData.size());
        ElVisFloat* data = m_monomialConversionTableBuffer.MapOptiXPointer();
        std::copy(monomialCoversionData.begin(), monomialCoversionData.end(), data);
        m_monomialConversionTableBuffer.UnmapOptiXPointer();
      }


      SynchronizeWithOptix();
      DoSynchronize(view);
    }
    catch(optixu::Exception& e)
    {
      std::cout << "Exception encountered setting up isosurface." << std::endl;
      std::cerr << e.getErrorString() << std::endl;
      std::cout << e.getErrorString().c_str() << std::endl;
    }
    catch(std::exception& e)
    {
      std::cout << "Exception encountered setting up isosurface." << std::endl;
      std::cout << e.what() << std::endl;
    }
    catch(...)
    {
      std::cout << "Exception encountered setting up isosurface." << std::endl;
    }
  }

  void IsosurfaceModule::DoSynchronize(SceneView* view)
  {
    optixu::Context context = view->GetContext();
    CUmodule module = view->GetScene()->GetCudaModule();
    std::cout << "Isosurface Module Sync." << std::endl;
    if( m_isovalues.size() > 0 )
    {
      std::cout << "Setting Isovalue buffer." << std::endl;
      if( !m_isovalueBuffer.Initialized() )
      {
        m_isovalueBuffer.SetContextInfo(context, module);
        m_isovalueBuffer.SetDimensions(m_isovalues.size());
        m_isovalueBufferSize = m_isovalues.size();
      }

      if( m_isovalueBufferSize != m_isovalues.size())
      {
        m_isovalueBuffer.SetContextInfo(context, module);
        m_isovalueBuffer.SetDimensions(m_isovalues.size());
        m_isovalueBufferSize = m_isovalues.size();
      }

      ElVisFloat* isovalueData = m_isovalueBuffer.MapOptiXPointer();
      std::copy(m_isovalues.begin(), m_isovalues.end(), isovalueData);
      m_isovalueBuffer.UnmapOptiXPointer();
    }
  }

  void IsosurfaceModule::SynchronizeWithOptix()
  {

    //        RTsize curSize = 0;
    //        m_isovalueBuffer->getSize(curSize);
    //        if( curSize >= m_isovalues.size() )
    //        {
    //            m_isovalueBuffer->setSize(m_isovalues.size());
    //        }

    //        ElVisFloat* data = static_cast<ElVisFloat*>(m_isovalueBuffer->map());
    //        BOOST_FOREACH(ElVisFloat isovalue, m_isovalues)
    //        {
    //            *data = isovalue;
    //            data += 1;
    //        }
    //        m_isovalueBuffer->unmap();
  }

  void IsosurfaceModule::ReadFloatVector(const std::string& fileName, std::vector<ElVisFloat>& values)
  {
    std::ifstream inFile(fileName.c_str());

    while(!inFile.eof())
    {
      std::string line;
      std::getline(inFile, line);
      if( line.empty() )
      {
        continue;
      }

      try
      {
        ElVisFloat value = boost::lexical_cast<ElVisFloat>(line);
        values.push_back(value);
      }
      catch(boost::bad_lexical_cast&)
      {
        std::cout << "Unable to parse " << line << std::endl;
      }

    }
    inFile.close();
  }



}


