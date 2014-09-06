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

#include <ElVis/Core/Cuda.h>

#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/png_io.hpp>

#define png_infopp_NULL (png_infopp)NULL
#ifndef int_p_NULL
#define int_p_NULL (int*)NULL
#endif

namespace ElVis
{
    RayGeneratorProgram VolumeRenderingModule::m_PerformVolumeRendering;

    VolumeRenderingModule::VolumeRenderingModule() :
        RenderModule()
        ,DensityBreakpoints("DensityBreakpoints")
        ,RedBreakpoints("RedBreakpoints")
        ,GreenBreakpoints("GreenBreakpoints")
        ,BlueBreakpoints("BlueBreakpoints")
        ,DensityValues("DensityValues")
        ,RedValues("RedValues")
        ,GreenValues("GreenValues")
        ,BlueValues("BlueValues"),
        m_segmentIntegrationType(eRiemann_SingleThreadPerRay),
        m_enableSampleTracking(false),
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
          std::cout << "Volume Rendering Element Traversal." << std::endl;
          optixu::Context context = view->GetContext();

          if( m_transferFunction->Dirty() )
          {
              std::cout << "Updating transfer function." << std::endl;
              m_transferFunction->CopyToOptix(context, DensityBreakpoints, DensityValues, eDensity);
              m_transferFunction->CopyToOptix(context, RedBreakpoints, RedValues, eRed);
              m_transferFunction->CopyToOptix(context, GreenBreakpoints, GreenValues, eGreen);
              m_transferFunction->CopyToOptix(context, BlueBreakpoints, BlueValues, eBlue);
              m_transferFunction->Dirty() = true;
          }
          SetFloat(context["desiredH"], m_compositingStepSize);
          context->launch(m_PerformVolumeRendering.Index, view->GetWidth(), view->GetHeight());
        }
        catch(optixu::Exception& e)
        {
            std::cout << "Exception encountered rendering primary rays." << std::endl;
            std::cerr << e.getErrorString() << std::endl;
            std::cout << e.getErrorString().c_str() << std::endl;
        }
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
    }


    void VolumeRenderingModule::IntegrateSegment(SceneView* view)
    {
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
    }




    void VolumeRenderingModule::IntegrateTrapezoidal_SingleThreadPerRay(SceneView* view)
    {

    }

    void VolumeRenderingModule::IntegrateSegmentSingleThreadPerRayRiemann(SceneView* view)
    {
 
    }


    void VolumeRenderingModule::IntegrateSingleThreadPerRayFull(SceneView* view)
    {

    }

    void VolumeRenderingModule::IntegrateSingleWarpPerSegmentFull(SceneView* view)
    {

    }

    void VolumeRenderingModule::IntegrateGKOnly(SceneView* view)
    {

    }

    void VolumeRenderingModule::IntegrateSegmentSingleThreadPerRay(SceneView* view)
    {

    }

    void VolumeRenderingModule::IntegrateDensity_AdaptiveTrapezoidal_SingleBlockPerRay(SceneView* view)
    {
    }

    void VolumeRenderingModule::IntegrateDensity_NonAdaptiveTrapezoidal_SingleBlockPerRay(SceneView* view)
    {
    }


    void VolumeRenderingModule::DoSetup(SceneView* view)
    {
        optixu::Context context = view->GetContext();

        if( !m_initializationComplete )
        {

            if( !m_PerformVolumeRendering.IsValid() )
            {
              m_PerformVolumeRendering = view->AddRayGenerationProgram("PerformVolumeRendering");
            }
            if( m_transferFunction->Dirty() )
            {
                optixu::Context context = view->GetContext();
                m_transferFunction->CopyToOptix(context, DensityBreakpoints, DensityValues, eDensity);
                m_transferFunction->CopyToOptix(context, RedBreakpoints, RedValues, eRed);
                m_transferFunction->CopyToOptix(context, GreenBreakpoints, GreenValues, eGreen);
                m_transferFunction->CopyToOptix(context, BlueBreakpoints, BlueValues, eBlue);
                m_transferFunction->Dirty() = true;
            }
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

