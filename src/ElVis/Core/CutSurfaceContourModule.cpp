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


#include <ElVis/Core/CutSurfaceContourModule.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/PtxManager.h>
#include <boost/timer.hpp>
#include <boost/typeof/typeof.hpp>
#include <iostream>
#include <stdio.h>

namespace ElVis
{
    const std::string CutSurfaceContourModule::ReferencePointAtIntersectionBufferName("ReferencePointAtIntersectionBuffer");

    CutSurfaceContourModule::CutSurfaceContourModule() :
        m_contourSampleBuffer("ContourSampleBuffer"),
        m_isovalueBuffer("Isovalues"),
        m_referencePointAtIntersectionBuffer(ReferencePointAtIntersectionBufferName),
        m_dirty(true),
        m_treatElementBoundariesAsDiscontinuous(0),
        m_matchVisual3Contours(0)
    {}

    void CutSurfaceContourModule::DoSetup(SceneView* view)
    {
        optixu::Context context = view->GetContext();
        assert(context);

        m_contourSampleBuffer.SetContext(context);
        m_contourSampleBuffer.SetDimensions(view->GetWidth()+1, view->GetHeight()+1);
        
        m_isovalueBuffer.SetContext(context);
        m_isovalueBuffer.SetDimensions(m_isovalues.size());

        m_elementIdAtIntersectionBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, view->GetWidth()+1, view->GetHeight()+1);
        context["ElementIdAtIntersectionBuffer"]->set(m_elementIdAtIntersectionBuffer);

        m_elementTypeAtIntersectionBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, view->GetWidth()+1, view->GetHeight()+1);
        context["ElementTypeAtIntersectionBuffer"]->set(m_elementTypeAtIntersectionBuffer);

        m_referencePointAtIntersectionBuffer.SetContext(context);
        m_referencePointAtIntersectionBuffer.SetDimensions(view->GetWidth()+1, view->GetHeight()+1);

        m_sampleRayProgram = view->AddRayGenerationProgram("SamplePixelCornersRayGeneratorForCategorization");
        m_markPixelProgram = view->AddRayGenerationProgram("CategorizeContourPixels");

        context->setMissProgram(2, PtxManager::LoadProgram(context, view->GetPTXPrefix(), "ContourMiss"));
    }

    void CutSurfaceContourModule::DoSynchronize(SceneView* view)
    {
        if( m_dirty )
        {
            m_isovalueBuffer.SetDimensions(m_isovalues.size());
            BOOST_AUTO(isovalueData, m_isovalueBuffer.Map());
            int index = 0;
            for(std::set<ElVisFloat>::iterator iter = m_isovalues.begin(); iter != m_isovalues.end(); ++iter)
            {
                isovalueData[index] = *iter;
                ++index;
            }

            optixu::Context context = view->GetContext();
            context["TreatElementBoundariesAsDiscontinuous"]->setInt(m_treatElementBoundariesAsDiscontinuous);
            context["MatchVisual3Contours"]->setInt(m_matchVisual3Contours);

            m_dirty = false;
        }
    }

    void CutSurfaceContourModule::DoRender(SceneView* view)
    {
        optixu::Context context = view->GetContext();
        
        context->launch(m_sampleRayProgram.Index, view->GetWidth()+1, view->GetHeight()+1);
        context->launch(m_markPixelProgram.Index, view->GetWidth(), view->GetHeight());
    }


        
    void CutSurfaceContourModule::DoResize(unsigned int newWidth, unsigned int newHeight)
    {
        if( m_contourSampleBuffer.Initialized() )
        {
            m_contourSampleBuffer.SetDimensions(newWidth+1, newHeight+1);
            m_elementIdAtIntersectionBuffer->setSize(newWidth+1, newHeight+1);
            m_elementTypeAtIntersectionBuffer->setSize(newWidth+1, newHeight+1);
            m_referencePointAtIntersectionBuffer.SetDimensions(newWidth+1, newHeight+1);
        }
    }

    std::string CutSurfaceContourModule::DoGetName() const
    {
        return "Cut-Surface Contour";
    }

    void CutSurfaceContourModule::AddIsovalue(ElVisFloat newVal)
    {
        if( m_isovalues.find(newVal) == m_isovalues.end() )
        {
            m_isovalues.insert(newVal);
            SetSyncAndRenderRequired();
            m_dirty = true;
            OnIsovalueAdded(newVal);
            OnModuleChanged(*this);
        }
    }

    void CutSurfaceContourModule::RemoveIsovalue(ElVisFloat newVal)
    {
        std::set<ElVisFloat>::iterator found = m_isovalues.find(newVal);
        if( found != m_isovalues.end() )
        {
            m_isovalues.erase(found);
            SetSyncAndRenderRequired();
            m_dirty = true;
            OnIsovalueRemoved(newVal);
            OnModuleChanged(*this);
        }
    }

    int CutSurfaceContourModule::GetTreatElementBoundariesAsDiscontinuous() const
    {
        return m_treatElementBoundariesAsDiscontinuous;
    }

    void CutSurfaceContourModule::SetTreatElementBoundariesAsDiscontinuous(int value)
    {
        if( m_treatElementBoundariesAsDiscontinuous != value )
        {
            m_treatElementBoundariesAsDiscontinuous = value;
            SetSyncAndRenderRequired();
            m_dirty = true;
            OnModuleChanged(*this);
        }
    }

    int CutSurfaceContourModule::GetMatchVisual3Contours() const
    {
        return m_matchVisual3Contours;
    }

    void CutSurfaceContourModule::SetMatchVisual3Contours(int newValue)
    {
        if( m_matchVisual3Contours != newValue)
        {
            m_matchVisual3Contours = newValue;
            SetSyncAndRenderRequired();
            m_dirty = true;
            OnModuleChanged(*this);
        }
    }

}

