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


#include <ElVis/Core/CutSurfaceMeshModule.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/PtxManager.h>
#include <boost/timer.hpp>
#include <iostream>
#include <stdio.h>

namespace ElVis
{
    const std::string CutSurfaceMeshModule::ProgramName("SamplePixelCornersRayGeneratorForCategorization");
    const std::string CutSurfaceMeshModule::CategorizeProgramName("CategorizeMeshPixels");

    CutSurfaceMeshModule::CutSurfaceMeshModule() :
        m_program(),
        m_categorizeProgram()
    {
    }

    void CutSurfaceMeshModule::DoSetup(SceneView* view)
    {
        optixu::Context context = view->GetContext();
        assert(context);

        m_program = view->AddRayGenerationProgram(ProgramName.c_str());
        m_categorizeProgram = view->AddRayGenerationProgram(CategorizeProgramName.c_str());
    }

    void CutSurfaceMeshModule::DoRender(SceneView* view)
    {
        optixu::Context context = view->GetContext();

        context->launch(m_program.Index, view->GetWidth()+1, view->GetHeight()+1);
        context->launch(m_categorizeProgram.Index, view->GetWidth(), view->GetHeight());
    }

    std::string CutSurfaceMeshModule::DoGetName() const
    {
        return "Cut-Surface Mesh";
    }
}

