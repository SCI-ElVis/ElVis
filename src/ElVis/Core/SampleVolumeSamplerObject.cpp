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


#include <ElVis/Core/SampleVolumeSamplerObject.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/PtxManager.h>
#include <boost/bind.hpp>

namespace ElVis
{
    optixu::Material SampleVolumeSamplerObject::Material;
    bool SampleVolumeSamplerObject::Initialized = SampleVolumeSamplerObject::InitializeStatic();
    optixu::Program SampleVolumeSamplerObject::ClosestHitProgram;
    optixu::Program SampleVolumeSamplerObject::AnyHitProgram;

    SampleVolumeSamplerObject::SampleVolumeSamplerObject() 
    {
    }

    SampleVolumeSamplerObject::SampleVolumeSamplerObject(boost::shared_ptr<Object> obj) :
        PrimaryRayObject(obj)
    {
    }


    SampleVolumeSamplerObject::~SampleVolumeSamplerObject() {}
            
    optixu::Material SampleVolumeSamplerObject::GetMaterial(SceneView* view)
    {
        optixu::Context context = view->GetContext();
        if( !Material.get() )
        {
            Material = context->createMaterial();
            Material->setClosestHitProgram(0, ClosestHitProgram);
            Material->setAnyHitProgram(0, AnyHitProgram);

            // Hack - find a better way.
            Material->setClosestHitProgram(2, ClosestHitProgram);
            Material->setAnyHitProgram(2, AnyHitProgram);
        }
        return Material;
    }

    bool SampleVolumeSamplerObject::InitializeStatic()
    {
        PtxManager::GetOnPtxLoaded().connect(boost::bind(&SampleVolumeSamplerObject::LoadPrograms, _1, _2));
        return true;
    }

    void SampleVolumeSamplerObject::LoadPrograms(const std::string& prefix, optixu::Context context)
    {
        ClosestHitProgram = PtxManager::LoadProgram(prefix, "SamplerVolumeClosestHit");
        AnyHitProgram = PtxManager::LoadProgram(prefix, "IgnoreCutSurfacesOutOfVolume");
    }
}

