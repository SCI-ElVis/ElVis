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


#include <ElVis/Core/SampleFaceObject.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/PtxManager.h>
#include <boost/bind.hpp>

namespace ElVis
{
    optixu::Material SampleFaceObject::Material;
    bool SampleFaceObject::Initialized = SampleFaceObject::InitializeStatic();
    optixu::Program SampleFaceObject::ClosestHitProgram;

    SampleFaceObject::SampleFaceObject()
    {
    }

    SampleFaceObject::SampleFaceObject(boost::shared_ptr<FaceObject> obj) :
        PrimaryRayObject(obj)
    {
    }


    optixu::Material SampleFaceObject::GetMaterial(SceneView* view)
    {
        optixu::Context context = view->GetContext();
        if( !Material.get() )
        {
            Material = context->createMaterial();
            Material->setClosestHitProgram(0, ClosestHitProgram);

            // Hack - find a better way.
            Material->setClosestHitProgram(2, ClosestHitProgram);
        }
        return Material;
    }

    void SampleFaceObject::EnableFace(int faceId)
    {
        boost::shared_ptr<FaceObject> obj = boost::dynamic_pointer_cast<FaceObject>(this->GetObject());
        if( !obj )
        {
            std::cout << "Case failed. " << std::endl;

            return;
        }

        obj->EnableFace(faceId);
    }

    void SampleFaceObject::DisableFace(int faceId)
    {
        boost::shared_ptr<FaceObject> obj = boost::dynamic_pointer_cast<FaceObject>(this->GetObject());
        if( !obj ) return;

        obj->DisableFace(faceId);
    }

    void SampleFaceObject::SetFaces(const std::vector<int>& ids, bool flag)
    {
        boost::shared_ptr<FaceObject> obj = boost::dynamic_pointer_cast<FaceObject>(this->GetObject());
        if( !obj ) return;

        obj->SetFaces(ids, flag);
    }


    bool SampleFaceObject::InitializeStatic()
    {
        PtxManager::GetOnPtxLoaded().connect(boost::bind(&SampleFaceObject::LoadPrograms, _1, _2));
        return true;
    }

    void SampleFaceObject::LoadPrograms(const std::string& prefix, optixu::Context context)
    {
        ClosestHitProgram = PtxManager::LoadProgram(prefix, "SamplerFaceClosestHit");
    }
}

