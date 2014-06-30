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

#include <ElVis/Core/FaceObject.h>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/OptiXExtensions.hpp>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/Scene.h>

#include <boost/bind.hpp>
#include <boost/typeof/typeof.hpp>

namespace ElVis
{

    FaceObject::FaceObject(boost::shared_ptr<Scene> m) :
        Object(),
        m_group(0),
        m_curvedFaceInstance(0),
        m_planarFaceInstance(0),
        m_transform(0),
        m_curvedDeviceFlags(0),
        m_planarDeviceFlags(0),
        m_material(0),
        m_scene(m),
        m_faceIds(),
        m_curvedFaceFlags(),
        m_planarFaceFlags(),
        m_facesEnabledBuffer()
    {
        SetupSubscriptions();
    }

    void FaceObject::SetupSubscriptions()
    {
    }


    optixu::Geometry FaceObject::DoCreateOptiXGeometry(SceneView* view)
    {
        // TODO - the fact that this should not be called indicates an error in
        // class structure here.
        assert(0);
        return optixu::Geometry();
    }

    optixu::Material FaceObject::DoCreateMaterial(SceneView* view)
    {
        optixu::Context context = view->GetContext();
        m_material = context->createMaterial();
        // This material and ray type 0 uses the cut surface closest hit program.
//        optixu::Program closestHit = PtxManager::LoadProgram(context, view->GetPTXPrefix(), "FaceObjectClosestHit");
//        m_material->setClosestHitProgram(0, closestHit);
        return m_material;
    }

    void FaceObject::EnableFace(int faceId)
    {
        if( m_faceIds.find(faceId) != m_faceIds.end() )
        {
            return;
        }
        m_faceIds.insert(faceId);

        if( !m_facesEnabledBuffer.get() )
        {
            return;
        }

        RTsize bufSize;
        m_facesEnabledBuffer->getSize(bufSize);

        if( faceId >= bufSize )
        {
            std::cout << "Face id " << faceId << " is not in range " << bufSize << std::endl;
        }

        unsigned char* data = static_cast<unsigned char*>(m_facesEnabledBuffer->map());
        data[faceId] = 1;
        m_facesEnabledBuffer->unmap();
        m_group->getAcceleration()->markDirty();
        this->OnObjectChanged(*this);
    }

    void FaceObject::DisableFace(int faceId)
    {
        m_faceIds.erase(faceId);

        if( !m_facesEnabledBuffer.get() ) return;

        RTsize bufSize;
        m_facesEnabledBuffer->getSize(bufSize);

        if( faceId >= bufSize )
        {
            std::cout << "Face id " << faceId << " is not in range " << bufSize << std::endl;
        }

        unsigned char* data = static_cast<unsigned char*>(m_facesEnabledBuffer->map());
        data[faceId] = 0;
        m_facesEnabledBuffer->unmap();
        m_group->getAcceleration()->markDirty();
        this->OnObjectChanged(*this);
    }


    void FaceObject::DoCreateNode(SceneView* view,
                optixu::Transform& transform, optixu::GeometryGroup& group)
    {
        optixu::Context context = view->GetContext();
        if( m_group.get() )
        {
            group = m_group;
            return;
        }

        BOOST_AUTO(model, view->GetScene()->GetModel());

        group = context->createGeometryGroup();

        m_curvedFaceInstance = context->createGeometryInstance();
        m_planarFaceInstance = context->createGeometryInstance();

        optixu::Geometry curvedGeometry = model->GetPlanarFaceGeometry();
        optixu::Geometry planarGeometry = model->GetCurvedFaceGeometry();

        m_planarFaceInstance->setGeometry(planarGeometry);
        m_curvedFaceInstance->setGeometry(curvedGeometry);

        m_group = group;

        group->setChildCount(2);
        group->setChild(0, m_planarFaceInstance);
        group->setChild(1, m_curvedFaceInstance);

        // Somehow sharing the acceleration structure didnt' work.  Revisit if performance indicates.
        m_group->setAcceleration( context->createAcceleration("Sbvh","Bvh") );
        m_facesEnabledBuffer = view->GetScene()->GetModel()->GetFacesEnabledBuffer();

        std::vector<int> temp(m_faceIds.begin(), m_faceIds.end());
        SetFaces(temp, true);
    }

    void FaceObject::SetFaces(const std::vector<int>& ids, bool flag)
    {
        if( !m_facesEnabledBuffer.get() )
        {
            std::cout << "Enable buffer not valid." << std::endl;
            return;
        }

        RTsize bufSize;
        m_facesEnabledBuffer->getSize(bufSize);
        std::cout << "Face buffer size: " << bufSize << std::endl;

        unsigned char* data = static_cast<unsigned char*>(m_facesEnabledBuffer->map());
        for(int i = 0; i < ids.size(); ++i)
        {
            if( ids[i] < bufSize )
            {
                data[ids[i]] = flag;
            }
            else
            {
                std::cout << "Face id " << ids[i] << " is not in range " << bufSize << std::endl;
            }
        }
        m_facesEnabledBuffer->unmap();
        m_group->getAcceleration()->markDirty();
        this->OnObjectChanged(*this);
    }

}


