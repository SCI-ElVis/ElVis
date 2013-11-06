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

#include <ElVis/Core/Plane.h>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/OptiXExtensions.hpp>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/SceneView.h>

#include <boost/bind.hpp>

namespace ElVis
{
    bool Plane::Initialized = Plane::InitializeStatic();
    optixu::Program Plane::IntersectionProgram;
    optixu::Program Plane::BoundingProgram;

    Plane::Plane() :
        Object(),
        m_normal(1.0, 0.0, 0.0),
        m_point(0.0, 0.0, 0.0)
    {
        SetupSubscriptions();
    }

    Plane::Plane(const WorldPoint& normal, const WorldPoint& p) :
        Object(),
        m_normal(normal),
        m_point(p)
    {
        SetupSubscriptions();
    }

    void Plane::SetupSubscriptions()
    {
        m_normal.OnPointChanged.connect(boost::bind(&Plane::HandlePointChanged, this, _1));
        m_point.OnPointChanged.connect(boost::bind(&Plane::HandlePointChanged, this, _1));
    }

    void Plane::HandlePointChanged(const WorldPoint& p)
    {
        CopyDataToOptiX();
        OnObjectChanged(*this);
    }


    optixu::Geometry Plane::DoCreateOptiXGeometry(SceneView* view)
    {
        optixu::Context context = view->GetContext();
        optixu::Geometry result = context->createGeometry();
        result->setPrimitiveCount(1u);

        result->setBoundingBoxProgram(BoundingProgram);
        result->setIntersectionProgram(IntersectionProgram);
//        result->setBoundingBoxProgram( PtxManager::LoadProgram(context, view->GetPTXPrefix(), "Plane_bounding") );
//        result->setIntersectionProgram( PtxManager::LoadProgram(context, view->GetPTXPrefix(), "Plane_intersect") );
        return result;
    }

    optixu::Material Plane::DoCreateMaterial(SceneView* view)
    {
        optixu::Context context = view->GetContext();
        m_material = context->createMaterial();
        // This material and ray type 0 uses the cut surface closest hit program.
        optixu::Program closestHit = PtxManager::LoadProgram(context, view->GetPTXPrefix(), "PlaneClosestHit");
        m_material->setClosestHitProgram(0, closestHit);
        //m_material["ObjectLightingType"]->setInt(static_cast<int>(GetLightingType()));
        return m_material;
    }

    void Plane::DoCreateNode(SceneView* view,
                optixu::Transform& transform, optixu::GeometryGroup& group)
    {
        optixu::Context context = view->GetContext();

        if( m_group.get() )
        {
            group = m_group;
            return;
        }

        group = context->createGeometryGroup();
        group->setAcceleration( context->createAcceleration("NoAccel","NoAccel") );

        group->setChildCount(1);
        optixu::GeometryInstance instance = context->createGeometryInstance();
        optixu::Geometry geom = CreateOptiXGeometry(view);
        instance->setGeometry(geom);

        m_group = group;
        m_instance = instance;

        CopyDataToOptiX();

        group->setChild(0, instance);

    }

    void Plane::CopyDataToOptiX()
    {
        if( m_instance )
        {
            ElVisFloat D = -m_normal.x()*m_point.x() - m_normal.y()*m_point.y() - m_normal.z()*m_point.z();
            SetFloat(m_instance["PlaneNormal"], m_normal.x(), m_normal.y(), m_normal.z(), D);
            SetFloat(m_instance["PlanePoint"], m_point.x(), m_point.y(), m_point.z());
        }
    }


    std::ostream& operator<<(std::ostream& os, const Plane& obj)
    {
        os << "Normal: " << obj.GetNormal() << std::endl;
        os << "Point: " << obj.GetPoint() << std::endl;
        return os;
    }

    bool Plane::InitializeStatic()
    {
        PtxManager::GetOnPtxLoaded().connect(boost::bind(&Plane::LoadPrograms, _1, _2));
        return true;
    }

    void Plane::LoadPrograms(const std::string& prefix, optixu::Context context)
    {
        BoundingProgram = PtxManager::LoadProgram(prefix, "Plane_bounding");
        IntersectionProgram = PtxManager::LoadProgram(prefix, "Plane_intersect");
    }
}


