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

#include <ElVis/Core/Triangle.h>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/OptiXExtensions.hpp>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/SceneView.h>

namespace ElVis
{

    Triangle::Triangle() :
        Object()
    {
    }

//    Triangle::Triangle(const Triangle& rhs) :
//        Object(rhs)
//    {
//    }


    optixu::Geometry Triangle::DoCreateOptiXGeometry(SceneView* view)
    {
        optixu::Context context = view->GetContext();
        optixu::Geometry result = context->createGeometry();
        result->setPrimitiveCount(1u);

        result->setBoundingBoxProgram( PtxManager::LoadProgram(context, view->GetPTXPrefix(), "triangle_bounding") );
        result->setIntersectionProgram( PtxManager::LoadProgram(context, view->GetPTXPrefix(), "triangle_intersect") );
        return result;
    }

    optixu::Material Triangle::DoCreateMaterial(SceneView* view)
    {
        optixu::Context context = view->GetContext();
        m_material = context->createMaterial();
        // This material and ray type 0 uses the cut surface closest hit program.
        optixu::Program closestHit = PtxManager::LoadProgram(context, view->GetPTXPrefix(), "TriangleClosestHit");
        m_material->setClosestHitProgram(0, closestHit);
        //m_material["ObjectLightingType"]->setInt(static_cast<int>(GetLightingType()));
        return m_material;
    }

    void Triangle::DoCreateNode(SceneView* view, 
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

        SetFloat(instance["TriangleVertex0"], static_cast<ElVisFloat>(m_p0.x()), static_cast<ElVisFloat>(m_p0.y()), static_cast<ElVisFloat>(m_p0.z()));
        SetFloat(instance["TriangleVertex1"], static_cast<ElVisFloat>(m_p1.x()), static_cast<ElVisFloat>(m_p1.y()), static_cast<ElVisFloat>(m_p1.z()));
        SetFloat(instance["TriangleVertex2"], static_cast<ElVisFloat>(m_p2.x()), static_cast<ElVisFloat>(m_p2.y()), static_cast<ElVisFloat>(m_p2.z()));
        instance["s0"]->setFloat(m_scalar0);
        instance["s1"]->setFloat(m_scalar1);
        instance["s2"]->setFloat(m_scalar2);
        group->setChild(0, instance);

        m_group = group;
        m_instance = instance;
    }
    
    void Triangle::SetP0(const WorldPoint& value) 
    { 
        m_p0 = value; 
        if( m_instance.get() )
        {
            SetFloat(m_instance["p0"], static_cast<ElVisFloat>(m_p0.x()), static_cast<ElVisFloat>(m_p0.y()), static_cast<ElVisFloat>(m_p0.z()));
        }
    }
    void Triangle::SetP1(const WorldPoint& value) 
    { 
        m_p1 = value; 
        if( m_instance.get() )
        {
            SetFloat(m_instance["p1"], static_cast<ElVisFloat>(m_p1.x()), static_cast<ElVisFloat>(m_p1.y()), static_cast<ElVisFloat>(m_p1.z()));
        }
    }
    void Triangle::SetP2(const WorldPoint& value) 
    { 
        m_p2 = value; 
        if( m_instance.get() )
        {
            SetFloat(m_instance["p2"], static_cast<ElVisFloat>(m_p2.x()), static_cast<ElVisFloat>(m_p2.y()), static_cast<ElVisFloat>(m_p2.z()));
        }
    }

    void Triangle::SetPoints(const WorldPoint& p0, const WorldPoint& p1, const WorldPoint& p2) 
    {
        SetP0(p0);
        SetP1(p1);
        SetP2(p2);
    }

    std::ostream& operator<<(std::ostream& os, const Triangle& tri)
    {
        os << "P0: " << tri.GetP0() << std::endl;
        os << "P1: " << tri.GetP1() << std::endl;
        os << "P2: " << tri.GetP2() << std::endl;
        return os;
    }
}


