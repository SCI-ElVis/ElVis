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

#include <ElVis/Core/Cylinder.h>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/SceneView.h>

#include <iostream>

namespace ElVis
{
    Cylinder::Cylinder() :
        Object(),
        m_transformationMatrix(optix::Matrix<4,4>::identity())
    {
    }


    optixu::Geometry Cylinder::DoCreateOptiXGeometry(SceneView* view)
    {
        optixu::Context context = view->GetContext();
        optixu::Geometry result = context->createGeometry();
        result->setPrimitiveCount(1u);

        result->setBoundingBoxProgram( PtxManager::LoadProgram(context, view->GetPTXPrefix(), "CylinderBounding") );
        result->setIntersectionProgram( PtxManager::LoadProgram(context, view->GetPTXPrefix(), "CylinderIntersect") );
        return result;
    }

    optixu::Material Cylinder::DoCreateMaterial(SceneView* view)
    {
        optixu::Context context = view->GetContext();
        return context->createMaterial();
    }

    void Cylinder::DoCreateNode(SceneView* view, 
                optixu::Transform& transform, optixu::GeometryGroup& group)
    {
        optixu::Context context = view->GetContext();
        if( m_transform.get() )
        {
            transform = m_transform;
            group = m_group;
            return;
        }
        
        optixu::Geometry geom = CreateOptiXGeometry(view);
        transform = context->createTransform();
        transform->setMatrix(false, m_transformationMatrix.getData(), 0);
        
        group = context->createGeometryGroup();
        group->setAcceleration( context->createAcceleration("NoAccel","NoAccel") );
        transform->setChild(group);

        group->setChildCount(1);
        optixu::GeometryInstance instance = context->createGeometryInstance();
        instance->setGeometry(geom);
        
        group->setChild(0, instance);

        m_transform = transform;
        m_group = group;
        m_instance = instance;
    }

}


