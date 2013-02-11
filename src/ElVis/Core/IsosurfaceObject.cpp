/////////////////////////////////////////////////////////////////////////////////
////
//// The MIT License
////
//// Copyright (c) 2006 Scientific Computing and Imaging Institute,
//// University of Utah (USA)
////
//// License for the specific language governing rights and limitations under
//// Permission is hereby granted, free of charge, to any person obtaining a
//// copy of this software and associated documentation files (the "Software"),
//// to deal in the Software without restriction, including without limitation
//// the rights to use, copy, modify, merge, publish, distribute, sublicense,
//// and/or sell copies of the Software, and to permit persons to whom the
//// Software is furnished to do so, subject to the following conditions:
////
//// The above copyright notice and this permission notice shall be included
//// in all copies or substantial portions of the Software.
////
//// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//// DEALINGS IN THE SOFTWARE.
////
/////////////////////////////////////////////////////////////////////////////////

//#include <ElVis/Core/IsosurfaceObject.h>
//#include <ElVis/Core/SceneView.h>
//#include <ElVis/Core/PtxManager.h>

//#include <boost/foreach.hpp>

//namespace ElVis
//{
//    IsosurfaceObject::IsosurfaceObject() :
//        Object(),
//        OnIsovalueAdded(),
//        m_group(),
//        m_instance(),
//        m_transform(),
//        m_isovalues(),
//        m_isovalueBuffer("IsosurfaceIsovalues", 1)
//    {
//    }

//    void IsosurfaceObject::AddIsovalue(const ElVisFloat& value)
//    {
//        if( m_isovalues.find(value) == m_isovalues.end() )
//        {
//            m_isovalues.insert(value);
//            OnIsovalueAdded(value);
//            SynchronizeWithOptix();
//        }
//    }

//    optixu::Geometry IsosurfaceObject::DoCreateOptiXGeometry(SceneView* view)
//    {
//        optixu::Context context = view->GetContext();
//        optixu::Geometry result = context->createGeometry();
//        result->setPrimitiveCount(1u);

//        result->setBoundingBoxProgram( PtxManager::LoadProgram(context, view->GetPTXPrefix(), "IsosurfaceBounding") );
//        result->setIntersectionProgram( PtxManager::LoadProgram(context, view->GetPTXPrefix(), "IsosurfaceIntersection") );
//        return result;
//    }

//    optixu::Material IsosurfaceObject::DoCreateMaterial(SceneView* view)
//    {
//        return optixu::Material();
//    }

//    void IsosurfaceObject::DoCreateNode(SceneView* view, optixu::Transform& transform, optixu::GeometryGroup& group, optixu::GeometryInstance& instance)
//    {
//        optixu::Context context = view->GetContext();
//        if( m_group.get() )
//        {
//            group = m_group;
//            instance = m_instance;
//            return;
//        }

//        group = context->createGeometryGroup();
//        group->setAcceleration( context->createAcceleration("NoAccel","NoAccel") );

//        group->setChildCount(1);
//        instance = context->createGeometryInstance();
//        optixu::Geometry geom = CreateOptiXGeometry(view);
//        instance->setGeometry(geom);
//        group->setChild(0, instance);
//        m_isovalueBuffer.Create(context, RT_BUFFER_INPUT, m_isovalues.size());
//        instance[m_isovalueBuffer.Name().c_str()]->set(*m_isovalueBuffer);

//        m_group = group;
//        m_instance = instance;

//        SynchronizeWithOptix();
//    }

//    void IsosurfaceObject::SynchronizeWithOptix()
//    {
//        if( !m_instance ) return;

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
//    }

//}

