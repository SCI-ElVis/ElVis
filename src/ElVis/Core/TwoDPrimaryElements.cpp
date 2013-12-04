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

#include <ElVis/Core/TwoDPrimaryElements.h>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/OptiXExtensions.hpp>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/Scene.h>

#include <boost/bind.hpp>
#include <boost/typeof/typeof.hpp>

#include <boost/range.hpp>
#include <boost/range/algorithm/for_each.hpp>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>

namespace ElVis
{

    TwoDPrimaryElements::TwoDPrimaryElements(boost::shared_ptr<Scene> m) :
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

        if( m_scene->GetModel() )
        {
            Initialize();
        }

        //// Query the model to see if there are any 2D elements that are not 
        //// embedded in 3D space.  If so, create a node that can be used by 
        //// the primary ray module to render them.
        //BOOST_AUTO(model, m->GetModel());
        //if( model->Has2DElementsIn2DSpace() )
        //{
        //    int numberOf2DElementTypes = model->GetNumberOf2DElementTypesIn2DSpace();

        //    for(int i = 0; i < numberOf2DElementTypes; ++i)
        //    {
        //    }
        //}
    }

    void TwoDPrimaryElements::SetupSubscriptions()
    {
        m_scene->OnModelChanged.connect(boost::bind(&TwoDPrimaryElements::HandleModelChanged, this, _1));
    }

    void TwoDPrimaryElements::HandleModelChanged(boost::shared_ptr<Model> m)
    {
        Initialize();
    }

    void TwoDPrimaryElements::Initialize()
    {
    }

    optixu::Geometry TwoDPrimaryElements::DoCreateOptiXGeometry(SceneView* view)
    {
        // TODO - the fact that this should not be called indicates an error in
        // class structure here.
        assert(0);
        return optixu::Geometry();
    }

    optixu::Material TwoDPrimaryElements::DoCreateMaterial(SceneView* view)
    {
        if( !m_material.get() ) 
        {
            optixu::Context context = view->GetContext();
            BOOST_AUTO( model, m_scene->GetModel());
            m_material = model->Get2DPrimaryGeometryMaterial(view);
        }
        return m_material;
    }

    void TwoDPrimaryElements::DoCreateNode(SceneView* view,
                optixu::Transform& transform, optixu::GeometryGroup& group)
    {
        optixu::Context context = view->GetContext();
        BOOST_AUTO( model, m_scene->GetModel());

        if( m_group.get() )
        {
            group = m_group;
            return;
        }


        std::vector<optixu::GeometryInstance> twoDGroups = model->Get2DPrimaryGeometry(m_scene, context);
        if( twoDGroups.empty() )
        {
            return;
        }

        group = context->createGeometryGroup();
        int idx = 0;
        group->setChildCount(twoDGroups.size());

        boost::for_each(twoDGroups, 
          (boost::lambda::bind(&optixu::GeometryGroupObj::setChild, group.get(), idx, boost::lambda::_1), ++idx));
        //boost::for_each(twoDGroups, [&](optixu::GeometryInstance childGroup)
        //{
        //    group->setChild(idx, childGroup);
        //    ++idx;
        //});
        
        
        optixu::Acceleration acc = context->createAcceleration("Sbvh","Bvh");
        //optixu::Acceleration acc = context->createAcceleration("NoAccel", "NoAccel");
        group->setAcceleration(acc);

        m_group = group;
        
        
       
    }

    void TwoDPrimaryElements::CopyDataToOptiX()
    {
//        if( m_curvedFaceInstance && m_curvedFaceFlags.size() > 0 )
//        {
//            unsigned char* data = static_cast<unsigned char*>(m_curvedDeviceFlags->map());
//            std::copy(m_curvedFaceFlags.begin(), m_curvedFaceFlags.end(), data);
//            m_curvedDeviceFlags->unmap();
//        }

//        if( m_planarFaceInstance && m_planarFaceFlags.size() > 0)
//        {
//            unsigned char* data = static_cast<unsigned char*>(m_planarDeviceFlags->map());
//            std::copy(m_planarFaceFlags.begin(), m_planarFaceFlags.end(), data);
//            m_planarDeviceFlags->unmap();
//        }
    }



}


