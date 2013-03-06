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


#include <ElVis/Core/PrimaryRayObject.h>
#include <ElVis/Core/SceneView.h>

#include <boost/bind.hpp>

namespace ElVis
{
    PrimaryRayObject::PrimaryRayObject() :
        m_object()
    {
        SetupSubscriptions();
    }

    PrimaryRayObject::PrimaryRayObject(boost::shared_ptr<Object> obj) :
        m_object(obj)    
    {
        SetupSubscriptions();
    }


    PrimaryRayObject::~PrimaryRayObject()
    {
    }


    void PrimaryRayObject::CreateNode(SceneView* view, 
        optixu::Transform& transform, optixu::GeometryGroup& group)
    {
        m_object->CreateNode(view, transform, group);

        if( !group.get() ) return;

        for(unsigned int i = 0; i < group->getChildCount(); ++i)
        {
            optixu::GeometryInstance instance = group->getChild(i);
            instance->setMaterialCount(1);
            instance->setMaterial(0, GetMaterial(view));
        }
    }

    void PrimaryRayObject::SetupSubscriptions()
    {
        m_object->OnObjectChanged.connect(boost::bind(&PrimaryRayObject::HandleObjectChanged, this, _1));
    }

    void PrimaryRayObject::HandleObjectChanged(const Object&)
    {
        OnObjectChanged(*this);
    }
}

