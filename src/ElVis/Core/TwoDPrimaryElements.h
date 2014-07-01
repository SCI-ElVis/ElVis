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

#ifndef ELVIS_ELVISNATIVE_TWO_D_PRIMARY_ELEMENTS_H
#define ELVIS_ELVISNATIVE_TWO_D_PRIMARY_ELEMENTS_H

#include <optixu/optixpp.h>
#include <optixu/optixu_matrix.h>

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/Object.h>
#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/Float.h>

#include <iostream>
#include <set>

#include <boost/signals2.hpp>

namespace ElVis
{
    class Scene;
    class Model;

    class TwoDPrimaryElements : public Object
    {
        public:
            ELVIS_EXPORT TwoDPrimaryElements(boost::shared_ptr<Scene> m);
            ELVIS_EXPORT virtual ~TwoDPrimaryElements() {}

        protected:

            ELVIS_EXPORT virtual optixu::Geometry DoCreateOptiXGeometry(SceneView* view);
            ELVIS_EXPORT virtual optixu::Material DoCreateMaterial(SceneView* view);
            ELVIS_EXPORT virtual void DoCreateNode(SceneView* view,
                optixu::Transform& transform, optixu::GeometryGroup& group);

        private:
            TwoDPrimaryElements& operator=(const TwoDPrimaryElements& rhs);
            ELVIS_EXPORT TwoDPrimaryElements(const TwoDPrimaryElements& rhs);

            void HandleModelChanged(boost::shared_ptr<Model> m);
            void Initialize();

            void CopyDataToOptiX();
            void SetupSubscriptions();

            optixu::GeometryGroup m_group;
            optixu::GeometryInstance m_curvedFaceInstance;
            optixu::GeometryInstance m_planarFaceInstance;
            optixu::Transform m_transform;
            optixu::Buffer m_curvedDeviceFlags;
            optixu::Buffer m_planarDeviceFlags;

            optixu::Material m_material;

            boost::shared_ptr<Scene> m_scene;
            std::set<int> m_faceIds;
            std::vector<unsigned char> m_curvedFaceFlags;
            std::vector<unsigned char> m_planarFaceFlags;
            optixu::Buffer m_facesEnabledBuffer;

    };

}

#endif
