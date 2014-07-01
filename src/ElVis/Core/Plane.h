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

#ifndef ELVIS_ELVISNATIVE_PLANE_H
#define ELVIS_ELVISNATIVE_PLANE_H

#include <optixu/optixpp.h>
#include <optixu/optixu_matrix.h>

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/Object.h>
#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/Float.h>

#include <iostream>

#include <boost/signals2.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

namespace ElVis
{
    class Plane : public Object
    {
        public:
            friend class boost::serialization::access;
            ELVIS_EXPORT Plane();
            ELVIS_EXPORT Plane(const WorldPoint& normal, const WorldPoint& p);
            ELVIS_EXPORT virtual ~Plane() {}

            ELVIS_EXPORT const WorldPoint& GetNormal() const { return m_normal; }
            ELVIS_EXPORT const WorldPoint& GetPoint() const { return m_point; }
            ELVIS_EXPORT WorldPoint& GetNormal() { return m_normal; }
            ELVIS_EXPORT WorldPoint& GetPoint() { return m_point; }

            ELVIS_EXPORT void SetNormal(const WorldPoint& value) { m_normal = value; }
            ELVIS_EXPORT void SetPoint(const WorldPoint& value) { m_point = value; }

        protected:

            ELVIS_EXPORT virtual optixu::Geometry DoCreateOptiXGeometry(SceneView* view);
            ELVIS_EXPORT virtual optixu::Material DoCreateMaterial(SceneView* view);
            ELVIS_EXPORT virtual void DoCreateNode(SceneView* view,
                optixu::Transform& transform, optixu::GeometryGroup& group);

        private:
            Plane& operator=(const Plane& rhs);
            ELVIS_EXPORT Plane(const Plane& rhs);
            static bool Initialized;
            static bool InitializeStatic();
            static void LoadPrograms(const std::string& prefix, optixu::Context context);

            void CopyDataToOptiX();
            void HandlePointChanged(const WorldPoint& p);
            void SetupSubscriptions();

            template<typename Archive>
            void serialize(Archive& ar, const unsigned int version)
            {
                ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Object);
                ar & BOOST_SERIALIZATION_NVP(m_normal);
                ar & BOOST_SERIALIZATION_NVP(m_point);
            }

            optixu::GeometryGroup m_group;
            optixu::GeometryInstance m_instance;
            optixu::Transform m_transform;

            optixu::Material m_material;

            WorldPoint m_normal;
            WorldPoint m_point;

            static optixu::Program BoundingProgram;
            static optixu::Program IntersectionProgram;
    };

    ELVIS_EXPORT std::ostream& operator<<(std::ostream& os, const Plane& tri);
}

#endif
