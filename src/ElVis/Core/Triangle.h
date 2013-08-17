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

#ifndef ELVIS_ELVISNATIVE_TRIANGLE_H
#define ELVIS_ELVISNATIVE_TRIANGLE_H

#include <optixu/optixpp.h>
#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/Object.h>
#include <optixu/optixu_matrix.h>
#include <ElVis/Core/Point.hpp>

#include <iostream>

namespace ElVis
{
    class Triangle : public Object
    {
        public:
            ELVIS_EXPORT Triangle();
            ELVIS_EXPORT virtual ~Triangle() {}

            ELVIS_EXPORT const WorldPoint& GetP0() const { return m_p0; }
            ELVIS_EXPORT const WorldPoint& GetP1() const { return m_p1; }
            ELVIS_EXPORT const WorldPoint& GetP2() const { return m_p2; }

            ELVIS_EXPORT void SetP0(const WorldPoint& value);
            ELVIS_EXPORT void SetP1(const WorldPoint& value);
            ELVIS_EXPORT void SetP2(const WorldPoint& value);

            ELVIS_EXPORT void SetPoints(const WorldPoint& p0, const WorldPoint& p1, const WorldPoint& p2);

            ELVIS_EXPORT void SetScalars(float s0, float s1, float s2)
            {
                m_scalar0 = s0;
                m_scalar1 = s1;
                m_scalar2 = s2;
            }

        protected:
            
            ELVIS_EXPORT virtual optixu::Geometry DoCreateOptiXGeometry(SceneView* view);
            ELVIS_EXPORT virtual optixu::Material DoCreateMaterial(SceneView* view);
            ELVIS_EXPORT virtual void DoCreateNode(SceneView* view, 
                optixu::Transform& transform, optixu::GeometryGroup& group);

        private:
            Triangle& operator=(const Triangle& rhs);
            ELVIS_EXPORT Triangle(const Triangle& rhs);

            optixu::GeometryGroup m_group;
            optixu::GeometryInstance m_instance;
            optixu::Transform m_transform;
            
            optixu::Material m_material;
            WorldPoint m_p0;
            WorldPoint m_p1;
            WorldPoint m_p2;

            float m_scalar0;
            float m_scalar1;
            float m_scalar2;
    };

    ELVIS_EXPORT std::ostream& operator<<(std::ostream& os, const Triangle& tri);
}

#endif 
