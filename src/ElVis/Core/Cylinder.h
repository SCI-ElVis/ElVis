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

#ifndef ELVIS_ELVISNATIVE_CYLINDER_H
#define ELVIS_ELVISNATIVE_CYLINDER_H

#include <optixu/optixpp.h>
#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/Object.h>
#include <optixu/optixu_matrix.h>
#include <ElVis/Core/Float.h>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

namespace ElVis
{
    class Cylinder : public Object
    {
        public:
            friend class boost::serialization::access;
            ELVIS_EXPORT Cylinder();
            ELVIS_EXPORT virtual ~Cylinder() {}

            ELVIS_EXPORT optix::Matrix<4,4>& GetTransformationMatrix() { return m_transformationMatrix; }

            ELVIS_EXPORT void SetTransformationMatrixEntry(int index, float value)
            {
                m_transformationMatrix[index] = value;
                if( m_transform.get() )
                {
                    m_transform->setMatrix(false, m_transformationMatrix.getData(), 0);
                }
            }
            
            ELVIS_EXPORT float GetTransformationMatrixEntry(int index)
            {       
                return m_transformationMatrix[index];
            }
            
        protected:
            
            virtual optixu::Geometry DoCreateOptiXGeometry(SceneView* view);
            virtual optixu::Material DoCreateMaterial(SceneView* view);
            virtual void DoCreateNode(SceneView* view, 
                optixu::Transform& transform, optixu::GeometryGroup& group);

        private:
            ELVIS_EXPORT Cylinder(const Cylinder& rhs);
            Cylinder& operator=(const Cylinder& rhs);

            template<typename Archive>
            void serialize(Archive& ar, const unsigned int version)
            {
                ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Object);
            }

            optixu::GeometryGroup m_group;
            optixu::GeometryInstance m_instance;
            optixu::Transform m_transform;
            optix::Matrix<4,4> m_transformationMatrix;
    };
}

#endif //ELVIS_ELVISNATIVE_CYLINDER_H
