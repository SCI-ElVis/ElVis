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


#ifndef ELVISNATIVE_PRIMARY_RAY_OBJECT_H
#define ELVISNATIVE_PRIMARY_RAY_OBJECT_H

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/Object.h>
#include <optixu/optixpp.h>
#include <optix_math.h>

#include <boost/shared_ptr.hpp>
#include <boost/signals2.hpp>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/list.hpp>

namespace ElVis
{
    /// \brief Adapts objects so they can be used by the primary ray module.
    class PrimaryRayObject
    {
        public:
            friend class boost::serialization::access;
            ELVIS_EXPORT PrimaryRayObject();
            ELVIS_EXPORT explicit PrimaryRayObject(boost::shared_ptr<Object> obj);
            ELVIS_EXPORT virtual ~PrimaryRayObject();
            
            ELVIS_EXPORT boost::shared_ptr<Object> GetObject() const { return m_object; }

            ELVIS_EXPORT void CreateNode(SceneView* view, optixu::Transform& transform, optixu::GeometryGroup& group);

            boost::signals2::signal<void (const PrimaryRayObject&)> OnObjectChanged;

        protected:
            virtual optixu::Material GetMaterial(SceneView* view) = 0;

        private:
            PrimaryRayObject& operator=(const PrimaryRayObject& rhs);
            PrimaryRayObject(const PrimaryRayObject& rhs);

            void HandleObjectChanged(const Object&);
            void SetupSubscriptions();

            template<typename Archive>
            void serialize(Archive& ar, const unsigned int version)
            {
                ar & BOOST_SERIALIZATION_NVP(m_object);
            }

            boost::shared_ptr<Object> m_object;
    };
}

#endif
