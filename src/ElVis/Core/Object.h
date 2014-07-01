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

#ifndef ELVIS_ELVISNATIVE_OBJECT_H
#define ELVIS_ELVISNATIVE_OBJECT_H

#include <optixu/optixpp.h>
#include <ElVis/Core/ElVisDeclspec.h>

#include <boost/signals2.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

namespace ElVis
{
    class SceneView;
    class Object
    {
        public:
            friend class boost::serialization::access;
            Object()
            {
            }

            virtual ~Object() {}

            ELVIS_EXPORT optixu::Geometry CreateOptiXGeometry(SceneView* view);
            ELVIS_EXPORT optixu::Material CreateMaterial(SceneView* view);

            // Creates a single branch in the tree for this object.  The return is anticipated
            // to be either the group or the transform, and will be added to an optixu::Group
            // as a child.
            ELVIS_EXPORT void CreateNode(SceneView* view, 
                optixu::Transform& transform, optixu::GeometryGroup& group);

            boost::signals2::signal<void (const Object&)> OnObjectChanged;

        protected:
            Object(const Object& rhs);
            Object& operator=(const Object&);

            // Deprecated
            ELVIS_EXPORT virtual optixu::Geometry DoCreateOptiXGeometry(SceneView* view) = 0;

            // Deprecated
            ELVIS_EXPORT virtual optixu::Material DoCreateMaterial(SceneView* view) = 0;

            // Creates the geometry and instances required for this object to enter 
            // the scene graph.  Subclasses can fill in a default geometry if desired, 
            // but callers may replace it.
            ELVIS_EXPORT virtual void DoCreateNode(SceneView* view, 
                optixu::Transform& transform, optixu::GeometryGroup& group) = 0;


        private:
            template<typename Archive>
            void serialize(Archive& ar, const unsigned int version)
            {
            }
    };
}

#endif //ELVIS_ELVISNATIVE_OBJECT_H
