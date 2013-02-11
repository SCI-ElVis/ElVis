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

//#ifndef ELVIS_CORE_TRIANGLE_H
//#define ELVIS_CORE_TRIANGLE_H

//#include <optixu/optixpp.h>
//#include <optixu/optixu_matrix.h>


//#include <ElVis/Core/ElVisDeclspec.h>
//#include <ElVis/Core/Float.h>
//#include <ElVis/Core/Object.h>
//#include <ElVis/Core/Point.hpp>
//#include <ElVis/Core/Buffer.h>

//#include <iostream>
//#include <set>

//namespace ElVis
//{
//    class IsosurfaceObject : public Object
//    {
//        public:
//            ELVIS_EXPORT IsosurfaceObject();
//            ELVIS_EXPORT virtual ~IsosurfaceObject() {}

//            ELVIS_EXPORT void AddIsovalue(const ElVisFloat& value);

//            boost::signal< void (ElVisFloat) > OnIsovalueAdded;

//        protected:

//            ELVIS_EXPORT virtual optixu::Geometry DoCreateOptiXGeometry(SceneView* view);
//            ELVIS_EXPORT virtual optixu::Material DoCreateMaterial(SceneView* view);
//            ELVIS_EXPORT virtual void DoCreateNode(SceneView* view,
//                optixu::Transform& transform, optixu::GeometryGroup& group, optixu::GeometryInstance& instance);

//        private:
//            IsosurfaceObject& operator=(const IsosurfaceObject& rhs);
//            IsosurfaceObject(const IsosurfaceObject& rhs);

//            void SynchronizeWithOptix();

//            optixu::GeometryGroup m_group;
//            optixu::GeometryInstance m_instance;
//            optixu::Transform m_transform;

//            std::set<ElVisFloat> m_isovalues;
//            FloatingPointBuffer m_isovalueBuffer;

//    };

//}

//#endif
