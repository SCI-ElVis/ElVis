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


#ifndef ELVISNATIVE_ELEMENT_FACE_RENDERING_OBJECT
#define ELVISNATIVE_ELEMENT_FACE_RENDERING_OBJECT

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/PrimaryRayObject.h>
#include <ElVis/Core/FaceObject.h>

#include <optixu/optixpp.h>
#include <optix_math.h>

namespace ElVis
{
    class SampleFaceObject : public PrimaryRayObject
    {
        public:
            ELVIS_EXPORT SampleFaceObject();
            ELVIS_EXPORT explicit SampleFaceObject(boost::shared_ptr<FaceObject> obj);
            ELVIS_EXPORT virtual ~SampleFaceObject() {}

            ELVIS_EXPORT void EnableFace(int faceId);
            ELVIS_EXPORT void DisableFace(int faceId);

            ELVIS_EXPORT void SetFaces(const std::vector<int>& ids, bool flag);

        protected:
            virtual optixu::Material GetMaterial(SceneView* view);

        private:
            SampleFaceObject& operator=(const SampleFaceObject& rhs);
            ELVIS_EXPORT SampleFaceObject(const SampleFaceObject& rhs);

            static bool Initialized;
            static bool InitializeStatic();
            static void LoadPrograms(const std::string& prefix, optixu::Context context);

            static optixu::Material Material;
            static optixu::Program ClosestHitProgram;
    };
}

#endif
