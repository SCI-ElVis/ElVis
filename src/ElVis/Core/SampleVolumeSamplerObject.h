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


#ifndef ELVISNATIVE_SAMPLE_VOLUME_SAMPLER_OBJECT
#define ELVISNATIVE_SAMPLE_VOLUME_SAMPLER_OBJECT

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/PrimaryRayObject.h>
#include <optixu/optixpp.h>
#include <optix_math.h>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/list.hpp>

namespace ElVis
{
    class SampleVolumeSamplerObject : public PrimaryRayObject
    {
        public:
            friend class boost::serialization::access;
            ELVIS_EXPORT SampleVolumeSamplerObject();
            ELVIS_EXPORT explicit SampleVolumeSamplerObject(boost::shared_ptr<Object> obj);
            ELVIS_EXPORT virtual ~SampleVolumeSamplerObject();
            
        protected:
            virtual optixu::Material GetMaterial(SceneView* view);

        private:
            SampleVolumeSamplerObject& operator=(const SampleVolumeSamplerObject& rhs);
            ELVIS_EXPORT SampleVolumeSamplerObject(const SampleVolumeSamplerObject& rhs);

            static bool Initialized;
            static bool InitializeStatic();
            static void LoadPrograms(const std::string& prefix, optixu::Context context);

            template<typename Archive>
            void serialize(Archive& ar, const unsigned int version)
            {
                ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(PrimaryRayObject);
            }

            static optixu::Material Material;
            static optixu::Program ClosestHitProgram;
            static optixu::Program AnyHitProgram;
    };
}

#endif
