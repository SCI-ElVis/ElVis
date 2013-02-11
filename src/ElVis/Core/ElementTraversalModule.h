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

#ifndef ELVIS_OPTIX_CUDA_INTEROP_MODULE_H
#define ELVIS_OPTIX_CUDA_INTEROP_MODULE_H

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/RenderModule.h>
#include <ElVis/Core/PrimaryRayObject.h>
#include <ElVis/Core/RayGeneratorProgram.h>
#include <ElVis/Core/Buffer.h>
#include <ElVis/Core/OpenGL.h>
#include <ElVis/Core/InteropBuffer.hpp>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/TransferFunction.h>
#include <ElVis/Core/ElementId.h>

#include <vector>

#include <cuda.h>

namespace ElVis
{
    // This approach doesn't work with both volume rendering and isosurfacing inheriting.
    // We end up defining the same variables over again.  We need to either find out if the
    // variables have already been defined, or do something with statics, or change the approach.
    class ElementTraversalModule : public RenderModule
    {
        public:
            ELVIS_EXPORT ElementTraversalModule();
            ELVIS_EXPORT virtual ~ElementTraversalModule() {}

        protected:
            ELVIS_EXPORT virtual void DoSetup(SceneView* view);
            ELVIS_EXPORT virtual void DoRender(SceneView* view);

            ELVIS_EXPORT virtual void DoEvaluateSegment(SceneView* view) = 0;
            ELVIS_EXPORT virtual bool HasWork() const { return true; }

            virtual void DoSetupAfterInteropModule(SceneView* view) = 0;

            ELVIS_EXPORT int GetSomeSegmentsNeedToBeIntegrated();
            ELVIS_EXPORT void ResetSomeSegmentsNeedToBeIntegrated();

            RayGeneratorProgram& GetInitTraversalProgram() { return m_ElementByElementVolumeTraversalInitProgram; }
            RayGeneratorProgram& GetTraveralProgram() { return m_ElementByElementVolumeTraversalProgram; }
            InteropBuffer<int>& GetSegmentElementIdBuffer() { return m_SegmentElementIdBuffer; }
            InteropBuffer<int>& GetSegmentElementTypeBuffer() { return m_SegmentElementTypeBuffer; }
            InteropBuffer<ElVisFloat>& GetSegmentStartBuffer() { return m_SegmentStart; }
            InteropBuffer<ElVisFloat>& GetSegmentEndBuffer() { return m_SegmentEnd; }
            InteropBuffer<ElVisFloat3>& GetSegmentRayDirectionBuffer() {return  m_SegmentRayDirection;}
            InteropBuffer<int>& GetSegmentIdBuffer() { return m_SegmentIdBuffer; }
            void RunCopyElementIdKeyData(SceneView* view);
            InteropBuffer<ElementId>& GetElementSortBuffer() { return m_IdSortBuffer; }

            optixu::Buffer& GetSegmentsNeedToBeIntegratedBuffer() { return m_SomeSegmentsNeedToBeIntegrated; }

        private:
            ElementTraversalModule& operator=(const ElementTraversalModule& rhs);
            ElementTraversalModule(const ElementTraversalModule& rhs);

            static bool Initialized;
            static bool InitializeStatic();
            static void LoadPrograms(const std::string& prefix, optixu::Context context);

            // Element by Element Approach
            static RayGeneratorProgram m_ElementByElementVolumeTraversalInitProgram;
            static RayGeneratorProgram m_ElementByElementVolumeTraversalProgram;

            static InteropBuffer<int> m_SegmentElementIdBuffer;
            static InteropBuffer<int> m_SegmentElementTypeBuffer;
            static InteropBuffer<ElVisFloat> m_SegmentStart;
            static InteropBuffer<ElVisFloat> m_SegmentEnd;
            static InteropBuffer<ElVisFloat3> m_SegmentRayDirection;
            static InteropBuffer<int> m_SegmentIdBuffer;
            static InteropBuffer<ElementId> m_IdSortBuffer;
            static optixu::Buffer m_SomeSegmentsNeedToBeIntegrated;
            static CUfunction m_copyElementIdKeyData;
    };
}


#endif
