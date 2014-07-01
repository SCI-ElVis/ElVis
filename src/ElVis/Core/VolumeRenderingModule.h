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

#ifndef ELVIS_VOLUME_RENDERING_MODULE_H
#define ELVIS_VOLUME_RENDERING_MODULE_H

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/RenderModule.h>
#include <ElVis/Core/RenderModule.h>
#include <ElVis/Core/PrimaryRayObject.h>
#include <ElVis/Core/RayGeneratorProgram.h>
#include <ElVis/Core/OpenGL.h>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/TransferFunction.h>
#include <ElVis/Core/HostTransferFunction.h>
#include <ElVis/Core/OptiXBuffer.hpp>

#include <vector>

namespace ElVis
{
    enum VolumeRenderingIntegrationType
    {
        eRiemann_SingleThreadPerRay = 0,
        eTrapezoidal_SingleThreadPerRay = 1,
        eIntegrateSegment_FullAlgorithm = 7
    };


    class VolumeRenderingModule : public RenderModule
    {
        public:
            ELVIS_EXPORT VolumeRenderingModule();
            ELVIS_EXPORT virtual ~VolumeRenderingModule() {}

            ELVIS_EXPORT void SetIntegrationType(VolumeRenderingIntegrationType type);
            VolumeRenderingIntegrationType GetIntegrationType() const { return m_segmentIntegrationType; }

            boost::shared_ptr<HostTransferFunction> GetTransferFunction() { return m_transferFunction; }

            ELVIS_EXPORT void SetCompositingStepSize(ElVisFloat value);
            ElVisFloat GetCompositingStepSize() const { return m_compositingStepSize; }

            ELVIS_EXPORT void SetEpsilon(ElVisFloat value);
            ElVisFloat GetEpsilon() const { return m_epsilon; }

            ELVIS_EXPORT void WriteAccumulatedDensityBuffer(const std::string& fileName, SceneView* view);

            ELVIS_EXPORT void SetTrackNumberOfSamples(bool value);
            ELVIS_EXPORT bool GetTrackNumberOfSamples() const { return m_enableSampleTracking; }

            void SetRenderIntegrationType(bool value) { m_renderIntegrationType = value; }

            void SetEnableEmptySpaceSkipping(bool value) { m_enableEmptySpaceSkipping = value; }

            boost::signals2::signal<void (const RenderModule&, VolumeRenderingIntegrationType)> OnIntegrationTypeChanged;

        protected:
            ELVIS_EXPORT virtual void DoRender(SceneView* view);
            ELVIS_EXPORT virtual void DoSetup(SceneView* view);
            ELVIS_EXPORT virtual void DoEvaluateSegment(SceneView* view);

            virtual int DoGetNumberOfRequiredEntryPoints() { return 2; }
            virtual void DoResize(unsigned int newWidth, unsigned int newHeight) {}
            virtual std::string DoGetName() const { return "Volume Rendering"; }

        private:
            VolumeRenderingModule& operator=(const VolumeRenderingModule& rhs);
            VolumeRenderingModule(const VolumeRenderingModule& rhs);
            
            static bool InitializeStatic();
            static bool Initialized;

            //virtual void DoAfterSetup(SceneView* view);

            void IntegrateSegment(SceneView* view);
            void IntegrateSegmentSingleThreadPerRayRiemann(SceneView* view);
            void IntegrateSingleThreadPerRayFull(SceneView* view);
            void IntegrateSingleWarpPerSegmentFull(SceneView* view);
            void IntegrateSegmentSingleThreadPerRay(SceneView* view);
            void IntegrateGKOnly(SceneView* view);
            void IntegrateTrapezoidal_SingleThreadPerRay(SceneView* view);

            void HandleTransferFunctionChanged();

            ///////////////////////////////////////////////////////
            // The following methods are primarily test methods to see how a user-defined
            // he over each entire segment performs when compared to adaptively sampling
            // in warp sized chunks.  Both cases use error estimation, it's just that
            // the first one cuts sample spacing in half globally when the error isn't
            // good enough, while the second tries to be more adaptive.
            void IntegrateDensity_NonAdaptiveTrapezoidal_SingleBlockPerRay(SceneView* view);
            void IntegrateDensity_AdaptiveTrapezoidal_SingleBlockPerRay(SceneView* view);

            ///////////////////////////////////////////////////////////
            // the following methods are candidates for the final code.
            ///////////////////////////////////////////////////////////
            void ResetSampleCount();

            OptiXBuffer<ElVisFloat> DensityBreakpoints;
            OptiXBuffer<ElVisFloat> RedBreakpoints;
            OptiXBuffer<ElVisFloat> GreenBreakpoints;
            OptiXBuffer<ElVisFloat> BlueBreakpoints;

            OptiXBuffer<ElVisFloat> DensityValues;
            OptiXBuffer<ElVisFloat> RedValues;
            OptiXBuffer<ElVisFloat> GreenValues;
            OptiXBuffer<ElVisFloat> BlueValues;

            RayGeneratorProgram m_program;
            static RayGeneratorProgram m_PerformVolumeRendering;

            // Element by Element Approach
            VolumeRenderingIntegrationType m_segmentIntegrationType;

            bool m_enableSampleTracking;
            boost::shared_ptr<HostTransferFunction> m_transferFunction;
            ElVisFloat m_compositingStepSize;
            ElVisFloat m_epsilon;
            bool m_renderIntegrationType;
            bool m_enableEmptySpaceSkipping;
            bool m_initializationComplete;


    };
}


#endif
