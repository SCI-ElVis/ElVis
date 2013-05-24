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

#ifndef ELVIS_ISOSURFACE_MODULE_H
#define ELVIS_ISOSURFACE_MODULE_H

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/RenderModule.h>
#include <ElVis/Core/ElementTraversalModule.h>
#include <ElVis/Core/RayGeneratorProgram.h>
#include <ElVis/Core/Buffer.h>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/ElementId.h>

#include <set>

#include <boost/signals.hpp>

namespace ElVis
{
    class IsosurfaceModule : public ElementTraversalModule
    {
        public:
            ELVIS_EXPORT IsosurfaceModule();
            ELVIS_EXPORT virtual ~IsosurfaceModule() {}

            ELVIS_EXPORT virtual void DoRender(SceneView* view);

            ELVIS_EXPORT void AddIsovalue(const ElVisFloat& value);
            ELVIS_EXPORT void RemoveIsovalue(const ElVisFloat& value);

            ELVIS_EXPORT const std::set<ElVisFloat> GetIsovalues() const { return m_isovalues; }

            boost::signal< void (ElVisFloat) > OnIsovalueAdded;
            boost::signal< void (ElVisFloat, ElVisFloat)> OnIsovalueChanged;
            boost::signal< void (ElVisFloat)> OnIsovalueRemoved;

        protected:

            ELVIS_EXPORT virtual void DoSynchronize(SceneView* view);
            ELVIS_EXPORT virtual void DoSetupAfterInteropModule(SceneView* view);
            ELVIS_EXPORT virtual void DoEvaluateSegment(SceneView* view);
            ELVIS_EXPORT virtual bool HasWork() const { return m_isovalues.size() > 0; }

            virtual int DoGetNumberOfRequiredEntryPoints() { return 2; }
            virtual void DoResize(unsigned int newWidth, unsigned int newHeight);
            virtual std::string DoGetName() const { return "Isosurface Rendering"; }

        private:
            IsosurfaceModule& operator=(const IsosurfaceModule& rhs);
            IsosurfaceModule(const IsosurfaceModule& rhs);

            void SynchronizeWithOptix();
            static void ReadFloatVector(const std::string& fileName, std::vector<ElVisFloat>& values);
            static bool InitializeStaticForIsosuface();

            std::set<ElVisFloat> m_isovalues;
            unsigned int m_isovalueBufferSize;

            InteropBuffer<ElVisFloat> m_isovalueBuffer;
            InteropBuffer<ElVisFloat> m_gaussLegendreNodesBuffer;
            InteropBuffer<ElVisFloat> m_gaussLegendreWeightsBuffer;
            InteropBuffer<ElVisFloat> m_monomialConversionTableBuffer;

            CUfunction m_findIsosurfaceFunction;

            static RayGeneratorProgram m_ElementByElementVolumeTraversalInitProgramForIsosurface;
            static RayGeneratorProgram m_ElementByElementVolumeTraversalProgramForIsosurface;
            static RayGeneratorProgram m_FindIsosurface;
            static bool Initialized;

    };
}


#endif
