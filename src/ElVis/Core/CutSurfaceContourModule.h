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

#ifndef ELVIS_CUT_SURFACE_CONTOUR_MODULE_H
#define ELVIS_CUT_SURFACE_CONTOUR_MODULE_H

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/RenderModule.h>
#include <ElVis/Core/PrimaryRayObject.h>
#include <ElVis/Core/RayGeneratorProgram.h>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/OptiXBuffer.hpp>

#include <boost/signals2.hpp>

#include <set>
#include <vector>

namespace ElVis
{
    class CutSurfaceContourModule : public RenderModule
    {
        public:
            ELVIS_EXPORT CutSurfaceContourModule();
            ELVIS_EXPORT virtual ~CutSurfaceContourModule() {}

            ELVIS_EXPORT void AddObject(boost::shared_ptr<PrimaryRayObject> obj) { m_objects.push_back(obj); }

            ELVIS_EXPORT void AddIsovalue(ElVisFloat newVal);
            ELVIS_EXPORT void RemoveIsovalue(ElVisFloat newVal);

            ELVIS_EXPORT int GetTreatElementBoundariesAsDiscontinuous() const;
            ELVIS_EXPORT void SetTreatElementBoundariesAsDiscontinuous(int value);

            ELVIS_EXPORT int GetMatchVisual3Contours() const;
            ELVIS_EXPORT void SetMatchVisual3Contours(int newValue);

            boost::signals2::signal<void (ElVisFloat)> OnIsovalueAdded;
            boost::signals2::signal<void (ElVisFloat)> OnIsovalueRemoved;

        protected:
            ELVIS_EXPORT virtual void DoSetup(SceneView* view);
            ELVIS_EXPORT virtual void DoSynchronize(SceneView* view);
            ELVIS_EXPORT virtual void DoRender(SceneView* view);

            virtual int DoGetNumberOfRequiredEntryPoints() { return 2; }
            ELVIS_EXPORT virtual void DoResize(unsigned int newWidth, unsigned int newHeight);
            virtual std::string DoGetName() const;

        private:
            CutSurfaceContourModule(const CutSurfaceContourModule& rhs);
            CutSurfaceContourModule& operator=(const CutSurfaceContourModule& rhs);

            static const std::string ReferencePointAtIntersectionBufferName;

            OptiXBuffer<ElVisFloat> m_contourSampleBuffer;
            OptiXBuffer<ElVisFloat> m_isovalueBuffer;
            OptiXBuffer<ElVisFloat3> m_referencePointAtIntersectionBuffer;
            optixu::Buffer m_elementIdAtIntersectionBuffer;
            optixu::Buffer m_elementTypeAtIntersectionBuffer;
            std::vector<boost::shared_ptr<PrimaryRayObject> > m_objects;
            std::set<ElVisFloat> m_isovalues;
            RayGeneratorProgram m_sampleRayProgram;
            RayGeneratorProgram m_markPixelProgram;
            bool m_dirty;
            int m_treatElementBoundariesAsDiscontinuous;
            int m_matchVisual3Contours;

    };
}


#endif 
