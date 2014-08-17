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

#ifndef ELVIS_CUT_SURFACE_MESH_MODULE_H
#define ELVIS_CUT_SURFACE_MESH_MODULE_H

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/RenderModule.h>
#include <ElVis/Core/PrimaryRayObject.h>
#include <ElVis/Core/RayGeneratorProgram.h>
#include <ElVis/Core/Float.h>

#include <boost/signals2.hpp>

#include <set>
#include <vector>

namespace ElVis
{
    class CutSurfaceMeshModule : public RenderModule
    {
        public:
            ELVIS_EXPORT CutSurfaceMeshModule();
            ELVIS_EXPORT virtual ~CutSurfaceMeshModule() {}

        protected:
            ELVIS_EXPORT virtual void DoSetup(SceneView* view);
            ELVIS_EXPORT virtual void DoRender(SceneView* view);


            virtual int DoGetNumberOfRequiredEntryPoints() { return 1; }
            virtual std::string DoGetName() const;

        private:
            CutSurfaceMeshModule(const CutSurfaceMeshModule& rhs);
            CutSurfaceMeshModule& operator=(const CutSurfaceMeshModule& rhs);

            static const std::string ProgramName;
            static const std::string CategorizeProgramName;

            RayGeneratorProgram m_program;
            RayGeneratorProgram m_categorizeProgram;
    };
}


#endif
