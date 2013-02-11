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

#ifndef ELVIS_NATIVE_LIGHTING_MODULE_H
#define ELVIS_NATIVE_LIGHTING_MODULE_H


#include <ElVis/Core/RenderModule.h>
#include <ElVis/Core/OpenGL.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/ColorMap.h>
#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/RayGeneratorProgram.h>

#include <optixu/optixpp.h>

namespace ElVis
{
    class LightingModule : public RenderModule
    {
        public:
            ELVIS_EXPORT LightingModule();
            ELVIS_EXPORT LightingModule(const LightingModule& rhs);
            ELVIS_EXPORT virtual ~LightingModule();
            
        protected:
            virtual void DoRender(SceneView* view); 
            virtual void DoSetup(SceneView* view); 
            virtual int DoGetNumberOfRequiredEntryPoints() 
            {
                // 7 Ray programs in the CutSurfceColorMap object
                // 1 in the color map.
                return 1; 
            }
            virtual void DoResize(unsigned int newWidth, unsigned int newHeight) {}
            virtual std::string DoGetName() const  { return "Lighting Module"; }

        private:
            const LightingModule& operator=(const LightingModule& rhs);            
            RayGeneratorProgram m_program;
    };
}

#endif //CUT_SURFACE_COLOR_MAP_H
