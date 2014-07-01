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

#ifndef ELVIS_COLOR_MAPPER_MODULE_H
#define ELVIS_COLOR_MAPPER_MODULE_H

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/RenderModule.h>
#include <ElVis/Core/PrimaryRayObject.h>
#include <ElVis/Core/RayGeneratorProgram.h>
#include <ElVis/Core/ColorMap.h>
#include <ElVis/Core/OptiXExtensions.hpp>
#include <vector>
#include <boost/signals2.hpp>

namespace ElVis
{
    // Assumes SampleBuffer has been populated with values.
    // Defines the color map itself
    // Outputs to color_buffer and raw_color_buffer.
    class ColorMapperModule : public RenderModule
    {
        public:
            ELVIS_EXPORT ColorMapperModule();
            ELVIS_EXPORT virtual ~ColorMapperModule() {}

            ELVIS_EXPORT void SetColorMap(boost::shared_ptr<ColorMap> map);

            ELVIS_EXPORT boost::shared_ptr<ColorMap> GetColorMap() { return m_colorMap; }

        protected:
            ELVIS_EXPORT virtual void DoSetup(SceneView* view);
            ELVIS_EXPORT virtual void DoSynchronize(SceneView* view);
            ELVIS_EXPORT virtual void DoRender(SceneView* view); 

            virtual int DoGetNumberOfRequiredEntryPoints() { return 1; }
            virtual std::string DoGetName() const { return "Color Mapper"; }

        private:
            ColorMapperModule(const ColorMapperModule& rhs);
            ColorMapperModule& operator=(const ColorMapperModule& rhs);

            void HandleMinOrMaxChanged(float value);
            void HandleColorMapChanged(const ColorMap& rhs);

            boost::shared_ptr<ColorMap> m_colorMap;
            RayGeneratorProgram m_program;
            optixu::Buffer m_data;
            optixu::TextureSampler m_textureSampler;
            unsigned int m_size;
            float m_min;
            float m_max;
            bool m_dirty;
            boost::signals2::connection m_minConnection;
            boost::signals2::connection m_maxConnection;
            boost::signals2::connection m_changedConnection;
    };
}


#endif 
