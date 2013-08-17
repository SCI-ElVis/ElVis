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

#ifndef ELVIS_ELVIS_NATIVE_SCENE_H
#define ELVIS_ELVIS_NATIVE_SCENE_H

#include <list>
#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/Color.h>
#include <ElVis/Core/ColorMap.h>
#include <ElVis/Core/Object.h>
#include <ElVis/Core/PrimaryRayObject.h>
#include <ElVis/Core/HostTransferFunction.h>
#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/FaceDef.h>
#include <ElVis/Core/OptiXBuffer.hpp>

#include <optixu/optixpp.h>

#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/signal.hpp>

namespace ElVis
{
    class Model;
    class Light;
    class HostTransferFunction;

    class Scene
    {
        public:
            struct ColorMapInfo
            {
                ELVIS_EXPORT ColorMapInfo();
                ELVIS_EXPORT ColorMapInfo(const ColorMapInfo& rhs);
                ELVIS_EXPORT ColorMapInfo& operator=(const ColorMapInfo& rhs);

                boost::shared_ptr<PiecewiseLinearColorMap> Map;
                boost::filesystem::path Path;
                std::string Name;
            };

        public:
            ELVIS_EXPORT Scene();
            ELVIS_EXPORT ~Scene();

            ELVIS_EXPORT void AddLight(Light* value) { m_allLights.push_back(value); }
            ELVIS_EXPORT const std::list<Light*>& GetLights() const { return m_allLights; }

            ELVIS_EXPORT const Color& AmbientLightColor() const { return m_ambientLightColor; }
            ELVIS_EXPORT void SetAmbientLightColor(const Color& value) ;
                        
            ELVIS_EXPORT void SetModel(boost::shared_ptr<Model> value) { m_model = value; OnModelChanged(value); }
            ELVIS_EXPORT boost::shared_ptr<Model> GetModel() const { return m_model; }

            ELVIS_EXPORT optixu::Context GetContext();

            ELVIS_EXPORT void SetOptixStackSize(int size);
            ELVIS_EXPORT int GetOptixStackSize() const { return m_optixStackSize; }

            ELVIS_EXPORT void AddPrimaryRayObject(boost::shared_ptr<PrimaryRayObject> value) { m_allPrimaryObjects.push_back(value); }
            ELVIS_EXPORT const std::list<boost::shared_ptr<PrimaryRayObject> >& GetPrimaryRayObjects() { return m_allPrimaryObjects; }

            ELVIS_EXPORT boost::shared_ptr<ColorMap> LoadColorMap(const boost::filesystem::path& p);
            ELVIS_EXPORT boost::shared_ptr<PiecewiseLinearColorMap> GetColorMap(const std::string& name) const;
            ELVIS_EXPORT const std::map<std::string, ColorMapInfo>& GetColorMaps() const { return m_colorMaps; }

            ELVIS_EXPORT void SetEnableOptixTrace(bool newValue);
            ELVIS_EXPORT bool GetEnableOptixTrace() const { return m_enableOptiXTrace; }

            ELVIS_EXPORT void SetOptixTracePixelIndex(const Point<int, TwoD>& newValue);
            ELVIS_EXPORT const Point<int, TwoD>& GetOptixTracePixelIndex() const { return m_optixTraceIndex; }

            ELVIS_EXPORT void SetOptixTraceBufferSize(int newValue);
            ELVIS_EXPORT int GetOptixTraceBufferSize() const { return m_optiXTraceBufferSize; }

            ELVIS_EXPORT void SynchronizeWithOptiXIfNeeded();

//            ELVIS_EXPORT optixu::Program GetNewtonIntersectionProgram() const { return m_newtonIntersectionProgram; }
            ELVIS_EXPORT optixu::Buffer GetFaceIdBuffer() const { return m_faceIdBuffer; }
            ELVIS_EXPORT OptiXBuffer<ElVisFloat3>& GetFaceMinExtentBuffer() { return m_faceMinExtentBuffer; }
            ELVIS_EXPORT OptiXBuffer<ElVisFloat3>& GetFaceMaxExtentBuffer() { return m_faceMaxExtentBuffer; }

//            ELVIS_EXPORT optixu::Geometry GetCurvedFaceGeometry() const { return m_curvedFaceGeometry; }
//            ELVIS_EXPORT optixu::Geometry GetPlanarFaceGeometry() const { return m_planarFaceGeometry; }
            ELVIS_EXPORT optixu::Geometry GetFaceGeometry() const { return m_faceGeometry; }
            ELVIS_EXPORT optixu::Acceleration GetFaceAcceleration() const { return m_faceAcceleration; }

            ELVIS_EXPORT optixu::Buffer GetFacesEnabledBuffer() { return m_facesEnabledBuffer; }

            boost::signal<void (const ColorMapInfo&)> OnColorMapAdded;
            boost::signal< void (boost::shared_ptr<Model>) > OnModelChanged;
            boost::signal< void (const Scene&)> OnSceneInitialized;
            boost::signal<void (const Scene&)> OnSceneChanged;
            boost::signal<void (int)> OnOptixPrintBufferSizeChanged;
            boost::signal<void (bool)> OnEnableTraceChanged;

        private:
            Scene(const Scene& rhs);
            Scene& operator=(const Scene& rhs);

            void InitializeFaces();
            void Get3DModelInformation();

            std::list<Light*> m_allLights;
            Color m_ambientLightColor;
            boost::shared_ptr<Model> m_model;
            optixu::Context m_context;

            std::list<boost::shared_ptr<PrimaryRayObject> > m_allPrimaryObjects;

            int m_optixStackSize;

            std::map<std::string, ColorMapInfo> m_colorMaps;

            bool m_enableOptiXTrace;
            int m_optiXTraceBufferSize;
            Point<int, TwoD> m_optixTraceIndex;
            bool m_enableOptiXExceptions;
            bool m_optixDataDirty;
            bool m_tracePixelDirty;
            bool m_enableTraceDirty;

            // Optix variables and programs for use in the newton intersection program.
            optixu::Program m_faceBoundingBoxProgram;
            optixu::Program m_faceIntersectionProgram;
            optixu::Buffer m_faceIdBuffer;
            OptiXBuffer<ElVisFloat3> m_faceMinExtentBuffer;
            OptiXBuffer<ElVisFloat3> m_faceMaxExtentBuffer;
            optixu::Geometry m_faceGeometry;
            optixu::Acceleration m_faceAcceleration;
            optixu::Buffer m_facesEnabledBuffer;
    };
}

#endif //ELVIS_ELVIS_NATIVE_SCENE_H
