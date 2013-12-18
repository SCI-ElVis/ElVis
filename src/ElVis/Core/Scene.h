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
#include <stdexcept>
#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/Color.h>
#include <ElVis/Core/ColorMap.h>
#include <ElVis/Core/Object.h>
#include <ElVis/Core/PrimaryRayObject.h>
#include <ElVis/Core/HostTransferFunction.h>
#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/FaceInfo.h>
#include <ElVis/Core/OptiXBuffer.hpp>
#include <ElVis/Core/Plugin.h>
#include <optixu/optixpp.h>

#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/signal.hpp>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/string.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <QDir>

namespace ElVis
{
    class Model;
    class Light;
    class HostTransferFunction;

    /// \brief The scene represents the data to be visualized, independent 
    /// of the specific visualization algorithms.  OptiX data structures can 
    /// be included in the scene.
    class Scene : public boost::enable_shared_from_this<Scene>
    {
        public:
            friend class boost::serialization::access;
            struct ColorMapInfo
            {
                friend class boost::serialization::access;
                ELVIS_EXPORT ColorMapInfo();
                ELVIS_EXPORT ColorMapInfo(const ColorMapInfo& rhs);
                ELVIS_EXPORT ColorMapInfo& operator=(const ColorMapInfo& rhs);

                boost::shared_ptr<PiecewiseLinearColorMap> Map;
                boost::filesystem::path Path;
                std::string Name;
                PiecewiseLinearColorMap Map1;
            private:
                template<typename Archive>
                void serialize(Archive& ar, const unsigned int version)
                {
                    ar & BOOST_SERIALIZATION_NVP(Map1);
                    //ar & BOOST_SERIALIZATION_NVP(Path);
                    ar & BOOST_SERIALIZATION_NVP(Name);
                }
            };

        public:
            ELVIS_EXPORT Scene();
            ELVIS_EXPORT ~Scene();

            ELVIS_EXPORT void AddLight(boost::shared_ptr<Light> value) { m_allLights.push_back(value); }
            ELVIS_EXPORT const std::list<boost::shared_ptr<Light>>& GetLights() const { return m_allLights; }

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

            //ELVIS_EXPORT OptiXBuffer<FaceInfo>& GetFaceInfoBuffer() { return m_faceIdBuffer; }

//            ELVIS_EXPORT optixu::Geometry GetCurvedFaceGeometry() const { return m_curvedFaceGeometry; }
//            ELVIS_EXPORT optixu::Geometry GetPlanarFaceGeometry() const { return m_planarFaceGeometry; }
            ELVIS_EXPORT optixu::Geometry GetFaceGeometry() const { return m_faceGeometry; }
            ELVIS_EXPORT optixu::Acceleration GetFaceAcceleration() const { return m_faceAcceleration; }

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

            template<typename Archive>
            void do_serialize(Archive& ar, const unsigned int version, 
                typename boost::enable_if<typename Archive::is_saving>::type* p = 0)
            {
                // On output, write the path to the model.  If possible, make 
                // it relative to the execution directory for maximum portability.
                BOOST_AUTO(path, m_model->GetPath());

                QDir dir;
                std::string relativeModelPath = dir.relativeFilePath(QString(path.c_str())).toStdString();
                ar & BOOST_SERIALIZATION_NVP(relativeModelPath);   

                BOOST_AUTO(pluginName, m_model->GetPlugin()->GetName());
                ar & BOOST_SERIALIZATION_NVP(pluginName);
            }

            template<typename Archive>
            void do_serialize(Archive& ar, const unsigned int version, 
                typename boost::enable_if<typename Archive::is_loading>::type* p = 0)
            {
                // On input, if there is a model defined, load it.  We will need the
                // plugin pointer as well.  Get it by name, and assume it is already
                // loaded.  Future iterations can relax this restriction.


                // When serializing, we expect a model to already be loaded.
                if( m_model )
                {
                    throw new std::runtime_error("Can't load state with a model already loaded.");
                }

                
                m_optixDataDirty = true;
                m_tracePixelDirty = true;
                m_enableTraceDirty = true;
                OnSceneChanged(*this);
            }

            template<typename Archive>
            void serialize(Archive& ar, const unsigned int version)
            {
                ar & BOOST_SERIALIZATION_NVP(m_allLights);
                ar & BOOST_SERIALIZATION_NVP(m_ambientLightColor);
                ar & BOOST_SERIALIZATION_NVP(m_allPrimaryObjects);
                ar & BOOST_SERIALIZATION_NVP(m_optixStackSize);

                ar & BOOST_SERIALIZATION_NVP(m_colorMaps);
                ar & BOOST_SERIALIZATION_NVP(m_enableOptiXTrace);
                ar & BOOST_SERIALIZATION_NVP(m_optiXTraceBufferSize);
                ar & BOOST_SERIALIZATION_NVP(m_optixTraceIndex);
                ar & BOOST_SERIALIZATION_NVP(m_enableOptiXExceptions);
                ar & BOOST_SERIALIZATION_NVP(m_enableOptiXExceptions);
                ar & BOOST_SERIALIZATION_NVP(m_enableOptiXExceptions);
                ar & BOOST_SERIALIZATION_NVP(m_enableOptiXExceptions);

                do_serialize(ar, version);
            }

            std::list<boost::shared_ptr<Light>> m_allLights;
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
            OptiXBuffer<FaceInfo> m_faceIdBuffer;
            optixu::Geometry m_faceGeometry;
            optixu::Acceleration m_faceAcceleration;
    };
}

#endif //ELVIS_ELVIS_NATIVE_SCENE_H
