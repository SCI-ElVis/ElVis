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

#include <ElVis/Core/Scene.h>
#include <ElVis/Core/DirectionalLight.h>
#include <ElVis/Core/PointLight.h>
#include <ElVis/Core/Model.h>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/Util.hpp>
#include <ElVis/Core/ElVisConfig.h>
#include <ElVis/Core/OpenGL.h>
#include <ElVis/Core/Timer.h>
#include <ElVis/Core/HostTransferFunction.h>
#include <ElVis/Core/OptiXExtensions.hpp>
#include <ElVis/Core/Cuda.h>

#include <boost/foreach.hpp>
#include <boost/typeof/typeof.hpp>
#include <iostream>

#include <tinyxml.h>



namespace ElVis
{
    Scene::Scene() :
        m_allLights(),
        m_ambientLightColor(),
        m_model(),
        m_context(0),
        m_allPrimaryObjects(),
        m_optixStackSize(8000),
        m_colorMaps(),
        m_enableOptiXTrace(true),
        m_optiXTraceBufferSize(100000),
        m_optixTraceIndex(),
        m_enableOptiXExceptions(false),
        m_optixDataDirty(true),
        m_tracePixelDirty(true),
        m_enableTraceDirty(true),
        m_faceIntersectionProgram(),
//        m_planarFaceIntersectionProgram(),
        m_faceBoundingBoxProgram(),
        m_faceIdBuffer("FaceInfoBuffer"),
//        m_curvedFaceGeometry(0),
//        m_planarFaceGeometry(0),
        m_faceAcceleration()
    {
        m_optixTraceIndex.SetX(0);
        m_optixTraceIndex.SetY(0);
        m_optixTraceIndex.SetZ(-1);

        // For some reason in gcc, setting this in the constructor initialization list
        // doesn't work.
        m_enableOptiXExceptions = false;
        m_optixDataDirty = true;
    }

    Scene::~Scene()
    {
        m_context = 0;
    }

    void Scene::SetAmbientLightColor(const Color& value) 
    { 
        m_ambientLightColor = value; 
        if( !m_context.get() ) return;
        m_context["ambientColor"]->setFloat(m_ambientLightColor.Red(), m_ambientLightColor.Green(), m_ambientLightColor.Blue());
    }
    

    void Scene::SetOptixStackSize(int size)
    {
        if( size <= 0 ) return;
        m_optixStackSize = size;

        if( m_context )
        {
            m_context->setStackSize(m_optixStackSize);
            OnSceneChanged(*this);
        }
    }

    optixu::Context Scene::GetContext()
    {
        // Context is not valid without a model, as we don't know which extension specific code to load until 
        // the model is selected.
        if( !m_model ) return m_context;
        try
        {
            if( !m_context.get() )
            {
                GLenum err = glewInit();
                if( GLEW_OK != err  )
                {
                    std::cout << "Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
                }

                unsigned int deviceCount = 0;
                rtDeviceGetDeviceCount(&deviceCount);

                m_context = optixu::Context::create();

                PtxManager::SetupContext(m_model->GetPTXPrefix(), m_context);

                unsigned int optixVersion;
                rtGetVersion(&optixVersion);
                std::cout << "OptiX Version: " << optixVersion << std::endl;


                // Ray Type 0 - Primary rays that intersect actual geometry.  Closest
                // hit programs determine exactly how the geometry is handled.
                // Ray Type 1 - Rays that find the current element and evaluate the scalar
                // value at a point.
                // Ray Type 2 - Rays that perform volume rendering.
                m_context->setRayTypeCount(3);

                // Setup Lighting
                // TODO - Move this into the base
                // Overall goal will be to have an OptixScene, which handles setting up the
                // context and the lighting.  OptixSceneViews will allow different access to the
                // same scene.
                m_context["ambientColor"]->setFloat(m_ambientLightColor.Red(), m_ambientLightColor.Green(), m_ambientLightColor.Blue());

                std::list<boost::shared_ptr<DirectionalLight> > allDirectionalLights;
                std::list<boost::shared_ptr<PointLight> > allPointLights;
                std::cout << "Total Lights: " << m_allLights.size() << std::endl;
                for(std::list<boost::shared_ptr<Light> >::iterator iter = m_allLights.begin(); iter != m_allLights.end(); ++iter)
                {
                    BOOST_AUTO(asDirectional, boost::dynamic_pointer_cast<DirectionalLight>(*iter));
                    BOOST_AUTO(asPointLight, boost::dynamic_pointer_cast<PointLight>(*iter));

                    if( asDirectional )
                    {
                        allDirectionalLights.push_back(asDirectional);
                    }
                    else if (asPointLight)
                    {
                        allPointLights.push_back(asPointLight);
                    }
                }

                // Setup Directional Lights.


                // Setup Point Lights.
                optixu::Buffer lightPositionBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, allPointLights.size()*3);
                m_context["lightPosition"]->set(lightPositionBuffer);
                float* positionData = static_cast<float*>(lightPositionBuffer->map());

                optixu::Buffer lightColorBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, allPointLights.size()*3);
                m_context["lightColor"]->set(lightColorBuffer);
                float* colorData = static_cast<float*>(lightColorBuffer->map());

                int i = 0;
                for(std::list<boost::shared_ptr<PointLight> >::const_iterator iter = allPointLights.begin(); iter != allPointLights.end(); ++iter)
                {
                    positionData[i] = static_cast<float>((*iter)->Position().x());
                    positionData[i+1] = static_cast<float>((*iter)->Position().y());
                    positionData[i+2] = static_cast<float>((*iter)->Position().z());

                    colorData[i] = (*iter)->GetColor().Red();
                    colorData[i+1] = (*iter)->GetColor().Green();
                    colorData[i+2] = (*iter)->GetColor().Blue();
                    i += 3;
                }

                lightPositionBuffer->unmap();
                lightColorBuffer->unmap();


                if( GetModel() )
                {

                    GetModel()->CalculateExtents();
                    SetFloat(m_context["VolumeMinExtent"], GetModel()->MinExtent());
                    SetFloat(m_context["VolumeMaxExtent"], GetModel()->MaxExtent());

                    // Version 2.0 Interface.
                    GetModel()->CopyToOptiX(m_context);
                }

                m_context->setStackSize(m_optixStackSize);
                m_context->setPrintLaunchIndex(-1, -1, -1);

                //// Miss program
                //m_context->setMissProgram( 0, PtxManager::LoadProgram(m_context, "ElVis.cu.ptx", "miss" ) );
                //m_context["bg_color"]->setFloat( 1.0f, 1.0f, 1.0f );

                SynchronizeWithOptiXIfNeeded();
                OnSceneInitialized(*this);
            }
        }
        catch(optixu::Exception& e)
        {
            std::cout << "Exception encountered setting up the scene." << std::endl;
            std::cerr << e.getErrorString() << std::endl;
            std::cout << e.getErrorString().c_str() << std::endl;
        }
        return m_context;
    }

    void Scene::SynchronizeWithOptiXIfNeeded()
    {
        if( !m_context ) return;
        if( !m_optixDataDirty ) return;

        if( m_context->getPrintEnabled() != m_enableOptiXTrace )
        {
            m_context->setPrintEnabled(m_enableOptiXTrace);
        }

        if( m_context->getPrintBufferSize() != m_optiXTraceBufferSize )
        {
            m_context->setPrintBufferSize(m_optiXTraceBufferSize);
        }

        if( m_context->getExceptionEnabled(RT_EXCEPTION_ALL) != m_enableOptiXExceptions )
        {
            if( m_enableOptiXExceptions )
            {
                std::cout << "Enabling optix exceptions." << std::endl;
            }
            else
            {
                std::cout << "Disabling optix exceptions." << std::endl;
            }

            std::cout << "Setting exception flag to " << (m_enableOptiXExceptions ? "true" : "false") << std::endl;
            m_context->setExceptionEnabled(RT_EXCEPTION_ALL, m_enableOptiXExceptions);
        }

        if( m_tracePixelDirty )
        {
            m_context["TracePixel"]->setInt(m_optixTraceIndex.x(), m_optixTraceIndex.y());
            m_tracePixelDirty = false;
        }

        if( m_enableTraceDirty )
        {
            m_context["EnableTrace"]->setInt((m_enableOptiXTrace ? 1 : 0));
            m_enableTraceDirty = false;
        }

        m_optixDataDirty = false;
    }

    boost::shared_ptr<ColorMap> Scene::LoadColorMap(const boost::filesystem::path& p)
    {
        boost::shared_ptr<ColorMap> result;
        if( p.extension() != ".xml" )
        {
            return result;
        }

        tinyxml::TiXmlDocument doc(p.string().c_str());
        bool loadOkay = doc.LoadFile();

        if( !loadOkay )
        {
            std::cout << "Unable to load file " << p.string() << std::endl;
            return result;
        }

        tinyxml::TiXmlHandle docHandle(&doc);
        tinyxml::TiXmlNode* node = 0;
        tinyxml::TiXmlElement* rootElement = doc.FirstChildElement("ColorMap");

        if( !rootElement )
        {
            std::cout << "Not a color map xml file" << std::endl;
            return result;
        }

        const char* name = rootElement->Attribute("name");
        const char* colorSpace = rootElement->Attribute("space");

        if( !name )
        {
            std::cout << "No color map name. " << std::endl;
            return result;
        }

        std::string colorMapName(name);

        if( m_colorMaps.find(colorMapName) != m_colorMaps.end() )
        {
            std::cout << "Color map already exists." << std::endl;
            return (*m_colorMaps.find(colorMapName)).second.Map;
            //return result;
        }

        ColorMapInfo info;
        info.Map = boost::shared_ptr<PiecewiseLinearColorMap>(new PiecewiseLinearColorMap());
        info.Path = p;
        info.Name = colorMapName;

        tinyxml::TiXmlElement* pointElement = rootElement->FirstChildElement("Point");

        while( pointElement )
        {
            float scalar, r, g, b, o;
            int scalarResult = pointElement->QueryFloatAttribute("x", &scalar);
            int rResult = pointElement->QueryFloatAttribute("r", &r);
            int gResult = pointElement->QueryFloatAttribute("g", &g);
            int bResult = pointElement->QueryFloatAttribute("b", &b);
            int oResult = pointElement->QueryFloatAttribute("o", &o);

            if( rResult == tinyxml::TIXML_SUCCESS &&
                gResult == tinyxml::TIXML_SUCCESS &&
                bResult == tinyxml::TIXML_SUCCESS &&
                oResult == tinyxml::TIXML_SUCCESS &&
                scalarResult == tinyxml::TIXML_SUCCESS)
            {
                Color c(r, g, b, o);
                info.Map->SetBreakpoint(scalar, c);
            }

            pointElement = pointElement->NextSiblingElement("Point");
        }

        if( !info.Map->IsValid() )
        {
            std::cout << "Transfer function is not valid." << std::endl;
            return result;
        }

        m_colorMaps[colorMapName] = info;
        OnColorMapAdded(info);
        return info.Map;
    }

    Scene::ColorMapInfo::ColorMapInfo() :
        Map(),
        Path(),
        Name()
    {}

    Scene::ColorMapInfo::ColorMapInfo(const Scene::ColorMapInfo& rhs) :
        Map(rhs.Map),
        Path(rhs.Path),
        Name(rhs.Name)
    {
    }

    Scene::ColorMapInfo& Scene::ColorMapInfo::operator=(const Scene::ColorMapInfo& rhs)
    {
        Map = rhs.Map;
        Path = rhs.Path;
        Name = rhs.Name;
        return *this;
    }

    boost::shared_ptr<PiecewiseLinearColorMap> Scene::GetColorMap(const std::string& name) const
    {
        std::map<std::string, ColorMapInfo>::const_iterator found = m_colorMaps.find(name);
        if( found != m_colorMaps.end())
        {
            return (*found).second.Map;
        }
        else
        {
            return boost::shared_ptr<PiecewiseLinearColorMap>();
        }
    }

    void Scene::SetEnableOptixTrace(bool newValue)
    {
        if( m_enableOptiXTrace == newValue ) return;

        m_enableOptiXTrace = newValue;

        if( m_context )
        {
            m_context["EnableTrace"]->setInt((m_enableOptiXTrace ? 1 : 0));
        }
        m_optixDataDirty = true;
        m_enableTraceDirty = true;
        OnSceneChanged(*this);
        OnEnableTraceChanged(newValue);
    }

    void Scene::SetOptixTracePixelIndex(const Point<int, TwoD>& newValue)
    {
        if( m_optixTraceIndex == newValue ) return;

        m_optixTraceIndex = newValue;
        m_optixDataDirty = true;
        m_tracePixelDirty = true;
        OnSceneChanged(*this);
    }


    void Scene::SetOptixTraceBufferSize(int newValue)
    {
        if( m_optiXTraceBufferSize == newValue ) return;

        m_optiXTraceBufferSize = newValue;
        OnOptixPrintBufferSizeChanged(newValue);
        m_optixDataDirty = true;
        OnSceneChanged(*this);
    }


}
