////////////////////////////////////////////////////////////////////////////////
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description:
//
////////////////////////////////////////////////////////////////////////////////


#include <ElVis/Gui/ApplicationState.h>
#include <ElVis/Core/PrimaryRayModule.h>
#include <ElVis/Core/ColorMapperModule.h>
#include <ElVis/Core/IsosurfaceModule.h>
#include <ElVis/Core/VolumeRenderingModule.h>
#include <ElVis/Core/LightingModule.h>
#include <ElVis/Core/PointLight.h>
#include <ElVis/Core/Color.h>
#include <ElVis/Core/HostTransferFunction.h>
#include <ElVis/Core/CutSurfaceContourModule.h>
#include <ElVis/Core/CutSurfaceMeshModule.h>
#include <ElVis/Core/TwoDPrimaryElements.h>
#include <ElVis/Core/TwoDPrimaryElementsPrimaryObject.h>
#include <ElVis/Core/Cylinder.h>
#include <ElVis/Core/SampleVolumeSamplerObject.h>
#include <boost/bind.hpp>
#include <boost/make_shared.hpp>

//#define OFFSET -.1

// nice
#define OFFSET -.005

//#define OFFSET -.045

namespace ElVis
{

    namespace Gui
    {

        ElVisFloat Temp(ElVisFloat value)
        {
            return value*2.703 - .703 + .5;
        }

        ApplicationState::ApplicationState() :
            m_scene(boost::make_shared<ElVis::Scene>()),
            m_surfaceSceneView(new SceneView()),
            m_selectedObject(),
            m_selectedTransferFunction(),
            m_colorMapperModule(),
            m_cutSurfaceContourModule(),
            m_cutSurfaceMeshModule(),
            m_selectedPointInfo(),
            m_minConnection(),
            m_maxConnection(),
            m_showLookAtPoint(true),
            m_showBoundingBox(true),
            m_volumeRenderingModule(),
            m_primaryRayModule(),
            m_isosurfaceModule()
        {
            m_surfaceSceneView->SetScene(m_scene);

            {
                m_primaryRayModule.reset(new PrimaryRayModule());
                m_surfaceSceneView->AddRenderModule(m_primaryRayModule);
                m_surfaceSceneView->OnSceneViewChanged.connect(boost::bind(&RenderModule::SetRenderRequired, m_primaryRayModule));
                m_scene->OnSceneChanged.connect(boost::bind(&RenderModule::SetRenderRequired, m_primaryRayModule));

                m_cutSurfaceContourModule.reset(new CutSurfaceContourModule());
                m_cutSurfaceContourModule->SetEnabled(false);
                //m_cutSurfaceContourModule->AddIsovalue(0);
                m_surfaceSceneView->AddRenderModule(m_cutSurfaceContourModule);
                m_primaryRayModule->OnRenderFlagsChanged.connect(boost::bind(&RenderModule::SetRenderRequired, m_cutSurfaceContourModule));
                m_cutSurfaceContourModule->OnRenderFlagsChanged.connect(boost::bind(&RenderModule::SetRenderRequired, m_primaryRayModule));


                m_cutSurfaceMeshModule.reset(new CutSurfaceMeshModule());
                m_cutSurfaceMeshModule->SetEnabled(false);
                m_surfaceSceneView->AddRenderModule(m_cutSurfaceMeshModule);
                m_primaryRayModule->OnRenderFlagsChanged.connect(boost::bind(&RenderModule::SetRenderRequired, m_cutSurfaceMeshModule));
                m_cutSurfaceMeshModule->OnRenderFlagsChanged.connect(boost::bind(&RenderModule::SetRenderRequired, m_primaryRayModule));


                m_isosurfaceModule.reset(new ElVis::IsosurfaceModule());
                m_surfaceSceneView->AddRenderModule(m_isosurfaceModule);
                m_isosurfaceModule->SetEnabled(false);
                m_primaryRayModule->OnRenderFlagsChanged.connect(boost::bind(&RenderModule::SetRenderRequired, m_isosurfaceModule));
                m_isosurfaceModule->OnRenderFlagsChanged.connect(boost::bind(&RenderModule::SetRenderRequired, m_primaryRayModule));


                m_colorMapperModule = boost::shared_ptr<ElVis::ColorMapperModule>(new ElVis::ColorMapperModule());
                m_surfaceSceneView->AddRenderModule(m_colorMapperModule);
                m_primaryRayModule->OnRenderFlagsChanged.connect(boost::bind(&RenderModule::SetRenderRequired, m_colorMapperModule));
                m_isosurfaceModule->OnRenderFlagsChanged.connect(boost::bind(&RenderModule::SetRenderRequired, m_colorMapperModule));
                m_cutSurfaceContourModule->OnRenderFlagsChanged.connect(boost::bind(&RenderModule::SetRenderRequired, m_colorMapperModule));
                m_cutSurfaceMeshModule->OnRenderFlagsChanged.connect(boost::bind(&RenderModule::SetRenderRequired, m_colorMapperModule));


                boost::shared_ptr<LightingModule> lightingModule(new LightingModule());
                lightingModule->SetEnabled(true);
                m_surfaceSceneView->AddRenderModule(lightingModule);
                m_primaryRayModule->OnRenderFlagsChanged.connect(boost::bind(&RenderModule::SetRenderRequired, lightingModule));
                m_isosurfaceModule->OnRenderFlagsChanged.connect(boost::bind(&RenderModule::SetRenderRequired, lightingModule));
                m_colorMapperModule->OnRenderFlagsChanged.connect(boost::bind(&RenderModule::SetRenderRequired, lightingModule));


                m_volumeRenderingModule.reset(new VolumeRenderingModule());
                m_volumeRenderingModule->SetEnabled(false);
                m_surfaceSceneView->AddRenderModule(m_volumeRenderingModule);
                m_volumeRenderingModule->SetIntegrationType(eRiemann_SingleThreadPerRay);
                m_surfaceSceneView->OnSceneViewChanged.connect(boost::bind(&RenderModule::SetRenderRequired, m_volumeRenderingModule));
                m_volumeRenderingModule->OnRenderFlagsChanged.connect(boost::bind(&RenderModule::SetRenderRequired, m_primaryRayModule));

                boost::shared_ptr<ElVis::HostTransferFunction> transferFunction = m_volumeRenderingModule->GetTransferFunction();

//#define isovalue .2
//#define step .02
//                // Delta Wing Mach Number
//                Color c0(.2, 0.0, .2, 0.0);
//                transferFunction->SetBreakpoint(isovalue-step, c0);

//                //Color c1(1.0, 0.0, 1.0, 100.0);
//                Color c1(1.0, 0.0, 1.0, 100.0);
//                transferFunction->SetBreakpoint(isovalue, c1);

//                Color c2(.2, 0.0, .2, 0.0);
//                transferFunction->SetBreakpoint(isovalue+step, c2);

                //////////////////////////////////////
                // Negative Jacobian Transfer Function.
                //////////////////////////////////////

                double color0 = -.4;
                double color1 = -.2;
                double color2 = -.6;

                Color c0(0.0, 0.0, 0.0, 0.0);
                transferFunction->SetBreakpoint(color0-.1, c0);

                //Color c1(1.0, 0.0, 1.0, 100.0);
                Color c1(1.0, 1.0, 0.0, 10.0);
                transferFunction->SetBreakpoint(color0, c1);

                Color c2(0.0, 0.0, 0.0, 0.0);
                transferFunction->SetBreakpoint(color0+.1, c2);


                Color c3(0.0, 0.0, 0.0, 0.0);
                transferFunction->SetBreakpoint(color1-.1, c3);

                Color c4(1.0, 0.0, 1.0, 20.0);
                transferFunction->SetBreakpoint(color1, c4);

                Color c5(0.0, 0.0, 0.0, 0.0);
                transferFunction->SetBreakpoint(color1+.1, c5);

                Color c6(0.0, 0.0, 0.0, 0.0);
                transferFunction->SetBreakpoint(color2-.1, c6);

                Color c7(0.0, 1.0, 1.0, 20.0);
                transferFunction->SetBreakpoint(color2, c7);

                Color c8(0.0, 0.0, 0.0, 0.0);
                transferFunction->SetBreakpoint(color2+.1, c8);

                //////////////////////////////////////
                // Bullet Transfer Function.
                //////////////////////////////////////
//                Color c0(0.0, 0.0, 0.0, 0.0);
//                transferFunction->SetBreakpoint(-.04, c0);

//                //Color c1(1.0, 0.0, 1.0, 100.0);
//                Color c1(1.0, 1.0, 0.0, 10.0);
//                transferFunction->SetBreakpoint(-.04+.005, c1);

//                Color c2(0.0, 0.0, 0.0, 0.0);
//                transferFunction->SetBreakpoint(-.04+.01, c2);


//                Color c3(0.0, 0.0, 0.0, 0.0);
//                transferFunction->SetBreakpoint(.01, c3);

//                Color c4(1.0, 0.0, 1.0, 20.0);
//                transferFunction->SetBreakpoint(.01+.005, c4);

//                Color c5(0.0, 0.0, 0.0, 0.0);
//                transferFunction->SetBreakpoint(.01+.01, c5);

//                Color c6(0.0, 0.0, 0.0, 0.0);
//                transferFunction->SetBreakpoint(-.0001, c6);

//                Color c7(0.0, 1.0, 1.0, 20.0);
//                transferFunction->SetBreakpoint(0.0, c7);

//                Color c8(0.0, 0.0, 0.0, 0.0);
//                transferFunction->SetBreakpoint(.0001, c8);

                //////////////////////////////////////
                // Paddle - attempt for performance with GK.  Appears to work, huge decrease in samples, but ugly image.
                //////////////////////////////////////
//                Color c0(0.0, 0.0, 0.0, 0.0);
//                transferFunction->SetBreakpoint(-.703, c0);

//                Color c1(1.0, 1.0, 0.0, 100.0);
//                transferFunction->SetBreakpoint(52.0/255.0*2.703 - .703, c1);

//                Color c2(0.0, 0.0, 0.0, 0.0);
//                transferFunction->SetBreakpoint(60.0/255.0*2.703 - .703, c2);

//                // Paddle - pretty image.
//                Color c0(0.0, 0.0, 0.0, 0.0);
//                transferFunction->SetBreakpoint(.03-OFFSET, c0);

//                Color c1(0.0, 1.0, 1.0, 10.0);
//                transferFunction->SetBreakpoint(.03, c1);

//                Color c2(0.0, 0.0, 0.0, 0.0);
//                transferFunction->SetBreakpoint(0.03+OFFSET, c2);


//                Color c3(0.0, 0.0, 0.0, 0.0);
//                transferFunction->SetBreakpoint(0.05-OFFSET, c3);

//                Color c4(1.0, 0.0, 1.0, 1.0);
//                transferFunction->SetBreakpoint(.05, c4);

//                Color c5(0.0, 0.0, 0.0, 0.0);
//                transferFunction->SetBreakpoint(0.05+OFFSET, c5);

//                Color c6(0.0, 0.0, 0.0, 0.0);
//                transferFunction->SetBreakpoint(0.1-OFFSET, c6);

//                Color c7(1.0, 1.0, 0.0, 10.0);
//                transferFunction->SetBreakpoint(0.1, c7);

//                Color c8(0.0, 0.0, 0.0, 0.0);
//                transferFunction->SetBreakpoint(0.1+OFFSET, c8);
            }

            ElVis::PointLight* l = new ElVis::PointLight();
            ElVis::Color lightColor;
            lightColor.SetRed(.2);
            lightColor.SetGreen(.2);
            lightColor.SetBlue(.2);

            ElVis::WorldPoint lightPos(10.0, 0.0, 0.0);
            l->SetColor(lightColor);
            l->SetPosition(lightPos);
            //m_scene->AddLight(l);

            ElVis::Color ambientColor;
            ambientColor.SetRed(1.0-86.0/255.0);
            ambientColor.SetGreen(1.0-86.0/255.0);
            ambientColor.SetBlue(1.0-86.0/255.0);
            m_scene->SetAmbientLightColor(ambientColor);
                                boost::shared_ptr<FaceObject> faceObject(new FaceObject(GetScene()));
                    m_faceSampler.reset(new SampleFaceObject(faceObject));
                    m_primaryRayModule->AddObject(m_faceSampler);

                    boost::shared_ptr<TwoDPrimaryElements> twoDObject(new TwoDPrimaryElements(GetScene()));
                    boost::shared_ptr<TwoDPrimaryElementsPrimaryObject> wrapper(new TwoDPrimaryElementsPrimaryObject(twoDObject));
                    m_primaryRayModule->AddObject(wrapper);

        }

        ApplicationState::~ApplicationState()
        {
        }

        boost::shared_ptr<HostTransferFunction> ApplicationState::GetHostTransferFunction()
        { 
            return m_volumeRenderingModule->GetTransferFunction(); 
        }

        void ApplicationState::SetShowLookAtPoint(bool newValue)
        {
            if( newValue != m_showLookAtPoint )
            {
                m_showLookAtPoint = newValue;
                OnApplicationStateChanged();
            }
        }

        void ApplicationState::SetShowBoundingBox(bool newValue)
        {
            if( newValue != m_showBoundingBox )
            {
                m_showBoundingBox = newValue;
                OnApplicationStateChanged();
            }
        }

        void ApplicationState::SetSelectedColorMap(boost::shared_ptr<PiecewiseLinearColorMap> f)
        {
            m_selectedTransferFunction = f;
            m_colorMapperModule->SetColorMap(f);
            OnSelectedTransferFunctionChanged(f);
            OnApplicationStateChanged();

            m_minConnection.disconnect();
            m_maxConnection.disconnect();
            m_maxConnection = f->OnMaxChanged.connect(boost::bind(&ApplicationState::HandleSelectedColorMapMinOrMaxChanged, this, _1));
            m_minConnection = f->OnMinChanged.connect(boost::bind(&ApplicationState::HandleSelectedColorMapMinOrMaxChanged, this, _1));
        }

        void ApplicationState::HandleSelectedColorMapMinOrMaxChanged(float value)
        {
            OnApplicationStateChanged();
        }

        void ApplicationState::SetLookAtPointToIntersectionPoint(unsigned int pixel_x, unsigned int pixel_y)
        {
            if( pixel_x < m_surfaceSceneView->GetWidth() && pixel_y < m_surfaceSceneView->GetHeight() )
            {
                if( m_surfaceSceneView->GetElementId(pixel_x, pixel_y) >= 0 )
                {
                    WorldPoint p = m_surfaceSceneView->GetIntersectionPoint(pixel_x, pixel_y);
                    m_surfaceSceneView->GetViewSettings()->SetLookAt(p);
                }
            }
            else
            {
                m_selectedPointInfo.Valid = false;
            }
            OnSelectedPointChanged(m_selectedPointInfo);
        }

        void ApplicationState::GeneratePointInformation(unsigned int pixel_x, unsigned int pixel_y)
        {
            if( pixel_x < m_surfaceSceneView->GetWidth() && pixel_y < m_surfaceSceneView->GetHeight() )
            {
                m_selectedPointInfo.Valid = true;
                m_selectedPointInfo.IntersectionPoint = m_surfaceSceneView->GetIntersectionPoint(pixel_x, pixel_y);
                m_selectedPointInfo.Normal = m_surfaceSceneView->GetNormal(pixel_x, pixel_y);
                m_selectedPointInfo.Pixel.SetX(pixel_x);
                m_selectedPointInfo.Pixel.SetY(pixel_y);

                m_scene->SetOptixTracePixelIndex(m_selectedPointInfo.Pixel);

                m_selectedPointInfo.ElementId = m_surfaceSceneView->GetElementId(pixel_x, pixel_y);
                m_selectedPointInfo.ElementTypeId = m_surfaceSceneView->GetElementTypeId(pixel_x, pixel_y);
                m_selectedPointInfo.Scalar = m_surfaceSceneView->GetScalarSample(pixel_x, pixel_y);
            }
            else
            {
                m_selectedPointInfo.Valid = false;
            }
            OnSelectedPointChanged(m_selectedPointInfo);
        }
    }
}
