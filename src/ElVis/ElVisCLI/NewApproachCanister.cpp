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


#include <ElVis/Core/OpenGL.h>
#include <ElVis/Extensions/JacobiExtension/JacobiExtensionElVisModel.h>
#include <ElVis/Core/Camera.h>
#include <ElVis/Core/ColorMap.h>
#include <ElVis/Core/Camera.h>
#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/Scene.h>
#include <ElVis/Core/Light.h>
#include <ElVis/Core/Color.h>
#include <ElVis/Core/PointLight.h>
#include <ElVis/Core/Triangle.h>
#include <ElVis/Core/SurfaceObject.h>
#include <ElVis/Core/ElVisConfig.h>
#include <ElVis/Core/CutSurfaceMeshModule.h>
#include <ElVis/Core/IsosurfaceModule.h>
#include <ElVis/Core/VolumeRenderingModule.h>
#include <ElVis/Core/Plane.h>

#include <ElVis/Core/LightingModule.h>
#include <ElVis/Core/PrimaryRayModule.h>
#include <ElVis/Core/CutSurfaceContourModule.h>
#include <string>
#include <boost/bind.hpp>
#include <boost/timer.hpp>
#include <ElVis/Core/Cylinder.h>
#include "NewApproachCanister.h"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <ElVis/Core/ColorMapperModule.h>
#include <ElVis/Core/SampleVolumeSamplerObject.h>
#include <boost/typeof/typeof.hpp>

#include <boost/make_shared.hpp>


int ColorMapBulletNewApproachVolumeSampling(int argc, char** argv, boost::shared_ptr<ElVis::Model> model, unsigned int width, unsigned int height, const std::string& outFilePath)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(100, 100);
    glutCreateWindow("fake");

    boost::shared_ptr<ElVis::Scene> scene = boost::make_shared<ElVis::Scene>();
    scene->SetModel(model);

    boost::shared_ptr<ElVis::Cylinder> cylinder(new ElVis::Cylinder());
    cylinder->GetTransformationMatrix()[11] = 2.0;

    cylinder->GetTransformationMatrix()[0] = .10001f;
    cylinder->GetTransformationMatrix()[5] = .10001f;
    cylinder->GetTransformationMatrix()[10] = .41f;
   
    
    BOOST_AUTO(l, boost::make_shared<ElVis::PointLight>());
    ElVis::Color lightColor;
    lightColor.SetRed(.5);
    lightColor.SetGreen(.5);
    lightColor.SetBlue(.5);

    ElVis::WorldPoint lightPos(10.0, 0.0, 0.0);
    l->SetColor(lightColor);
    l->SetPosition(lightPos);
    scene->AddLight(l);

    ElVis::Color ambientColor;
    ambientColor.SetRed(.5);
    ambientColor.SetGreen(.5);
    ambientColor.SetBlue(.5);
    scene->SetAmbientLightColor(ambientColor);
    
    boost::shared_ptr<ElVis::Triangle> triangle1(new ElVis::Triangle());
    triangle1->SetP0(ElVis::WorldPoint(0, -1, 0));
    triangle1->SetP1(ElVis::WorldPoint(0, -1, 18));
    triangle1->SetP2(ElVis::WorldPoint(0, 1, 18));

    boost::shared_ptr<ElVis::Triangle> triangle2(new ElVis::Triangle());
    triangle2->SetP0(ElVis::WorldPoint(0, -1, 0));
    triangle2->SetP2(ElVis::WorldPoint(0, 1, 0));
    triangle2->SetP1(ElVis::WorldPoint(0, 1, 18));
  
    boost::shared_ptr<ElVis::PrimaryRayModule> primaryRayModule(new ElVis::PrimaryRayModule());
    boost::shared_ptr<ElVis::SampleVolumeSamplerObject> t1Sampler(new ElVis::SampleVolumeSamplerObject(triangle1));
    boost::shared_ptr<ElVis::SampleVolumeSamplerObject> t2Sampler(new ElVis::SampleVolumeSamplerObject(triangle2));
    boost::shared_ptr<ElVis::SampleVolumeSamplerObject> cylinderSurface(new ElVis::SampleVolumeSamplerObject(cylinder));
    primaryRayModule->AddObject(t1Sampler);
    primaryRayModule->AddObject(t2Sampler);
    primaryRayModule->AddObject(cylinderSurface);

    boost::shared_ptr<ElVis::ColorMapperModule> colorMapperModule(new ElVis::ColorMapperModule());

    boost::shared_ptr<ElVis::TextureColorMap> textureColorMapper(new ElVis::TextureColorMap(ElVis::GetColorMapPath() + "/diverging257.cmap"));
    textureColorMapper->SetMin(-.12);
    textureColorMapper->SetMax(0);
    colorMapperModule->SetColorMap(textureColorMapper);
   
    
    ElVis::Camera c;

    // Zoomed to region of interest.
    //c.SetParameters(ElVis::WorldPoint(.5, 0, 1.5), ElVis::WorldPoint(0, 0, 1.5), ElVis::WorldVector(0, 1, 0));

    // Overall view
    //c.SetParameters(ElVis::WorldPoint(6, 0, 3.5), ElVis::WorldPoint(0, 0, 3.5), ElVis::WorldVector(0, 1, 0));

    //c.SetParameters(ElVis::WorldPoint(1.8, 1.2, 3.0), ElVis::WorldPoint(0, 0, 1), ElVis::WorldVector(0, 1, 0));
    c.SetParameters(ElVis::WorldPoint(1.8, .46, 3.7), ElVis::WorldPoint(0, 0, 2.7), ElVis::WorldVector(0, 1, 0));

    boost::shared_ptr<ElVis::SceneView> view(new ElVis::SceneView());
    view->SetCamera(c);
    
    boost::shared_ptr<ElVis::LightingModule> lighting(new ElVis::LightingModule());


    view->AddRenderModule(primaryRayModule);
    view->AddRenderModule(colorMapperModule);
    view->AddRenderModule(lighting);
    view->SetScene(scene);
    view->Resize(width, height);
    
    //view->GetScene()->SetEnableOptixTrace(true);
    //ElVis::Point<int, ElVis::TwoD> pixel1(10, 10);
    //view->GetScene()->SetOptixTracePixelIndex(pixel1);
    
    view->Draw();

    //std::cout << "Second pixel." << std::endl;
    //ElVis::Point<int, ElVis::TwoD> pixel(10, 1000-10-1);
    //view->GetScene()->SetOptixTracePixelIndex(pixel);
    //view->Draw();

    //ElVis::Point<int, ElVis::TwoD> pixel2(10, 1000-10-1);
    //view->GetScene()->SetOptixTracePixelIndex(pixel2);
    //view->GetScene()->SetEnableOptixTrace(true);
    //view->Draw();



    view->WriteColorBufferToFile(outFilePath.c_str());
    return 0;
}


int GenericCLIInterface(int argc, char** argv,
                        boost::shared_ptr<ElVis::Scene> scene,
                        boost::shared_ptr<ElVis::Model> model,
                        boost::shared_ptr<ElVis::ColorMap> colorMap,
                        unsigned int width, unsigned int height, const std::string& outFilePath, const ElVis::Camera& c)
{
  try
  {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(100, 100);
    glutCreateWindow("fake");

    std::cout << "Generic CLI Interface." << std::endl;
    const char* isovaluesLabel = "Isovalues";
    const char* isosurfaceModuleEnabledLabel = "IsosurfaceModuleEnabled";
    const char* volumeRenderingModuleEnabledLabel = "VolumeRenderingModuleEnabled";
    const char* contourModuleEnabledLabel = "ContourModuleEnabled";
    const char* meshModuleEnabledLabel = "MeshModuleEnabled";
    const char* boundarySurfacesLabel = "BoundarySurfaces";
    const char* renderFacesLabel = "RenderFaces";

    // Volume Rendering labels
    const char* integrationTypeLabel = "IntegrationType";
    const char* breakpointLabel = "Breakpoints";
    const char* colorsLabel = "Colors";
    const char* hLabel = "h";
    const char* epsilonLabel = "Epsilon";
    const char* traceLabel = "EnableTrace";
    const char* traceXLabel = "TraceX";
    const char* traceYLabel = "TraceY";
    const char* trackNumSamplesLabel = "TrackNumSamples";
    const char* renderIntegrationTypeLabel = "RenderIntegrationType";
    const char* emptySpaceSkippingLabel = "EnableEmptySpaceSkipping";

    const char* cutPlaneNormalLabel = "CutPlaneNormal";
    const char* cutPlanePointLabel= "CutPlanePoint";
    const char* fieldLabel="Field";

    const char* numTestsLabel = "NumTests";

    std::vector<double> cutPlaneNormal;
    std::vector<double> cutPlanePoint;
    std::vector<double> breakpoints;
    std::vector<double> colors;
    std::vector<int> boundarySurfaces;
    std::vector<int> faces;
    std::vector<double> isovalues;
    bool isosurfaceModuleEnabled = false;
    bool volumeRenderingModuleEnabled = false;
    bool contourModuleEnabled = false;
    bool meshModuleEnabled = false;
    unsigned int numTests = 1;
    std::string configFile;
    int fieldIndex = 0;

    boost::program_options::options_description desc("GenericCLIOptions");
    desc.add_options()
        (isovaluesLabel, boost::program_options::value<std::vector<double> >(&isovalues)->multitoken(), "Isovalues")
        (boundarySurfacesLabel, boost::program_options::value<std::vector<int> >(&boundarySurfaces)->multitoken(), "Boundary Surfaces")
        (renderFacesLabel, boost::program_options::value<std::vector<int> >(&faces)->multitoken(), "Faces")
        (isosurfaceModuleEnabledLabel, boost::program_options::value<bool>(&isosurfaceModuleEnabled), "Isosurface Module Enabled")
        (volumeRenderingModuleEnabledLabel, boost::program_options::value<bool>(&volumeRenderingModuleEnabled), "Volume Rendering Module Enabled")
        (contourModuleEnabledLabel, boost::program_options::value<bool>(&contourModuleEnabled), "Contour Module Enabled")
        (meshModuleEnabledLabel, boost::program_options::value<bool>(&meshModuleEnabled), "Mesh Module Enabled")
        (integrationTypeLabel, boost::program_options::value<int>(), "Integration Type")
        (breakpointLabel, boost::program_options::value<std::vector<double> >(&breakpoints)->multitoken(), "Breakpoints")
        (colorsLabel, boost::program_options::value<std::vector<double> >(&colors)->multitoken(), "Colors")
        (hLabel, boost::program_options::value<double>(), "h")
        (epsilonLabel, boost::program_options::value<double>(), "Epsilon")
        (traceLabel, boost::program_options::value<int>(), "Enable Trace")
        (traceXLabel, boost::program_options::value<int>(), "Trace X")
        (traceYLabel, boost::program_options::value<int>(), "Trace Y")
        (trackNumSamplesLabel, boost::program_options::value<int>(), "Track Num Samples")
        (renderIntegrationTypeLabel, boost::program_options::value<int>(), "RenderIntegrationType")
        (emptySpaceSkippingLabel, boost::program_options::value<int>(), "EnableEmptySpaceSkipping")
        (numTestsLabel, boost::program_options::value<unsigned int>(&numTests), "Number of Tests")
        (cutPlaneNormalLabel, boost::program_options::value<std::vector<double> >(&cutPlaneNormal)->multitoken(), "Cut Plane Normal")
        (cutPlanePointLabel, boost::program_options::value<std::vector<double> >(&cutPlanePoint)->multitoken(), "Cut Plane Point")
        (fieldLabel, boost::program_options::value<int>(&fieldIndex), "Field Index")
        ;

    const char* configFileNameLabel = "ConfigFile";

    boost::program_options::options_description configFileOptions("ConfigFileOptions");
    configFileOptions.add_options()
            (configFileNameLabel, boost::program_options::value<std::string>(&configFile), "Config File")
            ;

    boost::program_options::options_description commandLineOptions("CommandLineOptions");
    commandLineOptions.add(desc).add(configFileOptions);


    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(commandLineOptions).style(boost::program_options::command_line_style::allow_long |
                                                                                                              boost::program_options::command_line_style::long_allow_adjacent).allow_unregistered().run(), vm);
    boost::program_options::notify(vm);

    if( !configFile.empty() )
    {
        std::ifstream inFile(configFile.c_str());
        if( inFile )
        {
            boost::program_options::store(boost::program_options::parse_config_file(inFile, desc, true), vm);
            boost::program_options::notify(vm);
        }
        inFile.close();
    }

    #ifdef __GNUC__
    system("nvidia-smi");
    #endif

    bool trace = false;
    if( vm.count(traceLabel) == 1 )
    {
        trace = vm[traceLabel].as<int>();
    }

    int tracex = -1;
    int tracey = -1;
    if( vm.count(traceXLabel) == 1)
    {
        tracex = vm[traceXLabel].as<int>();
    }

    if( vm.count(traceYLabel) == 1)
    {
        tracey = vm[traceYLabel].as<int>();
    }

    scene->SetEnableOptixTrace(trace);
    scene->SetOptixTracePixelIndex(ElVis::Point<int, ElVis::TwoD>(tracex, tracey));

    BOOST_AUTO(l, boost::make_shared<ElVis::PointLight>());
    ElVis::Color lightColor;
    lightColor.SetRed(.5);
    lightColor.SetGreen(.5);
    lightColor.SetBlue(.5);

    ElVis::WorldPoint lightPos(10.0, 0.0, 0.0);
    l->SetColor(lightColor);
    l->SetPosition(lightPos);
    scene->AddLight(l);

    ElVis::Color ambientColor;
    ambientColor.SetRed(.5);
    ambientColor.SetGreen(.5);
    ambientColor.SetBlue(.5);
    scene->SetAmbientLightColor(ambientColor);

    boost::shared_ptr<ElVis::SceneView> view(new ElVis::SceneView());
    view->SetCamera(c);


    view->SetScalarFieldIndex(fieldIndex);

    boost::shared_ptr<ElVis::PrimaryRayModule> primaryRayModule(new ElVis::PrimaryRayModule());
    view->AddRenderModule(primaryRayModule);

    if( cutPlaneNormal.size() == 3 &&
        cutPlanePoint.size() == 3 )
    {
        ElVis::WorldPoint normal(cutPlaneNormal[0], cutPlaneNormal[1], cutPlaneNormal[2]);
        ElVis::WorldPoint p(cutPlanePoint[0], cutPlanePoint[1], cutPlanePoint[2]);
        
        boost::shared_ptr<ElVis::Plane> cutPlane(new ElVis::Plane(normal, p));
        boost::shared_ptr<ElVis::SampleVolumeSamplerObject> sampler(new ElVis::SampleVolumeSamplerObject(cutPlane));
        primaryRayModule->AddObject(sampler);
    }

    if( contourModuleEnabled )
    {
        std::cout << "Contour Module enabled.  Number of isovalues = " << isovalues.size() << std::endl;
        boost::shared_ptr<ElVis::CutSurfaceContourModule> contourModule(new ElVis::CutSurfaceContourModule());
        view->AddRenderModule(contourModule);

        for(unsigned int i = 0; i < isovalues.size(); ++i)
        {
            contourModule->AddIsovalue(isovalues[i]);
        }
    }

    if( meshModuleEnabled )
    {
        boost::shared_ptr<ElVis::CutSurfaceMeshModule> meshModule(new ElVis::CutSurfaceMeshModule());
        view->AddRenderModule(meshModule);
    }

    if( isosurfaceModuleEnabled )
    {
        boost::shared_ptr<ElVis::IsosurfaceModule> isosurfaceModule(new ElVis::IsosurfaceModule());
        view->AddRenderModule(isosurfaceModule);

        for(unsigned int i = 0; i < isovalues.size(); ++i)
        {
            isosurfaceModule->AddIsovalue(isovalues[i]);
        }
    }

    boost::shared_ptr<ElVis::ColorMapperModule> colorMapperModule(new ElVis::ColorMapperModule());
    view->AddRenderModule(colorMapperModule);
    colorMapperModule->SetColorMap(colorMap);

    if( boundarySurfaces.size() > 0)
    {
        boost::shared_ptr<ElVis::FaceObject> faceObject(new ElVis::FaceObject(scene));
        boost::shared_ptr<ElVis::SampleFaceObject> obj(new ElVis::SampleFaceObject(faceObject));
        primaryRayModule->AddObject(obj);
        for(int i = 0; i < boundarySurfaces.size(); ++i)
        {
            std::vector<int> faceIds;
            std::string boundaryName;
            model->GetBoundarySurface(boundarySurfaces[i], boundaryName, faceIds);

            if( faceIds.empty() ) continue;

            for(unsigned int j = 0; j < faceIds.size(); ++j)
            {
                obj->EnableFace(faceIds[j]);
            }
        }
    }

    if( faces.size() > 0 )
    {
        boost::shared_ptr<ElVis::FaceObject> faceObject(new ElVis::FaceObject(scene));
        boost::shared_ptr<ElVis::SampleFaceObject> obj(new ElVis::SampleFaceObject(faceObject));
        primaryRayModule->AddObject(obj);
        for(int i = 0; i < faces.size(); ++i)
        {
            obj->EnableFace(faces[i]);
        }
    }
    if( volumeRenderingModuleEnabled )
    {
        if( vm.count(integrationTypeLabel) == 0 )
        {
            return 1;
        }

        if( breakpoints.size()*4 != colors.size() )
        {
            std::cerr << "Invalid transfer specification." << std::endl;
            std::cerr << "Breakpoint size " << breakpoints.size() << std::endl;
            std::cerr << "Color size " << colors.size() << std::endl;
            return 1;
        }

        ElVis::VolumeRenderingIntegrationType integrationType = static_cast<ElVis::VolumeRenderingIntegrationType>(vm[integrationTypeLabel].as<int>());

        double h = .1;
        if( vm.count(hLabel) == 1 )
        {
            h = vm[hLabel].as<double>();
        }

        double epsilon = .001;
        if( vm.count(epsilonLabel) == 1 )
        {
            epsilon = vm[epsilonLabel].as<double>();
        }

        bool trackNumSamples = false;
        if( vm.count(trackNumSamplesLabel) == 1 )
        {
            trackNumSamples = (vm[trackNumSamplesLabel].as<int>() == 1);
        }

        bool renderIntegrationType = false;
        if( vm.count(renderIntegrationTypeLabel) == 1 )
        {
            renderIntegrationType = (vm[renderIntegrationTypeLabel].as<int>() == 1);
        }

        bool enableEmptySpaceSkipping = true;
        if( vm.count(emptySpaceSkippingLabel) == 1 )
        {
            enableEmptySpaceSkipping = (vm[emptySpaceSkippingLabel].as<int>() == 1);
        }

        boost::shared_ptr<ElVis::VolumeRenderingModule> m_volumeRenderingModule(new ElVis::VolumeRenderingModule());
        m_volumeRenderingModule->SetIntegrationType(integrationType);
        m_volumeRenderingModule->SetCompositingStepSize(h);
        m_volumeRenderingModule->SetEpsilon(epsilon);

        m_volumeRenderingModule->SetRenderIntegrationType(renderIntegrationType);
        m_volumeRenderingModule->SetEnableEmptySpaceSkipping(enableEmptySpaceSkipping);

        m_volumeRenderingModule->SetTrackNumberOfSamples(trackNumSamples);

        // Setup the transfer function.
        boost::shared_ptr<ElVis::HostTransferFunction> transferFunction = m_volumeRenderingModule->GetTransferFunction();
        for(unsigned int i = 0; i < breakpoints.size(); ++ i)
        {
            ElVis::Color c(colors[i*4], colors[i*4+1], colors[i*4+2], colors[i*4+3]);
            transferFunction->SetBreakpoint(breakpoints[i], c);
        }

        view->AddRenderModule(m_volumeRenderingModule);
    }

    boost::shared_ptr<ElVis::LightingModule> lighting(new ElVis::LightingModule());
    view->AddRenderModule(lighting);


    view->SetScene(scene);
    view->Resize(width, height);


    // Don't time to take care of initialization artifacts.
    view->Draw();


    double* times = new double[numTests-1];
    for(unsigned int testNum = 1; testNum < numTests; ++testNum)
    {
        // Repeated redraws will do nothing if we don't signal that the view has changed in some way.
        view->OnSceneViewChanged(*view);
        ElVis::Timer t = view->Draw();
        times[testNum-1] = t.TimePerTest(1);
    }

    view->WriteColorBufferToFile(outFilePath.c_str());

    if( numTests > 1 )
    {
        ElVis::Stat runtimeStats(times, std::numeric_limits<ElVisFloat>::max(), numTests-1, .95);
        std::cout << "Average Time Per Run: " << runtimeStats.Mean << std::endl;
        #ifdef __GNUC__
        system("nvidia-smi");
        #endif

    }
  }
  catch(std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
  catch(...)
  {
    std::cout << "Unknown exception." << std::endl;
  }

    return 0;
}
