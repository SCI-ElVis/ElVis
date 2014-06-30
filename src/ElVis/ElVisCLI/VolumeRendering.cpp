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

#include <boost/program_options.hpp>

#include <ElVis/ElVisCLI/VolumeRendering.h>

#include <ElVis/Core/VolumeRenderingModule.h>
#include <ElVis/Core/Scene.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/PointLight.h>
#include <ElVis/Core/Color.h>
#include <ElVis/Core/PrimaryRayModule.h>
#include <ElVis/Core/HostTransferFunction.h>
#include <string>
#include <ElVis/Core/ColorMapperModule.h>
#include <ElVis/Core/IsosurfaceModule.h>
#include <ElVis/Core/LightingModule.h>
#include <ElVis/Core/OpenGL.h>
#include <ElVis/Core/ElVisConfig.h>
#include <boost/make_shared.hpp>
#include <boost/typeof/typeof.hpp>

int IsosurfaceBullet(int argc, char** argv, boost::shared_ptr<ElVis::Model> model, unsigned int width, unsigned int height, const std::string& outFilePath, ElVis::Camera& c)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(100, 100);
    glutCreateWindow("fake");

    const char* isovaluesLabel = "Isovalues";
    std::vector<double> isovalues;


    boost::program_options::options_description desc("VolumeRenderingOptions");
    desc.add_options()
        (isovaluesLabel, boost::program_options::value<std::vector<double> >(&isovalues)->multitoken(), "Isovalues")
        ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).style(boost::program_options::command_line_style::allow_long |
                                                                                                              boost::program_options::command_line_style::long_allow_adjacent).allow_unregistered().run(), vm);
    boost::program_options::notify(vm);

    boost::shared_ptr<ElVis::Scene> scene = boost::make_shared<ElVis::Scene>();
    scene->SetModel(model);


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

    boost::shared_ptr<ElVis::PrimaryRayModule> primaryRayModule(new ElVis::PrimaryRayModule());
    boost::shared_ptr<ElVis::IsosurfaceModule> isosurfaceModule(new ElVis::IsosurfaceModule());
    boost::shared_ptr<ElVis::ColorMapperModule> m_colorMapperModule(new ElVis::ColorMapperModule());
    boost::shared_ptr<ElVis::LightingModule> lightingModule(new ElVis::LightingModule());

    for(unsigned int i = 0; i < isovalues.size(); ++i)
    {
        isosurfaceModule->AddIsovalue(isovalues[i]);
    }

    boost::shared_ptr<ElVis::SceneView> view(new ElVis::SceneView());
    view->SetCamera(c);

    boost::shared_ptr<ElVis::TextureColorMap> textureColorMapper(new ElVis::TextureColorMap(ElVis::GetColorMapPath() + "/diverging257.cmap"));
    textureColorMapper->SetMin(-.12);
    textureColorMapper->SetMax(0);
    m_colorMapperModule->SetColorMap(textureColorMapper);


    view->AddRenderModule(primaryRayModule);
    view->AddRenderModule(isosurfaceModule);
    view->AddRenderModule(m_colorMapperModule);
    view->AddRenderModule(lightingModule);
    view->SetScene(scene);
    view->Resize(width, height);


    ///////////////////////////////////
    // Single Shot
    ///////////////////////////////////
    view->Draw();
    //std::map<boost::shared_ptr<ElVis::RenderModule>, double> accum;
    //unsigned int numTests = 1;
    //for(unsigned int i = 0; i < numTests; ++i)
    //{
    //    view->Draw();
    //    const std::map<boost::shared_ptr<ElVis::RenderModule>, double>& timings = view->GetTimings();
    //    for( std::map<boost::shared_ptr<ElVis::RenderModule>, double>::const_iterator iter = timings.begin(); iter != timings.end(); ++iter)
    //    {
    //        accum[(*iter).first] += (*iter).second;
    //    }
    //}

    //// Make sure the data getting into the output file is always in the same order.
    //std::vector<boost::shared_ptr<ElVis::RenderModule> > headers;
    //for( std::map<boost::shared_ptr<ElVis::RenderModule>, double>::iterator iter = accum.begin(); iter != accum.end(); ++iter)
    //{
    //    boost::shared_ptr<ElVis::RenderModule> module = (*iter).first;
    //    double totalTime = (*iter).second;
    //    std::cout << typeid(*module).name() << " - " << totalTime/numTests << std::endl;
    //}

    view->WriteColorBufferToFile(outFilePath);
    return 0;
}

int VolumeRendering(int argc, char** argv, boost::shared_ptr<ElVis::Model> model, unsigned int width, unsigned int height, const std::string& outFilePath, ElVis::Camera& c)
{
    glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(100, 100);
  glutCreateWindow("fake");

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
    std::vector<double> breakpoints;
    std::vector<double> colors;


    boost::program_options::options_description desc("VolumeRenderingOptions");
    desc.add_options()
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
        ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).style(boost::program_options::command_line_style::allow_long |
                                                                                                              boost::program_options::command_line_style::long_allow_adjacent).allow_unregistered().run(), vm);
    boost::program_options::notify(vm);


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

    boost::shared_ptr<ElVis::Scene> scene = boost::make_shared<ElVis::Scene>();
    scene->SetModel(model);


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

    boost::shared_ptr<ElVis::PrimaryRayModule> primaryRayModule(new ElVis::PrimaryRayModule());
    boost::shared_ptr<ElVis::VolumeRenderingModule> m_volumeRenderingModule(new ElVis::VolumeRenderingModule());
    m_volumeRenderingModule->SetIntegrationType(integrationType);
    m_volumeRenderingModule->SetCompositingStepSize(h);
    m_volumeRenderingModule->SetEpsilon(epsilon);
    scene->SetEnableOptixTrace(trace);
    scene->SetOptixTracePixelIndex(ElVis::Point<int, ElVis::TwoD>(tracex, tracey));

    m_volumeRenderingModule->SetRenderIntegrationType(renderIntegrationType);
    m_volumeRenderingModule->SetEnableEmptySpaceSkipping(enableEmptySpaceSkipping);

    std::cout << "Track number of samples: " << (trackNumSamples ? "Yes" : "No") << std::endl;
    m_volumeRenderingModule->SetTrackNumberOfSamples(trackNumSamples);

    // Setup the transfer function.
    boost::shared_ptr<ElVis::HostTransferFunction> transferFunction = m_volumeRenderingModule->GetTransferFunction();
    for(unsigned int i = 0; i < breakpoints.size(); ++ i)
    {
        ElVis::Color c(colors[i*4], colors[i*4+1], colors[i*4+2], colors[i*4+3]);
        transferFunction->SetBreakpoint(breakpoints[i], c);
    }

    boost::shared_ptr<ElVis::SceneView> view(new ElVis::SceneView());
    view->SetCamera(c);


    boost::shared_ptr<ElVis::LightingModule> lightingModule(new ElVis::LightingModule());

    view->AddRenderModule(primaryRayModule);
    view->AddRenderModule(m_volumeRenderingModule);
    view->AddRenderModule(lightingModule);
    view->SetScene(scene);
    view->Resize(width, height);


    ///////////////////////////////////
    // Single Shot
    ///////////////////////////////////
    view->Draw();
    //std::map<boost::shared_ptr<ElVis::RenderModule>, double> accum;
    //unsigned int numTests = 1;
    //for(unsigned int i = 0; i < numTests; ++i)
    //{
    //    view->Draw();
    //    const std::map<boost::shared_ptr<ElVis::RenderModule>, double>& timings = view->GetTimings();
    //    for( std::map<boost::shared_ptr<ElVis::RenderModule>, double>::const_iterator iter = timings.begin(); iter != timings.end(); ++iter)
    //    {
    //        accum[(*iter).first] += (*iter).second;
    //    }
    //}
    //
    //// Make sure the data getting into the output file is always in the same order.
    //std::vector<boost::shared_ptr<ElVis::RenderModule> > headers;
    //for( std::map<boost::shared_ptr<ElVis::RenderModule>, double>::iterator iter = accum.begin(); iter != accum.end(); ++iter)
    //{
    //    boost::shared_ptr<ElVis::RenderModule> module = (*iter).first;
    //    double totalTime = (*iter).second;
    //    std::cout << typeid(*module).name() << " - " << totalTime/numTests << std::endl;
    //}

    view->WriteColorBufferToFile(outFilePath);
    return 0;
}


int VolumeRenderSphereForVerification(int argc, char** argv, boost::shared_ptr<ElVis::Model> model, unsigned int width, unsigned int height, const std::string& outFilePath, ElVis::Camera& c)
{
    glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(100, 100);
  glutCreateWindow("fake");

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
    std::vector<double> breakpoints;
    std::vector<double> colors;

    boost::program_options::options_description desc("VolumeRenderingOptions");
    desc.add_options()
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
        ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).style(boost::program_options::command_line_style::allow_long |
                                                                                                              boost::program_options::command_line_style::long_allow_adjacent).allow_unregistered().run(), vm);
    boost::program_options::notify(vm);

    if( vm.count(integrationTypeLabel) == 0 ) 
    {
        return 1;
    }

    if( breakpoints.size()*4 != colors.size() )
    {
        std::cerr << "Invalid transfer specification." << std::endl;
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

    boost::shared_ptr<ElVis::Scene> scene = boost::make_shared<ElVis::Scene>();
    scene->SetModel(model);

    boost::shared_ptr<ElVis::PrimaryRayModule> primaryRayModule(new ElVis::PrimaryRayModule());
    boost::shared_ptr<ElVis::VolumeRenderingModule> m_volumeRenderingModule(new ElVis::VolumeRenderingModule());
    m_volumeRenderingModule->SetIntegrationType(integrationType);
    m_volumeRenderingModule->SetCompositingStepSize(h);
    m_volumeRenderingModule->SetEpsilon(epsilon);
    scene->SetEnableOptixTrace(trace);
    scene->SetOptixTracePixelIndex(ElVis::Point<int, ElVis::TwoD>(tracex, tracey));
    m_volumeRenderingModule->SetRenderIntegrationType(renderIntegrationType);

    std::cout << "Track number of samples: " << (trackNumSamples ? "Yes" : "No") << std::endl;
    m_volumeRenderingModule->SetTrackNumberOfSamples(trackNumSamples);

    // Setup the transfer function.
    boost::shared_ptr<ElVis::HostTransferFunction> transferFunction = m_volumeRenderingModule->GetTransferFunction();
    for(unsigned int i = 0; i < breakpoints.size(); ++ i)
    {
        ElVis::Color c(colors[i*4], colors[i*4+1], colors[i*4+2], colors[i*4+3]);
        transferFunction->SetBreakpoint(breakpoints[i], c);
    }

//    ElVis::Camera c;
//    c.SetParameters(ElVis::WorldPoint(1.6, 0, 1.5), ElVis::WorldPoint(0, 0, 0), ElVis::WorldVector(0, 1, 0));

    // Zoomed to region of interest.
    

    // Overall view
    //c.SetParameters(ElVis::WorldPoint(6, 0, 3.5), ElVis::WorldPoint(0, 0, 3.5), ElVis::WorldVector(0, 1, 0));

    //c.SetParameters(ElVis::WorldPoint(1.8, 1.2, 3.0), ElVis::WorldPoint(0, 0, 1), ElVis::WorldVector(0, 1, 0));
    
    
    //c.SetParameters(ElVis::WorldPoint(1.8, .46, 3.7), ElVis::WorldPoint(0, 0, 2.7), ElVis::WorldVector(0, 1, 0));

    boost::shared_ptr<ElVis::SceneView> view(new ElVis::SceneView());
    view->SetCamera(c);


    view->AddRenderModule(primaryRayModule);
    view->AddRenderModule(m_volumeRenderingModule);
    view->SetScene(scene);
    view->Resize(width, height);


    ///////////////////////////////////
    // Single Shot
    ///////////////////////////////////
    //view->Draw();
    std::map<boost::shared_ptr<ElVis::RenderModule>, double> accum;

    unsigned int numTests = 1;
    for(unsigned int i = 0; i < numTests; ++i)
    {
        view->Draw();
        const std::map<boost::shared_ptr<ElVis::RenderModule>, double>& timings = view->GetTimings();
        for( std::map<boost::shared_ptr<ElVis::RenderModule>, double>::const_iterator iter = timings.begin(); iter != timings.end(); ++iter)
        {
            accum[(*iter).first] += (*iter).second;
        }
    }
    
    // Make sure the data getting into the output file is always in the same order.
    std::vector<boost::shared_ptr<ElVis::RenderModule> > headers;
    for( std::map<boost::shared_ptr<ElVis::RenderModule>, double>::iterator iter = accum.begin(); iter != accum.end(); ++iter)
    {
        boost::shared_ptr<ElVis::RenderModule> module = (*iter).first;
        double totalTime = (*iter).second;
        std::cout << typeid(*module).name() << " - " << totalTime/numTests << std::endl;
    }

    view->WriteColorBufferToFile(outFilePath);
    std::cout << "Done volume rendering." << std::endl;
    return 0;
}

//void PaddleVolumeRendering(int argc, char** argv)
//{
//    // Load volume.
//    if( argc != 6 )
//    {
//        std::cerr << "Usage: ElVisCLI <path_to_volume> <w> <h> <IntegrationType> <numTests>" << std::endl;
//        return;
//    }

//    glutInit(&argc, argv);
//	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
//	glutInitWindowSize(100, 100);
//	glutCreateWindow("fake");


//    std::string fileName(argv[1]);
//    int width = atoi(argv[2]);
//    int height = atoi(argv[3]);
//    int numTests = atoi(argv[5]);
//    unsigned int integrationType = atoi(argv[4]);

//    boost::shared_ptr<ElVis::JacobiExtension::JacobiExtensionModel> model(new ElVis::JacobiExtension::JacobiExtensionModel());
//    model->LoadVolume(fileName);
//    boost::shared_ptr<ElVis::JacobiExtension::FiniteElementVolume> m_volume = model->Volume();

//    boost::shared_ptr<ElVis::Scene> scene = boost::make_shared<ElVis::Scene>();
//    scene->SetModel(model);


//    ElVis::PointLight* l = new ElVis::PointLight();
//    ElVis::Color lightColor;
//    lightColor.SetRed(.5);
//    lightColor.SetGreen(.5);
//    lightColor.SetBlue(.5);

//    ElVis::WorldPoint lightPos(10.0, 0.0, 0.0);
//    l->SetColor(lightColor);
//    l->SetPosition(lightPos);
//    scene->AddLight(l);

//    ElVis::Color ambientColor;
//    ambientColor.SetRed(.5);
//    ambientColor.SetGreen(.5);
//    ambientColor.SetBlue(.5);
//    scene->SetAmbientLightColor(ambientColor);

//    ElVis::PrimaryRayModule* primaryRayModule = new ElVis::PrimaryRayModule();
//    ElVis::VolumeRenderingModule* m_volumeRenderingModule = new ElVis::VolumeRenderingModule();
//    m_volumeRenderingModule->SetIntegrationType(integrationType);
//    ElVis::Camera c;

//    // Zoomed to region of interest.
//    c.SetParameters(ElVis::WorldPoint(-8, 13, 15), ElVis::WorldPoint(5, 0, 5), ElVis::WorldVector(0, 1, 0));
//    //c.SetParameters(ElVis::WorldPoint(35, 40, 5), ElVis::WorldPoint(35, 0, 5), ElVis::WorldVector(0, 0, -1));
//    //c.SetParameters(ElVis::WorldPoint(6, 0, 3.5), ElVis::WorldPoint(0, 0, 3.5), ElVis::WorldVector(0, 1, 0));
//    //c.SetParameters(ElVis::WorldPoint(1.8, 1.2, 3.0), ElVis::WorldPoint(0, 0, 1), ElVis::WorldVector(0, 1, 0));

//    boost::shared_ptr<ElVis::SceneView> view(new ElVis::SceneView());
//    view->SetCamera(c);


//    view->AddRenderModule(primaryRayModule);
//    view->AddRenderModule(m_volumeRenderingModule);
//    view->SetScene(scene);
//    view->Resize(width, height);


//    ///////////////////////////////////
//    // Single Shot
//    ///////////////////////////////////
//    view->Draw();
//    std::map<ElVis::RenderModule*, double> accum;

//    for(unsigned int i = 0; i < numTests; ++i)
//    {
//        view->Draw();
//        const std::map<ElVis::RenderModule*, double>& timings = view->GetTimings();
//        for( std::map<ElVis::RenderModule*, double>::const_iterator iter = timings.begin(); iter != timings.end(); ++iter)
//        {
//            accum[(*iter).first] += (*iter).second;
//        }
//    }
    
//    // Make sure the data getting into the output file is always in the same order.
//    std::vector<ElVis::RenderModule*> headers;
//    for( std::map<ElVis::RenderModule*, double>::iterator iter = accum.begin(); iter != accum.end(); ++iter)
//    {
//        ElVis::RenderModule* module = (*iter).first;
//        double totalTime = (*iter).second;
//        std::cout << typeid(*module).name() << " - " << totalTime/numTests << std::endl;
//    }

//    view->WriteColorBufferToFile("volume_single_shot.ppm");

//}
