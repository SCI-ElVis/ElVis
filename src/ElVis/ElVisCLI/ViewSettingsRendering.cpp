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
#include <ElVis/Core/Camera.h>
#include <ElVis/Core/ColorMap.h>
#include <ElVis/Core/Camera.h>
#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/Scene.h>
#include <ElVis/Core/Light.h>
#include <ElVis/Core/Color.h>
#include <ElVis/Core/Light.h>
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
#include "ViewSettingsRendering.h"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <ElVis/Core/ColorMapperModule.h>
#include <ElVis/Core/SampleVolumeSamplerObject.h>
#include <fstream>
#include <boost/make_shared.hpp>

int ViewSettingsRendering(int argc,
                         char** argv,
                         boost::shared_ptr<ElVis::Scene> scene,
                         boost::shared_ptr<ElVis::Model> model,
                         unsigned int width,
                         unsigned int height,
                         const std::string& outFilePath)
{
  try
  {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(100, 100);
    glutCreateWindow("fake");

    std::cout << "ViewSettingsRedering CLI Interface." << std::endl;

    const char* traceLabel = "EnableTrace";
    const char* traceXLabel = "TraceX";
    const char* traceYLabel = "TraceY";
    const char* settingsPathLabel = "Settings";
    const char* numTestsLabel = "NumTests";

    unsigned int numTests = 1;
    std::string settingsPath("settings.xml");

    boost::program_options::options_description desc(
      "ViewSettingsRederingOptions");
    desc.add_options()(
      traceLabel, boost::program_options::value<int>(), "Enable Trace")(
      traceXLabel, boost::program_options::value<int>(), "Trace X")(
      traceYLabel, boost::program_options::value<int>(), "Trace Y")(
      numTestsLabel, boost::program_options::value<unsigned int>(&numTests), "Number of Tests")
          (settingsPathLabel, boost::program_options::value<std::string>(&settingsPath), "Settings");

    boost::program_options::options_description commandLineOptions(
      "CommandLineOptions");
    commandLineOptions.add(desc);

    boost::program_options::variables_map vm;
    boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
        .options(commandLineOptions)
        .style(boost::program_options::command_line_style::allow_long |
               boost::program_options::command_line_style::long_allow_adjacent)
        .allow_unregistered()
        .run(),
      vm);
    boost::program_options::notify(vm);

    bool trace = false;
    if (vm.count(traceLabel) == 1)
    {
      trace = vm[traceLabel].as<int>();
    }

    int tracex = -1;
    int tracey = -1;
    if (vm.count(traceXLabel) == 1)
    {
      tracex = vm[traceXLabel].as<int>();
    }

    if (vm.count(traceYLabel) == 1)
    {
      tracey = vm[traceYLabel].as<int>();
    }

    scene->SetEnableOptixTrace(trace);
    scene->SetOptixTracePixelIndex(
      ElVis::Point<unsigned int, ElVis::TwoD>(tracex, tracey));

    auto l = boost::make_shared<ElVis::Light>();
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

    boost::shared_ptr<ElVis::PrimaryRayModule> primaryRayModule(
      new ElVis::PrimaryRayModule());
    primaryRayModule->SetEnabled(true);
    view->AddRenderModule(primaryRayModule);

    boost::shared_ptr<ElVis::CutSurfaceContourModule> contourModule(
      new ElVis::CutSurfaceContourModule());
    contourModule->SetEnabled(false);
    view->AddRenderModule(contourModule);

    boost::shared_ptr<ElVis::CutSurfaceMeshModule> meshModule(
      new ElVis::CutSurfaceMeshModule());
    meshModule->SetEnabled(false);
    view->AddRenderModule(meshModule);

    boost::shared_ptr<ElVis::IsosurfaceModule> isosurfaceModule(
      new ElVis::IsosurfaceModule());
    isosurfaceModule->SetEnabled(false);
    view->AddRenderModule(isosurfaceModule);

    boost::shared_ptr<ElVis::ColorMapperModule> colorMapperModule(
      new ElVis::ColorMapperModule());
    colorMapperModule->SetEnabled(true);
    view->AddRenderModule(colorMapperModule);

    boost::shared_ptr<ElVis::VolumeRenderingModule> m_volumeRenderingModule(
      new ElVis::VolumeRenderingModule());
    m_volumeRenderingModule->SetEnabled(false);

    view->AddRenderModule(m_volumeRenderingModule);

    boost::shared_ptr<ElVis::LightingModule> lighting(
      new ElVis::LightingModule());
    lighting->SetEnabled(true);
    view->AddRenderModule(lighting);

    view->SetScene(scene);
    view->Resize(width, height);

    std::ifstream inFile(settingsPath);
    boost::archive::xml_iarchive ia(inFile);
    ia >> boost::serialization::make_nvp("SceneView", *view);
    inFile.close();

    // Don't time to take care of initialization artifacts.
    view->Draw();

    double* times = new double[numTests - 1];
    for (unsigned int testNum = 1; testNum < numTests; ++testNum)
    {
      // Repeated redraws will do nothing if we don't signal that the view has
      // changed in some way.
      view->OnSceneViewChanged(*view);
      ElVis::Timer t = view->Draw();
      times[testNum - 1] = t.TimePerTest(1);
    }

    view->WriteColorBufferToFile(outFilePath.c_str());

    if (numTests > 1)
    {
      ElVis::Stat runtimeStats(
        times, std::numeric_limits<ElVisFloat>::max(), numTests - 1, .95);
      std::cout << "Average Time Per Run: " << runtimeStats.Mean << std::endl;
    }
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
  catch (...)
  {
    std::cout << "Unknown exception." << std::endl;
  }

  return 0;
}
