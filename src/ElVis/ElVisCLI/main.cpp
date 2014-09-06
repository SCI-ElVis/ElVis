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

#include <ElVis/Core/Camera.h>
#include <ElVis/Core/ColorMap.h>
#include <ElVis/Core/Camera.h>
#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/Scene.h>
#include <ElVis/Core/Light.h>
#include <ElVis/Core/Color.h>
#include <ElVis/Core/PointLight.h>
#include <ElVis/Core/Triangle.h>
#include <string>
#include <ElVis/Core/Cylinder.h>
#include "NewApproachCanister.h"
#include <boost/program_options.hpp>
#include <ElVis/Core/Interval.hpp>
#include <ElVis/Core/Jacobi.hpp>
#include <ElVis/ElVisCLI/VolumeRendering.h>
#include "Nektar++Models.h"
#include <fstream>
#include <boost/filesystem.hpp>
#include <ElVis/Core/Plugin.h>
#include <ElVis/Core/DynamicLib.h>
#include <ElVis/Core/ElVisConfig.h>

#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/bind.hpp>
#include <boost/timer.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>

#define png_infopp_NULL (png_infopp)NULL
#ifndef int_p_NULL
#define int_p_NULL (int*)NULL
#endif

namespace
{

  void PrintUsage()
  {
      std::cerr << "Usage: ElVisCLI --TestName <testname> --ModelPath <path> --Width <int> --Height <int> ..." << std::endl;
      exit(1);
  }

  void TestParam(boost::program_options::variables_map& vm, const char* p)
  {
      if( vm.count(p) == 0 ) PrintUsage();
  }


  template<typename T>
  bool testChannel(size_t idx, const T& lhs, const T& rhs)
  {
    if( lhs[idx] == rhs[idx] ) return true;

    auto dist = std::max(lhs[idx], rhs[idx]) -
      std::min(lhs[idx], rhs[idx]);
    return dist <= 1;
  }
}

int main(int argc, char** argv)
{
    const char* testNameLabel = "TestName";
    const char* modelPathLabel = "ModelPath";
    const char* widthLabel = "Width";
    const char* heightLabel = "Height";
    const char* outFileLabel = "OutFile";
    const char* compareFileLabel = "CompareFile";
    const char* eyeLabel = "Eye";
    const char* atLabel = "At";
    const char* upLabel = "Up";
    const char* moduleLabel = "Module";
    const char* colorMapLabel = "ColorMap";
    const char* colorMapMinLabel = "ColorMapMin";
    const char* colorMapMaxLabel = "ColorMapMax";


    std::vector<ElVisFloat> eyeInput;
    std::vector<ElVisFloat> atInput;
    std::vector<ElVisFloat> upInput;

    double min = 0.0;
    double max = 1.0;

    std::string configFile;

    boost::program_options::options_description desc("ElVisCLIOptions");
    desc.add_options()
        (testNameLabel, boost::program_options::value<std::string>(), "Test Name")
        (modelPathLabel, boost::program_options::value<std::string>(), "Model Path")
        (colorMapLabel, boost::program_options::value<std::string>(), "Color Map")
        (colorMapMinLabel, boost::program_options::value<double>(&min), "Color Map Min")
        (colorMapMaxLabel, boost::program_options::value<double>(&max), "Color Map Max")
        (moduleLabel, boost::program_options::value<std::string>(), "Module")
        (widthLabel, boost::program_options::value<unsigned int>(), "Width")
        (heightLabel, boost::program_options::value<unsigned int>(), "Height")
        (outFileLabel, boost::program_options::value<std::string>(), "Out File")
        (compareFileLabel, boost::program_options::value<std::string>(), "Compare File")
        (eyeLabel, boost::program_options::value<std::vector<ElVisFloat> >(&eyeInput)->multitoken(), "Eye")
        (atLabel, boost::program_options::value<std::vector<ElVisFloat> >(&atInput)->multitoken(), "At")
        (upLabel, boost::program_options::value<std::vector<ElVisFloat> >(&upInput)->multitoken(), "Up")
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
        std::cout << "Loading " << configFile << std::endl;
        if( inFile )
        {
            boost::program_options::store(boost::program_options::parse_config_file(inFile, desc, true), vm);
            boost::program_options::notify(vm);
        }
        inFile.close();
    }


    
    TestParam(vm, testNameLabel);
    TestParam(vm, modelPathLabel);
    TestParam(vm, widthLabel);
    TestParam(vm, heightLabel);
    TestParam(vm, outFileLabel);
    TestParam(vm, moduleLabel);

    ElVis::WorldPoint eye;
    ElVis::WorldVector up;
    ElVis::WorldPoint at;
    if( eyeInput.size() == 3 )
    {
        eye = ElVis::WorldPoint(eyeInput[0], eyeInput[1], eyeInput[2]);
        std::cout << "Eye: " << eye << std::endl;
    }
    if( upInput.size() == 3 )
    {
        up = ElVis::WorldVector(upInput[0], upInput[1], upInput[2]);
        std::cout << "Up: " << up << std::endl;
    }
    if( atInput.size() == 3 )
    {
        at = ElVis::WorldPoint(atInput[0], atInput[1], atInput[2]);
        std::cout << "At: " << at << std::endl;
    }

    ElVis::Camera c;
    c.SetParameters(eye, at, up);

    std::string testName = vm[testNameLabel].as<std::string>();
    std::string modelPath = vm[modelPathLabel].as<std::string>();
    std::string outFilePath = vm[outFileLabel].as<std::string>();
    std::string modulePath = vm[moduleLabel].as<std::string>();
    int width = vm[widthLabel].as<unsigned int>();
    int height = vm[heightLabel].as<unsigned int>();

    boost::shared_ptr<ElVis::Plugin> plugin(new ElVis::Plugin(modulePath));
    boost::shared_ptr<ElVis::Model> model(plugin->LoadModel(modelPath.c_str()));

    boost::shared_ptr<ElVis::Scene> scene = boost::make_shared<ElVis::Scene>();
    scene->SetModel(model);

    boost::shared_ptr<ElVis::ColorMap> colorMap;
    if( vm.count(colorMapLabel) > 0 )
    {
        std::string path = vm[colorMapLabel].as<std::string>();
        std::cout << "Loading " << path << std::endl;
        colorMap = scene->LoadColorMap(path);
        colorMap->SetMin(min);
        colorMap->SetMax(max);
    }
    else
    {
        boost::shared_ptr<ElVis::TextureColorMap> textureColorMapper(new ElVis::TextureColorMap(ElVis::GetColorMapPath() + "/diverging257.cmap"));
        textureColorMapper->SetMin(-.12);
        textureColorMapper->SetMax(0);
        colorMap = textureColorMapper;
    }



    int result = 0;

    if( testName == "CutSurfaceBullet" )
    {
        result = ColorMapBulletNewApproachVolumeSampling(argc, argv, model, width, height, outFilePath);
    }
    else if( testName == "CutSurfaceNektarSynthetic" )
    {
        result = TestNektarModelLoad(argc, argv, model, width, height, outFilePath);
    }
    else if( testName == "VolumeRenderBullet" )
    {
        result = VolumeRendering(argc, argv, model, width, height, outFilePath, c);
    }
    else if( testName == "VolumeRenderSphere" )
    {
        result = VolumeRendering(argc, argv, model, width, height, outFilePath, c);
    }
    else if( testName == "IsosurfaceBullet" )
    {
        result = IsosurfaceBullet(argc, argv, model, width, height, outFilePath, c);
    }
    else if( testName == "Generic" )
    {
        result = GenericCLIInterface(argc, argv, scene, model, colorMap, width, height, outFilePath, c);
    }
    if( result != 0 )
    {
        return result;
    }

    if( vm.count(compareFileLabel) == 1 )
    {
        std::string compareFile = vm[compareFileLabel].as<std::string>();

        std::string baselinePngPath = compareFile + ".png";
        std::string testPngPath = outFilePath + ".png";

        if( !boost::filesystem::exists(testPngPath) )
        {
            std::cout << "Error generating " << testPngPath << std::endl;
            return 1;
        }

        if( !boost::filesystem::exists(baselinePngPath) )
        {
            std::cout << "Copying results to compare location." << std::endl;
            boost::filesystem::path destPath(baselinePngPath);
            boost::filesystem::path srcPath(testPngPath);
            boost::filesystem::create_directories(destPath.parent_path());
            boost::filesystem::copy_file(srcPath, destPath, boost::filesystem::copy_option::overwrite_if_exists);
            return 0;
        }

        std::cout << "Comparing " << baselinePngPath << " with " << testPngPath << std::endl;
        
        boost::gil::rgba8_image_t baselinePngImage;
        boost::gil::rgba8_image_t testPngImage;

        boost::gil::png_read_image(baselinePngPath, baselinePngImage);
        boost::gil::png_read_image(testPngPath, testPngImage);

        auto baselineView = boost::gil::const_view(baselinePngImage);
        auto testView = boost::gil::const_view(testPngImage);

        if( baselineView.width() != testView.width() ||
            baselineView.height() != testView.height() )
        {
          return 1;
        }

        for(int j = 0; j < baselineView.height(); ++j)
        {
          for(int i = 0; i < baselineView.width(); ++i)
          {
            auto srcPixel = baselineView(i, j);
            auto testPixel = testView(i,j);
            auto pixelText = testChannel(0, srcPixel, testPixel) &&
              testChannel(1, srcPixel, testPixel) &&
              testChannel(2, srcPixel, testPixel);
            if( !pixelText ) return 1;
          }
        }
    }

    return result;
}
