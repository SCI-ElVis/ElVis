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
#include <ElVis/Core/ColorMapperModule.h>
#include <ElVis/ElVisCLI/Nektar++Models.h>
#include <boost/program_options.hpp>
#include <ElVis/Core/Camera.h>
#include <ElVis/Core/Scene.h>
#include <ElVis/Core/Light.h>
#include <ElVis/Core/Color.h>
#include <ElVis/Core/PointLight.h>
#include <ElVis/Core/SceneView.h>
#include <ElVis/Core/PrimaryRayModule.h>
#include <ElVis/Core/Triangle.h>
#include <ElVis/Core/LightingModule.h>
#include <ElVis/Core/SampleVolumeSamplerObject.h>
#include <ElVis/Core/ElVisConfig.h>
#include <boost/make_shared.hpp>
#include <boost/typeof/typeof.hpp>

int TestNektarModelLoad(int argc, char** argv, boost::shared_ptr<ElVis::Model> model, unsigned int width, unsigned int height, const std::string& outFilePath)
{
    glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
        glutInitWindowSize(100, 100);
        glutCreateWindow("fake");

    ElVis::Camera c;
    c.SetParameters(ElVis::WorldPoint(.5, .5, 1.2), ElVis::WorldPoint(.5, .5, 0), ElVis::WorldVector(0, 1, 0));

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

    ElVis::SceneView* view = new ElVis::SceneView();
    view->SetCamera(c);
    view->SetScene(scene);
    view->Resize(width, height);

    boost::shared_ptr<ElVis::PrimaryRayModule> primaryRayModule(new ElVis::PrimaryRayModule());
    view->AddRenderModule(primaryRayModule);

    boost::shared_ptr<ElVis::Triangle> triangle1(new ElVis::Triangle());
    triangle1->SetP0(ElVis::WorldPoint(0, 0, .5));
    triangle1->SetP1(ElVis::WorldPoint(1, 0, .5));
    triangle1->SetP2(ElVis::WorldPoint(1, 1, .5));
    boost::shared_ptr<ElVis::SampleVolumeSamplerObject> t1Sampler(new ElVis::SampleVolumeSamplerObject(triangle1));
    primaryRayModule->AddObject(t1Sampler);


    boost::shared_ptr<ElVis::Triangle> triangle2(new ElVis::Triangle());
    triangle2->SetP0(ElVis::WorldPoint(1, 1, .5));
    triangle2->SetP1(ElVis::WorldPoint(0, 1, .5));
    triangle2->SetP2(ElVis::WorldPoint(0, 0, .5));
    boost::shared_ptr<ElVis::SampleVolumeSamplerObject> t2Sampler(new ElVis::SampleVolumeSamplerObject(triangle2));
    primaryRayModule->AddObject(t2Sampler);

    boost::shared_ptr<ElVis::ColorMapperModule> colorMapperModule(new ElVis::ColorMapperModule());
    boost::shared_ptr<ElVis::TextureColorMap> textureColorMapper(new ElVis::TextureColorMap(ElVis::GetColorMapPath() + "/diverging257.cmap"));
    textureColorMapper->SetMin(0);
    textureColorMapper->SetMax(1.5);
    colorMapperModule->SetColorMap(textureColorMapper);

    view->AddRenderModule(colorMapperModule);

    boost::shared_ptr<ElVis::LightingModule> lighting(new ElVis::LightingModule());
    view->AddRenderModule(lighting);
    view->Draw();
    view->WriteColorBufferToFile(outFilePath);

    return 0;
}
