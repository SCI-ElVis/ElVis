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

#include <ElVis/Gui/ViewSettingsUI.h>
#include <QScrollArea>
#include <boost/typeof/typeof.hpp>

namespace ElVis
{
    namespace Gui
    {
        ViewSettingsUI::ViewSettingsUI(boost::shared_ptr<ApplicationState> appData) :
            QDockWidget("View Settings"),
            m_appData(appData),
            m_cameraPropertyManager(new CameraPropertyManager()),
            m_cameraProperty(0),
            m_browser(0),
            m_layout(0),
            m_groupPropertyManager(new QtGroupPropertyManager()),
            m_intPropertyManager(new QtIntPropertyManager()),
            m_spinBoxFactory(new QtSpinBoxFactory()),
            m_viewportProperty(0),
            m_viewportXProperty(0),
            m_viewportYProperty(0),
            m_enumPropertyManager(new QtEnumPropertyManager()),
            m_enumEditorFactory(new QtEnumEditorFactory()),
            m_projectionTypeProperty(0)
        {
            this->setObjectName("ViewSettings");
            QScrollArea* m_scrollArea = new QScrollArea();

            m_browser = new QtTreePropertyBrowser();

            /////////////////////
            // Camera
            ////////////////////
            m_cameraProperty = m_cameraPropertyManager->addProperty("Camera", appData->GetSurfaceSceneView()->GetViewSettings());
            m_cameraPropertyManager->SetupPropertyManagers(m_browser, m_browser);
            m_browser->addProperty(m_cameraProperty);

            ///////////////////
            // Viewport
            ///////////////////
            m_viewportProperty = m_groupPropertyManager->addProperty("Viewport");
            m_browser->addProperty(m_viewportProperty);
            m_viewportXProperty = m_intPropertyManager->addProperty("Width");
            m_viewportYProperty = m_intPropertyManager->addProperty("Height");
            m_viewportProperty->addSubProperty(m_viewportXProperty);
            m_viewportProperty->addSubProperty(m_viewportYProperty);

            m_intPropertyManager->setValue(m_viewportXProperty, m_appData->GetSurfaceSceneView()->GetWidth());
            m_intPropertyManager->setValue(m_viewportYProperty, m_appData->GetSurfaceSceneView()->GetHeight());
            m_appData->GetSurfaceSceneView()->OnWindowSizeChanged.connect(boost::bind(&ViewSettingsUI::HandleWindowSizeChanged, this, _1, _2));

            m_projectionTypeProperty = m_enumPropertyManager->addProperty("Projection Type");
            QStringList names;
            names.push_back("Perspective");
            names.push_back("Orthographic");
            m_enumPropertyManager->setEnumNames(m_projectionTypeProperty, names);
            m_browser->addProperty(m_projectionTypeProperty);
            m_browser->setFactoryForManager(m_enumPropertyManager, m_enumEditorFactory);

            connect(m_enumPropertyManager, SIGNAL(valueChanged(QtProperty*,int)), this, SLOT(HandleProjectionChangedInGui(QtProperty*,int)));

            m_scrollArea->setWidget(m_browser);
            m_layout = new QGridLayout(m_scrollArea);

            m_layout->addWidget(m_browser,0,0);
            this->setWidget(m_scrollArea);
        }

        ViewSettingsUI::~ViewSettingsUI()
        {
        }

        void ViewSettingsUI::HandleWindowSizeChanged(int w, int h)
        {
            m_intPropertyManager->setValue(m_viewportXProperty, m_appData->GetSurfaceSceneView()->GetWidth());
            m_intPropertyManager->setValue(m_viewportYProperty, m_appData->GetSurfaceSceneView()->GetHeight());
        }

        void ViewSettingsUI::HandleProjectionChangedInGui(QtProperty* prop, int value)
        {
            SceneViewProjection newProjectionType = static_cast<SceneViewProjection>(value);
            BOOST_AUTO(view, m_appData->GetSurfaceSceneView());
            if( view && view->GetProjectionType() != newProjectionType )
            {
                view->SetProjectionType(newProjectionType);
            }
        }
    }
}
