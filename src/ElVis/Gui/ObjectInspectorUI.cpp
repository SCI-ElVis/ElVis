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

#include <ElVis/Gui/ObjectInspectorUI.h>
#include <ElVis/Core/Plane.h>
#include <ElVis/Core/SampleVolumeSamplerObject.h>
#include <ElVis/Gui/PlanePropertyManager.h>

#include <boost/bind.hpp>

namespace ElVis
{
    namespace Gui
    {
        ObjectInspectorUI::ObjectInspectorUI(boost::shared_ptr<ApplicationState> appData) :
            QDockWidget("Object Inspector"),
            m_appData(appData),
            m_browser(0),
            m_dockWidgetContents(new QScrollArea()),
            m_layout(0)
        {
            this->setObjectName("ObjectInspector");
            m_browser = new QtGroupBoxPropertyBrowser();
//            m_layout = new QGridLayout(m_dockWidgetContents);
//            m_layout->addWidget(m_browser, 0, 0);
            m_dockWidgetContents->setWidget(m_browser);
//            m_dockWidgetContents->setLayout(m_layout);
            this->setWidget(m_dockWidgetContents);

            m_appData->OnSelectedObjectChanged.connect(boost::bind(&ObjectInspectorUI::SelectedObjectChanged, this, _1));
        }

        ObjectInspectorUI::~ObjectInspectorUI()
        {
        }

        void ObjectInspectorUI::SelectedObjectChanged(boost::shared_ptr<PrimaryRayObject> obj)
        {
            // For the initial test, assume it is a plane.
            std::cout << "ObjectInspectorUI::SelectedObjectChanged." << std::endl;

            if( !obj )
            {
                std::cout << "Selected object is nothing, clear the gui." << std::endl;
                return;
            }

            boost::shared_ptr<SampleVolumeSamplerObject> asSampler = boost::dynamic_pointer_cast<SampleVolumeSamplerObject>(obj);
            if( asSampler )
            {
                std::cout << "Selected object is sampler" << std::endl;
                //asSampler->GetObject();
                boost::shared_ptr<Plane> asPlane = boost::dynamic_pointer_cast<Plane>(asSampler->GetObject());
                if( asPlane )
                {
                    std::cout << "Is a plane." << std::endl;

                    // Now put in the plane gui.  Eventually this goes in the factor, but we want to test it here.
//                    m_layout->removeWidget(m_browser);
//                    m_dockWidgetContents->removeWidget(m_browser);
                    delete m_browser;
                    m_browser = new QtGroupBoxPropertyBrowser();
//                    m_layout->addWidget(m_browser);

                    PlanePropertyManager* planeManager = new PlanePropertyManager();
                    planeManager->SetupPropertyManagers(m_browser, m_dockWidgetContents);
                    QtProperty* planeProperty = planeManager->addProperty("Plane", asPlane);
                    m_browser->addProperty(planeProperty);

                    m_dockWidgetContents->setWidget(m_browser);
//                    m_browser->show();

                    // Editors are maps between managers and editor factories
                    // that are set at the top level.  My approach is to delegate this responsibility.
                    // This whole framework seems to have an extra level if indirectin that isn't needed.
                    // I think the approach will be to pass the browser to my top level manager, which will
                    // pass it to children, and they'll set the factories.


                    //editor1->setFactoryForManager(stringManager, lineEditFactory);

                    // Step 1 - Create a base class for the virtual function to do this.
                    // Step 2 - Create the factory so each object I select with have its own editor.
                    // Step 3 - Test the full round trip.

                }
            }
            boost::shared_ptr<Plane> asPlane = boost::dynamic_pointer_cast<Plane>(obj);
            if( asPlane )
            {
                std::cout << "Selected object is a plane." << std::endl;
            }
        }
    }
}
