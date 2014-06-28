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

#include <ElVis/Gui/ModelInspectorWidget.h>
#include <boost/bind.hpp>
#include <QtWidgets/QScrollArea>

namespace ElVis
{
    namespace Gui
    {
        ModelInspectorWidget::ModelInspectorWidget(boost::shared_ptr<Scene> scene, QWidget* parent, Qt::WindowFlags f) :
            QDockWidget("Model Properties", parent, f),
            m_scene(scene),
            m_dockWidgetContents(new QScrollArea()),
            m_intManager(0),
            m_numElementsProperty(0),
            m_browser(0),
            m_layout(0)
        {
            this->setObjectName("ModelProperties");
            m_intManager = new QtIntPropertyManager(m_dockWidgetContents);
            m_numElementsProperty = m_intManager->addProperty("Number of Elements");
            m_intManager->setValue(m_numElementsProperty, 0);
            m_browser = new QtTreePropertyBrowser(m_dockWidgetContents);
            m_layout = new QGridLayout(m_dockWidgetContents);
            m_browser->addProperty(m_numElementsProperty);
            m_layout->addWidget(m_browser, 0, 0);
            this->setWidget(m_dockWidgetContents);
            scene->OnModelChanged.connect(boost::bind(&ModelInspectorWidget::HandleModelChanged, this, _1));
        }

        ModelInspectorWidget::~ModelInspectorWidget()
        {
        }

        void ModelInspectorWidget::HandleModelChanged(boost::shared_ptr<Model> model)
        {
            // Properties I want to display about a model.
            // Number of Elements
            // Number of elements of each type.
            // Spatial extent.

            const WorldPoint& minExtent = model->MinExtent();
            const WorldPoint& maxExtent = model->MaxExtent();
            unsigned int numElements = model->GetNumberOfElements();

            m_intManager->setValue(m_numElementsProperty, numElements);
        }
    }
}
