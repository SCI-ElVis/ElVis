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

#ifndef ELVIS_GUI_MODEL_INSPECTOR_WIDGET_H
#define ELVIS_GUI_MODEL_INSPECTOR_WIDGET_H

#include <QDockWidget>
#include <ElVis/QtPropertyBrowser/qtpropertybrowser.h>
#include <ElVis/QtPropertyBrowser/qtpropertymanager.h>
#include <ElVis/QtPropertyBrowser/qttreepropertybrowser.h>
#include <ElVis/Core/Model.h>
#include <ElVis/Core/Scene.h>

#include <boost/shared_ptr.hpp>

#include <QtWidgets/QWidget>
#include <QtWidgets/QScrollArea>

namespace ElVis
{
    namespace Gui
    {
        class ModelInspectorWidget : public QDockWidget
        {
            Q_OBJECT

            // Model inspector should be updated whenever the actual model changes,
            // so it is a view of the model that is part of the scene.  I can pass
            // a pointer to the scene here, which can then be informed of the change
            // with event.

            // The model change event in the scene will be a boost::signals event, but
            // I want to hook it up to a slot.
            public:
                ModelInspectorWidget(boost::shared_ptr<Scene> scene, QWidget* parent = NULL, Qt::WindowFlags f = 0);

                virtual ~ModelInspectorWidget();

            public Q_SLOTS:
                void HandleModelChanged(boost::shared_ptr<Model> m);

            private:
                boost::shared_ptr<Scene> m_scene;
                QScrollArea* m_dockWidgetContents;
                QtIntPropertyManager* m_intManager;
                QtProperty* m_numElementsProperty;
                QtTreePropertyBrowser* m_browser;
                QGridLayout* m_layout;
        };
    }
}

#endif
