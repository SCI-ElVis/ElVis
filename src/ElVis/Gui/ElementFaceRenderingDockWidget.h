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

#ifndef ELVIS_GUI_ELEMENT_FACE_RENDERING_DOCK_WIDGET_H
#define ELVIS_GUI_ELEMENT_FACE_RENDERING_DOCK_WIDGET_H

#include <QtWidgets/QWidget>
#include <boost/shared_ptr.hpp>
#include <ElVis/Gui/ApplicationState.h>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QGraphicsScene>
#include <QtWidgets/QGraphicsRectItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QDoubleSpinBox>

#include <ElVis/Gui/ColorMapRect.h>
#include <ElVis/Core/HostTransferFunction.h>

#include <ElVis/QtPropertyBrowser/qtgroupboxpropertybrowser.h>
#include <ElVis/QtPropertyBrowser/qtpropertymanager.h>
#include <ElVis/QtPropertyBrowser/qteditorfactory.h>

namespace ElVis
{
    namespace Gui
    {

        class ElementFaceRenderingDockWidget : public QDockWidget
        {
            Q_OBJECT;

            public:
                ElementFaceRenderingDockWidget(boost::shared_ptr<ApplicationState> appData, QWidget* parent = 0, Qt::WindowFlags f = 0);
                virtual ~ElementFaceRenderingDockWidget() {}

            public Q_SLOTS:
                void HandleModelChanged(boost::shared_ptr<Model>);
                void HandleAddButtonPressed();
                void HandleSceneInitialized(const Scene& s);
                void HandleGeometryFaceMapSelected(QtProperty* p, bool v);
                void HandleToleranceChanged();

            protected:
                virtual void keyPressEvent(QKeyEvent *);

            private:
                ElementFaceRenderingDockWidget(const ElementFaceRenderingDockWidget& rhs);
                ElementFaceRenderingDockWidget& operator=(const ElementFaceRenderingDockWidget& rhs);

                boost::shared_ptr<ApplicationState> m_appData;
                QListWidget* m_list;
                QGridLayout* m_layout;
                QGridLayout* m_leftLayout;
                QGridLayout* m_rightLayout;
                QWidget* m_leftWidget;
                QWidget* m_rightWidget;
                QPushButton* m_addElementIdButton;
                QSpinBox* m_elementId;
                QtBoolPropertyManager* m_boolPropertyManager;
                QtCheckBoxFactory* m_checkBoxFactory;
                std::map<QtProperty*, std::vector<int> > m_faceMap;
                QtAbstractPropertyBrowser* m_browser;
                QDoubleSpinBox* m_toleranceSpinBox;
                std::map<QListWidgetItem*, int> m_shownFaces;


        };
    }
}

#endif
