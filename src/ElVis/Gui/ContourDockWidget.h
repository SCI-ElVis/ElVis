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

#ifndef ELVIS_GUI_CONTOUR_DOCK_WIDGET_H
#define ELVIS_GUI_CONTOUR_DOCK_WIDGET_H

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
#include <QtWidgets/QCheckBox>

namespace ElVis
{
    namespace Gui
    {

        class ContourDockWidget : public QDockWidget
        {
            Q_OBJECT;

            public:
                ContourDockWidget(boost::shared_ptr<ApplicationState> appData, QWidget* parent = 0, Qt::WindowFlags f = 0);
                virtual ~ContourDockWidget() {}

            public Q_SLOTS:
                void HandleModelChanged(boost::shared_ptr<Model>);
                void HandleAddContourButtonPressed();
                void HandleContourAdded(ElVisFloat newValue);
                void HandleToggleDiscontinuousElementBoundaries(bool newValue);
                void HandleMatchVisual3Toggled(bool newValue);

            protected:
                virtual void keyPressEvent(QKeyEvent *);
            private:
                ContourDockWidget(const ContourDockWidget& rhs);
                ContourDockWidget& operator=(const ContourDockWidget& rhs);

                boost::shared_ptr<ApplicationState> m_appData;
                QListWidget* m_list;
                QGridLayout* m_layout;
                QPushButton* m_addContourButtonButton;
                QDoubleSpinBox* m_contourSpinBox;
                std::map<QListWidgetItem*, ElVisFloat> m_values;
                QCheckBox* m_enabledCheckBox;
                QCheckBox* m_matchVisual3CheckBox;


        };
    }
}

#endif
