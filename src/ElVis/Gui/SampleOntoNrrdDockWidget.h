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

#ifndef ELVIS_GUI_SAMPLE_ONTO_NRRD_DOCK_WIDGET_H
#define ELVIS_GUI_SAMPLE_ONTO_NRRD_DOCK_WIDGET_H

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
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <ElVis/Core/SampleOntoNrrdModule.h>

namespace ElVis
{
    namespace Gui
    {

        class SampleOntoNrrdDockWidget : public QDockWidget
        {
            Q_OBJECT;

            public:
                SampleOntoNrrdDockWidget(boost::shared_ptr<ApplicationState> appData, QWidget* parent = 0, Qt::WindowFlags f = 0);
                virtual ~SampleOntoNrrdDockWidget() {}

            public Q_SLOTS:
                void HandleGeneratePressed();
            protected:

            private:
                SampleOntoNrrdDockWidget(const SampleOntoNrrdDockWidget& rhs);
                SampleOntoNrrdDockWidget& operator=(const SampleOntoNrrdDockWidget& rhs);

                boost::shared_ptr<ApplicationState> m_appData;
                QLineEdit* m_fileName;
                QLabel* m_fileNameLabel;
                QPushButton* m_generateButton;
                QDoubleSpinBox* m_spacing;
                QGridLayout* m_layout;
                QLabel* m_hLabel;
                SampleOntoNrrd* m_module;

        };
    }
}

#endif
