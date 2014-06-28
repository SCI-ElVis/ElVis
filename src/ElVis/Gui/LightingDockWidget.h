////////////////////////////////////////////////////////////////////////////////
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description:
//
////////////////////////////////////////////////////////////////////////////////


#ifndef ELVIS_GUI_LIGHTING_DOCK_WIDGET_H
#define ELVIS_GUI_LIGHTING_DOCK_WIDGET_H

#include <QtWidgets/QDockWidget>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QTreeWidgetItem>
#include <QtWidgets/QGridLayout>

#include <ElVis/QtPropertyBrowser/qtgroupboxpropertybrowser.h>
#include <ElVis/QtPropertyBrowser/qtpropertymanager.h>
#include <ElVis/QtPropertyBrowser/qteditorfactory.h>

#include <boost/shared_ptr.hpp>

#include <ElVis/Core/Scene.h>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>


namespace ElVis
{
    class RenderModule;

    namespace Gui
    {
        class ElVisUI;
        class ApplicationState;

        class LightingDockWidget : public QDockWidget
        {
            Q_OBJECT

            public:
                LightingDockWidget(boost::shared_ptr<ApplicationState> appData,
                    QWidget* parent = NULL, Qt::WindowFlags f = 0);

                virtual ~LightingDockWidget() {}

            public Q_SLOTS:
                void HandleChangeHeadlightColorButtonPushed();
                void HandleChangeAmbientButtonPushed();
                void HandleBackgroundButtonPushed();

            private:
                Color Convert(const QColor& c);
                QColor Convert(const Color& c);
                LightingDockWidget(const LightingDockWidget& rhs);
                LightingDockWidget& operator=(const LightingDockWidget& rhs);

                boost::shared_ptr<ApplicationState> m_appData;
                QGridLayout* m_layout;
                QPushButton* m_changeHeadlightColor;
                QLabel* m_headlightColorLabel;
                QPushButton* m_changeBackgroundColor;
                QLabel* m_backgroundColorLabel;
                QPushButton* m_changeAmbientColor;
                QLabel* m_ambientColorLabel;

        };
    }
}

#endif
