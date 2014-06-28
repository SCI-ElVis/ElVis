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

#ifndef ELVIS_GUI_DEBUG_SETTINGS_DOCK_WIDGET_H
#define ELVIS_GUI_DEBUG_SETTINGS_DOCK_WIDGET_H

#include <QDockWidget>
#include <QtWidgets/QWidget>
#include <QtWidgets/QScrollArea>
#include <ElVis/QtPropertyBrowser/qttreepropertybrowser.h>

#include <boost/shared_ptr.hpp>

#include <ElVis/Core/Scene.h>

#include <ElVis/Gui/ApplicationState.h>
#include <ElVis/Gui/PointPropertyManager.h>
#include <ElVis/Gui/ColorPropertyManager.h>
#include <ElVis/Gui/Property.hpp>

#include <QtWidgets/QScrollArea>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QScrollArea>

namespace ElVis
{
    namespace Gui
    {
        class DebugSettingDockWidget : public QDockWidget
        {
            Q_OBJECT

            public:
                DebugSettingDockWidget(boost::shared_ptr<ApplicationState> scene, QWidget* parent = 0, Qt::WindowFlags f = 0);

            public Q_SLOTS:
                void HandleSelectedPointChanged(const PointInfo& info);

            private:
                DebugSettingDockWidget(const DebugSettingDockWidget& rhs);
                DebugSettingDockWidget& operator=(const DebugSettingDockWidget& rhs);

                boost::shared_ptr<ApplicationState> m_appData;
                QScrollArea* m_dockWidgetContents;
                QtTreePropertyBrowser* m_browser;
                QGridLayout* m_layout;
                PointPropertyManager* m_pointManager;
                QtIntPropertyManager* m_intPropertyManager;
                ColorPropertyManager* m_colorPropertyManager;
                QtDoublePropertyManager* m_doublePropertyManager;
                QtProperty* m_intersectionPointProperty;
                PointPropertyManager* m_normalManager;
                QtProperty* m_normalProperty;
                PointInfo m_pointInfo;
                QtGroupPropertyManager* m_groupPropertyManager;
                QtProperty* m_optixProperties;
                Property<int>* m_optixStackSizeProperty;
                Property<bool>* m_enableTraceProperty;
                QtProperty* m_pixelXProperty;
                QtProperty* m_pixelYProperty;
                QtProperty* m_pixelProperty;
                QtProperty* m_elementIdProperty;
                QtProperty* m_elementTypeProperty;
                QtProperty* m_selectedPixelProperty;
                QtProperty* m_sampleProperty;
        };
    }
}


#endif
