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

#ifndef ELVIS_GUI_RENDER_SETTTINGS_DOCK_WIDGET_H
#define ELVIS_GUI_RENDER_SETTTINGS_DOCK_WIDGET_H

#include <ElVis/Gui/ApplicationState.h>
#include <ElVis/QtPropertyBrowser/qtgroupboxpropertybrowser.h>
#include <ElVis/QtPropertyBrowser/qtpropertymanager.h>
#include <ElVis/QtPropertyBrowser/qteditorfactory.h>

#include <QtWidgets/QDockWidget>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QScrollArea>

#include <map>

namespace ElVis
{
    namespace Gui
    {
        class RenderSettingsDockWidget : public QDockWidget
        {
            Q_OBJECT;

            public:
                RenderSettingsDockWidget(boost::shared_ptr<ApplicationState> appData);
                virtual ~RenderSettingsDockWidget() {}

            public Q_SLOTS:
                void HandleEnabledChangedByGui(QtProperty* property, bool newValue);
                void HandleStackSizeChanged(QtProperty* property, int newValue);
                void HandleShowLookAtChanged(QtProperty* property, bool newValue);

                void HandleEnabledChanged(QtProperty* property, const RenderModule& module, bool newValue);

            protected:

            private:
                RenderSettingsDockWidget(const RenderSettingsDockWidget& rhs);
                RenderSettingsDockWidget& operator=(const RenderSettingsDockWidget& rhs);

                boost::shared_ptr<ApplicationState> m_appData;
                QtAbstractPropertyBrowser* m_browser;
                QtGroupPropertyManager* m_groupManager;
                QtBoolPropertyManager* m_boolPropertyManager;
                std::map<QtProperty*, boost::shared_ptr<RenderModule> > m_renderModules;
                QtCheckBoxFactory* m_checkBoxFactory;

                QtIntPropertyManager* m_intPropertyManager;
                QtProperty* m_stackSizeProperty;
                QtSpinBoxFactory* m_spinBoxFactory;
                QtProperty* m_showLookAtPointProperty;
                QtProperty* m_showBoundingBoxProperty;
                QGridLayout* m_layout;
        };

    }
}

#endif
