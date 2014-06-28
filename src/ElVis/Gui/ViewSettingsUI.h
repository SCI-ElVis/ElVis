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

#ifndef ELVIS_GUI_VIEW_SETTINGS_UI_H
#define ELVIS_GUI_VIEW_SETTINGS_UI_H

#include <ElVis/Gui/CameraPropertyManager.h>
#include <ElVis/Gui/ApplicationState.h>
#include <ElVis/QtPropertyBrowser/qtgroupboxpropertybrowser.h>
#include <ElVis/QtPropertyBrowser/qttreepropertybrowser.h>

#include <QtWidgets/QDockWidget>
#include <QtWidgets/QGridLayout>

namespace ElVis
{
    namespace Gui
    {
        class ViewSettingsUI : public QDockWidget
        {
            Q_OBJECT;

            public:
                ViewSettingsUI(boost::shared_ptr<ApplicationState> appData);
                ~ViewSettingsUI();

            public Q_SLOTS:
                void HandleWindowSizeChanged(int w, int h);
                void HandleProjectionChangedInGui(QtProperty* prop, int value);

            protected:

            private:
                ViewSettingsUI(const ViewSettingsUI& rhs);
                ViewSettingsUI& operator=(const ViewSettingsUI& rhs);

                boost::shared_ptr<ApplicationState> m_appData;
                QtAbstractPropertyBrowser* m_browser;
                CameraPropertyManager* m_cameraPropertyManager;
                QtProperty* m_cameraProperty;
                QGridLayout* m_layout;
                QtGroupPropertyManager* m_groupPropertyManager;
                QtIntPropertyManager* m_intPropertyManager;
                QtSpinBoxFactory* m_spinBoxFactory;
                QtProperty* m_viewportProperty;
                QtProperty* m_viewportXProperty;
                QtProperty* m_viewportYProperty;
                QtEnumPropertyManager* m_enumPropertyManager;
                QtEnumEditorFactory* m_enumEditorFactory;
                QtProperty* m_projectionTypeProperty;
        };

    }
}

#endif
