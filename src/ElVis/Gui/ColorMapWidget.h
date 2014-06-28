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

#ifndef ELVIS_GUI_COLOR_MAP_WIDGET_H
#define ELVIS_GUI_COLOR_MAP_WIDGET_H

#include <QDockWidget>
#include <ElVis/QtPropertyBrowser/qtgroupboxpropertybrowser.h>

#include <ElVis/Gui/ApplicationState.h>
#include <ElVis/Gui/ColorPropertyManager.h>
#include <ElVis/Gui/ColorMapEditorWidget.h>

#include <ElVis/Core/Object.h>

#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QGridLayout>

namespace ElVis
{
    namespace Gui
    {
        class ColorMapWidget : public QDockWidget
        {
            Q_OBJECT

            public:
                ColorMapWidget(boost::shared_ptr<ApplicationState> appData);
                virtual ~ColorMapWidget() {}

            public Q_SLOTS:
                void HandleColorMapSelectionChanged(QTreeWidgetItem* current, QTreeWidgetItem* previous);
                void HandleColorMapRangeChanged(QtProperty* prop, double value);
                void HandleColorMapChanged();
                void HandleSelectedPointChanged(const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator& value);
                void HandleSelectedColorChangedInGui();
            protected:

            private:
                ColorMapWidget(const ColorMapWidget& rhs);
                ColorMapWidget& operator=(const ColorMapWidget& rhs);

                void SetupSubscriptions();
                void AddColorMap(const Scene::ColorMapInfo& data);

                boost::shared_ptr<ApplicationState> m_appData;

                QHBoxLayout* m_topLevelLayout;
                QGridLayout* m_fileLayout;
                QTreeWidget* m_listOfColorMaps;
                QPushButton* m_loadColorMapButton;
                QPushButton* m_saveColorMapButton;
                ColorPropertyManager* m_selectedColorManager;
                QtGroupBoxPropertyBrowser* m_selectedColorBrowser;
                QtDoublePropertyManager* m_doublePropertyManager;
                QtGroupPropertyManager* m_groupPropertyManager;
                QtDoubleSpinBoxFactory* m_spinBoxFactory;
                QtProperty* m_minProperty;
                QtProperty* m_maxProperty;
                ColorMapEditorWidget* m_colorMapEditor;
                QGridLayout* m_editorLayout;
                Color m_selectedColor;
                QtProperty* m_colorProperty;
        };
    }
}

#endif
