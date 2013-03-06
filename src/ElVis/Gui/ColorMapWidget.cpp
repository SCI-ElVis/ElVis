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

#include <ElVis/Core/HostTransferFunction.h>

#include <ElVis/Gui/ColorMapWidget.h>
#include <ElVis/Gui/ColorMapRect.h>
#include <boost/bind.hpp>
#include <boost/typeof/typeof.hpp>

namespace ElVis
{
    namespace Gui
    {
        ColorMapWidget::ColorMapWidget(boost::shared_ptr<ApplicationState> appData) :
            QDockWidget("Color Map Editor"),
            m_appData(appData),
            m_topLevelLayout(new QHBoxLayout()),
            m_fileLayout(new QGridLayout()),
            m_listOfColorMaps(new QTreeWidget()),
            m_loadColorMapButton(new QPushButton("Load Color Map")),
            m_saveColorMapButton(new QPushButton("Save Color Map")),
            m_selectedColorManager(new ColorPropertyManager()),
            m_selectedColorBrowser(new QtGroupBoxPropertyBrowser()),
            m_doublePropertyManager(new QtDoublePropertyManager()),
            m_groupPropertyManager(new QtGroupPropertyManager()),
            m_spinBoxFactory(new QtDoubleSpinBoxFactory()),
            m_minProperty(0),
            m_maxProperty(0),
            m_colorMapEditor(new ColorMapEditorWidget(appData)),
            m_editorLayout(new QGridLayout()),
            m_selectedColor(),
            m_colorProperty(0)
        {
            this->setObjectName("ColorMapEditor");
            QScrollArea* m_scrollArea = new QScrollArea();
            QWidget* widget = new QWidget();

            // Left showing files.
            QWidget* fileWidget = new QWidget();
            m_fileLayout->addWidget(m_listOfColorMaps, 0, 0, 1, 2);
            m_fileLayout->addWidget(m_loadColorMapButton, 1, 0);
            m_fileLayout->addWidget(m_saveColorMapButton, 1, 1);
            //fileWidget->setLayout(m_fileLayout);

            // Right with editors.
            m_selectedColorManager->SetupPropertyManagers(m_selectedColorBrowser, m_selectedColorBrowser);
            m_selectedColorBrowser->setFactoryForManager(m_doublePropertyManager, m_spinBoxFactory);

            m_colorProperty = m_selectedColorManager->addProperty("Selected Color", m_selectedColor);


            m_minProperty = m_doublePropertyManager->addProperty("Min");
            m_doublePropertyManager->setValue(m_minProperty, 0.0);
            m_doublePropertyManager->setDecimals(m_minProperty, 10);

            m_maxProperty = m_doublePropertyManager->addProperty("Max");
            m_doublePropertyManager->setValue(m_maxProperty, 1.0);
            m_doublePropertyManager->setMinimum(m_maxProperty, 0.0);
            m_doublePropertyManager->setMaximum(m_minProperty, 1.0);
            m_doublePropertyManager->setSingleStep(m_maxProperty, .01);
            m_doublePropertyManager->setSingleStep(m_minProperty, .01);
            m_doublePropertyManager->setDecimals(m_maxProperty, 10);

            QtProperty* colorMapProperties = m_groupPropertyManager->addProperty("Color Map Properties");
            colorMapProperties->addSubProperty(m_minProperty);
            colorMapProperties->addSubProperty(m_maxProperty);


            m_selectedColorBrowser->addProperty(colorMapProperties);
            m_selectedColorBrowser->addProperty(m_colorProperty);


            m_editorLayout->addWidget(m_colorMapEditor, 0, 0);
            m_editorLayout->addWidget(m_selectedColorBrowser, 1, 0);

            QWidget* editorWidget = new QWidget();
            //editorWidget->setLayout(m_editorLayout);

            // Combine it all.
            m_topLevelLayout->addLayout(m_fileLayout);
            m_topLevelLayout->addLayout(m_editorLayout);
//            m_topLevelLayout->addWidget(fileWidget);
//            m_topLevelLayout->addWidget(editorWidget);
            widget->setLayout(m_topLevelLayout);
            m_scrollArea->setWidget(widget);
            this->setWidget(m_scrollArea);

            // 2 sections.  The first is a list of loaded color maps, with buttons for
            // loading and saving.

            // The second section shows the currently selected color map.
            typedef std::pair<std::string, Scene::ColorMapInfo> IterType;
            BOOST_FOREACH(IterType iter, appData->GetScene()->GetColorMaps())
            {
                AddColorMap(iter.second);
            }

            SetupSubscriptions();
        }

        void ColorMapWidget::SetupSubscriptions()
        {
            m_appData->GetScene()->OnColorMapAdded.connect(boost::bind(&ColorMapWidget::AddColorMap, this, _1));
            connect(m_listOfColorMaps, SIGNAL(currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)), this, SLOT(HandleColorMapSelectionChanged(QTreeWidgetItem*,QTreeWidgetItem*)));
            connect(m_doublePropertyManager, SIGNAL(valueChanged(QtProperty*,double)), this, SLOT(HandleColorMapRangeChanged(QtProperty*,double)));
            ColorMapRect* m_rect = m_colorMapEditor->GetRect();
            connect(m_rect, SIGNAL(OnColorMapChanged()), this, SLOT(HandleColorMapChanged()));
            connect(m_rect, SIGNAL(OnSelectedPointChanged(const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator&)),
                    this, SLOT(HandleSelectedPointChanged(const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator&)));
            connect(m_selectedColorManager, SIGNAL(OnColorChangedInGui()), this, SLOT(HandleSelectedColorChangedInGui()));
        }

        void ColorMapWidget::HandleSelectedColorChangedInGui()
        {
            ColorMapRect* m_rect = m_colorMapEditor->GetRect();
            std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator selectedPoint = m_rect->GetSelectedBreakpoint();
            if( m_rect->GetTransferFunction() && 
                selectedPoint != m_rect->GetTransferFunction()->GetBreakpoints().end() )
            {
                BOOST_AUTO(transferFunction, m_rect->GetTransferFunction());
                transferFunction->SetBreakpoint(selectedPoint, m_selectedColor);
                m_rect->ForceUpdate();
            }
        }

        void ColorMapWidget::HandleSelectedPointChanged(const std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator& value)
        {
            ColorMapRect* m_rect = m_colorMapEditor->GetRect();
            std::map<ElVisFloat, ColorMapBreakpoint>::const_iterator selectedPoint = m_rect->GetSelectedBreakpoint();
            if( m_rect->GetTransferFunction() && 
                selectedPoint != m_rect->GetTransferFunction()->GetBreakpoints().end() )
            {
                m_selectedColor = (*selectedPoint).second.Col;
            }
            else
            {
                m_selectedColor = Color();
            }
            m_rect->update();
        }

        void ColorMapWidget::HandleColorMapChanged()
        {
            
        }

        void ColorMapWidget::AddColorMap(const Scene::ColorMapInfo& data)
        {
            QTreeWidgetItem* item = new QTreeWidgetItem(m_listOfColorMaps);
            item->setText(0, data.Name.c_str());

            // If this is the first one, make sure it is selected.
            if( m_listOfColorMaps->topLevelItemCount() == 1 )
            {
                m_listOfColorMaps->setCurrentItem(item);
            }
        }

        void ColorMapWidget::HandleColorMapSelectionChanged(QTreeWidgetItem* current, QTreeWidgetItem* previous)
        {
            if( current )
            {
                boost::shared_ptr<PiecewiseLinearColorMap> colorMap = m_appData->GetScene()->GetColorMap(current->text(0).toStdString());
                m_appData->SetSelectedColorMap(colorMap);
                colorMap->SetMin(m_doublePropertyManager->value(m_minProperty));
                colorMap->SetMax(m_doublePropertyManager->value(m_maxProperty));
            }
            else
            {
                m_appData->ClearSelectedTransferFunction();
            }
        }

        void ColorMapWidget::HandleColorMapRangeChanged(QtProperty* prop, double value)
        {
            boost::shared_ptr<PiecewiseLinearColorMap> map = m_appData->GetSelectedColorMap();

            if( prop == m_minProperty )
            {
                map->SetMin(value);
                m_doublePropertyManager->setMinimum(m_maxProperty, value);
            }
            else if( prop == m_maxProperty )
            {
                map->SetMax(value);
                m_doublePropertyManager->setMaximum(m_minProperty, value);
            }
        }

    }
}
