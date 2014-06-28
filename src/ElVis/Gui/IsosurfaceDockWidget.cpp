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

#include <ElVis/Gui/IsosurfaceDockWidget.h>
#include <ElVis/Gui/ApplicationState.h>

#include <ElVis/QtPropertyBrowser/qttreepropertybrowser.h>

#include <ElVis/Core/VolumeRenderingModule.h>

#include <QtWidgets/QScrollArea>

#include <boost/bind.hpp>

namespace ElVis
{
    namespace Gui
    {
        IsosurfaceDockWidget::IsosurfaceDockWidget(boost::shared_ptr<ApplicationState> appData,
            QWidget* parent, Qt::WindowFlags f) :
            QDockWidget("Isosurface Settings", parent, f),
            m_appData(appData),
            m_layout(0),
            m_list(new QListWidget()),
            m_addContourButton(new QPushButton("Add Isovalue")),
            m_contourSpinBox(new QDoubleSpinBox()),
            m_enabledCheckBox(new QCheckBox()),
            m_values()
        {
            m_list->setSortingEnabled(true);
            m_list->setSelectionMode(QAbstractItemView::SingleSelection);
            setObjectName("IsosurfaceDockWidget");
            QScrollArea* m_scrollArea = new QScrollArea();

            if (m_appData->GetIsosurfaceModule() )
            {
                QWidget* widget = new QWidget();

                m_layout = new QGridLayout(widget);
                m_layout->addWidget(m_list, 0, 0, 1, 2);

                m_contourSpinBox->setMinimum(-std::numeric_limits<ElVisFloat>::max());
                m_contourSpinBox->setMaximum(std::numeric_limits<ElVisFloat>::max());
                m_contourSpinBox->setDecimals(8);

                m_enabledCheckBox->setText("Enabled");
                m_enabledCheckBox->setChecked(false);

                m_layout->addWidget(m_contourSpinBox, 1, 0);
                m_layout->addWidget(m_addContourButton, 1, 1);
                m_layout->addWidget(m_enabledCheckBox, 2, 0);
                this->setWidget(widget);

                m_list->setFocusProxy(this);
                ///////////////////////////////////
                // Subscriptions
                ///////////////////////////////////
                connect(m_enabledCheckBox, SIGNAL(stateChanged(int)), this, SLOT(HandleEnabledStateChangedInGui(int)));
                m_appData->GetIsosurfaceModule()->OnEnabledChanged.connect(boost::bind(&IsosurfaceDockWidget::HandleEnabledChanged, this, _1, _2));
                connect(m_addContourButton, SIGNAL(clicked()), this, SLOT(HandleAddContourButtonPressed()));
                m_appData->GetIsosurfaceModule()->OnIsovalueAdded.connect(boost::bind(&IsosurfaceDockWidget::HandleIsovalueAdded, this, _1));
//                //////////////////////////////////////
//                // Isovalue
//                //////////////////////////////////////
//                m_isovalueProperty = m_doublePropertyManager->addProperty("Isovalue");
//                m_doublePropertyManager->setMinimum(m_isovalueProperty, 1e-10);
//                m_doublePropertyManager->setDecimals(m_isovalueProperty, 8);
//                //m_doublePropertyManager->setValue(m_sampleSpacingProperty, m_appData->GetIsosurfaceModule()->);
//                m_browser->addProperty(m_isovalueProperty);
//                connect(m_doublePropertyManager, SIGNAL(valueChanged(QtProperty*,double)), this, SLOT(HandleDoublePropertyChangedInGui(QtProperty*,double)));


//                m_browser->setFactoryForManager(m_doublePropertyManager, m_doubleSpinBoxFactory);
//                m_browser->setFactoryForManager(m_boolPropertyManager, m_checkBoxFactory);
//                m_scrollArea->setWidget(m_browser);

//                m_layout = new QGridLayout(m_scrollArea);
//                m_layout->addWidget(m_browser, 0, 0);

//                this->setWidget(m_scrollArea);
            }
        }

        void IsosurfaceDockWidget::HandleEnabledStateChangedInGui(int state)
        {
            bool enabled = (state == Qt::Checked);

            if( enabled != m_appData->GetIsosurfaceModule()->GetEnabled() )
            {
                m_appData->GetIsosurfaceModule()->SetEnabled(enabled);
            }
        }


        void IsosurfaceDockWidget::HandleEnabledChanged(const RenderModule&, bool newValue)
        {
            if( m_appData->GetIsosurfaceModule()->GetEnabled() != m_enabledCheckBox->isChecked() )
            {
                m_enabledCheckBox->setChecked(m_appData->GetIsosurfaceModule()->GetEnabled());
            }
        }

        void IsosurfaceDockWidget::HandleIsovalueAdded(ElVisFloat newValue)
        {
//            QListWidgetItem* item = new QListWidgetItem();
//            item->setData(Qt::DisplayRole, newValue);
//            m_list->addItem(item);
            PopulateListWidget();
        }

        void IsosurfaceDockWidget::HandleAddContourButtonPressed()
        {
            m_appData->GetIsosurfaceModule()->AddIsovalue(m_contourSpinBox->value());
        }

        void IsosurfaceDockWidget::PopulateListWidget()
        {
            m_list->clear();

            boost::shared_ptr<IsosurfaceModule> module = m_appData->GetIsosurfaceModule();
            if( !module ) return;

            const std::set<ElVisFloat>& isovalues = module->GetIsovalues();

            for(std::set<ElVisFloat>::const_iterator iter = isovalues.begin(); iter != isovalues.end(); ++iter)
            {
                QListWidgetItem* item = new QListWidgetItem();
                item->setData(Qt::DisplayRole, *iter);
                m_list->addItem(item);
                m_values[item] = *iter;
            }
        }

        void IsosurfaceDockWidget::keyPressEvent(QKeyEvent* event)
        {
            boost::shared_ptr<IsosurfaceModule> module = m_appData->GetIsosurfaceModule();
            if( !module )
            {
                return;
            }

            if( event->key() == Qt::Key_Delete )
            {
                std::cout << "Key delete." << std::endl;
                QList<QListWidgetItem*> items = m_list->selectedItems();

                for(int i = 0; i < items.size(); ++i)
                {
                    QListWidgetItem* item = items.at(i);

                    std::map<QListWidgetItem*, ElVisFloat>::iterator found = m_values.find(item);
                    if( found != m_values.end() )
                    {
                        module->RemoveIsovalue((*found).second);
                        m_values.erase(found);
                    }
                    delete item;
                }

            }


        }

    }
}
