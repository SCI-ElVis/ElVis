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

#include <ElVis/Gui/ContourDockWidget.h>
#include <QtGui/QKeyEvent>

#include <boost/bind.hpp>

namespace ElVis
{
    namespace Gui
    {
        ContourDockWidget::ContourDockWidget(boost::shared_ptr<ApplicationState> appData, QWidget* parent, Qt::WindowFlags f) :
            QDockWidget("Cut-Surface Contour Module", parent, f),
            m_appData(appData),
            m_list(new QListWidget()),
            m_layout(0),
            m_addContourButtonButton(new QPushButton("Add Contour")),
            m_contourSpinBox(new QDoubleSpinBox()),
            m_values(),
            m_enabledCheckBox(new QCheckBox("Disc. Element Boundaries")),
            m_matchVisual3CheckBox(new QCheckBox("Match Visual 3"))
        {
            m_list->setSortingEnabled(true);
            setObjectName("CutSurfaceContourModule");

            QWidget* widget = new QWidget();

            m_layout = new QGridLayout(widget);
            m_layout->addWidget(m_list, 0, 0, 1, 2);

            m_contourSpinBox->setMinimum(-std::numeric_limits<ElVisFloat>::max());
            m_contourSpinBox->setMaximum(std::numeric_limits<ElVisFloat>::max());
            m_contourSpinBox->setDecimals(10);
            //m_elementId->setSingleStep(1);

            m_enabledCheckBox->setChecked( m_appData->GetContourModule()->GetTreatElementBoundariesAsDiscontinuous());
            m_matchVisual3CheckBox->setChecked( m_appData->GetContourModule()->GetMatchVisual3Contours());

            m_layout->addWidget(m_contourSpinBox, 1, 0);
            m_layout->addWidget(m_addContourButtonButton, 1, 1);
            m_layout->addWidget(m_enabledCheckBox, 2, 0);
            m_layout->addWidget(m_matchVisual3CheckBox, 3, 0);

            this->setWidget(widget);

            m_list->setFocusProxy(this);
//            m_contourSpinBox->setFocusProxy(this);

            m_appData->GetScene()->OnModelChanged.connect(boost::bind(&ContourDockWidget::HandleModelChanged, this, _1));
            m_appData->GetContourModule()->OnIsovalueAdded.connect(boost::bind(&ContourDockWidget::HandleContourAdded, this, _1));
            connect(m_addContourButtonButton, SIGNAL(clicked()), this, SLOT(HandleAddContourButtonPressed()));

            connect(m_enabledCheckBox, SIGNAL(toggled(bool)), SLOT(HandleToggleDiscontinuousElementBoundaries(bool)));
            connect(m_matchVisual3CheckBox, SIGNAL(toggled(bool)), SLOT(HandleMatchVisual3Toggled(bool)));
        }

        void ContourDockWidget::HandleToggleDiscontinuousElementBoundaries(bool newValue)
        {
            boost::shared_ptr<CutSurfaceContourModule> module = m_appData->GetContourModule();
            if( newValue != module->GetTreatElementBoundariesAsDiscontinuous() )
            {
                module->SetTreatElementBoundariesAsDiscontinuous(newValue);
            }
        }

        void ContourDockWidget::HandleModelChanged(boost::shared_ptr<Model> model)
        {
            m_list->clear();

//            m_elementId->setMinimum(0);
//            m_elementId->setMaximum(model->GetNumberOfElements());
        }

        void ContourDockWidget::HandleAddContourButtonPressed()
        {
            // Add to the model, and rely on the update to change the list widget.
            m_appData->GetContourModule()->AddIsovalue(m_contourSpinBox->value());
        }

        void ContourDockWidget::HandleContourAdded(ElVisFloat newValue)
        {
            QListWidgetItem* item = new QListWidgetItem();
            item->setData(Qt::DisplayRole, newValue);
            m_list->addItem(item);
            m_values[item] = newValue;
        }

        void ContourDockWidget::HandleMatchVisual3Toggled(bool newValue)
        {
            boost::shared_ptr<CutSurfaceContourModule> module = m_appData->GetContourModule();
            if( newValue != module->GetMatchVisual3Contours() )
            {
                module->SetMatchVisual3Contours(newValue);
            }
        }

        void ContourDockWidget::keyPressEvent(QKeyEvent* event)
        {
            boost::shared_ptr<CutSurfaceContourModule> module = m_appData->GetContourModule();
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

