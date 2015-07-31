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

#include <ElVis/Gui/ElementFaceRenderingDockWidget.h>
#include <boost/bind.hpp>

#include <ElVis/QtPropertyBrowser/qttreepropertybrowser.h>

namespace ElVis
{
    namespace Gui
    {
        ElementFaceRenderingDockWidget::ElementFaceRenderingDockWidget(boost::shared_ptr<ApplicationState> appData, QWidget* parent, Qt::WFlags f) :
            QDockWidget("Element Face Rendering", parent, f),
            m_appData(appData),
            m_list(new QListWidget()),
            m_layout(0),
            m_leftLayout(0),
            m_rightLayout(0),
            m_leftWidget(new QWidget()),
            m_rightWidget(new QWidget()),
            m_addFaceIdButton(new QPushButton("Add Face")),
            m_addElementIdButton(new QPushButton("Add Element")),
            m_faceId(new QSpinBox()),
            m_elementId(new QSpinBox()),
            m_boolPropertyManager(new QtBoolPropertyManager()),
            m_checkBoxFactory(new QtCheckBoxFactory()),
            m_faceMap(),
            m_browser(new QtTreePropertyBrowser()),
            m_toleranceSpinBox(new QDoubleSpinBox()),
            m_shownFaces()
        {
            m_list->setSortingEnabled(true);
            setObjectName("ElementFaceRenderingWidget");

            //////////////////////////////////////////
            // Right Widget
            ///////////////////////////////////////////
            m_browser->setFactoryForManager(m_boolPropertyManager, m_checkBoxFactory);
            m_toleranceSpinBox->setMinimum(1e-30);
            m_toleranceSpinBox->setMaximum(10);
            m_toleranceSpinBox->setDecimals(8);
            m_toleranceSpinBox->setValue(m_appData->GetSurfaceSceneView()->GetFaceIntersectionTolerance());
            QLabel* label = new QLabel("Face Intersection Tolerance");

            /////////////////////////////////////////
            // Combine it all
            /////////////////////////////////////////
            QWidget* widget = new QWidget();

            m_layout = new QGridLayout(widget);
            m_layout->addWidget(m_list, 0, 0, 1, 2);

            m_faceId->setMinimum(-1);
            m_faceId->setMaximum(-1);
            m_faceId->setSingleStep(1);
            m_elementId->setMinimum(-1);
            m_elementId->setMaximum(-1);
            m_elementId->setSingleStep(1);

            m_layout->addWidget(m_faceId, 1, 0);
            m_layout->addWidget(m_addFaceIdButton, 1, 1);
            m_layout->addWidget(m_elementId, 3, 0);
            m_layout->addWidget(m_addElementIdButton, 3, 1);

            m_layout->addWidget(m_browser, 0, 2, 1, 2);
            m_layout->addWidget(label, 1, 2);
            m_layout->addWidget(m_toleranceSpinBox, 1, 3);

            m_list->setFocusProxy(this);
            this->setWidget(widget);

            m_appData->GetScene()->OnModelChanged.connect(boost::bind(&ElementFaceRenderingDockWidget::HandleModelChanged, this, _1));
            m_appData->GetScene()->OnSceneInitialized.connect(boost::bind(&ElementFaceRenderingDockWidget::HandleSceneInitialized, this, _1));
            connect(m_addFaceIdButton, SIGNAL(clicked()), this, SLOT(HandleAddFaceButtonPressed()));
            connect(m_addElementIdButton, SIGNAL(clicked()), this, SLOT(HandleAddElementButtonPressed()));
            connect(m_boolPropertyManager, SIGNAL(valueChanged(QtProperty*,bool)), this, SLOT(HandleGeometryFaceMapSelected(QtProperty*,bool)));
            connect(m_toleranceSpinBox, SIGNAL(valueChanged(double)), this, SLOT(HandleToleranceChanged(double)));
        }

        void ElementFaceRenderingDockWidget::HandleToleranceChanged(double x)
        {
            m_appData->GetSurfaceSceneView()->SetFaceIntersectionTolerance(m_toleranceSpinBox->value());
        }

        void ElementFaceRenderingDockWidget::HandleModelChanged(boost::shared_ptr<Model> model)
        {
            m_list->clear();
            m_shownFaces.clear();

            m_faceId->setMinimum(0);
            m_elementId->setMinimum(0);
            //m_elementId->setMaximum(model->GetNumberOfElements());

            int numSurfaceGroups = model->GetNumberOfBoundarySurfaces();
            for(int i = 0; i < numSurfaceGroups; ++i)
            {
                std::string name;
                std::vector<int> ids;
                model->GetBoundarySurface(i, name, ids);

                if( ids.empty() )
                {
                    continue;
                }

                QtProperty* property = m_boolPropertyManager->addProperty(name.c_str());
                m_boolPropertyManager->setValue(property, false);
                m_faceMap[property] = ids;
                m_browser->addProperty(property);
            }
        }

        void ElementFaceRenderingDockWidget::HandleGeometryFaceMapSelected(QtProperty* p, bool v)
        {
            std::map<QtProperty*, std::vector<int> >::const_iterator found = m_faceMap.find(p);
            if( found == m_faceMap.end() ) return;

            m_appData->GetFaceSampler()->SetFaces((*found).second, v);
        }

        void ElementFaceRenderingDockWidget::HandleSceneInitialized(const Scene& s)
        {
            m_faceId->setMaximum(
              static_cast<int>(m_appData->GetScene()->GetModel()->GetNumberOfFaces()));
            m_elementId->setMaximum(
              static_cast<int>(m_appData->GetScene()->GetModel()->GetNumberOfElements()));
        }

        void ElementFaceRenderingDockWidget::HandleAddFaceButtonPressed()
        {
            // Add to the model, and rely on the update to change the list widget.
            int faceId = m_faceId->value();
            std::cout << "Enabling face " << faceId << std::endl;
            m_appData->GetFaceSampler()->EnableFace(faceId);
            QListWidgetItem* item = new QListWidgetItem();
            item->setData(Qt::DisplayRole, faceId);
            m_list->addItem(item);
            m_shownFaces[item] = faceId;
        }

        void ElementFaceRenderingDockWidget::HandleAddElementButtonPressed()
        {
            // Add to the model, and rely on the update to change the list widget.
            unsigned int elemId = m_elementId->value();
            std::vector<unsigned int> facesMatched = m_appData->GetScene()->GetModel()->GetFacesBelongingToElement(elemId);

            if( facesMatched.size() != 0 )
            {
                if( facesMatched.size() > 1 )
                    std::cout << "Enabling faces ";
                else
                    std::cout << "Enabling face ";

                unsigned int listedFaces = 0;
                for (int v : facesMatched)
                {
                    m_appData->GetFaceSampler()->EnableFace(v);
                    std::cout << v;
                    listedFaces++;
                    if( listedFaces < facesMatched.size() )
                        std::cout << ", ";
                    QListWidgetItem* item = new QListWidgetItem();
                    item->setData(Qt::DisplayRole, v);
                    m_list->addItem(item);
                    m_shownFaces[item] = v;
                }
                std::cout << std::endl;
            }
        }

        void ElementFaceRenderingDockWidget::keyPressEvent(QKeyEvent* event)
        {
            if( event->key() == Qt::Key_Delete )
            {
                QList<QListWidgetItem*> items = m_list->selectedItems();

                for(int i = 0; i < items.size(); ++i)
                {
                    QListWidgetItem* item = items.at(i);

                    std::map<QListWidgetItem*, int>::iterator found = m_shownFaces.find(item);
                    if( found != m_shownFaces.end() )
                    {
                        m_appData->GetFaceSampler()->DisableFace((*found).second);
                        m_shownFaces.erase(found);
                    }
                    delete item;
                }

            }
        }


    }
}

