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


#include <ElVis/Gui/SceneItemsDockWidget.h>
#include <ElVis/Gui/ApplicationState.h>

#include <ElVis/Core/PrimaryRayModule.h>
#include <ElVis/Core/SampleVolumeSamplerObject.h>
#include <ElVis/Core/PrimaryRayObject.h>

#include <boost/bind.hpp>

namespace ElVis
{
    namespace Gui
    {
        SceneItemsDockWidget::SceneItemsDockWidget(boost::shared_ptr<ApplicationState> appData,
            QWidget* parent, Qt::WindowFlags f) : QDockWidget(QString::fromUtf8("Pipeline Browser"), parent, f),
            m_listWidget(0),
            m_cutSurfaceItem(0),
            m_objectItem(0),
            m_volumeItem(0),
            m_treeObjects(),
            m_appData(appData)
        {
            this->setObjectName("SceneItems");
            m_listWidget = new QTreeWidget(this);
            m_listWidget->setSelectionMode(QAbstractItemView::SingleSelection);
            this->setWidget(m_listWidget);

            m_cutSurfaceItem = new QTreeWidgetItem(m_listWidget);
            m_cutSurfaceItem->setText(0, QString::fromUtf8("Cut Surfaces"));
            m_objectItem = new QTreeWidgetItem(m_listWidget);
            m_objectItem->setText(0, QString::fromUtf8("Surfaces"));
            m_volumeItem = new QTreeWidgetItem(m_listWidget);
            m_volumeItem->setText(0, QString::fromUtf8("Volume"));

            boost::shared_ptr<PrimaryRayModule> primaryRayModule = m_appData->GetSurfaceSceneView()->GetPrimaryRayModule();
            primaryRayModule->OnObjectAdded.connect(boost::bind(&SceneItemsDockWidget::HandleSurfaceObjectAdded, this, _1));

            connect(m_listWidget, SIGNAL(itemSelectionChanged()), this, SLOT(SelectionChanged()));
        }

        SceneItemsDockWidget::~SceneItemsDockWidget()
        {
        }

        void SceneItemsDockWidget::HandleSurfaceObjectAdded(boost::shared_ptr<PrimaryRayObject> obj)
        {
            // If a sample object, then use as a cut surface.
            boost::shared_ptr<SampleVolumeSamplerObject> cutSurface = boost::dynamic_pointer_cast<SampleVolumeSamplerObject>(obj);
            std::cout << "HandleSurfaceObjectAdded" << std::endl;
            if( cutSurface )
            {
                std::cout << "HandleSurfaceObjectAdded is a sample volume samples object." << std::endl;
                QTreeWidgetItem* surfaceItem = new QTreeWidgetItem(m_cutSurfaceItem);
                surfaceItem->setText(0, "Surface");
                m_treeObjects[surfaceItem] = obj;
            }
        }

        void SceneItemsDockWidget::SelectionChanged()
        {
            QList<QTreeWidgetItem *> selectedItems = m_listWidget->selectedItems();
            if( selectedItems.count() == 0 ) return;

            QTreeWidgetItem* selectedItem = selectedItems.first();

            typedef std::map<QTreeWidgetItem*, boost::shared_ptr<PrimaryRayObject> >::const_iterator IterType;
            IterType found = m_treeObjects.find(selectedItem);
            if( found != m_treeObjects.end())
            {
                m_appData->SetSelectedObject((*found).second);
            }
        }
    }
}
