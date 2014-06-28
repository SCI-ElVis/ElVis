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


#ifndef ELVIS_GUI_SCENEITEMSDOCKWIDGET_H
#define ELVIS_GUI_SCENEITEMSDOCKWIDGET_H

#include <QtWidgets/QDockWidget>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QTreeWidgetItem>

#include <boost/shared_ptr.hpp>

#include <ElVis/Core/Scene.h>

namespace ElVis
{
    class PrimaryRayObject;

    namespace Gui
    {
        class ElVisUI;
        class ApplicationState;

        class SceneItemsDockWidget : public QDockWidget
        {
            Q_OBJECT

            public:
                SceneItemsDockWidget(boost::shared_ptr<ApplicationState> appData,
                    QWidget* parent = NULL, Qt::WindowFlags f = 0);

                virtual ~SceneItemsDockWidget();

            public Q_SLOTS:
                void HandleSurfaceObjectAdded(boost::shared_ptr<PrimaryRayObject> cutSurface);
                void SelectionChanged();

            private:
                QTreeWidget* m_listWidget;
                QTreeWidgetItem* m_cutSurfaceItem;
                QTreeWidgetItem* m_objectItem;
                QTreeWidgetItem* m_volumeItem;
                std::map<QTreeWidgetItem*, boost::shared_ptr<PrimaryRayObject> > m_treeObjects;
                boost::shared_ptr<ApplicationState> m_appData;
        };
    }
}

#endif
