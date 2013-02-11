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

#ifndef ELVIS_GUI_CUT_PLANE_PROPERTY_MANAGER_H
#define ELVIS_GUI_CUT_PLANE_PROPERTY_MANAGER_H

#include <ElVis/QtPropertyBrowser/qtpropertymanager.h>
#include <ElVis/QtPropertyBrowser/qteditorfactory.h>
#include <ElVis/QtPropertyBrowser/MemberProperty.hpp>
#include <ElVis/QtPropertyBrowser/MemberPropertyManager.hpp>
#include <ElVis/Gui/PointPropertyManager.h>

#include <ElVis/Core/Plane.h>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>

namespace ElVis
{
    class PlanePropertyManager : public QtAbstractPropertyManager
    {
        Q_OBJECT

        public:
            PlanePropertyManager() :
                QtAbstractPropertyManager(),
                m_normalPropertyManager(new PointPropertyManager()),
                m_pointPropertyManager(new PointPropertyManager()),
                m_groupPropertyManager(new QtGroupPropertyManager())
            {
            }

            virtual ~PlanePropertyManager() {}

            // Hide AddProperty on purpose.
            QtProperty* addProperty(const QString& name, boost::shared_ptr<Plane> obj)
            {
                QtProperty* topLevelItem = m_groupPropertyManager->addProperty(name);


                QtProperty* normalProperty = m_normalPropertyManager->addProperty("Normal", obj->GetNormal());
                QtProperty* pointProperty = m_pointPropertyManager->addProperty("Point", obj->GetPoint());

                topLevelItem->addSubProperty(normalProperty);
                topLevelItem->addSubProperty(pointProperty);
                return topLevelItem;
            }

            void SetupPropertyManagers(QtAbstractPropertyBrowser* browser, QObject* parent)
            {
                // Main purpose is to setup default editors.
                m_normalPropertyManager->SetupPropertyManagers(browser, parent);
                m_pointPropertyManager->SetupPropertyManagers(browser, parent);
            }

        protected Q_SLOTS:


        protected:

            virtual void initializeProperty(QtProperty *property)
            {

            }

            virtual QtProperty* createProperty()
            {
                return 0;
            }

        private:
            PlanePropertyManager(const PlanePropertyManager& rhs);
            PlanePropertyManager& operator=(const PlanePropertyManager& rhs);


            PointPropertyManager* m_normalPropertyManager;
            PointPropertyManager* m_pointPropertyManager;
            QtGroupPropertyManager* m_groupPropertyManager;

    };
}

#endif
