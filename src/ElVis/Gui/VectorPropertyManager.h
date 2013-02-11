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

#ifndef ELVIS_GUI_VECTOR_PROPERTY_MANAGER_H
#define ELVIS_GUI_VECTOR_PROPERTY_MANAGER_H

#include <ElVis/QtPropertyBrowser/qtpropertymanager.h>
#include <ElVis/QtPropertyBrowser/qteditorfactory.h>
#include <ElVis/QtPropertyBrowser/MemberProperty.hpp>
#include <ElVis/QtPropertyBrowser/MemberPropertyManager.hpp>
#include <ElVis/Core/Vector.hpp>

#include <boost/bind.hpp>
#include <boost/foreach.hpp>

namespace ElVis
{
    class VectorPropertyManager : public QtAbstractPropertyManager
    {
        Q_OBJECT

        public:
            VectorPropertyManager() :
                QtAbstractPropertyManager(),
                m_xPropertyManager(new QtDoublePropertyManager()),
                m_yPropertyManager(new QtDoublePropertyManager()),
                m_zPropertyManager(new QtDoublePropertyManager()),
                m_groupPropertyManager(new QtGroupPropertyManager()),
                m_doubleSpinBoxFactory(new QtDoubleSpinBoxFactory()),
                m_data()
            {
            }

            virtual ~VectorPropertyManager() {}

            // Hide AddProperty on purpose.
            QtProperty* addProperty(const QString& name, WorldVector& p)
            {
                QtProperty* topLevelItem = m_groupPropertyManager->addProperty(name);
                Data data(p);
                data.XProperty = this->m_xPropertyManager->addProperty("X");
                data.YProperty = this->m_yPropertyManager->addProperty("Y");
                data.ZProperty = this->m_zPropertyManager->addProperty("Z");

                topLevelItem->addSubProperty(data.XProperty);
                topLevelItem->addSubProperty(data.YProperty);
                topLevelItem->addSubProperty(data.ZProperty);

                m_xPropertyManager->setValue(data.XProperty, p.x());
                m_yPropertyManager->setValue(data.YProperty, p.y());
                m_zPropertyManager->setValue(data.ZProperty, p.z());

                m_xPropertyManager->setDecimals(data.XProperty, 8);
                m_yPropertyManager->setDecimals(data.YProperty, 8);
                m_zPropertyManager->setDecimals(data.ZProperty, 8);

                connect(m_xPropertyManager, SIGNAL(valueChanged(QtProperty*,double)), this, SLOT(XChanged(QtProperty*,double)));
                connect(m_yPropertyManager, SIGNAL(valueChanged(QtProperty*,double)), this, SLOT(YChanged(QtProperty*,double)));
                connect(m_zPropertyManager, SIGNAL(valueChanged(QtProperty*,double)), this, SLOT(ZChanged(QtProperty*,double)));

                p.OnVectorChanged.connect(boost::bind(&VectorPropertyManager::UpdateViewForVectorXChange, this, m_xPropertyManager, data.XProperty, _1));
                p.OnVectorChanged.connect(boost::bind(&VectorPropertyManager::UpdateViewForVectorYChange, this, m_yPropertyManager, data.YProperty, _1));
                p.OnVectorChanged.connect(boost::bind(&VectorPropertyManager::UpdateViewForVectorZChange, this, m_zPropertyManager, data.ZProperty, _1));

                typedef std::map<QtProperty*, Data>::value_type ValueType;
                ValueType value(topLevelItem, data);
                m_data.insert(value);
                return topLevelItem;
            }

            void SetupPropertyManagers(QtAbstractPropertyBrowser* browser, QObject* parent)
            {
                // Main purpose is to setup default editors.
                browser->setFactoryForManager(this->m_xPropertyManager, this->m_doubleSpinBoxFactory);
                browser->setFactoryForManager(this->m_yPropertyManager, this->m_doubleSpinBoxFactory);
                browser->setFactoryForManager(this->m_zPropertyManager, this->m_doubleSpinBoxFactory);
            }

        protected Q_SLOTS:

            void UpdateViewForVectorXChange(QtDoublePropertyManager* manager, QtProperty* property, const WorldVector& p)
            {
                manager->setValue(property, p.x());
            }

            void UpdateViewForVectorYChange(QtDoublePropertyManager* manager, QtProperty* property, const WorldVector& p)
            {
                manager->setValue(property, p.y());
            }

            void UpdateViewForVectorZChange(QtDoublePropertyManager* manager, QtProperty* property, const WorldVector& p)
            {
                manager->setValue(property, p.z());
            }

            void XChanged(QtProperty* property, double value)
            {
                WorldVector& p = FindPointForProperty(property);
                if( p.x() != value )
                {
                    p.SetX(value);
                }
            }

            void YChanged(QtProperty* property, double value)
            {
                WorldVector& p = FindPointForProperty(property);
                if( p.y() != value )
                {
                    p.SetY(value);
                }
            }

            void ZChanged(QtProperty* property, double value)
            {
                WorldVector& p = FindPointForProperty(property);
                if( p.z() != value )
                {
                    p.SetZ(value);
                }
            }


        protected:
            virtual void initializeProperty(QtProperty *property)
            {

            }

            virtual QtProperty* createProperty()
            {
                return 0;
            }

        private:
            VectorPropertyManager(const VectorPropertyManager& rhs);
            VectorPropertyManager& operator=(const VectorPropertyManager& rhs);


            struct Data
            {
                public:
                    Data(WorldVector& p) :
                        Point(p)
                    {

                    }

                    Data(const Data& rhs) :
                        Point(rhs.Point),
                        XProperty(rhs.XProperty),
                        YProperty(rhs.YProperty),
                        ZProperty(rhs.ZProperty)
                    {
                    }

                    WorldVector& Point;
                    QtProperty* XProperty;
                    QtProperty* YProperty;
                    QtProperty* ZProperty;

                    bool ContainsDoubleProperty(QtProperty* test) const
                    {
                        return XProperty == test ||
                               YProperty == test ||
                               ZProperty == test;
                    }

                private:
                    Data& operator=(const Data&);
            };

            WorldVector& FindPointForProperty(QtProperty* test) const
            {
                typedef std::map<QtProperty*, Data>::value_type IterType;
                BOOST_FOREACH(IterType iter, m_data)
                {
                    if( iter.second.ContainsDoubleProperty(test))
                    {
                        return iter.second.Point;
                    }
                }

                static WorldVector defaultResult;
                return defaultResult;
            }

            QtDoublePropertyManager* m_xPropertyManager;
            QtDoublePropertyManager* m_yPropertyManager;
            QtDoublePropertyManager* m_zPropertyManager;
            QtGroupPropertyManager* m_groupPropertyManager;
            QtDoubleSpinBoxFactory* m_doubleSpinBoxFactory;
            std::map<QtProperty*, Data> m_data;

    };
}

#endif
