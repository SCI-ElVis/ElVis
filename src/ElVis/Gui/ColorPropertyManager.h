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

#ifndef ELVIS_GUI_COLOR_PROPERTY_MANAGER_H
#define ELVIS_GUI_COLOR_PROPERTY_MANAGER_H

#include <ElVis/QtPropertyBrowser/qtpropertymanager.h>
#include <ElVis/QtPropertyBrowser/qteditorfactory.h>
#include <ElVis/QtPropertyBrowser/MemberProperty.hpp>
#include <ElVis/QtPropertyBrowser/MemberPropertyManager.hpp>
#include <ElVis/Core/Color.h>

#include <boost/bind.hpp>
#include <boost/foreach.hpp>

namespace ElVis
{
    class ColorPropertyManager : public QtAbstractPropertyManager
    {
        Q_OBJECT

        public:
            ColorPropertyManager() :
                QtAbstractPropertyManager(),
                m_redPropertyManager(new QtIntPropertyManager()),
                m_greenPropertyManager(new QtIntPropertyManager()),
                m_bluePropertyManager(new QtIntPropertyManager()),
                m_alphaPropertyManager(new QtIntPropertyManager()),
                m_intSpinBoxFactory(new QtSpinBoxFactory()),
                m_groupPropertyManager(new QtGroupPropertyManager()),
                m_data()
            {
            }

            virtual ~ColorPropertyManager() {}

            // Hide AddProperty on purpose.
            QtProperty* addProperty(const QString& name, Color& p)
            {
                QtProperty* topLevelItem = m_groupPropertyManager->addProperty(name);
                Data data(p);
                data.RedProperty = this->m_redPropertyManager->addProperty("Red");
                data.GreenProperty = this->m_greenPropertyManager->addProperty("Green");
                data.BlueProperty = this->m_bluePropertyManager->addProperty("Blue");
                data.AlphaProperty = this->m_alphaPropertyManager->addProperty("Alpha");

                topLevelItem->addSubProperty(data.RedProperty);
                topLevelItem->addSubProperty(data.GreenProperty);
                topLevelItem->addSubProperty(data.BlueProperty);
                topLevelItem->addSubProperty(data.AlphaProperty);

                m_redPropertyManager->setValue(data.RedProperty, p.Red());
                m_greenPropertyManager->setValue(data.GreenProperty, p.Green());
                m_bluePropertyManager->setValue(data.BlueProperty, p.Blue());
                m_alphaPropertyManager->setValue(data.AlphaProperty, p.Alpha());

                //////////////////////////
                connect(m_redPropertyManager, SIGNAL(valueChanged(QtProperty*,int)), this, SLOT(RedChanged(QtProperty*,int)));
                connect(m_greenPropertyManager, SIGNAL(valueChanged(QtProperty*,int)), this, SLOT(GreenChanged(QtProperty*,int)));
                connect(m_bluePropertyManager, SIGNAL(valueChanged(QtProperty*,int)), this, SLOT(BlueChanged(QtProperty*,int)));
                connect(m_alphaPropertyManager, SIGNAL(valueChanged(QtProperty*,int)), this, SLOT(AlphaChanged(QtProperty*,int)));

                p.OnColorChanged.connect(boost::bind(&ColorPropertyManager::UpdateViewForRedChange, this, m_redPropertyManager, data.RedProperty, _1));
                p.OnColorChanged.connect(boost::bind(&ColorPropertyManager::UpdateViewForGreenChange, this, m_greenPropertyManager, data.GreenProperty, _1));
                p.OnColorChanged.connect(boost::bind(&ColorPropertyManager::UpdateViewForBlueChange, this, m_bluePropertyManager, data.BlueProperty, _1));
                p.OnColorChanged.connect(boost::bind(&ColorPropertyManager::UpdateViewForAlphaChange, this, m_alphaPropertyManager, data.AlphaProperty, _1));

                typedef std::map<QtProperty*, Data>::value_type ValueType;
                ValueType value(topLevelItem, data);
                m_data.insert(value);
                return topLevelItem;
            }

            void SetupPropertyManagers(QtAbstractPropertyBrowser* browser, QObject* parent)
            {
                // Main purpose is to setup default editors.
                browser->setFactoryForManager(this->m_redPropertyManager, this->m_intSpinBoxFactory);
                browser->setFactoryForManager(this->m_greenPropertyManager, this->m_intSpinBoxFactory);
                browser->setFactoryForManager(this->m_bluePropertyManager, this->m_intSpinBoxFactory);
                browser->setFactoryForManager(this->m_alphaPropertyManager, this->m_intSpinBoxFactory);
            }

        Q_SIGNALS:
            void OnColorChangedInGui();

        protected Q_SLOTS:

            void UpdateViewForRedChange(QtIntPropertyManager* manager, QtProperty* property, const Color& p)
            {
                manager->setValue(property, p.RedAsInt());
            }

            void UpdateViewForGreenChange(QtIntPropertyManager* manager, QtProperty* property, const Color& p)
            {
                manager->setValue(property, p.GreenAsInt());
            }

            void UpdateViewForBlueChange(QtIntPropertyManager* manager, QtProperty* property, const Color& p)
            {
                manager->setValue(property, p.BlueAsInt());
            }

            void UpdateViewForAlphaChange(QtIntPropertyManager* manager, QtProperty* property, const Color& p)
            {
                manager->setValue(property, p.AlphaAsInt());
            }

            void RedChanged(QtProperty* property, int value)
            {
                Color& p = FindColorForProperty(property);
                if( p.RedAsInt() != value )
                {
                    p.SetRed(value);
                    OnColorChangedInGui();
                }
            }

            void GreenChanged(QtProperty* property, int value)
            {
                Color& p = FindColorForProperty(property);
                if( p.GreenAsInt() != value )
                {
                    p.SetGreen(value);
                    OnColorChangedInGui();
                }
            }

            void BlueChanged(QtProperty* property, int value)
            {
                Color& p = FindColorForProperty(property);
                if( p.BlueAsInt() != value )
                {
                    p.SetBlue(value);
                    OnColorChangedInGui();
                }
            }

            void AlphaChanged(QtProperty* property, int value)
            {
                Color& p = FindColorForProperty(property);
                if( p.AlphaAsInt() != value )
                {
                    p.SetAlpha(value);
                    OnColorChangedInGui();
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
            ColorPropertyManager(const ColorPropertyManager& rhs);
            ColorPropertyManager& operator=(const ColorPropertyManager& rhs);



            struct Data
            {
                public:
                    Data(Color& p) :
                        Obj(p),
                        RedProperty(0),
                        GreenProperty(0),
                        BlueProperty(0),
                        AlphaProperty(0)
                    {

                    }

                    Data(const Data& rhs) :
                        Obj(rhs.Obj),
                        RedProperty(rhs.RedProperty),
                        GreenProperty(rhs.GreenProperty),
                        BlueProperty(rhs.BlueProperty),
                        AlphaProperty(rhs.AlphaProperty)
                    {
                    }

                    Color& Obj;
                    QtProperty* RedProperty;
                    QtProperty* GreenProperty;
                    QtProperty* BlueProperty;
                    QtProperty* AlphaProperty;

                    bool ContainsProperty(QtProperty* test) const
                    {
                        return RedProperty == test ||
                               GreenProperty == test ||
                               BlueProperty == test ||
                               AlphaProperty == test;
                    }

                private:
                    Data& operator=(const Data&);
            };

            Color& FindColorForProperty(QtProperty* test) const
            {
                typedef std::map<QtProperty*, Data>::value_type IterType;
                BOOST_FOREACH(IterType iter, m_data)
                {
                    if( iter.second.ContainsProperty(test))
                    {
                        return iter.second.Obj;
                    }
                }

                static Color defaultResult;
                return defaultResult;
            }

            QtIntPropertyManager* m_redPropertyManager;
            QtIntPropertyManager* m_greenPropertyManager;
            QtIntPropertyManager* m_bluePropertyManager;
            QtIntPropertyManager* m_alphaPropertyManager;
            QtSpinBoxFactory* m_intSpinBoxFactory;
            QtGroupPropertyManager* m_groupPropertyManager;
            std::map<QtProperty*, Data> m_data;

    };
}

#endif
