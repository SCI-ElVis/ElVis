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

#ifndef ELVIS_GUI_PROPERTY_HPP
#define ELVIS_GUI_PROPERTY_HPP

#include <ElVis/QtPropertyBrowser/QtProperty>
#include <ElVis/QtPropertyBrowser/QtIntPropertyManager>
#include <ElVis/QtPropertyBrowser/QtSpinBoxFactory>
#include <ElVis/QtPropertyBrowser/QtCheckBoxFactory>
#include <QObject>
#include <boost/function.hpp>
#include <boost/signals2.hpp>
#include <boost/signals2/connection.hpp>
#include <boost/bind.hpp>

namespace ElVis
{
    namespace Gui
    {
        template<typename T>
        struct DefaultTypeManager;

        template<>
        struct DefaultTypeManager<int>
        {
            typedef QtIntPropertyManager ManagerType;
        };

        template<>
        struct DefaultTypeManager<bool>
        {
            typedef QtBoolPropertyManager ManagerType;
        };

        template<typename T>
        struct DefaultFactoryManager;

        template<>
        struct DefaultFactoryManager<int>
        {
            typedef QtSpinBoxFactory FactoryType;
        };

        template<>
        struct DefaultFactoryManager<bool>
        {
            typedef QtCheckBoxFactory FactoryType;
        };

        // I can't make Property a Q_OBJECT, so this base class allows me to connect the subscriptions
        // that I would prefer to have in Property.
        class BaseProperty : public QObject
        {
            Q_OBJECT

            public:
                void AddSubscription(QtAbstractPropertyManager* manager)
                {
                    connect(manager, SIGNAL(propertyChanged(QtProperty*)), SLOT(HandlePropertyChangedByInterface(QtProperty*)));
                }

            public Q_SLOTS:
                void HandlePropertyChangedByInterface(QtProperty* property)
                {
                    DoHandlePropertyChangedByInterface(property);
                }

            protected:
                virtual void DoHandlePropertyChangedByInterface(QtProperty* property) = 0;
        };

        template<typename T, typename ManagerType = typename DefaultTypeManager<T>::ManagerType,
                 typename FactoryType = typename DefaultFactoryManager<T>::FactoryType >
        class Property : public BaseProperty
        {

            public:
                typedef boost::function<void (T)> SetFunctionType;
                typedef boost::function<T () > GetFunctionType;

            public:
                Property(const std::string& name, const SetFunctionType& setFunc, const GetFunctionType& getFunc,
                         boost::signals2::signal<void (T)>& changedSignal) :
                    m_factory(new FactoryType()),
                    m_property(0),
                    m_manager(new ManagerType()),
                    m_setFunc(setFunc),
                    m_getFunc(getFunc)
                {
                    m_property = m_manager->addProperty(name.c_str());
                    m_manager->setValue(m_property, m_getFunc());
                    changedSignal.connect(boost::bind(&Property::HandleUnderlyingDataChanged, this, _1));
                    AddSubscription(m_manager);
                }

                void SetupPropertyManagers(QtAbstractPropertyBrowser* browser)
                {
                    browser->setFactoryForManager(m_manager, m_factory);
                }

                virtual ~Property()
                {
                    //m_connection.disconnect();
                }

                QtProperty* GetProperty() { return m_property; }

                void HandleUnderlyingDataChanged(const T& newValue)
                {
                    if( m_manager->value(m_property) != newValue )
                    {
                        m_manager->setValue(m_property, newValue);
                    }
                }

            private:
                void DoHandlePropertyChangedByInterface(QtProperty* property)
                {
                    T newValue = m_manager->value(property);
                    if( newValue != m_getFunc() )
                    {
                        m_setFunc(newValue);
                    }
                }

                FactoryType* m_factory;
                QtProperty* m_property;
                ManagerType* m_manager;
                SetFunctionType m_setFunc;
                GetFunctionType m_getFunc;
                //boost::signals2::connection m_connection;
        };

    }
}

#endif
