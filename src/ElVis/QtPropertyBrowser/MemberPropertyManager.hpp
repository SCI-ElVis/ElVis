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

#ifndef ELVIS_QTPROPERTYBROWSER_MEMBER_PROPERTY_MANAGER_H
#define ELVIS_QTPROPERTYBROWSER_MEMBER_PROPERTY_MANAGER_H

#include <ElVis/QtPropertyBrowser/qtpropertymanager.h>
#include <ElVis/QtPropertyBrowser/MemberProperty.hpp>

#include <boost/lexical_cast.hpp>

namespace ElVis
{

    // Unlike the normal QtProperty, this property is directly connected to the
    // data source, so an changes in the property browser will be propogated to the
    // object.  It makes the property browser more similar to the .NET property grid.
    template<typename ClassType, typename DataType>
    class MemberPropertyManager : public QtAbstractPropertyManager
    {
        public:
            typedef MemberProperty<ClassType, DataType> PropertyType;
            typedef typename PropertyType::SetFunctionType SetFunctionType;
            typedef typename PropertyType::GetFunctionType GetFunctionType;
        public:
            MemberPropertyManager() :
                QtAbstractPropertyManager()
            {
            }

            virtual ~MemberPropertyManager() {}

            // Hide AddProperty on purpose.
            PropertyType* addProperty(const QString& name, boost::shared_ptr<ClassType> obj,
                                      const GetFunctionType& getFunction, const SetFunctionType& setFunction)
            {
                PropertyType* property = new PropertyType(this, obj, getFunction, setFunction);
                property->setPropertyName(name);
                d_ptr->m_properties.insert(property);
                return property;
            }

        protected:
            virtual bool hasValue(const QtProperty *property) const
            {
                return true;
            }
            //virtual QIcon valueIcon(const QtProperty *property) const;
            virtual QString valueText(const QtProperty *property) const
            {
                const PropertyType* memberProperty = dynamic_cast<const PropertyType*>(property);
                if( memberProperty )
                {
                    std::string asString = boost::lexical_cast<std::string>(memberProperty->GetValue());
                    return QString(asString.c_str());
                }
                else
                {
                    return "Invalid Property";
                }
            }
            virtual QString displayText(const QtProperty *property) const
            {
                const PropertyType* memberProperty = dynamic_cast<const PropertyType*>(property);
                if( memberProperty )
                {
                    std::string asString = boost::lexical_cast<std::string>(memberProperty->GetValue());
                    return QString(asString.c_str());
                }
                else
                {
                    return "Invalid Property";
                }
            }
            //virtual EchoMode echoMode(const QtProperty *) const;
            virtual void initializeProperty(QtProperty *property)
            {

            }

            virtual void uninitializeProperty(QtProperty *property)
            {

            }

            virtual QtProperty* createProperty()
            {
                return 0;
            }

        private:
            MemberPropertyManager(const MemberPropertyManager& rhs);
            MemberPropertyManager& operator=(const MemberPropertyManager& rhs);
    };
}

#endif
