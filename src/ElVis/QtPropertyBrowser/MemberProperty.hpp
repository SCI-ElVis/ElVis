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

#ifndef ELVIS_QTPROPERTYBROWSER_MEMBER_PROPERTY_H
#define ELVIS_QTPROPERTYBROWSER_MEMBER_PROPERTY_H

#include <ElVis/QtPropertyBrowser/QtProperty>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <iostream>

namespace ElVis
{
    // Unlike the normal QtProperty, this property is directly connected to the
    // data source, so an changes in the property browser will be propogated to the
    // object.  It makes the property browser more similar to the .NET property grid.
    template<typename ClassType, typename DataType>
    class MemberProperty : public QtProperty
    {
        public:
            typedef boost::function<DataType (ClassType*)> GetFunctionType;
            typedef boost::function<void (ClassType*, DataType)> SetFunctionType;

        public:
            MemberProperty(QtAbstractPropertyManager* owner, boost::shared_ptr<ClassType> obj, const GetFunctionType& getFunction, const SetFunctionType& setFunction) :
                QtProperty(owner),
                m_obj(obj),
                m_getFunction(getFunction),
                m_setFunction(setFunction)
            {
            }

            DataType GetValue() const
            {
                std::cout << "Getting value" << std::endl;
                return m_getFunction(m_obj.get());
            }

            void SetValue(const DataType& value)
            {
                std::cout << "Setting values" << std::endl;
                m_setFunction(m_obj.get(), value);
            }

            virtual ~MemberProperty() {}

        private:
            MemberProperty(const MemberProperty& rhs);
            MemberProperty& operator=(const MemberProperty& rhs);

            boost::shared_ptr<ClassType> m_obj;
            GetFunctionType m_getFunction;
            SetFunctionType m_setFunction;
    };
}

#endif
