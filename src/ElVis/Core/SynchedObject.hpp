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

#ifndef ELVIS_CORE_SYNCHED_OBJECT_HPP
#define ELVIS_CORE_SYNCHED_OBJECT_HPP

#include <boost/signals2.hpp>

namespace ElVis
{
    template<typename T>
    class SynchedObject
    {
        public:
            SynchedObject() :
                m_data(),
                m_isDirty(true)
            {
            }

            explicit SynchedObject(const T& value) :
                m_data(value),
                m_isDirty(true)
            {
            }

            SynchedObject(const SynchedObject<T>& rhs) :
                m_data(rhs.m_data),
                m_isDirty(true)
            {
            }

            SynchedObject<T>& operator=(const SynchedObject<T>& rhs)
            {
                Assign(rhs.m_data);
                return *this;
            }

            SynchedObject<T>& operator=(const T& rhs)
            {
                Assign(rhs);
                return *this;
            }

            ~SynchedObject()
            {
            }

            const T& operator*() const { return m_data; }
            T& operator*() { return m_data; }

            bool IsDirty() const { return m_isDirty; }

            void MarkDirty()
            {
                Mark(true);
            }

            void MarkClean()
            {
                Mark(false);
            }


            boost::signals2::signal<void (const SynchedObject<T>&)> OnDirtyFlagChanged;

        private:
            void Assign(const T& data)
            {
                if( m_data != data )
                {
                    m_data = data;
                    MarkDirty();
                }
            }

            void Mark(bool newValue)
            {
                if( newValue != m_isDirty )
                {
                    m_isDirty = newValue;
                    OnDirtyFlagChanged(*this);
                }
            }

            T m_data;
            bool m_isDirty;
    };
}

#endif
