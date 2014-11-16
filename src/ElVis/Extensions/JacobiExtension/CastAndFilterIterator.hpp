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
//  Provides an iterator to iterate a list of base objects and return 
//  only those of a given derived type.
//
////////////////////////////////////////////////////////////////////////////////


#ifndef ELVIS_JACOBI_EXTENSION_CAST_AND_FILTER_ITERATOR_HPP
#define ELVIS_JACOBI_EXTENSION_CAST_AND_FILTER_ITERATOR_HPP

#include <boost/shared_ptr.hpp>
#include <boost/iterator/filter_iterator.hpp>

template<typename BaseType, typename DerivedType, typename RawIteratorType>
class CastAndFilterIterator
{
    private:
        struct IsObjectOfDerivedType
        {
            bool operator()(boost::shared_ptr<BaseType> e)
            {
                boost::shared_ptr<DerivedType> AsDerived =
                    boost::dynamic_pointer_cast<DerivedType>(e);
                return (bool)AsDerived;
            }
        };

    public:
        CastAndFilterIterator(RawIteratorType begin, RawIteratorType end) :
            m_begin(begin),
            m_end(end)
        {
        }

        CastAndFilterIterator(const CastAndFilterIterator<BaseType, DerivedType, RawIteratorType>& rhs) :
            m_begin(rhs.m_begin),
            m_end(rhs.m_end)
        {
        }

        CastAndFilterIterator<BaseType, DerivedType, RawIteratorType>& operator=
            (const CastAndFilterIterator<BaseType, DerivedType, RawIteratorType>& rhs)
        {
            m_begin = rhs.m_begin;
            m_end = rhs.m_end;
            return *this;
        }

        typedef boost::filter_iterator<IsObjectOfDerivedType, RawIteratorType> iterator;
        typedef boost::filter_iterator<IsObjectOfDerivedType, RawIteratorType> const_iterator;

        iterator begin()
        {
            return boost::make_filter_iterator(IsObjectOfDerivedType(), m_begin, m_end);
        }

        iterator end()
        {
            return boost::make_filter_iterator(IsObjectOfDerivedType(), m_end, m_end);
        }

        const_iterator begin() const
        {
            return boost::make_filter_iterator(IsObjectOfDerivedType(), m_begin, m_end);
        }

        const_iterator end() const
        {
            return boost::make_filter_iterator(IsObjectOfDerivedType(), m_end, m_end);
        }

    private:
        
        RawIteratorType m_begin;
        RawIteratorType m_end;
};

#endif 


