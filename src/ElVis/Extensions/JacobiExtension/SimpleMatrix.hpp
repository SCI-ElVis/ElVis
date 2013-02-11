////////////////////////////////////////////////////////////////////////////////
//
//  File: hoPointTransformations.hpp
//
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
//  A very simple, unoptimized matrix class.
//
////////////////////////////////////////////////////////////////////////////////


#ifndef ELVIS_JACOBI_EXTENSION_ELVIS_HIGH_ORDER_ISOSURFACE_SIMPLE_MATRIX_H
#define ELVIS_JACOBI_EXTENSION_ELVIS_HIGH_ORDER_ISOSURFACE_SIMPLE_MATRIX_H

#include <ElVis/Core/Vector.hpp>
#include <boost/array.hpp>

#include <ElVis/Core/Vector.hpp>

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace ElVis
{
    namespace JacobiExtension
    {
        enum MatrixOrder { ROW_MAJOR, COLUMN_MAJOR };

        template<typename DataType, unsigned int numRows, unsigned int numColumns>
        class Matrix
        {
        public:
            Matrix()
            {
            }

            ~Matrix()
            {
            }

            Matrix(const Matrix<DataType, numRows, numColumns>& rhs)
            {
                for(unsigned int i = 0; i < numRows; ++i)
                {
                    for(unsigned int j = 0; j < numColumns; ++j)
                    {
                        m_data[i][j] = rhs.m_data[i][j];
                    }
                }
            }


            Matrix(boost::array<DataType, numRows*numColumns>& values,
                MatrixOrder o)
            {
                if( o == ROW_MAJOR )
                {
                    unsigned int i = 0;
                    for(unsigned int row = 0; row < numRows; ++row)
                    {
                        for(unsigned int col = 0; col < numColumns; ++col)
                        {
                            m_data[row][col] = values[i];
                            ++i;
                        }
                    }
                }
                else
                {
                    unsigned int i = 0;
                    for(unsigned int col = 0; col < numColumns; ++col)
                    {
                        for(unsigned int row = 0; row < numRows; ++row)
                        {
                            m_data[row][col] = values[i];
                            ++i;
                        }
                    }
                }
            }

            Matrix& operator=(const Matrix& rhs)
            {
                for(unsigned int i = 0; i < numRows; ++i)
                {
                    for(unsigned int j = 0; j < numRows; ++j)
                    {
                        m_data[i][j] = rhs.m_data[i][j];
                    }
                }

                return *this;
            }

            const DataType& getData(unsigned int row, unsigned int column) const
            {
                assert(row < numRows && column < numColumns);
                return m_data[row][column];
            }

            DataType& getData(unsigned int row, unsigned int column)
            {
                assert(row < numRows && column < numColumns);
                return m_data[row][column];
            }

            DataType& setData(unsigned int row, unsigned int column)
            {
                assert(row < numRows && column < numColumns);
                return m_data[row][column];
            }

            DataType infinityNorm() const
            {
                DataType result = -std::numeric_limits<DataType>::max();
                for(unsigned int i = 0; i < numRows; ++i)
                {
                    DataType temp = 0.0;
                    for(unsigned int j = 0; j < numColumns; ++j)
                    {
                        temp += fabs(getData(i,j));
                    }

                    if( temp > result )
                    {
                        result = temp;
                    }
                }

                return result;
            }

            void transpose()
            {
                assert(numRows == numColumns);
                for(unsigned int i = 0; i < numRows; ++i)
                {
                    for(unsigned int j = i; j < numColumns; ++j)
                    {
                        if( i != j )
                        {
                            std::swap(m_data[i][j], m_data[j][i]);
                        }
                    }
                }
            }

            DataType& operator()(unsigned int i, unsigned int j)
            {
                return m_data[i][j];
            }

            const DataType& operator()(unsigned int i, unsigned int j) const
            {
                return m_data[i][j];
            }

            const DataType& operator[](unsigned int i) const
            {
                unsigned int row = i/numColumns;
                unsigned int col = i%numRows;
                return getData(row, col);
            }

            DataType& operator[](unsigned int i)
            {
                unsigned int row = i/numColumns;
                unsigned int col = i%numRows;
                return getData(row, col);
            }

        private:
            DataType m_data[numRows][numColumns];
        };

        template<typename DataType, unsigned int lhsRows, unsigned int lhsColumns,
            unsigned int rhsRows, unsigned int rhsColumns>
            Matrix<DataType, lhsRows, rhsColumns> operator*(
            const Matrix<DataType, lhsRows, lhsColumns>& lhs,
            const Matrix<DataType, rhsRows, rhsColumns>& rhs)
        {
            Matrix<DataType, lhsRows, rhsColumns> result;

            for(unsigned int i = 0; i < lhsRows; ++i)
            {
                for(unsigned int j = 0; j < rhsColumns; ++j)
                {
                    DataType t = DataType(0);

                    // Set the result(i,j) element.
                    for(unsigned int k = 0; k < lhsColumns; ++k)
                    {
                        t += lhs(i,k)*rhs(k,j);
                    }
                    result(i,j) = t;
                }
            }

            return result;
        }

        template<typename space, typename DataType, unsigned int numRows, unsigned int numColumns>
        ElVis::Vector<DataType, space> operator*(
            const Matrix<DataType, numRows, numColumns>& lhs,
            const Vector<DataType, space>& rhs)
        {
            Vector<DataType, space> result(rhs.GetRows());
            for(unsigned int i = 0; i < numRows; ++i)
            {
                DataType elementValue = 0.0;
                for(unsigned int j = 0; j < numColumns; ++j)
                {
                    elementValue += lhs.getData(i,j)*rhs[j];
                }
                result[i] = elementValue;
            }

            return result;
        }

        //rt::Vector<4, double> operator*(const Matrix<double, 4, 4>& lhs, const rt::Vector<4, double>& rhs)
        //{
        //    rt::Vector<4, double> result;
        //    return result;
        //}
        //     rt::Vector<4, double> operator*(const Matrix<double, 4, 4>& lhs, const rt::Vector<4, double>& rhs)
        //     {
        //         rt::Vector<4, double> result;
        //         for(unsigned int i = 0; i < 4; ++i)
        //         {
        //             double elementValue = 0.0;
        //             for(unsigned int j = 0; j < 4; ++j)
        //             {
        //                 elementValue += lhs.getData(i,j)*rhs[j];
        //             }
        //             result[i] = elementValue;
        //         }
        //
        //         return result;
        //     }
    }
}

#endif //ELVIS_HIGH_ORDER_ISOSURFACE_SIMPLE_MATRIX_H

