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

#ifndef ELVIS_INTERVAL_MATRIX_CU
#define ELVIS_INTERVAL_MATRIX_CU

#include <ElVis/Core/Cuda.h>
#include <ElVis/Core/Float.cu>
#include <ElVis/Core/Interval.hpp>
#include <ElVis/Core/IntervalPoint.cu>
#include <ElVis/Core/matrix.cu>

namespace ElVis
{
    template<unsigned int Rows, unsigned int Columns>
    class IntervalMatrix
    {
        public:
            ELVIS_DEVICE IntervalMatrix()
            {
                m_data[0] = Interval<ElVisFloat>(MAKE_FLOAT(1.0));
                m_data[1] = Interval<ElVisFloat>(MAKE_FLOAT(0.0));
                m_data[2] = Interval<ElVisFloat>(MAKE_FLOAT(0.0));

                m_data[3] = Interval<ElVisFloat>(MAKE_FLOAT(0.0));
                m_data[4] = Interval<ElVisFloat>(MAKE_FLOAT(1.0));
                m_data[5] = Interval<ElVisFloat>(MAKE_FLOAT(0.0));

                m_data[6] = Interval<ElVisFloat>(MAKE_FLOAT(0.0));
                m_data[7] = Interval<ElVisFloat>(MAKE_FLOAT(0.0));
                m_data[8] = Interval<ElVisFloat>(MAKE_FLOAT(1.0));

            }

            ELVIS_DEVICE IntervalMatrix& operator=(const IntervalMatrix& rhs)
            {
                m_data[0] = rhs.m_data[0];
                m_data[1] = rhs.m_data[1];
                m_data[2] = rhs.m_data[2];

                m_data[3] = rhs.m_data[3];
                m_data[4] = rhs.m_data[4];
                m_data[5] = rhs.m_data[5];

                m_data[6] = rhs.m_data[6];
                m_data[7] = rhs.m_data[7];
                m_data[8] = rhs.m_data[8];
                return *this;
            }

            ELVIS_DEVICE IntervalMatrix(const IntervalMatrix& rhs)
            {
                for(unsigned int i = 0; i < Rows*Columns; ++i)
                {
                    m_data[i] = rhs.m_data[i];
                }
            }

            ELVIS_DEVICE ElVis::Matrix<Rows, Columns> GetMidpoint() const
            {
                ElVis::Matrix<Rows, Columns> result;
                for(unsigned int i = 0; i < Rows*Columns; ++i)
                {
                    result[i] = m_data[i].GetMidpoint();
                }
                return result;
            }

            ELVIS_DEVICE const Interval<ElVisFloat>& operator[](unsigned int i) const { return m_data[i]; }
            ELVIS_DEVICE Interval<ElVisFloat>& operator[](unsigned int i) { return m_data[i]; }

            ELVIS_DEVICE ElVisFloat Norm() const
            {
                ElVisFloat t0 = fmaxf(m_data[0].GetMax(), m_data[1].GetMax());
                ElVisFloat t1 = fmaxf(m_data[2].GetMax(), m_data[3].GetMax());
                ElVisFloat t2 = fmaxf(m_data[4].GetMax(), m_data[5].GetMax());
                ElVisFloat t3 = fmaxf(m_data[6].GetMax(), m_data[7].GetMax());
                return fmaxf(t0, fmaxf(t1, fmaxf(t2, fmaxf(m_data[8].GetMax(), t3))));
            }

        private:

            Interval<ElVisFloat> m_data[Rows*Columns];
    };

    ELVIS_DEVICE Interval<ElVisFloat> Determinant(const IntervalMatrix<3,3>& rhs)
    {
        return rhs[0]*rhs[4]*rhs[8] -
                rhs[0]*rhs[5]*rhs[7] -
                rhs[1]*rhs[3]*rhs[8] +
                rhs[1]*rhs[5]*rhs[6] +
                rhs[2]*rhs[3]*rhs[7] -
                rhs[2]*rhs[4]*rhs[6];
    }

    ELVIS_DEVICE IntervalMatrix<3,3> Invert(const IntervalMatrix<3,3>& rhs)
    {
        IntervalMatrix<3,3> result;

        Interval<ElVisFloat> InvDeterm = MAKE_FLOAT(1.0)*Determinant(rhs);

        result[0] = InvDeterm* (rhs[4]*rhs[8] - rhs[5]*rhs[7]);
        result[1] = InvDeterm* (rhs[2]*rhs[7] - rhs[1]*rhs[8]);
        result[2] = InvDeterm* (rhs[1]*rhs[5] - rhs[2]*rhs[4]);

        result[3] = InvDeterm* (rhs[5]*rhs[6] - rhs[3]*rhs[8]);
        result[4] = InvDeterm* (rhs[0]*rhs[8] - rhs[2]*rhs[6]);
        result[5] = InvDeterm* (rhs[2]*rhs[3] - rhs[0]*rhs[5]);

        result[6] = InvDeterm* (rhs[3]*rhs[7] - rhs[4]*rhs[6]);
        result[7] = InvDeterm* (rhs[1]*rhs[6] - rhs[0]*rhs[7]);
        result[8] = InvDeterm* (rhs[0]*rhs[4] - rhs[1]*rhs[3]);
        return result;
    }



    template<unsigned int Rows, unsigned int Columns>
    ELVIS_DEVICE IntervalMatrix<Rows, Columns> operator+(const IntervalMatrix<Rows, Columns>& lhs, const IntervalMatrix<Rows, Columns>& rhs)
    {
        IntervalMatrix<Rows, Columns> result;
        for(unsigned int i = 0; i < Rows*Columns; ++i)
        {
            result[i] = lhs[i] + rhs[i];
        }
        return result;
    }

    template<unsigned int Rows, unsigned int Columns>
    ELVIS_DEVICE IntervalMatrix<Rows, Columns> operator-(const IntervalMatrix<Rows, Columns>& lhs, const IntervalMatrix<Rows, Columns>& rhs)
    {
        IntervalMatrix<Rows, Columns> result;
        for(unsigned int i = 0; i < Rows*Columns; ++i)
        {
            result[i] = lhs[i] - rhs[i];
        }
        return result;
    }

    template<typename VectorType>
    ELVIS_DEVICE VectorType operator*(const IntervalMatrix<3,3>& m, const VectorType& p)
    {
        VectorType result;
        result.x = m[0]*p[0] + m[1]*p[1] + m[2]*p[2];
        result.y = m[3]*p[0] + m[4]*p[1] + m[5]*p[2];
        result.z = m[6]*p[0] + m[7]*p[1] + m[8]*p[2];
        return result;
    }

    template<typename LhsMatrixType, typename RhsMatrixType>
    ELVIS_DEVICE IntervalMatrix<3,3> MatrixMultiply(const LhsMatrixType& lhs, const RhsMatrixType& rhs)
    {
        IntervalMatrix<3,3> result;
        result[0] = lhs[0]*rhs[0] + lhs[1]*rhs[3] + lhs[2]*rhs[6];
        result[1] = lhs[0]*rhs[1] + lhs[1]*rhs[4] + lhs[2]*rhs[7];
        result[2] = lhs[0]*rhs[2] + lhs[1]*rhs[5] + lhs[2]*rhs[8];

        result[3] = lhs[3]*rhs[0] + lhs[4]*rhs[3] + lhs[5]*rhs[6];
        result[4] = lhs[3]*rhs[1] + lhs[4]*rhs[4] + lhs[5]*rhs[7];
        result[5] = lhs[3]*rhs[2] + lhs[4]*rhs[5] + lhs[5]*rhs[8];

        result[6] = lhs[6]*rhs[0] + lhs[7]*rhs[3] + lhs[8]*rhs[6];
        result[7] = lhs[6]*rhs[1] + lhs[7]*rhs[4] + lhs[8]*rhs[7];
        result[8] = lhs[6]*rhs[2] + lhs[7]*rhs[5] + lhs[8]*rhs[8];
        return result;
    }

    ELVIS_DEVICE IntervalMatrix<3,3> operator*(const IntervalMatrix<3,3>& lhs, const IntervalMatrix<3,3>&  rhs)
    {
        return MatrixMultiply(lhs, rhs);
    }

    ELVIS_DEVICE IntervalMatrix<3,3> operator*(const Matrix<3,3>& lhs, const IntervalMatrix<3,3>&  rhs)
    {
        return MatrixMultiply(lhs, rhs);
    }

    ELVIS_DEVICE IntervalMatrix<3,3> operator*(const IntervalMatrix<3,3>& lhs, const Matrix<3,3>&  rhs)
    {
        return MatrixMultiply(lhs, rhs);
    }

    ELVIS_DEVICE IntervalPoint operator*(const IntervalMatrix<3,3>& lhs, const IntervalPoint&  rhs)
    {
        IntervalPoint result;
        result.x = lhs[0]*rhs.x + lhs[1]*rhs.y + lhs[2]*rhs.z;
        result.y = lhs[3]*rhs.x + lhs[4]*rhs.y + lhs[5]*rhs.z;
        result.z = lhs[6]*rhs.x + lhs[7]*rhs.y + lhs[8]*rhs.z;

        return result;
    }

}

#endif

