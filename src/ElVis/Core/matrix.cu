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

#ifndef ELVIS_MATRIX_CU
#define ELVIS_MATRIX_CU

#include <optixu/optixu_math.h>
#include <optixu/optixu_math_namespace.h>
#include <optix_math.h>
#include <assert.h>
#include <ElVis/Core/Float.cu>

#define MATRIX_ACCESS(m,i,j) m[i*N+j]
#define MAT_DECL template <unsigned int M, unsigned int N>

// This matrix class is a direct copy of the OptiX matrix class, modified to 
// use ElVisFloat instead of float.
namespace ElVis 
{
  template <int DIM> struct VectorDim { };
  template <> struct VectorDim<2> { typedef ElVisFloat2 VectorType; };
  template <> struct VectorDim<3> { typedef ElVisFloat3 VectorType; };
  template <> struct VectorDim<4> { typedef ElVisFloat4 VectorType; };


  template <unsigned int M, unsigned int N> class Matrix;

  template <unsigned int M> Matrix<M,M>& operator*=(Matrix<M,M>& m1, const Matrix<M,M>& m2);
  MAT_DECL Matrix<M,N>& operator-=(Matrix<M,N>& m1, const Matrix<M,N>& m2);
  MAT_DECL Matrix<M,N>& operator+=(Matrix<M,N>& m1, const Matrix<M,N>& m2);
  MAT_DECL Matrix<M,N>& operator*=(Matrix<M,N>& m1, ElVisFloat f);
  MAT_DECL Matrix<M,N>& operator/=(Matrix<M,N>& m1, ElVisFloat f);
  MAT_DECL Matrix<M,N> operator-(const Matrix<M,N>& m1, const Matrix<M,N>& m2);
  MAT_DECL Matrix<M,N> operator+(const Matrix<M,N>& m1, const Matrix<M,N>& m2);
  MAT_DECL Matrix<M,N> operator/(const Matrix<M,N>& m, ElVisFloat f);
  MAT_DECL Matrix<M,N> operator*(const Matrix<M,N>& m, ElVisFloat f);
  MAT_DECL Matrix<M,N> operator*(ElVisFloat f, const Matrix<M,N>& m);
  MAT_DECL typename Matrix<M,N>::floatM operator*(const Matrix<M,N>& m, const typename Matrix<M,N>::floatN& v );
  MAT_DECL typename Matrix<M,N>::floatN operator*(const typename Matrix<M,N>::floatM& v, const Matrix<M,N>& m);
  template<unsigned int M, unsigned int N, unsigned int R> Matrix<M,R> operator*(const Matrix<M,N>& m1, const Matrix<N,R>& m2);


  // Partial specializations to make matrix vector multiplication more efficient
  template <unsigned int N>
  RT_HOSTDEVICE ElVisFloat2 operator*(const Matrix<2,N>& m, const typename Matrix<2,N>::floatN& vec );
  template <unsigned int N>
  RT_HOSTDEVICE ElVisFloat3 operator*(const Matrix<3,N>& m, const typename Matrix<3,N>::floatN& vec );
  template <unsigned int N>
  RT_HOSTDEVICE ElVisFloat4 operator*(const Matrix<4,N>& m, const typename Matrix<4,N>::floatN& vec );
  RT_HOSTDEVICE ElVisFloat4 operator*(const Matrix<4,4>& m, const ElVisFloat4& vec );

  // A matrix with M rows and N columns
  template <unsigned int M, unsigned int N>
  class Matrix
  {
  public:
    typedef typename VectorDim<N>::VectorType  floatN; // A row of the matrix
    typedef typename VectorDim<M>::VectorType  floatM; // A column of the matrix

    // Create an unitialized matrix.
    RT_HOSTDEVICE              Matrix();

    // Create a matrix from the specified float array.
    RT_HOSTDEVICE explicit     Matrix( const ElVisFloat data[M*N] ) { for(unsigned int i = 0; i < M*N; ++i) _data[i] = data[i]; }

    // Copy the matrix.
    RT_HOSTDEVICE              Matrix( const Matrix& m );

    // Assignment operator.
    RT_HOSTDEVICE Matrix&      operator=( const Matrix& b );

    // Access the specified element 0..N*M-1
    RT_HOSTDEVICE const ElVisFloat&        operator[]( unsigned int i )const { return _data[i]; }

    // Access the specified element 0..N*M-1
    RT_HOSTDEVICE ElVisFloat&       operator[]( unsigned int i )      { return _data[i]; }

    // Access the specified row 0..M.  Returns float, float2, float3 or float4 depending on the matrix size.
    RT_HOSTDEVICE floatN       getRow( unsigned int m )const;

    // Access the specified column 0..N.  Returns float, float2, float3 or float4 depending on the matrix size.
    RT_HOSTDEVICE floatM       getCol( unsigned int n )const;

    // Returns a pointer to the internal data array.  The data array is stored in row-major order.
    RT_HOSTDEVICE ElVisFloat*       getData();

    // Returns a const pointer to the internal data array.  The data array is stored in row-major order.
    RT_HOSTDEVICE const ElVisFloat* getData()const;

    // Assign the specified row 0..M.  Takes a float, float2, float3 or float4 depending on the matrix size.
    RT_HOSTDEVICE void         setRow( unsigned int m, const floatN &r );

    // Assign the specified column 0..N.  Takes a float, float2, float3 or float4 depending on the matrix size.
    RT_HOSTDEVICE void         setCol( unsigned int n, const floatM &c );

    // Returns the transpose of the matrix.
    RT_HOSTDEVICE Matrix<N,M>         transpose() const;

    // Returns the inverse of the matrix.
    RT_HOSTDEVICE Matrix<4,4>         inverse() const;

    // Returns the determinant of the matrix.
    RT_HOSTDEVICE ElVisFloat               det() const;

    // Returns a rotation matrix.
    RT_HOSTDEVICE static Matrix<4,4>  rotate(const ElVisFloat radians, const ElVisFloat3& axis);

    // Returns a translation matrix.
    RT_HOSTDEVICE static Matrix<4,4>  translate(const ElVisFloat3& vec);

    // Returns a scale matrix.
    RT_HOSTDEVICE static Matrix<4,4>  scale(const ElVisFloat3& vec);

    // Returns the identity matrix.
    RT_HOSTDEVICE static Matrix<N,N>  identity();

    // Ordered comparison operator so that the matrix can be used in an STL container.
    RT_HOSTDEVICE bool         operator<( const Matrix<M, N>& rhs ) const;
  private:
    ElVisFloat _data[M*N]; // The data array is stored in row-major order.
  };



  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N>::Matrix()
  {
  }

  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N>::Matrix( const Matrix<M,N>& m )
  {
    for(unsigned int i = 0; i < M*N; ++i)
      _data[i] = m._data[i];
  }

  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N>&  Matrix<M,N>::operator=( const Matrix& b )
  {
    for(unsigned int i = 0; i < M*N; ++i)
      _data[i] = b._data[i];
    return *this;
  }


  /*
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE float Matrix<M,N>::operator[]( unsigned int i )const
  {
  assert( i < M*N );
  return _data[i];
  }


  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE float& Matrix<M,N>::operator[]( unsigned int i )
  {
  return _data[i];
  }
  */

  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE typename Matrix<M,N>::floatN Matrix<M,N>::getRow( unsigned int m )const
  {
    typename Matrix<M,N>::floatN temp;
    ElVisFloat* v = reinterpret_cast<ElVisFloat*>( &temp );
    const ElVisFloat* row = &( _data[m*N] );
    for(unsigned int i = 0; i < N; ++i)
      v[i] = row[i];

    return temp;
  }


  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE typename Matrix<M,N>::floatM Matrix<M,N>::getCol( unsigned int n )const
  {
    typename Matrix<M,N>::floatM temp;
    ElVisFloat* v = reinterpret_cast<ElVisFloat*>( &temp );
    for ( unsigned int i = 0; i < M; ++i )
      v[i] = MATRIX_ACCESS( _data, i, n );

    return temp;
  }


  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE ElVisFloat* Matrix<M,N>::getData()
  {
    return _data;
  }


  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE const ElVisFloat* Matrix<M,N>::getData() const
  {
    return _data;
  }


  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE void Matrix<M,N>::setRow( unsigned int m, const typename Matrix<M,N>::floatN &r )
  {
    const ElVisFloat* v = reinterpret_cast<const ElVisFloat*>( &r );
    ElVisFloat* row = &( _data[m*N] );
    for(unsigned int i = 0; i < N; ++i)
      row[i] = v[i];
  }


  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE void Matrix<M,N>::setCol( unsigned int n, const typename Matrix<M,N>::floatM &c )
  {
    const ElVisFloat* v = reinterpret_cast<const ElVisFloat*>( &c );
    for ( unsigned int i = 0; i < M; ++i )
      MATRIX_ACCESS( _data, i, n ) = v[i];
  }


  // Subtract two matrices of the same size.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N> operator-(const Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    Matrix<M,N> temp( m1 );
    temp -= m2;
    return temp;
  }


  // Subtract two matrices of the same size.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N>& operator-=(Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    for ( unsigned int i = 0; i < M*N; ++i )
      m1[i] -= m2[i];
    return m1;
  }


  // Add two matrices of the same size.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N> operator+(const Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    Matrix<M,N> temp( m1 );
    temp += m2;
    return temp;
  }


  // Add two matrices of the same size.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N>& operator+=(Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    for ( unsigned int i = 0; i < M*N; ++i )
      m1[i] += m2[i];
    return m1;
  }


  // Multiply two compatible matrices.
  template<unsigned int M, unsigned int N, unsigned int R>
  RT_HOSTDEVICE Matrix<M,R> operator*( const Matrix<M,N>& m1, const Matrix<N,R>& m2)
  {
    Matrix<M,R> temp;

    for ( unsigned int i = 0; i < M; ++i ) {
      for ( unsigned int j = 0; j < R; ++j ) {
        ElVisFloat sum = MAKE_FLOAT(0.0);
        for ( unsigned int k = 0; k < N; ++k ) {
          ElVisFloat ik = m1[ i*N+k ];
          ElVisFloat kj = m2[ k*R+j ];
          sum += ik * kj;
        }
        temp[i*R+j] = sum;
      }
    }
    return temp;
  }


  // Multiply two compatible matrices.
  template<unsigned int M>
  RT_HOSTDEVICE Matrix<M,M>& operator*=(Matrix<M,M>& m1, const Matrix<M,M>& m2)
  {
    m1 = m1*m2;
    return m1;
  }


  // Multiply matrix by vector
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE typename Matrix<M,N>::floatM operator*(const Matrix<M,N>& m, const typename Matrix<M,N>::floatN& vec )
  {
    typename Matrix<M,N>::floatM temp;
    ElVisFloat* t = reinterpret_cast<ElVisFloat*>( &temp );
    const ElVisFloat* v = reinterpret_cast<const ElVisFloat*>( &vec );

    for (unsigned int i = 0; i < M; ++i) {
      ElVisFloat sum = MAKE_FLOAT(0.0);
      for (unsigned int j = 0; j < N; ++j) {
        sum += MATRIX_ACCESS( m, i, j ) * v[j];
      }
      t[i] = sum;
    }

    return temp;
  }

  // Multiply matrix2xN by floatN
  template<unsigned int N>
  RT_HOSTDEVICE ElVisFloat2 operator*(const Matrix<2,N>& m, const typename Matrix<2,N>::floatN& vec )
  {
    ElVisFloat2 temp = { MAKE_FLOAT(0.0), MAKE_FLOAT(0.0) };
    const ElVisFloat* v = reinterpret_cast<const ElVisFloat*>( &vec );

    int index = 0;
    for (unsigned int j = 0; j < N; ++j)
      temp.x += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.y += m[index++] * v[j];

    return temp;
  }

  // Multiply matrix3xN by floatN
  template<unsigned int N>
  RT_HOSTDEVICE ElVisFloat3 operator*(const Matrix<3,N>& m, const typename Matrix<3,N>::floatN& vec )
  {
    ElVisFloat3 temp = { MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0) };
    const ElVisFloat* v = reinterpret_cast<const ElVisFloat*>( &vec );

    int index = 0;
    for (unsigned int j = 0; j < N; ++j)
      temp.x += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.y += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.z += m[index++] * v[j];

    return temp;
  }

  // Multiply matrix4xN by floatN
  template<unsigned int N>
  RT_HOSTDEVICE ElVisFloat4 operator*(const Matrix<4,N>& m, const typename Matrix<4,N>::floatN& vec )
  {
    ElVisFloat4 temp = { MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0) };

    const ElVisFloat* v = reinterpret_cast<const ElVisFloat*>( &vec );

    int index = 0;
    for (unsigned int j = 0; j < N; ++j)
      temp.x += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.y += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.z += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.w += m[index++] * v[j];

    return temp;
  }

  // Multiply matrix4x4 by float4
  RT_HOSTDEVICE inline ElVisFloat4 operator*(const Matrix<4,4>& m, const ElVisFloat4& vec )
  {
    ElVisFloat4 temp;
    temp.x  = m[ 0] * vec.x +
              m[ 1] * vec.y +
              m[ 2] * vec.z +
              m[ 3] * vec.w;
    temp.y  = m[ 4] * vec.x +
              m[ 5] * vec.y +
              m[ 6] * vec.z +
              m[ 7] * vec.w;
    temp.z  = m[ 8] * vec.x +
              m[ 9] * vec.y +
              m[10] * vec.z +
              m[11] * vec.w;
    temp.w  = m[12] * vec.x +
              m[13] * vec.y +
              m[14] * vec.z +
              m[15] * vec.w;

    return temp;
  }

  // Multiply vector by matrix
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE typename Matrix<M,N>::floatN operator*(const typename Matrix<M,N>::floatM& vec, const Matrix<M,N>& m)
  {
    typename Matrix<M,N>::floatN  temp;
    ElVisFloat* t = reinterpret_cast<ElVisFloat*>( &temp );
    const ElVisFloat* v = reinterpret_cast<const ElVisFloat*>( &vec);

    for (unsigned int i = 0; i < N; ++i) {
      ElVisFloat sum = MAKE_FLOAT(0.0);
      for (unsigned int j = 0; j < M; ++j) {
        sum += v[j] * MATRIX_ACCESS( m, j, i ) ;
      }
      t[i] = sum;
    }

    return temp;
  }


  // Multply matrix by a scalar.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N> operator*(const Matrix<M,N>& m, ElVisFloat f)
  {
    Matrix<M,N> temp( m );
    temp *= f;
    return temp;
  }


  // Multply matrix by a scalar.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N>& operator*=(Matrix<M,N>& m, ElVisFloat f)
  {
    for ( unsigned int i = 0; i < M*N; ++i )
      m[i] *= f;
    return m;
  }


  // Multply matrix by a scalar.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N>  operator*(ElVisFloat f, const Matrix<M,N>& m)
  {
    Matrix<M,N> temp;

    for ( unsigned int i = 0; i < M*N; ++i )
      temp[i] = m[i]*f;

    return temp;
  }


  // Divide matrix by a scalar.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N> operator/(const Matrix<M,N>& m, ElVisFloat f)
  {
    Matrix<M,N> temp( m );
    temp /= f;
    return temp;
  }


  // Divide matrix by a scalar.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N>& operator/=(Matrix<M,N>& m, ElVisFloat f)
  {
    ElVisFloat inv_f = MAKE_FLOAT(1.0) / f;
    for ( unsigned int i = 0; i < M*N; ++i )
      m[i] *= inv_f;
    return m;
  }

  // Returns the transpose of the matrix.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE inline Matrix<N,M> Matrix<M,N>::transpose() const
  {
    Matrix<N,M> ret;
    for( unsigned int row = 0; row < M; ++row )
      for( unsigned int col = 0; col < N; ++col )
        ret._data[col*M+row] = _data[row*N+col];
    return ret;
  }

  // Returns the determinant of the matrix.
  template<>
  RT_HOSTDEVICE inline ElVisFloat Matrix<3,3>::det() const
  {
    const ElVisFloat* m   = _data;
    ElVisFloat d = m[0]*m[4]*m[8] + m[1]*m[5]*m[6] + m[2]*m[3]*m[7]
      - m[0]*m[5]*m[7] - m[1]*m[3]*m[8] - m[2]*m[4]*m[6];
    return d;
  }

  // Returns the determinant of the matrix.
  template<>
  RT_HOSTDEVICE inline ElVisFloat Matrix<4,4>::det() const
  {
    const ElVisFloat* m   = _data;
    ElVisFloat d =
      m[0]*m[5]*m[10]*m[15]-
      m[0]*m[5]*m[11]*m[14]+m[0]*m[9]*m[14]*m[7]-
      m[0]*m[9]*m[6]*m[15]+m[0]*m[13]*m[6]*m[11]-
      m[0]*m[13]*m[10]*m[7]-m[4]*m[1]*m[10]*m[15]+m[4]*m[1]*m[11]*m[14]-
      m[4]*m[9]*m[14]*m[3]+m[4]*m[9]*m[2]*m[15]-
      m[4]*m[13]*m[2]*m[11]+m[4]*m[13]*m[10]*m[3]+m[8]*m[1]*m[6]*m[15]-
      m[8]*m[1]*m[14]*m[7]+m[8]*m[5]*m[14]*m[3]-
      m[8]*m[5]*m[2]*m[15]+m[8]*m[13]*m[2]*m[7]-
      m[8]*m[13]*m[6]*m[3]-
      m[12]*m[1]*m[6]*m[11]+m[12]*m[1]*m[10]*m[7]-
      m[12]*m[5]*m[10]*m[3]+m[12]*m[5]*m[2]*m[11]-
      m[12]*m[9]*m[2]*m[7]+m[12]*m[9]*m[6]*m[3];
    return d;
  }

  // Returns the inverse of the matrix.
  template<>
  RT_HOSTDEVICE inline Matrix<4,4> Matrix<4,4>::inverse() const
  {
    Matrix<4,4> dst;
    const ElVisFloat* m   = _data;
    const ElVisFloat d = MAKE_FLOAT(1.0) / det();

    dst[0]  = d * (m[5] * (m[10] * m[15] - m[14] * m[11]) + m[9] * (m[14] * m[7] - m[6] * m[15]) + m[13] * (m[6] * m[11] - m[10] * m[7]));
    dst[4]  = d * (m[6] * (m[8] * m[15] - m[12] * m[11]) + m[10] * (m[12] * m[7] - m[4] * m[15]) + m[14] * (m[4] * m[11] - m[8] * m[7]));
    dst[8]  = d * (m[7] * (m[8] * m[13] - m[12] * m[9]) + m[11] * (m[12] * m[5] - m[4] * m[13]) + m[15] * (m[4] * m[9] - m[8] * m[5]));
    dst[12] = d * (m[4] * (m[13] * m[10] - m[9] * m[14]) + m[8] * (m[5] * m[14] - m[13] * m[6]) + m[12] * (m[9] * m[6] - m[5] * m[10]));
    dst[1]  = d * (m[9] * (m[2] * m[15] - m[14] * m[3]) + m[13] * (m[10] * m[3] - m[2] * m[11]) + m[1] * (m[14] * m[11] - m[10] * m[15]));
    dst[5]  = d * (m[10] * (m[0] * m[15] - m[12] * m[3]) + m[14] * (m[8] * m[3] - m[0] * m[11]) + m[2] * (m[12] * m[11] - m[8] * m[15]));
    dst[9]  = d * (m[11] * (m[0] * m[13] - m[12] * m[1]) + m[15] * (m[8] * m[1] - m[0] * m[9]) + m[3] * (m[12] * m[9] - m[8] * m[13]));
    dst[13] = d * (m[8] * (m[13] * m[2] - m[1] * m[14]) + m[12] * (m[1] * m[10] - m[9] * m[2]) + m[0] * (m[9] * m[14] - m[13] * m[10]));
    dst[2]  = d * (m[13] * (m[2] * m[7] - m[6] * m[3]) + m[1] * (m[6] * m[15] - m[14] * m[7]) + m[5] * (m[14] * m[3] - m[2] * m[15]));
    dst[6]  = d * (m[14] * (m[0] * m[7] - m[4] * m[3]) + m[2] * (m[4] * m[15] - m[12] * m[7]) + m[6] * (m[12] * m[3] - m[0] * m[15]));
    dst[10] = d * (m[15] * (m[0] * m[5] - m[4] * m[1]) + m[3] * (m[4] * m[13] - m[12] * m[5]) + m[7] * (m[12] * m[1] - m[0] * m[13]));
    dst[14] = d * (m[12] * (m[5] * m[2] - m[1] * m[6]) + m[0] * (m[13] * m[6] - m[5] * m[14]) + m[4] * (m[1] * m[14] - m[13] * m[2]));
    dst[3]  = d * (m[1] * (m[10] * m[7] - m[6] * m[11]) + m[5] * (m[2] * m[11] - m[10] * m[3]) + m[9] * (m[6] * m[3] - m[2] * m[7]));
    dst[7]  = d * (m[2] * (m[8] * m[7] - m[4] * m[11]) + m[6] * (m[0] * m[11] - m[8] * m[3]) + m[10] * (m[4] * m[3] - m[0] * m[7]));
    dst[11] = d * (m[3] * (m[8] * m[5] - m[4] * m[9]) + m[7] * (m[0] * m[9] - m[8] * m[1]) + m[11] * (m[4] * m[1] - m[0] * m[5]));
    dst[15] = d * (m[0] * (m[5] * m[10] - m[9] * m[6]) + m[4] * (m[9] * m[2] - m[1] * m[10]) + m[8] * (m[1] * m[6] - m[5] * m[2]));
    return dst;
  }

  RT_HOSTDEVICE inline void Invert(const Matrix<3,3>& matrix, Matrix<3,3>& dst)
  {
    const ElVisFloat* rhs   = matrix.getData();
    const ElVisFloat InvDeterm = MAKE_FLOAT(1.0) / matrix.det();

    dst[0] = InvDeterm* (rhs[4]*rhs[8] - rhs[5]*rhs[7]);
    dst[1] = InvDeterm* (rhs[2]*rhs[7] - rhs[1]*rhs[8]);
    dst[2] = InvDeterm* (rhs[1]*rhs[5] - rhs[2]*rhs[4]);

    dst[3] = InvDeterm* (rhs[5]*rhs[6] - rhs[3]*rhs[8]);
    dst[4] = InvDeterm* (rhs[0]*rhs[8] - rhs[2]*rhs[6]);
    dst[5] = InvDeterm* (rhs[2]*rhs[3] - rhs[0]*rhs[5]);

    dst[6] = InvDeterm* (rhs[3]*rhs[7] - rhs[4]*rhs[6]);
    dst[7] = InvDeterm* (rhs[1]*rhs[6] - rhs[0]*rhs[7]);
    dst[8] = InvDeterm* (rhs[0]*rhs[4] - rhs[1]*rhs[3]);
  }

  RT_HOSTDEVICE inline Matrix<3,3> Invert(const Matrix<3,3>& matrix)
  {
    Matrix<3,3> dst;
    const ElVisFloat* rhs   = matrix.getData();
    const ElVisFloat InvDeterm = MAKE_FLOAT(1.0) / matrix.det();

    dst[0] = InvDeterm* (rhs[4]*rhs[8] - rhs[5]*rhs[7]);
    dst[1] = InvDeterm* (rhs[2]*rhs[7] - rhs[1]*rhs[8]);
    dst[2] = InvDeterm* (rhs[1]*rhs[5] - rhs[2]*rhs[4]);

    dst[3] = InvDeterm* (rhs[5]*rhs[6] - rhs[3]*rhs[8]);
    dst[4] = InvDeterm* (rhs[0]*rhs[8] - rhs[2]*rhs[6]);
    dst[5] = InvDeterm* (rhs[2]*rhs[3] - rhs[0]*rhs[5]);

    dst[6] = InvDeterm* (rhs[3]*rhs[7] - rhs[4]*rhs[6]);
    dst[7] = InvDeterm* (rhs[1]*rhs[6] - rhs[0]*rhs[7]);
    dst[8] = InvDeterm* (rhs[0]*rhs[4] - rhs[1]*rhs[3]);

    return dst;
  }

  // Returns a rotation matrix.
  // This is a static member.
  template<>
  RT_HOSTDEVICE Matrix<4,4> inline Matrix<4,4>::rotate(const ElVisFloat radians, const ElVisFloat3& axis)
  {
    Matrix<4,4> Mat = Matrix<4,4>::identity();
    ElVisFloat *m = Mat.getData();

    // NOTE: Element 0,1 is wrong in Foley and Van Dam, Pg 227!
    // TODO - Double sin/cos?
    ElVisFloat sintheta=sinf(radians);
    ElVisFloat costheta=cosf(radians);
    ElVisFloat ux=axis.x;
    ElVisFloat uy=axis.y;
    ElVisFloat uz=axis.z;
    m[0*4+0]=ux*ux+costheta*(1-ux*ux);
    m[0*4+1]=ux*uy*(1-costheta)-uz*sintheta;
    m[0*4+2]=uz*ux*(1-costheta)+uy*sintheta;
    m[0*4+3]=0;

    m[1*4+0]=ux*uy*(1-costheta)+uz*sintheta;
    m[1*4+1]=uy*uy+costheta*(1-uy*uy);
    m[1*4+2]=uy*uz*(1-costheta)-ux*sintheta;
    m[1*4+3]=0;

    m[2*4+0]=uz*ux*(1-costheta)-uy*sintheta;
    m[2*4+1]=uy*uz*(1-costheta)+ux*sintheta;
    m[2*4+2]=uz*uz+costheta*(1-uz*uz);
    m[2*4+3]=0;

    m[3*4+0]=0;
    m[3*4+1]=0;
    m[3*4+2]=0;
    m[3*4+3]=1;

    return Matrix<4,4>( m );
  }

  // Returns a translation matrix.
  // This is a static member.
  template<>
  RT_HOSTDEVICE Matrix<4,4> inline Matrix<4,4>::translate(const ElVisFloat3& vec)
  {
    Matrix<4,4> Mat = Matrix<4,4>::identity();
    ElVisFloat *m = Mat.getData();

    m[3] = vec.x;
    m[7] = vec.y;
    m[11]= vec.z;

    return Matrix<4,4>( m );
  }

  // Returns a scale matrix.
  // This is a static member.
  template<>
  RT_HOSTDEVICE Matrix<4,4> inline Matrix<4,4>::scale(const ElVisFloat3& vec)
  {
    Matrix<4,4> Mat = Matrix<4,4>::identity();
    ElVisFloat *m = Mat.getData();

    m[0] = vec.x;
    m[5] = vec.y;
    m[10]= vec.z;

    return Matrix<4,4>( m );
  }

  // Returns the identity matrix.
  // This is a static member.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<N,N> Matrix<M,N>::identity()
  {
    ElVisFloat temp[N*N];
    for(unsigned int i = 0; i < N*N; ++i)
      temp[i] = 0;
    for( unsigned int i = 0; i < N; ++i )
      MATRIX_ACCESS( temp,i,i ) = 1.0f;
    return Matrix<N,N>( temp );
  }

  // Ordered comparison operator so that the matrix can be used in an STL container.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE bool Matrix<M,N>::operator<( const Matrix<M, N>& rhs ) const
  {
    for( unsigned int i = 0; i < N*M; ++i ) {
      if( _data[i] < rhs._data[i] )
        return true;
      else if( _data[i] > rhs._data[i] )
        return false;
    }
    return false;
  }

  typedef Matrix<2, 2> Matrix2x2;
  typedef Matrix<2, 3> Matrix2x3;
  typedef Matrix<2, 4> Matrix2x4;
  typedef Matrix<3, 2> Matrix3x2;
  typedef Matrix<3, 3> Matrix3x3;
  typedef Matrix<3, 4> Matrix3x4;
  typedef Matrix<4, 2> Matrix4x2;
  typedef Matrix<4, 3> Matrix4x3;
  typedef Matrix<4, 4> Matrix4x4;

} // end namespace ElVis

#undef MATRIX_ACCESS
#undef MAT_DECL

template <unsigned int N, typename DataType>
__device__ __forceinline__ void SetColumn(optix::Matrix<4, N>& matrix, int column, DataType& data)
{
    matrix[column] = data.x;
    matrix[column+N] = data.y;
    matrix[column+2*N] = data.z;
    matrix[column+3*N] = data.w;
}

template <unsigned int N, typename DataType>
__device__ __forceinline__ void SetColumn(optix::Matrix<3, N>& matrix, int column, DataType& data)
{
    matrix[column] = data.x;
    matrix[column+N] = data.y;
    matrix[column+2*N] = data.z;
}

class SquareMatrix
{
    public:
        __device__ __forceinline__ SquareMatrix(ElVisFloat* data, int n)
        {
            m_data = data;
            m_n = n;
        }

        __device__ __forceinline__ ElVisFloat& operator()(int row, int column)
        {
            return m_data[row*m_n + column];
        }

        __device__ __forceinline__ const ElVisFloat& operator()(int row, int column) const
        {
            return m_data[row*m_n + column];
        }

        __device__ __forceinline__ unsigned int GetSize() const { return m_n; }

    private:
        SquareMatrix(const SquareMatrix& rhs);
        SquareMatrix& operator=(const SquareMatrix& rhs);

        unsigned int m_n;
        ElVisFloat* m_data;

};

#endif
