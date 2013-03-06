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

#ifndef ELVIS_CORE_ISOSURFACE_MODULE_CUDA_CU
#define ELVIS_CORE_ISOSURFACE_MODULE_CUDA_CU

#include <ElVis/Core/Jacobi.hpp>
#include <ElVis/Core/matrix.cu>
#include <ElVis/Core/Cuda.h>
#include <ElVis/Core/Interval.hpp>

class OrthogonalLegendreBasis
{
    public:
        __device__ static ElVisFloat Eval(unsigned int i, const ElVisFloat& x)
        {
            return Sqrtf((MAKE_FLOAT(2.0)*i+MAKE_FLOAT(1.0))/MAKE_FLOAT(2.0)) * ElVis::OrthoPoly::P(i, 0, 0, x);
        }
};

template<typename FuncType>
__device__
void GenerateLeastSquaresPolynomialProjection(unsigned int order, const ElVisFloat* __restrict__ allNodes, const ElVisFloat* __restrict__ allWeights, const FuncType& f, ElVisFloat* workspace, ElVisFloat* coeffs)
{
    // Nodes and weights start with two point rules
    unsigned int index = (order-1)*(order);
    index = index >> 1;
    index += order-1;

//    ELVIS_PRINTF("Index %d\n", index);
    const ElVisFloat* nodes = &allNodes[index];
    const ElVisFloat* weights = &allWeights[index];


    for(unsigned int j = 0; j <= order; ++j)
    {
        workspace[j] = f(nodes[j]);
    }

    for(unsigned int c_index = 0; c_index <= order; ++c_index)
    {
        coeffs[c_index] = MAKE_FLOAT(0.0);
        for(unsigned int k = 0; k <= order; ++k)
        {
//            ELVIS_PRINTF("K %d, node %2.15f, weight %2.15f, sample %2.15f, basis %2.15f\n",
//                     k, nodes[k], weights[k], workspace[k], OrthogonalLegendreBasis::Eval(c_index, nodes[k]));
            coeffs[c_index] += workspace[k] * OrthogonalLegendreBasis::Eval(c_index, nodes[k]) *
                weights[k];
        }
    }
}

template<typename FuncType>
__device__
void GenerateLeastSquaresPolynomialProjectionParallel(unsigned int order, const ElVisFloat* __restrict__ allNodes, const ElVisFloat* __restrict__ allWeights, const FuncType& f, ElVisFloat* workspace, ElVisFloat* coeffs)
{
}

__device__ ElVisFloat& AccessArray(ElVisFloat* a, int i, int j, int n)
{
    return a[i*n + j];
}

template<typename T1, typename T2>
__device__ T1 SIGN(const T1 &a, const T2 &b)
    {return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);}

__device__ void balance(SquareMatrix& a)
{
    int n = a.GetSize();
    const ElVisFloat RADIX = 2;
    bool done=false;
    ElVisFloat sqrdx=RADIX*RADIX;
    while (!done)
    {
        done=true;
        for (int i=0;i<n;i++)
        {
            ElVisFloat r=0.0,c=0.0;
            for (int j=0;j<n;j++)
            {
                if (j != i)
                {
                    c += abs(a(j,i));
                    r += abs(a(i,j));
                }
            }
            if (c != 0.0 && r != 0.0)
            {
                ElVisFloat g=r/RADIX;
                ElVisFloat f=1.0;
                ElVisFloat s=c+r;
                while (c<g)
                {
                    f *= RADIX;
                    c *= sqrdx;
                }
                g=r*RADIX;
                while (c>g)
                {
                    f /= RADIX;
                    c /= sqrdx;
                }
                if ((c+r)/f < 0.95*s)
                {
                    done=false;
                    g=1.0/f;
                    //scale[i] *= f;
                    for (int j=0;j<n;j++) a(i,j) *= g;
                    for (int j=0;j<n;j++) a(j,i) *= f;
                }
            }
        }
    }
}

// returns roots in wri.  Since we don't care about complex roots, they are just set to -1.0
__device__ void hqr(SquareMatrix& a, int n, ElVisFloat* wri)
{
    int nn,m,l,k,j,its,i,mmin;
    ElVisFloat z,y,x,w,v,u,t,s,r,q,p,anorm=MAKE_FLOAT(0.0);

    const ElVisFloat EPS=MAKE_FLOAT(1e-8);

    for (i=0;i<n;i++)
    {
        for (j=max(i-1,0);j<n;j++)
        {
            anorm += abs(a(i, j));
        }
    }

    nn=n-1;
    t=0.0;
    while (nn >= 0)
    {
        its=0;
        do
        {
            for (l=nn;l>0;l--)
            {
                s=abs(a(l-1, l-1))+abs(a(l, l));
                if (s == 0.0) s=anorm;
                if (abs(a(l,l-1)) <= EPS*s)
                {
                    a(l,l-1) = 0.0;
                    break;
                }
            }
            x=a(nn,nn);
            if (l == nn)
            {
                wri[nn--]=x+t;
            } else
            {
                y=a(nn-1,nn-1);
                w=a(nn,nn-1)*a(nn-1,nn);
                if (l == nn-1)
                {
                    p=0.5*(y-x);
                    q=p*p+w;
                    z=sqrt(abs(q));
                    x += t;
                    if (q >= 0.0)
                    {
                        z=p+SIGN(z,p);
                        wri[nn-1]=wri[nn]=x+z;
                        if (z != 0.0) wri[nn]=x-w/z;
                    } else
                    {
                        //wri[nn]=Complex(x+p,-z);
                        //wri[nn-1]=conj(wri[nn]);
                        wri[nn] = MAKE_FLOAT(-10.0);
                        wri[nn-1] = MAKE_FLOAT(-10.0);
                    }
                    nn -= 2;
                }
                else
                {
                    if (its == 30) return;
                    if (its == 10 || its == 20)
                    {
                        t += x;
                        for (i=0;i<nn+1;i++) a(i,i) -= x;
                        s=abs(a(nn,nn-1))+abs(a(nn-1,nn-2));
                        y=x=0.75*s;
                        w = -0.4375*s*s;
                    }
                    ++its;
                    for (m=nn-2;m>=l;m--)
                    {
                        z=a(m,m);
                        r=x-z;
                        s=y-z;
                        p=(r*s-w)/a(m+1,m)+a(m,m+1);
                        q=a(m+1,m+1)-z-r-s;
                        r=a(m+2,m+1);
                        s=abs(p)+abs(q)+abs(r);
                        p /= s;
                        q /= s;
                        r /= s;
                        if (m == l) break;
                        u=abs(a(m,m-1))*(abs(q)+abs(r));
                        v=abs(p)*(abs(a(m-1,m-1))+abs(z)+abs(a(m+1,m+1)));
                        if (u <= EPS*v) break;
                    }
                    for (i=m;i<nn-1;i++)
                    {
                        a(i+2,i)=0.0;
                        if (i != m) a(i+2,i-1)=0.0;
                    }
                    for (k=m;k<nn;k++)
                    {
                        if (k != m)
                        {
                            p=a(k,k-1);
                            q=a(k+1,k-1);
                            r=0.0;
                            if (k+1 != nn) r=a(k+2,k-1);
                            if ((x=abs(p)+abs(q)+abs(r)) != 0.0)
                            {
                                p /= x;
                                q /= x;
                                r /= x;
                            }
                        }
                        if ((s=SIGN(sqrt(p*p+q*q+r*r),p)) != 0.0)
                        {
                            if (k == m)
                            {
                                if (l != m)
                                a(k,k-1) = -a(k,k-1);
                            }
                            else
                            {
                                a(k,k-1) = -s*x;
                            }
                            p += s;
                            x=p/s;
                            y=q/s;
                            z=r/s;
                            q /= p;
                            r /= p;
                            for (j=k;j<nn+1;j++)
                            {
                                p=a(k,j)+q*a(k+1,j);
                                if (k+1 != nn)
                                {
                                    p += r*a(k+2,j);
                                    a(k+2,j) -= p*z;
                                }
                                a(k+1,j) -= p*y;
                                a(k,j) -= p*x;
                            }
                            mmin = nn < k+3 ? nn : k+3;
                            for (i=l;i<mmin+1;i++)
                            {
                                p=x*a(i,k)+y*a(i,k+1);
                                if (k+1 != nn)
                                {
                                    p += z*a(i,k+2);
                                    a(i,k+2) -= p*r;
                                }
                                a(i,k+1) -= p*q;
                                a(i,k) -= p;
                            }
                        }
                    }
                }
            }
        } while (l+1 < nn);
    }
}

struct IsosurfaceFieldEvaluator
{
    public:
        ELVIS_DEVICE IsosurfaceFieldEvaluator() :
            Origin(),
            Direction(),
            A(),
            B(),
            ElementId(0),
            ElementType(0),
            ReferencePointType(ElVis::eReferencePointIsInvalid),
            InitialGuess()
        {
        }

        __device__ ElVisFloat operator()(const ElVisFloat& t) const
        {
            // Incoming t is [-1..1], we need to scale to [A,B]
            ElVisFloat scaledT = (t+MAKE_FLOAT(1.0))/MAKE_FLOAT(2.0) * (B-A) + A;
            ElVisFloat3 p = Origin + scaledT*Direction;
            ElVisFloat s = EvaluateFieldCuda(ElementId, ElementType, FieldId, p, ReferencePointType, InitialGuess);
            ReferencePointType = ElVis::eReferencePointIsInitialGuess;
            return s;
        }

        ElVisFloat3 Origin;
        ElVisFloat3 Direction;
        ElVisFloat A;
        ElVisFloat B;
        unsigned int ElementId;
        unsigned int ElementType;
        int FieldId;
        mutable ElVis::ReferencePointParameterType ReferencePointType;
        mutable ElVisFloat3 InitialGuess;

    private:
        IsosurfaceFieldEvaluator(const IsosurfaceFieldEvaluator& rhs);
        IsosurfaceFieldEvaluator& operator=(const IsosurfaceFieldEvaluator& rhs);
};

__device__ void GenerateRowMajorHessenbergMatrix(const ElVisFloat* monomialCoefficients, int n, SquareMatrix& h)
{

    // First row
    for(int column = 0; column < n-1; ++column)
    {
        h(0, column) = MAKE_FLOAT(0.0);
    }

    for(int row = 1; row < n; ++row)
    {
        for(int column = 0; column < n-1; ++column)
        {
            if( row == column+1 )
            {
                h(row, column) = MAKE_FLOAT(1.0);
            }
            else
            {
                h(row, column) = MAKE_FLOAT(0.0);
            }
        }
    }

    ElVisFloat inverse = MAKE_FLOAT(-1.0)/monomialCoefficients[n];
    for(int row = 0; row < n; ++row)
    {
        h(row, n-1) = monomialCoefficients[row]*inverse;
    }
}

__device__ void ConvertToMonomial(unsigned int order, ElVisFloat* monomialConversionBuffer, const ElVisFloat* legendreCoeffs, ElVisFloat* monomialCoeffs)
{
    int tableIndex = 0;
    for(int i = 2; i <= order; ++i)
    {
        tableIndex += i*i;
    }
    //ELVIS_PRINTF("Table Index %d\n", tableIndex);
    SquareMatrix m(&monomialConversionBuffer[tableIndex], order+1);

    // Now that we have the coefficient table we can convert.
    for(unsigned int coeffIndex = 0; coeffIndex <= order; ++coeffIndex)
    {
        monomialCoeffs[coeffIndex] = MAKE_FLOAT(0.0);
        for(unsigned int legCoeffIndex = 0; legCoeffIndex <= order; ++legCoeffIndex)
        {
            //ElVisFloat multiplier = AccessArray(buffer,legCoeffIndex,coeffIndex,order+1);
            //ELVIS_PRINTF("Legendre Coeff %2.15f, multiplier %2.15f\n", legendreCoeffs[legCoeffIndex], multiplier);
            monomialCoeffs[coeffIndex] += legendreCoeffs[legCoeffIndex]*
                m(legCoeffIndex,coeffIndex);
        }
    }
}

__device__ void PrintMatrix(SquareMatrix& m)
{
    for(unsigned int row = 0; row < m.GetSize(); ++row)
    {
        for(unsigned int column = 0; column < m.GetSize(); ++column)
        {
            ELVIS_PRINTF("%2.15f, ", m(row, column));
        }
        ELVIS_PRINTF("\n");
    }
}

extern "C" __global__ void CopyToElementId(const int* __restrict__ elementIdBuffer, const int* __restrict__ elementTypeBuffer,
                                           ElVis::ElementId* out, bool enableTrace, int tracex, int tracey,
                                           int bufferSize)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if( index >= bufferSize ) return;

    out[index].Id = elementIdBuffer[index];
    out[index].Type = elementTypeBuffer[index];
}

extern "C" __global__ void FindIsosurfaceInSegment(ElVisFloat3 origin, const int* __restrict__ segmentElementId, const int* __restrict__ segmentElementType,
                                                   const int* __restrict__ segmentdIdBuffer,
                                                   const ElVisFloat3* __restrict__ segmentDirection,
                                                   const ElVisFloat* __restrict__ segmentStart, const ElVisFloat* __restrict__ segmentEnd,
                                                   int fieldId,
                                                   int numIsosurfaces, const ElVisFloat* __restrict__ isovalues,
                                                   bool enableTrace, int tracex, int tracey,
                                                   int screen_x, int screen_y,
                                                   const ElVisFloat* __restrict__ gaussNodes, const ElVisFloat* __restrict__ gaussWeights,
                                                   ElVisFloat* __restrict__ monomialConversionTable,
                                                   ElVisFloat* __restrict__ SampleBuffer, ElVisFloat3* __restrict__ intersection_buffer)
{
    if( numIsosurfaces == 0 ) return;

    int2 trace = make_int2(tracex, tracey);

    uint2 pixel;
    pixel.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixel.y = blockIdx.y * blockDim.y + threadIdx.y;

    bool traceEnabled = (pixel.x == trace.x && pixel.y == trace.y && enableTrace);
    if( traceEnabled )
    {
        ELVIS_PRINTF("FindIsosurfaceInSegment: Find Isosurface .\n");
    }

    uint2 screen;
    screen.x = screen_x;
    screen.y = screen_y;

//    screen.x = gridDim.x * blockDim.x;
//    screen.y = gridDim.y * blockDim.y;

    if( pixel.x >= screen.x ||
        pixel.y >= screen.y )
    {
        return;
    }

    int pixelIndex = pixel.x + screen.x*pixel.y;
    int segmentIndex = segmentdIdBuffer[pixelIndex];
    if( traceEnabled )
    {
        ELVIS_PRINTF("FindIsosurfaceInSegment: Segment index %d, pixel index %d\n", segmentIndex, pixelIndex);
    }
    if( segmentEnd[segmentIndex] < MAKE_FLOAT(0.0) )
    {
        if( traceEnabled )
        {
            ELVIS_PRINTF("FindIsosurfaceInSegment: Exiting because ray has left volume based on segment end\n", segmentIndex);
        }
        return;
    }

    int elementId = segmentElementId[segmentIndex];
    if( traceEnabled )
    {
        ELVIS_PRINTF("FindIsosurfaceInSegment: Element id %d\n", elementId);
    }
    if( elementId == -1 )
    {
        if( traceEnabled )
        {
            ELVIS_PRINTF("FindIsosurfaceInSegment: Exiting because element id is 0\n", segmentIndex);
        }
        return;
    }

    int elementTypeId = segmentElementType[segmentIndex];

    ElVisFloat a = segmentStart[segmentIndex];
    ElVisFloat b = segmentEnd[segmentIndex];

    ElVisFloat3 rayDirection = segmentDirection[segmentIndex];
    ElVisFloat d = (b-a);

    if( traceEnabled )
    {
        ELVIS_PRINTF("FindIsosurfaceInSegment: Ray Direction (%2.10f, %2.10f, %2.10f), segment distance %2.10f and endopints [%2.10f, %2.10f]\n", rayDirection.x, rayDirection.y, rayDirection.z, d, a, b);
    }
    if( d == MAKE_FLOAT(0.0) )
    {
        if( traceEnabled )
        {
            ELVIS_PRINTF("FindIsosurfaceInSegment: Exiting because d is 0\n", rayDirection.x, rayDirection.y, rayDirection.z, d);
        }
        return;
    }

    ElVisFloat bestDepth = depth_buffer[segmentIndex];
    if( traceEnabled )
    {
        ELVIS_PRINTF("FindIsosurfaceInSegment: Best Depth %2.10f and a %2.10f\n", bestDepth, a);
    }
//    if( bestDepth <= a )
//    {
//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Exiting because existing depth value %2.10f exists before segment start %2.10f\n", bestDepth, a);
//        }
//        return;
//    }

    ElVisFloat3 p0 = origin + a*rayDirection;
    ElVisFloat3 p1 = origin + b*rayDirection;
    ElVis::Interval<ElVisFloat> range;
    EstimateRangeCuda(elementId, elementTypeId, fieldId, p0, p1, range);

    if( traceEnabled )
    {
        ELVIS_PRINTF("Range of scalar field is (%2.10f, %2.10f)\n", range.GetLow(), range.GetHigh());
        ELVIS_PRINTF("Origin (%f, %f, %f)\n", origin.x, origin.y, origin.z);

        ELVIS_PRINTF("Direction (%f, %f, %f)\n", rayDirection.x, rayDirection.y, rayDirection.z);
        ELVIS_PRINTF("Integration domain [%f, %f]\n", a, b);
    }

    for(int isosurfaceId = 0; isosurfaceId < numIsosurfaces; ++isosurfaceId )
    {
        if( !range.IsEmpty() && !range.Contains(isovalues[isosurfaceId]) )
        {
            continue;
        }


        if( traceEnabled )
        {
            ELVIS_PRINTF("Searching for isovalue %f\n", isovalues[isosurfaceId]);
        }

        // Project onto a polynomial along the ray.
        // Generate an nth order polynomial projection.
        // First pass, create an mth element local array to store the value, and exit out if the required order is
        // too large.
        ElVisFloat polynomialCoefficients[32];
        ElVisFloat monomialCoefficients[32];
        ElVisFloat workspace[32];
        ElVisFloat h_data[10*10];

        int requiredOrder = 8;
        for(int i = 0; i < 32; ++i)
        {
            polynomialCoefficients[i] = -73.45;
            workspace[i] = -73.45;
            monomialCoefficients[i] = -73.45;
        }

        IsosurfaceFieldEvaluator f;
        f.Origin = origin;
        f.Direction = rayDirection;
        f.ElementId = elementId;
        f.ElementType = elementTypeId;
        f.A = a;
        f.B = b;
        f.FieldId = fieldId;


        GenerateLeastSquaresPolynomialProjection(requiredOrder, gaussNodes, gaussWeights, f, workspace, polynomialCoefficients);
        if( traceEnabled )
        {
            ELVIS_PRINTF("Legendre %2.15f, %2.15f, %2.15f, %2.15f, %2.15f, %2.15f, %2.15f\n",
                 polynomialCoefficients[0],
                 polynomialCoefficients[1],
                 polynomialCoefficients[2],
                 polynomialCoefficients[3],
                 polynomialCoefficients[4],
                 polynomialCoefficients[5],
                 polynomialCoefficients[6],
                 polynomialCoefficients[7],
                 polynomialCoefficients[8]);
        }

        // Fix up the polynomial order if we requested higher than necessary.
        int reducedOrder = requiredOrder;
        ElVisFloat epsilon = MAKE_FLOAT(1e-8);

        for(int i = requiredOrder; i >= 1; --i)
        {
            if( Fabsf(polynomialCoefficients[i]) > epsilon )
            {
                reducedOrder = i;
                break;
            }
        }


        if( traceEnabled )
        {
            ELVIS_PRINTF("Reduced order %d\n", reducedOrder );
        }

        ConvertToMonomial(reducedOrder, monomialConversionTable, polynomialCoefficients, monomialCoefficients);
        if( traceEnabled )
        {
            ELVIS_PRINTF("Monomial %2.15f, %2.15f, %2.15f, %2.15f, %2.15f, %2.15f, %2.15f\n",
                 monomialCoefficients[0],
                 monomialCoefficients[1],
                 monomialCoefficients[2],
                 monomialCoefficients[3],
                 monomialCoefficients[4],
                 monomialCoefficients[5],
                 monomialCoefficients[6],
                 monomialCoefficients[7],
                 monomialCoefficients[8]);
        }

        monomialCoefficients[0] -= isovalues[isosurfaceId];

        SquareMatrix h(h_data, reducedOrder);
        GenerateRowMajorHessenbergMatrix(monomialCoefficients, reducedOrder, h);
    //    if( traceEnabled )
    //    {
    //        ELVIS_PRINTF("Before balancing.\n");
    //        PrintMatrix(h);
    //    }

        balance(h);

    //    if( traceEnabled )
    //    {
    //        ELVIS_PRINTF("After balancing.\n");
    //        PrintMatrix(h);
    //    }
        ElVisFloat roots[8];
        for(int i = 0; i < 8; ++i)
        {
            roots[i] = -4582.23;
        }

        hqr(h, reducedOrder, roots);

    //    if( traceEnabled )
    //    {
    //        ELVIS_PRINTF("Roots %2.15f, %2.15f, %2.15f, %2.15f, %2.15f, %2.15f\n",
    //             roots[0],
    //             roots[1],
    //             roots[2],
    //             roots[3],
    //             roots[4],
    //             roots[5]);
    //    }

        ElVisFloat foundRoot = ELVIS_FLOAT_MAX;
        for(int i = 0; i < reducedOrder; ++i)
        {
            ElVisFloat root = roots[i];
            if( root >= MAKE_FLOAT(-1.0) &&
                root <= MAKE_FLOAT(1.0) &&
                root <= foundRoot )
            {
                ElVisFloat foundT = (root + MAKE_FLOAT(1.0))/MAKE_FLOAT(2.0) * (f.B-f.A) + f.A;
                if( foundT < bestDepth )
                {
                    foundRoot = root;
                }
            }
        }

        if( foundRoot != ELVIS_FLOAT_MAX )
        {

            ElVisFloat foundT = (foundRoot + MAKE_FLOAT(1.0))/MAKE_FLOAT(2.0) * (f.B-f.A) + f.A;



            ElVisFloat3 foundIntersectionPoint = origin + foundT*rayDirection;

            intersection_buffer[segmentIndex] = foundIntersectionPoint;
            SampleBuffer[segmentIndex] = EvaluateFieldCuda(elementId,
                                                           elementTypeId,
                                                           fieldId,
                                                           foundIntersectionPoint);
            if( traceEnabled )
            {
                ELVIS_PRINTF("FindIsosurfaceInSegment: ######################## Found root %2.15f, in world %2.15f with value %f \n", foundRoot, foundT, SampleBuffer[segmentIndex]);
            }

            EvaluateNormalCuda(elementId,
                             elementTypeId,
                             fieldId,
                             foundIntersectionPoint, normal_buffer[segmentIndex]);
            depth_buffer[segmentIndex] = foundT;
            bestDepth = foundT;
            //        // This depth buffer is wrong, need accumulated.
            //        depth_buffer[launch_index] = (far+near)/(far-near) - 2.0f/foundT * far*near/(far-near);
            //        depth_buffer[launch_index] = (depth_buffer[launch_index]+1.0)/2.0;

        }
    }

}

#endif //ELVIS_CORE_ISOSURFACE_MODULE_CUDA_CU
