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

#include <optix_cuda.h>
#include <optix_math.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_aabb.h>
#include <ElVis/Core/DiffuseLighting.cu>
#include <ElVis/Core/OptixVariables.cu>

rtDeclareVariable(ElVisFloat3, TriangleVertex0, , );
rtDeclareVariable(ElVisFloat3, TriangleVertex1, , );
rtDeclareVariable(ElVisFloat3, TriangleVertex2, , );

rtDeclareVariable(float, s0, , );
rtDeclareVariable(float, s1, , );
rtDeclareVariable(float, s2, , );


RT_PROGRAM void triangle_intersect( int primIdx )
{
    ElVisFloat3 e0 = TriangleVertex1 - TriangleVertex0;
    ElVisFloat3 e1 = TriangleVertex0 - TriangleVertex2;
    ElVisFloat3 n  = cross( e0, e1 );

    ElVisFloat v   = dot( n, MakeFloat3(ray.direction) );
    ElVisFloat r   = MAKE_FLOAT(1.0) / v;

    ElVisFloat3 e2 = TriangleVertex0 - MakeFloat3(ray.origin);
    ElVisFloat va  = dot( n, e2 );
    ElVisFloat t   = r*va;

    if(t < ray.tmax && t > ray.tmin) 
    {
        ElVisFloat3 i   = cross( e2, MakeFloat3(ray.direction) );
        ElVisFloat v1   = dot( i, e1 );
        ElVisFloat beta = r*v1;
        if(beta >= MAKE_FLOAT(0.0))
        {
            ElVisFloat v2 = dot( i, e0 );
            ElVisFloat gamma = r*v2;
            if( (v1+v2)*v <= v*v && gamma >= MAKE_FLOAT(0.0) ) 
            {
                if(  rtPotentialIntersection( t ) ) 
                {
                    normal = n;
                    rtReportIntersection(0);
                }
            }
        }   
    }
}

RT_PROGRAM void triangle_bounding (int, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = fminf( fminf( ConvertToFloat3(TriangleVertex0), ConvertToFloat3(TriangleVertex1)), ConvertToFloat3(TriangleVertex2) );
    aabb->m_max = fmaxf( fmaxf( ConvertToFloat3(TriangleVertex0), ConvertToFloat3(TriangleVertex1)), ConvertToFloat3(TriangleVertex2) );
}

