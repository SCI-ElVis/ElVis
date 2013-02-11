/////////////////////////////////////////////////////////////////////////////////
////
//// The MIT License
////
//// Copyright (c) 2006 Scientific Computing and Imaging Institute,
//// University of Utah (USA)
////
//// License for the specific language governing rights and limitations under
//// Permission is hereby granted, free of charge, to any person obtaining a
//// copy of this software and associated documentation files (the "Software"),
//// to deal in the Software without restriction, including without limitation
//// the rights to use, copy, modify, merge, publish, distribute, sublicense,
//// and/or sell copies of the Software, and to permit persons to whom the
//// Software is furnished to do so, subject to the following conditions:
////
//// The above copyright notice and this permission notice shall be included
//// in all copies or substantial portions of the Software.
////
//// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//// DEALINGS IN THE SOFTWARE.
////
/////////////////////////////////////////////////////////////////////////////////
//
//#include <optix_cuda.h>
//#include <optix_math.h>
//#include <optixu/optixu_matrix.h>
//#include <optixu/optixu_aabb.h>
//#include <ElVis/Float.cu>
//
//rtDeclareVariable(float3, center, , );
//rtDeclareVariable(float, radius, , );
//rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
//rtDeclareVariable(ElVisFloat3, normal, attribute normal_vec, );
//
//RT_PROGRAM void SphereIntersection( int primIdx )
//{
//    float3 d = ray.direction;
//	float3 o = ray.origin;
//
//	float3 oc = o-center;
//
//	float A = dot(d,d);
//	float B = 2*(dot(oc,d));
//	float C = dot(oc,oc) - radius*radius;
//	
//	float D = B*B - 4*A*C;
//
//	if( D < 0 )
//	{
//		return;
//    }
//                    
//	// In this case we know that there is at least 1 intersection.
//	float denom = 2.0f * A;
//	float square_D = sqrt(D);
//
//	// Of the two roots, this is the one which is closest to the viewer.
//	float t1 = (-B - square_D)/denom;
//
//	if( t1 > 0.0f )
//	{
//		if(  rtPotentialIntersection( t1 ) ) 
//        {
//            const float3 intersectionPoint = ray.origin + t1 * ray.direction;
//            normal = intersectionPoint - center;
//            normalize(normal);
//            rtReportIntersection(0);
//        }
//	}
//	else
//	{
//		float t2 = (-B + square_D)/denom;
//		
//		if( t2 > 0.0f )
//		{
//		    if(  rtPotentialIntersection( t2 ) ) 
//            {
//                const float3 intersectionPoint = ray.origin + t2 * ray.direction;
//                normal = intersectionPoint - center;
//                normalize(normal);
//                rtReportIntersection(0);
//            }
//        }
//	}
//}
//
//RT_PROGRAM void SphereBounding (int, float result[6])
//{
//    optix::Aabb* aabb = (optix::Aabb*)result;
//    aabb->m_min = center - make_float3(radius);
//    aabb->m_max = center + make_float3(radius);
//}
//
