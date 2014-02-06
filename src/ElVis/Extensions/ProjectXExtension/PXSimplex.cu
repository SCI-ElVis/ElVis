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

#ifndef _PXSIMPLEX_CU
#define _PXSIMPLEX_CU

#include <float.h>
#include <optix_cuda.h>
#include <optix_math.h>
#include <ElVis/Core/matrix.cu>
#include <optixu/optixu_aabb.h>
#include <ElVis/Core/CutSurfacePayloads.cu>
#include <ElVis/Core/typedefs.cu>
#include <ElVis/Core/util.cu>


// rtBuffer<uint> PXSimplexSeed;
// rtDeclareVariable(ElVisFloat, randomFloat, attribute randomFloat, );
// rtDeclareVariable(ElVisFloat4, randomPoint, attribute randomPoint, );
//rtDeclareVariable(unsigned int, intersectedEgrp, attribute sharedUint, );
//rtDeclareVariable(unsigned int, intersectedElem, attribute sharedUint, );


//static unsigned int seed = 3873;
// __device__ ElVisFloat myRandom(unsigned int *localseed){
//   const float invintmax = 1.0f/65535.0f;
//   const unsigned int P1 = 16901;
//   const unsigned int P2 = 1949;

//   *localseed = (P1*(*localseed) + P2) % 65535;
  
//   return 2.0f*((float) (*localseed))*invintmax - 1.0f;
  
// }



//RT_PROGRAM void PXSimplexContainsOriginRootFindingMethod(int primIdx)
//{
//    WorldPoint w = ray.origin;
//    TensorPoint tp = TransformWorldToTensor(w);
//    if( tp.x <= -1.001f || tp.x >= 1.001f || 
//        tp.y <= -1.001f || tp.y >= 1.001f || 
//        tp.z <= -1.001f || tp.z >= 1.001f )
//    {
//        return;
//    }
//
//    //float t;
//    //FindPlaneIntersection(ray, plane_buffer[primIdx], t);
//    //if(  rtPotentialIntersection( t ) ) 
//    //{
//    //    rtReportIntersection(0);
//    //}
//    if( rtPotentialIntersection(.1) )
//    {
//        rtReportIntersection(0);
//    }
//}

#endif //end _PXSIMPLEX_CU
