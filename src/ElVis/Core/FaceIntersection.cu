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

#ifndef ELVIS_CORE_FACE_INTERSECTION_CU
#define ELVIS_CORE_FACE_INTERSECTION_CU

__device__ VolumeRenderingPayload FindNextFaceIntersection(const ElVisFloat3& origin, 
           const ElVisFloat3& rayDirection)
{
  VolumeRenderingPayload payload;
  payload.Initialize();

  optix::Ray ray = optix::make_Ray(ConvertToFloat3(origin), ConvertToFloat3(rayDirection), 2, 1e-3, RT_DEFAULT_MAX);

  // do linear faces first, since they are fast.  Intersections with linear 
  // faces may help weed out bad curved matches.
  rtTrace(PlanarFaceGroup, ray, payload);
  rtTrace(CurvedFaceGroup, ray, payload);

  return payload;
}

__device__ void FaceBoundingBox(int globalFaceIdx, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;

    ElVisFloat3 p0 = FaceInfoBuffer[globalFaceIdx].MinExtent;
    ElVisFloat3 p1 = FaceInfoBuffer[globalFaceIdx].MaxExtent;

    //rtPrintf("FaceBoundingBoxProgram: (%f, %f, %f) - (%f, %f, %f)\n", 
    //  p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);
    aabb->m_min = make_float3(p0.x, p0.y, p0.z);
    aabb->m_max = make_float3(p1.x, p1.y, p1.z);
}

RT_PROGRAM void PlanarFaceBoundingBoxProgram(int primitiveId, float result[6])
{
  FaceBoundingBox(PlanarFaceToGlobalIdxMap[primitiveId], result);
}

RT_PROGRAM void CurvedFaceBoundingBoxProgram(int primitiveId, float result[6])
{
  FaceBoundingBox(CurvedFaceToGlobalIdxMap[primitiveId], result);
}

#endif
