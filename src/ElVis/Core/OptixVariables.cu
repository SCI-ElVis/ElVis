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

#ifndef ELVIS_OPTIX_VARIABLES_CU
#define ELVIS_OPTIX_VARIABLES_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_aabb.h>
#include <ElVis/Core/CutSurfacePayloads.cu>
#include <ElVis/Core/VolumeRenderingPayload.cu>
#include <ElVis/Core/FaceDef.h>

rtDeclareVariable(ElVisFloat3, normal, attribute normal_vec, );

rtDeclareVariable(rtObject, PrimaryRayGeometry, , );

rtDeclareVariable(int, FieldId, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(int, EnableTrace, , );
rtDeclareVariable(int2, TracePixel, , );

rtDeclareVariable(ElementFinderPayload, intersectionPointPayload, rtPayload, );
rtDeclareVariable(VolumeRenderingPayload, volumePayload, rtPayload, );
rtDeclareVariable(CutSurfaceScalarValuePayload, payload, rtPayload, );

rtDeclareVariable(ElVisFloat3, VolumeMinExtent, , );
rtDeclareVariable(ElVisFloat3, VolumeMaxExtent, , );

rtBuffer<uchar4, 2> color_buffer;
rtBuffer<ElVisFloat3, 2> raw_color_buffer;

rtBuffer<ElVisFloat, 2> SampleBuffer;

rtBuffer<ElVisFloat3, 2> normal_buffer;
rtBuffer<ElVisFloat3, 2> intersection_buffer;

rtBuffer<int, 2> ElementIdBuffer;
rtBuffer<int, 2> ElementTypeBuffer;

rtDeclareVariable(float, closest_t, rtIntersectionDistance, );

rtDeclareVariable(rtObject,      element_group, , );

rtBuffer<float, 2> depth_buffer;

// For depth buffer calculations for interop with OpenGL.
rtDeclareVariable(float, near, , );
rtDeclareVariable(float, far, , );
rtDeclareVariable(int, DepthBits, , );


// For face intersections.
rtBuffer<ElVis::FaceDef, 1> FaceIdBuffer;
rtBuffer<ElVisFloat3, 1> FaceMinExtentBuffer;
rtBuffer<ElVisFloat3, 1> FaceMaxExtentBuffer;

rtDeclareVariable(rtObject, DummyGroup, , );

rtDeclareVariable(int, intersectedFaceId, attribute IntersectedFaceId, );
rtDeclareVariable(ElVisFloat2, faceIntersectionReferencePoint, attribute FaceIntersectionReferencePoint, );
rtDeclareVariable(bool, faceIntersectionReferencePointIsValid, attribute FaceIntersectionReferencePointIsValid, );

rtDeclareVariable(ElVisFloat3, HeadlightColor, ,);

RT_PROGRAM void Fake()
{
    int payload;
    optix::size_t2 screen = color_buffer.size();

    ElVisFloat2 d = MakeFloat2(launch_index) / MakeFloat2(screen) * MAKE_FLOAT(2.0) - MAKE_FLOAT(1.0);

    ElVisFloat3 ray_origin;
    ElVisFloat3 ray_direction;

    optix::Ray ray = optix::make_Ray( ConvertToFloat3(ray_origin), ConvertToFloat3(ray_direction), 0, .001f, RT_DEFAULT_MAX);

    rtTrace(DummyGroup, ray, payload);
}

#endif

