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

#ifndef ELVIS_VOLUME_RENDERING_SINGLE_RAY_PER_SEGMENT_CU
#define ELVIS_VOLUME_RENDERING_SINGLE_RAY_PER_SEGMENT_CU

#include <ElVis/Core/Float.cu>
#include <ElVis/Core/FieldEvaluator.cu>
#include <ElVis/Math/TrapezoidalIntegration.hpp>
#include <ElVis/Core/TransferFunction.h>
#include <math_functions.h>

namespace ElVis
{

  extern "C" __global__ void IntegrateSegmentSingleThreadPerRayRiemann(
    ElVisFloat3 origin,
    const int* __restrict__ segmentElementId,
    const int* __restrict__ segmentElementType,
    const ElVisFloat3* __restrict__ segmentDirection,
    const ElVisFloat* __restrict__ segmentStart,
    const ElVisFloat* __restrict__ segmentEnd,
    int fieldId,
    TransferFunction* transferFunction,
    ElVisFloat epsilon,
    ElVisFloat desiredH,
    uint screenx,
    uint screeny,
    bool enableTrace,
    int tracex,
    int tracey,
    int* numSamples,
    ElVisFloat* __restrict__ densityAccumulator,
    ElVisFloat3* __restrict__ colorAccumulator)
  {
    int2 trace = make_int2(tracex, tracey);

    uint2 pixel;
    pixel.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixel.y = blockIdx.y * blockDim.y + threadIdx.y;

    bool traceEnabled =
      (pixel.x == trace.x && pixel.y == trace.y && enableTrace);
    uint2 screen = make_uint2(screenx, screeny);

    if (pixel.x >= screen.x || pixel.y >= screen.y)
    {
      return;
    }

    int segmentIndex = pixel.x + screen.x * pixel.y;
    if (segmentEnd[segmentIndex] < MAKE_FLOAT(0.0))
    {
      return;
    }

    int elementId = segmentElementId[segmentIndex];
    if (elementId == -1)
    {
      return;
    }

    int elementTypeId = segmentElementType[segmentIndex];
    ElVisFloat accumulatedDensity = densityAccumulator[segmentIndex];
    ElVisFloat3 color = colorAccumulator[segmentIndex];
    ElVisFloat a = segmentStart[segmentIndex];
    ElVisFloat b = segmentEnd[segmentIndex];

    ElVisFloat3 dir = segmentDirection[segmentIndex];
    ElVisFloat d = (b - a);

    if (d == MAKE_FLOAT(0.0))
    {
      return;
    }

    int n = Floor(d / desiredH);

    ElVisFloat h;

    if (n <= 1)
    {
      h = b - a;
      n = 1;
    }
    else
    {
      h = d / (ElVisFloat)(n - 1);
    }

    if (traceEnabled)
    {
      // ELVIS_PRINTF("Total segment range: [%2.15f, %2.15f], segment Id %d\n",
      // segmentStart[segmentIndex], segmentEnd[segmentIndex], segmentIndex);
      // ELVIS_PRINTF("D = %2.15f, H = %2.15f, N = %d\n", d, h, n);
    }

    // First test for density identically 0.  This means the segment does not
    // contribute at
    // all to the integral and can be skipped.
    FieldEvaluator f;
    f.Origin = origin;
    f.Direction = dir;
    f.ElementId = elementId;
    f.ElementType = elementTypeId;
    f.sampleCount = numSamples;
    f.FieldId = fieldId;

    ElVisFloat s0 = f(a);
    ElVisFloat d0 = transferFunction->Sample(eDensity, s0);
    ElVisFloat3 color0 = transferFunction->SampleColor(s0);
    ElVisFloat atten = expf(-accumulatedDensity);
    color += h * color0 * d0 * atten;

    accumulatedDensity += d0 * h;

    for (int i = 1; i < n; ++i)
    {
      ElVisFloat t = a + i * h;
      ElVisFloat sample = f(t);
      ElVisFloat densityValue = transferFunction->Sample(eDensity, sample);

      ElVisFloat3 sampleColor = transferFunction->SampleColor(sample);

      ElVisFloat atten = expf(-accumulatedDensity);

      color += h * sampleColor * densityValue * atten;

      accumulatedDensity += densityValue * h;
    }

    densityAccumulator[segmentIndex] = accumulatedDensity;
    colorAccumulator[segmentIndex] = color;
  }

  extern "C" __global__ void Trapezoidal_SingleThreadPerRay(
    ElVisFloat3 origin,
    const int* __restrict__ segmentElementId,
    const int* __restrict__ segmentElementType,
    const ElVisFloat3* __restrict__ segmentDirection,
    const ElVisFloat* __restrict__ segmentStart,
    const ElVisFloat* __restrict__ segmentEnd,
    int fieldId,
    TransferFunction* transferFunction,
    ElVisFloat epsilon,
    ElVisFloat desiredH,
    uint screenx,
    uint screeny,
    bool enableTrace,
    int tracex,
    int tracey,
    int* numSamples,
    ElVisFloat* __restrict__ densityAccumulator,
    ElVisFloat3* __restrict__ colorAccumulator)
  {
    int2 trace = make_int2(tracex, tracey);

    uint2 pixel;
    pixel.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixel.y = blockIdx.y * blockDim.y + threadIdx.y;

    bool traceEnabled =
      (pixel.x == trace.x && pixel.y == trace.y && enableTrace);

    uint2 screen = make_uint2(screenx, screeny);

    if (pixel.x >= screen.x || pixel.y >= screen.y)
    {
      return;
    }

    int segmentIndex = pixel.x + screen.x * pixel.y;
    if (segmentEnd[segmentIndex] < MAKE_FLOAT(0.0))
    {
      return;
    }

    int elementId = segmentElementId[segmentIndex];
    if (elementId == -1)
    {
      return;
    }

    int elementTypeId = segmentElementType[segmentIndex];

    if (traceEnabled)
    {
      // ELVIS_PRINTF("Trapezoidal_SingleThreadPerRay: Processing segment id
      // %d\n", segmentIndex);
    }
    ElVisFloat accumulatedDensity = densityAccumulator[segmentIndex];
    ElVisFloat3 color = colorAccumulator[segmentIndex];
    ElVisFloat a = segmentStart[segmentIndex];
    ElVisFloat b = segmentEnd[segmentIndex];

    ElVisFloat3 dir = segmentDirection[segmentIndex];
    ElVisFloat d = (b - a);

    if (d == MAKE_FLOAT(0.0))
    {
      return;
    }

    int n = Floor(d / desiredH);

    ElVisFloat h;

    if (n == 0)
    {
      h = b - a;
      n = 1;
    }
    else
    {
      h = d / (ElVisFloat)(n);
    }

    // First test for density identically 0.  This means the segment does not
    // contribute at
    // all to the integral and can be skipped.
    FieldEvaluator f;
    f.Origin = origin;
    f.Direction = dir;
    f.ElementId = elementId;
    f.ElementType = elementTypeId;
    f.sampleCount = numSamples;
    f.FieldId = fieldId;

    ElVisFloat s0 = f(a);
    ElVisFloat3 color0 = transferFunction->SampleColor(s0);
    ElVisFloat d0 = transferFunction->Sample(eDensity, s0);
    ElVisFloat atten = expf(-accumulatedDensity);
    color += h * MAKE_FLOAT(.5) * color0 * d0 * atten;

    for (int i = 1; i < n; ++i)
    {
      ElVisFloat t = a + i * h;
      ElVisFloat sample = f(t);
      ElVisFloat d1 = transferFunction->Sample(eDensity, sample);

      accumulatedDensity += MAKE_FLOAT(.5) * h * (d0 + d1);

      ElVisFloat3 colorSample = transferFunction->SampleColor(sample);
      ElVisFloat atten = expf(-accumulatedDensity);

      color += h * colorSample * d1 * atten;

      d0 = d1;
    }

    ElVisFloat sn = f(b);
    ElVisFloat3 colorn = transferFunction->SampleColor(sn);
    ElVisFloat dn = transferFunction->Sample(eDensity, sn);
    accumulatedDensity += MAKE_FLOAT(.5) * h * (d0 + dn);
    atten = expf(-accumulatedDensity);
    color += h * MAKE_FLOAT(.5) * colorn * dn * atten;

    densityAccumulator[segmentIndex] = accumulatedDensity;
    colorAccumulator[segmentIndex] = color;
  }
}

#endif // ELVIS_VOLUME_RENDERING_SINGLE_RAY_PER_SEGMENT_CU
