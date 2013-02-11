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

#ifndef ELVIS_CORE_VOLUME_RENDERING_CU
#define ELVIS_CORE_VOLUME_RENDERING_CU

#include <ElVis/Core/Float.cu>

namespace ElVis
{
    extern "C" __global__ void PopulateColorBuffer(ElVisFloat3* __restrict__ accumulatedColor, ElVisFloat* __restrict__ accumulatedOpacity, uchar4* __restrict__ colorBuffer, int bufSize,
                                                   ElVisFloat3 bgColor)
    {
        int index = blockIdx.x*blockDim.x + threadIdx.x;
        if( index >= bufSize ) 
        {
            return;
        }

        ElVisFloat atten = expf(-accumulatedOpacity[index]);

        ElVisFloat3 accumColor = accumulatedColor[index];
        uchar4 incomingColor = colorBuffer[index];

        ElVisFloat3 incomingColorAsFloat = MakeFloat3(incomingColor.x/MAKE_FLOAT(255.0), incomingColor.y/MAKE_FLOAT(255.0), incomingColor.z/MAKE_FLOAT(255.0));
        accumColor += incomingColorAsFloat*atten;

        uchar4 color;
        ElVisFloat red  = fminf(MAKE_FLOAT(1.0), accumColor.x);
        ElVisFloat green  = fminf(MAKE_FLOAT(1.0), accumColor.y);
        ElVisFloat blue  = fminf(MAKE_FLOAT(1.0), accumColor.z);
        color.x = 255u*red;
        color.y = 255u*green;
        color.z = 255u*blue;
        color.w = 255u;//*opacity[index];
        colorBuffer[index] = color;
    }

    extern "C" __global__ void ClearAccumulatorBuffers(ElVisFloat* __restrict__ opacity, ElVisFloat3* __restrict__ color, int bufSize)
    {
        int index = blockIdx.x*blockDim.x + threadIdx.x;
        if( index >= bufSize ) 
        {
            return;
        }

        opacity[index] = MAKE_FLOAT(0.0);
        color[index] = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    }

}

#endif //ELVIS_CORE_VOLUME_RENDERING_CU
