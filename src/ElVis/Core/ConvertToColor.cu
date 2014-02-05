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


#ifndef ELVIS_CONVERT_TO_COLOR_CU
#define ELVIS_CONVERT_TO_COLOR_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include <ElVis/Core/Float.cu>

// Converts a floating point color representation, where every channel is 
// in the range [0,1] to a 1-byte per channel RGBA color in the range [0,255], 
// with an alpha of 255.
__device__ __forceinline__ uchar4 ConvertToColor(const ElVisFloat3& c)
{
    return make_uchar4( static_cast<unsigned char>(c.x*MAKE_FLOAT(255.99)),  /* R */
                        static_cast<unsigned char>(c.y*MAKE_FLOAT(255.99)),  /* G */
                        static_cast<unsigned char>(c.z*MAKE_FLOAT(255.99)),  /* B */
                        255u);                                                 /* A */
}

#endif
