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

#ifndef ELVIS_CORE_PRINTF_CU
#define ELVIS_CORE_PRINTF_CU

#ifdef ELVIS_OPTIX_MODULE

    #ifdef ELVIS_ENABLE_PRINTF
        #include <ElVis/Core/OptixVariables.cu>

        #define ELVIS_PRINTF(...) \
            if( EnableTrace ) \
            { \
                if( (TracePixel.x == -1 || TracePixel.x == launch_index.x) && \
                    (TracePixel.y == -1 || TracePixel.y == (color_buffer.size().y - launch_index.y -1) ) ) \
                { \
                    rtPrintf(__VA_ARGS__); \
                }\
             }
    #else
        #define ELVIS_PRINTF(...)
    #endif

#else

#include <ElVis/Core/Cuda.h>


//The macro CUPRINTF is defined for architectures
//with different compute capabilities.
#if __CUDA_ARCH__ < 200 	//Compute capability 1.x architectures
#define ELVIS_PRINTF(fmt, ...) ;
#else						//Compute capability 2.x architectures
#define ELVIS_PRINTF printf
#endif


#endif

#endif //ELVIS_CORE_PRINTF_CU


