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

#ifndef ELVIS_VOLUME_RENDERING_PAYLOAD_CU
#define ELVIS_VOLUME_RENDERING_PAYLOAD_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include <ElVis/Core/util.cu>
#include <ElVis/Core/Float.cu>
#include <ElVis/Core/OptixVariables.cu>

struct VolumeRenderingPayload
{
    ELVIS_DEVICE void Initialize()
    {
        FoundIntersection = false;
        ElementId = 0;
        ElementTypeId = 0;
        IntersectionT = MAKE_FLOAT(-1.0);
        FaceId.Value = -1;
    }

    bool FoundIntersection;
    unsigned int ElementId;
    unsigned int ElementTypeId;
    ElVisFloat IntersectionT;
    GlobalFaceIdx FaceId;
};

rtDeclareVariable(VolumeRenderingPayload, volumePayload, rtPayload, );

#endif //ELVIS_VOLUME_RENDERING_PAYLOAD_CU
