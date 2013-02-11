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

#ifndef TEXTURE_COLOR_MAP_CU
#define TEXTURE_COLOR_MAP_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include <optixu/optixu_matrix.h>
#include <ElVis/Core/util.cu>
#include <ElVis/Core/ConvertToColor.cu>
#include <ElVis/Core/Float.cu>
#include <ElVis/Core/OptixVariables.cu>

// Color mapping is implemented as a separate invocation to the OptiX engine.
// Any launch_index with a scalar that is not FLT_MAX will go through the 
// color mapping process.

// To use textures, comment out this line and uncomment the textureSampler code.
// Testing has shown that textures are slightly slower and take longer to compile
rtBuffer<float4, 1> ColorMapTexture;
//rtTextureSampler<float4, 1> t;


rtDeclareVariable(float, TextureMinScalar, ,);
rtDeclareVariable(float, TextureMaxScalar, ,);


RT_PROGRAM void TextureColorMap()
{

    ElVisFloat scalar = SampleBuffer[launch_index];

    // Use a comparison because I can't get an equality to ELVIS_FLOAT_MAX to 
    // work.
    if( scalar > ELVIS_FLOAT_COMPARE ) 
    {
        return;
    }

    ElVisFloat normalized = (scalar - TextureMinScalar)/(TextureMaxScalar - TextureMinScalar);
    normalized = fmaxf(normalized, MAKE_FLOAT(0.0));
    normalized = fminf(normalized, MAKE_FLOAT(1.0));
    
    int index = normalized*ColorMapTexture.size();

    ELVIS_PRINTF("Color map index %d\n", index);
    ELVIS_PRINTF("Min %f, Max %f, scalar %f Normalized %f\n", TextureMinScalar, TextureMaxScalar, scalar, normalized);
    if( index == ColorMapTexture.size() )
    {
        // TODO - Setup the texture so I don't have this branch.
        index = index - 1;
    }
    float4 resultColor = ColorMapTexture[index];

    //float4 resultColor = tex1D( t, normalized);

    ELVIS_PRINTF("(%f, %f) - %f to %f give color (%f, %f, %f)\n", TextureMinScalar, TextureMaxScalar, scalar, normalized,
        resultColor.x, resultColor.y, resultColor.z);

    // Set the actual color in the color buffer since we don't 
    // know for sure if the lighting module will be called.
    raw_color_buffer[launch_index] = MakeFloat3(resultColor.x, resultColor.y, resultColor.z);
    color_buffer[launch_index] = ConvertToColor(MakeFloat3(resultColor.x, resultColor.y, resultColor.z));
}


#endif 
