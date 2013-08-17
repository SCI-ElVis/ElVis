// Copyright NVIDIA Corporation 2009
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES


#include "sunsky.h"

using namespace optix;

/******************************************************************************\
 *
 * Minimal background shader using preetham model.  Applies constant scaling
 * factor to returned result in case post-process tonemapping is not desired
 * and only shows visible sun for 0-depth (eye) rays.
 *
\******************************************************************************/

struct PerRayData_radiance
{
  float3 result;
  float importance;
  int depth;
};

rtDeclareVariable( optix::Ray,          ray,          rtCurrentRay, );
rtDeclareVariable( PerRayData_radiance, prd_radiance, rtPayload, );

rtDeclareVariable( float, sky_scale, , )=1.0f;

RT_PROGRAM void miss()
{
  prd_radiance.result = sky_scale * querySkyModel( prd_radiance.depth == 0 , ray.direction );
}   
