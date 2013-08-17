
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include "ringsOfSpheres.h"

using namespace optix;

ringOfSpheres_t::ringOfSpheres_t(float ringRadius_,
                                 float sphRadius_,
                                 float gap, //< length of gap from one sphere to next and one ring to next
                                 const int maxSpheresPerRing)
                                 : ringRadius(ringRadius_), sphRadius(sphRadius_)
{
  // Figure out how many spheres will fit, given the radii and gap.

  ringRadius += 0.0f;
  float ringCircum = 2.f * M_PIf * ringRadius;
  float perSph = sphRadius * 2.f + gap;
  float nSpheresfrac = ringCircum / perSph;
  int nSpheres = std::max(1, std::min(int(nSpheresfrac), maxSpheresPerRing));
  if(ringRadius <= 0.0f) 
	  nSpheres = 1;

  MakeSpheres(nSpheres);
}

// Fills in sphereCenters, with the spheres in their default orientation
void ringOfSpheres_t::MakeSpheres(const int nSpheres)
{
  sphCenters.resize(nSpheres);
  for(int i=0; i<nSpheres; i++) {
    float ang = 2.f * M_PIf * float(i) / float(nSpheres+0.01);
    sphCenters[i] = RotXZ(make_float3(ringRadius,0,0), ang);
  }
}

void allRings_t::MakeRings(const Wave_t &sphRad, //< The function for computing the abs of the radius of the spheres in each ring
                           const Wave_t &sphRadDamp, //< Dampen sphRad by this wave
                           const float gap, //< Distance between balls in the same ring and from ring to ring
                           const size_t maxRings, //< Maximum number of rings to allow
                           const int maxSpheresPerRing, //< limit the number of spheres in a ring
                           const float maxDiameter) //< limit diameter of the largest ring
{
  if(maxRings <= 0u) return;

  float usedRad = gap; // How far out we've already filled with rings, counting outer gap

  for(float r=0.f; Rings.size() < maxRings && r < maxDiameter; r+=0.001f) {
    float targetr = fabsf(sphRad(r)) * sphRadDamp(r);
    if(targetr < 0.f) break; // Once the target sphere diameter goes negative we're done
    if(targetr < gap) continue; // gap also specifies the minimum sphere radius

    if((r - usedRad > targetr) || // There's space to fit this size of sphere here
      (r == 0.f && targetr >= gap)) { // A sphere at the center is wanted
        if(r > 0.0f) targetr = r - usedRad; // Make the spheres as big as will fit so there are no gaps
        ringOfSpheres_t tmpRing(r, targetr, gap, maxSpheresPerRing);
        Rings.push_back(tmpRing);

        r = usedRad = r + targetr + gap;
    }
  }
}

void allRings_t::StepTime(const float t)
{
  // Apply acceleration to each enabled ring
  for(size_t i=0; i<Rings.size(); i++) {
    if(Rings[i].ringRot.useWave) {
      Rings[i].ringRot.rotVel += Rings[i].ringRot.rotAccelWave(t);
    }
    if(Rings[i].ringRot.freezeFrames > 0.0f)
      Rings[i].ringRot.rotVel = 0.0f;
  }

  // Dampen all the velocities
  if(Rings.size() > 1) {
    std::vector<float> tmpRotVel(Rings.size());

    for(size_t i=0; i<Rings.size(); i++) {
      if(i==0) {
        tmpRotVel[i] = Rings[i].ringRot.rotVel * 0.666667f + Rings[i+1].ringRot.rotVel * 0.333333f;
      } else if(i==Rings.size()-1) {
        tmpRotVel[i] = Rings[i].ringRot.rotVel * 0.666667f + Rings[i-1].ringRot.rotVel * 0.333333f;
      } else {
        tmpRotVel[i] = Rings[i].ringRot.rotVel * 0.5f + Rings[i-1].ringRot.rotVel * 0.25f + Rings[i+1].ringRot.rotVel * 0.25f;
      }
    }

    // Copy back from tmp array
    for(size_t i=0; i<Rings.size(); i++) {
      Rings[i].ringRot.rotVel = tmpRotVel[i];

      // Stop a ring if the user pressed its number
      if(Rings[i].ringRot.freezeFrames > 0.0f) {
        Rings[i].ringRot.rotVel = 0.0f;
        Rings[i].ringRot.freezeFrames--;
      }

      // Don't rotate a ring fast enough to temporally alias
	  float r = optix::fmaxf(0.001f, Rings[i].ringRadius);
	  float x = Rings[i].sphRadius * 0.4f / r;
	  x = optix::clamp(x, -1.0f, 1.0f);
      float ringMaxVel = asin(x);
      if(Rings[i].ringRot.rotVel > ringMaxVel) Rings[i].ringRot.rotVel = ringMaxVel;
      if(Rings[i].ringRot.rotVel < -ringMaxVel) Rings[i].ringRot.rotVel = -ringMaxVel;

      Rings[i].ringRot.rotPos += Rings[i].ringRot.rotVel;
    }
  } else {
      Rings[0].ringRot.rotPos += Rings[0].ringRot.rotVel;
  }
}

// Returns the number of spheres in all the rings
unsigned int allRings_t::TotalSpheres()
{
  unsigned int nSpheres = 0u;
  for(size_t r = 0; r < Rings.size(); r++)
    nSpheres += static_cast<unsigned int>( Rings[r].sphCenters.size() );

  return nSpheres;
}

allRings_t::allRings_t(const Wave_t &sphRad, const Wave_t &sphRadDamp, const float gap,
                       const size_t maxRings, const int maxSpheresPerRing, const float maxDiameter)
{
  MakeRings(sphRad, sphRadDamp, gap, maxRings, maxSpheresPerRing, maxDiameter);
}
