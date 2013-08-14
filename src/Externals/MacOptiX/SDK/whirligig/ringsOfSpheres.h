
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

#ifndef ringOfSpheres_h
#define ringOfSpheres_h

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

#include <vector>

// Represents a sine wave with amplitude, frequency, phase, and bias
struct Wave_t {
  Wave_t(const float ampl_=1.f, const float freq_=1.f, const float phase_=0.f, const float bias_=0.f)
    : ampl(ampl_), freq(freq_), phase(phase_), bias(bias_) {}

  float operator()(const float t) const
  {
    return ampl * sinf(t * freq + phase) + bias;
  }

private:
  float ampl, freq, phase, bias;
};

// Describes everything about how this ring rotates
struct ringRotation_t {
  Wave_t rotAccelWave; // Acceleration function to add to rotVel each time step, if not just following neighbors
  float rotPos; // Ring's rotational position, in radians
  float rotVel; // Ring's rotational velocity, in radians per timestep
  float freezeFrames; // Clamp rotVel to 0 for this many more frames
  bool useWave; // True to set velocity to wave value prior to damping

  ringRotation_t() : rotPos(0.f), rotVel(0.f), freezeFrames(0.f), useWave(false) {}
};

class ringOfSpheres_t {
public:
  ringOfSpheres_t(float ringRadius_, float sphRadius_, float gap, const int maxSpheresPerRing);

  ringRotation_t ringRot; // everything about how this ring rotates

  float ringRadius; // radius from the center of the ring to the center of all spheres in the ring
  float sphRadius; // radius of the spheres in this ring
  std::vector<optix::float3> sphCenters; // Sphere locations (in X,Y plane)

private:
  // Fills in sphCenters, with the spheres in their default orientation
  void MakeSpheres(const int nSpheres);
};

class allRings_t {
public:
  allRings_t(const Wave_t &sphRad = Wave_t(), const Wave_t &sphRadDamp = Wave_t(), const float gap = 0.1f,
    const size_t maxRings = 0u, const int maxSpheresPerRing = 0u, const float maxDiameter = 0.0f);

  // Call once per frame to compute new rotational velocities of each ring
  void StepTime(const float t);

  // Returns the number of spheres in all the rings
  unsigned int TotalSpheres();

  std::vector<ringOfSpheres_t> Rings;

private:
  void MakeRings(const Wave_t &sphRad, //< The function for computing the abs of the radius of the spheres in each ring
    const Wave_t &sphRadDamp, //< Dampen sphRad by this wave
    const float gap, //< Distance between balls in the same ring and from ring to ring
    const size_t maxRings, //< Maximum number of rings to allow
    const int maxSpheresPerRing, //< limit the number of spheres in a ring
    const float maxDiameter); //< limit diameter of the largest ring
};

// Rotate the vector v about the Y axis
inline optix::float3 RotXZ(const optix::float3 v, const float ang)
{
  float ca = cosf(ang), sa = sinf(ang);
  return optix::make_float3(ca*v.x - sa*v.z, v.y, sa*v.x + ca*v.z);
}

#endif
