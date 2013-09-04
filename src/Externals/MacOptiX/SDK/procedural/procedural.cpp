
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

//-----------------------------------------------------------------------------
//
//  procedural.cpp - Demonstrates 1D color ramp lookup, 3D solid noise, and
//  turbulence implementation.  This is the basis for all sorts of famous
//  procedural textures like marble, wood, onion, etc. which are shown in
//  procedural.cu.
//
//-----------------------------------------------------------------------------



/*
  Manual:
  - Call application with Maya OBJ file on command line and 
    optional parameters to initialize the effect parameters.
    See the usage printout in the main() routine.

  Keys:  (Small letters decrease, capital letters increase.)
  t/T:   turbulence (default 1.0)
         Factor multiplied on the turbulence result. 
         Decreasing to 0.0 will show the root pattern!
  e/E:   frequency (default 1.0)
         ('f' was taken by the fullscreen toggle) 
         Factor multiplied on the input 3D point, bigger numbers let the pattern repeat more often.
  o/O:   octaves (default 1)
         Number of noise lookups in the turbulence function.
  l/L:   lambda (default 0.0)
         Factor multiplied on the input position to the turbulence. (A little like frequency.)
         Scaled by itself on each iteration.
         Only has an effect if octaves > 1.
  m/M:   omega (default 0.0)
         Factor multiplied on the noise() result in the turbulence function.
         Scaled by itself on each iteration.
         Only has an effect if octaves > 1.
  x/X:
  y/Y:
  z/Z:   Adjust the pattern origin in the resp. coordinate axis. 
         Default origin is 0.0, 0.0, 0.0;
  SPACE: Reset all the above values to their defaults.
  p/P:   Roll through the different procedural texture modes.

  Implemented procedural 3D textures:
  marble      : Base pattern maps x-axis to 1D. Planar slices orthogonal to x-axis.
  wood        : Base pattern maps distance to x-axis to 1D. Concentric circles around x-axis.
  onion       : Base pattern maps distance to pattern origin to 1D. Concentric spheres around pattern origin.
  noise_cubed : No orientation, just 3D noise lookup in range [0, 1] cubed to give smoother noise than
                the 3D solid noise data itself.  (Easiest to make look nice with the tweakable values.)
  voronoi     : Distance to equidistant surfaces beween the two nearest control points in a repeated pattern. Very nice!
  sphere      : 3D spheres. Only reacts on origin and frequency.
  checker     : 3D cubes in alternating colors in all directions. Only reacts on origin and frequency.
                The checker function itself return 0.0 or 1.0, but picks 1D color ramp at u = 0.0 and 0.5
                in this sample to unify the code.)

  Hints:
  - All patterns are defined around the unit cube area; visuals depend on the size of the geometry.
  - Try reducing the turbulence value with 't' to see the original pattern.
  - For wood you'd need to increase the frequency to get more rings (if your geometry fits into the unit cube).
  - Then increase the turbulence again, increase the octaves to 2 or higher and play with omega and lambda.
  - Shift the origin around with the xX, yY, zZ keys to see the solid nature of the textures.
  - Run the demo using different color ramps.
*/


#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <sutil.h>
#include "commonStructs.h"
#include <GLUTDisplay.h>
#include <ObjLoader.h>
#include <string>
#include <iostream>
#include <cstdlib>
#include <cstring>

using namespace optix;


//------------------------------------------------------------------------------
//
//  Auxiliary types
//
//-----------------------------------------------------------------------------

enum Procedure 
{
  PROCEDURE_MARBLE = 0, // default
  PROCEDURE_WOOD,
  PROCEDURE_ONION,
  PROCEDURE_VORONOI,
  PROCEDURE_NOISE_CUBED,
  PROCEDURE_SPHERE,
  PROCEDURE_CHECKER,
  PROCEDURE_COUNT // Must be last. Used to size default parameters array.
};

enum ColorRamp
{
  COLOR_RAMP_HUE = 0, // default
  COLOR_RAMP_MARBLE_BLUE,
  COLOR_RAMP_MARBLE_BROWN,
  COLOR_RAMP_WOOD,
  COLOR_RAMP_CRACKED,
  COLOR_RAMP_BLOODSTONE,
  COLOR_RAMP_TEMPERATURE,
};

typedef struct 
{
  Procedure proc;
  ColorRamp ramp;
  float     frequency;
  float     turbulence;
  int       octaves;
  float     lambda;
  float     omega;
  float     origin[3];
} ProceduralParameters;


typedef struct
{
  float u;      // u-coordinate in range [0.0, 1.0]
  float c[3];   // color, components in range [0.0, 1.0]
} color_ramp_element;


//-----------------------------------------------------------------------------
//
//  Global variables
//
//-----------------------------------------------------------------------------

// These are some nice looking defaults for the knot.obj
ProceduralParameters g_ParametersDefault[7] = 
{ 
  // Procedure,            ramp,                    frequency, turbulence, octaves, lambda, omega, origin[3]
  { PROCEDURE_MARBLE,      COLOR_RAMP_MARBLE_BROWN,  0.3f,     1.2f,       4,       0.35f,  0.75f, {0.0f, 0.0f, 0.0f} },
  { PROCEDURE_WOOD,        COLOR_RAMP_WOOD,         15.0f,     0.1f,       1,       0.0f,   0.0f,  {0.0f, 1.1f, 0.0f} },
  { PROCEDURE_ONION,       COLOR_RAMP_BLOODSTONE,    1.5f,     0.2f,       4,       0.2f,   2.3f,  {0.0f, 0.0f, 0.1f} },
  { PROCEDURE_VORONOI,     COLOR_RAMP_CRACKED,       2.0f,     0.0f,       1,       0.0f,   0.0f,  {0.0f, 0.0f, 0.0f} },
  { PROCEDURE_NOISE_CUBED, COLOR_RAMP_MARBLE_BLUE,   0.2f,     0.0f,       1,       0.0f,   0.0f,  {0.0f, 0.0f, 0.0f} },
  { PROCEDURE_SPHERE,      COLOR_RAMP_BLOODSTONE,    1.5f,     0.0f,       1,       0.0f,   0.0f,  {0.0f, 0.0f, 0.0f} },
  { PROCEDURE_CHECKER,     COLOR_RAMP_TEMPERATURE,   4.0f,     0.0f,       1,       0.0f,   0.0f,  {0.0f, 0.0f, 0.0f} }
};


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

static float rand_range(float min, float max)
{
  return min + (max - min) * (float) rand() / (float) RAND_MAX;
}


// To index into default parameters.
Procedure ProcedureStringToEnum( const std::string& name )
{
  if (name == "wood")
  {
    return PROCEDURE_WOOD;
  }
  else if (name == "onion")
  {
    return PROCEDURE_ONION;
  }
  else if (name == "voronoi")
  {
    return PROCEDURE_VORONOI;
  }
  else if (name == "noise_cubed")
  {
    return PROCEDURE_NOISE_CUBED;
  }
  else if (name == "sphere")
  {
    return PROCEDURE_SPHERE;
  }
  else if (name == "checker")
  {
    return PROCEDURE_CHECKER;
  }
  else if (name == "marble")
  { 
    return PROCEDURE_MARBLE;
  }
  else
  {
    std::cerr << "Unknown procedure '" << name << "' specified.  Exiting." << std::endl;
    exit(2);
  }
}


// To index into default parameters.
ColorRamp ColorRampStringToEnum( const std::string& name)
{
  if (name == "marble_blue")
  {
    return COLOR_RAMP_MARBLE_BLUE;
  }
  else if (name == "marble_brown")
  {
    return COLOR_RAMP_MARBLE_BROWN;
  }
  else if (name == "wood")
  {
    return COLOR_RAMP_WOOD;
  }
  else if (name == "cracked")
  {
    return COLOR_RAMP_CRACKED;
  }
  else if (name == "bloodstone")
  {
    return COLOR_RAMP_BLOODSTONE;
  }
  else if (name == "temperature")
  {
    return COLOR_RAMP_TEMPERATURE;
  }
  else if (name == "hue")
  {
    return COLOR_RAMP_HUE;
  }
  else 
  {
    std::cerr << "Unknown color ramp '" << name << "' specified.  Exiting." << std::endl;
    exit(2);
  }
}


//-----------------------------------------------------------------------------
// 
//  Color ramp building
// 
//  Build a color ramp in a 1D float4 texture.
//  Input is a vector of u-coordinates and colors which define the color ramp.
//  Example:
//  Input:
//  0.4, {1.0, 0.0, 0.0}
//  0.8, {0.0, 1.0, 0.0}
//  Output color ramp:
//  [0.0, 0.4) red to red 
//  [0.4, 0.8) red to green
//  [0.8, 1.0] green to green
//
//-----------------------------------------------------------------------------

bool color_ramp(std::vector<color_ramp_element> v, int size, float *ramp)
{
  if (v.size() < 1 || !size || !ramp) 
  {
    return false;
  }

  float r;
  float g;
  float b;
  float a = 1.0f; // CUDA doesn't support float3 textures.
  float *p = ramp;

  if (v.size() == 1)
  {
    // Special case, only one color in the input, means the whole color ramp is that color.
    r = v[0].c[0];
    g = v[0].c[1];
    b = v[0].c[2];

    for (int i = 0; i < size; i++)
    {
      *p++ = r;
      *p++ = g;
      *p++ = b;
      *p++ = a;
    }
    return true;
  }

  // Here v.size() is at least 2.
  color_ramp_element left;
  color_ramp_element right;
  size_t entry = 0;

  left = v[entry];
  if (left.u > 0.0f)
  {
    left.u = 0.0f;
  }
  else // left u == 0;
  {
    entry++;
  }
  right = v[entry++];

  for (int i = 0; i < size; i++)
  {
    // The 1D coordinate at which we need to calculate the color.
    float u = (float) i / (float) (size - 1);
    
    // Check if it's in the range [left.u, right.u)
    while (!(left.u <= u && u < right.u))
    {
      left = right;
      if (entry < v.size())
      {
        right = v[entry++];
      }
      else
      {
        // left is already the last entry, move right.u to the end of the range.
        right.u = 1.0001f; // Make sure we pass 1.0 < right.u in the last iteration.
        break;
      }
    }

    float t = (u - left.u) / (right.u - left.u);
    r = left.c[0] + (right.c[0] - left.c[0]) * t;
    g = left.c[1] + (right.c[1] - left.c[1]) * t;
    b = left.c[2] + (right.c[2] - left.c[2]) * t;

    *p++ = r;
    *p++ = g;
    *p++ = b;
    *p++ = a;
  }
  return true;
}


void create_color_ramp(ColorRamp ramp, int size, float *data)
{
  // Generate the color ramp definition vector.
  std::vector<color_ramp_element> color_ramp_definition;
  color_ramp_element cre;

  switch (ramp)
  {
    case COLOR_RAMP_MARBLE_BLUE:
      // white-blue
      cre.u = 0.0f;
      cre.c[0] = 1.0f;
      cre.c[1] = 1.0f;
      cre.c[2] = 1.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.15f;
      cre.c[0] = 0.0f;
      cre.c[1] = 0.0f;
      cre.c[2] = 0.8f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.3f;
      cre.c[0] = 1.0f;
      cre.c[1] = 1.0f;
      cre.c[2] = 1.0f;  
      color_ramp_definition.push_back(cre);
      break;

    case COLOR_RAMP_MARBLE_BROWN:
      // white with fine dark brown line
      cre.u = 0.0f;
      cre.c[0] = 255.0f / 255.0f;
      cre.c[1] = 255.0f / 255.0f;
      cre.c[2] = 240.0f / 255.0f;
      color_ramp_definition.push_back(cre);
      cre.u = 0.4f;
      cre.c[0] = 202.0f / 255.0f;
      cre.c[1] = 173.0f / 255.0f;
      cre.c[2] = 139.0f / 255.0f;
      color_ramp_definition.push_back(cre);
      cre.u = 0.5f;
      cre.c[0] = 23.0f / 255.0f;
      cre.c[1] =  7.0f / 255.0f;
      cre.c[2] =  0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.6f;
      cre.c[0] = 202.0f / 255.0f;
      cre.c[1] = 173.0f / 255.0f;
      cre.c[2] = 139.0f / 255.0f;
      color_ramp_definition.push_back(cre);
      cre.u = 1.0f;
      cre.c[0] = 255.0f / 255.0f;
      cre.c[1] = 255.0f / 255.0f;
      cre.c[2] = 240.0f / 255.0f;
      color_ramp_definition.push_back(cre);
      break;

    case COLOR_RAMP_WOOD:
      // warm red-brown
      cre.u = 0.0f; // dark
      cre.c[0] = 136.0f / 255.0f; 
      cre.c[1] =  43.0f / 255.0f;
      cre.c[2] =   9.0f / 255.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.2f; // middle
      cre.c[0] = 157.0f / 255.0f;
      cre.c[1] =  69.0f / 255.0f;
      cre.c[2] =   0.0f / 255.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.65f; // light
      cre.c[0] = 191.0f / 255.0f;
      cre.c[1] =  99.0f / 255.0f;
      cre.c[2] =  36.0f / 255.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.9f; // middle
      cre.c[0] = 157.0f / 255.0f;
      cre.c[1] =  69.0f / 255.0f;
      cre.c[2] =   0.0f / 255.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 1.0f; // dark
      cre.c[0] = 136.0f / 255.0f; 
      cre.c[1] =  43.0f / 255.0f;
      cre.c[2] =   9.0f / 255.0f;  
      color_ramp_definition.push_back(cre);
      break;

    case COLOR_RAMP_CRACKED:
      // Used with Voronoi this gives a mosaic like appearance.
      cre.u = 0.0f;
      cre.c[0] = 0.0f; // Black
      cre.c[1] = 0.0f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.1f;
      cre.c[0] = 0.0f;
      cre.c[1] = 0.0f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.1f;
      cre.c[0] = 74.0f / 255.0f; 
      cre.c[1] =  9.0f / 255.0f;
      cre.c[2] =  7.0f / 255.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.3f;
      cre.c[0] = 240.0f / 255.0f; 
      cre.c[1] = 192.0f / 255.0f;
      cre.c[2] =  10.0f / 255.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 1.0f;
      cre.c[0] = 1.0f; 
      cre.c[1] = 206.0f / 255.0f;
      cre.c[2] =  23.0f / 255.0f;  
      color_ramp_definition.push_back(cre);
      break;

    case COLOR_RAMP_BLOODSTONE:
      // green with sparkling red veins
      cre.u = 0.0f;
      cre.c[0] = 0.0f;
      cre.c[1] = 0.6f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.40f;
      cre.c[0] = 0.0f;
      cre.c[1] = 0.4f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.40f;
      cre.c[0] = 0.3f;
      cre.c[1] = 0.0f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.45f;
      cre.c[0] = 1.0f;
      cre.c[1] = 0.0f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.55f;
      cre.c[0] = 1.0f;
      cre.c[1] = 0.0f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.6f;
      cre.c[0] = 0.3f;
      cre.c[1] = 0.0f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.6f;
      cre.c[0] = 0.0f;
      cre.c[1] = 0.4f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 1.0f;
      cre.c[0] = 0.0f;
      cre.c[1] = 0.6f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      break;

    case COLOR_RAMP_TEMPERATURE:
      // Cold to hot: blue, green, red, yellow, white
      cre.u = 0.0f;
      cre.c[0] = 0.0f; // blue
      cre.c[1] = 0.0f;
      cre.c[2] = 1.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.25f;
      cre.c[0] = 0.0f; // green
      cre.c[1] = 1.0f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.5f;
      cre.c[0] = 1.0f; // red
      cre.c[1] = 0.0f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 0.75f;
      cre.c[0] = 1.0f; // yellow
      cre.c[1] = 1.0f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 1.0f;
      cre.c[0] = 1.0f; // white
      cre.c[1] = 1.0f;
      cre.c[2] = 1.0f;  
      color_ramp_definition.push_back(cre);
      break;

    default:
    case COLOR_RAMP_HUE:
      // standard color circle
      cre.u = 0.0f;
      cre.c[0] = 1.0f; // red
      cre.c[1] = 0.0f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 1.0f / 6.0f;
      cre.c[0] = 1.0f; // yellow
      cre.c[1] = 1.0f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 2.0f / 6.0f;
      cre.c[0] = 0.0f; // green
      cre.c[1] = 1.0f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 2.0f / 6.0f;
      cre.c[0] = 0.0f; // cyan
      cre.c[1] = 1.0f;
      cre.c[2] = 1.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 4.0f / 6.0f;
      cre.c[0] = 0.0f; // blue
      cre.c[1] = 0.0f;
      cre.c[2] = 1.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 5.0f / 6.0f;
      cre.c[0] = 1.0f; // magenta
      cre.c[1] = 0.0f;
      cre.c[2] = 1.0f;  
      color_ramp_definition.push_back(cre);
      cre.u = 1.0f;
      cre.c[0] = 1.0f; // red
      cre.c[1] = 0.0f;
      cre.c[2] = 0.0f;  
      color_ramp_definition.push_back(cre);
      break;
  }
  color_ramp(color_ramp_definition, size, data);
}


//-----------------------------------------------------------------------------
// 
//  Voronai texture routines. Expensive to build.
//
//-----------------------------------------------------------------------------

void minima(float x, float y, float z, float &a, float &b)
{
  float t = sqrtf(x * x + y * y + z * z);
  if (t < a) // New absolute minimum?
  {
    b = a;  // Move old minimum to second place.
    a = t;  // Remember the absolute minimum.
  }
  else if (t < b) // New second place minimum?
  {
    b = t;
  }
}


// Mind, this one-time 3D distance value setup grows with O(n^3) in texture dimensions, 
// don't overdo the 3D texture size.
void Voronoi_repeat(int count, int width, int height, int depth, float *data)
{
  // Initialize the control points.
  float *p = new float[count * 3];
  // Partitioning distance of the individual components.
  // (Means, ten times as many buckets as needed.)
  float d = 0.1f / (float) (count * 3);
  float f;
  
  p[0] = rand_range(0.0f, 1.0f);
  for (int i = 1; i < count * 3; i++)
  {
    float m0; // Minimum distance to components.
    // Make sure two coordinates are never hitting the same spot,
    // otherwise minima() could return 0.0 for the second nearest distance 
    // which would result in a division by zero.
    // Doing this on all components also spaces the points more equally.
    do {
      f = rand_range(0.0f, 1.0f); // New float component.
      
      m0 = fabs(p[0] - f);
      for (int j = 1; j < i; j++)
      {
        float m1 = fabs(p[j] - f);
        if (m1 < m0)
        {
          m0 = m1;
        }
      }
    } while (m0 < d);

    p[i] = f;
  }

  // Increments.
  float xs = 1.0f / (float) width;
  float ys = 1.0f / (float) height;
  float zs = 1.0f / (float) depth;

  // Coordinates.
  float zc = zs * 0.5f;
  for (int z = 0; z < depth; z++)
  {
    float yc = ys * 0.5f;
    for (int y = 0; y < height; y++)
    {
      float xc = xs * 0.5f;
      for (int x = 0; x < width; x++)
      {
        // Distances to the next point can only be in the range [0, 1] for the repeating pattern.
        float m0 = 100.0f;
        float m1 = 100.0f;

        for (int i = 0; i < count; i++)
        {
          // The Voronoi control point.
          float xp = p[i * 3    ]; 
          float yp = p[i * 3 + 1]; 
          float zp = p[i * 3 + 2]; 

          // For a repeating effect we need to take 27 cubes into account!
          minima(xp - 1.0f - xc, yp - 1.0f - yc, zp - 1.0f - zc, m0, m1);
          minima(xp        - xc, yp - 1.0f - yc, zp - 1.0f - zc, m0, m1);
          minima(xp + 1.0f - xc, yp - 1.0f - yc, zp - 1.0f - zc, m0, m1);
          minima(xp - 1.0f - xc, yp        - yc, zp - 1.0f - zc, m0, m1);
          minima(xp        - xc, yp        - yc, zp - 1.0f - zc, m0, m1);
          minima(xp + 1.0f - xc, yp        - yc, zp - 1.0f - zc, m0, m1);
          minima(xp - 1.0f - xc, yp + 1.0f - yc, zp - 1.0f - zc, m0, m1);
          minima(xp        - xc, yp + 1.0f - yc, zp - 1.0f - zc, m0, m1);
          minima(xp + 1.0f - xc, yp + 1.0f - yc, zp - 1.0f - zc, m0, m1);
                                                             
          minima(xp - 1.0f - xc, yp - 1.0f - yc, zp        - zc, m0, m1);
          minima(xp        - xc, yp - 1.0f - yc, zp        - zc, m0, m1);
          minima(xp + 1.0f - xc, yp - 1.0f - yc, zp        - zc, m0, m1);
          minima(xp - 1.0f - xc, yp        - yc, zp        - zc, m0, m1);
          minima(xp        - xc, yp        - yc, zp        - zc, m0, m1);
          minima(xp + 1.0f - xc, yp        - yc, zp        - zc, m0, m1);
          minima(xp - 1.0f - xc, yp + 1.0f - yc, zp        - zc, m0, m1);
          minima(xp        - xc, yp + 1.0f - yc, zp        - zc, m0, m1);
          minima(xp + 1.0f - xc, yp + 1.0f - yc, zp        - zc, m0, m1);
                                                             
          minima(xp - 1.0f - xc, yp - 1.0f - yc, zp + 1.0f - zc, m0, m1);
          minima(xp        - xc, yp - 1.0f - yc, zp + 1.0f - zc, m0, m1);
          minima(xp + 1.0f - xc, yp - 1.0f - yc, zp + 1.0f - zc, m0, m1);
          minima(xp - 1.0f - xc, yp        - yc, zp + 1.0f - zc, m0, m1);
          minima(xp        - xc, yp        - yc, zp + 1.0f - zc, m0, m1);
          minima(xp + 1.0f - xc, yp        - yc, zp + 1.0f - zc, m0, m1);
          minima(xp - 1.0f - xc, yp + 1.0f - yc, zp + 1.0f - zc, m0, m1);
          minima(xp        - xc, yp + 1.0f - yc, zp + 1.0f - zc, m0, m1);
          minima(xp + 1.0f - xc, yp + 1.0f - yc, zp + 1.0f - zc, m0, m1);
        }

        *data++ = 1.0f - m0 / m1;
        xc += xs;
      }
      yc += ys;
    }
    zc += zs;
  }
  delete [] p;
}


//------------------------------------------------------------------------------
//
//  ObjScene
//
//------------------------------------------------------------------------------

class ObjScene : public SampleScene
{
public:
  ObjScene(const std::string& filename,
           const std::string& proc_name,
           const ProceduralParameters& parameters) 
    : m_filename(filename), m_proc_name(proc_name), m_parameters(parameters)  {}

  virtual void   initScene( InitialCameraData& camera_data);
  virtual void   trace(const RayGenCameraData& camera_data);
  virtual Buffer getOutputBuffer();

  virtual bool keyPressed(unsigned char key, int x, int y);

private:
  std::string m_filename;
  std::string m_proc_name;
  ProceduralParameters m_parameters;
  const static int WIDTH;
  const static int HEIGHT;
};

const int ObjScene::WIDTH  = 1024;
const int ObjScene::HEIGHT = 768;


// Demo interaction:
bool ObjScene::keyPressed(unsigned char key, int x, int y)
{
  bool update = false;

  // Mind: some keys are taken by the parent class.
  switch (key)
  {
    // Frequency
    case 'e':
      m_parameters.frequency -= 0.5f;
      update = true;
      break;
    case 'E':
      m_parameters.frequency += 0.5f;
      update = true;
      break;

    // Turbulence (amount)
    case 't':
      m_parameters.turbulence -= 0.05f;
      update = true;
      break;
    case 'T':
      m_parameters.turbulence += 0.05f;
      update = true;
      break;

    // Octaves
    case 'o':
      m_parameters.octaves--;
      if (m_parameters.octaves < 1)
      {
        m_parameters.octaves = 1;
      }
      update = true;
      break;
    case 'O':
      m_parameters.octaves++;
      update = true;
      break;

    // Lambda
    case 'l':
      m_parameters.lambda -= 0.05f;
      update = true;
      break;
    case 'L':
      m_parameters.lambda += 0.05f;
      update = true;
      break;

    // Omega
    case 'g':
      m_parameters.omega -= 0.05f;
      update = true;
      break;
    case 'G':
      m_parameters.omega += 0.05f;
      update = true;
      break;

    // Origin.x
    case 'x':
      m_parameters.origin[0] -= 0.1f;
      update = true;
      break;
    case 'X':
      m_parameters.origin[0] += 0.1f;
      update = true;
      break;

    // Origin.y
    case 'y':
      m_parameters.origin[1] -= 0.1f;
      update = true;
      break;
    case 'Y':
      m_parameters.origin[1] += 0.1f;
      update = true;
      break;

    // Origin.x
    case 'z':
      m_parameters.origin[2] -= 0.1f;
      update = true;
      break;
    case 'Z':
      m_parameters.origin[2] += 0.1f;
      update = true;
      break;
    
    // Reset to defaults
    case ' ': // SPACE
      m_parameters = g_ParametersDefault[ProcedureStringToEnum(m_proc_name)];
      update = true;
      break;
  }

  if (update)
  {
    m_context["frequency"]->setFloat(m_parameters.frequency);
    m_context["turbulence"]->setFloat(m_parameters.turbulence);
    m_context["octaves"]->setInt(m_parameters.octaves);
    m_context["lambda"]->setFloat(m_parameters.lambda);
    m_context["omega"]->setFloat(m_parameters.omega);
    m_context["origin"]->setFloat(m_parameters.origin[0],
                                  m_parameters.origin[1],
                                  m_parameters.origin[2]);

    std::cerr << std::endl << "freqency = " << m_parameters.frequency << std::endl;
    std::cerr << "turbulence = " << m_parameters.turbulence << std::endl;
    std::cerr << "octaves = " << m_parameters.octaves << std::endl;
    std::cerr << "lambda = " << m_parameters.lambda << std::endl;
    std::cerr << "omega = " << m_parameters.omega << std::endl;
    std::cerr << "origin = " << m_parameters.origin[0] << ", " 
                             << m_parameters.origin[1] << ", " 
                             << m_parameters.origin[2] << std::endl;
  }
  return update;
}



void ObjScene::initScene(InitialCameraData& camera_data)
{
  // Setup context
  m_context->setRayTypeCount(2);
  m_context->setEntryPointCount(1);
  m_context->setStackSize(400);

  m_context["radiance_ray_type"]->setUint(0u);
  m_context["shadow_ray_type"]->setUint(1u);
  m_context["scene_epsilon"]->setFloat(1.e-4f);
  m_context["max_depth"]->setInt(5);
  m_context["ambient_light_color"]->setFloat(0.2f, 0.2f, 0.2f);

  // Output buffer
  Buffer outputBuffer = createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT);
  m_context["output_buffer"]->set(outputBuffer);

  // Lights buffer
  BasicLight lights[] = 
  { // Light at left top front
    { make_float3(-60.0f, 60.0f, 60.0f), make_float3(1.0f, 1.0f, 1.0f), 1 }
  };

  Buffer lightBuffer = m_context->createBuffer(RT_BUFFER_INPUT);
  lightBuffer->setFormat(RT_FORMAT_USER);
  lightBuffer->setElementSize(sizeof(BasicLight));
  lightBuffer->setSize(sizeof(lights) / sizeof(lights[0]));
  memcpy(lightBuffer->map(), lights, sizeof(lights));
  lightBuffer->unmap();
  m_context["lights"]->set(lightBuffer);

  // Ray generation program
  std::string ptx_path = ptxpath( "procedural", "pinhole_camera.cu");
  Program ray_gen_program = m_context->createProgramFromPTXFile(ptx_path, "pinhole_camera");
  m_context->setRayGenerationProgram(0, ray_gen_program);

  // Exception / miss programs
  m_context->setExceptionProgram(0, m_context->createProgramFromPTXFile(ptx_path, "exception"));
  m_context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);

  m_context->setMissProgram(0, m_context->createProgramFromPTXFile(ptxpath("procedural", "constantbg.cu"), "miss"));
  m_context["bg_color"]->setFloat(0.0f, 0.0f, 0.0f);

  // Procedural materials (marble, wood, onion, noise_cubed, voronoi, sphere, checker)
  // If a wrong --proc parameter is used this will fail.
  std::string closest_hit_func_name( std::string( "closest_hit_" ) + m_proc_name );
  Program closest_hit_program = m_context->createProgramFromPTXFile(ptxpath("procedural", "procedural.cu"), closest_hit_func_name );
  Program any_hit_program     = m_context->createProgramFromPTXFile(ptxpath("procedural", "procedural.cu"), "any_hit_shadow");

  Material material = m_context->createMaterial();
  material->setClosestHitProgram(0u, closest_hit_program);
  material->setAnyHitProgram(1u, any_hit_program);

  // 3D solid noise buffer, 1 float channel, all entries in the range [0.0, 1.0].
  srand(0); // Make sure the pseudo random numbers are the same every run.

  int tex_width  = 64;
  int tex_height = 64;
  int tex_depth  = 64;
  Buffer noiseBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, tex_width, tex_height, tex_depth);
  float *tex_data = (float *) noiseBuffer->map();
  
  if (m_parameters.proc == PROCEDURE_VORONOI)
  {
    // Distances to Voronoi control points (repeated version, taking the 26 surrounding cubes into account!)
    Voronoi_repeat(16, tex_width, tex_height, tex_depth, tex_data);
  }
  else
  {
    // Random noise in range [0, 1]
    for (int i = tex_width * tex_height * tex_depth;  i > 0; i--)
    {
      // One channel 3D noise in [0.0, 1.0] range.
      *tex_data++ = rand_range(0.0f, 1.0f);
    }
  }
  noiseBuffer->unmap(); 


  // Noise texture sampler
  TextureSampler noiseSampler = m_context->createTextureSampler();

  noiseSampler->setWrapMode(0, RT_WRAP_REPEAT);
  noiseSampler->setWrapMode(1, RT_WRAP_REPEAT);
  noiseSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
  noiseSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
  noiseSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
  noiseSampler->setMaxAnisotropy(1.0f);
  noiseSampler->setMipLevelCount(1);
  noiseSampler->setArraySize(1);
  noiseSampler->setBuffer(0, 0, noiseBuffer);

  m_context["noise_texture"]->setTextureSampler(noiseSampler);


  // 1D color ramp buffer, 4 float channels, all entries in the range [0.0, 1.0].
  Buffer colorRampBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, 1024);
  tex_data = (float *) colorRampBuffer->map();
  create_color_ramp(m_parameters.ramp, 1024, tex_data);
  colorRampBuffer->unmap();


  // Color ramp texture sampler
  TextureSampler colorRampSampler = m_context->createTextureSampler();

  // Prefer RT_CLAMP_TO_EDGE for non-cyclic patterns! (COLOR_RAMP_GREYSCALE, COLOR_RAMP_TEMPERATURE) 
  colorRampSampler->setWrapMode(0, RT_WRAP_REPEAT);
  colorRampSampler->setWrapMode(1, RT_WRAP_REPEAT);

  // checker's texture lookup has been adjusted to 0.0 and 0.5 to result in distinguishable colors.
  // 0.0 and 1.0 return the same color for repeated textures.
  colorRampSampler->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
  colorRampSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
  colorRampSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
  colorRampSampler->setMaxAnisotropy(1.0f);
  colorRampSampler->setMipLevelCount(1);
  colorRampSampler->setArraySize(1);
  colorRampSampler->setBuffer(0, 0, colorRampBuffer);

  m_context["color_ramp_texture"]->setTextureSampler(colorRampSampler);

  // Load OBJ scene 
  GeometryGroup geometry_group = m_context->createGeometryGroup();
  ObjLoader loader( m_filename.c_str(), m_context, geometry_group, material); 
  loader.load();
  m_context[ "top_object" ]->set( geometry_group ); 
  m_context[ "top_shadower" ]->set( geometry_group ); 

  // Set up camera
  optix::Aabb aabb = loader.getSceneBBox();
  float max_dim = fmaxf(aabb.extent(0), aabb.extent(1)); // max of x, y components
  float3 eye = aabb.center();
  eye.x -= max_dim * 0.25f;  // left
  eye.y += max_dim;         // above
  eye.z += max_dim * 1.25f; // front
  camera_data = InitialCameraData(eye,                             // eye
                                  aabb.center(),                   // lookat
                                  make_float3( 0.0f, 1.0f, 0.0f ), // up
                                  45.0f);                          // vfov

  // Declare camera variables.  The values do not matter, they will be overwritten in trace.
  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

  // Material
  m_context["specular_exp"]->setFloat(20.0f);

  // Use keys to adjust:
  m_context["frequency"]->setFloat(m_parameters.frequency);    // e/E
  m_context["turbulence"]->setFloat(m_parameters.turbulence);  // t/T
  m_context["octaves"]->setInt(m_parameters.octaves);          // o/O
  m_context["lambda"]->setFloat(m_parameters.lambda);          // l/L
  m_context["omega"]->setFloat(m_parameters.omega);            // m/M
  m_context["origin"]->setFloat(m_parameters.origin[0],
                                m_parameters.origin[1],
                                m_parameters.origin[2]);

  m_context["diameter"]->setFloat(1.0f); // for solid sphere texture

  // Prepare to run 
  m_context->validate();
  m_context->compile();
}


void ObjScene::trace(const RayGenCameraData& camera_data)
{
  m_context["eye"]->setFloat(camera_data.eye);
  m_context["U"]->setFloat(camera_data.U);
  m_context["V"]->setFloat(camera_data.V);
  m_context["W"]->setFloat(camera_data.W);

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize(buffer_width, buffer_height);

  m_context->launch( 0, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height) );
}

Buffer ObjScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


//------------------------------------------------------------------------------
//
//  main driver
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -o  | --obj <obj_file>                     Specify .OBJ model to be rendered\n"
    << "  --proc <marble | wood | onion | noise_cubed | voronoi | sphere | checker>\n"
    << "  --ramp <marble_blue | marble_brown | wood | cracked | bloodstone | hue | temperature>\n"
    << "  --freq <float>\n"
    << "  --turb <float>\n"
    << "  --lambda <float>\n"
    << "  --oct <int>\n"
    << "  --omega <float>\n"
    << "  --origin <float> <float> <float>\n"
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "Decrease/Increase (default)\n"
    << "  e/E: Pattern frequency (1.0)\n"
    << "  t/T: Turbulence (1.0)\n"
    << "  o/O: Turbulence octaves (1)\n"
    << "  l/L: Turbulence lambda (0.0) affects pattern waviness when octaves > 1\n"
    << "  g/G: Turbulence omega (0.0) affects noise strength when octaves > 1\n"
    << "  x/X, y/Y, z/Z: (0.0, 0.0, 0.0) pans pattern origin\n"
    << "SPACE: Reset all the above values to their defaults\n"
    << std::endl;

  if ( doExit ) exit(1);
}


// Functor used to check if a given index is in [1, argc-1]
class CheckIndex 
{
public:
  CheckIndex(unsigned int argc, const std::string& argv0 ) : m_argc(argc), m_argv0(argv0)
  {}
  void operator()(unsigned int i )      
  { if (i <= 0 || i >= m_argc ) printUsageAndExit( m_argv0 ); }
private:
  const unsigned int m_argc;
  const std::string  m_argv0; 
};


void process_commandline(int argc, char **argv, std::string& obj_filename,
                                                std::string& procedure_name,
                                                ProceduralParameters& parameters )
{
  // First grab the obj file
  CheckIndex checkIndex( argc, argv[0] );

  // Grab the procedure name, if present, before any overrides.  Default to marble.
  procedure_name = "marble";                              //  Set marble as default
  parameters     = g_ParametersDefault[PROCEDURE_MARBLE]; //
  for (int i = 1; i < argc-1; i++) {
    std::string arg( argv[i] );
    if (arg == "--proc") {
      checkIndex(++i);
      procedure_name = argv[i];
      parameters     = g_ParametersDefault[ProcedureStringToEnum(procedure_name)];
    }
  }

  // Finally process the parameter overrides
  for (int i = 1; i < argc; i++)
  {
    std::string arg( argv[i] );
    if (arg == "--proc")
    {
      // We already processed this option above, so eat next token and continue
      checkIndex( ++i ); 
    }
    else if (arg == "--ramp")
    {
      checkIndex(++i);
      parameters.ramp = ColorRampStringToEnum(argv[i]);
    }
    else if (arg == "--freq")
    {
      checkIndex(++i);
      parameters.frequency = static_cast<float>( atof(argv[i]) );
    }
    else if (arg == "--turb")
    {
      checkIndex(++i);
      parameters.turbulence = static_cast<float>( atof(argv[i]) );
    }
    else if (arg == "--oct")
    {
      checkIndex(++i);
      parameters.octaves = atoi(argv[i]);
    }
    else if (arg == "--lambda")
    {
      checkIndex(++i);
      parameters.lambda = static_cast<float>( atof(argv[i]) );
    }
    else if (arg == "--omega")
    {
      checkIndex(++i);
      parameters.omega = static_cast<float>( atof(argv[i]) );
    }
    else if (arg == "--origin")
    {
      checkIndex(i+3);
      parameters.origin[0] = static_cast<float>( atof(argv[++i]) );
      parameters.origin[1] = static_cast<float>( atof(argv[++i]) );
      parameters.origin[2] = static_cast<float>( atof(argv[++i]) );
    }
    else if (arg == "--help" || arg == "-h")
    {
      printUsageAndExit( argv[0] );
    }
    else if (arg == "--obj" || arg == "-o")
    {
      checkIndex(++i);
      obj_filename = argv[i];
    }
    else 
    {
      std::cerr << "Unknown option specified: '" << arg << "'" << std::endl;
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );
}


int main(int argc, char** argv) 
{
  GLUTDisplay::init(argc, argv);

  std::string obj_filename;
  std::string procedure_name;
  ProceduralParameters params;
  process_commandline(argc, argv, obj_filename, procedure_name, params );
  if ( obj_filename.empty() ) {
    obj_filename = std::string( sutilSamplesDir() ) + "/procedural/knot.obj";
  }
  
  try {
    ObjScene scene( obj_filename, procedure_name, params );
    GLUTDisplay::run("Procedural Scene", &scene);
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(2);
  }

  return 0;
}
