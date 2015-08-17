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

#ifndef ELVISNATIVE_SURFACE_OBJECT_H
#define ELVISNATIVE_SURFACE_OBJECT_H

#include <ElVis/Core/ElVisDeclspec.h>
#include <ElVis/Core/PrimaryRayObject.h>
#include <ElVis/Core/Object.h>

#include <optixu/optixpp.h>
#include <optix_math.h>

#include <boost/shared_ptr.hpp>

namespace ElVis
{
  class SurfaceObject : public PrimaryRayObject
  {
  public:
    ELVIS_EXPORT SurfaceObject();
    ELVIS_EXPORT explicit SurfaceObject(boost::shared_ptr<Object> obj);
    ELVIS_EXPORT virtual ~SurfaceObject();

  protected:
    virtual optixu::Material GetMaterial(SceneView* view);

  private:
    SurfaceObject& operator=(const SurfaceObject& rhs);
    ELVIS_EXPORT SurfaceObject(const SurfaceObject& rhs);

    static optixu::Material Material;
  };
}

#endif
