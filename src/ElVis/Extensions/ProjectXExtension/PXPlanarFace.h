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

#ifndef PX_PLANAR_FACE_H
#define PX_PLANAR_FACE_H

#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/Vector.hpp>
#include <ElVis/Core/FaceInfo.h>
#include <algorithm>

namespace ElVis
{
  struct FaceNodeInfo
  {
  private:
    FaceNodeInfo() : nNode(0), vertexIdx(NULL) {}

  public:
    FaceNodeInfo(const int nNode) : nNode(nNode) { vertexIdx = new unsigned int[nNode]; }
    FaceNodeInfo(const FaceNodeInfo& Info) : nNode(Info.nNode), vertexIdx(NULL)
    {
      vertexIdx = new unsigned int[nNode];
      for(int i = 0; i < nNode; i++) vertexIdx[i] = Info.vertexIdx[i];
    }
    FaceNodeInfo& operator=(const FaceNodeInfo& Info)
    {
      nNode = Info.nNode;
      vertexIdx = new unsigned int[nNode];
      for(int i = 0; i < nNode; i++) vertexIdx[i] = Info.vertexIdx[i];
      return *this;
    }
    ~FaceNodeInfo() { delete [] vertexIdx; }

    int nNode;
    TwoDElementType Type;
    unsigned int *vertexIdx;
  };

  struct PXPlanarFace
  {
    explicit PXPlanarFace(const WorldVector& n, const FaceInfo& info, const FaceNodeInfo& planarInfo) :
              normal(n),
              info(info),
              planarInfo(planarInfo)
    {
    }

    PXPlanarFace(const PXPlanarFace& rhs) :
      normal(rhs.normal),
      info(rhs.info),
      planarInfo(rhs.planarInfo)
    {
    }

    PXPlanarFace& operator=(const PXPlanarFace& rhs)
    {
      normal = rhs.normal;
      info = rhs.info;
      planarInfo = rhs.planarInfo;
      return *this;
    }

    WorldVector normal;
    FaceInfo info;
    FaceNodeInfo planarInfo;
  };
}


#endif
