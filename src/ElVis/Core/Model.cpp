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

#include "Model.h"
#include "Util.hpp"

namespace ElVis
{

    Model::Model() :
        m_minExtent(std::numeric_limits<ElVisFloat>::max(),std::numeric_limits<ElVisFloat>::max(),std::numeric_limits<ElVisFloat>::max()),
        m_maxExtent(-std::numeric_limits<ElVisFloat>::max(), -std::numeric_limits<ElVisFloat>::max(), -std::numeric_limits<ElVisFloat>::max()),
        m_center()
    {
    }


    Model::~Model()
    {
    }

    void Model::CalculateExtents()
    {
        m_minExtent = WorldPoint(std::numeric_limits<ElVisFloat>::max(),std::numeric_limits<ElVisFloat>::max(),std::numeric_limits<ElVisFloat>::max());
        m_maxExtent = WorldPoint(-std::numeric_limits<ElVisFloat>::max(), -std::numeric_limits<ElVisFloat>::max(), -std::numeric_limits<ElVisFloat>::max());
        DoCalculateExtents(m_minExtent, m_maxExtent);
        for(unsigned int i = 0; i < m_center.dimension(); ++i)
        {
            m_center.SetValue(i,(m_maxExtent[i] + m_minExtent[i])/2.0);
        }
    }
    
    const WorldPoint& Model::GetMidpoint()
    {
        CalculateExtents();
        return m_center;
    }

    std::vector<optixu::GeometryGroup> Model::GetPointLocationGeometry(Scene* scene, optixu::Context context, CUmodule module)
    {
        return DoGetPointLocationGeometry(scene, context, module);
    }

    void Model::GetFaceGeometry(Scene* scene, optixu::Context context, CUmodule module, optixu::Geometry& faces)
    {
        return DoGetFaceGeometry(scene, context, module, faces);
    }

    int Model::GetNumberOfBoundarySurfaces() const
    {
        return DoGetNumberOfBoundarySurfaces();
    }

    void Model::GetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds)
    {
        DoGetBoundarySurface(surfaceIndex, name, faceIds);
    }

    std::vector<optixu::GeometryInstance> Model::Get2DPrimaryGeometry(Scene* scene, optixu::Context context, CUmodule module)
    {
        return DoGet2DPrimaryGeometry(scene, context, module);
    }

    int Model::GetModelDimension() const
    {
        return DoGetModelDimension();
    }
}
