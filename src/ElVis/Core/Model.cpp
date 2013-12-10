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
#include <boost/filesystem.hpp>

namespace ElVis
{

    Model::Model(const std::string& modelPath) :
        m_modelPath(modelPath),
        m_plugin(),
        m_minExtent(std::numeric_limits<ElVisFloat>::max(),std::numeric_limits<ElVisFloat>::max(),std::numeric_limits<ElVisFloat>::max()),
        m_maxExtent(-std::numeric_limits<ElVisFloat>::max(), -std::numeric_limits<ElVisFloat>::max(), -std::numeric_limits<ElVisFloat>::max()),
        m_center(),
        m_faceIdBuffer("FaceIdBuffer"),
        m_planarFaceVertexBuffer("LinearFaceVertexBuffer")
    {
    }


    Model::~Model()
    {
    }

    std::string Model::GetModelName() const
    {
        boost::filesystem::path path(m_modelPath);
        return path.filename().string();
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

    std::vector<optixu::GeometryGroup> Model::GetPointLocationGeometry(boost::shared_ptr<Scene> scene, optixu::Context context)
    {
        return DoGetPointLocationGeometry(scene, context);
    }

    void Model::GetFaceGeometry(boost::shared_ptr<Scene> scene, optixu::Context context, optixu::Geometry& faces)
    {
        return DoGetFaceGeometry(scene, context, faces);
    }

    int Model::GetNumberOfBoundarySurfaces() const
    {
        return DoGetNumberOfBoundarySurfaces();
    }

    void Model::GetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds)
    {
        DoGetBoundarySurface(surfaceIndex, name, faceIds);
    }

    std::vector<optixu::GeometryInstance> Model::Get2DPrimaryGeometry(boost::shared_ptr<Scene> scene, optixu::Context context)
    {
        return DoGet2DPrimaryGeometry(scene, context);
    }

    void Model::copyFaceDefsToOptiX(optixu::Context context, size_t& numPlanarFaces)
    {
      numPlanarFaces = 0;
      
      // Get information about faces and copy to OptiX.
      // Array of [0, numFaces] of faceDefs.
      BOOST_AUTO(numFaces, GetNumberOfFaces());

      // Populate the face id buffer.
      m_faceIdBuffer.SetContext(context);
      m_faceIdBuffer.SetDimensions(numFaces);
      BOOST_AUTO(mappedFaceIdBuffer, m_faceIdBuffer.Map());

      for(size_t i = 0; i < numFaces; ++i)
      {
        BOOST_AUTO(faceDef, GetFaceDefinition(i));
        if( faceDef.Type == ePlanar ) ++numPlanarFaces;
        mappedFaceIdBuffer[i] = faceDef;
      }
    }

    void Model::copyPlanarFaceVerticesToOptiX(optixu::Context context)
    {
      BOOST_AUTO(numPlanarVertices, GetNumberOfPlanarFaceVertices());

      // Populate the linear fields that we will handle.
      m_planarFaceVertexBuffer.SetContext(context);
      m_planarFaceVertexBuffer.SetDimensions(numPlanarVertices);
      BOOST_AUTO(mappedPlanarVertices, m_planarFaceVertexBuffer.Map());

      for(size_t i = 0; i < numPlanarVertices; ++i)
      {
        mappedPlanarVertices[i] = GetPlanarFaceVertex(i);
      }
    }

    void Model::CopyToOptiX(optixu::Context context)
    {
      size_t numPlanarFaces = 0;
      copyFaceDefsToOptiX(context, numPlanarFaces);
      copyPlanarFaceVerticesToOptiX(context);
      // Populate custom faces.

      // Populate fields.
    }

    void Model::InitializeOptiX(boost::shared_ptr<Scene> scene, optixu::Context context)
    {
      DoInitializeOptiX(scene, context);
    }

    int Model::GetModelDimension() const
    {
        return DoGetModelDimension();
    }

    size_t Model::GetNumberOfFaces() const
    {
      return DoGetNumberOfFaces();
    }

    /// \brief Returns the given face definition.
    FaceDef Model::GetFaceDefinition(size_t globalFaceId) const
    {
      return DoGetFaceDefinition(globalFaceId);
    }

    /// \brief Returns the number of vertices associated with the linear
    /// faces.
    ///
    /// This method returns the total number of vertices associated with
    /// linear faces.  Vertices shared among faces are counted only once.
    size_t Model::GetNumberOfPlanarFaceVertices() const
    {
      return DoGetNumberOfPlanarFaceVertices();
    }

    /// Get the vertex for a linear face.
    WorldPoint Model::GetPlanarFaceVertex(size_t vertexIdx) const
    {
      return DoGetPlanarFaceVertex(vertexIdx);
    }

    /// \brief Returns the number of vertices associated with a single linear face.
    size_t Model::GetNumberOfVerticesForPlanarFace(size_t globalFaceId) const
    {
      return DoGetNumberOfVerticesForPlanarFace(globalFaceId);
    }

    /// \brief Returns the vertex id in the range [0, DoGetNumberOfPlanarFaceVertices)
    size_t Model::GetFaceVertexIndex(size_t globalFaceId, size_t vertexId)
    {
      return DoGetFaceVertexIndex(globalFaceId, vertexId);
    }
}
