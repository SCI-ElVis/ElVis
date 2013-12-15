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
        m_faceIdBuffer("FaceInfoBuffer"),
        m_PlanarFaceToGlobalIdxMap("PlanarFaceToGlobalIdxMap"),
        m_CurvedFaceToGlobalIdxMap("CurvedFaceToGlobalIdxMap"),
        m_PlanarFaceInfoBuffer("PlanarFaceInfoBuffer"),
        m_VertexBuffer("VertexBuffer"),
        m_PlanarFaceNormalBuffer("PlanarFaceNormalBuffer")
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

    WorldVector Model::GetPlanarFaceNormal(size_t localFaceId) const
    {
      return DoGetPlanarFaceNormal(localFaceId);
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
      BOOST_AUTO(mappedFaceInfoBuffer, m_faceIdBuffer.Map());

      for(size_t i = 0; i < numFaces; ++i)
      {
        BOOST_AUTO(faceDef, GetFaceDefinition(i));
        if( faceDef.Type == ePlanar ) ++numPlanarFaces;
        mappedFaceInfoBuffer[i] = faceDef;
      }
    }

    void Model::copyPlanarFaceVerticesToOptiX(optixu::Context context)
    {
      BOOST_AUTO(numPlanarVertices, GetNumberOfPlanarFaceVertices());
      std::cout << "############ Num planar face vertices: " << numPlanarVertices << std::endl;

      //// Populate the linear fields that we will handle.
      m_VertexBuffer.SetContext(context);
      m_VertexBuffer.SetDimensions(numPlanarVertices);
      BOOST_AUTO(mappedPlanarVertices, m_VertexBuffer.Map());

      for(size_t i = 0; i < numPlanarVertices; ++i)
      {
        mappedPlanarVertices[i] = MakeFloat4(GetPlanarFaceVertex(i));
      }
    }

    void Model::createLinearFaceGeometry(optixu::Context context)
    {
    }

    void Model::copyPlanarNormalsToOptiX(optixu::Context context, size_t numPlanarFaces)
    {
      m_PlanarFaceNormalBuffer.SetContext(context);
      m_PlanarFaceNormalBuffer.SetDimensions(numPlanarFaces);
      BOOST_AUTO(mappedPlanarFaceNormalBuffer, m_PlanarFaceNormalBuffer.Map());

      for(unsigned int i = 0; i < numPlanarFaces; ++i)
      {
        BOOST_AUTO(normal, GetPlanarFaceNormal(i));
        mappedPlanarFaceNormalBuffer[i] = MakeFloat4(normal);  
      }
    }

    void Model::copyPlanarFaces(optixu::Context context, size_t numPlanarFaces)
    {
      copyPlanarNormalsToOptiX(context, numPlanarFaces);

      copyPlanarFaceVerticesToOptiX(context);

      m_PlanarFaceToGlobalIdxMap.SetContext(context);
      m_PlanarFaceToGlobalIdxMap.SetDimensions(1);

      m_PlanarFaceInfoBuffer.SetContext(context);
      m_PlanarFaceInfoBuffer.SetDimensions(numPlanarFaces);

      BOOST_AUTO(numFaces, GetNumberOfFaces());
      BOOST_AUTO(faceBuffer, m_PlanarFaceInfoBuffer.Map());

      size_t localFaceIdx = 0;
      for(size_t globalFaceIdx = 0; globalFaceIdx < numFaces; ++globalFaceIdx)
      {
        BOOST_AUTO(faceDef, GetFaceDefinition(globalFaceIdx));
        if( faceDef.Type != ePlanar ) continue;

        PlanarFaceInfo info;
        BOOST_AUTO(numVertices, GetNumberOfVerticesForPlanarFace(globalFaceIdx));
        if( numVertices == 3 )
        {
          info.Type = eTriangle;
        }
        else
        {
          info.Type = eQuad;
        }

        for(size_t vertexIdx = 0; vertexIdx < numVertices; ++vertexIdx)
        {
          info.vertexIdx[vertexIdx] = DoGetPlanarFaceVertexIndex(globalFaceIdx, vertexIdx);
        }
        if( info.Type == eTriangle )
        {
          info.vertexIdx[3] = info.vertexIdx[2];
        }
        if( localFaceIdx >= numPlanarFaces) 
        {
          throw std::runtime_error("Invalid planar face count.");
        }
        faceBuffer[localFaceIdx] = info;
        ++localFaceIdx;
      }


      // Planar faces need the following:
      //facesForTraversal->setBoundingBoxProgram(faceForTraversalBBProgram);
      //  facesForTraversal->setIntersectionProgram(faceForTraversalIntersectionProgram);
      //  optixu::GeometryGroup ElementTraversalGroup = m_context->createGeometryGroup();
      //  ElementTraversalGroup->setChildCount(1);

      //  optixu::GeometryInstance faceForTraversalInstance = m_context->createGeometryInstance();
      //  optixu::Material faceForTraversalMaterial = m_context->createMaterial();
      //  faceForTraversalMaterial->setClosestHitProgram(2, closestHit);
      //  faceForTraversalInstance->setMaterialCount(1);
      //  faceForTraversalInstance->setMaterial(0, faceForTraversalMaterial);
      //  faceForTraversalInstance->setGeometry(facesForTraversal);

    }
    
    void Model::copyCurvedFaces(optixu::Context context)
    {
      m_CurvedFaceToGlobalIdxMap.SetContext(context);
      m_CurvedFaceToGlobalIdxMap.SetDimensions(1);
    }

    void Model::CopyToOptiX(optixu::Context context)
    {
      size_t numPlanarFaces = 0;
      copyFaceDefsToOptiX(context, numPlanarFaces);

      std::cout << "Copy planar faces: "<< numPlanarFaces << std::endl;
      copyPlanarFaces(context, numPlanarFaces);
      std::cout << "Copy field info" << std::endl;
      CopyFieldInfoToOptiX(context);
      //copyCurvedFaces(context);
      // CopyPlanarFaces(context);
      // CopyCurvedFaces(context);
      // CopyFaceAdjacency(context);
      // CopyFields(context);

      
      //copyPlanarFaceVerticesToOptiX(context);
      //createLinearFaceGeometry(context);
      // Populate custom faces.

      // Populate fields.
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
    FaceInfo Model::GetFaceDefinition(size_t globalFaceId) const
    {
      FaceInfo result = DoGetFaceDefinition(globalFaceId);
      result.widenExtents();
      return result;
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
    size_t Model::GetPlanarFaceVertexIndex(size_t globalFaceId, size_t vertexId)
    {
      return DoGetPlanarFaceVertexIndex(globalFaceId, vertexId);
    }

    void Model::CopyFieldInfoToOptiX(optixu::Context context)
    {
      DoCopyFieldInfoToOptiX(context);
    }
}
