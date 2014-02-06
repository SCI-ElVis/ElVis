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

#include <ElVis/Extensions/JacobiExtension/JacobiExtensionElVisModel.h>
#include <ElVis/Extensions/JacobiExtension/Hexahedron.h>
#include <ElVis/Extensions/JacobiExtension/Tetrahedron.h>
#include <ElVis/Extensions/JacobiExtension/Prism.h>
#include <ElVis/Extensions/JacobiExtension/JacobiExtensionPTXConfig.h>

#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/Scene.h>

#include <ElVis/Core/Util.hpp>

#include <list>

#include <boost/typeof/typeof.hpp>

#include <ElVis/Core/Float.h>

namespace ElVis
{
    namespace JacobiExtension
    {
        JacobiExtensionModel::JacobiExtensionModel(const std::string& modelPath) :
            Model(modelPath),
            m_volume(),
            m_numberOfCopies(1),
            m_numberOfModes(-1),
            HexCoefficientBufferIndices("HexCoefficientIndices"),
            PrismCoefficientBufferIndices("PrismCoefficientIndices"),
            HexCoefficientBuffer("HexCoefficients"),
            PrismCoefficientBuffer("PrismCoefficients"),
            HexPlaneBuffer("HexPlaneBuffer"),
            PrismPlaneBuffer("PrismPlaneBuffer")
        {
        }

        JacobiExtensionModel::~JacobiExtensionModel()
        {}

        const std::string& JacobiExtensionModel::DoGetPTXPrefix() const
        {
            return GetJacobiExtensionPTXPrefix();
        }

        void JacobiExtensionModel::writeCellVolumeForVTK(const char* fileName)
        {
            if( !m_volume ) return;
            m_volume->writeCellVolumeForVTK(fileName);
        }

        void JacobiExtensionModel::LoadVolume(const std::string& filePath)
        {
            m_volume = boost::shared_ptr<FiniteElementVolume>(new FiniteElementVolume(filePath.c_str()));
            m_volume->calcOverallMinMaxScalar();
            std::cout << "Min: " << m_volume->get_min() << std::endl;
            std::cout << "Max: " << m_volume->get_max() << std::endl;
            ElVis::WorldPoint minExtent;
            ElVis::WorldPoint maxExtent;
            m_volume->calcOverallBoundingBox(minExtent, maxExtent);
            std::cout << "Min Extent: " << minExtent << std::endl;
            std::cout << "Max Extent: " << maxExtent << std::endl;
            SetMinExtent(minExtent);
            SetMaxExtent(maxExtent);

            std::map<JacobiFaceKey, JacobiFace> faceLookupMap;
            PopulateFaces<Hexahedron>(m_volume, faceLookupMap);
            PopulateFaces<Prism>(m_volume, faceLookupMap);


            // Find all unique vertices.  The OptiX extension requires a unique list of vertices, 
            // which each face references via vertex index.
            std::set<WorldPoint, bool(*)(const WorldPoint&, const WorldPoint&)> verticesLookupMap(closePointLessThan);
            for(unsigned int i = 0; i < m_volume->numElements(); i++)
            {
                BOOST_AUTO(element, m_volume->getElement(i));
                for(unsigned int j = 0; j < element->numVertices(); ++j)
                {
                    BOOST_AUTO(vertex, element->vertex(j));
                    if( verticesLookupMap.find(vertex) == verticesLookupMap.end() )
                    {
                        verticesLookupMap.insert(vertex);
                    }
                }
            }

            // Put the unique vertices in a easily-indexed list for future use.  The list will be sorted, 
            // so we can easily find the vertex index with a binary search.
            std::copy(verticesLookupMap.begin(), verticesLookupMap.end(),
              std::back_inserter(m_vertices));

            // Update the indices for each vertex.
            for(std::map<JacobiFaceKey, JacobiFace>::iterator iter = faceLookupMap.begin();
                iter != faceLookupMap.end(); ++iter)
            {
              const JacobiFaceKey& key = (*iter).first;
              JacobiFace& value = (*iter).second;
              for(unsigned int i = 0; i < 4; ++i)
              {
                BOOST_AUTO(iter, std::find_if(m_vertices.begin(), m_vertices.end(), boost::bind(closePointEqual, _1, key.p[i])));
                value.planarInfo.vertexIdx[i] = std::distance(m_vertices.begin(), iter);
              }
            }

            for(std::map<JacobiFaceKey, JacobiFace>::const_iterator iter = faceLookupMap.begin();
                iter != faceLookupMap.end(); ++iter)
            {
              BOOST_AUTO(face, (*iter).second);
              face.info.widenExtents();
              m_faces.push_back(face);
            }
        }

        void JacobiExtensionModel::DoCalculateExtents(WorldPoint& min, WorldPoint& max)
        {
            m_volume->calcOverallBoundingBox(min, max);
        }

        unsigned int JacobiExtensionModel::DoGetNumberOfElements() const
        {
            return m_volume->numElements();
        }

        int JacobiExtensionModel::DoGetNumFields() const
        {
            return 1;
        }

        FieldInfo JacobiExtensionModel::DoGetFieldInfo(unsigned int index) const
        {
            FieldInfo info;
            info.Name = "Density";
            info.Id = 0;
            info.Shortcut = "";
            return info;
        }

        bool closePointLessThan(const WorldPoint& lhs, const WorldPoint& rhs)
        {
            double tol = .001;
            for(unsigned int i = 0; i < 3; ++i)
            {
                if( lhs[i] < (rhs[i]-tol) ) return true;
                if( lhs[i] > (rhs[i]+tol) ) return false;
            }
            return false;
        }

        void JacobiExtensionModel::DoCopyExtensionSpecificDataToOptiX(optixu::Context context)
        {
  
            CopyFieldsForElementType<Hexahedron>(m_volume, context, "Hex");
            CopyFieldsForElementType<Prism>(m_volume, context, "Prism");
        }

        std::vector<optixu::GeometryInstance> JacobiExtensionModel::DoGet2DPrimaryGeometry(boost::shared_ptr<Scene> scene, optixu::Context context)
        {
            return std::vector<optixu::GeometryInstance>();
        }

        optixu::Material JacobiExtensionModel::DoGet2DPrimaryGeometryMaterial(SceneView* view)
        {
            return optixu::Material();
        }

        template<>
        ElVis::OptiXBuffer<int>& JacobiExtensionModel::GetCoefficientIndexBuffer<Hexahedron>()
        {
            return HexCoefficientBufferIndices;
        }

        template<>
        ElVis::OptiXBuffer<int>& JacobiExtensionModel::GetCoefficientIndexBuffer<Prism>()
        {
            return PrismCoefficientBufferIndices;
        }

        template<>
        ElVis::OptiXBuffer<ElVisFloat>& JacobiExtensionModel::GetCoefficientBuffer<Hexahedron>()
        {
            return HexCoefficientBuffer;
        }

        template<>
        ElVis::OptiXBuffer<ElVisFloat>& JacobiExtensionModel::GetCoefficientBuffer<Prism>()
        {
            return PrismCoefficientBuffer;
        }

        template<>
        ElVis::OptiXBuffer<ElVisFloat4>& JacobiExtensionModel::GetPlaneBuffer<Hexahedron>()
        {
            return HexPlaneBuffer;
        }

        template<>
        ElVis::OptiXBuffer<ElVisFloat4>& JacobiExtensionModel::GetPlaneBuffer<Prism>()
        {
            return PrismPlaneBuffer;
        }

        int JacobiExtensionModel::DoGetNumberOfBoundarySurfaces() const
        {
            return 0;
        }

        void JacobiExtensionModel::DoGetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds)
        {
        }

        size_t JacobiExtensionModel::DoGetNumberOfFaces() const
        {
          return m_faces.size();
        }

        FaceInfo JacobiExtensionModel::DoGetFaceDefinition(size_t globalFaceId) const
        {
          return m_faces[globalFaceId].info;
        }

        size_t JacobiExtensionModel::DoGetNumberOfPlanarFaceVertices() const
        {
          return m_vertices.size();
        }

        WorldPoint JacobiExtensionModel::DoGetPlanarFaceVertex(size_t vertexIdx) const
        {
          return m_vertices[vertexIdx];
        }

        size_t JacobiExtensionModel::DoGetNumberOfVerticesForPlanarFace(size_t localFaceIdx) const
        {
          if( m_faces[localFaceIdx].planarInfo.Type == eTriangle )
          {
            return 3;
          }
          else
          {
            return 4;
          }
        }

        size_t JacobiExtensionModel::DoGetPlanarFaceVertexIndex(size_t localFaceIdx, size_t vertexId)
        {
          return m_faces[localFaceIdx].planarInfo.vertexIdx[vertexId];
        }

        WorldVector JacobiExtensionModel::DoGetPlanarFaceNormal(size_t localFaceId) const
        {
          return m_faces[localFaceId].normal;
        }

    }
}
