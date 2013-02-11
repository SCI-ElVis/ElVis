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


#include "IsosurfaceConverter.h"

namespace ElVis
{
    IsosurfaceConverter::IsosurfaceConverter() :
        m_volume()
    {
    }

    IsosurfaceConverter::IsosurfaceConverter(const boost::filesystem::path& fileName) :
        IModelConverter(),
        m_volume()
    {
        SetInputFileName(fileName);
    }


    void IsosurfaceConverter::SetInputFileName(const boost::filesystem::path& fileName)
    {
        m_volume = boost::shared_ptr<OriginalNektar::FiniteElementVolume>(
            new OriginalNektar::FiniteElementVolume(fileName.string().c_str()));

        InitializeVertices();
        InitializeEdges();
        InitializeFaces();
        InitializeElements();
    }

    void IsosurfaceConverter::InitializeVertices()
    {
        for(unsigned int i = 0; i < m_volume->numElements(); ++i)
        {
            boost::shared_ptr<OriginalNektar::Polyhedron> poly = m_volume->getElement(i);
            for(int j = 0; j < poly->numVertices(); ++j)
            {
                const OriginalNektar::WorldPoint& v = poly->vertex(j);
                InsertIntoIdList(v, m_vertexMap);
            }
        }
    }

    OriginalNektar::Edge IsosurfaceConverter::GetGlobalEdge(boost::shared_ptr<OriginalNektar::Polyhedron> poly, unsigned int edgeId)
    {
        OriginalNektar::Edge localEdge = poly->GetEdge(edgeId);
        const OriginalNektar::WorldPoint& v0 = poly->vertex(localEdge.GetVertex0());
        const OriginalNektar::WorldPoint& v1 = poly->vertex(localEdge.GetVertex1());

        unsigned int v0Id = (*m_vertexMap.right.find(v0)).second;
        unsigned int v1Id = (*m_vertexMap.right.find(v1)).second;
        OriginalNektar::Edge globalEdge(v0Id, v1Id);
        return globalEdge;
    }

    OriginalNektar::Face IsosurfaceConverter::GetGlobalFace(boost::shared_ptr<OriginalNektar::Polyhedron> poly, unsigned int faceId)
    {
        OriginalNektar::Face localFace = poly->GetFace(faceId);
        
        std::vector<unsigned int> globalEdgeIds;

        // Local face has the indices of local edges.  For each local edge, convert it to 
        // a global edge, then lookup the global index.
        for(unsigned int i = 0; i < localFace.NumberOfEdges(); ++i)
        {
            unsigned int edgeId = localFace.EdgeId(i);
            OriginalNektar::Edge globalEdge = GetGlobalEdge(poly, edgeId);
            globalEdgeIds.push_back(m_edgeMap.right.find(globalEdge)->second);
        }

        if( globalEdgeIds.size() == 3 )
        {
            return OriginalNektar::Face(globalEdgeIds[0], globalEdgeIds[1], globalEdgeIds[2]);
        }
        else
        {
            return OriginalNektar::Face(globalEdgeIds[0], globalEdgeIds[1], globalEdgeIds[2], globalEdgeIds[3]);
        }
    }

    unsigned int IsosurfaceConverter::GetGlobalFaceId(boost::shared_ptr<OriginalNektar::Polyhedron> poly, unsigned int faceId)
    {
        OriginalNektar::Face globalFace = GetGlobalFace(poly, faceId);
        if( globalFace.NumberOfEdges() == 3 )
        {
            return (*m_triangularFaceMap.right.find(globalFace)).second;
        }
        else if( globalFace.NumberOfEdges() == 4 )
        {
            return (*m_quadFaceMap.right.find(globalFace)).second;
        }
        return 0;
    }

    void IsosurfaceConverter::InitializeEdges()
    {
        for(unsigned int i = 0; i < m_volume->numElements(); ++i)
        {
            boost::shared_ptr<OriginalNektar::Polyhedron> poly = m_volume->getElement(i);
            for(unsigned int edgeId = 0; edgeId < poly->NumberOfEdges(); ++edgeId)
            {
                OriginalNektar::Edge globalEdge = GetGlobalEdge(poly, edgeId);
                InsertIntoIdList(globalEdge, m_edgeMap);
            }
        }
    }

    void IsosurfaceConverter::InitializeFaces()
    {
        for(unsigned int i = 0; i < m_volume->numElements(); ++i)
        {
            boost::shared_ptr<OriginalNektar::Polyhedron> poly = m_volume->getElement(i);

            for(unsigned int j = 0; j < poly->NumberOfFaces(); ++j)
            {
                OriginalNektar::Face globalFace = GetGlobalFace(poly, j);

                if( globalFace.NumberOfEdges() == 3 )
                {
                    InsertIntoIdList(globalFace, m_triangularFaceMap);
                }
                else if( globalFace.NumberOfEdges() == 4 )
                {
                    InsertIntoIdList(globalFace, m_quadFaceMap);                
                }
            }
        }
    }

    void IsosurfaceConverter::InitializeElements()
    {
        for(unsigned int i = 0; i < m_volume->numElements(); ++i)
        {
            boost::shared_ptr<OriginalNektar::Polyhedron> poly = m_volume->getElement(i);
            boost::shared_ptr<OriginalNektar::Hexahedron> hex = 
                boost::dynamic_pointer_cast<OriginalNektar::Hexahedron>(poly);
            boost::shared_ptr<OriginalNektar::Prism> prism = 
                boost::dynamic_pointer_cast<OriginalNektar::Prism>(poly);

            if( hex )
            {
                unsigned int hexId = m_allHexahedra.size();
                m_allHexahedra.push_back(hex);

                std::vector<unsigned int> faceIds;
                for(unsigned int j = 0; j < poly->NumberOfFaces(); ++j)
                {
                    faceIds.push_back(GetGlobalFaceId(poly, j));
                }
                m_hexFaceIdMap[hexId] = faceIds;
            }
            else if( prism )
            {
                unsigned int prismId = m_allPrisms.size();
                m_allPrisms.push_back(prism);

                std::vector<unsigned int> triFaceIds;
                std::vector<unsigned int> quadFaceIds;
                for(unsigned int j = 0; j < poly->NumberOfFaces(); ++j)
                {
                    if( poly->GetFace(j).NumberOfEdges() == 3 )
                    {
                        triFaceIds.push_back(GetGlobalFaceId(poly, j));    
                    }
                    else if( poly->GetFace(j).NumberOfEdges() == 4 )
                    {
                        quadFaceIds.push_back(GetGlobalFaceId(poly, j));    
                    }
                }

                m_prismTriangularFaceIdMap[prismId] = triFaceIds;
                m_prismQuadFaceIdMap[prismId] = quadFaceIds;
            }
        }
    }

    unsigned int IsosurfaceConverter::GetNumberOfVertices() const
    {
        return m_vertexMap.size();
    }

    void IsosurfaceConverter::GetVertex(unsigned int id, double& x, double& y, double& z) const
    {
        VertexMap::left_const_iterator found = m_vertexMap.left.find(id);
        const OriginalNektar::WorldPoint& v = (*found).second;
        x = v.x();
        y = v.y();
        z = v.z();
    }

    unsigned int IsosurfaceConverter::GetNumberOfEdges() const
    {
        return m_edgeMap.size();
    }

    void IsosurfaceConverter::GetEdge(unsigned int id, unsigned int& vertex0Id, unsigned int& vertex1Id) const
    {
        EdgeMap::left_const_iterator found = m_edgeMap.left.find(id);
        const OriginalNektar::Edge& e = (*found).second;
        vertex0Id = e.GetVertex0();
        vertex1Id = e.GetVertex1();
    }

    unsigned int IsosurfaceConverter::GetNumberOfTriangularFaces() const
    {
        return m_triangularFaceMap.size();
    }

    unsigned int IsosurfaceConverter::GetNumberOfQuadrilateralFaces() const
    {
        return m_quadFaceMap.size();
    }

    void IsosurfaceConverter::GetTriangleFaceEdgeIds(unsigned int faceId, unsigned int* edgeIds) const
    {
        FaceMap::left_const_iterator found = m_triangularFaceMap.left.find(faceId);
        const OriginalNektar::Face& face = (*found).second;
        for(int i = 0; i < 3; ++i)
        {
            edgeIds[i] = face.EdgeId(i);
        }
    }

    void IsosurfaceConverter::GetQuadrilateralFaceEdgeIds(unsigned int faceId, unsigned int* edgeIds) const
    {
        FaceMap::left_const_iterator found = m_quadFaceMap.left.find(faceId);
        const OriginalNektar::Face& face = (*found).second;
        for(int i = 0; i < 4; ++i)
        {
            edgeIds[i] = face.EdgeId(i);
        }
    }

    unsigned int IsosurfaceConverter::GetNumberOfHexahedra() const
    {
        return m_allHexahedra.size();
    }

    void IsosurfaceConverter::GetHexahedronFaceIds(unsigned int hexId, unsigned int* faceIds) const
    {
        const std::vector<unsigned int>& ids = (*m_hexFaceIdMap.find(hexId)).second;

        for(unsigned int i = 0; i < 6; ++i)
        {
            faceIds[i] = ids[i];
        }
    }

    void IsosurfaceConverter::GetHexahedronDegree(unsigned int hexId, unsigned int* degrees) const
    {
        degrees[0] = m_allHexahedra[hexId]->degree(0);
        degrees[1] = m_allHexahedra[hexId]->degree(1);
        degrees[2] = m_allHexahedra[hexId]->degree(2);
    }

    void IsosurfaceConverter::GetPrismDegree(unsigned int prismId, unsigned int* degrees) const
    {
        degrees[0] = m_allPrisms[prismId]->degree(0);
        degrees[1] = m_allPrisms[prismId]->degree(1);
        degrees[2] = m_allPrisms[prismId]->degree(2);
    }

    unsigned int IsosurfaceConverter::GetNumberOfPrisms() const
    {
        return m_allPrisms.size();
    }

    void IsosurfaceConverter::GetPrismQuadFaceIds(unsigned int prismId, unsigned int* faceIds) const
    {
        const std::vector<unsigned int>& ids = (*m_prismQuadFaceIdMap.find(prismId)).second;

        for(unsigned int i = 0; i < ids.size(); ++i)
        {
            faceIds[i] = ids[i];
        }
    }

    void IsosurfaceConverter::GetPrismTriangleFaceIds(unsigned int prismId, unsigned int* faceIds) const
    {
        const std::vector<unsigned int>& ids = (*m_prismTriangularFaceIdMap.find(prismId)).second;

        for(unsigned int i = 0; i < ids.size(); ++i)
        {
            faceIds[i] = ids[i];
        }
    }

    double IsosurfaceConverter::CalculateScalarValue(double x, double y, double z) const
    {
        OriginalNektar::WorldPoint p(x, y, z);
        return m_volume->calculateScalarValue(p);
    }

    double IsosurfaceConverter::CalculateScalarValue(double x, double y, double z, unsigned int elementId) const
    {
        boost::shared_ptr<OriginalNektar::Polyhedron> poly = m_volume->getElement(elementId);
        OriginalNektar::WorldPoint p(x, y, z);
        OriginalNektar::TensorPoint tp = poly->transformWorldToTensor(p);
        if( tp.a() >= -1.0001 && tp.a() <= 1.0001 &&
            tp.b() >= -1.0001 && tp.b() <= 1.0001 &&
            tp.c() >= -1.0001 && tp.c() <= 1.0001 )
        {
            return poly->f(tp);
        }
        return CalculateScalarValue(x, y, z);
    }
}

ElVis::IModelConverter* CreateConverter()
{
    return new ElVis::IsosurfaceConverter();
}

