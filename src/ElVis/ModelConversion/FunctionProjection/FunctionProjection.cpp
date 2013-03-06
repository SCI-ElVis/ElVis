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


#include "FunctionProjection.h"

namespace ElVis
{
    FunctionProjection::FunctionProjection() :
        bcs(0),
        m_model(0)
    {
    }

    FunctionProjection::FunctionProjection(const boost::filesystem::path& fileName) :
        IModelConverter(),
        bcs(0),
        m_model(0)
    {
        SetInputFileName(fileName);
    }

    void FunctionProjection::SetInputFileName(const boost::filesystem::path& fileName)
    {
        boost::filesystem::path prefix = fileName.root_path();
        prefix /= fileName.stem();
        m_model = new ElVis::NektarPlusPlusExtension::NektarModel(prefix.string());
        //m_model->InitializeWithGeometry(fileName.string());
        //m_model->GetMesh()->ReadExpansions(fileName.string());
        //bcs = new SpatialDomains::BoundaryConditions(m_model->GetMesh().get());
        //bcs->Read(fileName.string());

        for(unsigned int i = 0; i < GetNumberOfVertices(); ++i)
        {
            double x, y, z;
            GetVertex(i, x, y, z);
            m_allPoints.push_back(ElVis::WorldPoint(x,y,z));
        }
    }

    unsigned int FunctionProjection::GetNumberOfVertices() const
    {
        return m_model->GetMesh()->GetNvertices();
    }

    void FunctionProjection::GetVertex(unsigned int id, double& x, double& y, double& z) const
    {
        Nektar::SpatialDomains::VertexComponentSharedPtr vertex = 
            m_model->GetMesh()->GetVertex(id);

        x = vertex->x();
        y = vertex->y();
        z = vertex->z();
    }

    unsigned int FunctionProjection::GetNumberOfEdges() const
    {
        return m_model->GetMesh()->GetAllSegGeoms().size();
    }

    void FunctionProjection::GetEdge(unsigned int id, unsigned int& vertex0Id, unsigned int& vertex1Id) const
    {
        Nektar::SpatialDomains::SegGeomSharedPtr edge = m_model->GetMesh()->GetEdge(id);
        vertex0Id = edge->GetVid(0);
        vertex1Id = edge->GetVid(1);
    }

    unsigned int FunctionProjection::GetNumberOfTriangularFaces() const
    {
        return m_model->GetMesh()->GetAllTriGeoms().size();
    }

    unsigned int FunctionProjection::GetNumberOfQuadrilateralFaces() const
    {
        return m_model->GetMesh()->GetAllQuadGeoms().size();
    }

    void FunctionProjection::GetTriangleFaceEdgeIds(unsigned int faceId, unsigned int* edgeIds) const
    {
        //FaceMap::left_const_iterator found = m_triangularFaceMap.left.find(faceId);
        //const OriginalNektar::Face& face = (*found).second;
        //for(int i = 0; i < 3; ++i)
        //{
        //    edgeIds[i] = face.EdgeId(i);
        //}
    }

    void FunctionProjection::GetQuadrilateralFaceEdgeIds(unsigned int faceId, unsigned int* edgeIds) const
    {
        BOOST_AUTO(mesh, m_model->GetMesh());
        BOOST_AUTO(iter, mesh->GetAllQuadGeoms().find(faceId));
        if( iter != mesh->GetAllQuadGeoms().end() )
        {
            BOOST_AUTO(quad, (*iter).second);
            for(unsigned int i = 0; i < 4; ++i)
            {
                edgeIds[i] = quad->GetEid(i);
            }
        }
    }

    unsigned int FunctionProjection::GetNumberOfHexahedra() const
    {
        return m_model->GetMesh()->GetAllHexGeoms().size();
    }

    void FunctionProjection::GetHexahedronFaceIds(unsigned int hexId, unsigned int* faceIds) const
    {
        BOOST_AUTO(mesh, m_model->GetMesh());
        BOOST_AUTO(iter, mesh->GetAllHexGeoms().find(hexId));
        if( iter != mesh->GetAllHexGeoms().end() )
        {
            BOOST_AUTO(hex, (*iter).second);
            for(unsigned int i = 0; i < 6; ++i)
            {
                faceIds[i] = hex->GetFid(i);
            }
        }
    }

    void FunctionProjection::GetHexahedronDegree(unsigned int hexId, unsigned int* degrees) const
    {
        BOOST_AUTO(mesh, m_model->GetMesh());
        BOOST_AUTO(iter, mesh->GetAllHexGeoms().find(hexId));
        if( iter != mesh->GetAllHexGeoms().end() )
        {
            BOOST_AUTO(hex, (*iter).second);
            Nektar::SpatialDomains::ExpansionShPtr expansion = mesh->GetExpansion(hex);
            for(int i = 0; i < 3; ++i)
            {
                degrees[i] = expansion->m_basisKeyVector[i].GetNumModes()-1;
            }
        }
    }

    void FunctionProjection::GetPrismDegree(unsigned int prismId, unsigned int* degrees) const
    {
        BOOST_AUTO(mesh, m_model->GetMesh());
        BOOST_AUTO(iter, mesh->GetAllPrismGeoms().find(prismId));
        if( iter != mesh->GetAllPrismGeoms().end() )
        {
            BOOST_AUTO(prism, (*iter).second);
        
            Nektar::SpatialDomains::ExpansionShPtr expansion = mesh->GetExpansion(prism);
            for(int i = 0; i < 3; ++i)
            {
                degrees[i] = expansion->m_basisKeyVector[i].GetNumModes()-1;
            }
        }
    }

    unsigned int FunctionProjection::GetNumberOfPrisms() const
    {
        return m_model->GetMesh()->GetAllPrismGeoms().size();
    }

    void FunctionProjection::GetPrismQuadFaceIds(unsigned int prismId, unsigned int* faceIds) const
    {

    }

    void FunctionProjection::GetPrismTriangleFaceIds(unsigned int prismId, unsigned int* faceIds) const
    {
    }

    double FunctionProjection::CalculateScalarValue(double x, double y, double z) const
    {
        LibUtilities::EquationSharedPtr exac = m_model->GetSession()->GetFunction("ExactSolution", 0);
        return exac->Evaluate(x, y, z);
        //SpatialDomains::ConstExactSolutionShPtr ex_sol
        //                        = bcs->GetExactSolution(bcs->GetVariable(0));
        //return ex_sol->Evaluate(x, y, z);
    }

    double FunctionProjection::CalculateScalarValue(double x, double y, double z, unsigned int elementId) const
    {
        return CalculateScalarValue(x, y, z);
    }

}

ElVis::IModelConverter* CreateConverter()
{
    return new ElVis::FunctionProjection();
}

