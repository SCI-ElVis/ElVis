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


#include "NektarConverter.h"

namespace ElVis
{
    NektarConverter::NektarConverter() 
    {
    }

    NektarConverter::NektarConverter(const boost::filesystem::path& fileName) :
        IModelConverter()
    {
        SetInputFileName(fileName);
    }


    void NektarConverter::SetInputFileName(const boost::filesystem::path& fileName)
    {
        FILE* inFile = fopen(fileName.string().c_str(), "r");
        ReadParams(inFile);
        m_mesh = ReadMesh(inFile, "volume_name");
        fclose(inFile);
    }



    void NektarConverter::IntializeVertices()
    {
        // Iterate vertices
        for(unsigned int i = 0; i < m_mesh->nel; ++i)
        {
            Element* curElement = m_mesh->flist[i];
            
            for(int vertexId = 0; i < vertexId; ++i)
            {
                OriginalNektar::WorldPoint p(curElement->vert[i].x, curElement->vert[i].y, curElement->vert[i].z);
                InsertIntoIdList(p, m_vertexMap);
            }
        }
        // add them to the map.
    }
    
    unsigned int NektarConverter::GetNumberOfVertices() const
    {
        return 0;
    }

    void NektarConverter::GetVertex(unsigned int id, double& x, double& y, double& z) const
    {
    }

    unsigned int NektarConverter::GetNumberOfEdges() const
    {
        return 0;
    }

    void NektarConverter::GetEdge(unsigned int id, unsigned int& vertex0Id, unsigned int& vertex1Id) const
    {
        
    }

    unsigned int NektarConverter::GetNumberOfTriangularFaces() const
    {
        return 0;
    }

    unsigned int NektarConverter::GetNumberOfQuadrilateralFaces() const
    {
        return 0;
    }

    void NektarConverter::GetTriangleFaceEdgeIds(unsigned int faceId, unsigned int* edgeIds) const
    {
        
    }

    void NektarConverter::GetQuadrilateralFaceEdgeIds(unsigned int faceId, unsigned int* edgeIds) const
    {
        
    }

    unsigned int NektarConverter::GetNumberOfHexahedra() const
    {
        return 0;
    }

    void NektarConverter::GetHexahedronFaceIds(unsigned int hexId, unsigned int* faceIds) const
    {
    }

    void NektarConverter::GetHexahedronDegree(unsigned int hexId, unsigned int* degrees) const
    {
    }

    void NektarConverter::GetPrismDegree(unsigned int prismId, unsigned int* degrees) const
    {
    }

    unsigned int NektarConverter::GetNumberOfPrisms() const
    {
        return 0;
    }

    void NektarConverter::GetPrismQuadFaceIds(unsigned int prismId, unsigned int* faceIds) const
    {
        
    }

    void NektarConverter::GetPrismTriangleFaceIds(unsigned int prismId, unsigned int* faceIds) const
    {
        
    }

    double NektarConverter::CalculateScalarValue(double x, double y, double z) const
    {
        return 0.0;
    }

    double NektarConverter::CalculateScalarValue(double x, double y, double z, unsigned int elementId) const
    {
        return 0.0;
    }
}

ElVis::IModelConverter* CreateConverter()
{
    return new ElVis::NektarConverter();
}

