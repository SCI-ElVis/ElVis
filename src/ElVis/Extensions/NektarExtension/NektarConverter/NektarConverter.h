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

#ifndef ELVIS_PLUGINS_ISOSURFACE_CONVERTER_H
#define ELVIS_PLUGINS_ISOSURFACE_CONVERTER_H

#include <ElVisModelConversion/IElVisModelConverter.h>
#include <boost/filesystem.hpp>
#include <map>
#include "NektarConverterDeclspec.h"
#include <boost/bimap.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/bimap.hpp>
#include <Nektar/Point.hpp>
#include <nektar.h>

namespace ElVis
{
    class NektarConverter : public IModelConverter
    {
        public:
            NEKTAR_CONVERTER_EXPORT NektarConverter();
            NEKTAR_CONVERTER_EXPORT explicit NektarConverter(const boost::filesystem::path& fileName);

            NEKTAR_CONVERTER_EXPORT virtual void SetInputFileName(const boost::filesystem::path& fileName);
            NEKTAR_CONVERTER_EXPORT virtual unsigned int GetNumberOfVertices() const;
            NEKTAR_CONVERTER_EXPORT virtual void GetVertex(unsigned int id, double& x, double& y, double& z) const;

            NEKTAR_CONVERTER_EXPORT virtual unsigned int GetNumberOfEdges() const;
            NEKTAR_CONVERTER_EXPORT virtual void GetEdge(unsigned int id, unsigned int& vertex0Id, unsigned int& vertex1Id) const;

            NEKTAR_CONVERTER_EXPORT virtual unsigned int GetNumberOfTriangularFaces() const;
            NEKTAR_CONVERTER_EXPORT virtual void GetTriangleFaceEdgeIds(unsigned int faceId, unsigned int* edgeIds) const;
            NEKTAR_CONVERTER_EXPORT virtual unsigned int GetNumberOfQuadrilateralFaces() const;
            NEKTAR_CONVERTER_EXPORT virtual void GetQuadrilateralFaceEdgeIds(unsigned int faceId, unsigned int* edgeIds) const;

            NEKTAR_CONVERTER_EXPORT virtual unsigned int GetNumberOfHexahedra() const;
            NEKTAR_CONVERTER_EXPORT virtual void GetHexahedronFaceIds(unsigned int hexId, unsigned int* faceIds) const;
            NEKTAR_CONVERTER_EXPORT virtual void GetHexahedronDegree(unsigned int hexId, unsigned int* degrees) const;

            NEKTAR_CONVERTER_EXPORT virtual unsigned int GetNumberOfPrisms() const;
            NEKTAR_CONVERTER_EXPORT virtual void GetPrismQuadFaceIds(unsigned int prismId, unsigned int* faceIds) const;
            NEKTAR_CONVERTER_EXPORT virtual void GetPrismTriangleFaceIds(unsigned int prismId, unsigned int* faceIds) const;
            NEKTAR_CONVERTER_EXPORT virtual void GetPrismDegree(unsigned int prismId, unsigned int* degrees) const;

            NEKTAR_CONVERTER_EXPORT virtual double CalculateScalarValue(double x, double y, double z) const;
            NEKTAR_CONVERTER_EXPORT virtual double CalculateScalarValue(double x, double y, double z, unsigned int elementId) const;
            
        private:
            NektarConverter(const NektarConverter& rhs);
            NektarConverter& operator=(const NektarConverter& rhs);

            void IntializeVertices();
            
            template<typename BimapType, typename ObjectType>
            void InsertIntoIdList(const ObjectType& t, BimapType& map)
            {
                if( map.right.find(t) == map.right.end() )
                {
                    unsigned int id = map.size();
                    map.insert(typename BimapType::value_type(id, t));
                }
            }

                    
            Element_List* m_mesh;
            
            typedef boost::bimap<unsigned int, OriginalNektar::WorldPoint> VertexMap;
            VertexMap m_vertexMap;
    };
}

extern "C"
{
    NEKTAR_CONVERTER_EXPORT ElVis::IModelConverter* CreateConverter();
}

#endif //ELVIS_PLUGINS_CONVER_ISOSURFACE_H

