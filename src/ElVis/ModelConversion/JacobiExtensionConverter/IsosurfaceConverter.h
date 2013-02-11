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
#include <Nektar/FiniteElementVolume.h>
#include <map>
#include "IsosurfaceConverterDeclspec.h"
#include <boost/bimap.hpp>
#include <boost/shared_ptr.hpp>
#include <Nektar/FiniteElementVolume.h>
#include <Nektar/Hexahedron.h>
#include <Nektar/Prism.h>

namespace ElVis
{
    class IsosurfaceConverter : public IModelConverter
    {
        public:
            ISOSURFACE_CONVERTER_EXPORT IsosurfaceConverter();
            ISOSURFACE_CONVERTER_EXPORT explicit IsosurfaceConverter(const boost::filesystem::path& fileName);

            ISOSURFACE_CONVERTER_EXPORT virtual void SetInputFileName(const boost::filesystem::path& fileName);
            ISOSURFACE_CONVERTER_EXPORT virtual unsigned int GetNumberOfVertices() const;
            ISOSURFACE_CONVERTER_EXPORT virtual void GetVertex(unsigned int id, double& x, double& y, double& z) const;

            ISOSURFACE_CONVERTER_EXPORT virtual unsigned int GetNumberOfEdges() const;
            ISOSURFACE_CONVERTER_EXPORT virtual void GetEdge(unsigned int id, unsigned int& vertex0Id, unsigned int& vertex1Id) const;

            ISOSURFACE_CONVERTER_EXPORT virtual unsigned int GetNumberOfTriangularFaces() const;
            ISOSURFACE_CONVERTER_EXPORT virtual void GetTriangleFaceEdgeIds(unsigned int faceId, unsigned int* edgeIds) const;
            ISOSURFACE_CONVERTER_EXPORT virtual unsigned int GetNumberOfQuadrilateralFaces() const;
            ISOSURFACE_CONVERTER_EXPORT virtual void GetQuadrilateralFaceEdgeIds(unsigned int faceId, unsigned int* edgeIds) const;

            ISOSURFACE_CONVERTER_EXPORT virtual unsigned int GetNumberOfHexahedra() const;
            ISOSURFACE_CONVERTER_EXPORT virtual void GetHexahedronFaceIds(unsigned int hexId, unsigned int* faceIds) const;
            ISOSURFACE_CONVERTER_EXPORT virtual void GetHexahedronDegree(unsigned int hexId, unsigned int* degrees) const;

            ISOSURFACE_CONVERTER_EXPORT virtual unsigned int GetNumberOfPrisms() const;
            ISOSURFACE_CONVERTER_EXPORT virtual void GetPrismQuadFaceIds(unsigned int prismId, unsigned int* faceIds) const;
            ISOSURFACE_CONVERTER_EXPORT virtual void GetPrismTriangleFaceIds(unsigned int prismId, unsigned int* faceIds) const;
            ISOSURFACE_CONVERTER_EXPORT virtual void GetPrismDegree(unsigned int prismId, unsigned int* degrees) const;

            ISOSURFACE_CONVERTER_EXPORT virtual double CalculateScalarValue(double x, double y, double z) const;
            ISOSURFACE_CONVERTER_EXPORT virtual double CalculateScalarValue(double x, double y, double z, unsigned int elementId) const;
        private:
            IsosurfaceConverter(const IsosurfaceConverter& rhs);
            IsosurfaceConverter& operator=(const IsosurfaceConverter& rhs);

            void InitializeVertices();
            void InitializeEdges();
            void InitializeFaces();
            void InitializeElements();

            boost::shared_ptr<OriginalNektar::FiniteElementVolume> m_volume;
            
            OriginalNektar::Edge GetGlobalEdge(boost::shared_ptr<OriginalNektar::Polyhedron> poly, unsigned int edgeId);
            OriginalNektar::Face GetGlobalFace(boost::shared_ptr<OriginalNektar::Polyhedron> poly, unsigned int faceId);
            unsigned int GetGlobalFaceId(boost::shared_ptr<OriginalNektar::Polyhedron> poly, unsigned int faceId);

            typedef boost::bimap<unsigned int, OriginalNektar::WorldPoint> VertexMap;
            VertexMap m_vertexMap;


            typedef boost::bimap<unsigned int, OriginalNektar::Edge> EdgeMap;
            EdgeMap m_edgeMap;

            typedef boost::bimap<unsigned int, OriginalNektar::Face> FaceMap;
            FaceMap m_triangularFaceMap;
            FaceMap m_quadFaceMap;
            
            template<typename BimapType, typename ObjectType>
            void InsertIntoIdList(const ObjectType& t, BimapType& map)
            {
                if( map.right.find(t) == map.right.end() )
                {
                    unsigned int id = map.size();
                    map.insert(typename BimapType::value_type(id, t));
                }
            }

            std::map<unsigned int, std::vector<unsigned int> > m_hexFaceIdMap;
            std::vector<boost::shared_ptr<OriginalNektar::Hexahedron> > m_allHexahedra;

            std::map<unsigned int, std::vector<unsigned int> > m_prismTriangularFaceIdMap;
            std::map<unsigned int, std::vector<unsigned int> > m_prismQuadFaceIdMap;
            std::vector<boost::shared_ptr<OriginalNektar::Prism> > m_allPrisms;

    };
}

extern "C"
{
    ISOSURFACE_CONVERTER_EXPORT ElVis::IModelConverter* CreateConverter();
}

#endif //ELVIS_PLUGINS_CONVER_ISOSURFACE_H

