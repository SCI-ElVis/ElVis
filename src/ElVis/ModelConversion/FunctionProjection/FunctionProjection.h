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

#ifndef ELVIS_FUNCTION_PROJECTION_H
#define ELVIS_FUNCTION_PROJECTION_H

#include <ElVis/ModelConversion/Interface/IElVisModelConverter.h>

#include <map>

#include <boost/filesystem.hpp>
#include <boost/bimap.hpp>
#include <boost/shared_ptr.hpp>

#include "FunctionProjectionDeclspec.h"

#include <ElVis/Core/Point.hpp>

#include <ElVis/Extensions/NektarPlusPlusExtension/NektarModel.h>

#include <SpatialDomains/Conditions.h>

namespace ElVis
{
    // Projects a global function onto a Nektar++ mesh.  It expects a Nektar++ mesh file with analytic solution specified.
    class FunctionProjection : public IModelConverter
    {
        public:
            FUNCTION_PROJECTION_EXPORT FunctionProjection();
            FUNCTION_PROJECTION_EXPORT explicit FunctionProjection(const boost::filesystem::path& fileName);

            FUNCTION_PROJECTION_EXPORT virtual void SetInputFileName(const boost::filesystem::path& fileName);
            FUNCTION_PROJECTION_EXPORT virtual unsigned int GetNumberOfVertices() const;
            FUNCTION_PROJECTION_EXPORT virtual void GetVertex(unsigned int id, double& x, double& y, double& z) const;

            FUNCTION_PROJECTION_EXPORT virtual unsigned int GetNumberOfEdges() const;
            FUNCTION_PROJECTION_EXPORT virtual void GetEdge(unsigned int id, unsigned int& vertex0Id, unsigned int& vertex1Id) const;

            FUNCTION_PROJECTION_EXPORT virtual unsigned int GetNumberOfTriangularFaces() const;
            FUNCTION_PROJECTION_EXPORT virtual void GetTriangleFaceEdgeIds(unsigned int faceId, unsigned int* edgeIds) const;
            FUNCTION_PROJECTION_EXPORT virtual unsigned int GetNumberOfQuadrilateralFaces() const;
            FUNCTION_PROJECTION_EXPORT virtual void GetQuadrilateralFaceEdgeIds(unsigned int faceId, unsigned int* edgeIds) const;

            FUNCTION_PROJECTION_EXPORT virtual unsigned int GetNumberOfHexahedra() const;
            FUNCTION_PROJECTION_EXPORT virtual void GetHexahedronFaceIds(unsigned int hexId, unsigned int* faceIds) const;
            FUNCTION_PROJECTION_EXPORT virtual void GetHexahedronDegree(unsigned int hexId, unsigned int* degrees) const;

            FUNCTION_PROJECTION_EXPORT virtual unsigned int GetNumberOfPrisms() const;
            FUNCTION_PROJECTION_EXPORT virtual void GetPrismQuadFaceIds(unsigned int prismId, unsigned int* faceIds) const;
            FUNCTION_PROJECTION_EXPORT virtual void GetPrismTriangleFaceIds(unsigned int prismId, unsigned int* faceIds) const;
            FUNCTION_PROJECTION_EXPORT virtual void GetPrismDegree(unsigned int prismId, unsigned int* degrees) const;

            FUNCTION_PROJECTION_EXPORT virtual double CalculateScalarValue(double x, double y, double z) const;
            FUNCTION_PROJECTION_EXPORT virtual double CalculateScalarValue(double x, double y, double z, unsigned int elementId) const;

        protected:


        private:
            FunctionProjection(const FunctionProjection& rhs);
            FunctionProjection& operator=(const FunctionProjection& rhs);


            ElVis::NektarPlusPlusExtension::NektarModel* m_model;
            SpatialDomains::BoundaryConditions* bcs;
            std::vector<ElVis::WorldPoint> m_allPoints;
    };
}

extern "C"
{
    FUNCTION_PROJECTION_EXPORT ElVis::IModelConverter* CreateConverter();
}

#endif //ELVIS_FUNCTION_PROJECTION_H

