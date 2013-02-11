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

#ifndef ELVIS_MODEL_CONVERSION_MODEL_CONVERSION_H
#define ELVIS_MODEL_CONVERSION_MODEL_CONVERSION_H

#include <boost/filesystem/path.hpp>
#include <vector>

namespace ElVis
{
    class IModelConverter
    {
        public:
            IModelConverter();
            virtual ~IModelConverter() = 0;

            virtual void SetInputFileName(const boost::filesystem::path& fileName) = 0;
            virtual unsigned int GetNumberOfVertices() const = 0;
            virtual void GetVertex(unsigned int id, double& x, double& y, double& z) const = 0;

            virtual unsigned int GetNumberOfEdges() const = 0;
            virtual void GetEdge(unsigned int id, unsigned int& vertex0Id, unsigned int& vertex1Id) const = 0;

            virtual unsigned int GetNumberOfQuadrilateralFaces() const = 0;
            virtual unsigned int GetNumberOfTriangularFaces() const = 0;
            virtual void GetTriangleFaceEdgeIds(unsigned int faceId, unsigned int* edgeIds) const = 0;
            virtual void GetQuadrilateralFaceEdgeIds(unsigned int faceId, unsigned int* edgeIds) const = 0;

           
            virtual unsigned int GetNumberOfHexahedra() const = 0;
            virtual void GetHexahedronFaceIds(unsigned int hexId, unsigned int* faceIds) const = 0;
            virtual void GetHexahedronDegree(unsigned int hexId, unsigned int* degrees) const = 0;

            virtual unsigned int GetNumberOfPrisms() const = 0;
            virtual void GetPrismQuadFaceIds(unsigned int prismId, unsigned int* faceIds) const = 0;
            virtual void GetPrismTriangleFaceIds(unsigned int prismId, unsigned int* faceIds) const = 0;
            virtual void GetPrismDegree(unsigned int prismId, unsigned int* degrees) const = 0;

            virtual double CalculateScalarValue(double x, double y, double z) const = 0;
            virtual double CalculateScalarValue(double x, double y, double z, unsigned int elementId) const = 0;

        private:
    };
}

#endif //ELVIS_MODEL_CONVERSION_MODEL_CONVERSION_H
