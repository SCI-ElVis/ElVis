////////////////////////////////////////////////////////////////////////////////
//
//  File: hoHexahedron.h
//
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description:
//
//
////////////////////////////////////////////////////////////////////////////////

#ifndef ELVIS_JACOBI_EXTENSION___FINITE_ELEMENT_HEXAHEDRON_H__
#define ELVIS_JACOBI_EXTENSION___FINITE_ELEMENT_HEXAHEDRON_H__

#include <ElVis/Extensions/JacobiExtension/Polyhedra.h>
#include <ElVis/Extensions/JacobiExtension/Jacobi.hpp>

#include <boost/concept_check.hpp>
#include <assert.h>

namespace ElVis
{
    namespace JacobiExtension
    {
#define NUM_HEX_VERTICES 8
#define NUM_HEX_FACES 6

        /// \brief Implements a planar faced, high order hexahedron.
        ///
        /// For hexahedra, the interpolating polynomial is given by
        ///
        ///  N[i]
        /// ------- /N[j]  /N[k]
        /// \      |----- |-----                                                     ||
        ///  \     | \    | \                                                        ||
        ///   )    |  )   |  )   a[i, j, k] phi[i](xi[1]) phi[j](xi[2]) phi[k](xi[3])||
        ///  /     | /    | /                                                        ||
        /// /      |----- |-----                                                     ||
        /// ------- \j = 0 \k = 0
        /// i = 0
        ///
        /// Where phi_i is an ith degree polynomial, and all phi_i are orthoganal
        /// with respect to each other.
        ///
        /// For the Hexahedron we have chosen to use the Legendre Polynomials for phi.
        ///
        /// Note that when creating a hexahedron that the number of coefficients a that
        /// are needed is equal to (i+1)*(j+1)*(k+1), where i,j,k is the degree of the
        /// polynomial in the given direction.
        ///
        /// So if you wish to use second degree polynomials in all directions, then you
        /// will need (2+1)*(2+1)*(2+1) = 27 coefficients.
        class Hexahedron : public PolyhedronWithVertices<NUM_HEX_VERTICES>
        {
        public:
            static const int TypeId = 0;

            // Creates the hexahedron from the given input file following the
            // defined format.
            // points is the array which will be indexed from the file for each vertex.
            Hexahedron(FILE* in, bool reverseBytes=false);
            virtual ~Hexahedron();

            ///////////////////////////////////////////
            /// \name Point Transformation Routines
            /// These methods define point transformation between spaces.
            //{@
            ///////////////////////////////////////////

            /// \brief Transforms a point from Reference to Tensor space.
            virtual ElVis::TensorPoint transformReferenceToTensor(const ElVis::ReferencePoint& p);
            virtual ElVis::ReferencePoint transformTensorToReference(const ElVis::TensorPoint& p);
            virtual ElVis::WorldPoint transformReferenceToWorld(const ElVis::ReferencePoint& p);

            ElVis::TensorPoint transformWorldToTensorCartesianHex(const ElVis::WorldPoint& p);

            JACOBI_EXTENSION_EXPORT virtual unsigned int NumberOfCoefficientsForOrder(unsigned int order);

            //virtual IntervalTensorPoint transformReferenceToTensor(const IntervalReferencePoint& p);
            //virtual IntervalReferencePoint transformTensorToReference(const IntervalTensorPoint& p);
            //virtual IntervalWorldPoint transformReferenceToWorld(const IntervalReferencePoint& p);
            ///////////////////////////////////////////////
            //@}
            ///////////////////////////////////////////////

            //virtual bool findIntersectionWithGeometryInWorldSpace(const rt::Ray& ray, double& min, double& max);

            virtual void writeElementGeometryForVTK(const char* fileName);
            virtual void writeElementForVTKAsCell(const char*) {}
            virtual void outputVertexOrderForVTK(std::ofstream& outFile, int startPoint=0);
            virtual int vtkCellType();
            virtual int GetCellType() { return HEXAHEDRON; }
            virtual void calculateMinAndMax() const;

            ///////////////////////////////////////////
            /// \name Scalar Field Evaluation
            /// These methods are used to evaluate the scalar field in the element.
            //{@
            ///////////////////////////////////////////

            /// \brief Evaluates the scalar field at the given point.
            virtual double f(const ElVis::TensorPoint& p) const ;

            virtual double df_da(const ElVis::TensorPoint& p) const;
            virtual double df_db(const ElVis::TensorPoint& p) const;
            virtual double df_dc(const ElVis::TensorPoint& p) const;

            virtual double df2_da2(const ElVis::TensorPoint& p) const;
            virtual double df2_db2(const ElVis::TensorPoint& p) const;
            virtual double df2_dc2(const ElVis::TensorPoint& p) const;
            virtual double df2_dab(const ElVis::TensorPoint& p) const;
            virtual double df2_dac(const ElVis::TensorPoint& p) const;
            virtual double df2_dbc(const ElVis::TensorPoint& p) const;


            ///////////////////////////////////////////
            //@}
            ///////////////////////////////////////////

            // The functions are used for the numerical computations needed
            // to transform a point to reference space.
            //
            // This function calculates the jacobian of f (where f is the
            // transform from reference to world space).  (r,s,t) is the
            // point where we will evaluate the Jacobian, J is the output.
            // This code was largely generated by Maple (with some slight
            // hand tuning).
            virtual void getWorldToReferenceJacobian(const ElVis::ReferencePoint& p, JacobiExtension::Matrix<double, 3, 3>& J);

            virtual void calculateTensorToWorldSpaceMappingJacobian(const ElVis::TensorPoint& p, JacobiExtension::Matrix<double, 3, 3>& J);

            virtual void calculateTensorToWorldSpaceMappingHessian(const ElVis::TensorPoint& p, JacobiExtension::Matrix<double, 3, 3>& Hx,
                JacobiExtension::Matrix<double, 3, 3>& Hy, JacobiExtension::Matrix<double, 3, 3>& Hz);

            /// \brief Gets the normal for the face.
            /// Note that the normal is facing inwards.
            const ElVis::WorldVector& GetFaceNormal(int i) const
            {
                return faceNormals[i];
            }

            template<typename T>
            void GetFace(int i, T& a, T& b, T& c, T& d) const
            {
                a = (T)faceNormals[i].x();
                b = (T)faceNormals[i].y();
                c = (T)faceNormals[i].z();
                d = (T)D[i];
            }

            static const unsigned int NumFaces = NUM_HEX_FACES;
            static const unsigned int VerticesForEachFace[];
            static const unsigned int NumEdgesForEachFace[];

            static const Edge Edges[];
            static const Face Faces[];

        protected:

            void calculateFaceNormals();

            // Holds the normals for each face of the hex.
            // This is needed to help us intersect the plane that
            // each face belongs to without calculating it each time.
            ElVis::WorldVector faceNormals[NUM_HEX_FACES];

            // For each normal, the D which will give us the plane
            // equation.
            double D[NUM_HEX_FACES];

            //void intersectsFacePlane(const rt::Ray& ray, int face, double& min, double& max);

            // Must be a pointer since the constructor requires information
            // we don't have until near the end of the constructor.
            //HexahedralJacobiExpansion<double>* m_scalarField;
            //HexahedralJacobiExpansion<boost::numeric::interval<double> >* m_intervalScalarField;

        private:
            virtual double do_maxScalarValue() const;
            virtual double do_minScalarValue() const;
            virtual unsigned int DoNumberOfEdges() const;
            virtual Edge DoGetEdge(unsigned int id) const;
            virtual unsigned int DoNumberOfFaces() const;
            virtual Face DoGetFace(unsigned int id) const;

            double performSummation(const std::vector<double>& v1,
                const std::vector<double>& v2, const std::vector<double>& v3) const;
        };




    }
}

#endif
