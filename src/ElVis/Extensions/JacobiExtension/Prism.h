////////////////////////////////////////////////////////////////////////////////
//
//  File: hoPrism.h
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

#ifndef ELVIS_JACOBI_EXTENSION__PRISM_H_
#define ELVIS_JACOBI_EXTENSION__PRISM_H_

#include <ElVis/Extensions/JacobiExtension/Polyhedra.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4267)
#endif

#include <boost/multi_array.hpp>

#ifdef _MSC_VER
#pragma warning( pop )
#endif

#include <ElVis/Extensions/JacobiExtension/Jacobi.hpp>

namespace ElVis
{
    namespace JacobiExtension
    {
#define NUM_PRISM_VERTICES 6
#define NUM_PRISM_FACES 5

        // Prism Object.
        class Prism : public PolyhedronWithVertices<NUM_PRISM_VERTICES>
        {
        public:
            static const int TypeId = 1;

            Prism(FILE* in, bool reverseBytes=false);
            virtual ~Prism();

            virtual ElVis::TensorPoint transformReferenceToTensor(const ElVis::ReferencePoint& p);
            virtual ElVis::ReferencePoint transformTensorToReference(const ElVis::TensorPoint& p);
            virtual ElVis::WorldPoint transformReferenceToWorld(const ElVis::ReferencePoint& p);

            //virtual bool findIntersectionWithGeometryInWorldSpace(const rt::Ray& ray, double& min, double& max);

            // Writes the geometry in vtk format.
            virtual void writeElementGeometryForVTK(const char* fileName);
            virtual void writeElementForVTKAsCell(const char* fileName);
            virtual void outputVertexOrderForVTK(std::ofstream& outFile,int startPoint=0);
            virtual int vtkCellType();
            virtual int GetCellType() { return PRISM; }
            JACOBI_EXTENSION_EXPORT virtual unsigned int NumberOfCoefficientsForOrder(unsigned int order);

            ///////////////////////////////////////////
            /// \name Scalar Field Evaluation
            /// These methods are used to evaluate the scalar field in the element.
            //{@
            ///////////////////////////////////////////

            /// \brief Evaluates the scalar field at the given point.
            virtual double f(const ElVis::TensorPoint& p) const;

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

            static const unsigned int NumFaces = NUM_PRISM_FACES;
            static const unsigned int VerticesForEachFace[];
            static const unsigned int NumEdgesForEachFace[];
            static const Edge Edges[];
            static const Face Faces[];

        protected:
            // The functions are used for the numerical computations needed
            // to transform a point to reference space.
            //
            // This function calculates the jacobian of f (where f is the
            // transform from reference to world space).  (r,s,t) is the
            // point where we will evaluate the Jacobian, J is the output.
            // This code was largely generated by Maple (with some slight
            // hand tuning).
            void getWorldToReferenceJacobian(const ElVis::ReferencePoint& p, JacobiExtension::Matrix<double, 3, 3>& J);

            void calculateInverseJacobian(double r, double s, double t, double* inverse);

            void calculateFaceNormals();

            // Holds the normals for each face of the hex.
            // This is needed to help us intersect the plane that
            // each face belongs to without calculating it each time.
            ElVis::WorldVector faceNormals[NUM_PRISM_FACES];

            // For each normal, the D which will give us the plane
            // equation.
            double D[NUM_PRISM_FACES];

            //void intersectsFacePlane(const rt::Ray& ray, int face, double& min, double& max);

        private:
            virtual double do_maxScalarValue() const;
            virtual double do_minScalarValue() const;
            virtual unsigned int DoNumberOfEdges() const;
            virtual Edge DoGetEdge(unsigned int id) const;
            virtual unsigned int DoNumberOfFaces() const;
            virtual Face DoGetFace(unsigned int id) const;

            double performSummation(const std::vector<double>& v1,
                const std::vector<double>& v2, const boost::multi_array<double, 2>& v3) const;
        };


    }
}

#endif
