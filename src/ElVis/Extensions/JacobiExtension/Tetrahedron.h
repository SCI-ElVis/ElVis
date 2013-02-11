////////////////////////////////////////////////////////////////////////////////
//
//  File: hoTetrahedron.h
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

#ifndef ELVIS_JACOBI_EXTENSION__TETRAHEDRON_H_
#define ELVIS_JACOBI_EXTENSION__TETRAHEDRON_H_

#include <ElVis/Extensions/JacobiExtension/Polyhedra.h>
#include <ElVis/Extensions/JacobiExtension/Jacobi.hpp>
#include <assert.h>

namespace ElVis
{
    namespace JacobiExtension
    {
        //#define NUM_TET_VERTICES 4
        //#define NUM_TET_FACES 4
        ////#include "TetrahedralJacobiExpansion.hpp"

        //class Tetrahedron : public PolyhedronWithVertices<NUM_TET_VERTICES>
        //{
        //    public:
        //        Tetrahedron(FILE* in, bool reverseBytes=false);
        //        Tetrahedron(unsigned int degreeDir1, unsigned int degreeDir2,
        //            unsigned int degreeDir3, const Polyhedron::CoefficientListType& basisCoeffs,
        //            const ElVis::WorldPoint& p0, const ElVis::WorldPoint& p1,
        //            const ElVis::WorldPoint& p2, const ElVis::WorldPoint& p3);

        //        virtual ~Tetrahedron();

        //        //virtual ElVis::ReferencePoint transformWorldToReference(const ElVis::WorldPoint& p);
        //        virtual ElVis::TensorPoint transformReferenceToTensor(const ElVis::ReferencePoint& p);
        //        virtual ElVis::ReferencePoint transformTensorToReference(const ElVis::TensorPoint& p);
        //        virtual ElVis::WorldPoint transformReferenceToWorld(const ElVis::ReferencePoint& p);

        //        //virtual bool findIntersectionWithGeometryInWorldSpace(const rt::Ray& ray, double& min, double& max);

        //        virtual void writeElement(FILE* outFile);

        //        // Writes the geometry in vtk format.
        //        virtual void writeElementGeometryForVTK(const char* fileName);
        //        virtual void writeElementForVTKAsCell(const char* fileName);
        //        virtual void outputVertexOrderForVTK(std::ofstream& outFile,int startPoint=0);
        //        virtual int vtkCellType();

        //        ///////////////////////////////////////////
        //        /// \name Scalar Field Evaluation
        //        /// These methods are used to evaluate the scalar field in the element.
        //        //{@
        //        ///////////////////////////////////////////

        //        /// \brief Evaluates the scalar field at the given point.
        //        virtual double f(const ElVis::TensorPoint& p) const;

        //        virtual double df_da(const ElVis::TensorPoint& p) const;
        //        virtual double df_db(const ElVis::TensorPoint& p) const;
        //        virtual double df_dc(const ElVis::TensorPoint& p) const;

        //        virtual double df2_da2(const ElVis::TensorPoint& p) const;
        //        virtual double df2_db2(const ElVis::TensorPoint& p) const;
        //        virtual double df2_dc2(const ElVis::TensorPoint& p) const;
        //        virtual double df2_dab(const ElVis::TensorPoint& p) const;
        //        virtual double df2_dac(const ElVis::TensorPoint& p) const;
        //        virtual double df2_dbc(const ElVis::TensorPoint& p) const;

        //        ///////////////////////////////////////////
        //        //@}
        //        ///////////////////////////////////////////

        //        static unsigned int calcNumCoefficients(unsigned int degree1,
        //            unsigned int degree2, unsigned int degree3);

        //        virtual void calculateTensorToWorldSpaceMappingJacobian(const ElVis::TensorPoint& p, JacobiExtension::Matrix<double, 3, 3>& J);

        //        virtual void calculateTensorToWorldSpaceMappingHessian(const ElVis::TensorPoint& p, JacobiExtension::Matrix<double, 3, 3>& Hx,
        //            JacobiExtension::Matrix<double, 3, 3>& Hy, JacobiExtension::Matrix<double, 3, 3>& Hz);

        //    protected:
        //        // The functions are used for the numerical computations needed
        //        // to transform a point to reference space.
        //        //
        //        // This function calculates the jacobian of f (where f is the
        //        // transform from reference to world space).  (r,s,t) is the
        //        // point where we will evaluate the Jacobian, J is the output.
        //        // This code was largely generated by Maple (with some slight
        //        // hand tuning).
        //        void getWorldToReferenceJacobian(const ElVis::ReferencePoint& p, JacobiExtension::Matrix<double, 3, 3>& J);

        //        void calculateInverseJacobian(double r, double s, double t, double* inverse);

        //        void calculateFaceNormals();

        //        // Holds the normals for each face of the hex.
        //        // This is needed to help us intersect the plane that
        //        // each face belongs to without calculating it each time.
        //        ElVis::WorldVector faceNormals[NUM_TET_FACES];

        //        // For each normal, the D which will give us the plane
        //        // equation.
        //        double D[NUM_TET_FACES];

        //        //bool intersectsFacePlane(const rt::Ray& ray, int face, double& min, double& max);

        //    private:
        //        double firstComponent(unsigned int i, const double& val) const;
        //        double d_firstComponent(unsigned int i, const double& val) const;
        //        double dd_firstComponent(unsigned int i, const double& val) const;
        //        double secondComponent(unsigned int i, unsigned int j, const double& val) const;
        //        double d_secondComponent(unsigned int i, unsigned int j, const double& val) const;
        //        double dd_secondComponent(unsigned int i, unsigned int j, const double& val) const;
        //        double thirdComponent(unsigned int i, unsigned int j, unsigned int k, const double& val) const;
        //        double d_thirdComponent(unsigned int i, unsigned int j, unsigned int k, const double& val) const;
        //        double dd_thirdComponent(unsigned int i, unsigned int j, unsigned int k, const double& val) const;

        //        virtual double do_maxScalarValue() const;
        //        virtual double do_minScalarValue() const;

        //        Tetrahedron(const Tetrahedron& rhs);
        //        Tetrahedron& operator=(const Tetrahedron& rhs);
        //        void populateWorldToReferenceJacobian();
        //        JacobiExtension::Matrix<double, 3, 3> m_worldToReferenceJacobian;
        //};


    }
}

#endif
