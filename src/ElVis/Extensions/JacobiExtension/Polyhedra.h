////////////////////////////////////////////////////////////////////////////////
//
//  File: hoPolyhedra.h
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
//  Abstract base class for 3D polyhedra.
//
////////////////////////////////////////////////////////////////////////////////

// Some definitions
// Each polyhedra has an interpolating function f, which for any point (x,y,z) in the
// element, returns the scalar value at that point.  This interpolating function
// is based on several basis functions and is a polynomial of arbitrary degree.
//
// We have defined another function g which is the interpolating
// function along a given ray.  Therefore instead of being a function of (x,y,z), it is
// a function of t and gives the scalar value of the field along the ray for any point
// t.  We call this function the ray interpolating function.
//
// The user supplies us with the coefficients of f when the polyhedra is created.
// These are static.  The coefficients of g are dynamic and depend on the ray being
// cast through the element.  Therefore these coefficients will need to be calculated
// each time a ray is cast through the element.
//
#ifdef _MSC_VER
#pragma warning( disable : 4786 )
#endif

#ifndef ELVIS_JACOBI_EXTENSION__POLYHEDRA_H_
#define ELVIS_JACOBI_EXTENSION__POLYHEDRA_H_

#include <iostream>
#include <ElVis/Core/Vector.hpp>
#include <ElVis/Core/Point.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <float.h>
#include <vector>

#include <ElVis/Extensions/JacobiExtension/Typedefs.h>
#include <ElVis/Extensions/JacobiExtension/EndianConvert.h>
#include <ElVis/Extensions/JacobiExtension/SimpleMatrix.hpp>
#include <ElVis/Extensions/JacobiExtension/Declspec.h>
#include <ElVis/Extensions/JacobiExtension/Edge.h>
#include <ElVis/Extensions/JacobiExtension/Face.h>

namespace ElVis
{
    namespace JacobiExtension
    {
        enum PolyhedronType
        {
            TETRAHEDRON = 0,
            HEXAHEDRON,
            PYRAMID,
            PRISM
        };

        class FiniteElementVolume;

        /// \brief Represents a polyhedron in world space.
        class Polyhedron 
        {
        public:
            typedef double CoefficientType;
            typedef std::vector<CoefficientType> CoefficientListType;

        public:
            JACOBI_EXTENSION_EXPORT Polyhedron();
            JACOBI_EXTENSION_EXPORT Polyhedron(const Polyhedron& rhs);
            JACOBI_EXTENSION_EXPORT virtual ~Polyhedron();
            JACOBI_EXTENSION_EXPORT const Polyhedron& operator=(const Polyhedron& rhs);

            int id() const { return m_id; }

            JACOBI_EXTENSION_EXPORT virtual unsigned int NumberOfCoefficientsForOrder(unsigned int order) = 0;

            const std::vector<double>& basisCoefficients() const { return m_basisCoefficients; }
            unsigned int degree(int index) const { return m_degree[index]; }

            // Used when creating the bounding box acceleration routines.  This helps us know
            // the extent of the element so our bounding boxes can be as tight as possible.
            JACOBI_EXTENSION_EXPORT void elementBounds(double& min_x, double& min_y, double& min_z,
                double& max_x, double& max_y, double& max_z);
            JACOBI_EXTENSION_EXPORT void elementBounds(ElVis::WorldPoint& min, ElVis::WorldPoint& max);

            // Calculates the normal at a given point.  The result is stored in n.
            // This is done in world space.
            JACOBI_EXTENSION_EXPORT void calculateNormal(const ElVis::TensorPoint& p, ElVis::WorldVector* n);

            JACOBI_EXTENSION_EXPORT double findScalarValueAtPoint(const ElVis::WorldPoint& p);
            //JACOBI_EXTENSION_EXPORT double findScalarValueAtPoint(const rt::Ray& theRay, double t);

            /// This function finds the value at point t where t \in [-1, 1].
            /// So a mapping is performed so that t = -1 is mapped to min
            /// and t = 1 is mapped to max before evaluation.
            //JACOBI_EXTENSION_EXPORT double findScalarValueAtPoint(const rt::Ray& theRay, double min,
            //    double max, double t);


            JACOBI_EXTENSION_EXPORT double maxScalarValue() const;
            JACOBI_EXTENSION_EXPORT double minScalarValue() const;


            // Sometimes it is useful to refer to the derivatives in terms of
            // indicies instead of (a,b,c).
            // For example, df2(0,0) is the same as df2_da2, and df2(0,1) is the
            // same as df2_dab.
            //double df2(unsigned int dir1, unsigned int dir2, const ElVis::TensorPoint& p);

            /// \brief Calculates the world hessian.
            /// \param tp The point at which the hessian is evaluated.
            /// \param J The inverse transposed Jacobian of the mapping.
            /// \param hessian Output.
            //void worldHessian(const ElVis::TensorPoint& tp, JacobiExtension::Matrix<double, 3,3>& J, JacobiExtension::Matrix<double, 3, 3>& hessian);
            //void worldHessian(const ElVis::TensorPoint& tp, const JacobiExtension::Matrix<double, 3, 3>& J, JacobiExtension::Matrix<double, 3, 3> H);

            /// \brief Tranfrorms tensor vectors to world vectors.
            JACOBI_EXTENSION_EXPORT ElVis::WorldVector transformTensorToWorldVector(const ElVis::TensorPoint& origin,
                const ElVis::TensorVector& tv);


            virtual unsigned int numVertices() const = 0;
            virtual const ElVis::WorldPoint& vertex(int i) const = 0;

            // These functions deal with converting Points to and from world and reference space.
            // transformRayToReferenceSpace is provided as a convenience, it simply uses
            // transformPointFromWorldSpaceToTensorSpace.
            //
            // These functions are pure virtual functions as the transformation will be dependent upon
            // the type of element.  Hexahedrons will have different mappings than tetrahedrons, for
            // example.
            ElVis::TensorPoint transformWorldToTensor(const ElVis::WorldPoint& p)
            {
                ElVis::ReferencePoint ref = transformWorldToReference(p);
                return transformReferenceToTensor(ref);
            }

            bool containsPoint(const ElVis::WorldPoint& p)
            {
                ElVis::TensorPoint tp = transformWorldToTensor(p);
                return tp.a() >= -1.0001 && tp.a() <= 1.0001 &&
                    tp.b() >= -1.0001 && tp.b() <= 1.0001 &&
                    tp.c() >= -1.0001 && tp.c() <= 1.0001;
            }

            ElVis::WorldPoint transformTensorToWorld(const ElVis::TensorPoint& p)
            {
                ElVis::ReferencePoint ref = transformTensorToReference(p);
                return transformReferenceToWorld(ref);
            }

            JACOBI_EXTENSION_EXPORT virtual ElVis::ReferencePoint transformWorldToReference(const ElVis::WorldPoint& p);
            virtual ElVis::TensorPoint transformReferenceToTensor(const ElVis::ReferencePoint& p) = 0;

            virtual ElVis::ReferencePoint transformTensorToReference(const ElVis::TensorPoint& p) = 0;
            virtual ElVis::WorldPoint transformReferenceToWorld(const ElVis::ReferencePoint& p) = 0;


            // We will map each element to a hex bounded by (-1,-1,-1) and (1,1,1).
            // Therefore the intersection routine for each element type, once we have
            // transformed the ray, will be identical.
            //
            // The intersection routines for each element will perform high level geometric
            // intersection tests and, if the ray really intersects the element, will
            // pass it on to this function, which will transform the ray and find if
            // it intersects the isovalue.
            //JACOBI_EXTENSION_EXPORT double intersectsIsovalueAt(const rt::Ray& ray, double isovalue, rt::IntersectionInfo& hit,
            //						boost::shared_ptr<const FiniteElementVolume> vol);
            //         JACOBI_EXTENSION_EXPORT double intersectsIsovalueProjectionAt(const rt::Ray& ray, double isovalue, rt::IntersectionInfo& hit,
            //                                     boost::shared_ptr<const FiniteElementVolume> vol);
            //JACOBI_EXTENSION_EXPORT double intersectsIsovalueDynamicProjectionAt(const rt::Ray& ray, double isovalue, rt::IntersectionInfo& hit,
            //                            boost::shared_ptr<FiniteElementVolume> vol);
            //JACOBI_EXTENSION_EXPORT virtual bool findIntersectionWithGeometryInWorldSpace(const rt::Ray& ray, double& min, double& max) = 0;

            virtual double smallestX() const = 0;
            virtual double smallestY() const = 0;
            virtual double smallestZ() const = 0;
            virtual double largestX() const = 0;
            virtual double largestY() const = 0;
            virtual double largestZ() const = 0;

            virtual double getMax() const { return m_maxValue; }
            virtual double getMin() const { return m_minValue; }

            bool minHasBeenSet() const { return m_minValue != -DBL_MAX; }
            bool maxHasBeenSet() const { return m_maxValue != DBL_MAX; }

            // writes this element out in binary format, just as it will
            // be read.
            void writeElement(FILE* outFile);
            JACOBI_EXTENSION_EXPORT void writeForUnstructuredVTK(int x_samples, int y_samples, int z_samples, const char* fileName);
            JACOBI_EXTENSION_EXPORT void writeForStructuredVTK(double min_x, double min_y, double min_z, double max_x, double max_y, double max_z,
                int dim_x, int dim_y, int dim_z, const char* fileName);
            JACOBI_EXTENSION_EXPORT virtual void writeElementGeometryForVTK(const char* fileName) = 0;
            JACOBI_EXTENSION_EXPORT void writeElementAsIndividualVolume(const char* fileName);
            JACOBI_EXTENSION_EXPORT virtual void writeElementForVTKAsCell(const char* fileName) = 0;
            JACOBI_EXTENSION_EXPORT virtual void outputVertexOrderForVTK(std::ofstream& outFile, int startPoint) = 0;
            JACOBI_EXTENSION_EXPORT virtual int vtkCellType() = 0;
            JACOBI_EXTENSION_EXPORT virtual int GetCellType() = 0;


//            void interpolatingPolynomialDegreeOverride(unsigned int override)
//            {
//                if( override > 0 )
//                {
//                    m_interpolatingPolynomialDegreeOverride = override;
//                }
//            }

//            unsigned int interpolatingPolynomialDegreeOverride() const
//            {
//                return m_interpolatingPolynomialDegreeOverride;
//            }

//            double referenceToWorldTolerance() const { return m_referenceToWorldTolerance; }
//            double referenceToWorldTolerance(double tol) { return m_referenceToWorldTolerance = tol; }

//            const std::vector<double>& relativeErrorStats() { return m_relativeErrorStats; }
//            const std::vector<double>& absoluteErrorStates() { return m_absoluteErrorStats; }
//            const std::vector<double>& interpolationL2NormStats() { return m_interpolationL2NormStats; }
//            const std::vector<double>& projectionL2NormStats() { return m_projectionL2NormStats; }
//            const std::vector<double>& interpolationInfinityNormStats() { return m_interpolationInfinityNormStats; }
//            const std::vector<double>& projectionInfinityNormStats() { return m_projectionInfinityNormStats; }
//            const std::vector<double>& interp_relativeErrorStats() { return m_interp_relativeErrorStats; }
//            const std::vector<double>& interp_absoluteErrorStats() { return m_interp_absoluteErrorStats; }
//            const std::vector<double>& rootFindingErrorStats() { return m_rootFindingErrorStats; }


            /// \brief Calculates the scalar value at the point.
            /// \param p The point where we evaluate the scalar field.
            virtual double f(const ElVis::TensorPoint& p) const = 0;

            /// \brief Calculates \f$\frac{\partial f}{\partial a}\f$
            virtual double df_da(const ElVis::TensorPoint& p) const = 0;

            /// \brief Calculates \f$\frac{\partial f}{\partial b}\f$
            virtual double df_db(const ElVis::TensorPoint& p) const = 0;

            /// \brief Calculates \f$\frac{\partial f}{\partial c}\f$
            virtual double df_dc(const ElVis::TensorPoint& p) const = 0;

            /// \brief Calculates \f$\frac{\partial^2 f}{\partial a^2}\f$
            virtual double df2_da2(const ElVis::TensorPoint& p) const = 0;

            /// \brief Calculates \f$\frac{\partial^2 f}{\partial b^2}\f$
            virtual double df2_db2(const ElVis::TensorPoint& p) const = 0;

            /// \brief Calculates \f$\frac{\partial^2 f}{\partial c^2}\f$
            virtual double df2_dc2(const ElVis::TensorPoint& p) const = 0;

            /// \brief Calculates \f$\frac{\partial^2 f}{\partial a \partial b}\f$
            virtual double df2_dab(const ElVis::TensorPoint& p) const = 0;

            /// \brief Calculates \f$\frac{\partial^2 f}{\partial a \partial c}\f$
            virtual double df2_dac(const ElVis::TensorPoint& p) const = 0;

            /// \brief Calculates \f$\frac{\partial^2 f}{\partial b \partial c}\f$
            virtual double df2_dbc(const ElVis::TensorPoint& p) const = 0;

            //////////////////////////////////////////////
            /// Reference space stuff.
            //////////////////////////////////////////////

            /// \brief Calculates the hessian of the scalar field in the element.
            ///
            /// \param p The point where the hessian will be evaluated.
            /// \param H The output hessian at p.
            ///
            /// Calculates the hessian of the scalar function p.
            /// The index to J is to list the row first, then the column.
            /// The tensor space coordinates are a, b, and c.
            ///
            /// \f[
            ///     \left(
            ///         \begin{array}{ccc}
            ///             \frac{\partial^2\rho}{\partial a^2} & \frac{\partial^2 \rho}{\partial a \partial b} & \frac{\partial^2 \rho}{\partial a \partial c}\\
            ///             \frac{\partial^2\rho}{\partial b \partial a} & \frac{\partial^2 \rho}{\partial b^2} & \frac{\partial^2 \rho}{\partial b \partial c}\\
            ///             \frac{\partial^2\rho}{\partial c \partial a} & \frac{\partial^2 \rho}{\partial c \partial b} & \frac{\partial^2 \rho}{\partial c^2}\\
            ///         \end{array}\right)
            /// \f]
            JACOBI_EXTENSION_EXPORT void calculateScalarFunctionHessian(const ElVis::TensorPoint& p, JacobiExtension::Matrix<double, 3, 3>& H);

            /// \brief Calculates the gradient of the scalar field in world space.
            /// \param tp The point where the gradient will be evaluated.
            /// \return The gradient in world space.
            ///
            /// Calculates the gradient of the scalar function in tensor space.  This is:
            ///
            /// \f[
            ///     \left(
            ///         \begin{array}{c}
            ///             \frac{\partial \rho}{\partial x} \\
            ///             \frac{\partial \rho}{\partial y} \\
            ///             \frac{\partial \rho}{\partial z} \\
            ///         \end{array}\right)
            /// \f]
            JACOBI_EXTENSION_EXPORT ElVis::WorldVector calculateScalarFunctionWorldGradient(const ElVis::TensorPoint& tp);

            /// \brief Calculates the gradient of the scalar field in world space.
            /// \param tv The point where the gradient will be evaluated.
            /// \param J The inverse transpose Jacobian of the mapping function.
            /// \return The gradient in world space.
            ///
            /// Calculates the gradient of the scalar function in world space.  This is:
            ///
            /// \f[
            ///     \left(
            ///         \begin{array}{c}
            ///             \frac{\partial \rho}{\partial x} \\
            ///             \frac{\partial \rho}{\partial y} \\
            ///             \frac{\partial \rho}{\partial z} \\
            ///         \end{array}\right)
            /// \f]
            JACOBI_EXTENSION_EXPORT ElVis::WorldVector calculateScalarFunctionWorldGradient(const ElVis::TensorVector& tv, JacobiExtension::Matrix<double, 3, 3>& J);

            /// \brief Calculates the gradient of the scalar field in tensor space.
            /// \param p The point where the gradient will be evaluated.
            /// \return The gradient in tensor space.
            ///
            /// Calculates the gradient of the scalar function in tensor space.  This is:
            ///
            /// \f[
            ///     \left(
            ///         \begin{array}{c}
            ///             \frac{\partial f}{\partial a} \\
            ///             \frac{\partial f}{\partial b} \\
            ///             \frac{\partial f}{\partial c} \\
            ///         \end{array}\right)
            /// \f]
            JACOBI_EXTENSION_EXPORT ElVis::TensorVector calculateScalarFunctionTensorGradient(const ElVis::TensorPoint& p);


            /// \brief Calculates the jacobian of the mapping between tensor and world spaces.
            /// \param p The point where the jacobian will be evaluated.
            /// \param J The output jacobian.
            ///
            /// The mapping functions are defined as:
            ///
            /// \f$ x = f_x(a,b,c) \f$
            ///
            /// \f$ y = f_y(a,b,c) \f$
            ///
            /// \f$ z = f_z(a,b,c) \f$
            ///
            ///
            /// The jacobian of this mapping is:
            ///
            /// \f[
            ///     \left(
            ///         \begin{array}{ccc}
            ///             \frac{\partial f_x}{\partial a} & \frac{\partial f_x}{\partial b} & \frac{\partial f_x}{\partial c }\\
            ///             \frac{\partial f_y}{\partial a} & \frac{\partial f_y}{\partial b} & \frac{\partial f_y}{\partial c }\\
            ///             \frac{\partial f_z}{\partial a} & \frac{\partial f_z}{\partial b} & \frac{\partial f_z}{\partial c }\\
            ///         \end{array}\right)
            /// \f]
            JACOBI_EXTENSION_EXPORT virtual void calculateTensorToWorldSpaceMappingJacobian(const ElVis::TensorPoint& p, JacobiExtension::Matrix<double, 3, 3>& J) = 0;

            /// \brief Calculates the hessian of the tensor to world mapping function.
            /// \param p The point for which the hessian will be computed.
            /// \param H The output hessian.
            ///
            /// The mapping functions are defined as:
            ///
            /// \f$ x = f_x(a,b,c) \f$
            ///
            /// \f$ y = f_y(a,b,c) \f$
            ///
            /// \f$ z = f_z(a,b,c) \f$
            ///
            /// \f[ H_x =
            ///     \left(
            ///         \begin{array}{ccc}
            ///             \frac{\partial^2f_x}{\partial a^2} & \frac{\partial^2 f_x}{\partial a \partial b} & \frac{\partial^2 f_x}{\partial a \partial c}\\
            ///             \frac{\partial^2f_x}{\partial b \partial a} & \frac{\partial^2 f_x}{\partial b^2} & \frac{\partial^2 f_x}{\partial b \partial c}\\
            ///             \frac{\partial^2f_x}{\partial c \partial a} & \frac{\partial^2 f_x}{\partial c \partial b} & \frac{\partial^2 f_x}{\partial c^2}\\
            ///         \end{array}\right)
            /// \f]
            ///
            /// \f[ H_y =
            ///     \left(
            ///         \begin{array}{ccc}
            ///             \frac{\partial^2f_y}{\partial a^2} & \frac{\partial^2 f_y}{\partial a \partial b} & \frac{\partial^2 f_y}{\partial a \partial c}\\
            ///             \frac{\partial^2f_y}{\partial b \partial a} & \frac{\partial^2 f_y}{\partial b^2} & \frac{\partial^2 f_y}{\partial b \partial c}\\
            ///             \frac{\partial^2f_y}{\partial c \partial a} & \frac{\partial^2 f_y}{\partial c \partial b} & \frac{\partial^2 f_y}{\partial c^2}\\
            ///         \end{array}\right)
            /// \f]
            ///
            /// \f[ H_z =
            ///     \left(
            ///         \begin{array}{ccc}
            ///             \frac{\partial^2f_z}{\partial a^2} & \frac{\partial^2 f_z}{\partial a \partial b} & \frac{\partial^2 f_z}{\partial a \partial c}\\
            ///             \frac{\partial^2f_z}{\partial b \partial a} & \frac{\partial^2 f_z}{\partial b^2} & \frac{\partial^2 f_z}{\partial b \partial c}\\
            ///             \frac{\partial^2f_z}{\partial c \partial a} & \frac{\partial^2 f_z}{\partial c \partial b} & \frac{\partial^2 f_z}{\partial c^2}\\
            ///         \end{array}\right)
            /// \f]
            JACOBI_EXTENSION_EXPORT virtual void calculateTensorToWorldSpaceMappingHessian(const ElVis::TensorPoint& p, JacobiExtension::Matrix<double, 3, 3>& Hx,
                JacobiExtension::Matrix<double, 3, 3>& Hy, JacobiExtension::Matrix<double, 3, 3>& Hz) = 0;
            JACOBI_EXTENSION_EXPORT void calculateTensorToWorldSpaceMappingMiriahHessian(const ElVis::TensorPoint& p, JacobiExtension::Matrix<double, 3, 3>& Hr,
                JacobiExtension::Matrix<double, 3, 3>& Hs, JacobiExtension::Matrix<double, 3, 3>& Ht);

            /// \brief Calculated the Jacobian of the mapping from tensor to world space.
            /// \prarm
            JACOBI_EXTENSION_EXPORT virtual void getWorldToReferenceJacobian(const ElVis::ReferencePoint& rp, JacobiExtension::Matrix<double, 3, 3>& J) = 0;

            JACOBI_EXTENSION_EXPORT void calculateInverseJacobian(double r, double s, double t, JacobiExtension::Matrix<double, 3, 3>& inverse);


            JACOBI_EXTENSION_EXPORT void transposeAndInvertMatrix(const JacobiExtension::Matrix<double, 3, 3>& J, JacobiExtension::Matrix<double, 3, 3>& inverse);

            JACOBI_EXTENSION_EXPORT void calculateWorldToTensorInverseTransposeJacobian(const ElVis::TensorPoint& p, JacobiExtension::Matrix<double, 3, 3>& J);

            JACOBI_EXTENSION_EXPORT unsigned int NumberOfEdges() const { return DoNumberOfEdges(); }
            JACOBI_EXTENSION_EXPORT Edge GetEdge(unsigned int id) const { return DoGetEdge(id); }
            JACOBI_EXTENSION_EXPORT unsigned int NumberOfFaces() const { return DoNumberOfFaces(); }
            JACOBI_EXTENSION_EXPORT Face GetFace(unsigned int id) const { return DoGetFace(id); }
        protected:
//            JACOBI_EXTENSION_EXPORT void setInterpolatingPolynomialDegree(unsigned int newDegree) { m_interpolatingPolynomialDegree = newDegree; }

            // The actual degree of the interpolating polynomial which lies along
            // the ray.  This is defined to be the sum of the degrees in each
            // direction.
//            unsigned int m_interpolatingPolynomialDegree;
//            unsigned int m_interpolatingPolynomialDegreeOverride;

            // The degree of the interpolating polynomials in each direction.
            unsigned int m_degree[3];

            int m_id;
            static int ID_COUNTER;

            /// These functions are constructor helpers which set portions
            /// of the class from a file.
            JACOBI_EXTENSION_EXPORT void setMinMax(FILE* inFile, bool reverseBytes = false);
            JACOBI_EXTENSION_EXPORT void readDegree(FILE* inFile, bool reverseBytes = false);
            JACOBI_EXTENSION_EXPORT void writeDegree(FILE* outFile);
            JACOBI_EXTENSION_EXPORT void writeBasisCoefficients(FILE* inFile);
            JACOBI_EXTENSION_EXPORT virtual void writeVertices(FILE* outFile) = 0;

            JACOBI_EXTENSION_EXPORT void readBasisCoefficients(FILE* inFile, int numCoefss, bool reverseBytes = false);
            JACOBI_EXTENSION_EXPORT void setBasisCoefficients(const CoefficientListType& newCoeffs) { m_basisCoefficients = newCoeffs; }
            JACOBI_EXTENSION_EXPORT void setDegree(unsigned int degree1, unsigned int degree2,
                unsigned int degree3)
            {
                m_degree[0] = degree1;
                m_degree[1] = degree2;
                m_degree[2] = degree3;
            }

        private:

            void calculateMinMax() const;

            /// \brief Returns the element's maximum scalar value.
            virtual double do_maxScalarValue() const = 0;

            /// \brief Returns the element's minimum scalar value.
            virtual double do_minScalarValue() const = 0;

            virtual unsigned int DoNumberOfEdges() const = 0;
            virtual Edge DoGetEdge(unsigned int id) const = 0;
            virtual unsigned int DoNumberOfFaces() const = 0;
            virtual Face DoGetFace(unsigned int id) const = 0;

            /// rt::RayTraceableObject interface.  These methods deal with geometry only.
            //virtual double do_findParametricIntersection(const rt::Ray& theRay);
            //virtual ElVis::WorldVector do_calculateNormal(const rt::IntersectionInfo& intersection);

            // Each element will be represented by a series of basis functions
            // sum(i,j,k) a_i,j,k * phi_i * phi_j * phi_k
            //
            // The number of coefficients (a_i,j,k) depends on the order of the basis functions.
            //
            // Ex.  If we defined phi_i, phi_j, and phi_k up to second order then we'll need
            // a total of 27 coefficients (since i,j,k will range from 0..2, giving a_0,0,0 to a_2,2,2).
            CoefficientListType m_basisCoefficients;

            // The minimum and maximum scalar values of the element.
            mutable double m_minValue;
            mutable double m_maxValue;

            void assignId();
            void copy(const Polyhedron& rhs);
//            double m_referenceToWorldTolerance;

            // Stats
//            std::vector<double> m_relativeErrorStats;
//            std::vector<double> m_rootFindingErrorStats;
//            std::vector<double> m_absoluteErrorStats;
//            std::vector<double> m_interp_relativeErrorStats;
//            std::vector<double> m_interp_absoluteErrorStats;
//            std::vector<double> m_interpolationL2NormStats;
//            std::vector<double> m_projectionL2NormStats;
//            std::vector<double> m_interpolationInfinityNormStats;
//            std::vector<double> m_projectionInfinityNormStats;

        };

        /// \brief Creates a polyhedron from the given file.
        boost::shared_ptr<Polyhedron> createNewPolyhedron(FILE* is, bool reverseBytes = false);

        typedef std::vector<boost::shared_ptr<Polyhedron> > PolyhedraVector;

        template<unsigned int NUM_VERTICES>
        class PolyhedronWithVertices : public Polyhedron
        {
        public:
            static const unsigned int VertexCount = NUM_VERTICES;

            PolyhedronWithVertices() {}
            virtual ~PolyhedronWithVertices() {}
            void readVerticesFromFile(FILE* inFile, bool reverseBytes);

            virtual unsigned int numVertices() const { return NUM_VERTICES; }
            virtual const ElVis::WorldPoint& vertex(int i) const;

            virtual double smallestX() const;
            virtual double smallestY() const;
            virtual double smallestZ() const;
            virtual double largestX() const;
            virtual double largestY() const;
            virtual double largestZ() const;


        protected:
            void setVertex(const ElVis::WorldPoint& v, unsigned int index);
            virtual void writeVertices(FILE* outFile);

        private:
            /// Finds the minimum value for the given index among
            /// all vertices.
            double findMinIndex(unsigned int index) const;
            double findMaxIndex(unsigned int index) const;
            ElVis::WorldPoint vertices[NUM_VERTICES];
        };


        template<unsigned int NUM_VERTICES>
        const ElVis::WorldPoint& PolyhedronWithVertices<NUM_VERTICES>::vertex(int i) const
        {
#ifdef _DEBUG
            if( i < 0 || i >= NUM_VERTICES )
            {
                std::cerr << "ERROR:  Attempted to access illegal vertex " <<
                    i << " of a polyhedron with " << NUM_VERTICES
                    << "vertices." << std::endl;
                static ElVis::WorldPoint p(0,0,0);
                return p;
            }
#endif

            return vertices[i];
        }

        template<unsigned int NUM_VERTICES>
        void PolyhedronWithVertices<NUM_VERTICES>::readVerticesFromFile(FILE* inFile, bool reverseBytes)
        {
            for(unsigned int i = 0; i < NUM_VERTICES; ++i)
            {
                double points[3];
                if( fread(points, sizeof(double), 3, inFile) != 3 )
                {
                    std::cerr << "Error reading point " << i << std::endl;
                    exit(1);
                }

                if( reverseBytes )
                {
                    JacobiExtension::reverseBytes(points[0]);
                    JacobiExtension::reverseBytes(points[1]);
                    JacobiExtension::reverseBytes(points[2]);
                }
                vertices[i] = ElVis::WorldPoint(points[0], points[1], points[2]);
            }

            for(unsigned int i = 0; i < 8u-NUM_VERTICES; ++i)
            {
                double points[3];
                if( fread(points, sizeof(double), 3, inFile) != 3 )
                {
                    std::cerr << "Error reading empty points in prism." << std::endl;
                    exit(1);
                }
            }
        }

        template<unsigned int NUM_VERTICES>
        void PolyhedronWithVertices<NUM_VERTICES>::writeVertices(FILE* outFile)
        {
            for(unsigned int i = 0; i < NUM_VERTICES; ++i)
            {
                fwrite(vertices[i].GetPtr(), sizeof(double), 3, outFile);
            }

            for(unsigned int i = 0; i < 8u-NUM_VERTICES; ++i)
            {
                double points[3];
                fwrite(points, sizeof(double), 3, outFile);
            }
        }


        template<unsigned int NUM_VERTICES>
        double PolyhedronWithVertices<NUM_VERTICES>::smallestX() const
        {
            return findMinIndex(0);
        }

        template<unsigned int NUM_VERTICES>
        double PolyhedronWithVertices<NUM_VERTICES>::smallestY() const
        {
            return findMinIndex(1);
        }

        template<unsigned int NUM_VERTICES>
        double PolyhedronWithVertices<NUM_VERTICES>::smallestZ() const
        {
            return findMinIndex(2);
        }

        template<unsigned int NUM_VERTICES>
        double PolyhedronWithVertices<NUM_VERTICES>::largestX() const
        {
            return findMaxIndex(0);
        }

        template<unsigned int NUM_VERTICES>
        double PolyhedronWithVertices<NUM_VERTICES>::largestY() const
        {
            return findMaxIndex(1);
        }

        template<unsigned int NUM_VERTICES>
        double PolyhedronWithVertices<NUM_VERTICES>::largestZ() const
        {
            return findMaxIndex(2);
        }

        template<unsigned int NUM_VERTICES>
        double PolyhedronWithVertices<NUM_VERTICES>::findMinIndex(unsigned int index) const
        {
            double result = std::numeric_limits<double>::max();
            for(unsigned int i = 0; i < NUM_VERTICES; ++i)
            {
                if( vertices[i][index] < result )
                {
                    result = vertices[i][index];
                }
            }
            return result;
        }

        template<unsigned int NUM_VERTICES>
        double PolyhedronWithVertices<NUM_VERTICES>::findMaxIndex(unsigned int index) const
        {
#ifdef min
#undef min
#endif
            double result = -std::numeric_limits<double>::min();
            for(unsigned int i = 0; i < NUM_VERTICES; ++i)
            {
                if( vertices[i][index] > result )
                {
                    result = vertices[i][index];
                }
            }
            return result;
        }

        template<unsigned int NUM_VERTICES>
        void PolyhedronWithVertices<NUM_VERTICES>::setVertex(const ElVis::WorldPoint& v, unsigned int index)
        {
            vertices[index] = v;
        }
    }
}

#endif
