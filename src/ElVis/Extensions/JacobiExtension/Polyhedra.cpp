////////////////////////////////////////////////////////////////////////////////
//
//  File: hoPolyhedra.cpp
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

#include <ElVis/Extensions/JacobiExtension/Isosurface.h>
#include <limits>

#include <iostream>
#include <vector>

#include <algorithm>

#include <ElVis/Extensions/JacobiExtension/Polyhedra.h>
#include <ElVis/Extensions/JacobiExtension/Hexahedron.h>
#include <ElVis/Extensions/JacobiExtension/Prism.h>
//#include <ray_tracer/rtIntersectionInfo.h>
#include <ElVis/Extensions/JacobiExtension/FiniteElementMath.h>
#include <ElVis/Extensions/JacobiExtension/EndianConvert.h>
#include <ElVis/Core/Vector.hpp>
#include <ElVis/Extensions/JacobiExtension/PolynomialInterpolation.hpp>
#include <ElVis/Extensions/JacobiExtension/Polynomial.hpp>
#include <ElVis/Extensions/JacobiExtension/NumericalIntegration.hpp>
#include <ElVis/Extensions/JacobiExtension/Tetrahedron.h>
#include <ElVis/Extensions/JacobiExtension/Writers.h>
#include <ElVis/Extensions/JacobiExtension/FiniteElementVolume.h>

#include <ElVis/Extensions/JacobiExtension/FiniteElementVolume.h>
#include <ElVis/Extensions/JacobiExtension/Writers.h>
#include <ElVis/Extensions/JacobiExtension/Tetrahedron.h>

#include <math.h>
#include <set>
#include <limits>
#include <boost/bind.hpp>

using std::set;
using std::vector;


using namespace std;

namespace ElVis
{
    namespace JacobiExtension
    {
#ifdef max
#undef max
#endif

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif 

#ifndef M_PI_2
#define M_PI_2 1.570796326794896619
#endif


        int Polyhedron::ID_COUNTER = 0;

        boost::shared_ptr<Polyhedron> createNewPolyhedron(FILE* in, bool reverseBytes)
        {
            int type;
            if( fread(&type, sizeof(int), 1, in) != 1 )
            {
                cerr << "Error reading type." << endl;
                exit(1);
            }

            //std::cout << "Read type " << type << std::endl;
            if( reverseBytes ) JacobiExtension::reverseBytes(type);

            if( type == HEXAHEDRON )
            {
                return boost::shared_ptr<Polyhedron>(new Hexahedron(in, reverseBytes));
            }
            else if( type == PRISM )
            {
                return boost::shared_ptr<Polyhedron>(new Prism(in, reverseBytes));
            }
            //else if( type == TETRAHEDRON )
            //{
            //    return boost::shared_ptr<Polyhedron>(new Tetrahedron(in, reverseBytes));
            //}
            else
            {
                cerr << "Tried to read type " << type << endl;
                exit(1);
            }

            return boost::shared_ptr<Polyhedron>();
        }

        Polyhedron::Polyhedron() :
            //m_interpolatingPolynomialDegree(0),
            //m_interpolatingPolynomialDegreeOverride(0),
            m_id(0),
            m_basisCoefficients(),
            m_minValue(-std::numeric_limits<double>::max()),
            m_maxValue(std::numeric_limits<double>::max())//,
            //m_referenceToWorldTolerance(numeric_limits<double>::epsilon())
        {
            m_degree[0] = 0;
            m_degree[1] = 0;
            m_degree[2] = 0;
            assignId();
        }

        Polyhedron::Polyhedron(const Polyhedron& rhs) :
            //m_interpolatingPolynomialDegree(0),
            //m_interpolatingPolynomialDegreeOverride(0),
            m_id(0),
            m_basisCoefficients(),
            m_minValue(-std::numeric_limits<double>::max()),
            m_maxValue(std::numeric_limits<double>::max())
        {
            copy(rhs);
            assignId();
        }

        const Polyhedron& Polyhedron::operator=(const Polyhedron& rhs)
        {
            if( this != &rhs )
            {
                copy(rhs);
                assignId();
            }

            return *this;
        }

        void Polyhedron::copy(const Polyhedron& rhs)
        {
            m_basisCoefficients = rhs.m_basisCoefficients;
//            m_interpolatingPolynomialDegree = rhs.m_interpolatingPolynomialDegree;
//            m_interpolatingPolynomialDegreeOverride = rhs.m_interpolatingPolynomialDegreeOverride;
            m_minValue = rhs.m_minValue;
            m_maxValue = rhs.m_maxValue;
            m_id = rhs.m_id;
            m_degree[0] = rhs.m_degree[0];
            m_degree[1] = rhs.m_degree[1];
            m_degree[2] = rhs.m_degree[2];
        }

        Polyhedron::~Polyhedron()
        {
        }

        void Polyhedron::writeForStructuredVTK(double min_x, double min_y, double min_z, double max_x, double max_y, double max_z,
            int dim_x, int dim_y, int dim_z, const char* fileName)
        {
            if( dim_x <=1 || dim_y <= 1 || dim_z <= 1 ) return;

            double x_h = (max_x-min_x)/(dim_x-1);
            double y_h = (max_y-min_y)/(dim_y-1);
            double z_h = (max_z-min_z)/(dim_z-1);

            std::vector<double> values;
            for(int z = 0; z < dim_z; z++)
            {
                for(int y = 0; y < dim_y; y++)
                {
                    for(int x = 0; x < dim_x; x++)
                    {
                        ElVis::WorldPoint wp(min_x + x*x_h, min_y + y*y_h, min_z + z*z_h);
                        ElVis::TensorPoint tp = transformWorldToTensor(wp);

                        double a = tp[0];
                        double b = tp[1];
                        double c = tp[2];

                        if( a >= -1 && a <= 1 &&
                            b >= -1 && b <= 1 &&
                            c >= -1 && c <= 1 )
                        {
                            values.push_back(findScalarValueAtPoint(wp));
                        }
                        else
                        {
                            values.push_back(FLT_MAX);
                        }
                    }
                }
            }

            writeStructuredData(min_x, min_y, min_z, x_h, y_h, z_h,
                dim_x, dim_y, dim_z,
                values, fileName);
        }

        void Polyhedron::writeForUnstructuredVTK(int num_x_samples, int num_y_samples, int num_z_samples, const char* fileName)
        {
            double min_x = -1;
            double min_y = -1;
            double min_z = -1;

            double max_x = 1;
            double max_y = 1;
            double max_z = 1;

            double x_h = (max_x-min_x)/(num_x_samples-1);
            double y_h = (max_y-min_y)/(num_y_samples-1);
            double z_h = (max_z-min_z)/(num_z_samples-1);

            std::vector<ElVis::WorldPoint > points;
            std::vector<double> values;

            for(int k = 0; k < num_z_samples; k++)
            {
                for(int j = 0; j < num_y_samples; j++)
                {
                    for(int i = 0; i < num_x_samples; i++)
                    {
                        ElVis::TensorPoint tp(min_x + i*x_h, min_y + j*y_h, min_z + k*z_h);
                        double val = f(tp);
                        values.push_back(val);

                        ElVis::WorldPoint wp = transformTensorToWorld(tp);
                        points.push_back(wp);
                    }
                }
            }

            ofstream outFile(fileName, ios::out);
            writeUnstructuredData(points, values, outFile);
            outFile.close();
        }

        void Polyhedron::writeElementAsIndividualVolume(const char* fileName)
        {
            FILE* outFile = fopen(fileName, "wb");
            FiniteElementVolume::writeHeader(outFile);

            int endianCheck = 1;
            fwrite(&endianCheck, sizeof(int), 1, outFile);

            fwrite(&endianCheck, sizeof(int), 1, outFile);
            writeElement(outFile);
            fclose(outFile);
        }



        // Lagrange version.
        //double Polyhedron::intersectsIsovalueAt(const rt::Ray& ray, double isovalue, rt::IntersectionInfo& hit,
        //                                    boost::shared_ptr<const FiniteElementVolume> vol)
        //{
        //    return intersectsIsovalueProjectionAt(ray, isovalue, hit, vol);
        //    //return intersectsIsovalueDynamicProjection(ray, isovalue, hit, vol);

        //}

        //double Polyhedron::intersectsIsovalueProjectionAt(const rt::Ray& ray, double isovalue, rt::IntersectionInfo& hit,
        //                        boost::shared_ptr<const FiniteElementVolume> vol)
        //{
        //    ElVis::WorldPoint origin(ray.getOrigin().x(),
        //        ray.getOrigin().y(), ray.getOrigin().z());

        //    double min, max;
        //    if( !findIntersectionWithGeometryInWorldSpace(ray, min, max) )
        //    {
        //        return false;
        //    }

        //    int actualDegree = m_interpolatingPolynomialDegree;

        //    if( interpolatingPolynomialDegreeOverride() > 0 )
        //    {
        //        actualDegree = interpolatingPolynomialDegreeOverride();
        //    }

        //    // Generate the polynomial projection.
        //    boost::function1<double, double> actual_func = boost::bind(
        //        &Polyhedron::findScalarValueAtPoint,
        //        this, ray, min, max, _1);
        //    Polynomial::OrthogonalLegendrePolynomial poly =
        //            PolynomialInterpolation::generateLeastSquaresPolynomialProjection<double>(
        //                actualDegree, actual_func);
        //    poly.reducePolynomialByRemovingSmallLeadingCoefficients(1e-15);

        //    // Calculate the L2 Norm by integration.
        ////  SubtractFuncs<double> sub(poly, actual_func);
        ////  double l2_norm = NumericalIntegration::GaussLegendreQuadrature(
        ////      boost::function1<double, double>(sub),
        ////      5*actualDegree+3, -1.0, 1.0);
        ////  m_projectionL2NormStats.push_back(l2_norm);
        ////
        ////  // Get a feel for the infinity norm by sampling.
        ////  double infinity_norm = -std::numeric_limits<double>::max();
        ////  int n = 1000;
        ////  double h = 2.0/n;
        ////  for(int i = 0; i < n; ++i)
        ////  {
        ////      double t = -1.0 + h*i;
        ////      double real_val = actual_func(t);
        ////      double projected_val = poly(t);
        ////      double err = fabs(real_val-projected_val);
        ////      if( err > infinity_norm )
        ////      {
        ////          infinity_norm = err;
        ////      }
        ////  }
        ////  m_projectionInfinityNormStats.push_back(infinity_norm);

        //    Polynomial::Monomial mon_poly = Polynomial::convertToMonomial(poly);
        //    mon_poly.coeff(0) = mon_poly.coeff(0) - isovalue;
        //    std::vector<double> roots = Polynomial::findRealRoots(mon_poly);
        //    size_t numRoots = roots.size();

        //    if( numRoots > 0 )
        //    {
        //        // TODO - We haven't dealt with the camera being inside the element yet.
        //        double found_root = DBL_MAX;
        //        for(size_t i = 0; i < numRoots; i++)
        //        {
        //            if( roots[i] >= -1 && roots[i] <= 1 && roots[i] < found_root )
        //            {
        //                found_root = roots[i];
        //            }
        //        }

        //        if( found_root != DBL_MAX )
        //        {
        //            double worldT = (found_root+1)/2.0 * (max-min) + min;
        //            return worldT;
        //        }
        //    }

        //    return -1;
        //}


        //double Polyhedron::findScalarValueAtPoint(const rt::Ray& theRay, double min,
        //            double max, double t)
        //{
        //    assert(t >= -1.0 && t <= 1.0);
        //    assert(min <= max);

        //    // Find a new value of t in the range [min, max]
        //    double world_t = (t+1.0)/2.0 * (max-min) + min;
        //    return findScalarValueAtPoint(theRay, world_t);
        //}

        //double Polyhedron::findScalarValueAtPoint(const rt::Ray& theRay, double t)
        //{
        //    ElVis::WorldPoint p = theRay.getOrigin() + findPointAlongVector(theRay.getDirection(),t);
        //    return findScalarValueAtPoint(p);
        //}

        double Polyhedron::findScalarValueAtPoint(const ElVis::WorldPoint& p)
        {
            ElVis::TensorPoint tp = transformWorldToTensor(p);
            return f(tp);
        }

        void Polyhedron::calculateScalarFunctionHessian(const ElVis::TensorPoint& p, Matrix<double, 3, 3>& H)
        {
            H(0,0) = df2_da2(p);
            H(0,1) = df2_dab(p);
            H(0,2) = df2_dac(p);

            // H(1,0) = df2_dab
            H(1,0) = H(0,1);
            H(1,1) = df2_db2(p);
            H(1,2) = df2_dbc(p);

            // H(2,0) = df2_dac
            // H(2,1) = df2_dbc
            H(2,0) = H(0,2);
            H(2,1) = H(1,2);
            H(2,2) = df2_dc2(p);
        }

        //void Polyhedron::worldHessian(const ElVis::TensorPoint& tp, const Matrix<double, 3, 3>& J, Matrix<double, 3, 3> H)
        //{
        //  Matrix<double, 3, 3> tH;
        //  calculateScalarFunctionHessian(tp, tH);

        //  for(unsigned int i = 0; i < 3; ++i)
        //  {
        //      for(unsigned int j = 0; j < 3; ++j)
        //      {
        //          double elementVal = 0.0;
        //          for(unsigned int m = 0; m < 3; ++m)
        //          {
        //              for(unsigned int n = 0; n < 3; ++n)
        //              {
        //                  elementVal += tH.getData(m,n) * J.getData(i, m) * J.getData(j, n);
        //              }
        //          }
        //          H.setData(i,j) = elementVal;
        //      }
        //  }
        //}

        //double Polyhedron::df2(unsigned int dir1, unsigned int dir2, const ElVis::TensorPoint& p)
        //{
        //  if( dir1 == 0 )
        //  {
        //      if( dir2 == 0 )
        //      {
        //          return df2_da2(p);
        //      }
        //      else if( dir2 == 1 )
        //      {
        //          return df2_dab(p);
        //      }
        //      else if( dir2 == 2 )
        //      {
        //          return df2_dac(p);
        //      }
        //  }
        //  else if( dir1 == 1 )
        //  {
        //      if( dir2 == 0 )
        //      {
        //          return df2_da2(p);
        //      }
        //      else if( dir2 == 1 )
        //      {
        //          return df2_dab(p);
        //      }
        //      else if( dir2 == 2 )
        //      {
        //          return df2_dac(p);
        //      }
        //  }
        //  else if( dir1 == 2 )
        //  {
        //      if( dir2 == 0 )
        //      {
        //          return df2_da2(p);
        //      }
        //      else if( dir2 == 1 )
        //      {
        //          return df2_dab(p);
        //      }
        //      else if( dir2 == 2 )
        //      {
        //          return df2_dac(p);
        //      }
        //  }
        //}

        //void Polyhedron::worldHessian(const ElVis::TensorPoint& tp, Matrix<double, 3,3>& J, Matrix<double, 3, 3>& hessian)
        //{
        // The inverse transpose of the mapping Jacobian looks like this:
        //
        // [ da/dx db/dx dc/dc ]
        // [ da/dy db/dy dc/dy ]
        // [ da/dz db/dz dc/dz ]
        //
        // So where these derivative values are needed we can obtain them directly
        // from J.
        //
        // The final value of hessian[i][j] = Diff(phi,i,j) = Diff(phi,m,n)*Diff(m,i)*Diff(n,j)+Diff(phi,m)*Diff(m,i,j)
        // Where we loop over all possible m and n.
        //
        // This isn't necessarily optimal, let's see if it works first.

        //for(unsigned int i = 0; i < 2; ++i)
        //{
        //  for(unsigned int j = 0; j < 2; ++j)
        //  {
        //      unsigned int hessianIndex = i*3 + j;
        //      hessian[hessianIndex] = 0.0;

        //      for(unsigned int m = 0; m < 2; ++m)
        //      {
        //          for(unsigned int n = 0; n < 2; ++n)
        //          {
        //          }
        //      }
        //  }
        //}
        //}

        ElVis::TensorVector Polyhedron::calculateScalarFunctionTensorGradient(const ElVis::TensorPoint& p)
        {
            return ElVis::TensorVector(
              static_cast<ElVis::TensorVector::value_type>(df_da(p)), 
              static_cast<ElVis::TensorVector::value_type>(df_db(p)), 
              static_cast<ElVis::TensorVector::value_type>(df_dc(p)));
        }

        ElVis::WorldVector Polyhedron::calculateScalarFunctionWorldGradient(const ElVis::TensorPoint& tp)
        {
            ElVis::TensorVector tv = calculateScalarFunctionTensorGradient(tp);
            return transformTensorToWorldVector(tp, tv);
        }

        ElVis::WorldVector Polyhedron::transformTensorToWorldVector(const ElVis::TensorPoint& origin,
            const ElVis::TensorVector& tv)
        {
            Matrix<double, 3, 3> J;
            calculateWorldToTensorInverseTransposeJacobian(origin, J);

            return calculateScalarFunctionWorldGradient(tv, J);
        }

        ElVis::WorldVector Polyhedron::calculateScalarFunctionWorldGradient(const ElVis::TensorVector& tv, Matrix<double, 3, 3>& J)
        {
            ElVis::WorldVector result;
            result.SetX(tv.x()*J[0] + tv.y()*J[1] + tv.z()*J[2]);
            result.SetY(tv.x()*J[3] + tv.y()*J[4] + tv.z()*J[5]);
            result.SetZ(tv.x()*J[6] + tv.y()*J[7] + tv.z()*J[8]);

            return result;
        }

        void Polyhedron::calculateNormal(const ElVis::TensorPoint& p, ElVis::WorldVector* n)
        {
            ElVis::TensorVector grad = calculateScalarFunctionTensorGradient(p);
            (*n) = transformTensorToWorldVector(p, grad);
            n->Normalize();
        }

        void Polyhedron::elementBounds(ElVis::WorldPoint& min, ElVis::WorldPoint& max)
        {
            double min_x, min_y, min_z, max_x, max_y, max_z;
            elementBounds(min_x, min_y, min_z, max_x, max_y, max_z);

            min = ElVis::WorldPoint(min_x, min_y, min_z);
            max = ElVis::WorldPoint(max_x, max_y, max_z);
            //min.set(min_x, min_y, min_z);
            //max.set(max_x, max_y, max_z);
        }

        void Polyhedron::elementBounds(double& min_x, double& min_y, double& min_z,
            double& max_x, double& max_y, double& max_z)
        {
            min_x = FLT_MAX;
            min_y = FLT_MAX;
            min_z = FLT_MAX;
            max_x = -FLT_MAX;
            max_y = -FLT_MAX;
            max_z = -FLT_MAX;

            for(unsigned int i = 0; i < numVertices(); i++)
            {
                ElVis::WorldPoint v = vertex(i);
                if( v.x() < min_x )
                    min_x = v.x();
                if( v.x() > max_x )
                    max_x = v.x();

                if( v.y() < min_y )
                    min_y = v.y();
                if( v.y() > max_y )
                    max_y = v.y();

                if( v.z() < min_z )
                    min_z = v.z();
                if( v.z() > max_z )
                    max_z = v.z();
            }
        }

        //ElVis::TensorPoint Polyhedron::transformWorldToTensor(const ElVis::WorldPoint& p)
        //{
        //  ElVis::ReferencePoint ref = transformWorldToReference(p);
        //  return transformReferenceToTensor(ref);
        //}
        //
        //ElVis::WorldPoint Polyhedron::transformTensorToWorld(const ElVis::TensorPoint& p)
        //{
        //  ElVis::ReferencePoint ref = transformTensorToReference(p);
        //  return transformReferenceToWorld(ref);
        //}

        void Polyhedron::calculateInverseJacobian(double r, double s, double t, Matrix<double, 3, 3>& inverse)
        {
            Matrix<double, 3, 3> jacobian;
            getWorldToReferenceJacobian(ElVis::ReferencePoint(r,s,t),jacobian);

            double J[] = {jacobian.getData(0,0), jacobian.getData(0,1), jacobian.getData(0,2),
                jacobian.getData(1,0), jacobian.getData(1,1), jacobian.getData(1,2),
                jacobian.getData(2,0), jacobian.getData(2,1), jacobian.getData(2,2) };

            // Now take the inverse.
            double determinant = (-J[0]*J[4]*J[8]+J[0]*J[5]*J[7]+J[3]*J[1]*J[8]-J[3]*J[2]*J[7]-J[6]*J[1]*J[5]+J[6]*J[2]*J[4]);
            inverse.setData(0,0) = (-J[4]*J[8]+J[5]*J[7])/determinant;
            inverse.setData(0,1) = -(-J[1]*J[8]+J[2]*J[7])/determinant;
            inverse.setData(0,2) = -(J[1]*J[5]-J[2]*J[4])/determinant;
            inverse.setData(1,0) = -(-J[3]*J[8]+J[5]*J[6])/determinant;
            inverse.setData(1,1) = (-J[0]*J[8]+J[2]*J[6])/determinant;
            inverse.setData(1,2) = (J[0]*J[5]-J[2]*J[3])/determinant;
            inverse.setData(2,0) = (-J[3]*J[7]+J[4]*J[6])/determinant;
            inverse.setData(2,1) = (J[0]*J[7]-J[1]*J[6])/determinant;
            inverse.setData(2,2) = -(J[0]*J[4]-J[1]*J[3])/determinant;

        }

        void Polyhedron::transposeAndInvertMatrix(const Matrix<double, 3, 3>& jacobian, Matrix<double, 3, 3>& inverse)
        {
            double J[] = {jacobian.getData(0,0), jacobian.getData(0,1), jacobian.getData(0,2),
                jacobian.getData(1,0), jacobian.getData(1,1), jacobian.getData(1,2),
                jacobian.getData(2,0), jacobian.getData(2,1), jacobian.getData(2,2) };

            double determinant = (-J[0]*J[4]*J[8]+J[0]*J[5]*J[7]+J[3]*J[1]*J[8]-J[3]*J[2]*J[7]-J[6]*J[1]*J[5]+J[6]*J[2]*J[4]);
            inverse.setData(0,0) = (-J[4]*J[8]+J[5]*J[7])/determinant;
            inverse.setData(0,1) = -(-J[1]*J[8]+J[2]*J[7])/determinant;
            inverse.setData(0,2) = -(J[1]*J[5]-J[2]*J[4])/determinant;
            inverse.setData(1,0) = -(-J[3]*J[8]+J[5]*J[6])/determinant;
            inverse.setData(1,1) = (-J[0]*J[8]+J[2]*J[6])/determinant;
            inverse.setData(1,2) = (J[0]*J[5]-J[2]*J[3])/determinant;
            inverse.setData(2,0) = (-J[3]*J[7]+J[4]*J[6])/determinant;
            inverse.setData(2,1) = (J[0]*J[7]-J[1]*J[6])/determinant;
            inverse.setData(2,2) = -(J[0]*J[4]-J[1]*J[3])/determinant;

            inverse.transpose();
            //std::swap(inverse[1], inverse[3]);
            //std::swap(inverse[2], inverse[6]);
            //std::swap(inverse[5], inverse[7]);
        }

        void Polyhedron::calculateWorldToTensorInverseTransposeJacobian(const ElVis::TensorPoint& p, Matrix<double, 3, 3>& result)
        {
            Matrix<double, 3, 3> J;
            calculateTensorToWorldSpaceMappingJacobian(p,J);
            transposeAndInvertMatrix(J, result);
        }


        void Polyhedron::calculateTensorToWorldSpaceMappingMiriahHessian(const ElVis::TensorPoint& p, Matrix<double, 3, 3>& Hr,
            Matrix<double, 3, 3>& Hs, Matrix<double, 3, 3>& Ht)
        {
            Matrix<double, 3, 3> Hx, Hy, Hz;
            calculateTensorToWorldSpaceMappingHessian(p, Hx, Hy, Hz);

            // Now change the ordering for Miriah's convention.
            Hr(0,0) = Hx(0,0);
            Hr(0,1) = Hx(0,1);
            Hr(0,2) = Hx(0,2);
            Hr(1,0) = Hy(0,0);
            Hr(1,1) = Hy(0,1);
            Hr(1,2) = Hy(0,2);
            Hr(2,0) = Hz(0,0);
            Hr(2,1) = Hz(0,1);
            Hr(2,2) = Hz(0,2);

            Hs(0,0) = Hx(1,0);
            Hs(0,1) = Hx(1,1);
            Hs(0,2) = Hx(1,2);
            Hs(1,0) = Hy(1,0);
            Hs(1,1) = Hy(1,1);
            Hs(1,2) = Hy(1,2);
            Hs(2,0) = Hz(1,0);
            Hs(2,1) = Hz(1,1);
            Hs(2,2) = Hz(1,2);

            Ht(0,0) = Hx(2,0);
            Ht(0,1) = Hx(2,1);
            Ht(0,2) = Hx(2,2);
            Ht(1,0) = Hy(2,0);
            Ht(1,1) = Hy(2,1);
            Ht(1,2) = Hy(2,2);
            Ht(2,0) = Hz(2,0);
            Ht(2,1) = Hz(2,1);
            Ht(2,2) = Hz(2,2);
        }

        ElVis::ReferencePoint Polyhedron::transformWorldToReference(const ElVis::WorldPoint& p)
        {
            static int exact = 0;
            static int runs = 0;
            //static int tolerance = 0;
            static int iteration = 0;
            static int adjust = 0;

            //cout << "Exact: " << exact<< endl;
            //cout << "Tolerance: " << tolerance<< endl;
            //cout << "Iteration: " << iteration<< endl;
            //cout << "Adjustment is too small: " << adjust << endl;
            //cout << "Runs: " << runs << endl;
            ++runs;

            // So we first need an initial guess.  We can probably make this smarter, but
            // for now let's go with 0,0,0.
            ElVis::ReferencePoint result(0.0, 0.0, 0.0);
            Matrix<double, 3, 3> inverse;

            int numIterations = 0;
            const int MAX_ITERATIONS = 10000;
            //bool done = false;
            do
            {
                ElVis::WorldPoint f = transformReferenceToWorld(result) - p;
                calculateInverseJacobian(result.r(), result.s(), result.t(), inverse);

                double r_adjust = (inverse[0]*f.x() + inverse[1]*f.y() + inverse[2]*f.z());
                double s_adjust = (inverse[3]*f.x() + inverse[4]*f.y() + inverse[5]*f.z());
                double t_adjust = (inverse[6]*f.x() + inverse[7]*f.y() + inverse[8]*f.z());

                //cout << "Point to transform to reference: " << p << endl;
                //cout << "F: " << f << endl;
                //cout << "Result: " << transformReferenceToWorld(result) << endl;

//                if( fabs(r_adjust) < m_referenceToWorldTolerance &&
//                    fabs(s_adjust) < m_referenceToWorldTolerance &&
//                    fabs(t_adjust) < m_referenceToWorldTolerance )
//                {
//                    ++tolerance;
//                    return result;
//                }

                ElVis::ReferencePoint pointAdjust(r_adjust, s_adjust, t_adjust);

                ElVis::ReferencePoint tempResult = result - pointAdjust;

                // Even if the tempResult isn't 0, it may be too small
                // for the real result.
                if( result == tempResult )
                {
                    ++adjust;
                    //done = true;
                }

                result = tempResult;

                //if( result.r() < -1.0 ) result.r() = -1.0;
                //if( result.r() > 1.0 ) result.r() = 1.0;
                //if( result.s() < -1.0 ) result.s() -1.0;
                //if( result.s() > 1.0 ) result.s() = 1.0;
                //if( result.t() < -1.0 ) result.t() = -1.0;
                //if( result.t() > 1.0 ) result.t() = 1.0;
                // Now check the inverse through interval arithmetic.  If the point
                // we want is in the interval then we are done.

                ElVis::WorldPoint inversePoint = transformReferenceToWorld(result);
                if( p.x() == inversePoint.x() &&
                    p.y() == inversePoint.y() &&
                    p.z() == inversePoint.z()  )
                {
                    ++exact;
                    return result;
                }

                numIterations++;
            }
            while( numIterations < MAX_ITERATIONS);
            ++iteration;
            return result;
        }

        void Polyhedron::assignId()
        {
            m_id = ID_COUNTER;
            ID_COUNTER++;
        }

        void Polyhedron::setMinMax(FILE* inFile, bool reverseBytes)
        {
            int hasMinMaxInFile;

            if( fread(&hasMinMaxInFile, sizeof(int), 1, inFile) != 1 )
            {
                cerr << "Error reading min max." << endl;
                exit(1);
            }

            if( reverseBytes ) JacobiExtension::reverseBytes(hasMinMaxInFile);

            if( hasMinMaxInFile != 0 )
            {
                if( fread(&m_minValue, sizeof(double), 1, inFile) != 1 )
                {
                    cerr << "Error reading minimum value." << endl;
                    exit(1);
                }

                if( fread(&m_maxValue, sizeof(double), 1, inFile) != 1 )
                {
                    cerr << "Error reading maximum value." << endl;
                    exit(1);
                }

                if( reverseBytes )
                {
                    JacobiExtension::reverseBytes(m_minValue);
                    JacobiExtension::reverseBytes(m_maxValue);
                }
            }
            else
            {
                // Read the values anyway, they'll be dummy values so
                // we won't populate our internal min/max data.
                double temp[2];
                if( fread(temp, sizeof(double), 2, inFile) != 2)
                {
                    cerr << "Error reading min/max values." << endl;
                    exit(1);
                }
            }
        }

        void Polyhedron::writeElement(FILE* outFile)
        {
            int type = GetCellType();
            if( fwrite(&type, sizeof(int), 1, outFile) != 1 )
            {
                cerr << "ERROR writing type." << endl;
                return;
            }

            double minValue = this->getMin();
            double maxValue = this->getMax();
            int hasMinMax = 0;
            if( minValue == -DBL_MAX )
            {
                hasMinMax = 1;
            }
            if( fwrite(&hasMinMax, sizeof(int), 1, outFile) != 1 )
            {
                cerr << "ERROR writing min max indicator." << endl;
                return;
            }

            if( fwrite(&minValue, sizeof(double), 1, outFile) != 1 )
            {
                cerr << "ERROR writing min value." << endl;
                return;
            }

            if( fwrite(&maxValue, sizeof(double), 1, outFile) != 1 )
            {
                cerr << "ERROR writing max value." << endl;
                return;
            }

            writeVertices(outFile);
            writeDegree(outFile);
            writeBasisCoefficients(outFile);
        }

        void Polyhedron::writeDegree(FILE* outFile)
        {
            fwrite(m_degree, sizeof(int), 3, outFile);
        }

        void Polyhedron::writeBasisCoefficients(FILE* outFile)
        {
            fwrite(&m_basisCoefficients[0], sizeof(double), m_basisCoefficients.size(), outFile);
        }

        void Polyhedron::readDegree(FILE* inFile, bool reverseBytes)
        {
            if( fread(m_degree, sizeof(int), 3, inFile) != 3 )
            {
                cerr << "Error reading degree." << endl;
                exit(1);
            }

            if( reverseBytes )
            {
                for(int i = 0; i < 3; ++i)
                {
                    JacobiExtension::reverseBytes(m_degree[i]);
                }
            }

        }

        void Polyhedron::readBasisCoefficients(FILE* inFile, int numCoefficients, bool reverseBytes)
        {
            m_basisCoefficients.reserve(numCoefficients);

            for(int i = 0; i < numCoefficients; i++)
            {
                double temp = 0.0;
                if( fread(&temp, sizeof(double), 1, inFile) != 1 )
                {
                    cerr << "Error reading coefficients." << endl;
                    exit(1);
                }
                m_basisCoefficients.push_back(temp);
            }

            if( reverseBytes )
            {
                for(int i = 0; i < numCoefficients; i++)
                {
                    JacobiExtension::reverseBytes(m_basisCoefficients[i]);
                }
            }
            std::vector<double>(m_basisCoefficients).swap(m_basisCoefficients);
        }

        //double Polyhedron::intersectsIsovalueDynamicProjectionAt(const rt::Ray& ray, double isovalue, rt::IntersectionInfo& hit,
        //                                    boost::shared_ptr<FiniteElementVolume> vol)
        //{
        //    return -1.0;

        //    //ElVis::WorldPoint origin(ray.getOrigin().x(),
        //    //  ray.getOrigin().y(), ray.getOrigin().z());

        //    //double actualMin, actualMax;
        //    //if( !findIntersectionWithGeometryInWorldSpace(ray, actualMin, actualMax) )
        //    //{
        //    //  return false;
        //    //}

        //    //int actualDegree = m_interpolatingPolynomialDegree;

        //    //if( interpolatingPolynomialDegreeOverride() > 0 )
        //    //{
        //    //  actualDegree = interpolatingPolynomialDegreeOverride();
        //    //}
        //    //
        //    //std::vector<double> min_values;
        //    //std::vector<double> max_values;
        //    //min_values.push_back(actualMin);
        //    //min_values.push_back(actualMin + (actualMax-actualMin)/2.0);
        //    //max_values.push_back(actualMin + (actualMax-actualMin)/2.0);
        //    //max_values.push_back(actualMax);

        //    //for(size_t minIndex = 0; minIndex < min_values.size(); ++minIndex)
        //    //{
        //    //  double min = min_values[minIndex];
        //    //  double max = max_values[minIndex];

        //    //  // Generate the polynomial projection.
        //    //  boost::function1<double, double> actual_func = boost::bind(
        //    //      &Polyhedron::findScalarValueAtPoint,
        //    //      this, ray, min, max, _1);
        //    //  Polynomial::OrthogonalLegendrePolynomial poly =
        //    //          PolynomialInterpolation::generateLeastSquaresPolynomialProjection<double>(
        //    //              actualDegree, actual_func);
        //    //  poly.reducePolynomialByRemovingSmallLeadingCoefficients(1e-10);
        //    //  // Calculate the L2 Norm by integration.
        //    ////    SubtractFuncs<double> sub(poly, actual_func);
        //    ////    double l2_norm = NumericalIntegration::GaussLegendreQuadrature(
        //    ////        boost::function1<double, double>(sub),
        //    ////        5*actualDegree+3, -1.0, 1.0);
        //    ////    m_projectionL2NormStats.push_back(l2_norm);
        //    ////
        //    ////    // Get a feel for the infinity norm by sampling.
        //    ////    double infinity_norm = -std::numeric_limits<double>::max();
        //    ////    int n = 1000;
        //    ////    double h = 2.0/n;
        //    ////    for(int i = 0; i < n; ++i)
        //    ////    {
        //    ////        double t = -1.0 + h*i;
        //    ////        double real_val = actual_func(t);
        //    ////        double projected_val = poly(t);
        //    ////        double err = fabs(real_val-projected_val);
        //    ////        if( err > infinity_norm )
        //    ////        {
        //    ////            infinity_norm = err;
        //    ////        }
        //    ////    }
        //    ////    m_projectionInfinityNormStats.push_back(infinity_norm);
        //    //
        //    //  Polynomial::Monomial mon_poly = Polynomial::convertToMonomial(poly);
        //    //  //cout << "Desired Isovalue: " << isovalue << endl;
        //    //  //cout << "Coeff 0: " << mon_poly.coeff(0) << endl;
        //    //  mon_poly.coeff(0) = mon_poly.coeff(0) - isovalue;
        //    //  //cout << "Coeff 0: " << mon_poly.coeff(0) << endl;
        //    //  std::vector<double> roots = Polynomial::findRealRoots(mon_poly);
        //    //  size_t numRoots = roots.size();

        //    //  if( numRoots > 0 )
        //    //  {
        //    //      // TODO - We haven't dealt with the camera being inside the element yet.
        //    //      double found_root = DBL_MAX;
        //    //      for(size_t i = 0; i < numRoots; i++)
        //    //      {
        //    //          if( roots[i] >= -1 && roots[i] <= 1 && roots[i] < found_root )
        //    //          {
        //    //              found_root = roots[i];
        //    //          }
        //    //      }

        //    //      if( found_root != DBL_MAX )
        //    //      {
        //    //          double worldT = (found_root+1)/2.0 * (max-min) + min;
        //    //
        //    //          if( hit.testIntersection(vol, (double)worldT) )
        //    //          {
        //    //              FiniteElementVolume* fvol = vol.get();//(FiniteElementVolume*)vol;
        //    //
        //    //              // Check the error between the approximation and the actual
        //    //              // polynomial.
        //    //
        //    //              // Note that the normal is calculated with the intersection in
        //    //              // reference space, not world space.  The interpolation function
        //    //              // is only defined in reference space.
        //    //              ElVis::WorldPoint intersection = origin + findPointAlongVector(ray.getDirection(),worldT);
        //    //              ElVis::TensorPoint intersectionInTensorSpace =
        //    //                  transformWorldToTensor(intersection);

        //    //              ElVis::WorldVector n;
        //    //              calculateNormal(intersectionInTensorSpace, &n);
        //    //              hit.setNormal(n);

        //    //              double val = f(intersectionInTensorSpace);
        //    //
        //    //              double error = fabs(val-isovalue);
        //    //              //double polyVal = poly(found_root);
        //    //              //double polyError = fabs(polyVal-isovalue);

        //    //              fvol->intersectedObject = this;
        //    //              fvol->errorAtIntersection = error;
        //    //              //m_absoluteErrorStats.push_back(error);
        //    //              //m_rootFindingErrorStats.push_back(polyError);
        //    //
        //    //              //static double infinity_norm = -std::numeric_limits<double>::max();
        //    //              //if( error > infinity_norm )
        //    //              //{
        //    //              //  infinity_norm = error;
        //    //              //  //cout << "Found t " << found_root << endl;
        //    //              //  //cout << "World t " << worldT << endl;
        //    //              //  //cout << "Infinity norm " << error << " at element: " << id() << endl;
        //    //              //  //cout << "Poly error: " << polyError << endl;
        //    //              //  //cout << "Origin: " << ray.origin() << endl;
        //    //              //  //cout << "Direction: " << ray.direction() << endl;
        //    //              //  //cout << "Second Point: " << ray.origin() + ray.direction()*100.0 << endl;
        //    //              //}

        //    //              //if( isovalue != 0.0 )
        //    //              //{
        //    //              //  m_relativeErrorStats.push_back(error/fabs(isovalue));
        //    //              //}
        //    //              //else
        //    //              //{
        //    //              //  m_relativeErrorStats.push_back(error);
        //    //              //}
        //    //
        //    //              return true;
        //    //          }
        //    //      }
        //    //  }
        //    //}

        //    //return false;
        //}

        /// \todo Finish this.
        //double Polyhedron::do_findParametricIntersection(const rt::Ray& theRay)
        //{
        //    return -1.0;
        //}

        /// \todo Finish this.
        //ElVis::WorldVector Polyhedron::do_calculateNormal(const rt::IntersectionInfo& intersection)
        //{
        //    return ElVis::WorldVector(0,0,0);
        //}

        void Polyhedron::calculateMinMax() const
        {
            if( m_minValue != -std::numeric_limits<double>::max() &&
                m_maxValue != std::numeric_limits<double>::max() )
            {
                return;
            }

            double tempMin = std::numeric_limits<double>::max();
            double tempMax = -std::numeric_limits<double>::max();

            const unsigned int numDivisions = 10;
            double h = 2.0/numDivisions;

            for(unsigned int i = 0; i < numDivisions; ++i)
            {
                for(unsigned int j = 0; j < numDivisions; ++j)
                {
                    for(unsigned int k = 0; k < numDivisions; ++k)
                    {
                        ElVis::TensorPoint tp(-1.0 + i*h, -1.0 + j*h, -1.0 + k*h);
                        double p = f(tp);

                        if( p < tempMin )
                        {
                            tempMin = p;
                        }

                        if( p > tempMax )
                        {
                            tempMax = p;
                        }
                    }
                }
            }

            m_minValue = tempMin;
            m_maxValue = tempMax;
        }

        double Polyhedron::maxScalarValue() const
        {
            calculateMinMax();
            return m_maxValue;
            //return do_maxScalarValue();
        }

        double Polyhedron::minScalarValue() const
        {
            calculateMinMax();
            return m_minValue;
            //return do_minScalarValue();
        }



    }
}

