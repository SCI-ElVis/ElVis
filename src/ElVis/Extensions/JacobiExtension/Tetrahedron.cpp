////////////////////////////////////////////////////////////////////////////////
//
//  File: hoTetrahedron.cpp
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
#include <ElVis/Extensions/JacobiExtension/Tetrahedron.h>
#include <ElVis/Extensions/JacobiExtension/EndianConvert.h>
#include <ElVis/Extensions/JacobiExtension/FiniteElementMath.h>
#include <ElVis/Extensions/JacobiExtension/Edge.h>
#include <ElVis/Extensions/JacobiExtension/PointTransformations.hpp>
#include <iostream>

using namespace std;

namespace ElVis
{
    namespace JacobiExtension
    {
        //unsigned int Tetrahedron::calcNumCoefficients(unsigned int degree1,
        //            unsigned int degree2, unsigned int degree3)
        //{
        //    return (degree1+1)*(degree2+2)*(degree3+3)/6;
        //}

        //Tetrahedron::Tetrahedron(unsigned int degree1, unsigned int degree2,
        //    unsigned int degree3, const Polyhedron::CoefficientListType& basisCoeffs,
        //    const ElVis::WorldPoint& p0, const ElVis::WorldPoint& p1,
        //    const ElVis::WorldPoint& p2, const ElVis::WorldPoint& p3)
        //{
        //    unsigned int numCoefficients = calcNumCoefficients(degree1, degree2, degree3);
        //    assert(basisCoeffs.size() == numCoefficients);
        //    setDegree(degree1, degree2, degree3);
        //    setBasisCoefficients(basisCoeffs);

        //    setVertex(p0, 0);
        //    setVertex(p1, 1);
        //    setVertex(p2, 2);
        //    setVertex(p3, 3);

        //    setInterpolatingPolynomialDegree(degree1+degree2+degree3);
        //    calculateFaceNormals();
        //    populateWorldToReferenceJacobian();
        //}

        //Tetrahedron::Tetrahedron(FILE* inFile, bool reverseBytes)
        //{
        //    setMinMax(inFile, reverseBytes);
        //    readVerticesFromFile(inFile, reverseBytes);
        //    readDegree(inFile, reverseBytes);

        //    //int numCoefficients = (degree(0)+1)*(degree(0)+2)*(degree(0)+3)/6;
        //    int numCoefficients = calcNumCoefficients(degree(0), degree(1), degree(2));

        //    readBasisCoefficients(inFile, numCoefficients, reverseBytes);

        //    setInterpolatingPolynomialDegree(degree(0)+degree(1)+degree(2));
        //    calculateFaceNormals();
        //    populateWorldToReferenceJacobian();
        //}

        //void Tetrahedron::populateWorldToReferenceJacobian()
        //{
        //    double v1x = vertex(0).x();
        //    double v2x = vertex(1).x();
        //    double v3x = vertex(2).x();
        //    double v4x = vertex(3).x();

        //    double v1y = vertex(0).y();
        //    double v2y = vertex(1).y();
        //    double v3y = vertex(2).y();
        //    double v4y = vertex(3).y();

        //    double v1z = vertex(0).z();
        //    double v2z = vertex(1).z();
        //    double v3z = vertex(2).z();
        //    double v4z = vertex(3).z();

        //    m_worldToReferenceJacobian.setData(0,0) = -v1x/2.0+v2x/2.0;
        //    m_worldToReferenceJacobian.setData(0,1) = -v1x/2.0+v3x/2.0;
        //    m_worldToReferenceJacobian.setData(0,2) = -v1x/2.0+v4x/2.0;
        //    m_worldToReferenceJacobian.setData(1,0) = -v1y/2.0+v2y/2.0;
        //    m_worldToReferenceJacobian.setData(1,1) = -v1y/2.0+v3y/2.0;
        //    m_worldToReferenceJacobian.setData(1,2) = -v1y/2.0+v4y/2.0;
        //    m_worldToReferenceJacobian.setData(2,0) = -v1z/2.0+v2z/2.0;
        //    m_worldToReferenceJacobian.setData(2,1) = -v1z/2.0+v3z/2.0;
        //    m_worldToReferenceJacobian.setData(2,2) = -v1z/2.0+v4z/2.0;

        //}

        //Tetrahedron::~Tetrahedron()
        //{
        //}


        //ElVis::TensorPoint Tetrahedron::transformReferenceToTensor(const ElVis::ReferencePoint& p)
        //{
        //    return PointTransformations::transformTetReferenceToTensor(p);
        //}

        //ElVis::ReferencePoint Tetrahedron::transformTensorToReference(const ElVis::TensorPoint& p)
        //{
        //    return PointTransformations::transformTetTensorToReference(p);
        //}

        //ElVis::WorldPoint Tetrahedron::transformReferenceToWorld(const ElVis::ReferencePoint& p)
        //{
        //    return PointTransformations::transformReferenceToWorld(*this, p);
        //}

        //int Tetrahedron::vtkCellType()
        //{
        //    return 10;
        //}

        //void Tetrahedron::calculateFaceNormals()
        //{
        //    // Calculate the normals for each face.
        //    ElVis::WorldVector d1 = createVectorFromPoints(vertex(2), vertex(0));
        //    ElVis::WorldVector d2 = createVectorFromPoints(vertex(1), vertex(0));
        //    faceNormals[0] = d1.Cross(d2);

        //    d1 = createVectorFromPoints(vertex(1), vertex(0));
        //    d2 = createVectorFromPoints(vertex(3), vertex(0));
        //    faceNormals[1] = d1.Cross(d2);

        //    d1 = createVectorFromPoints(vertex(2), vertex(1));
        //    d2 = createVectorFromPoints(vertex(3), vertex(1));
        //    faceNormals[2] = d1.Cross(d2);

        //    d1 = createVectorFromPoints(vertex(3), vertex(0));
        //    d2 = createVectorFromPoints(vertex(2), vertex(0));
        //    faceNormals[3] = d1.Cross(d2);

        //    for(int i = 0; i < NUM_TET_FACES; i++)
        //    {
        //        faceNormals[i].Normalize();
        //    }

        //    D[0] = -(faceNormals[0].x()*vertex(0).x() + faceNormals[0].y()*vertex(0).y() +
        //        faceNormals[0].z()*vertex(0).z());

        //    D[1] = -(faceNormals[1].x()*vertex(3).x() + faceNormals[1].y()*vertex(3).y() +
        //        faceNormals[1].z()*vertex(3).z());


        //    D[2] = -(faceNormals[2].x()*vertex(3).x() + faceNormals[2].y()*vertex(3).y() +
        //        faceNormals[2].z()*vertex(3).z());

        //    D[3] = -(faceNormals[3].x()*vertex(3).x() + faceNormals[3].y()*vertex(3).y() +
        //        faceNormals[3].z()*vertex(3).z());
        //}

        ////bool Tetrahedron::findIntersectionWithGeometryInWorldSpace(const rt::Ray& ray, double& min, double& max)
        ////{
        ////    ElVis::WorldVector u,v,w;
        ////    generateCoordinateSystemFromVector(ray.getDirection(), u, v, w);

        ////    ElVis::WorldPoint o = ray.getOrigin();
        ////    ElVis::WorldPoint transformedPoints[NUM_TET_VERTICES];
        ////    for(int i = 0; i < NUM_TET_VERTICES; ++i)
        ////    {
        ////        double tx = vertex(i).x()-o.x();
        ////        double ty = vertex(i).y()-o.y();
        ////        double tz = vertex(i).z()-o.z();
        ////        transformedPoints[i].SetX(u.x()*tx + u.y()*ty + u.z()*tz);
        ////        transformedPoints[i].SetY(v.x()*tx + v.y()*ty + v.z()*tz);

        ////        // Don't worry about the w component.  We want to project onto
        ////        // the uv plane so we'll set it to 0.
        ////    }

        ////    Edge e1(transformedPoints[0].x(), transformedPoints[0].y(),
        ////        transformedPoints[1].x(), transformedPoints[1].y());
        ////    Edge e2(transformedPoints[0].x(), transformedPoints[0].y(),
        ////        transformedPoints[3].x(), transformedPoints[3].y());

        ////    Edge e3(transformedPoints[1].x(), transformedPoints[1].y(),
        ////        transformedPoints[3].x(), transformedPoints[3].y());
        ////    Edge e4(transformedPoints[0].x(), transformedPoints[0].y(),
        ////        transformedPoints[2].x(), transformedPoints[2].y());

        ////    Edge e5(transformedPoints[1].x(), transformedPoints[1].y(),
        ////        transformedPoints[2].x(), transformedPoints[2].y());
        ////    Edge e6(transformedPoints[2].x(), transformedPoints[2].y(),
        ////        transformedPoints[3].x(), transformedPoints[3].y());

        ////    min = FLT_MAX;
        ////    max = -FLT_MAX;

        ////    if( (e1.numTimesCrossesPositiveXAxis() +
        ////        e2.numTimesCrossesPositiveXAxis() +
        ////        e3.numTimesCrossesPositiveXAxis()) & 0x01 )
        ////    {
        ////        intersectsFacePlane(ray, 1, min, max);
        ////    }

        ////    if( (e1.numTimesCrossesPositiveXAxis() +
        ////        e4.numTimesCrossesPositiveXAxis() +
        ////        e5.numTimesCrossesPositiveXAxis()) & 0x01 )
        ////    {
        ////        intersectsFacePlane(ray, 0, min, max);
        ////    }

        ////    if( (e3.numTimesCrossesPositiveXAxis() +
        ////        e5.numTimesCrossesPositiveXAxis() +
        ////        e6.numTimesCrossesPositiveXAxis()) & 0x01 )
        ////    {
        ////        intersectsFacePlane(ray, 2, min, max);
        ////    }

        ////    if( (e2.numTimesCrossesPositiveXAxis() +
        ////        e6.numTimesCrossesPositiveXAxis() +
        ////        e4.numTimesCrossesPositiveXAxis()) & 0x01 )
        ////    {
        ////        intersectsFacePlane(ray, 3, min, max);
        ////    }

        ////    // return min != FLT_MAX && min < max;
        ////    return min != FLT_MAX;
        ////}


        ////bool Tetrahedron::intersectsFacePlane(const rt::Ray& ray, int face, double& min, double& max)
        ////{
        ////    double t = -1;
        ////    if( planeIntersection(ray, faceNormals[face], D[face], t) )
        ////    {
        ////        if( t > max ) max = t;
        ////        if( t < min ) min = t;
        ////        return true;
        ////    }
        ////    return false;
        ////}

        //void Tetrahedron::calculateTensorToWorldSpaceMappingHessian(const ElVis::TensorPoint& p, Matrix<double, 3, 3>& Hx,
        //            Matrix<double, 3, 3>& Hy, Matrix<double, 3, 3>& Hz)
        //{
        //    double a = p.a();
        //    double b = p.b();
        //    double c = p.c();

        //    double v0x = vertex(0).x();
        //    double v1x = vertex(1).x();
        //    double v2x = vertex(2).x();

        //    double v0y = vertex(0).y();
        //    double v1y = vertex(1).y();
        //    double v2y = vertex(2).y();

        //    double v0z = vertex(0).z();
        //    double v1z = vertex(1).z();
        //    double v2z = vertex(2).z();

        //    // X
        //    Hx(0,0) = 0.0;
        //    Hx(0,1) = (1.0-c)*v0x/8.0-(1.0-c)*v1x/8.0;
        //    Hx(0,2) = (1.0-b)*v0x/8.0-(1.0-b)*v1x/8.0;

        //    Hx(1,0) = Hx(0,1);
        //    Hx(1,1) = 0.0;
        //    Hx(1,2) = -(-1.0/4.0+a/4.0)*v0x/2.0+(1.0+a)*v1x/8.0-v2x/4.0;

        //    Hx(2,0) = Hx(0,2);
        //    Hx(2,1) = Hx(1,2);
        //    Hx(2,2) = 0.0;


        //    // Y
        //    Hy(0,0) = 0.0;
        //    Hy(0,1) = (1.0-c)*v0y/8.0-(1.0-c)*v1y/8.0;
        //    Hy(0,2) = (1.0-b)*v0y/8.0-(1.0-b)*v1y/8.0;

        //    Hy(1,0) = Hy(0,1);
        //    Hy(1,1) = 0.0;
        //    Hy(1,2) = -(-1.0/4.0+a/4.0)*v0y/2.0+(1.0+a)*v1y/8.0-v2y/4.0;

        //    Hy(2,0) = Hy(0,2);
        //    Hy(2,1) = Hy(1,2);
        //    Hy(2,2) = 0.0;



        //    // Z
        //    Hz(0,0) = 0.0;
        //    Hz(0,1) = (1.0-c)*v0z/8.0-(1.0-c)*v1z/8.0;
        //    Hz(0,2) = (1.0-b)*v0z/8.0-(1.0-b)*v1z/8.0;

        //    Hz(1,0) = Hz(0,1);
        //    Hz(1,1) = 0.0;
        //    Hz(1,2) = -(-1.0/4.0+a/4.0)*v0z/2.0+(1.0+a)*v1z/8.0-v2z/4.0;

        //    Hz(2,0) = Hz(0,2);
        //    Hz(2,1) = Hz(1,2);
        //    Hz(2,2) = 0.0;
        //}

        //void Tetrahedron::calculateTensorToWorldSpaceMappingJacobian(const ElVis::TensorPoint& p, Matrix<double, 3, 3>& J)
        //{
        //    double a = p.a();
        //    double b = p.b();
        //    double c = p.c();
        //    double v1x = vertex(0).x();
        //    double v2x = vertex(1).x();
        //    double v3x = vertex(2).x();
        //    double v4x = vertex(3).x();

        //    double v1y = vertex(0).y();
        //    double v2y = vertex(1).y();
        //    double v3y = vertex(2).y();
        //    double v4y = vertex(3).y();

        //    double v1z = vertex(0).z();
        //    double v2z = vertex(1).z();
        //    double v3z = vertex(2).z();
        //    double v4z = vertex(3).z();

        //    double t1 = 1.0-b;
        //    double t2 = 1.0-c;
        //    double t3 = t1*t2;
        //    double t7 = 1.0+a;
        //    double t8 = t7*t2;
        //    double t9 = t8-1.0+c;
        //    double t16 = t7*t1;
        //    double t19 = t16/8.0-3.0/8.0+b/8.0;
        //    double t23 = 1.0+b;
        //    J.setData(0,0) = -t3*v1x/8.0+t3*v2x/8.0;
        //    J.setData(0,1) = t9*v1x/8.0-t8*v2x/8.0+t2*v3x/8.0;
        //    J.setData(0,2) = t19*v1x-t16*v2x/8.0-t23*v3x/8.0+v4x/2.0;
        //    J.setData(1,0) = -t3*v1y/8.0+t3*v2y/8.0;
        //    J.setData(1,1) = t9*v1y/8.0-t8*v2y/8.0+t2*v3y/8.0;
        //    J.setData(1,2) = t19*v1y-t16*v2y/8.0-t23*v3y/8.0+v4y/2.0;
        //    J.setData(2,0) = -t3*v1z/8.0+t3*v2z/8.0;
        //    J.setData(2,1) = t9*v1z/8.0-t8*v2z/8.0+t2*v3z/8.0;
        //    J.setData(2,2) = t19*v1z-t16*v2z/8.0-t23*v3z/8.0+v4z/2.0;

        //}

        //void Tetrahedron::getWorldToReferenceJacobian(const ElVis::ReferencePoint&, Matrix<double, 3, 3>& J)
        //{
        //    J = m_worldToReferenceJacobian;
        //    //std::copy(&m_worldToReferenceJacobian[0],
        //    //  &m_worldToReferenceJacobian[9],
        //    //  &J[0]);
        //}


        //void Tetrahedron::writeElement(FILE* outFile)
        //{
        //}


        //void Tetrahedron::writeElementGeometryForVTK(const char* fileName)
        //{
        //}

        //void Tetrahedron::writeElementForVTKAsCell(const char* fileName)
        //{
        //}

        //void Tetrahedron::outputVertexOrderForVTK(ofstream& outFile,int startPoint)
        //{
        //    outFile << startPoint << " " << startPoint+1 << " "
        //        << startPoint+2 << " " << startPoint+3;
        //}


        //double Tetrahedron::f(const ElVis::TensorPoint& p) const
        //{
        //    double result = 0.0;
        //    std::vector<double>::const_iterator coeff = basisCoefficients().begin();

        //    for(int i = 0; i <= static_cast<int>(degree(0)); ++i)
        //    {
        //        double val1 = firstComponent(i, p.a());
        //        for(int j = 0; j <= static_cast<int>(degree(1))-i; ++j)
        //        {
        //            double val2 = secondComponent(i, j, p.b());

        //            for(int k = 0; k <= (static_cast<int>(degree(2))-i)-j; ++k)
        //            {
        //                double val3 = thirdComponent(i, j, k, p.c());
        //                result += (*coeff) * val1 * val2 * val3;
        //                ++coeff;
        //            }
        //        }
        //    }

        //    assert(coeff == basisCoefficients().end());
        //    return result;
        //}

        //double Tetrahedron::df_da(const ElVis::TensorPoint& p) const
        //{
        //    double result = 0.0;
        //    std::vector<double>::const_iterator coeff = basisCoefficients().begin();

        //    for(int i = 0; i <= static_cast<int>(degree(0)); ++i)
        //    {
        //        double val1 = d_firstComponent(i, p.a());
        //        for(int j = 0; j <= static_cast<int>(degree(1))-i; ++j)
        //        {
        //            double val2 = secondComponent(i, j, p.b());

        //            for(int k = 0; k <= (static_cast<int>(degree(2))-i)-j; ++k)
        //            {
        //                double val3 = thirdComponent(i, j, k, p.c());
        //                result += (*coeff) * val1 * val2 * val3;
        //                ++coeff;
        //            }
        //        }
        //    }

        //    assert(coeff == basisCoefficients().end());
        //    return result;
        //}

        //double Tetrahedron::df_db(const ElVis::TensorPoint& p) const
        //{
        //    double result = 0.0;
        //    std::vector<double>::const_iterator coeff = basisCoefficients().begin();

        //    for(int i = 0; i <= static_cast<int>(degree(0)); ++i)
        //    {
        //        double val1 = firstComponent(i, p.a());
        //        for(int j = 0; j <= static_cast<int>(degree(1))-i; ++j)
        //        {
        //            double val2 = d_secondComponent(i, j, p.b());

        //            for(int k = 0; k <= (static_cast<int>(degree(2))-i)-j; ++k)
        //            {
        //                double val3 = thirdComponent(i, j, k, p.c());
        //                result += (*coeff) * val1 * val2 * val3;
        //                ++coeff;
        //            }
        //        }
        //    }

        //    assert(coeff == basisCoefficients().end());
        //    return result;
        //}

        //double Tetrahedron::df_dc(const ElVis::TensorPoint& p) const
        //{
        //    double result = 0.0;
        //    std::vector<double>::const_iterator coeff = basisCoefficients().begin();

        //    for(int i = 0; i <= static_cast<int>(degree(0)); ++i)
        //    {
        //        double val1 = firstComponent(i, p.a());
        //        for(int j = 0; j <= static_cast<int>(degree(1))-i; ++j)
        //        {
        //            double val2 = secondComponent(i, j, p.b());

        //            for(int k = 0; k <= (static_cast<int>(degree(2))-i)-j; ++k)
        //            {
        //                double val3 = d_thirdComponent(i, j, k, p.c());
        //                result += (*coeff) * val1 * val2 * val3;
        //                ++coeff;
        //            }
        //        }
        //    }

        //    assert(coeff == basisCoefficients().end());
        //    return result;
        //}


        //double Tetrahedron::firstComponent(unsigned int i, const double& val) const
        //{
        //    return Jacobi::P(i, 0, 0, val);
        //}

        //double Tetrahedron::d_firstComponent(unsigned int i, const double& val) const
        //{
        //    return Jacobi::dP(i, 0, 0, val);
        //}

        //double Tetrahedron::dd_firstComponent(unsigned int i, const double& val) const
        //{
        //    return Jacobi::ddP(i, 0, 0, val);
        //}

        //double Tetrahedron::secondComponent(unsigned int i, unsigned int j, const double& val) const
        //{
        //    return pow( 1.0-val, (double)i) * Jacobi::P(j, 2*i+1, 0, val);
        //}

        //double Tetrahedron::d_secondComponent(unsigned int i, unsigned int j, const double& val) const
        //{
        //    if( i == 0 )
        //    {
        //        return Jacobi::dP(j, 2*i+1, 0, val);
        //    }
        //    else
        //    {
        //        return pow( 1.0-val, static_cast<double>(i)) * Jacobi::dP(j, 2*i+1, 0, val) +
        //            -1.0*static_cast<double>(i)*
        //            pow(1.0-val, static_cast<double>(i)-1.0)*
        //            Jacobi::P(j, 2*i+1, 0, val);
        //    }
        //}

        //double Tetrahedron::dd_secondComponent(unsigned int i, unsigned int j, const double& val) const
        //{
        //    int a = 2*i+1;
        //    if( i == 0 )
        //    {
        //        return  Jacobi::ddP(j, a, 0, val);
        //    }
        //    else if( i == 1 )
        //    {
        //        return  2.0*pow(1.0-val,1.0*i-1.0)*i*Jacobi::dP(j, a, 0, val) +
        //            pow(1.0-val,1.0*i)*Jacobi::ddP(j, a, 0, val);
        //    }
        //    else
        //    {
        //        return  pow(1.0-val,1.0*i-2.0)*i*i*Jacobi::P(j, a, 0, val) -
        //            pow(1.0-val,1.0*i-2.0)*i*Jacobi::P(j, a, 0, val) -
        //            2.0*pow(1.0-val,1.0*i-1.0)*i*Jacobi::dP(j, a, 0, val) +
        //            pow(1.0-val,1.0*i)*Jacobi::ddP(j, a, 0, val);
        //    }
        //}

        //double Tetrahedron::thirdComponent(unsigned int i, unsigned int j, unsigned int k, const double& val) const
        //{
        //    return pow( 1.0-val, (double)(i+j)) * Jacobi::P(k, 2*i+2*j+2, 0, val);
        //}

        //double Tetrahedron::d_thirdComponent(unsigned int i, unsigned int j, unsigned int k, const double& val) const
        //{
        //    if( i+j == 0 )
        //    {
        //        return Jacobi::dP(k, 2*i+2*j+2, 0, val);
        //    }
        //    else
        //    {
        //        return pow( 1.0-val, static_cast<double>(i+j)) * Jacobi::dP(k, 2*i+2*j+2, 0, val) +
        //            -1.0*static_cast<double>(i+j)*
        //            pow(1.0-val, static_cast<double>(i+j)-1.0) *
        //            Jacobi::P(k, 2*i+2*j+2, 0, val);
        //    }
        //}

        //double Tetrahedron::dd_thirdComponent(unsigned int i, unsigned int j, unsigned int k, const double& val) const
        //{
        //    int a = 2*i+2*j+1;

        //    if( i+j == 0 )
        //    {
        //        return Jacobi::ddP(k, 2*i+2*j+2, 0, val);
        //    }
        //    else if( i+j == 1 )
        //    {
        //        return 2.0*pow(1.0-val,1.0*i+1.0*j-1.0)*(i+j)*Jacobi::dP(k, a, 0, val) +
        //            pow(1.0-val,1.0*i+1.0*j)*Jacobi::ddP(k, a, 0, val);
        //    }
        //    else
        //    {
        //        return pow(1.0-val,1.0*i+1.0*j-2.0)*pow(i+j,2.0)*Jacobi::P(k, a, 0, val) -
        //            pow(1.0-val,1.0*i+1.0*j-2.0)*(i+j)*Jacobi::P(k, a, 0, val) -
        //            2.0*pow(1.0-val,1.0*i+1.0*j-1.0)*(i+j)*Jacobi::dP(k, a, 0, val) +
        //            pow(1.0-val,1.0*i+1.0*j)*Jacobi::ddP(k, a, 0, val);
        //    }
        //}

        //double Tetrahedron::df2_da2(const ElVis::TensorPoint& p) const
        //{
        //    double result = 0.0;
        //    std::vector<double>::const_iterator coeff = basisCoefficients().begin();

        //    for(int i = 0; i <= static_cast<int>(degree(0)); ++i)
        //    {
        //        double val1 = dd_firstComponent(i, p.a());
        //        for(int j = 0; j <= static_cast<int>(degree(1))-i; ++j)
        //        {
        //            double val2 = secondComponent(i, j, p.b());

        //            for(int k = 0; k <= (static_cast<int>(degree(2))-i)-j; ++k)
        //            {
        //                double val3 = thirdComponent(i, j, k, p.c());
        //                result += (*coeff) * val1 * val2 * val3;
        //                ++coeff;
        //            }
        //        }
        //    }

        //    assert(coeff == basisCoefficients().end());
        //    return result;
        //}

        //double Tetrahedron::df2_db2(const ElVis::TensorPoint& p) const
        //{
        //    double result = 0.0;
        //    std::vector<double>::const_iterator coeff = basisCoefficients().begin();

        //    for(int i = 0; i <= static_cast<int>(degree(0)); ++i)
        //    {
        //        double val1 = firstComponent(i, p.a());
        //        for(int j = 0; j <= static_cast<int>(degree(1))-i; ++j)
        //        {
        //            double val2 = dd_secondComponent(i, j, p.b());

        //            for(int k = 0; k <= (static_cast<int>(degree(2))-i)-j; ++k)
        //            {
        //                double val3 = thirdComponent(i, j, k, p.c());
        //                result += (*coeff) * val1 * val2 * val3;
        //                ++coeff;
        //            }
        //        }
        //    }

        //    assert(coeff == basisCoefficients().end());
        //    return result;
        //}

        //double Tetrahedron::df2_dc2(const ElVis::TensorPoint& p) const
        //{
        //    double result = 0.0;
        //    std::vector<double>::const_iterator coeff = basisCoefficients().begin();

        //    for(int i = 0; i <= static_cast<int>(degree(0)); ++i)
        //    {
        //        double val1 = firstComponent(i, p.a());
        //        for(int j = 0; j <= static_cast<int>(degree(1))-i; ++j)
        //        {
        //            double val2 = secondComponent(i, j, p.b());

        //            for(int k = 0; k <= (static_cast<int>(degree(2))-i)-j; ++k)
        //            {
        //                double val3 = dd_thirdComponent(i, j, k, p.c());
        //                result += (*coeff) * val1 * val2 * val3;
        //                ++coeff;
        //            }
        //        }
        //    }

        //    assert(coeff == basisCoefficients().end());
        //    return result;
        //}

        //double Tetrahedron::df2_dab(const ElVis::TensorPoint& p) const
        //{
        //    double result = 0.0;
        //    std::vector<double>::const_iterator coeff = basisCoefficients().begin();

        //    for(int i = 0; i <= static_cast<int>(degree(0)); ++i)
        //    {
        //        double val1 = d_firstComponent(i, p.a());
        //        for(int j = 0; j <= static_cast<int>(degree(1))-i; ++j)
        //        {
        //            double val2 = d_secondComponent(i, j, p.b());

        //            for(int k = 0; k <= (static_cast<int>(degree(2))-i)-j; ++k)
        //            {
        //                double val3 = thirdComponent(i, j, k, p.c());
        //                result += (*coeff) * val1 * val2 * val3;
        //                ++coeff;
        //            }
        //        }
        //    }

        //    assert(coeff == basisCoefficients().end());
        //    return result;
        //}

        //double Tetrahedron::df2_dac(const ElVis::TensorPoint& p) const
        //{
        //    double result = 0.0;
        //    std::vector<double>::const_iterator coeff = basisCoefficients().begin();

        //    for(int i = 0; i <= static_cast<int>(degree(0)); ++i)
        //    {
        //        double val1 = d_firstComponent(i, p.a());
        //        for(int j = 0; j <= static_cast<int>(degree(1))-i; ++j)
        //        {
        //            double val2 = secondComponent(i, j, p.b());

        //            for(int k = 0; k <= (static_cast<int>(degree(2))-i)-j; ++k)
        //            {
        //                double val3 = d_thirdComponent(i, j, k, p.c());
        //                result += (*coeff) * val1 * val2 * val3;
        //                ++coeff;
        //            }
        //        }
        //    }

        //    assert(coeff == basisCoefficients().end());
        //    return result;
        //}

        //double Tetrahedron::df2_dbc(const ElVis::TensorPoint& p) const
        //{
        //    double result = 0.0;
        //    std::vector<double>::const_iterator coeff = basisCoefficients().begin();

        //    for(int i = 0; i <= static_cast<int>(degree(0)); ++i)
        //    {
        //        double val1 = firstComponent(i, p.a());
        //        for(int j = 0; j <= static_cast<int>(degree(1))-i; ++j)
        //        {
        //            double val2 = d_secondComponent(i, j, p.b());

        //            for(int k = 0; k <= (static_cast<int>(degree(2))-i)-j; ++k)
        //            {
        //                double val3 = d_thirdComponent(i, j, k, p.c());
        //                result += (*coeff) * val1 * val2 * val3;
        //                ++coeff;
        //            }
        //        }
        //    }

        //    assert(coeff == basisCoefficients().end());
        //    return result;
        //}

        //double Tetrahedron::do_maxScalarValue() const
        //{
        //    return std::numeric_limits<double>::max();
        //}

        //double Tetrahedron::do_minScalarValue() const
        //{
        //    return std::numeric_limits<double>::min();
        //}
    }
}