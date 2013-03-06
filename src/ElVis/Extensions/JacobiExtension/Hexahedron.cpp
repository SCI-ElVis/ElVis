////////////////////////////////////////////////////////////////////////////////
//
//  File: hoHexahedron.cpp
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
//#include <boost/numeric/interval.hpp>
//typedef boost::numeric::interval<double> IntType;

#include <ElVis/Extensions/JacobiExtension/Hexahedron.h>
#include <ElVis/Extensions/JacobiExtension/FiniteElementMath.h>
#include <ElVis/Extensions/JacobiExtension/EndianConvert.h>

#include <ElVis/Extensions/JacobiExtension/Edge.h>
//#include "Box.hpp"
#include <ElVis/Extensions/JacobiExtension/PointTransformations.hpp>

#include <iostream>
using namespace std;

namespace ElVis
{
    namespace JacobiExtension
    {
        const unsigned int Hexahedron::VerticesForEachFace[] = 
        {0, 1, 2, 3, 
        4, 5, 6, 7,
        3, 2, 6, 7,
        0, 4, 7, 3,
        0, 1, 5, 4, 
        1, 5, 6, 2 };

        const unsigned int Hexahedron::NumEdgesForEachFace[] =
        {4, 4, 4, 4, 4, 4};

        const Edge Hexahedron::Edges[] = 
        {
            Edge(0, 1),
            //Edge(1, 3),
            Edge(1, 2),
            //Edge(3, 2),
            Edge(2, 3),
            //Edge(0, 2),
            Edge(0, 3),
            Edge(0, 4),
            Edge(1, 5),
            //Edge(3, 7),
            Edge(2, 6),
            //Edge(2, 6),
            Edge(3, 7),
            Edge(4, 5),
            //Edge(5, 7),
            Edge(5, 6),
            //Edge(6, 7),
            Edge(7, 6),
            //Edge(4, 6)
            Edge(4, 7)
        };

        // These are not in the order expected by the OriginalNektar code (and 
        // are not used by OriginalNektar code).  These are the order expected 
        // by Nektar++ and are only used for the conversion.
        const Face Hexahedron::Faces[] = 
        {
            Face(0, 1, 2, 3),
            Face(0, 5, 8, 4),
            Face(1, 6, 9, 5),
            Face(2, 6, 10, 7),
            Face(3, 7, 11, 4),
            Face(8, 9, 10, 11)

        };

        Hexahedron::Hexahedron(FILE* inFile, bool reverseBytes) //:
            //  m_intervalScalarField(NULL)
        {
            setMinMax(inFile, reverseBytes);
            readVerticesFromFile(inFile, reverseBytes);
            readDegree(inFile, reverseBytes);

            int numCoefficients = (degree(0)+1) * (degree(1)+1) * (degree(2)+1);

            readBasisCoefficients(inFile, numCoefficients, reverseBytes);

            //setInterpolatingPolynomialDegree(degree(0)+degree(1)+degree(2));
            calculateFaceNormals();

            //m_scalarField = new HexahedralJacobiExpansion<double>(degree(0),
            //  degree(1), degree(2), basisCoefficients());
        }

        void Hexahedron::calculateFaceNormals()
        {
            // Calculate the normals for each face.
            ElVis::WorldVector d1 = createVectorFromPoints(vertex(3), vertex(0));
            ElVis::WorldVector d2 = createVectorFromPoints(vertex(1), vertex(0));
            faceNormals[0] = d1.Cross(d2);

            d1 = createVectorFromPoints(vertex(5), vertex(4));
            d2 = createVectorFromPoints(vertex(7), vertex(4));
            faceNormals[1] = d1.Cross(d2);

            d1 = createVectorFromPoints(vertex(6), vertex(7));
            d2 = createVectorFromPoints(vertex(3), vertex(7));
            faceNormals[2] = d1.Cross(d2);

            d1 = createVectorFromPoints(vertex(7), vertex(4));
            d2 = createVectorFromPoints(vertex(0), vertex(4));
            faceNormals[3] = d1.Cross(d2);

            d1 = createVectorFromPoints(vertex(0), vertex(4));
            d2 = createVectorFromPoints(vertex(5), vertex(4));
            faceNormals[4] = d1.Cross(d2);

            d1 = createVectorFromPoints(vertex(1), vertex(5));
            d2 = createVectorFromPoints(vertex(6), vertex(5));
            faceNormals[5] = d1.Cross(d2);

//            ElVis::WorldVector d1 = createVectorFromPoints(vertex(0), vertex(1));
//            ElVis::WorldVector d2 = createVectorFromPoints(vertex(0), vertex(3));
//            faceNormals[0] = d1.Cross(d2);

//            d1 = createVectorFromPoints(vertex(4), vertex(5));
//            d2 = createVectorFromPoints(vertex(4), vertex(7));
//            faceNormals[1] = -(d1.Cross(d2));

//            d1 = createVectorFromPoints(vertex(7), vertex(6));
//            d2 = createVectorFromPoints(vertex(7), vertex(3));
//            faceNormals[2] = -(d1.Cross(d2));

//            d1 = createVectorFromPoints(vertex(4), vertex(0));
//            d2 = createVectorFromPoints(vertex(4), vertex(7));
//            faceNormals[3] = d1.Cross(d2);

//            d1 = createVectorFromPoints(vertex(4), vertex(5));
//            d2 = createVectorFromPoints(vertex(4), vertex(0));
//            faceNormals[4] = d1.Cross(d2);

//            d1 = createVectorFromPoints(vertex(5), vertex(1));
//            d2 = createVectorFromPoints(vertex(5), vertex(6));
//            faceNormals[5] = -(d1.Cross(d2));

            for(int i = 0; i < 6; i++)
            {
                faceNormals[i].Normalize();
            }

            D[0] = -(faceNormals[0].x()*vertex(0).x() + faceNormals[0].y()*vertex(0).y() +
                faceNormals[0].z()*vertex(0).z());

            D[1] = -(faceNormals[1].x()*vertex(4).x() + faceNormals[1].y()*vertex(4).y() +
                faceNormals[1].z()*vertex(4).z());


            D[2] = -(faceNormals[2].x()*vertex(3).x() + faceNormals[2].y()*vertex(3).y() +
                faceNormals[2].z()*vertex(3).z());

            D[3] = -(faceNormals[3].x()*vertex(4).x() + faceNormals[3].y()*vertex(4).y() +
                faceNormals[3].z()*vertex(4).z());

            D[4] = -(faceNormals[4].x()*vertex(4).x() + faceNormals[4].y()*vertex(4).y() +
                faceNormals[4].z()*vertex(4).z());

            D[5] = -(faceNormals[5].x()*vertex(5).x() + faceNormals[5].y()*vertex(5).y() +
                faceNormals[5].z()*vertex(5).z());
        }


        void Hexahedron::writeElementGeometryForVTK(const char* fileName)
        {
            std::ofstream outFile(fileName, ios::out);
            outFile << "# vtk DataFile Version 2.0" << endl;
            outFile << "Hex." << endl;
            outFile << "ASCII" << endl;
            outFile << "DATASET POLYDATA" << endl;
            outFile << "POINTS " << NUM_HEX_VERTICES << " float" << endl;

            for(int i = 0; i < NUM_HEX_VERTICES; i++)
            {
                outFile << vertex(i).x() << " " <<
                    vertex(i).y() << " " <<
                    vertex(i).z() << endl;
            }

            outFile << "POLYGONS 6 30" << endl;

            outFile << "4 0 1 2 3" << endl;
            outFile << "4 4 5 6 7" << endl;
            outFile << "4 1 5 6 2" << endl;
            outFile << "4 4 0 3 7" << endl;
            outFile << "4 7 6 2 3" << endl;
            outFile << "4 4 5 1 0" << endl;
            outFile.close();
        }

        Hexahedron::~Hexahedron()
        {
            //delete m_scalarField;
            //delete m_intervalScalarField;
        }

        //void Hexahedron::intersectsFacePlane(const rt::Ray& ray, int face, double& min, double& max)
        //{
        //    double t = -1;
        //    if( planeIntersection(ray, faceNormals[face], D[face], t) )
        //    {
        //        if( t > max ) max = t;
        //        if( t < min ) min = t;
        //    }
        //}


        //bool Hexahedron::findIntersectionWithGeometryInWorldSpace(const rt::Ray& ray, double& min, double& max)
        //{
        //    // To intersect the hex, we'll transform all of the
        //    // points to a new coordinate system (u,v,w).  This
        //    // coordinate system is defined by the ray.  The origin of the
        //    // coordinate system is the origin of the ray.
        //    // The basis vectors of the new system is defined by the direction
        //    // of the ray.  We will not be able to specify the coordinate
        //    // system uniquely, but that doesn't matter for our purposes.
        //    //
        //    // The overall goal is to project each point of the element onto
        //    // the uv plane.  Once there we can do the intersection test.

        //    // So the first step is to find the uvw basis vectors.
        //    // w is easy, it is the direction of the ray.

        //    ElVis::WorldVector u,v,w;
        //    generateCoordinateSystemFromVector(ray.getDirection(), u, v, w);

        //    ElVis::WorldPoint o = ray.getOrigin();
        //    ElVis::WorldPoint transformedPoints[NUM_HEX_VERTICES];
        //    for(int i = 0; i < NUM_HEX_VERTICES; i++)
        //    {
        //        double tx = vertex(i).x()-o.x();
        //        double ty = vertex(i).y()-o.y();
        //        double tz = vertex(i).z()-o.z();
        //        transformedPoints[i].SetX(u.x()*tx + u.y()*ty + u.z()*tz);
        //        transformedPoints[i].SetY(v.x()*tx + v.y()*ty + v.z()*tz);

        //        // Don't worry about the w component.  We want to project onto
        //        // the uv plane so we'll set it to 0.
        //    }

        //    // A hex has 12 edges.
        //    Edge e01(transformedPoints[0].x(), transformedPoints[0].y(),
        //        transformedPoints[1].x(), transformedPoints[1].y());
        //    Edge e02(transformedPoints[2].x(), transformedPoints[2].y(),
        //        transformedPoints[1].x(), transformedPoints[1].y());

        //    Edge e03(transformedPoints[2].x(), transformedPoints[2].y(),
        //        transformedPoints[3].x(), transformedPoints[3].y());
        //    Edge e04(transformedPoints[0].x(), transformedPoints[0].y(),
        //        transformedPoints[3].x(), transformedPoints[3].y());


        //    Edge e05(transformedPoints[4].x(), transformedPoints[4].y(),
        //        transformedPoints[5].x(), transformedPoints[5].y());
        //    Edge e06(transformedPoints[5].x(), transformedPoints[5].y(),
        //        transformedPoints[6].x(), transformedPoints[6].y());

        //    Edge e07(transformedPoints[6].x(), transformedPoints[6].y(),
        //        transformedPoints[7].x(), transformedPoints[7].y());
        //    Edge e08(transformedPoints[7].x(), transformedPoints[7].y(),
        //        transformedPoints[4].x(), transformedPoints[4].y());

        //    Edge e09(transformedPoints[6].x(), transformedPoints[6].y(),
        //        transformedPoints[2].x(), transformedPoints[2].y());
        //    Edge e10(transformedPoints[5].x(), transformedPoints[5].y(),
        //        transformedPoints[1].x(), transformedPoints[1].y());

        //    Edge e11(transformedPoints[7].x(), transformedPoints[7].y(),
        //        transformedPoints[3].x(), transformedPoints[3].y());
        //    Edge e12(transformedPoints[0].x(), transformedPoints[0].y(),
        //        transformedPoints[4].x(), transformedPoints[4].y());

        //    min = FLT_MAX;
        //    max = -FLT_MAX;

        //    // Now check each face to see if there is an intersection.  If there
        //    // is then we simply need to find the intersection of the ray with
        //    // the plane defined by the face.
        //    if( (e01.numTimesCrossesPositiveXAxis() + e02.numTimesCrossesPositiveXAxis() +
        //        e03.numTimesCrossesPositiveXAxis() + e04.numTimesCrossesPositiveXAxis()) & 0x01 )
        //    {
        //        intersectsFacePlane(ray, 0, min, max);
        //    }

        //    if( (e05.numTimesCrossesPositiveXAxis() + e06.numTimesCrossesPositiveXAxis() +
        //        e07.numTimesCrossesPositiveXAxis() + e08.numTimesCrossesPositiveXAxis()) & 0x01 )
        //    {
        //        intersectsFacePlane(ray, 1, min, max);
        //    }

        //    if( (e02.numTimesCrossesPositiveXAxis() + e09.numTimesCrossesPositiveXAxis() +
        //        e10.numTimesCrossesPositiveXAxis() + e06.numTimesCrossesPositiveXAxis()) & 0x01 )
        //    {
        //        intersectsFacePlane(ray, 5, min, max);
        //    }

        //    if( (e04.numTimesCrossesPositiveXAxis() + e11.numTimesCrossesPositiveXAxis() +
        //        e12.numTimesCrossesPositiveXAxis() + e08.numTimesCrossesPositiveXAxis()) & 0x01 )
        //    {
        //        intersectsFacePlane(ray, 3, min, max);
        //    }

        //    if( (e09.numTimesCrossesPositiveXAxis() + e03.numTimesCrossesPositiveXAxis() +
        //        e11.numTimesCrossesPositiveXAxis() + e07.numTimesCrossesPositiveXAxis()) & 0x01 )
        //    {
        //        intersectsFacePlane(ray, 2, min, max);
        //    }

        //    if( (e01.numTimesCrossesPositiveXAxis() + e05.numTimesCrossesPositiveXAxis() +
        //        e10.numTimesCrossesPositiveXAxis() + e12.numTimesCrossesPositiveXAxis()) & 0x01 )
        //    {
        //        intersectsFacePlane(ray, 4, min, max);
        //    }

        //    //return min != FLT_MAX && min < max;
        //    return min != FLT_MAX;
        //}

        unsigned int Hexahedron::NumberOfCoefficientsForOrder(unsigned int order)
        {
            return (order+1)*(order+1)*(order+1);
        }

        ElVis::TensorPoint Hexahedron::transformReferenceToTensor(const ElVis::ReferencePoint& p)
        {
            return PointTransformations::transformHexReferenceToTensor(p);
        }

        //IntervalTensorPoint Hexahedron::transformReferenceToTensor(const IntervalReferencePoint& p)
        //{
        //  return PointTransformations::transformHexReferenceToTensor(p);
        //}

        ElVis::ReferencePoint Hexahedron::transformTensorToReference(const ElVis::TensorPoint& p)
        {
            return PointTransformations::transformHexTensorToReference(p);
        }

        //IntervalReferencePoint Hexahedron::transformTensorToReference(const IntervalTensorPoint& p)
        //{
        //  return PointTransformations::transformHexTensorToReference(p);
        //}
        //
        //IntervalWorldPoint Hexahedron::transformReferenceToWorld(const IntervalReferencePoint& p)
        //{
        //  return PointTransformations::transformReferenceToWorld(*this, p);
        //}

        ElVis::WorldPoint Hexahedron::transformReferenceToWorld(const ElVis::ReferencePoint& p)
        {
            return PointTransformations::transformReferenceToWorld(*this, p);
        }

        void Hexahedron::calculateTensorToWorldSpaceMappingHessian(const ElVis::TensorPoint& p, JacobiExtension::Matrix<double, 3, 3>& Hx,
            JacobiExtension::Matrix<double, 3, 3>& Hy, JacobiExtension::Matrix<double, 3, 3>& Hz)
        {
            double c = p.c();
            double b = p.b();
            double a = p.a();

            double v0x = vertex(0).x();
            double v1x = vertex(1).x();
            double v2x = vertex(2).x();
            double v3x = vertex(3).x();
            double v4x = vertex(4).x();
            double v5x = vertex(5).x();
            double v6x = vertex(6).x();
            double v7x = vertex(7).x();

            double v0y = vertex(0).y();
            double v1y = vertex(1).y();
            double v2y = vertex(2).y();
            double v3y = vertex(3).y();
            double v4y = vertex(4).y();
            double v5y = vertex(5).y();
            double v6y = vertex(6).y();
            double v7y = vertex(7).y();

            double v0z = vertex(0).z();
            double v1z = vertex(1).z();
            double v2z = vertex(2).z();
            double v3z = vertex(3).z();
            double v4z = vertex(4).z();
            double v5z = vertex(5).z();
            double v6z = vertex(6).z();
            double v7z = vertex(7).z();

            // X
            Hx(0,0) = 0.0;
            Hx(0,1) = (1.0-c)*v0x/8.0-(1.0-c)*v1x/8.0+(1.0-c)*v2x/8.0-
                (1.0-c)*v3x/8.0+(1.0+c)*v4x/8.0-(1.0+c)*v5x/8.0+
                (1.0+c)*v6x/8.0-(1.0+c)*v7x/8.0;
            Hx(0,2) = (1.0-b)*v0x/8.0-(1.0-b)*v1x/8.0-(1.0+b)*v2x/8.0+
                (1.0+b)*v3x/8.0-(1.0-b)*v4x/8.0+(1.0-b)*v5x/8.0+
                (1.0+b)*v6x/8.0-(1.0+b)*v7x/8.0;

            Hx(1,0) = Hx(0,1);
            Hx(1,1) = 0.0;
            Hx(1,2) = (1.0-a)*v0x/8.0+(1.0+a)*v1x/8.0-(1.0+a)*v2x/8.0-
                (1.0-a)*v3x/8.0-(1.0-a)*v4x/8.0-(1.0+a)*v5x/8.0+
                (1.0+a)*v6x/8.0+(1.0-a)*v7x/8.0;

            Hx(2,0) = Hx(0,2);
            Hx(2,1) = Hx(1,2);
            Hx(2,2) = 0.0;

            // Y
            Hy(0,0) = 0.0;
            Hy(0,1) = (1.0-c)*v0y/8.0-(1.0-c)*v1y/8.0+(1.0-c)*v2y/8.0-
                (1.0-c)*v3y/8.0+(1.0+c)*v4y/8.0-(1.0+c)*v5y/8.0+
                (1.0+c)*v6y/8.0-(1.0+c)*v7y/8.0;
            Hy(0,2) = (1.0-b)*v0y/8.0-(1.0-b)*v1y/8.0-(1.0+b)*v2y/8.0+
                (1.0+b)*v3y/8.0-(1.0-b)*v4y/8.0+(1.0-b)*v5y/8.0+
                (1.0+b)*v6y/8.0-(1.0+b)*v7y/8.0;

            Hy(1,0) = Hy(0,1);
            Hy(1,1) = 0.0;
            Hy(1,2) = (1.0-a)*v0y/8.0+(1.0+a)*v1y/8.0-(1.0+a)*v2y/8.0-
                (1.0-a)*v3y/8.0-(1.0-a)*v4y/8.0-(1.0+a)*v5y/8.0+
                (1.0+a)*v6y/8.0+(1.0-a)*v7y/8.0;

            Hy(2,0) = Hy(0,2);
            Hy(2,1) = Hy(1,2);
            Hy(2,2) = 0.0;

            // Z
            Hz(0,0) = 0.0;
            Hz(0,1) = (1.0-c)*v0z/8.0-(1.0-c)*v1z/8.0+(1.0-c)*v2z/8.0-
                (1.0-c)*v3z/8.0+(1.0+c)*v4z/8.0-(1.0+c)*v5z/8.0+
                (1.0+c)*v6z/8.0-(1.0+c)*v7z/8.0;
            Hz(0,2) = (1.0-b)*v0z/8.0-(1.0-b)*v1z/8.0-(1.0+b)*v2z/8.0+
                (1.0+b)*v3z/8.0-(1.0-b)*v4z/8.0+(1.0-b)*v5z/8.0+
                (1.0+b)*v6z/8.0-(1.0+b)*v7z/8.0;

            Hz(1,0) = Hz(0,1);
            Hz(1,1) = 0.0;
            Hz(1,2) = (1.0-a)*v0z/8.0+(1.0+a)*v1z/8.0-(1.0+a)*v2z/8.0-
                (1.0-a)*v3z/8.0-(1.0-a)*v4z/8.0-(1.0+a)*v5z/8.0+
                (1.0+a)*v6z/8.0+(1.0-a)*v7z/8.0;

            Hz(2,0) = Hz(0,2);
            Hz(2,1) = Hz(1,2);
            Hz(2,2) = 0.0;
        }

        void Hexahedron::calculateTensorToWorldSpaceMappingJacobian(const ElVis::TensorPoint& p, JacobiExtension::Matrix<double, 3, 3>& J)
        {
            double r = p.a();
            double s = p.b();
            double t = p.c();

            double t1 = 1.0-s;
            double t2 = 1.0-t;
            double t3 = t1*t2;
            double t6 = 1.0+s;
            double t7 = t6*t2;
            double t10 = 1.0+t;
            double t11 = t1*t10;
            double t14 = t6*t10;
            double t18 = 1.0-r;
            double t19 = t18*t2;
            double t21 = 1.0+r;
            double t22 = t21*t2;
            double t26 = t18*t10;
            double t28 = t21*t10;
            double t33 = t18*t1;
            double t35 = t21*t1;
            double t37 = t21*t6;
            double t39 = t18*t6;

            double v1x = vertex(0).x();
            double v2x = vertex(1).x();
            double v3x = vertex(2).x();
            double v4x = vertex(3).x();
            double v5x = vertex(4).x();
            double v6x = vertex(5).x();
            double v7x = vertex(6).x();
            double v8x = vertex(7).x();

            double v1y = vertex(0).y();
            double v2y = vertex(1).y();
            double v3y = vertex(2).y();
            double v4y = vertex(3).y();
            double v5y = vertex(4).y();
            double v6y = vertex(5).y();
            double v7y = vertex(6).y();
            double v8y = vertex(7).y();

            double v1z = vertex(0).z();
            double v2z = vertex(1).z();
            double v3z = vertex(2).z();
            double v4z = vertex(3).z();
            double v5z = vertex(4).z();
            double v6z = vertex(5).z();
            double v7z = vertex(6).z();
            double v8z = vertex(7).z();

            J.setData(0,0) = -t3*v1x/8.0+t3*v2x/8.0+t7*v3x/8.0-t7*v4x/8.0-t11*v5x/8.0+t11*
                v6x/8.0+t14*v7x/8.0-t14*v8x/8.0;
            J.setData(0,1) = -t19*v1x/8.0-t22*v2x/8.0+t22*v3x/8.0+t19*v4x/8.0-t26*v5x/8.0-
                t28*v6x/8.0+t28*v7x/8.0+t26*v8x/8.0;
            J.setData(0,2) = -t33*v1x/8.0-t35*v2x/8.0-t37*v3x/8.0-t39*v4x/8.0+t33*v5x/8.0+
                t35*v6x/8.0+t37*v7x/8.0+t39*v8x/8.0;
            J.setData(1,0) = -t3*v1y/8.0+t3*v2y/8.0+t7*v3y/8.0-t7*v4y/8.0-t11*v5y/8.0+t11*
                v6y/8.0+t14*v7y/8.0-t14*v8y/8.0;
            J.setData(1,1) = -t19*v1y/8.0-t22*v2y/8.0+t22*v3y/8.0+t19*v4y/8.0-t26*v5y/8.0-
                t28*v6y/8.0+t28*v7y/8.0+t26*v8y/8.0;
            J.setData(1,2) = -t33*v1y/8.0-t35*v2y/8.0-t37*v3y/8.0-t39*v4y/8.0+t33*v5y/8.0+
                t35*v6y/8.0+t37*v7y/8.0+t39*v8y/8.0;
            J.setData(2,0) = -t3*v1z/8.0+t3*v2z/8.0+t7*v3z/8.0-t7*v4z/8.0-t11*v5z/8.0+t11*
                v6z/8.0+t14*v7z/8.0-t14*v8z/8.0;
            J.setData(2,1) = -t19*v1z/8.0-t22*v2z/8.0+t22*v3z/8.0+t19*v4z/8.0-t26*v5z/8.0-
                t28*v6z/8.0+t28*v7z/8.0+t26*v8z/8.0;
            J.setData(2,2) = -t33*v1z/8.0-t35*v2z/8.0-t37*v3z/8.0-t39*v4z/8.0+t33*v5z/8.0+
                t35*v6z/8.0+t37*v7z/8.0+t39*v8z/8.0;
        }

        ElVis::TensorPoint Hexahedron::transformWorldToTensorCartesianHex(const ElVis::WorldPoint& p)
        {
            double x_low, y_low, z_low, x_high, y_high, z_high;
            elementBounds(x_low, y_low, z_low, x_high, y_high, z_high);

            double r = (2*p.x() - x_high - x_low)/(-x_low+x_high);
            double s = -(2*p.y()-y_low - y_high)/(y_low - y_high);
            double t = -(2*p.z()-z_low-z_high)/(z_low-z_high);
            return ElVis::TensorPoint(r,s,t);
        }

        void Hexahedron::getWorldToReferenceJacobian(const ElVis::ReferencePoint& p, JacobiExtension::Matrix<double, 3, 3>& J)
        {
            ElVis::TensorPoint tp = transformReferenceToTensor(p);
            calculateTensorToWorldSpaceMappingJacobian(tp,J);
        }

        void Hexahedron::outputVertexOrderForVTK(std::ofstream& outFile, int startPoint)
        {
            outFile << startPoint << " " << startPoint+1 << " "
                << startPoint+3 << " " << startPoint+2 << " "
                << startPoint+4 << " " << startPoint+5 << " "
                << startPoint+7 << " " << startPoint+6;
            //outFile << "0 1 3 2 4 5 7 6";
        }

        int Hexahedron::vtkCellType()
        {
            return 11;
        }

        void Hexahedron::calculateMinAndMax() const
        {
            //  cout << "Calculating min and max." << endl;
            //  //if( !minHasBeenSet() )
            //  //{
            //      cout << "Min has not been set." << endl;
            //      // We need to create a new scalar field using interval types.
            //      // We'll do it temporarily since we'll never need it again.
            //      HexahedralJacobiExpansion<IntType> intScalarField(m_scalarField->firstDirectionDegree(),
            //          m_scalarField->secondDirectionDegree(),
            //          m_scalarField->thirdDirectionDegree(),
            //          m_scalarField->basisCoefficients());
            //
            ////        Box<IntType, HexahedralJacobiExpansion<IntType> > box(
            ////            IntType(-1.0, 1.0), IntType(-1.0, 1.0), IntType(-1.0, 1.0),
            ////            intScalarField);
            ////        minValue = box.findMinimumValue();
            //  //}
            //
            //  if( !maxHasBeenSet() )
            //  {
            //  }
        }

        double Hexahedron::performSummation(const std::vector<double>& v1,
            const std::vector<double>& v2, const std::vector<double>& v3) const
        {
            // Precondition stuff.
            assert(v1.size()-1 == degree(0));
            assert(v2.size()-1 == degree(1));
            assert(v3.size()-1 == degree(2));
            boost::function_requires<boost::ConvertibleConcept<double, double> >();

            double result = 0.0;
            Polyhedron::CoefficientListType::const_iterator coeff = basisCoefficients().begin();

            for(unsigned int i = 0; i <= degree(0); ++i)
            {
                for(unsigned int j = 0; j <= degree(1); ++j)
                {
                    for(unsigned int k = 0; k <= degree(2); ++k)
                    {
                        assert(coeff != basisCoefficients().end());
                        result += (*coeff)*v1[i]*v2[j]*v3[k];
                        ++coeff;
                    }
                }
            }
            assert(coeff == basisCoefficients().end());
            return result;
        }

        double Hexahedron::f(const ElVis::TensorPoint& p) const
        {
            std::vector<double> firstDirection(degree(0)+1);
            for(unsigned int i = 0; i <= degree(0); ++i )
            {
                firstDirection[i] = Jacobi::P(i, 0, 0, p.a());
            }

            std::vector<double> secondDirection(degree(1)+1);
            for(unsigned int j = 0; j <= degree(1); ++j )
            {
                secondDirection[j] = Jacobi::P(j, 0, 0, p.b());
            }

            std::vector<double> thirdDirection(degree(2)+1);
            for(unsigned int k = 0; k <= degree(2); ++k )
            {
                thirdDirection[k] = Jacobi::P(k, 0, 0, p.c());
            }

            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Hexahedron::df_da(const ElVis::TensorPoint& p) const
        {
            std::vector<double> firstDirection(degree(0)+1);
            for(unsigned int i = 0; i <= degree(0); ++i )
            {
                firstDirection[i] = Jacobi::dP(i, 0, 0, p.a());
            }

            std::vector<double> secondDirection(degree(1)+1);
            for(unsigned int j = 0; j <= degree(1); ++j )
            {
                secondDirection[j] = Jacobi::P(j, 0, 0, p.b());
            }

            std::vector<double> thirdDirection(degree(2)+1);
            for(unsigned int k = 0; k <= degree(2); ++k )
            {
                thirdDirection[k] = Jacobi::P(k, 0, 0, p.c());
            }
            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Hexahedron::df_db(const ElVis::TensorPoint& p) const
        {
            std::vector<double> firstDirection(degree(0)+1);
            for(unsigned int i = 0; i <= degree(0); ++i )
            {
                firstDirection[i] = Jacobi::P(i, 0, 0, p.a());
            }

            std::vector<double> secondDirection(degree(1)+1);
            for(unsigned int j = 0; j <= degree(1); ++j )
            {
                secondDirection[j] = Jacobi::dP(j, 0, 0, p.b());
            }

            std::vector<double> thirdDirection(degree(2)+1);
            for(unsigned int k = 0; k <= degree(2); ++k )
            {
                thirdDirection[k] = Jacobi::P(k, 0, 0, p.c());
            }
            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Hexahedron::df_dc(const ElVis::TensorPoint& p) const
        {
            std::vector<double> firstDirection(degree(0)+1);
            for(unsigned int i = 0; i <= degree(0); ++i )
            {
                firstDirection[i] = Jacobi::P(i, 0, 0, p.a());
            }

            std::vector<double> secondDirection(degree(1)+1);
            for(unsigned int j = 0; j <= degree(1); ++j )
            {
                secondDirection[j] = Jacobi::P(j, 0, 0, p.b());
            }

            std::vector<double> thirdDirection(degree(2)+1);
            for(unsigned int k = 0; k <= degree(2); ++k )
            {
                thirdDirection[k] = Jacobi::dP(k, 0, 0, p.c());
            }
            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Hexahedron::df2_da2(const ElVis::TensorPoint& p) const
        {
            std::vector<double> firstDirection(degree(0)+1);
            for(unsigned int i = 0; i <= degree(0); ++i )
            {
                firstDirection[i] = Jacobi::ddP(i, 0, 0, p.a());
            }

            std::vector<double> secondDirection(degree(1)+1);
            for(unsigned int j = 0; j <= degree(1); ++j )
            {
                secondDirection[j] = Jacobi::P(j, 0, 0, p.b());
            }

            std::vector<double> thirdDirection(degree(2)+1);
            for(unsigned int k = 0; k <= degree(2); ++k )
            {
                thirdDirection[k] = Jacobi::P(k, 0, 0, p.c());
            }
            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Hexahedron::df2_db2(const ElVis::TensorPoint& p) const
        {
            std::vector<double> firstDirection(degree(0)+1);
            for(unsigned int i = 0; i <= degree(0); ++i )
            {
                firstDirection[i] = Jacobi::P(i, 0, 0, p.a());
            }

            std::vector<double> secondDirection(degree(1)+1);
            for(unsigned int j = 0; j <= degree(1); ++j )
            {
                secondDirection[j] = Jacobi::ddP(j, 0, 0, p.b());
            }

            std::vector<double> thirdDirection(degree(2)+1);
            for(unsigned int k = 0; k <= degree(2); ++k )
            {
                thirdDirection[k] = Jacobi::dP(k, 0, 0, p.c());
            }
            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Hexahedron::df2_dc2(const ElVis::TensorPoint& p) const
        {
            std::vector<double> firstDirection(degree(0)+1);
            for(unsigned int i = 0; i <= degree(0); ++i )
            {
                firstDirection[i] = Jacobi::P(i, 0, 0, p.a());
            }

            std::vector<double> secondDirection(degree(1)+1);
            for(unsigned int j = 0; j <= degree(1); ++j )
            {
                secondDirection[j] = Jacobi::P(j, 0, 0, p.b());
            }

            std::vector<double> thirdDirection(degree(2)+1);
            for(unsigned int k = 0; k <= degree(2); ++k )
            {
                thirdDirection[k] = Jacobi::ddP(k, 0, 0, p.c());
            }
            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Hexahedron::df2_dab(const ElVis::TensorPoint& p) const
        {
            std::vector<double> firstDirection(degree(0)+1);
            for(unsigned int i = 0; i <= degree(0); ++i )
            {
                firstDirection[i] = Jacobi::dP(i, 0, 0, p.a());
            }

            std::vector<double> secondDirection(degree(1)+1);
            for(unsigned int j = 0; j <= degree(1); ++j )
            {
                secondDirection[j] = Jacobi::dP(j, 0, 0, p.b());
            }

            std::vector<double> thirdDirection(degree(2)+1);
            for(unsigned int k = 0; k <= degree(2); ++k )
            {
                thirdDirection[k] = Jacobi::P(k, 0, 0, p.c());
            }
            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Hexahedron::df2_dac(const ElVis::TensorPoint& p) const
        {
            std::vector<double> firstDirection(degree(0)+1);
            for(unsigned int i = 0; i <= degree(0); ++i )
            {
                firstDirection[i] = Jacobi::dP(i, 0, 0, p.a());
            }

            std::vector<double> secondDirection(degree(1)+1);
            for(unsigned int j = 0; j <= degree(1); ++j )
            {
                secondDirection[j] = Jacobi::P(j, 0, 0, p.b());
            }

            std::vector<double> thirdDirection(degree(2)+1);
            for(unsigned int k = 0; k <= degree(2); ++k )
            {
                thirdDirection[k] = Jacobi::dP(k, 0, 0, p.c());
            }
            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Hexahedron::df2_dbc(const ElVis::TensorPoint& p) const
        {
            std::vector<double> firstDirection(degree(0)+1);
            for(unsigned int i = 0; i <= degree(0); ++i )
            {
                firstDirection[i] = Jacobi::P(i, 0, 0, p.a());
            }

            std::vector<double> secondDirection(degree(1)+1);
            for(unsigned int j = 0; j <= degree(1); ++j )
            {
                secondDirection[j] = Jacobi::dP(j, 0, 0, p.b());
            }

            std::vector<double> thirdDirection(degree(2)+1);
            for(unsigned int k = 0; k <= degree(2); ++k )
            {
                thirdDirection[k] = Jacobi::dP(k, 0, 0, p.c());
            }
            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Hexahedron::do_maxScalarValue() const
        {
            return std::numeric_limits<double>::max();
        }

        double Hexahedron::do_minScalarValue() const
        {
            return std::numeric_limits<double>::min();
        }

        unsigned int Hexahedron::DoNumberOfEdges() const
        {
            return 12;
        }

        Edge Hexahedron::DoGetEdge(unsigned int id) const
        {
            return Edges[id];
        }

        unsigned int Hexahedron::DoNumberOfFaces() const
        {
            return 6;
        }

        Face Hexahedron::DoGetFace(unsigned int id) const
        {
            return Faces[id];
        }

    }
}
