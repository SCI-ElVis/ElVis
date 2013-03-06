////////////////////////////////////////////////////////////////////////////////
//
//  File: hoPrism.hpp
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
#include <ElVis/Extensions/JacobiExtension/Prism.h>
#include <ElVis/Extensions/JacobiExtension/EndianConvert.h>
#include <ElVis/Extensions/JacobiExtension/FiniteElementMath.h>
#include <ElVis/Extensions/JacobiExtension/Edge.h>
#include <ElVis/Extensions/JacobiExtension/PointTransformations.hpp>
//#include "PrismJacobiExpansion.hpp"

#include <iostream>
using namespace std;

namespace ElVis
{
    namespace JacobiExtension
    {
        const unsigned int Prism::VerticesForEachFace[] = 
        {0, 1, 2, 3, 
        1, 2, 5, 4,
        0, 3, 5, 4,
        0, 1, 4, 4,
        3, 2, 5, 5};

        const unsigned int Prism::NumEdgesForEachFace[] =
        {4, 4, 4, 3, 3};

        const Edge Prism::Edges[] =
        {
            Edge(0, 1),
            Edge(1, 2),
            Edge(2, 3),
            Edge(3, 0),
            Edge(2, 5),
            Edge(5, 4),
            Edge(1, 4),
            Edge(0, 4),
            Edge(5, 3)
        };

        const Face Prism::Faces[] =
        {
            Face(0, 3, 2, 1),
            Face(1, 4, 5, 6),
            Face(8, 3, 7, 5),
            Face(0, 6, 7),
            Face(8, 4, 2)
        };

        unsigned int Prism::DoNumberOfEdges() const
        {
            return 9;
        }

        Edge Prism::DoGetEdge(unsigned int id) const
        {
            return Edges[id];
        }
        unsigned int Prism::DoNumberOfFaces() const
        {
            return 5;
        }
        Face Prism::DoGetFace(unsigned int id) const
        {
            return Faces[id];
        }

        Prism::Prism(FILE* inFile, bool reverseBytes) //:
            //  m_intervalScalarField(NULL)
        {
            setMinMax(inFile, reverseBytes);
            readVerticesFromFile(inFile, reverseBytes);
            readDegree(inFile, reverseBytes);

            int numCoefficients = (degree(0)+1)*(degree(1)+1)*(degree(2)+2)/2;
            readBasisCoefficients(inFile, numCoefficients, reverseBytes);

            //setInterpolatingPolynomialDegree(2*degree(0)+degree(1)+degree(2));

            calculateFaceNormals();

            //  m_scalarField = new PrismJacobiExpansion<double>(degree(0),
            //      degree(1), degree(2), basisCoefficients());
        }

        Prism::~Prism()
        {
            //delete m_scalarField;
            //delete m_intervalScalarField;
        }




        void Prism::calculateFaceNormals()
        {
            // Inward facing normals.
            ElVis::WorldVector d1 = createVectorFromPoints(vertex(3), vertex(0));
            ElVis::WorldVector d2 = createVectorFromPoints(vertex(1), vertex(0));
            faceNormals[0] = d1.Cross(d2);

            d2 = createVectorFromPoints(vertex(5), vertex(4));
            d1 = createVectorFromPoints(vertex(1), vertex(4));
            faceNormals[1] = d1.Cross(d2);

            d2 = createVectorFromPoints(vertex(4), vertex(5));
            d1 = createVectorFromPoints(vertex(3), vertex(5));
            faceNormals[2] = d1.Cross(d2);

            d2 = createVectorFromPoints(vertex(1), vertex(4));
            d1 = createVectorFromPoints(vertex(0), vertex(4));
            faceNormals[3] = d1.Cross(d2);

            d2 = createVectorFromPoints(vertex(2), vertex(3));
            d1 = createVectorFromPoints(vertex(5), vertex(3));
            faceNormals[4] = d1.Cross(d2);

//            ElVis::WorldVector d1 = createVectorFromPoints(vertex(0), vertex(1));
//            ElVis::WorldVector d2 = createVectorFromPoints(vertex(0), vertex(3));
//            faceNormals[0] = -(d1.Cross(d2));

//            d2 = createVectorFromPoints(vertex(2), vertex(5));
//            d1 = createVectorFromPoints(vertex(2), vertex(1));
//            faceNormals[1] = d1.Cross(d2);

//            d2 = createVectorFromPoints(vertex(3), vertex(5));
//            d1 = createVectorFromPoints(vertex(3), vertex(0));
//            faceNormals[2] = -(d1.Cross(d2));

//            d2 = createVectorFromPoints(vertex(0), vertex(1));
//            d1 = createVectorFromPoints(vertex(0), vertex(4));
//            faceNormals[3] = d1.Cross(d2);

//            d2 = createVectorFromPoints(vertex(3), vertex(2));
//            d1 = createVectorFromPoints(vertex(3), vertex(5));
//            faceNormals[4] = -(d1.Cross(d2));

            for(int i = 0; i < NUM_PRISM_FACES; i++)
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

            D[4] = -(faceNormals[4].x()*vertex(5).x() + faceNormals[4].y()*vertex(5).y() +
                faceNormals[4].z()*vertex(5).z());

        }


        //IntervalTensorPoint Prism::transformReferenceToTensor(const IntervalReferencePoint& p)
        //{
        //  return PointTransformations::transformPrismReferenceToTensor(p);
        //}

        ElVis::TensorPoint Prism::transformReferenceToTensor(const ElVis::ReferencePoint& p)
        {
            return PointTransformations::transformPrismReferenceToTensor(p);
        }

        //IntervalReferencePoint Prism::transformTensorToReference(const IntervalTensorPoint& p)
        //{
        //  return PointTransformations::transformPrismTensorToReference(p);
        //}

        ElVis::ReferencePoint Prism::transformTensorToReference(const ElVis::TensorPoint& p)
        {
            return PointTransformations::transformPrismTensorToReference(p);
        }

        ElVis::WorldPoint Prism::transformReferenceToWorld(const ElVis::ReferencePoint& p)
        {
            return PointTransformations::transformReferenceToWorld(*this, p);
        }


        //bool Prism::findIntersectionWithGeometryInWorldSpace(const rt::Ray& ray, double& min, double& max)
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
        //    ElVis::WorldPoint transformedPoints[NUM_PRISM_VERTICES];
        //    for(int i = 0; i < NUM_PRISM_VERTICES; i++)
        //    {
        //        double tx = vertex(i).x()-o.x();
        //        double ty = vertex(i).y()-o.y();
        //        double tz = vertex(i).z()-o.z();
        //        transformedPoints[i].SetX(u.x()*tx + u.y()*ty + u.z()*tz);
        //        transformedPoints[i].SetY(v.x()*tx + v.y()*ty + v.z()*tz);

        //        // Don't worry about the w component.  We want to project onto
        //        // the uv plane so we'll set it to 0.
        //    }

        //    // A Prism has 9 edges.
        //    Edge e01(transformedPoints[0], transformedPoints[1]);
        //    Edge e02(transformedPoints[1], transformedPoints[4]);
        //    Edge e03(transformedPoints[0], transformedPoints[4]);
        //    Edge e04(transformedPoints[5], transformedPoints[4]);
        //    Edge e05(transformedPoints[0], transformedPoints[3]);
        //    Edge e06(transformedPoints[1], transformedPoints[2]);
        //    Edge e07(transformedPoints[2], transformedPoints[3]);
        //    Edge e08(transformedPoints[5], transformedPoints[2]);
        //    Edge e09(transformedPoints[5], transformedPoints[3]);

        //    min = FLT_MAX;
        //    max = -FLT_MAX;

        //    // Now check each face to see if there is an intersection.  If there
        //    // is then we simply need to find the intersection of the ray with
        //    // the plane defined by the face.
        //    if( (e01.numTimesCrossesPositiveXAxis() + e02.numTimesCrossesPositiveXAxis() +
        //        e03.numTimesCrossesPositiveXAxis()) & 0x01 )
        //    {
        //        intersectsFacePlane(ray, 3, min, max);
        //    }

        //    if( (e09.numTimesCrossesPositiveXAxis() +
        //        e07.numTimesCrossesPositiveXAxis() + e08.numTimesCrossesPositiveXAxis()) & 0x01 )
        //    {
        //        intersectsFacePlane(ray, 4, min, max);
        //    }

        //    if( (e03.numTimesCrossesPositiveXAxis() + e04.numTimesCrossesPositiveXAxis() +
        //        e09.numTimesCrossesPositiveXAxis() + e05.numTimesCrossesPositiveXAxis()) & 0x01 )
        //    {
        //        intersectsFacePlane(ray, 2, min, max);
        //    }

        //    if( (e05.numTimesCrossesPositiveXAxis() + e07.numTimesCrossesPositiveXAxis() +
        //        e01.numTimesCrossesPositiveXAxis() + e06.numTimesCrossesPositiveXAxis()) & 0x01 )
        //    {
        //        intersectsFacePlane(ray, 0, min, max);
        //    }

        //    if( (e04.numTimesCrossesPositiveXAxis() + e08.numTimesCrossesPositiveXAxis() +
        //        e02.numTimesCrossesPositiveXAxis() + e06.numTimesCrossesPositiveXAxis()) & 0x01 )
        //    {
        //        intersectsFacePlane(ray, 1, min, max);
        //    }

        //    //return min != FLT_MAX && min < max;
        //    return min != FLT_MAX;

        //}

        //void Prism::intersectsFacePlane(const rt::Ray& ray, int face, double& min, double& max)
        //{
        //    double t = -1;
        //    if( planeIntersection(ray, faceNormals[face], D[face], t) )
        //    {
        //        if( t > max ) max = t;
        //        if( t < min ) min = t;
        //    }
        //}

        void Prism::calculateTensorToWorldSpaceMappingHessian(const ElVis::TensorPoint& p, JacobiExtension::Matrix<double, 3, 3>& Hx,
            JacobiExtension::Matrix<double, 3, 3>& Hy, JacobiExtension::Matrix<double, 3, 3>& Hz)
        {
            double a = p.a();
            double b = p.b();
            double c = p.c();

            double v0x = vertex(0).x();
            double v1x = vertex(1).x();
            double v2x = vertex(2).x();
            double v3x = vertex(3).x();
            double v4x = vertex(4).x();
            double v5x = vertex(5).x();

            double v0y = vertex(0).y();
            double v1y = vertex(1).y();
            double v2y = vertex(2).y();
            double v3y = vertex(3).y();
            double v4y = vertex(4).y();
            double v5y = vertex(5).y();

            double v0z = vertex(0).z();
            double v1z = vertex(1).z();
            double v2z = vertex(2).z();
            double v3z = vertex(3).z();
            double v4z = vertex(4).z();
            double v5z = vertex(5).z();

            // X
            Hx(0,0) = 0.0;
            Hx(0,1) = (1.0/2.0-c/2.0)*v0x/4.0-(1.0-c)*v1x/8.0+
                (1.0-c)*v2x/8.0-(1.0/2.0-c/2.0)*v3x/4.0;
            Hx(0,2) = (1.0-b)*v0x/8.0-(1.0-b)*v1x/8.0-(1.0+b)*v2x/8.0+
                (1.0+b)*v3x/8.0;

            Hx(1,0) = Hx(0,1);
            Hx(1,1) = 0.0;
            Hx(1,2) = (1.0/2.0-a/2.0)*v0x/4.0+(1.0+a)*v1x/8.0-
                (1.0+a)*v2x/8.0-(1.0/2.0-a/2.0)*v3x/4.0-
                v4x/4.0+v5x/4.0;

            Hx(2,0) = Hx(0,2);
            Hx(2,1) = Hx(1,2);
            Hx(2,2) = 0.0;


            // Y
            Hy(0,0) = 0.0;
            Hy(0,1) = (1.0/2.0-c/2.0)*v0y/4.0-(1.0-c)*v1y/8.0+
                (1.0-c)*v2y/8.0-(1.0/2.0-c/2.0)*v3y/4.0;
            Hy(0,2) = (1.0-b)*v0y/8.0-(1.0-b)*v1y/8.0-(1.0+b)*v2y/8.0+
                (1.0+b)*v3y/8.0;

            Hy(1,0) = Hy(0,1);
            Hy(1,1) = 0.0;
            Hy(1,2) = (1.0/2.0-a/2.0)*v0y/4.0+(1.0+a)*v1y/8.0-
                (1.0+a)*v2y/8.0-(1.0/2.0-a/2.0)*v3y/4.0-
                v4y/4.0+v5y/4.0;

            Hy(2,0) = Hy(0,2);
            Hy(2,1) = Hy(1,2);
            Hy(2,2) = 0.0;


            // Z
            Hz(0,0) = 0.0;
            Hz(0,1) = (1.0/2.0-c/2.0)*v0z/4.0-(1.0-c)*v1z/8.0+
                (1.0-c)*v2z/8.0-(1.0/2.0-c/2.0)*v3z/4.0;
            Hz(0,2) = (1.0-b)*v0z/8.0-(1.0-b)*v1z/8.0-(1.0+b)*v2z/8.0+
                (1.0+b)*v3z/8.0;

            Hz(1,0) = Hz(0,1);
            Hz(1,1) = 0.0;
            Hz(1,2) = (1.0/2.0-a/2.0)*v0z/4.0+(1.0+a)*v1z/8.0-
                (1.0+a)*v2z/8.0-(1.0/2.0-a/2.0)*v3z/4.0-
                v4z/4.0+v5z/4.0;

            Hz(2,0) = Hz(0,2);
            Hz(2,1) = Hz(1,2);
            Hz(2,2) = 0.0;

        }

        void Prism::calculateTensorToWorldSpaceMappingJacobian(const ElVis::TensorPoint& p, JacobiExtension::Matrix<double, 3, 3>& J)
        {
            double a = p.a();
            double b = p.b();
            double c = p.c();
            double t1 = 1.0-c;
            double t2 = 1.0-b;
            double t3 = t1*t2/2.0;
            double t6 = t1*t2;
            double t9 = 1.0+b;
            double t10 = t1*t9;
            double t13 = t1*t9/2.0;
            double t17 = 1.0+a;
            double t18 = t17*t1;
            double t20 = t18/2.0-1.0+c;
            double t29 = 1.0+c;
            double t35 = 1.0-a;
            double t36 = t35*t2/2.0;
            double t39 = t17*t2;
            double t42 = t17*t9;
            double t45 = t35*t9/2.0;

            double v1x = vertex(0).x();
            double v2x = vertex(1).x();
            double v3x = vertex(2).x();
            double v4x = vertex(3).x();
            double v5x = vertex(4).x();
            double v6x = vertex(5).x();

            double v1y = vertex(0).y();
            double v2y = vertex(1).y();
            double v3y = vertex(2).y();
            double v4y = vertex(3).y();
            double v5y = vertex(4).y();
            double v6y = vertex(5).y();

            double v1z = vertex(0).z();
            double v2z = vertex(1).z();
            double v3z = vertex(2).z();
            double v4z = vertex(3).z();
            double v5z = vertex(4).z();
            double v6z = vertex(5).z();

            J.setData(0,0) = -t3*v1x/4.0+t6*v2x/8.0+t10*v3x/8.0-t13*v4x/4.0;
            J.setData(0,1) = t20*v1x/4.0-t18*v2x/8.0+t18*v3x/8.0-t20*v4x/4.0-t29*v5x/4.0+t29
                *v6x/4.0;
            J.setData(0,2) = -t36*v1x/4.0-t39*v2x/8.0-t42*v3x/8.0-t45*v4x/4.0+t2*v5x/4.0+t9*
                v6x/4.0;
            J.setData(1,0) = -t3*v1y/4.0+t6*v2y/8.0+t10*v3y/8.0-t13*v4y/4.0;
            J.setData(1,1) = t20*v1y/4.0-t18*v2y/8.0+t18*v3y/8.0-t20*v4y/4.0-t29*v5y/4.0+t29
                *v6y/4.0;
            J.setData(1,2) = -t36*v1y/4.0-t39*v2y/8.0-t42*v3y/8.0-t45*v4y/4.0+t2*v5y/4.0+t9*
                v6y/4.0;
            J.setData(2,0) = -t3*v1z/4.0+t6*v2z/8.0+t10*v3z/8.0-t13*v4z/4.0;
            J.setData(2,1) = t20*v1z/4.0-t18*v2z/8.0+t18*v3z/8.0-t20*v4z/4.0-t29*v5z/4.0+t29
                *v6z/4.0;
            J.setData(2,2) = -t36*v1z/4.0-t39*v2z/8.0-t42*v3z/8.0-t45*v4z/4.0+t2*v5z/4.0+t9*
                v6z/4.0;

        }

        void Prism::getWorldToReferenceJacobian(const ElVis::ReferencePoint& p, JacobiExtension::Matrix<double, 3, 3>& J)
        {
            double r = p.r();
            double s = p.s();
            double t = p.t();

            double t1 = 1.0-s;
            double t2 = t1*vertex(0).x();
            double t4 = 1.0+s;
            double t6 = t4*vertex(3).x();
            double t8 = r+t;
            double t10 = 1.0+r;
            double t14 = 1.0+t;
            double t21 = t1*vertex(0).y();
            double t24 = t4*vertex(3).y();
            double t36 = t1*vertex(0).z();
            double t39 = t4*vertex(3).z();

            J.setData(0,0) = -t2/4.0+t1*vertex(1).x()/4.0+t4*vertex(2).x()/4.0-t6/4.0;

            J.setData(0,1) = t8*vertex(0).x()/4.0-t10*vertex(1).x()/4.0+t10*vertex(2).x()/4.0-t8*vertex(3).x()/4.0-
                t14*vertex(4).x()/4.0+t14*vertex(5).x()/4.0;

            J.setData(0,2) = -t2/4.0-t6/4.0+t1*vertex(4).x()/4.0+t4*vertex(5).x()/4.0;

            J.setData(1,0) = -t21/4.0+t1*vertex(1).y()/4.0+t4*vertex(2).y()/4.0-t24/4.0;

            J.setData(1,1) = t8*vertex(0).y()/4.0-t10*vertex(1).y()/4.0+t10*vertex(2).y()/4.0-t8*vertex(3).y()/4.0-
                t14*vertex(4).y()/4.0+t14*vertex(5).y()/4.0;

            J.setData(1,2) = -t21/4.0-t24/4.0+t1*vertex(4).y()/4.0+t4*vertex(5).y()/4.0;

            J.setData(2,0) = -t36/4.0+t1*vertex(1).z()/4.0+t4*vertex(2).z()/4.0-t39/4.0;

            J.setData(2,1) = t8*vertex(0).z()/4.0-t10*vertex(1).z()/4.0+t10*vertex(2).z()/4.0-t8*vertex(3).z()/4.0-
                t14*vertex(4).z()/4.0+t14*vertex(5).z()/4.0;

            J.setData(2,2) = -t36/4.0-t39/4.0+t1*vertex(4).z()/4.0+t4*vertex(5).z()/4.0;
        }

        void Prism::outputVertexOrderForVTK(std::ofstream& outFile, int startPoint)
        {
            outFile << startPoint << " " << startPoint+1 << " "
                << startPoint+4 << " " << startPoint+3 << " "
                << startPoint+2 << " " << startPoint+5;
            //outFile << "0 1 4 3 2 5";
        }

        int Prism::vtkCellType()
        {
            return 13;
        }

        void Prism::writeElementForVTKAsCell(const char* fileName)
        {
            std::ofstream outFile(fileName, ios::out);
            outFile << "# vtk DataFile Version 2.0" << endl;
            outFile << "Prism." << endl;
            outFile << "ASCII" << endl;
            outFile << "DATASET UNSTRUCTURED_GRID" << endl;
            outFile << "POINTS " << NUM_PRISM_VERTICES << " float" << endl;
            int i = 0;
            for(i = 0; i < NUM_PRISM_VERTICES; i++)
            {
                outFile << vertex(i).x() << " " <<
                    vertex(i).y() << " " <<
                    vertex(i).z() << endl;
            }

            outFile << "CELLS 1 7" << endl;
            outFile << "6 ";// 0 1 4 3 2 5" << endl;
            outputVertexOrderForVTK(outFile);
            outFile << endl;
            outFile << "CELL_TYPES 1" << endl;
            outFile << "13" << endl;
            outFile << "POINT_DATA " << NUM_PRISM_VERTICES << endl;
            outFile << "SCALARS scalars float 1" << endl;
            outFile << "LOOKUP_TABLE default" << endl;
            for(i = 0; i < NUM_PRISM_VERTICES; i++)
            {
                ElVis::WorldPoint wp = vertex(i);
                ElVis::TensorPoint tp = this->transformWorldToTensor(wp);
                double val = f(tp);
                outFile << val << " ";
            }
            outFile << endl;
            outFile.close();
        }

        void Prism::writeElementGeometryForVTK(const char* fileName)
        {
            std::ofstream outFile(fileName, ios::out);
            outFile << "# vtk DataFile Version 2.0" << endl;
            outFile << "Prism." << endl;
            outFile << "ASCII" << endl;
            outFile << "DATASET POLYDATA" << endl;
            outFile << "POINTS " << NUM_PRISM_VERTICES << " float" << endl;

            for(int i = 0; i < NUM_PRISM_VERTICES; i++)
            {
                outFile << vertex(i).x() << " " <<
                    vertex(i).y() << " " <<
                    vertex(i).z() << endl;
            }

            outFile << "POLYGONS 5 23" << endl;

            outFile << "4 0 1 2 3" << endl;
            outFile << "4 1 2 5 4" << endl;
            outFile << "4 0 3 5 4" << endl;
            outFile << "3 0 1 4" << endl;
            outFile << "3 2 3 5" << endl;
            outFile.close();
        }


        unsigned int Prism::NumberOfCoefficientsForOrder(unsigned int order)
        {
            unsigned int result = 0;
            for(unsigned int i = 0; i <= order; ++i)
            {
                for(unsigned int j = 0; j <= order; ++j)
                {
                    for(unsigned int k = 0; k <= order-i; ++k)
                    {
                        ++result;
                    }
                }
            }
            return result;
        }

        double Prism::f(const ElVis::TensorPoint& p) const
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

            boost::multi_array<double, 2> thirdDirection(boost::extents[degree(0)+1][degree(2)+1]);
            for(unsigned int i = 0; i <= degree(0); ++i)
            {
                for(unsigned int k = 0; k <= degree(2)-i; ++k )
                {
                    thirdDirection[i][k] = pow((1.0-p.c()), static_cast<double>(i)) *
                        Jacobi::P(k, 2*i+1, 0, p.c());
                }
            }

            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Prism::df_da(const ElVis::TensorPoint& p) const
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

            boost::multi_array<double, 2> thirdDirection(boost::extents[degree(0)+1][degree(2)+1]);
            for(unsigned int i = 0; i <= degree(0); ++i)
            {
                for(unsigned int k = 0; k <= degree(2)-i; ++k )
                {
                    thirdDirection[i][k] = pow((1.0-p.c()), static_cast<double>(i)) *
                        Jacobi::P(k, 2*i+1, 0, p.c());
                }
            }

            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Prism::df_db(const ElVis::TensorPoint& p) const
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

            boost::multi_array<double, 2> thirdDirection(boost::extents[degree(0)+1][degree(2)+1]);
            for(unsigned int i = 0; i <= degree(0); ++i)
            {
                for(unsigned int k = 0; k <= degree(2)-i; ++k )
                {
                    thirdDirection[i][k] = pow((1.0-p.c()), static_cast<double>(i)) *
                        Jacobi::P(k, 2*i+1, 0, p.c());
                }
            }

            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Prism::df_dc(const ElVis::TensorPoint& p) const
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

            boost::multi_array<double, 2> thirdDirection(boost::extents[degree(0)+1][degree(2)+1]);
            for(unsigned int i = 0; i <= degree(0); ++i)
            {
                for(unsigned int k = 0; k <= degree(2)-i; ++k )
                {
                    double d1 = pow((1.0-p.c()), static_cast<double>(i)) *
                        Jacobi::dP(k, 2*i+1, 0, p.c());
                    double d2 = 0.0;
                    if( i > 0 )
                    {
                        d2 = pow((1.0-p.c()), static_cast<double>(i-1)) * (-static_cast<double>(i)) *
                            Jacobi::P(k, 2*i+1, 0, p.c());
                    }
                    thirdDirection[i][k] = d1+d2;
                }
            }

            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Prism::performSummation(const std::vector<double>& v1,
            const std::vector<double>& v2, const boost::multi_array<double, 2>& v3) const
        {
            double result(0.0);
            std::vector<double>::const_iterator coeff = basisCoefficients().begin();
            for(unsigned int i = 0; i <= degree(0); ++i)
            {
                for(unsigned int j = 0; j <= degree(1); ++j)
                {
                    for(unsigned int k = 0; k <= degree(2)-i; ++k)
                    {
                        result += (*coeff)*v1[i]*v2[j]*v3[i][k];
                        ++coeff;
                    }
                }
            }

            return result;
        }

        double Prism::df2_da2(const ElVis::TensorPoint& p) const
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

            boost::multi_array<double, 2> thirdDirection(boost::extents[degree(0)+1][degree(2)+1]);
            for(unsigned int i = 0; i <= degree(0); ++i)
            {
                for(unsigned int k = 0; k <= degree(2)-i; ++k )
                {
                    thirdDirection[i][k] = pow((1.0-p.c()), static_cast<double>(i)) *
                        Jacobi::P(k, 2*i+1, 0, p.c());
                }
            }

            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Prism::df2_db2(const ElVis::TensorPoint& p) const
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

            boost::multi_array<double, 2> thirdDirection(boost::extents[degree(0)+1][degree(2)+1]);
            for(unsigned int i = 0; i <= degree(0); ++i)
            {
                for(unsigned int k = 0; k <= degree(2)-i; ++k )
                {
                    thirdDirection[i][k] = pow((1.0-p.c()), static_cast<double>(i)) *
                        Jacobi::P(k, 2*i+1, 0, p.c());
                }
            }

            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Prism::df2_dc2(const ElVis::TensorPoint& p) const
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

            boost::multi_array<double, 2> thirdDirection(boost::extents[degree(0)+1][degree(2)+1]);
            for(unsigned int i = 0; i <= degree(0); ++i)
            {
                int a = 2*i+1;
                for(unsigned int k = 0; k <= degree(2)-i; ++k )
                {
                    thirdDirection[i][k] = pow((1.0-p.c()), static_cast<double>(i)) *
                        Jacobi::P(k, 2*i+1, 0, p.c());

                    if( i == 0 )
                    {
                        thirdDirection[i][k] = Jacobi::ddP(k, a, 0, p.c());
                    }
                    else if( i == 1 )
                    {
                        thirdDirection[i][k] = 2.0*pow(1.0-p.c(),1.0*i-1.0)*i*Jacobi::dP(k, a, 0, p.c()) +
                            pow(1.0-p.c(),1.0*i)*Jacobi::ddP(k, a, 0, p.c());
                    }
                    else
                    {
                        thirdDirection[i][k] = pow(1.0-p.c(),1.0*i-2.0)*i*i*Jacobi::P(k, a, 0, p.c()) -
                            pow(1.0-p.c(),1.0*i-2.0)*i*Jacobi::P(k, a, 0, p.c()) -
                            2.0*pow(1.0-p.c(),1.0*i-1.0)*i*Jacobi::dP(k, a, 0, p.c()) +
                            pow(1.0-p.c(),1.0*i)*Jacobi::ddP(k, a, 0, p.c());
                    }
                }
            }

            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Prism::df2_dab(const ElVis::TensorPoint& p) const
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

            boost::multi_array<double, 2> thirdDirection(boost::extents[degree(0)+1][degree(2)+1]);
            for(unsigned int i = 0; i <= degree(0); ++i)
            {
                for(unsigned int k = 0; k <= degree(2)-i; ++k )
                {
                    thirdDirection[i][k] = pow((1.0-p.c()), static_cast<double>(i)) *
                        Jacobi::P(k, 2*i+1, 0, p.c());
                }
            }

            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Prism::df2_dac(const ElVis::TensorPoint& p) const
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

            boost::multi_array<double, 2> thirdDirection(boost::extents[degree(0)+1][degree(2)+1]);
            for(unsigned int i = 0; i <= degree(0); ++i)
            {
                for(unsigned int k = 0; k <= degree(2)-i; ++k )
                {
                    double d1 = pow((1.0-p.c()), static_cast<double>(i)) *
                        Jacobi::dP(k, 2*i+1, 0, p.c());
                    double d2 = 0.0;
                    if( i > 0 )
                    {
                        d2 = pow((1.0-p.c()), static_cast<double>(i-1)) * (-static_cast<double>(i)) *
                            Jacobi::P(k, 2*i+1, 0, p.c());
                    }
                    thirdDirection[i][k] = d1+d2;
                }
            }

            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Prism::df2_dbc(const ElVis::TensorPoint& p) const
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

            boost::multi_array<double, 2> thirdDirection(boost::extents[degree(0)+1][degree(2)+1]);
            for(unsigned int i = 0; i <= degree(0); ++i)
            {
                for(unsigned int k = 0; k <= degree(2)-i; ++k )
                {
                    double d1 = pow((1.0-p.c()), static_cast<double>(i)) *
                        Jacobi::dP(k, 2*i+1, 0, p.c());
                    double d2 = 0.0;
                    if( i > 0 )
                    {
                        d2 = pow((1.0-p.c()), static_cast<double>(i-1)) * (-static_cast<double>(i)) *
                            Jacobi::P(k, 2*i+1, 0, p.c());
                    }
                    thirdDirection[i][k] = d1+d2;
                }
            }

            return performSummation(firstDirection, secondDirection, thirdDirection);
        }

        double Prism::do_maxScalarValue() const
        {
            return std::numeric_limits<double>::max();
        }

        double Prism::do_minScalarValue() const
        {
            return std::numeric_limits<double>::min();
        }
    }
}


