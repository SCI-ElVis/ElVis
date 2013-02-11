////////////////////////////////////////////////////////////////////////////////
//
//  File: hoPyramid.cpp
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
//#include "Pyramid.h"
//#include "EndianConvert.h"
//#include "FiniteElementMath.h"
//#include "Edge.h"
//#include "PointTransformations.h"
//#include <iostream>
//using namespace std;
//
//Pyramid::Pyramid(FILE* inFile, bool reverseBytes)
//{
//  setMinMax(inFile, reverseBytes);
//  readVerticesFromFile(inFile, reverseBytes);
//  readDegree(inFile, reverseBytes);
//
//  int numCoefficients = (degree(0)+1)*(degree(0)+2)*(degree(0)+3)/6;
//  readBasisCoefficients(inFile, numCoefficients, reverseBytes);
//
//  // TODO - Check this as well.
//  setInterpolatingPolynomialDegree(degree(0)+degree(1)+degree(2));
//  calculateFaceNormals();
//}
//
//Pyramid::~Pyramid()
//{
//}
//
//ElVis::TensorPoint<double> Pyramid::transformReferenceToTensor(const ElVis::ReferencePoint& p)
//{
//  return PointTransformations::transformPyramidReferenceToTensor(p);
//}
//
//ElVis::ReferencePoint<double> Pyramid::transformTensorToReference(const ElVis::TensorPoint& p)
//{
//  return PointTransformations::transformPyramidTensorToReference(p);
//}
//
//ElVis::WorldPoint Pyramid::transformReferenceToWorld(const ElVis::ReferencePoint& p)
//{
//  return PointTransformations::transformReferenceToWorld(*this, p);
//}
//
//IntervalTensorPoint Pyramid::transformReferenceToTensor(const IntervalReferencePoint& p)
//{
//  return PointTransformations::transformPyramidReferenceToTensor(p);
//}
//
//IntervalReferencePoint Pyramid::transformTensorToReference(const IntervalTensorPoint& p)
//{
//  return PointTransformations::transformPyramidTensorToReference(p);
//}
//
//IntervalWorldPoint Pyramid::transformReferenceToWorld(const IntervalReferencePoint& p)
//{
//  return PointTransformations::transformReferenceToWorld(*this, p);
//}
//
//
//double Pyramid::findScalarValueAtPoint(const ElVis::TensorPoint& p)
//{
//  int coeffIndex = 0;
//  double result = 0.0;
//
//  for(int k = 0; k <= degree(2); k++)
//  {
//      for(int j = 0; j <= degree(1)-k; j++)
//      {
//          for(int i = 0; i <= degree(0)-k-j; i++)
//          {
//              result += basisCoefficients()[coeffIndex] *
//                  Jacobi::P(i, 0, 0, p.a()) *
//                  pow( 1.0-p.b(), (double)i) * Jacobi::P(j, 2*i+1, 0, p.b()) *
//                  pow( 1.0-p.c(), (double)(i+j)) * Jacobi::P(k, 2*i+2*j+2, 0, p.c());
//
//              coeffIndex++;
//          }
//      }
//  }
//
//  return result;
//}
//
//void Pyramid::calculateNormal(const ElVis::TensorPoint& p, ElVis::WorldVector* n)
//{
//  int coeffIndex = 0;
//  double result = 0.0;
//  int i,j,k;
//
//  for(k = 0; k <= degree(2); k++)
//  {
//      for(j = 0; j <= degree(1)-k; j++)
//      {
//          for(i = 0; i <= degree(0)-k-j; i++)
//          {
//              result += basisCoefficients()[coeffIndex] *
//                  Jacobi::dP(i, 0, 0, p.a()) *
//                  pow( 1.0-p.b(), (double)i) * Jacobi::P(j, 2*i+1, 0, p.b()) *
//                  pow( 1.0-p.c(), (double)(i+j)) * Jacobi::P(k, 2*i+2*j+2, 0, p.c());
//
//              coeffIndex++;
//          }
//      }
//  }
//  n->x(result);
//
//  coeffIndex = 0;
//  result = 0.0;
//
//  for(k = 0; k <= degree(2); k++)
//  {
//      for(j = 0; j <= degree(1)-k; j++)
//      {
//          for(i = 0; i <= degree(0)-k-j; i++)
//          {
//              double d1 = Jacobi::dP(i, 0, 0, p.a()) *
//                  pow( 1.0-p.b(), (double)i) * Jacobi::dP(j, 2*i+1, 0, p.b()) *
//                  pow( 1.0-p.c(), (double)(i+j)) * Jacobi::P(k, 2*i+2*j+2, 0, p.c());
//              double d2 = Jacobi::dP(i, 0, 0, p.a()) *
//                  Jacobi::P(j, 2*i+1, 0, p.b()) *
//                  pow( 1.0-p.c(), (double)(i+j)) * Jacobi::P(k, 2*i+2*j+2, 0, p.c());
//
//              if( i != 0 )
//              {
//                  d2 *= pow(1.0-p.b(), (double)(i-1));
//              }
//              result += basisCoefficients()[coeffIndex]*(d1+d2);
//              coeffIndex++;
//          }
//      }
//  }
//  n->y(result);
//
//  coeffIndex = 0;
//  result = 0.0;
//
//  for(k = 0; k <= degree(2); k++)
//  {
//      for(j = 0; j <= degree(1)-k; j++)
//      {
//          for(i = 0; i <= degree(0)-k-j; i++)
//          {
//              double d1 = Jacobi::dP(i, 0, 0, p.a()) *
//                  pow( 1.0-p.b(), (double)i) * Jacobi::P(j, 2*i+1, 0, p.b()) *
//                  pow( 1.0-p.c(), (double)(i+j)) * Jacobi::dP(k, 2*i+2*j+2, 0, p.c());
//              double d2 = Jacobi::dP(i, 0, 0, p.a()) *
//                  pow( 1.0-p.b(), (double)i) * Jacobi::P(j, 2*i+1, 0, p.b()) *
//                  Jacobi::P(k, 2*i+2*j+2, 0, p.c());
//
//              if( i+j != 0 )
//              {
//                  d2 *= pow( 1.0-p.c(), (double)(i+j)-1);
//              }
//              result += basisCoefficients()[coeffIndex]*(d1+d2);
//              coeffIndex++;
//          }
//      }
//  }
//  n->z(result);
//}
//
//int Pyramid::vtkCellType()
//{
//  return 10;
//}
//
//void Pyramid::calculateFaceNormals()
//{
//  // Calculate the normals for each face.
//  ElVis::WorldVector d1 = createVectorFromPoints(vertex(2), vertex(0));
//  ElVis::WorldVector d2 = createVectorFromPoints(vertex(1), vertex(0));
//  faceNormals[0] = d1.cross(d2);
//
//  d1 = createVectorFromPoints(vertex(1), vertex(0));
//  d2 = createVectorFromPoints(vertex(3), vertex(0));
//  faceNormals[1] = d1.cross(d2);
//
//  d1 = createVectorFromPoints(vertex(2), vertex(1));
//  d2 = createVectorFromPoints(vertex(3), vertex(1));
//  faceNormals[2] = d1.cross(d2);
//
//  d1 = createVectorFromPoints(vertex(3), vertex(0));
//  d2 = createVectorFromPoints(vertex(2), vertex(0));
//  faceNormals[3] = d1.cross(d2);
//
//  for(int i = 0; i < NUM_PYRAMID_FACES; i++)
//  {
//      faceNormals[i].normalize();
//  }
//
//  D[0] = -(faceNormals[0].x()*vertex(0).x() + faceNormals[0].y()*vertex(0).y() +
//      faceNormals[0].z()*vertex(0).z());
//
//  D[1] = -(faceNormals[1].x()*vertex(3).x() + faceNormals[1].y()*vertex(3).y() +
//      faceNormals[1].z()*vertex(3).z());
//
//
//  D[2] = -(faceNormals[2].x()*vertex(3).x() + faceNormals[2].y()*vertex(3).y() +
//      faceNormals[2].z()*vertex(3).z());
//
//  D[3] = -(faceNormals[3].x()*vertex(3).x() + faceNormals[3].y()*vertex(3).y() +
//      faceNormals[3].z()*vertex(3).z());
//}
//
//bool Pyramid::findIntersectionWithGeometryInWorldSpace(rt::Ray& ray, double& min, double& max)
//{
//  ElVis::WorldVector u,v,w;
//  generateCoordinateSystemFromVector(ray.direction(), u, v, w);
//
//  ElVis::WorldPoint o = ray.origin();
//  ElVis::WorldPoint transformedPoints[NUM_PYRAMID_VERTICES];
//  for(int i = 0; i < NUM_PYRAMID_VERTICES; i++)
//  {
//      double tx = vertex(i).x()-o.x();
//      double ty = vertex(i).y()-o.y();
//      double tz = vertex(i).z()-o.z();
//      transformedPoints[i].x(u.x()*tx + u.y()*ty + u.z()*tz);
//      transformedPoints[i].y(v.x()*tx + v.y()*ty + v.z()*tz);
//
//      // Don't worry about the w component.  We want to project onto
//      // the uv plane so we'll set it to 0.
//  }
//
//  Edge e1(transformedPoints[0].x(), transformedPoints[0].y(),
//      transformedPoints[1].x(), transformedPoints[1].y());
//  Edge e2(transformedPoints[0].x(), transformedPoints[0].y(),
//      transformedPoints[3].x(), transformedPoints[3].y());
//
//  Edge e3(transformedPoints[1].x(), transformedPoints[1].y(),
//      transformedPoints[3].x(), transformedPoints[3].y());
//  Edge e4(transformedPoints[0].x(), transformedPoints[0].y(),
//      transformedPoints[2].x(), transformedPoints[2].y());
//
//  Edge e5(transformedPoints[1].x(), transformedPoints[1].y(),
//      transformedPoints[2].x(), transformedPoints[2].y());
//  Edge e6(transformedPoints[2].x(), transformedPoints[2].y(),
//      transformedPoints[3].x(), transformedPoints[3].y());
//
//  double t = 0.0;
//  min = FLT_MAX;
//  max = -FLT_MAX;
//
//  if( (e1.numTimesCrossesPositiveXAxis() +
//      e2.numTimesCrossesPositiveXAxis() +
//      e3.numTimesCrossesPositiveXAxis()) & 0x01 )
//  {
//      intersectsFacePlane(ray, 1, min, max);
//  }
//
//  if( (e1.numTimesCrossesPositiveXAxis() +
//      e4.numTimesCrossesPositiveXAxis() +
//      e5.numTimesCrossesPositiveXAxis()) & 0x01 )
//  {
//      intersectsFacePlane(ray, 0, min, max);
//  }
//
//  if( (e3.numTimesCrossesPositiveXAxis() +
//      e5.numTimesCrossesPositiveXAxis() +
//      e6.numTimesCrossesPositiveXAxis()) & 0x01 )
//  {
//      intersectsFacePlane(ray, 2, min, max);
//  }
//
//  if( (e2.numTimesCrossesPositiveXAxis() +
//      e6.numTimesCrossesPositiveXAxis() +
//      e4.numTimesCrossesPositiveXAxis()) & 0x01 )
//  {
//      intersectsFacePlane(ray, 3, min, max);
//  }
//
//  return min != FLT_MAX && min < max;
//}
//
//
//bool Pyramid::intersectsFacePlane(const rt::Ray& ray, int face, double& min, double& max)
//{
//  double t = -1;
//  if( planeIntersection(ray, faceNormals[face], D[face], t) )
//  {
//      if( t > max ) max = t;
//      if( t < min ) min = t;
//      return true;
//  }
//  return false;
//}
//
//void Pyramid::getWorldToReferenceJacobian(double r, double s, double t, double* J)
//{
//  J[0] = -vertex(0)[0]/2.0+vertex(1)[0]/2.0;
//  J[1] = -vertex(0)[0]/2.0+vertex(2)[0]/2.0;
//  J[2] = -vertex(0)[0]/2.0+vertex(3)[0]/2.0;
//  J[3] = -vertex(0)[1]/2.0+vertex(1)[1]/2.0;
//  J[4] = -vertex(0)[1]/2.0+vertex(2)[1]/2.0;
//  J[5] = -vertex(0)[1]/2.0+vertex(3)[1]/2.0;
//  J[6] = -vertex(0)[2]/2.0+vertex(1)[2]/2.0;
//  J[7] = -vertex(0)[2]/2.0+vertex(2)[2]/2.0;
//  J[8] = -vertex(0)[2]/2.0+vertex(3)[2]/2.0;
//
//}
//
//void Pyramid::writeElement(FILE* outFile)
//{
//}
//
//
//void Pyramid::writeElementGeometryForVTK(const char* fileName)
//{
//}
//
//void Pyramid::writeElementForVTKAsCell(const char* fileName)
//{
//}
//
//void Pyramid::outputVertexOrderForVTK(std::ofstream& outFile,int startPoint)
//{
//}
//
//
