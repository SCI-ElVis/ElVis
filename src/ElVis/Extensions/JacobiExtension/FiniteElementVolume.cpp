////////////////////////////////////////////////////////////////////////////////
//
//  File: hoFiniteElementVolume.cpp
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
#include <iostream>

#include <ElVis/Extensions/JacobiExtension/FiniteElementVolume.h>
//#include <ray_tracer/rtIntersectionInfo.h>
#include <ElVis/Extensions/JacobiExtension/PointTransformations.hpp>
#include <ElVis/Extensions/JacobiExtension/PolyhedraBoundingBox.h>
#include <limits>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <functional>
#include <ElVis/Extensions/JacobiExtension/EndianConvert.h>
#include <boost/bind.hpp>
//#include <strstream>
#include <ElVis/Extensions/JacobiExtension/Hexahedron.h>
#include <ElVis/Extensions/JacobiExtension/Tetrahedron.h>
#include <ElVis/Extensions/JacobiExtension/Prism.h>
#include <ElVis/Extensions/JacobiExtension/Pyramid.h>
#include <ElVis/Extensions/JacobiExtension/NumericalIntegration.hpp>
#include <boost/timer.hpp>
//#include "Scene.h"

using namespace std;
using namespace boost;
using std::cout;
using std::endl;

namespace ElVis
{
    namespace JacobiExtension
    {
        const char* FiniteElementVolume::headerData = "Finite Element Volume  ";

        FiniteElementVolume::OutOfBoundsPoint::OutOfBoundsPoint(const std::string& what_arg) :
        std::runtime_error(what_arg)
        {
        }

        FiniteElementVolume::FiniteElementVolume() :
        LastFoundElement()
        {
        }

        //XERCES_CPP_NAMESPACE_USE
        //FiniteElementVolume::FiniteElementVolume(DOMElement* volumeElement, boost::shared_ptr<Scene> theScene) :
        //  Object(volumeElement, theScene),
        //  m_minValue(-numeric_limits<double>::max()),
        //  m_maxValue(numeric_limits<double>::max()),
        //  m_fileName(""),
        //  itsBBoxes(NULL),
        //  m_lowerBound(0.0), m_upperBound(1.0),
        //  m_enableColorMap(false),
        //  m_polynomialDegree(1),
        //  m_tolerance(.00000001),
        //  m_statFileName(""),
        //  m_createGrid(false)
        //{
        //  DOMElementIterator iter(volumeElement);
        //  while( ++iter )
        //  {
        //      DOMElement* element = *iter;
        //      std::string str(DOMString(element->getTextContent()).convertToString());

        //      if( element->getNodeName() == DOMString("File") )
        //      {
        //          m_fileName = str;
        //      }
        //      else if( element->getNodeName() == DOMString("PolynomialDegree") )
        //      {
        //          m_polynomialDegree = boost::lexical_cast<unsigned int>(str);
        //      }
        //      else if( element->getNodeName() == DOMString("CreateGrid") )
        //      {
        //          if( str == "true" )
        //          {
        //              m_createGrid = true;
        //          }
        //      }
        //  }

        //  init();
        //  theScene->setFiniteElementVolume(this);
        //}

        //class CreateFiniteElementVolume
        //{
        //  public:
        //      Object* operator()(DOMElement* sphereElement, boost::shared_ptr<Scene> scene)
        //      {
        //          return new FiniteElementVolume(sphereElement, scene);
        //      }
        //};


        //const bool registered =
        //  ObjectFactorySingleton::Instance().Register(std::string("FiniteElementVolume"), CreateFiniteElementVolume());

        FiniteElementVolume::FiniteElementVolume(
            const char* fileName)
            : //rt::RayTraceableObjectGroup(),
        //boost::enable_shared_from_this<FiniteElementVolume>(),
        m_minValue(-numeric_limits<double>::max()),
            m_maxValue(numeric_limits<double>::max()),
            LastFoundElement(),
            m_fileName(fileName),
            itsBBoxes(NULL),
            m_lowerBound(0.0), m_upperBound(1.0),
            m_enableColorMap(false),
            m_polynomialDegree(6),
            m_tolerance(0.0),
            m_createGrid(false)
        {
            init();
        }

        bool FiniteElementVolume::CanLoadVolume(const char* fileName)
        {
            FILE* in = fopen(fileName, "rb");
            if(!in)
            {
                return false;
            }

            char buf2[200];
            if( fread(buf2, sizeof(char), strlen(headerData)+1, in) != strlen(headerData)+1 )
            {
              fclose(in);
              std::cout << __FILE__ << ": " << __LINE__ << std::endl;
              throw std::runtime_error("Failed to read header data from: " + std::string(fileName) );
              //std::cout << "Failed to read header data from: " + std::string(fileName) << std::endl;
            }
            fclose(in);

            return strcmp(headerData, buf2) == 0;
        }

        void FiniteElementVolume::init()
        {
            NumericalIntegration::GaussLegendreNodesAndWeights<double>::InitializeNodesAndWeights();


            //setMaterial(matl);
            // Format of the input file:
            //
            // FiniteElementVolume file
            // number_of_points number_of_elements
            // binary floating ElVis::WorldPoint data, x y z scalar value
            // binary floating ElVis::WorldPoint data for elements.
            // Format is defined by the element, but contains an identifier
            // specifying the type of element, integer index into the ElVis::WorldPoint
            // array for each vertex, and information about the basis functions
            // used for interpolation.
            cout << "Creating a finite element volume from file " << m_fileName << endl;
            FILE* in = fopen(m_fileName.c_str(), "rb");
            if(!in)
            {
                std::runtime_error e("Error opening input file: " + m_fileName);
                throw e;
            }

            char buf2[200];
            if( fread(buf2, sizeof(char), strlen(headerData)+1, in) != strlen(headerData)+1 )
            {
              std::cout << __FILE__ << ": " << __LINE__ << std::endl;
              throw std::runtime_error("Failed to read header data from: " + m_fileName);
            }

            if(strcmp(headerData, buf2) != 0)
            {
              std::cout << __FILE__ << ": " << __LINE__ << std::endl;
              throw std::runtime_error(m_fileName + " is not a valid FiniteElementVolume file");
            }

            int endianCheck = -1;

            if( fread(&endianCheck, sizeof(int), 1, in) != 1 )
            {
                cerr << "Error reading the endian check flag." << endl;
                exit(1);
            }

            bool reverseBytes = (endianCheck != 1);
            std::cout << "Reverse bytes: " << reverseBytes << std::endl;
            int numElements = 0;

            if( fread(&numElements, sizeof(int), 1, in) != 1 )
            {
                cerr << "Error reading number of elements." << endl;
                exit(1);
            }

            if( reverseBytes ) JacobiExtension::reverseBytes(numElements);

            if(numElements <= 0 )
            {
                cerr << "Invalid number of elements specified." << endl;
                exit(1);
            }
            std::cout << "Number of elements: " << numElements << std::endl;

            for(int i = 0; i < numElements; i++)
            {
                elements.push_back(createNewPolyhedron(in, reverseBytes));
            }

//            if( m_polynomialDegree > 0 )
//            {
//                std::for_each(elements.begin(), elements.end(), boost::bind(&Polyhedron::interpolatingPolynomialDegreeOverride,
//                    _1, m_polynomialDegree));
//            }

//            if( m_tolerance > 0.0 )
//            {
//                std::for_each(elements.begin(), elements.end(), boost::bind(&Polyhedron::referenceToWorldTolerance,
//                    _1, m_tolerance));
//            }

            /*
            // The file has now been read in.  We now need to go through
            // each element and calculate the min/max values.
            static double largestMinDiffMag = -DBL_MAX;

            for(int i = 0; i < numElements; i++)
            {
            if( elements[i]->vtkCellType() == 11 &&
            elements[i]->id() == 696 )
            {
            cout << "Processing element " << elements[i]->id() << endl;
            double prevMin = elements[i]->getMin();
            elements[i]->calculateMinAndMax();
            double newMin = elements[i]->getMin();
            cout << "prev Min for element " << i << " is " <<
            prevMin << "." << endl;
            cout << "New min for element is " << newMin << endl;

            if( fabs(newMin-prevMin) > largestMinDiffMag )
            {
            largestMinDiffMag = fabs(newMin-prevMin);
            }
            cout << "####################" << endl;
            cout << largestMinDiffMag << endl;
            cout << "####################" << endl;
            }
            }
            */

            // Now that we've read in all of the data we need to
            // create the acceleration structures.
            if( m_createGrid )
            {
                std::string accelerationName = m_fileName + std::string(".bounding_boxes");
                //itsBBoxes = createSimpleBoundingBox(elements, accelerationName);
                //itsBBoxes = createSimpleBoundingBox(elements);
            }

            ElVis::WorldPoint _min, _max;
            calcOverallBoundingBox(_min, _max);
            //cout << "Data set from " << _min << " to " << _max << endl;
            calcOverallMinMaxScalar();
        }

        void calcStats(const std::vector<double>& vals,
            double& min, double& max, double& l_infinity, double& l_2,
            double& average)
        {
            min = std::numeric_limits<double>::max();
            max = -std::numeric_limits<double>::max();
            l_2 = 0.0;
            average = 0.0;

            for(unsigned int i = 0; i < vals.size(); ++i)
            {
                average += fabs(vals[i]);
                l_2 += vals[i]*vals[i];

                if( fabs(vals[i]) < min )
                {
                    min = fabs(vals[i]);
                }

                if( fabs(vals[i]) > max )
                {
                    max = fabs(vals[i]);
                }
            }

            l_2 = sqrt(l_2/static_cast<double>(vals.size()));
            average /= static_cast<double>(vals.size());
            l_infinity = max;
        }

        void FiniteElementVolume::printStats(std::ostream& os, bool printHeader)
        {
//            std::vector<double> relative_error;
//            std::vector<double> absolute_error;
//            std::vector<double> root_finding_error;
//            //std::vector<double> interp_relative_error;
//            //std::vector<double> interp_absolute_error;
//            //std::vector<double> interpolation_l2_norm;
//            //std::vector<double> projection_l2_norm;
//            //std::vector<double> interpolation_infinity_norm;
//            //std::vector<double> projection_infinity_norm;

//            for(unsigned int i = 0; i < elements.size(); ++i)
//            {
//                boost::shared_ptr<Polyhedron> poly = elements[i];
//                relative_error.insert(relative_error.end(),
//                    poly->relativeErrorStats().begin(),
//                    poly->relativeErrorStats().end());
//                absolute_error.insert(absolute_error.end(),
//                    poly->absoluteErrorStates().begin(),
//                    poly->absoluteErrorStates().end());
//                root_finding_error.insert(root_finding_error.end(),
//                    poly->rootFindingErrorStats().begin(),
//                    poly->rootFindingErrorStats().end());
//                /*interpolation_l2_norm.insert(interpolation_l2_norm.end(),
//                poly->interpolationL2NormStats().begin(),
//                poly->interpolationL2NormStats().end());
//                projection_l2_norm.insert(projection_l2_norm.end(),
//                poly->projectionL2NormStats().begin(),
//                poly->projectionL2NormStats().end());
//                interpolation_infinity_norm.insert(interpolation_infinity_norm.end(),
//                poly->interpolationInfinityNormStats().begin(),
//                poly->interpolationInfinityNormStats().end());
//                projection_infinity_norm.insert(projection_infinity_norm.end(),
//                poly->projectionInfinityNormStats().begin(),
//                poly->projectionInfinityNormStats().end());
//                interp_relative_error.insert(interp_relative_error.end(),
//                poly->interp_relativeErrorStats().begin(),
//                poly->interp_relativeErrorStats().end());
//                interp_absolute_error.insert(interp_absolute_error.end(),
//                poly->interp_absoluteErrorStats().begin(),
//                poly->interp_absoluteErrorStats().end());*/
//            }

//            double min;
//            double max;
//            double l_infinity;
//            double l_2;
//            double average;

//            if( printHeader )
//            {
//                os << "VOLUME\tSTATISTIC\tISOVALUE\tPOLYNOMIAL_DEGREE\tTOLERANCE\tMIN\tMAX\tINFINITY\tL2\tAVERAGE" << endl;
//            }

//            calcStats(relative_error, min, max, l_infinity, l_2, average);
//            os << m_fileName << "\t" << "RELATIVE\t" << m_isoval << "\t" <<
//                m_polynomialDegree << "\t" << m_tolerance << "\t" <<
//                min << "\t" << max << "\t" << l_infinity << "\t" <<
//                l_2 << "\t" << average << endl;

//            calcStats(absolute_error, min, max, l_infinity, l_2, average);
//            os << m_fileName << "\t" << "ABSOLUTE\t" << m_isoval << "\t" <<
//                m_polynomialDegree << "\t" << m_tolerance << "\t" <<
//                min << "\t" << max << "\t" << l_infinity << "\t" <<
//                l_2 << "\t" << average << endl;

//            calcStats(root_finding_error, min, max, l_infinity, l_2, average);
//            os << m_fileName << "\t" << "ROOT_FINDING\t" << m_isoval << "\t" <<
//                m_polynomialDegree << "\t" << m_tolerance << "\t" <<
//                min << "\t" << max << "\t" << l_infinity << "\t" <<
//                l_2 << "\t" << average << endl;

//            //calcStats(interpolation_l2_norm, min, max, l_infinity, l_2, average);
//            //os << m_fileName << "\t" << "INTERP_L2\t" << m_isoval << "\t" <<
//            //  m_polynomialDegree << "\t" << m_tolerance << "\t" <<
//            //  min << "\t" << max << "\t" << l_infinity << "\t" <<
//            //  l_2 << "\t" << average << endl;
//            //
//            //calcStats(projection_l2_norm, min, max, l_infinity, l_2, average);
//            //os << m_fileName << "\t" << "PROJECTION_L2\t" << m_isoval << "\t" <<
//            //  m_polynomialDegree << "\t" << m_tolerance << "\t" <<
//            //  min << "\t" << max << "\t" << l_infinity << "\t" <<
//            //  l_2 << "\t" << average << endl;

//            //calcStats(interpolation_infinity_norm, min, max, l_infinity, l_2, average);
//            //os << m_fileName << "\t" << "INTERPOLATION_INFINITY\t" << m_isoval << "\t" <<
//            //  m_polynomialDegree << "\t" << m_tolerance << "\t" <<
//            //  min << "\t" << max << "\t" << l_infinity << "\t" <<
//            //  l_2 << "\t" << average << endl;

//            //calcStats(projection_infinity_norm, min, max, l_infinity, l_2, average);
//            //os << m_fileName << "\t" << "PROJECTION_INFINITY\t" << m_isoval << "\t" <<
//            //  m_polynomialDegree << "\t" << m_tolerance << "\t" <<
//            //  min << "\t" << max << "\t" << l_infinity << "\t" <<
//            //  l_2 << "\t" << average << endl;
//            //
//            //calcStats(interp_relative_error, min, max, l_infinity, l_2, average);
//            //os << m_fileName << "\t" << "INTER_RELATIVE_ERRROR\t" << m_isoval << "\t" <<
//            //  m_polynomialDegree << "\t" << m_tolerance << "\t" <<
//            //  min << "\t" << max << "\t" << l_infinity << "\t" <<
//            //  l_2 << "\t" << average << endl;
//            //
//            //calcStats(interp_absolute_error, min, max, l_infinity, l_2, average);
//            //os << m_fileName << "\t" << "INTERP_ABSOLUTE_ERROR\t" << m_isoval << "\t" <<
//            //  m_polynomialDegree << "\t" << m_tolerance << "\t" <<
//            //  min << "\t" << max << "\t" << l_infinity << "\t" <<
//            //  l_2 << "\t" << average << endl;

        }

        FiniteElementVolume::~FiniteElementVolume()
        {
            //unsigned int numHexes = 0;
            //unsigned int numPrisms = 0;
            //unsigned int numTets = 0;
            //unsigned int numPyramids = 0;

            //for(size_t i = 0; i < elements.size(); ++i)
            //{
            //    Hexahedron* hex = dynamic_cast<Hexahedron*>(elements[i]);
            //    if( hex )
            //    {
            //        ++numHexes;
            //    }

            //    Prism* prism = dynamic_cast<Prism*>(elements[i]);
            //    if( prism )
            //    {
            //        ++numPrisms;
            //    }

            //    Tetrahedron* tet = dynamic_cast<Tetrahedron*>(elements[i]);
            //    if( tet )
            //    {
            //        ++numTets;
            //    }
            //}

            //cout << "Num hexes: " << numHexes << endl;
            //cout << "Num prisms: " << numPrisms << endl;
            //cout << "Num tets: " << numTets << endl;
            //cout << "Num Pyramids: " << numPyramids << endl;

            //if( m_statFileName.empty() )
            //{
            //    printStats(cout, true);
            //}
            //else
            //{
            //    bool printHeader = false;
            //    std::ifstream inFile(m_statFileName.c_str(), std::ios::in);
            //    if( !inFile.good() )
            //    {
            //        printHeader = true;
            //    }

            //    std::ofstream outFile(m_statFileName.c_str(), std::ios::app);
            //    printStats(outFile, printHeader);
            //    outFile.close();
            //}
        }

        void FiniteElementVolume::WriteCellVolume(unsigned int elementId, const std::string& filePrefix)
        {
            std::string fileName = filePrefix + "Element" + boost::lexical_cast<std::string>(elementId) + ".dat";
            elements[elementId]->writeElementAsIndividualVolume(fileName.c_str());
        }

        void FiniteElementVolume::writeVolume(FILE* outFile)
        {
            if( !writeHeader(outFile) )
            {
                cerr << "Error writing header." << endl;
                return;
            }

            int one = 1;
            if( fwrite(&one, sizeof(int), 1, outFile) != 1 )
            {
                cerr << "Error writing endian check." << endl;
                return;
            }

            size_t numElements = elements.size();
            if( fwrite(&numElements, sizeof(int), 1, outFile) != 1 )
            {
                cerr << "Error writing number of elements." << endl;
                return;
            }

            for(size_t i = 0; i < numElements; i++)
            {
                elements[i]->writeElement(outFile);
            }
        }

        boost::shared_ptr<Polyhedron> FiniteElementVolume::getElement(int index)
        {
            if( index >= 0 && index < static_cast<int>(elements.size()) )
                return elements[index];
            else
                return boost::shared_ptr<Polyhedron>();
        }

        void FiniteElementVolume::WriteCellVolumeForVTK(unsigned int elementId, const std::string& filePrefix)
        {
            std::string fileName = filePrefix + "Element" + boost::lexical_cast<std::string>(elementId) + ".vtk";
            ofstream outFile(fileName.c_str(), ios::out);

            outFile << "# vtk DataFile Version 2.0" << endl;
            outFile << "Volume as cells." << endl;
            outFile << "ASCII" << endl;
            outFile << "DATASET UNSTRUCTURED_GRID" << endl;

            int numPoints = 0;
            numPoints += elements[elementId]->numVertices();

            outFile << "POINTS " << numPoints << " float" << endl;
            for(unsigned int j = 0; j < elements[elementId]->numVertices(); j++)
            {
                outFile << elements[elementId]->vertex(j).x() << " " <<
                    elements[elementId]->vertex(j).y() << " " <<
                    elements[elementId]->vertex(j).z() << endl;
            }


            outFile << "CELLS " << 1 << " " << 1 + numPoints << endl;
            int currentPoint = 0;
            outFile << elements[elementId]->numVertices() << " ";
            elements[elementId]->outputVertexOrderForVTK(outFile, currentPoint);
            outFile << endl;

            currentPoint += elements[elementId]->numVertices();

            outFile << "CELL_TYPES " << 1 << endl;

            outFile << elements[elementId]->vtkCellType() << endl;

            outFile << "POINT_DATA " << numPoints << endl;
            outFile << "SCALARS scalars float 1" << endl;
            outFile << "LOOKUP_TABLE default" << endl;

            for(unsigned int j = 0; j < elements[elementId]->numVertices(); j++)
            {
                outFile << elements[elementId]->findScalarValueAtPoint(elements[elementId]->vertex(j)) << endl;
            }

            outFile.close();
        }

        void FiniteElementVolume::writeCellVolumeForVTK(const char* fileName)
        {
            std::cout << "Writing cell volume. " << fileName << std::endl;
            ofstream outFile(fileName, ios::out);
            outFile << "# vtk DataFile Version 2.0" << endl;
            outFile << "Volume as cells." << endl;
            outFile << "ASCII" << endl;
            outFile << "DATASET UNSTRUCTURED_GRID" << endl;

            int numPoints = 0;
            unsigned int i = 0;
            for(i = 0; i < elements.size(); i++)
            {
                numPoints += elements[i]->numVertices();
            }

            outFile << "POINTS " << numPoints << " float" << endl;
            for(i = 0; i < elements.size(); i++)
            {
                for(unsigned int j = 0; j < elements[i]->numVertices(); j++)
                {
                    outFile << elements[i]->vertex(j).x() << " " <<
                        elements[i]->vertex(j).y() << " " <<
                        elements[i]->vertex(j).z() << endl;
                }
            }

            outFile << "CELLS " << (unsigned int)elements.size() << " " << (unsigned int)elements.size() + numPoints << endl;
            int currentPoint = 0;
            for(i = 0; i < elements.size(); i++)
            {
                outFile << elements[i]->numVertices() << " ";
                elements[i]->outputVertexOrderForVTK(outFile, currentPoint);
                outFile << endl;

                currentPoint += elements[i]->numVertices();
            }

            outFile << "CELL_TYPES " << static_cast<unsigned int>(elements.size()) << endl;

            for(i = 0; i < elements.size(); i++)
            {
                outFile << elements[i]->vtkCellType() << endl;
            }

            outFile << "POINT_DATA " << numPoints << endl;
            outFile << "SCALARS scalars float 1" << endl;
            outFile << "LOOKUP_TABLE default" << endl;

            for(i = 0; i < elements.size(); i++)
            {
                for(unsigned int j = 0; j < elements[i]->numVertices(); j++)
                {
                    outFile << elements[i]->findScalarValueAtPoint(elements[i]->vertex(j)) << endl;
                }
            }

            outFile.close();
        }

        void writeSampledVolumeHeader(ofstream& outFile, int numSamples)
        {
        }



        bool FiniteElementVolume::writeHeader(FILE* file)
        {
            unsigned int headerLen = static_cast<unsigned int>(strlen(headerData));
            return headerLen+1 == fwrite(headerData, sizeof(char), headerLen+1, file);
        }

        bool CalculateValue(boost::shared_ptr<JacobiExtension::Polyhedron>& poly, const ElVis::WorldPoint& p, double& outValue)
        {
            ElVis::TensorPoint tp = poly->transformWorldToTensor(p);
            if( tp.a() >= -1.0001 && tp.a() <= 1.0001 &&
                tp.b() >= -1.0001 && tp.b() <= 1.0001 &&
                tp.c() >= -1.0001 && tp.c() <= 1.0001 )
            {
                outValue = poly->f(tp);
                return true;
            }
            return false;
        }

        double FiniteElementVolume::calculateScalarValue(const ElVis::WorldPoint& p)
        {
            double result;

            if( LastFoundElement && CalculateValue(LastFoundElement, p, result) )
            {
                return result;
            }

            for(unsigned int i = 0; i < elements.size(); ++i)
            {
                if( CalculateValue(elements[i], p, result) )
                {
                    LastFoundElement = elements[i];
                    return result;
                }
            }

            LastFoundElement.reset();
            throw OutOfBoundsPoint("Point out of bounds.");
        }

        bool FiniteElementVolume::sample(const ElVis::WorldPoint& p,
            double& scalarValue, ElVis::WorldVector& worldNormal,
            ElVis::TensorVector& tensorGradient,
            Matrix<double, 3, 3>& scalarHessian,
            Matrix<double, 3, 3>& mappingJacobian,
            Matrix<double, 3, 3>& mappingRHessian,
            Matrix<double, 3, 3>& mappingSHessian,
            Matrix<double, 3, 3>& mappingTHessian)
        {
            if( !itsBBoxes )
            {
                cerr << "ERROR - bounding box acceleration must be enabled for the sample method to work." << endl;
                return false;
            }

            ElVis::WorldVector v(1,1,1);
            v.Normalize();
            //rt::Ray theRay(p, v);
            //boost::shared_ptr<Polyhedron> poly = itsBBoxes->findIntersectedPolyhedron(theRay);

            //if( poly )
            //{
            //    ElVis::TensorPoint tp = poly->transformWorldToTensor(p);

            //    scalarValue = poly->f(tp);

            //    // We need the transposed, inverted Jacobian for both the
            //    // normal and the hessian, so just calculate it here to be used
            //    // both times.
            //    Matrix<double, 3, 3> Jinv;
            //    poly->calculateTensorToWorldSpaceMappingJacobian(tp, mappingJacobian);
            //    poly->transposeAndInvertMatrix(mappingJacobian, Jinv);

            //    tensorGradient = poly->calculateScalarFunctionTensorGradient(tp);
            //    worldNormal = poly->calculateScalarFunctionWorldGradient(tensorGradient, Jinv);

            //    poly->calculateScalarFunctionHessian(tp, scalarHessian);
            //    //poly->calculateTensorToWorldSpaceMappingHessian(tp, mappingXHessian, mappingYHessian, mappingZHessian);
            //    //poly->worldHessian(J, hessian);
            //    poly->calculateTensorToWorldSpaceMappingMiriahHessian(tp, mappingRHessian, mappingSHessian, mappingTHessian);
            //}
            //else
            //{
            //    return false;
            //    //std::strstream outString;
            //    //outString << p;
            //    //throw OutOfBoundsPoint(outString.str());
            //}

            return true;
        }

        //boost::shared_ptr<Polyhedron> FiniteElementVolume::findIntersectedPolyhedron(const rt::Ray& ray)
        //{
        //    if( itsBBoxes )
        //    {
        //        return itsBBoxes->findIntersectedPolyhedron(ray);
        //    }
        //    else
        //    {
        //        cerr << "findIntersectedPolyhedron not available without acceleration." << endl;
        //    }
        //    return boost::shared_ptr<Polyhedron>();
        //}


        //ElVis::WorldVector FiniteElementVolume::normal(const ElVis::WorldPoint&, const rt::IntersectionInfo& hit)
        //{
        //    return hit.getNormal();
        //}

        void FiniteElementVolume::calcOverallMinMaxScalar()
        {
            double tempMin = std::numeric_limits<double>::max();
            double tempMax = -std::numeric_limits<double>::min();

            for(unsigned int i = 0; i < elements.size(); i++)
            {
                if( elements[i]->getMax() > tempMax )
                    tempMax = elements[i]->getMax();
                if( elements[i]->getMin() < tempMin )
                    tempMin = elements[i]->getMin();
            }

            m_maxValue = tempMax;
            m_minValue = tempMin;
        }

        void FiniteElementVolume::calcOverallBoundingBox(ElVis::WorldPoint& min, ElVis::WorldPoint& max) const
        {
            min = ElVis::WorldPoint(FLT_MAX, FLT_MAX, FLT_MAX);
            max = ElVis::WorldPoint(-FLT_MAX, -FLT_MAX, -FLT_MAX);

            for(unsigned int i = 0; i < elements.size(); i++)
            {
                double min_x, min_y, min_z, max_x, max_y, max_z;
                elements[i]->elementBounds(min_x, min_y, min_z, max_x, max_y, max_z);

                if( min.x() > min_x )
                {
                    min.SetX(min_x);
                }

                if( min.y() > min_y )
                {
                    min.SetY(min_y);
                }

                if( min.z() > min_z )
                {
                    min.SetZ(min_z);
                }

                if( max.x() < max_x )
                {
                    max.SetX(max_x);
                }

                if( max.y() < max_y )
                {
                    max.SetY(max_y);
                }

                if( max.z() < max_z )
                {
                    max.SetZ(max_z);
                }
            }
        }

        /// Bad implementation
        double FiniteElementVolume::smallestX() const
        {
            ElVis::WorldPoint min, max;
            calcOverallBoundingBox(min, max);
            return min.x();
        }

        double FiniteElementVolume::smallestY() const
        {
            ElVis::WorldPoint min, max;
            calcOverallBoundingBox(min, max);
            return min.y();
        }

        double FiniteElementVolume::smallestZ() const
        {
            ElVis::WorldPoint min, max;
            calcOverallBoundingBox(min, max);
            return min.z();
        }

        double FiniteElementVolume::largestX() const
        {
            ElVis::WorldPoint min, max;
            calcOverallBoundingBox(min, max);
            return max.x();
        }

        double FiniteElementVolume::largestY() const
        {
            ElVis::WorldPoint min, max;
            calcOverallBoundingBox(min, max);
            return max.y();
        }

        double FiniteElementVolume::largestZ() const
        {
            ElVis::WorldPoint min, max;
            calcOverallBoundingBox(min, max);
            return max.z();
        }

        void FiniteElementVolume::get_minmax(double& min, double& max)
        {
            min = m_minValue;
            max = m_maxValue;
        }

        double FiniteElementVolume::get_min()
        {
            return m_minValue;
        }

        double FiniteElementVolume::get_max()
        {
            return m_maxValue;
        }

        //double FiniteElementVolume::do_findParametricIntersection(const rt::Ray& theRay)
        //   {
        //       intersectedObject = NULL;
        //       errorAtIntersection = 0.0;
        //       cn = 0.0;

        //       rt::IntersectionInfo hit;
        //       intersect(theRay, hit);

        //       if( hit.hasIntersection() )
        //       {
        //      return hit.getIntersectionT();
        //           // Fill in obj.intersection and obj.normal.
        //           //obj.intersection = obj.ray.origin() + obj.ray.direction()*hit.min_t;
        //           //ElVis::WorldVector* normal = (ElVis::WorldVector*)hit.scratchpad;
        //           //obj.normal = *normal;
        //           //obj.ray.t(hit.min_t);
        //           //obj.scalarValue = errorAtIntersection;
        //           //obj.conditionNumber = cn;
        //           //return true;
        //       }

        //       return -1;
        //   }

        void FiniteElementVolume::setMaterialForElements()
        {
            //std::for_each(elements.begin(), elements.end(), boost::bind(&MaterialObject::setMaterial, _1, getMaterial()));
        }

        //bool FiniteElementVolume::do_findIntersection(const rt::Ray& theRay, rt::IntersectionInfo& info) const
        //{
        //    //intersectedObject = boost::shared_ptr<Polyhedron>();
        //    //errorAtIntersection = 0.0;
        //    //cn = 0.0;

        //    boost::shared_ptr<rt::RayTraceableObject> prevObj = info.getIntersectedObject();

        //    if( itsBBoxes )
        //    {
        //        double t = itsBBoxes->intersectsIsovalueAt(theRay, isovalue(), info, shared_from_this());
        //        if( t > 0 && t < info.getIntersectionT() )
        //        {
        //            info.setIntersectionT(t);
        //            info.setIntersectedGroup(shared_from_this());
        //        }
        //    }
        //    else
        //    {
        //        for(unsigned int i = 0; i < elements.size(); i++)
        //        {
        //            double t = elements[i]->intersectsIsovalueAt(theRay, isovalue(), info, shared_from_this());
        //            if( t > 0 && t < info.getIntersectionT() )
        //            {
        //                info.setIntersectionT(t);
        //                info.setIntersectedGroup(shared_from_this());
        //                info.setIntersectedObject(elements[i]);
        //            }
        //        }
        //    }

        //    return info.getIntersectedGroup() == shared_from_this();
        //}

        //ElVis::WorldVector FiniteElementVolume::do_calculateNormal(const rt::IntersectionInfo& info) const
        //{
        //    assert(info.getIntersectedObject());

        //    boost::shared_ptr<Polyhedron> poly = boost::dynamic_pointer_cast<Polyhedron>(info.getIntersectedObject());
        //    assert(poly);

        //    const ElVis::WorldPoint& wp = info.getIntersectionPoint();
        //    ElVis::TensorPoint tp = poly->transformWorldToTensor(wp);

        //    ElVis::WorldVector n;
        //    poly->calculateNormal(tp, &n);
        //    return n;
        //}

        //void FiniteElementVolume::do_addObject(boost::shared_ptr<rt::RayTraceableObject> obj)
        //{
        //    boost::shared_ptr<Polyhedron> ptr = boost::dynamic_pointer_cast<Polyhedron>(obj);
        //    if( ptr )
        //    {
        //        elements.push_back(ptr);
        //        ptr->setMaterial(getMaterial());
        //    }
        //}

        //void FiniteElementVolume::do_addObjectGroup(boost::shared_ptr<const rt::RayTraceableObjectGroup> group)
        //{
        //}
    }
}

