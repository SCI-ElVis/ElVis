//////////////////////////////////////////////////////////////////////////////////
////
////  File: hoPolyhedraBoundingBox.cpp
////
////
////  The MIT License
////
////  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
////  Department of Aeronautics, Imperial College London (UK), and Scientific
////  Computing and Imaging Institute, University of Utah (USA).
////
////  License for the specific language governing rights and limitations under
////  Permission is hereby granted, free of charge, to any person obtaining a
////  copy of this software and associated documentation files (the "Software"),
////  to deal in the Software without restriction, including without limitation
////  the rights to use, copy, modify, merge, publish, distribute, sublicense,
////  and/or sell copies of the Software, and to permit persons to whom the
////  Software is furnished to do so, subject to the following conditions:
////
////  The above copyright notice and this permission notice shall be included
////  in all copies or substantial portions of the Software.
////
////  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
////  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
////  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
////  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
////  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
////  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
////  DEALINGS IN THE SOFTWARE.
////
////  Description:
////
//////////////////////////////////////////////////////////////////////////////////
//#include <ElVis/Extensions/JacobiExtension/Isosurface.h>
//#include <ElVis/Extensions/JacobiExtension/PolyhedraBoundingBox.h>
//#include <ray_tracer/rtIntersectionInfo.h>
//#include <algorithm>
//#include <list>
////#include <iostream>
//#include <ElVis/Extensions/JacobiExtension/FiniteElementVolume.h>
//#include <boost/lexical_cast.hpp>
//#include <boost/tokenizer.hpp>
//#include <iostream>
//
//using namespace std;
//
//namespace JacobiExtension
//{
//    bool PolyhedraBoundingBox::intersectsBoundingBox(const rt::Ray& ray, double& nearestInt)
//    {
//        // rt::Ray tracing book page 65
//        t_near = -DBL_MAX;
//        t_far = DBL_MAX;
//
//        const ElVis::WorldPoint& origin = ray.getOrigin();
//        const ElVis::WorldVector& dir = ray.getDirection();
//
//        // X Plane
//        if( dir.x() == 0 )
//        {
//            if( origin.x() < itsMin.x() ||
//                origin.x() > itsMax.x() )
//            {
//                return false;
//            }
//        }
//        else
//        {
//            double t1 = (itsMin.x() - origin.x())/dir.x();
//            double t2 = (itsMax.x() - origin.x())/dir.x();
//
//            if( t1 > t2 )
//            {
//                double temp = t1;
//                t1 = t2;
//                t2 = temp;
//            }
//
//            if( t1 > t_near )
//            {
//                t_near = t1;
//            }
//
//            if( t2 < t_far )
//            {
//                t_far = t2;
//            }
//
//            if( t_near > t_far )
//            {
//                return false;
//            }
//
//            if( t_far < 0 )
//            {
//                return false;
//            }
//        }
//
//
//        // Y Plane
//        if( dir.y() == 0 )
//        {
//            if( origin.y() < itsMin.y() ||
//                origin.y() > itsMax.y() )
//            {
//                return false;
//            }
//        }
//        else
//        {
//            double t1 = (itsMin.y() - origin.y())/dir.y();
//            double t2 = (itsMax.y() - origin.y())/dir.y();
//
//            if( t1 > t2 )
//            {
//                double temp = t1;
//                t1 = t2;
//                t2 = temp;
//            }
//
//            if( t1 > t_near )
//            {
//                t_near = t1;
//            }
//
//            if( t2 < t_far )
//            {
//                t_far = t2;
//            }
//
//            if( t_near > t_far )
//            {
//                return false;
//            }
//
//            if( t_far < 0 )
//            {
//                return false;
//            }
//        }
//
//        // Z Plane
//        if( dir.z() == 0 )
//        {
//            if( origin.z() < itsMin.z() ||
//                origin.z() > itsMax.z() )
//            {
//                return false;
//            }
//        }
//        else
//        {
//            double t1 = (itsMin.z() - origin.z())/dir.z();
//            double t2 = (itsMax.z() - origin.z())/dir.z();
//
//            if( t1 > t2 )
//            {
//                double temp = t1;
//                t1 = t2;
//                t2 = temp;
//            }
//
//            if( t1 > t_near )
//            {
//                t_near = t1;
//            }
//
//            if( t2 < t_far )
//            {
//                t_far = t2;
//            }
//
//            if( t_near > t_far )
//            {
//                return false;
//            }
//
//            if( t_far < 0 )
//            {
//                return false;
//            }
//        }
//
//        nearestInt = t_near;
//        return true;
//    }
//
//    void PolyhedraBoundingBox::updateMin(const ElVis::WorldPoint& p)
//    {
//        if( p.x() < itsMin.x() ) itsMin.SetX(p.x()-.000001);
//        if( p.y() < itsMin.y() ) itsMin.SetY(p.y()-.000001);
//        if( p.z() < itsMin.z() ) itsMin.SetZ(p.z()-.000001);
//    }
//
//    void PolyhedraBoundingBox::updateMax(const ElVis::WorldPoint& p)
//    {
//        if( p.x() > itsMax.x() ) itsMax.SetX(p.x()+.000001);
//        if( p.y() > itsMax.y() ) itsMax.SetY(p.y()+.000001);
//        if( p.z() > itsMax.z() ) itsMax.SetZ(p.z()+.000001);
//    }
//
//    void PolyhedraBoundingBox::updateMin(double newMin)
//    {
//        if( newMin < itsScalarMin ) itsScalarMin = newMin;
//    }
//
//    void PolyhedraBoundingBox::updateMax(double newMax)
//    {
//        if( newMax > itsScalarMax ) itsScalarMax = newMax;
//    }
//
//
//    double PolyhedraBoundingBoxNode::intersectsIsovalueAt(const rt::Ray& ray, double isovalue, rt::IntersectionInfo& hit,
//                                boost::shared_ptr<const FiniteElementVolume> vol)
//    {
//        LeafMap sortedResult;
//        findLeavesWhichMayContainIsovalue(ray, vol->isovalue(), sortedResult);
//        if( sortedResult.size() > 0 )
//        {
//            //// Sort by t value.
//            //std::sort(result.begin(), result.end(), SortBoxesByIntersection());
//            //// Now go through these one at a time and stop on the first intersection.
//            //for(unsigned int i = 0; i < result.size(); i++)
//            //{
//            //  if( result[i]->intersectsIsovalue(ray, vol->isovalue(), hit, vol) )
//            //  {
//            //      return true;
//            //  }
//            //}
//
//            for(LeafMap::iterator iter = sortedResult.begin(); iter != sortedResult.end(); ++iter)
//            {
//                double t = (*iter).second->poly()->intersectsIsovalueAt(ray, vol->isovalue(), hit, vol);
//                if(  t > 0 )
//                {
//                    hit.setIntersectedObject((*iter).second->poly());
//                    return t;
//                }
//            }
//        }
//        return -1.0;
//    }
//
//    boost::shared_ptr<Polyhedron> PolyhedraBoundingBoxNode::findIntersectedPolyhedron(const rt::Ray& ray)
//    {
//        LeafVector result;
//        if( findAllIntersectedPolyhedra(ray, result) )
//        {
//            //cout << result.size() << endl;
//            std::sort(result.begin(), result.end(), SortBoxesByIntersection());
//            return result[0]->poly();
//
//            //for(size_t i = 0; i < result.size(); ++i)
//            //{
//            //  double min, max;
//            //  result[i]->poly()->findIntersectionWithGeometryInWorldSpace(ray, min, max);
//            //  if( min <= 0 && max >= 0 )
//            //  {
//            //      return result[i]->poly();
//            //  }
//            //}
//            //return NULL;
//            //return result.back()->poly();
//        }
//        else
//        {
//            return boost::shared_ptr<Polyhedron>();
//        }
//    }
//
//    bool PolyhedraBoundingBoxNode::findAllIntersectedPolyhedra(const rt::Ray& ray, LeafVector& leaves)
//    {
//        double t = -1.0;
//        if( intersectsBoundingBox(ray, t) )
//        {
//            for(unsigned int i = 0; i < itsChildren.size(); i++)
//            {
//                itsChildren[i]->findAllIntersectedPolyhedra(ray, leaves);
//            }
//        }
//
//        return leaves.size() > 0;
//    }
//
//    void PolyhedraBoundingBoxNode::findLeavesWhichMayContainIsovalue(const rt::Ray& ray, double isovalue,
//                LeafMap& sortedResult)
//    {
//        double t = -1.0;
//        if( containsIsovalue(isovalue) && intersectsBoundingBox(ray, t) )
//        {
//            for(unsigned int i = 0; i < itsChildren.size(); i++)
//            {
//                itsChildren[i]->findLeavesWhichMayContainIsovalue(ray, isovalue, sortedResult);
//            }
//        }
//    }
//
//    PolyhedraBoundingBoxLeaf::PolyhedraBoundingBoxLeaf(boost::shared_ptr<Polyhedron> poly) : PolyhedraBoundingBox(), itsPoly(poly)
//    {
//        ElVis::WorldPoint min, max;
//
//        itsPoly->elementBounds(min, max);
//        updateMin(min);
//        updateMax(max);
//        updateMin(itsPoly->getMin());
//        updateMax(itsPoly->getMax());
//    }
//
//    boost::shared_ptr<Polyhedron> PolyhedraBoundingBoxLeaf::findIntersectedPolyhedron(const rt::Ray& ray)
//    {
//        double min, max;
//        itsPoly->findIntersectionWithGeometryInWorldSpace(ray, min, max);
//        if( (min <= 0.0 && max >= 0.0) || min == max )
//        {
//            ElVis::TensorPoint tp = itsPoly->transformWorldToTensor(ray.getOrigin());
//			double a = tp[0];
//			double b = tp[1];
//			double c = tp[2];
//            if( a >= -1.000001 && a <= 1.000001 &&
//                b >= -1.000001 && b <= 1.000001 &&
//                c >= -1.000001 && c <= 1.000001 )
//            {
//                return itsPoly;
//            }
//            else
//            {
//                return boost::shared_ptr<Polyhedron>();
//            }
//        }
//        else
//        {
//            return boost::shared_ptr<Polyhedron>();
//        }
//    }
//
//    bool PolyhedraBoundingBoxLeaf::findAllIntersectedPolyhedra(const rt::Ray& ray, LeafVector& leaves)
//    {
//        double t = -1.0;
//        if( intersectsBoundingBox(ray, t) )
//        {
//            boost::shared_ptr<Polyhedron> poly = findIntersectedPolyhedron(ray);
//            if( poly )
//            {
//                leaves.push_back(this);
//                return true;
//            }
//            else
//            {
//                return false;
//            }
//        }
//        else
//        {
//            return false;
//        }
//    }
//
//    double PolyhedraBoundingBoxLeaf::intersectsIsovalueAt(const rt::Ray& ray, double isovalue, rt::IntersectionInfo& hit,
//                                                      boost::shared_ptr<const FiniteElementVolume> vol)
//    {
//        double t = -1.0;
//        if( containsIsovalue(isovalue) && intersectsBoundingBox(ray, t) )
//        {
//            double t = itsPoly->intersectsIsovalueAt(ray, isovalue, hit, vol);
//            if( t > 0.0 )
//            {
//                hit.setIntersectedObject(itsPoly);
//                return t;
//            }
//        }
//        else
//        {
//            /*
//            Vector n;
//            if( itsPoly->findIntersectionWithGeometryInWorldSpace(ray, t_near, t_far, n))
//            {
//                cerr << "Found intersection with geometry but not with bounding box." << endl;
//                cerr << "Bounding: " << getMin() << " - " << getMax() << endl;
//                cerr << "Vertices: " << endl;
//                for(int i = 0; i < itsPoly->numVertices(); i++)
//                {
//                    cerr << itsPoly->vertex(i) << endl;
//                }
//
//                cerr << "P1: " << ray.origin() << endl;
//                cerr << "P2: " << ray.origin() + 100.0*ray.direction();
//                //exit(0);
//            }
//            */
//        }
//
//        return -1.0;
//    }
//
//    void PolyhedraBoundingBoxLeaf::findLeavesWhichMayContainIsovalue(const rt::Ray& ray, double isovalue,
//                LeafMap& sortedResult)
//    {
//        double t = -1.0;
//        if( containsIsovalue(isovalue) && intersectsBoundingBox(ray, t) )
//        {
//            double min, max;
//            if( poly()->findIntersectionWithGeometryInWorldSpace(ray, min, max) )
//            {
//                sortedResult.insert(LeafMap::value_type(min, this));
//            }
//        }
//    }
//
//    void PolyhedraBoundingBox::writeBoundingBox(const char* fileName)
//    {
//        FILE* outFile = fopen(fileName, "wb");
//        int count = nodeCount();
//
//        if( fwrite(&count, sizeof(int), 1, outFile) != 1 )
//        {
//            cerr << "Error writing to " << fileName << endl;
//        }
//    }
//
//    // The format of the file is the following:
//    // TYPE     ID      TYPE    ID
//    // TYPE = ELEMENT or BOX
//    const std::string PolyhedraBoundingBox::ELEMENT("ELEMENT");
//    const std::string PolyhedraBoundingBox::BOX("BOX");
//
//    PolyhedraBoundingBox* createSimpleBoundingBoxFromFile(
//        PolyhedraVector& allPolyhedra,
//        const std::string& treeFile)
//    {
//        std::vector<PolyhedraBoundingBox*> elements;
//        for(size_t i = 0; i < allPolyhedra.size(); ++i)
//        {
//            PolyhedraBoundingBoxLeaf* leaf = new PolyhedraBoundingBoxLeaf(allPolyhedra[i]);
//            elements.push_back(leaf);
//        }
//
//        try
//        {
//            PolyhedraBoundingBoxNode* result = NULL;
//            std::vector<PolyhedraBoundingBox*> allBoxes;
//
//            std::ifstream inFile(treeFile.c_str());
//            while(!inFile.bad() && !inFile.eof() )
//            {
//                std::string inputLine;
//                std::getline(inFile, inputLine, '\n');
//                if(inFile.eof()) continue;
//
//                boost::tokenizer<> tokens(inputLine);
//                result = new PolyhedraBoundingBoxNode();
//
//                boost::tokenizer<>::iterator iter = tokens.begin();
//
//                for(int i = 0; i < 2; ++i)
//                {
//                    std::string type = *iter;
//                    ++iter;
//                    size_t number = boost::lexical_cast<size_t>(*iter);
//                    ++iter;
//
//                    if( type == PolyhedraBoundingBox::ELEMENT )
//                    {
//                        result->addChild(elements[number]);
//                    }
//                    else
//                    {
//                        result->addChild(allBoxes[number]);
//                    }
//                }
//                allBoxes.push_back(result);
//            }
//
//            inFile.close();
//            // The last entry is the root of the tree.
//            return result;
//        }
//        catch(...)
//        {
//            // The lexical cast can throw.
//            return NULL;
//        }
//    }
//
//    PolyhedraBoundingBox* createSimpleBoundingBox(PolyhedraVector& elements,
//        const std::string& treeFileName)
//    {
//        cout << "Loading from: " << treeFileName << endl;
//        std::ifstream treeFile(treeFileName.c_str());
//        std::string temp;
//        std::getline(treeFile, temp, '\n');
//        bool exists = !treeFile.fail();
//        treeFile.close();
//        if( exists )
//        {
//            cout << "Creating from acceleration file." << endl;
//            PolyhedraBoundingBox* result = createSimpleBoundingBoxFromFile(elements, treeFileName);
//            if( result )
//            {
//                return result;
//            }
//        }
//
//        // Either the file does not exist or it is invalid.  Reconstruct
//        // from first principles.
//        cout << "Creating from volume file." << endl;
//        std::ofstream outFile(treeFileName.c_str());
//
//        // Needed to assign numbers.
//        std::list<PolyhedraBoundingBox*> allBoundingBoxes;
//
//        std::list<PolyhedraBoundingBox*> candidates;
//        for(size_t i = 0; i < elements.size(); i++)
//        {
//            PolyhedraBoundingBoxLeaf* leaf = new PolyhedraBoundingBoxLeaf(elements[i]);
//            candidates.push_back(leaf);
//        }
//
//        std::list<PolyhedraBoundingBox*> nextCandidateList;
//        bool newBoxCreated = true;
//        while( candidates.size() > 1 && newBoxCreated )
//        {
//            //cout << "Iteration with " << candidates.size() << endl;
//            newBoxCreated = false;
//            for(std::list<PolyhedraBoundingBox*>::iterator iter1 = candidates.begin(); iter1 != candidates.end(); iter1++)
//            {
//                PolyhedraBoundingBox* box1 = *iter1;
//
//                double smallestVolume = FLT_MAX;
//                std::list<PolyhedraBoundingBox*>::iterator smallestIter;
//
//                for(std::list<PolyhedraBoundingBox*>::iterator iter2 = iter1; iter2 != candidates.end(); iter2++)
//                {
//                    if( iter1 != iter2 )
//                    {
//                        PolyhedraBoundingBox* box2 = *iter2;
//                        double temp =calculateCombinedBoundingVolume(box1, box2);
//
//                        if( temp < smallestVolume )
//                        {
//                            smallestVolume = temp;
//                            smallestIter = iter2;
//                        }
//                    }
//                }
//
//                // This isn't necessarily optimal.
//                // Whichever two had the smallest volume are combined.
//                // If we found a good match then we create a new node, otherwise
//                // we just add the existing node to the next list.
//                if( smallestVolume != FLT_MAX )
//                {
//                    PolyhedraBoundingBoxNode* node = new PolyhedraBoundingBoxNode();
//                    node->addChild(*iter1);
//                    node->addChild(*smallestIter);
//
//                    allBoundingBoxes.push_back(node);
//                    node->setIndex(allBoundingBoxes.size()-1);
//
//                    outFile << (*iter1)->type() << "\t" << static_cast<unsigned int>((*iter1)->index()) <<
//                        "\t" << (*smallestIter)->type() << "\t" << static_cast<unsigned int>((*smallestIter)->index()) << endl;
//
//                    nextCandidateList.push_back(node);
//                    candidates.erase(smallestIter);
//                    newBoxCreated = true;
//                }
//                else
//                {
//                    nextCandidateList.push_back(*iter1);
//                }
//            }
//
//            // The next candidate list becomes the current candidate list.
//            candidates.clear();
//            candidates = nextCandidateList;
//            nextCandidateList.clear();
//        }
//
//        if( candidates.size() == 0 )
//        {
//            cerr << "Serious error creating bounding box." << endl;
//            exit(1);
//        }
//
//        if( candidates.size() == 1 )
//        {
//            PolyhedraBoundingBox* node = candidates.front();
//            //cout << node->numChildren() << endl;
//            outFile.close();
//            return node;
//        }
//        else
//        {
//            // Now create the hierarchy.
//            // Take each element and check to see how it combines with others.
//            PolyhedraBoundingBoxNode* node = new PolyhedraBoundingBoxNode();
//
//            for(std::list<PolyhedraBoundingBox*>::iterator firstIter = candidates.begin(); firstIter != candidates.end(); firstIter++)
//            {
//                node->addChild(*firstIter);
//                outFile << (*firstIter)->type() << "\t" << static_cast<unsigned int>((*firstIter)->index()) << "\t";
//            }
//            //cout << node->numChildren() << endl;
//            outFile << endl;
//            outFile.close();
//            return node;
//        }
//
//
//    }
//
//    double calculateCombinedBoundingVolume(PolyhedraBoundingBox* box1,
//                                        PolyhedraBoundingBox* box2)
//    {
//        double small_x = (box1->getMin().x() < box2->getMin().x() ? box1->getMin().x() : box2->getMin().x());
//        double small_y = (box1->getMin().y() < box2->getMin().y() ? box1->getMin().y() : box2->getMin().y());
//        double small_z = (box1->getMin().z() < box2->getMin().z() ? box1->getMin().z() : box2->getMin().z());
//
//        double large_x = (box1->getMax().x() > box2->getMax().x() ? box1->getMax().x() : box1->getMax().x());
//        double large_y = (box1->getMax().y() > box2->getMax().y() ? box1->getMax().y() : box1->getMax().y());
//        double large_z = (box1->getMax().z() > box2->getMax().z() ? box1->getMax().z() : box1->getMax().z());
//
//        return (large_x-small_x)*(large_y-small_y)*(large_z-small_z);
//    }
//}
