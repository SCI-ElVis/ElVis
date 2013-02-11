//////////////////////////////////////////////////////////////////////////////////
////
////  File: hoPolyhedraBoundingBox.h
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
//
//#ifndef _POLYHEDRA_BOUNDING_BOX_H_
//#define _POLYHEDRA_BOUNDING_BOX_H_
//
//
//#include <ray_tracer/rtPoint.hpp>
//#include <ElVis/Extensions/JacobiExtension/PointTransformations.hpp>
//#include <ElVis/Extensions/JacobiExtension/FiniteElementVolume.h>
//#include <vector>
//#include <ray_tracer/rtIntersectionInfo.h>
//#include <map>
//
//using std::vector;
//
//
//namespace JacobiExtension
//{
//    class PolyhedraBoundingBox;
//    class PolyhedraBoundingBoxLeaf;
//
//    typedef vector<PolyhedraBoundingBox*> BoxVector;
//    typedef vector<PolyhedraBoundingBoxLeaf*> LeafVector;
//    typedef std::multimap<double, PolyhedraBoundingBoxLeaf*> LeafMap;
//
//    // An axis aligned bounding box.
//    class PolyhedraBoundingBox
//    {
//        public:
//            PolyhedraBoundingBox() : t_near(FLT_MAX), t_far(-FLT_MAX), itsScalarMin(FLT_MAX), itsScalarMax(-FLT_MAX),
//                itsMin(FLT_MAX, FLT_MAX, FLT_MAX), itsMax(-FLT_MAX, -FLT_MAX, -FLT_MAX)
//            {
//            }
//
//            PolyhedraBoundingBox(const ElVis::WorldPoint& min, const ElVis::WorldPoint& max) : itsMin(min), itsMax(max) {}
//            virtual ~PolyhedraBoundingBox() {}
//
//            //virtual double intersectsIsovalueAt(const rt::Ray& ray, double isovalue, rt::IntersectionInfo& hit,
//            //    boost::shared_ptr<const FiniteElementVolume> vol) = 0;
//
//            virtual void findLeavesWhichMayContainIsovalue(const rt::Ray& ray, double isovalue,
//                LeafMap& leafMap) = 0;
//
//            // Checks to see if the ray intersects the geometry of the bounding
//            // box.
//            bool intersectsBoundingBox(const rt::Ray& ray, double& nearestInt);
//            bool containsIsovalue(double val)
//            {
//                return val >= scalarMin() && val <= scalarMax();
//            }
//            virtual boost::shared_ptr<Polyhedron> findIntersectedPolyhedron(const rt::Ray& ray) = 0;
//            virtual bool findAllIntersectedPolyhedra(const rt::Ray& ray, LeafVector& leaves) = 0;
//
//            const ElVis::WorldPoint& getMin() const { return itsMin; }
//            const ElVis::WorldPoint& getMax() const { return itsMax; }
//
//            double get_t_near() const { return t_near; }
//            double get_t_far() const { return t_far; }
//
//            double scalarMin() const { return itsScalarMin; }
//            double scalarMax() const { return itsScalarMax; }
//
//            void writeBoundingBox(const char* fileName);
//            virtual int nodeCount() = 0;
//
//            // Outputs ELEMENT or BOX depending on the type.
//            virtual const std::string& type() const = 0;
//
//            // Returns an index where we can find this box in a list.
//            // This is highly coupled to the creation algorithm.
//            virtual size_t index() const = 0;
//
//            static const std::string ELEMENT;
//            static const std::string BOX;
//
//        protected:
//            void updateMin(const ElVis::WorldPoint& p);
//            void updateMax(const ElVis::WorldPoint& p);
//            void updateMin(double newMin);
//            void updateMax(double newMax);
//
//            double t_near, t_far;
//            double itsScalarMin, itsScalarMax;
//
//        private:
//            ElVis::WorldPoint itsMin;
//            ElVis::WorldPoint itsMax;
//
//    };
//
//    class SortBoxesByIntersection
//    {
//        public:
//            bool operator()(const PolyhedraBoundingBox* lhs, const PolyhedraBoundingBox* rhs)
//            {
//                return lhs->get_t_near() < rhs->get_t_near();
//            }
//    };
//
//    class PolyhedraBoundingBoxNode : public PolyhedraBoundingBox
//    {
//        public:
//            PolyhedraBoundingBoxNode() : PolyhedraBoundingBox(),
//                m_id(0) {}
//            virtual ~PolyhedraBoundingBoxNode() {}
//
//            virtual double intersectsIsovalueAt(const rt::Ray& ray, double isovalue, rt::IntersectionInfo& hit,
//                boost::shared_ptr<const FiniteElementVolume> vol);
//
//            virtual void findLeavesWhichMayContainIsovalue(const rt::Ray& ray, double isovalue,
//                LeafMap& sortedResult);
//
//            virtual boost::shared_ptr<Polyhedron> findIntersectedPolyhedron(const rt::Ray& ray);
//            virtual bool findAllIntersectedPolyhedra(const rt::Ray& ray, LeafVector& leaves);
//
//            void addChild(PolyhedraBoundingBox* child)
//            {
//                itsChildren.push_back(child);
//                updateMin(child->getMin());
//                updateMax(child->getMax());
//                updateMin(child->scalarMin());
//                updateMax(child->scalarMax());
//            }
//            size_t numChildren() const { return itsChildren.size(); }
//
//            virtual int nodeCount()
//            {
//                int result = 1;
//                for(unsigned int i = 0; i < itsChildren.size(); i++)
//                {
//                    result += itsChildren[i]->nodeCount();
//                }
//                return result;
//            }
//
//            // Outputs ELEMENT or BOX depending on the type.
//            virtual const std::string& type() const { return BOX; }
//
//            // Returns an index where we can find this box in a list.
//            // This is highly coupled to the creation algorithm.
//            virtual size_t index() const { return m_id; }
//            void setIndex(size_t val) { m_id = val; }
//
//        private:
//            BoxVector itsChildren;
//            size_t m_id;
//    };
//
//    class PolyhedraBoundingBoxLeaf : public PolyhedraBoundingBox
//    {
//        public:
//            PolyhedraBoundingBoxLeaf(boost::shared_ptr<Polyhedron> poly);
//            virtual ~PolyhedraBoundingBoxLeaf() {}
//
//            virtual double intersectsIsovalueAt(const rt::Ray& ray, double isovalue, rt::IntersectionInfo& hit,
//                boost::shared_ptr<const FiniteElementVolume> vol);
//
//            virtual void findLeavesWhichMayContainIsovalue(const rt::Ray& ray, double isovalue,
//                LeafMap& sortedResult);
//            virtual boost::shared_ptr<Polyhedron> findIntersectedPolyhedron(const rt::Ray& ray);
//            virtual bool findAllIntersectedPolyhedra(const rt::Ray& ray, LeafVector& leaves);
//            boost::shared_ptr<Polyhedron> poly() const { return itsPoly; }
//            virtual int nodeCount()
//            {
//                return 1;
//            }
//
//            // Outputs ELEMENT or BOX depending on the type.
//            virtual const std::string& type() const { return ELEMENT; }
//
//            // Returns an index where we can find this box in a list.
//            // This is highly coupled to the creation algorithm.
//            virtual size_t index() const { return itsPoly->id(); }
//        private:
//            boost::shared_ptr<Polyhedron> itsPoly;
//    };
//
//    // Creates a simple scheme where each element is surrounded by a box
//    // and then combines with other elements to form the tree.  There will
//    // be a max of two children per node, the boxes will be axis aligned,
//    // and the fitness function is the smallest bounding box.
//    //
//    // The treeFileName contains the name of a file to read/store
//    // the structure of the bounding boxes.  This allows for
//    // faster image generation.
//    PolyhedraBoundingBox* createSimpleBoundingBox(PolyhedraVector& elements,
//        const std::string& treeFileName);
//
//    double calculateCombinedBoundingVolume(PolyhedraBoundingBox* box1,
//                                        PolyhedraBoundingBox* box2);
//}
//
//#endif
