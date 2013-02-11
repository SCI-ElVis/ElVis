///////////////////////////////////////////////////////////////////////////////
//
// The MIT License
//
// Copyright (c) 2006 Scientific Computing and Imaging Institute,
// University of Utah (USA)
//
// License for the specific language governing rights and limitations under
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ELVIS_OCTREE_HPP
#define ELVIS_OCTREE_HPP

#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/Float.h>

#include <iostream>

namespace ElVis
{
    // T is an standard library container
    // This Octree is for static scenes.
    template<typename T>
    class Octree
    {
        public:
            // Create a new, base level octree with all objects in begin..end.
            template<typename Iter>
            Octree(const Iter& begin, const Iter& end, unsigned int min) :
                m_minObjects(min)
            {
                WorldPoint p0, p1;
                CalculateExtents(begin, end, p0, p1);
                m_p0 = p0;
                m_p1 = p1;

                for(Iter iter = begin; iter != end; ++iter)
                {
                    m_objects.push_back(*iter);
                }

                PopulateSubTrees();
            }

            // Create a new octree with all of the objects in the specified bounding box.
            // keepAll is a bit of hack.  Top level trees pass this in when it is OK for this
            // tree to keep all elements.
            template<typename Iter>
            Octree(const Iter& begin, const Iter& end, unsigned int min,
                const WorldPoint& p0, const WorldPoint& p1, bool keepAll = false) :
                m_p0(p0),
                m_p1(p1),
                m_minObjects(min)
            {
                PopulateOctree(p0, p1, begin, end, keepAll);
            }

            T* GetObject(const ElVis::WorldPoint& p)
            {
                if( !InBox(p, m_p0, m_p1) )
                {
                    return 0;
                }

                if( m_children.empty() )
                {
                    for(typename std::vector<T*>::iterator iter = m_objects.begin(); iter != m_objects.end(); ++iter)
                    {
                        if( (*iter)->ContainsPoint(p) )
                        {
                            return *iter;
                        }
                    }

                    return 0;
                }
                else
                {
                    for(typename std::list<Octree<T>* >::iterator iter = m_children.begin(); iter != m_children.end(); ++iter)
                    {
                        T* childResult = (*iter)->GetObject(p);
                        if( childResult ) return childResult;
                    }
                }
                return 0;
            }

            const ElVis::WorldPoint& GetMin() const { return m_p0; }
            const ElVis::WorldPoint& GetMax() const { return m_p1; }

            unsigned int GetNumObjects() const { return m_objects.size(); }

        private:
            Octree(const Octree<T>& rhs);
            Octree<T>& operator=(const Octree<T>& rhs);

            template<typename Iter>
            void CalculateExtents(const Iter& begin, const Iter& end, WorldPoint& p0, WorldPoint& p1)
            {
                p0 = WorldPoint(std::numeric_limits<ElVisFloat>::max(), std::numeric_limits<ElVisFloat>::max(), std::numeric_limits<ElVisFloat>::max());
                p1 = WorldPoint(-std::numeric_limits<ElVisFloat>::max(), -std::numeric_limits<ElVisFloat>::max(), -std::numeric_limits<ElVisFloat>::max());

                for(Iter iter = begin; iter != end; ++iter)
                {
                    p0 = Min(p0, (*iter)->GetMinExtent());
                    p1 = Max(p1, (*iter)->GetMaxExtent());
                }
            }

            template<typename Iter>
            void PopulateOctree(const WorldPoint& p0, const WorldPoint& p1, const Iter& begin, const Iter& end, bool canKeepAll = false)
            {
                m_p0 = p0;
                m_p1 = p1;

                // Only include those elements that have a portion in this tree.
                for(Iter iter = begin; iter != end; ++iter)
                {
                    WorldPoint min = (*iter)->GetMinExtent();
                    WorldPoint max = (*iter)->GetMaxExtent();


                    if( BoxesIntersect(min, max, m_p0, m_p1) )
                    {
                        m_objects.push_back(*iter);
                    }
                }

                // If any child has the same number of objects as this level, then 
                // we'll hit an infinite loop.
                if( !canKeepAll && this->GetNumObjects() == std::distance(begin, end) )
                {
                    return;
                }

                if( m_objects.size() > m_minObjects )
                {
                    PopulateSubTrees();
                }
            }

            bool HasElement(unsigned int id)
            {
                for(unsigned int i = 0; i < m_objects.size(); ++i)
                {
                    if( m_objects[i]->GetId() == id ) return true;
                }
                return false;
            }
            T* GetElement(unsigned int id)
            {
                for(unsigned int i = 0; i < m_objects.size(); ++i)
                {
                    if( m_objects[i]->GetId() == id ) return m_objects[i];
                }
                return 0;
            }

            // Assumes that the extents for the current tree have been set and the items belonging 
            // to this level are in m_objects.
            void PopulateSubTrees()
            {
                // Create children trees.
                WorldPoint mid = m_p0 + (m_p1-m_p0)/2.0;

                Octree<T>* boxes[8];
                boxes[0] = new Octree<T>(m_objects.begin(), m_objects.end(), m_minObjects,
                    WorldPoint(m_p0.x()-.001, m_p0.y()-.001, m_p0.z()-.001), WorldPoint(mid.x()+.001, mid.y()+.001, mid.z()+.001));
                boxes[1] = new Octree<T>(m_objects.begin(), m_objects.end(), m_minObjects,
                    WorldPoint(mid.x()-.001, m_p0.y()-.001, m_p0.z()-.001), WorldPoint(m_p1.x()+.001, mid.y()+.001, mid.z()+.001));
                boxes[2] = new Octree<T>(m_objects.begin(), m_objects.end(), m_minObjects,
                    WorldPoint(m_p0.x()-.001, mid.y()-.001, m_p0.z()-.001), WorldPoint(mid.x()+.001, m_p1.y()+.001, mid.z()+.001));
                boxes[3] = new Octree<T>(m_objects.begin(), m_objects.end(), m_minObjects,
                    WorldPoint(mid.x()-.001, mid.y()-.001, m_p0.z()-.001), WorldPoint(m_p1.x()+.001, m_p1.y()+.001, mid.z()+.001));

                boxes[4] = new Octree<T>(m_objects.begin(), m_objects.end(), m_minObjects,
                    WorldPoint(m_p0.x()-.001, m_p0.y()-.001, mid.z()-.001), WorldPoint(mid.x()+.001, mid.y()+.001, m_p1.z()+.001));
                boxes[5] = new Octree<T>(m_objects.begin(), m_objects.end(), m_minObjects,
                    WorldPoint(mid.x()-.001, m_p0.y()-.001, mid.z()-.001), WorldPoint(m_p1.x()+.001, mid.y()+.001, m_p1.z()+.001));
                boxes[6] = new Octree<T>(m_objects.begin(), m_objects.end(), m_minObjects,
                    WorldPoint(m_p0.x()-.001, mid.y()-.001, mid.z()-.001), WorldPoint(mid.x()+.001, m_p1.y()+.001, m_p1.z()+.001));
                boxes[7] = new Octree<T>(m_objects.begin(), m_objects.end(), m_minObjects,
                    WorldPoint(mid.x()-.001, mid.y()-.001, mid.z()-.001), WorldPoint(m_p1.x()+.001, m_p1.y()+.001, m_p1.z()+.001));


                for(unsigned int i = 0; i < 8; ++i)
                {
                    if( boxes[i]->GetNumObjects() > 0 )
                    {
                        m_children.push_back(boxes[i]);
                    }
                    else
                    {
                        delete boxes[i];
                    }
                }     
            }
            bool InBox(const WorldPoint& testPoint, const WorldPoint& p0, const WorldPoint& p1)
            {
                return  testPoint.x() >= p0.x() &&
                        testPoint.y() >= p0.y() &&
                        testPoint.z() >= p0.z() &&
                        testPoint.x() <= p1.x() &&
                        testPoint.y() <= p1.y() &&
                        testPoint.z() <= p1.z();
            }

            bool LineSegmentsOverlap(double x0, double x1, double y0, double y1)
            {
                return x1 >= y0 && x0 <= y1;
            }

            bool BoxesIntersect(const WorldPoint& box0Min, const WorldPoint& box0Max,
                                const WorldPoint& box1Min, const WorldPoint& box1Max)
            {
                return LineSegmentsOverlap(box0Min.x(), box0Max.x(), box1Min.x(), box1Max.x()) &&
                    LineSegmentsOverlap(box0Min.y(), box0Max.y(), box1Min.y(), box1Max.y()) &&
                    LineSegmentsOverlap(box0Min.z(), box0Max.z(), box1Min.z(), box1Max.z());
            }


            WorldPoint m_p0;
            WorldPoint m_p1;

            // The objects that belong at this level.
            std::vector<T*> m_objects;

            // The 8 children of this octree, unless it is a leaf.
            std::list<Octree<T>* > m_children;

            unsigned int m_minObjects;
    };
}

#endif //ELVIS_OCTREE_HPP

