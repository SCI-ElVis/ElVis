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

#ifndef ELVIS_MODEL_H
#define ELVIS_MODEL_H

#include <ElVis/Core/Point.hpp>
#include <ElVis/Core/FieldInfo.h>

#include <optixu/optixpp.h>
#include <vector>
#include <ElVis/Core/ElVisDeclspec.h>

namespace ElVis
{
    class Scene;
    class SceneView;

    class Model
    {
        public:
            ELVIS_EXPORT Model();
            ELVIS_EXPORT virtual ~Model();

            const WorldPoint& MinExtent() const { return m_minExtent; }
            const WorldPoint& MaxExtent() const { return m_maxExtent; }

            // These are for the conversion extension and shouldn't be here.
//            ELVIS_EXPORT unsigned int GetNumberOfPoints() const { return DoGetNumberOfPoints(); }
//            ELVIS_EXPORT WorldPoint GetPoint(unsigned int id) const { return DoGetPoint(id); }

            ELVIS_EXPORT unsigned int GetNumberOfElements() const { return DoGetNumberOfElements(); }

            /// \brief Generates the OptiX geometry nodes necessary to perform ray/element intersections.
            ELVIS_EXPORT std::vector<optixu::GeometryGroup> GetPointLocationGeometry(Scene* scene, optixu::Context context);
            ELVIS_EXPORT void GetFaceGeometry(Scene* scene, optixu::Context context, optixu::Geometry& faces);
            ELVIS_EXPORT std::vector<optixu::GeometryInstance> Get2DPrimaryGeometry(Scene* scene, optixu::Context context);
            ELVIS_EXPORT virtual optixu::Material Get2DPrimaryGeometryMaterial(SceneView* view) { return DoGet2DPrimaryGeometryMaterial(view); }

            ELVIS_EXPORT int GetModelDimension() const; 

            // This doesn't really belong in the model class, but is a good, quick place to put it for now.
            // It really should be a part of the SceneView class, but the individual extensions don't currently 
            // created customized SceneViews.
            ELVIS_EXPORT const std::string& GetPTXPrefix() const { return DoGetPTXPrefix(); }

            ELVIS_EXPORT void CalculateExtents();
            ELVIS_EXPORT const WorldPoint& GetMidpoint();

            ELVIS_EXPORT int GetNumFields() const { return DoGetNumFields(); }
            ELVIS_EXPORT FieldInfo GetFieldInfo(unsigned int index) const { return DoGetFieldInfo(index); }

            ELVIS_EXPORT int GetNumberOfBoundarySurfaces() const;
            ELVIS_EXPORT void GetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds);

        protected:
            void SetMinExtent(const WorldPoint& min) { m_minExtent = min; }
            void SetMaxExtent(const WorldPoint& max) { m_maxExtent = max; }

            /// The method listed below must be reimplemnted by each extension to provide
            /// the customized behavior required for ElVis.

            /// \brief Returns the number of scalar field supported by the model.
            ///
            /// Simulations often generate multiple fields (e.g., pressure, temperature) in a
            /// single simulations.  This method returns how many fields the model has that
            /// can be visualized.
            virtual int DoGetNumFields() const = 0;

            virtual int DoGetModelDimension() const = 0;

            /// \brief Returns information about the requested field.
            /// \param index A value in the range [0, DoGetNumFields()).
            /// \return Information about the given field.
            ///
            /// This method returns additional information about the requested field.
            /// The values of the FieldInfo struct are populated as follows:
            ///
            /// FieldInfo::Name - A user-readable name that is used in the gui to identify
            /// the field.
            ///
            /// FieldInfo::Id - An unique identifier for the field.  This can be different
            /// than the index parameter.  This id can be used to uniquely identify the field
            /// in the extension.
            virtual FieldInfo DoGetFieldInfo(unsigned int index) const = 0;

            /// \brief Returns the number of boundary surfaces in the model.
            ///
            /// Some simulations are specifically interested in the behavior around certain
            /// surfaces.  It is convenient to be able to access these surfaces directly in
            /// ElVis.  If the model supports boundary surfaces, this method indicates how
            /// many surfaces there are.
            ELVIS_EXPORT virtual int DoGetNumberOfBoundarySurfaces() const = 0;


            /// \brief Returns the name of the given boundary surface and the element faces associated with it.
            /// \param surfaceIndex The id of the requested surface, in the range [0, DoGetNumberOfBoundarySurfaces()).
            /// \param name Output parameter that will return the surface's name.
            /// \param faceIds Output parameter that returns the ids of each face that belongs to this surface.
            ELVIS_EXPORT virtual void DoGetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds) = 0;

            /// \brief Calculates the axis-aligned bounding box of the model.
            /// \param min Output parameter storing the smallest point of the model's axis-aligned bounding box.
            /// \param max Output parameter storing the smallest point of the model's axis-aligned bounding box.
            ELVIS_EXPORT virtual void DoCalculateExtents(WorldPoint& min, WorldPoint& max) = 0;

            /// \brief Returns the number of elements in the model.
            virtual unsigned int DoGetNumberOfElements() const = 0;

            virtual const std::string& DoGetPTXPrefix() const = 0;

            ELVIS_EXPORT virtual std::vector<optixu::GeometryGroup> DoGetPointLocationGeometry(Scene* scene, optixu::Context context) = 0;
            ELVIS_EXPORT virtual void DoGetFaceGeometry(Scene* scene, optixu::Context context, optixu::Geometry& faces) = 0;
            ELVIS_EXPORT virtual std::vector<optixu::GeometryInstance> DoGet2DPrimaryGeometry(Scene* scene, optixu::Context context) = 0;
            ELVIS_EXPORT virtual optixu::Material DoGet2DPrimaryGeometryMaterial(SceneView* view) = 0;


            // These are for the conversion extension and shouldn't be here.
            //            virtual unsigned int DoGetNumberOfPoints() const = 0;
            //            virtual WorldPoint DoGetPoint(unsigned int id) const = 0;

        private:
            Model& operator=(const Model& rhs);
            Model(const Model& rhs);

            WorldPoint m_minExtent;
            WorldPoint m_maxExtent;
            WorldPoint m_center;
    };

}

#endif //ELVIS_MODEL_H
