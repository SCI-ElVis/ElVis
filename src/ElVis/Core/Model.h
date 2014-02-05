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
#include <ElVis/Core/Vector.hpp>
#include <ElVis/Core/FieldInfo.h>
#include <ElVis/Core/FaceInfo.h>
#include <ElVis/Core/OptiXBuffer.hpp>
#include <optixu/optixpp.h>
#include <vector>
#include <string>
#include <ElVis/Core/ElVisDeclspec.h>

#include <boost/enable_shared_from_this.hpp>

namespace ElVis
{
    class Scene;
    class SceneView;
    class Plugin;

    class Model
    {
        public:
            ELVIS_EXPORT explicit Model(const std::string& modelPath);
            ELVIS_EXPORT virtual ~Model();

            const WorldPoint& MinExtent() const { return m_minExtent; }
            const WorldPoint& MaxExtent() const { return m_maxExtent; }

            ELVIS_EXPORT unsigned int GetNumberOfElements() const { return DoGetNumberOfElements(); }

            ELVIS_EXPORT void GetFaceGeometry(boost::shared_ptr<Scene> scene, optixu::Context context, optixu::Geometry& faces);
            ELVIS_EXPORT std::vector<optixu::GeometryInstance> Get2DPrimaryGeometry(boost::shared_ptr<Scene> scene, optixu::Context context);
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

            ELVIS_EXPORT boost::shared_ptr<Plugin> GetPlugin() const { return m_plugin; }
            ELVIS_EXPORT void SetPlugin(boost::shared_ptr<Plugin> plugin) { m_plugin = plugin; }
            ELVIS_EXPORT std::string GetModelName() const; 
            ELVIS_EXPORT const std::string& GetPath() const { return m_modelPath; }

            /// \brief Copies the model to the given optix context
            ELVIS_EXPORT void CopyToOptiX(optixu::Context context);

            ELVIS_EXPORT virtual size_t GetNumberOfFaces() const;

            /// \brief Returns the given face definition.
            ELVIS_EXPORT virtual FaceInfo GetFaceDefinition(size_t globalFaceId) const;

            /// \brief Returns the number of vertices associated with the linear
            /// faces.
            ///
            /// This method returns the total number of vertices associated with
            /// linear faces.  Vertices shared among faces are counted only once.
            ELVIS_EXPORT virtual size_t GetNumberOfPlanarFaceVertices() const;

            /// Get the vertex for a linear face.
            ELVIS_EXPORT virtual WorldPoint GetPlanarFaceVertex(size_t vertexIdx) const;

            /// \brief Returns the number of vertices associated with a single linear face.
            ELVIS_EXPORT virtual size_t GetNumberOfVerticesForPlanarFace(size_t localFaceIdx) const;

            /// \brief Returns the vertex id in the range [0, GetNumberOfPlanarFaceVertices)
            ELVIS_EXPORT virtual size_t GetPlanarFaceVertexIndex(size_t localFaceIdx, size_t vertexId);

            /// \brief Returns the normal for the given face.
            ELVIS_EXPORT WorldVector GetPlanarFaceNormal(size_t localFaceIdx) const;

            ELVIS_EXPORT void CopyExtensionSpecificDataToOptiX(optixu::Context context);

            ELVIS_EXPORT optixu::Buffer GetFacesEnabledBuffer() { return m_facesEnabledBuffer; }

            ELVIS_EXPORT const std::vector<FaceInfo>& GetFaceInfo() const { return m_faceInfo; }

            ELVIS_EXPORT optixu::Geometry GetPlanarFaceGeometry() const { return m_planarFaceGeometry; }
            ELVIS_EXPORT optixu::Geometry GetCurvedFaceGeometry() const { return m_curvedFaceGeometry; }

        protected:
            void SetMinExtent(const WorldPoint& min) { m_minExtent = min; }
            void SetMaxExtent(const WorldPoint& max) { m_maxExtent = max; }

            /// The method listed below must be reimplemnted by each extension to provide
            /// the customized behavior required for ElVis.

            /// \brief Returns the number of scalar fields supported by the model.
            ///
            /// Simulations often generate multiple fields (e.g., pressure, temperature) in a
            /// single simulations.  This method returns how many fields the model has that
            /// can be visualized.
            virtual int DoGetNumFields() const = 0;

            /// \brief Returns the dimension of the model.
            /// \return 2 for two-dimensional models, and 3 for three-dimensional models.
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
            /// than the index parameter.  This id is used to uniquely identify the field
            /// in the extension.
            virtual FieldInfo DoGetFieldInfo(unsigned int index) const = 0;

            /// \brief Returns the number of boundary surfaces in the model.
            /// \return The number of boundary surfaces in the model. 
            ///
            /// A boundary surface is a collection of element faces.  A typical 
            /// use case is to specify a boundary surface that represents some 
            /// geometry of interest (e.g., wing surface) so it can be easily 
            /// selected and visualized.
            ELVIS_EXPORT virtual int DoGetNumberOfBoundarySurfaces() const = 0;


            /// \brief Obtains the specified boundary surface.  
            /// \param surfaceIndex The boundary surface in the range [0, DoGetNumberOfBoundarySurfaces())
            /// \param name Output parameter that will return the surface's name.
            /// \param faceIds Output parameter that returns the ids of each face that belongs to this surface.
            ///        Each entry is in the range [0, DoGetNumberOfFaces())
            ELVIS_EXPORT virtual void DoGetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds) = 0;

            /// \brief Calculates the axis-aligned bounding box of the model.
            /// \param min Output parameter storing the smallest point of the model's axis-aligned bounding box.
            /// \param max Output parameter storing the smallest point of the model's axis-aligned bounding box.
            ELVIS_EXPORT virtual void DoCalculateExtents(WorldPoint& min, WorldPoint& max) = 0;

            /// \brief Returns the number of elements in the model.
            virtual unsigned int DoGetNumberOfElements() const = 0;

            /// \brief Returns the prefix assigned to the extension's ptx files.  
            /// Must be the name of the extension as specified in the extensions CMakeLists.txt file.
            virtual const std::string& DoGetPTXPrefix() const = 0;

            ELVIS_EXPORT virtual std::vector<optixu::GeometryInstance> DoGet2DPrimaryGeometry(boost::shared_ptr<Scene> scene, optixu::Context context) = 0;
            ELVIS_EXPORT virtual optixu::Material DoGet2DPrimaryGeometryMaterial(SceneView* view) = 0;

            // 3D Face Information
            // When rendering 3D models, the ray-tracer uses the faces to
            // traverse the model during isosurface and volume rendering,
            // and to perform point location queries when rendering cut-surfaces.
            //
            // ElVis provides an implementation for linear triangles and
            // quadrilaterals, which are enabled by overriding the
            // functions listed below.  Curved faces require interface from the
            // extension.
            //
            // ElVis assumes that each face will have a global id that can 
            // be used to identify it.

            /// \brief Returns the total number of faces in the model.
            /// This number is used to assign a global id to each face in 
            /// the range [0, DoGetNumberOfFaces)
            ELVIS_EXPORT virtual size_t DoGetNumberOfFaces() const = 0;

            /// \brief Returns the given face definition.
            ELVIS_EXPORT virtual FaceInfo DoGetFaceDefinition(size_t globalFaceId) const = 0;

            /// \brief Returns the number of vertices associated with the linear
            /// faces.
            ///
            /// This method returns the total number of vertices associated with
            /// linear faces.  Vertices shared among faces are counted only once.
            ELVIS_EXPORT virtual size_t DoGetNumberOfPlanarFaceVertices() const = 0;

            /// Get the vertex for a linear face.
            ELVIS_EXPORT virtual WorldPoint DoGetPlanarFaceVertex(size_t vertexIdx) const = 0;

            /// \brief Returns the number of vertices associated with a single linear face.
            /// \param localFaceIdx - The index of the planar face in the range [0, numPlanarFaces)
            ELVIS_EXPORT virtual size_t DoGetNumberOfVerticesForPlanarFace(size_t localFaceIdx) const = 0;

            /// \brief Returns the vertex id in the range [0, GetNumberOfPlanarFaceVertices)
            /// \param localFaceIdx - The index of the planar face in the range [0, numPlanarFaces)
            /// \param vertexId - The face vertex in the range [0, DoGetNumberOfVerticesForPlanarFace)
            ELVIS_EXPORT virtual size_t DoGetPlanarFaceVertexIndex(size_t localFaceIdx, size_t vertexId) = 0;

            /// \brief Returns the normal for the given face.
            ELVIS_EXPORT virtual WorldVector DoGetPlanarFaceNormal(size_t localFaceIdx) const = 0;

            /// \brief Copies any data required by the extension to OptiX.
            /// 
            /// This method is responsible for copying any data needed by the extension
            /// to OptiX.  Field information and curved faces are generally copied 
            /// here.
            ELVIS_EXPORT virtual void DoCopyExtensionSpecificDataToOptiX(optixu::Context context) = 0;

        private:
            Model& operator=(const Model& rhs);
            Model(const Model& rhs);

            void copyFaceDefsToOptiX(optixu::Context context);
            void copyPlanarFaceVerticesToOptiX(optixu::Context context);
            void copyPlanarFaces(optixu::Context context);
            void copyCurvedFaces(optixu::Context context);
            void copyPlanarNormalsToOptiX(optixu::Context context);
            void createFaceIntersectionGeometry(optixu::Context context);

            std::string m_modelPath;
            boost::shared_ptr<Plugin> m_plugin;
            WorldPoint m_minExtent;
            WorldPoint m_maxExtent;
            WorldPoint m_center;

            std::vector<FaceInfo> m_faceInfo;
            size_t m_numPlanarFaces;
            size_t m_numCurvedFaces;

            OptiXBuffer<FaceInfo> m_faceIdBuffer;
            OptiXBuffer<uint> m_PlanarFaceToGlobalIdxMap;
            OptiXBuffer<uint> m_CurvedFaceToGlobalIdxMap;
            OptiXBuffer<int> m_GlobalFaceToPlanarFaceIdxMap;
            OptiXBuffer<int> m_GlobalFaceToCurvedFaceIdxMap;

            // Contains a PlanarFaceInfo object for each planar face in the model.
            // Indexing is by local planar face index.
            OptiXBuffer<PlanarFaceInfo> m_PlanarFaceInfoBuffer;

            OptiXBuffer<ElVisFloat4> m_VertexBuffer;
            ElVis::OptiXBuffer<ElVisFloat4> m_PlanarFaceNormalBuffer;

            optixu::Program m_planarFaceBoundingBoxProgram;
            optixu::Program m_planarFaceIntersectionProgram;

            optixu::Program m_curvedFaceBoundingBoxProgram;
            optixu::Program m_curvedFaceIntersectionProgram;

            optixu::Program m_faceClosestHitProgram;

            optixu::Buffer m_facesEnabledBuffer;
            optixu::Geometry m_planarFaceGeometry;
            optixu::Geometry m_curvedFaceGeometry; 

            optixu::Buffer m_planarFacesEnabledBuffer;
            optixu::Buffer m_curvedFacesEnabledBuffer;
    };

}

#endif //ELVIS_MODEL_H
