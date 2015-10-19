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

#ifndef ELVIS_NEKPP_MODEL_H
#define ELVIS_NEKPP_MODEL_H

#include <vector>
#include <boost/foreach.hpp>
#include <boost/utility.hpp>
#include <boost/filesystem.hpp>

#include <optixu/optixpp.h>

#include <ElVis/Core/Model.h>
#include <ElVis/Core/OptiXBuffer.hpp>
#include <ElVis/Core/Float.h>

#include <SpatialDomains/MeshGraph.h>
#include <MultiRegions/ExpList.h>

namespace ElVis
{
namespace NektarPlusPlusExtension
{

    class NektarModel : public Model
    {
    public:
        NektarModel(const std::string& modelPath);
        virtual ~NektarModel();

        static const std::string prefix;

        void LoadVolume(const std::string& filePath);
        void LoadFields(const boost::filesystem::path& fieldFile);

    protected:
        virtual int DoGetNumFields() const;
        virtual int DoGetModelDimension() const;
        virtual FieldInfo DoGetFieldInfo(unsigned int index) const;
        virtual int DoGetNumberOfBoundarySurfaces() const;
        virtual void DoGetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds);
        virtual void DoCalculateExtents(WorldPoint& min, WorldPoint& max);
        virtual unsigned int DoGetNumberOfElements() const;
        virtual const std::string& DoGetPTXPrefix() const;
        virtual std::vector<optixu::GeometryInstance> DoGet2DPrimaryGeometry(boost::shared_ptr<Scene> scene, optixu::Context context);
        virtual optixu::Material DoGet2DPrimaryGeometryMaterial(SceneView* view);
        virtual size_t DoGetNumberOfFaces() const;
        virtual FaceInfo DoGetFaceDefinition(size_t globalFaceId) const;
        virtual size_t DoGetNumberOfPlanarFaceVertices() const;
        virtual WorldPoint DoGetPlanarFaceVertex(size_t vertexIdx) const;
        virtual size_t DoGetNumberOfVerticesForPlanarFace(size_t localFaceIdx) const;
        virtual size_t DoGetPlanarFaceVertexIndex(size_t localFaceIdx, size_t vertexId);
        virtual WorldVector DoGetPlanarFaceNormal(size_t localFaceIdx) const ;
        virtual void DoCopyExtensionSpecificDataToOptiX(optixu::Context context);

    private:
        /// Session reader object
        Nektar::LibUtilities::SessionReaderSharedPtr m_session;

        /// MeshGraph containing mesh and geometry information
        Nektar::SpatialDomains::MeshGraphSharedPtr m_graph;

        /// Array of fields
        std::vector<Nektar::MultiRegions::ExpListSharedPtr> m_fields;

        /// Array of field definitions
        std::vector<Nektar::LibUtilities::FieldDefinitionsSharedPtr> m_fieldDef;

        /// Vector containing all faces
        std::vector<Nektar::SpatialDomains::Geometry2DSharedPtr> m_faces;

        std::vector<bool> m_faceNormalFlip;

        /// Map from geometry IDs to contiguous IDs (i.e. 0 -> nFaces-1)
        std::map<int, int> m_idToElmt;

        /// Vector of FaceInfo objects for each face in the domain
        std::vector<FaceInfo> m_faceInfo;

        /// Extent of mesh
        WorldPoint m_extentMin, m_extentMax;

        std::vector<Nektar::SpatialDomains::Geometry2DSharedPtr> m_planarFaces;

        std::map<int, int> m_planarVertIdMap;

        std::vector<WorldPoint> m_planarVerts;

        std::vector<WorldVector> m_planarFaceNormals;
    };
}
}

#endif
