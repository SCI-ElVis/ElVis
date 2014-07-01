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

#include <ElVis/Extensions/NektarPlusPlusExtension/NektarModelNew.h>

#include <SpatialDomains/MeshGraph3D.h>
#include <LocalRegions/TriExp.h>
#include <LocalRegions/QuadExp.h>
#include <LocalRegions/HexExp.h>
#include <MultiRegions/ExpList3D.h>

namespace ElVis
{
namespace NektarPlusPlusExtension
{
    NektarModel::NektarModel(const std::string& modelPath) :
        Model(modelPath)
    {

    }

    NektarModel::~NektarModel()
    {
        
    }

    void NektarModel::LoadVolume(const std::string &path)
    {
        int i, j;

        boost::filesystem::path geometryFile(path + ".xml");
        boost::filesystem::path fieldFile(path + ".fld");

        if (!boost::filesystem::exists(geometryFile))
        {
            std::string message = std::string("Geometry file ")
                + geometryFile.string() + " does not exist.";
            throw std::runtime_error(message.c_str());
        }

        if (!boost::filesystem::exists(fieldFile))
        {
            std::string message = std::string("Field file ")
                + fieldFile.string() + " does not exist.";
            throw std::runtime_error(message.c_str());
        }

        // Create session reader
        char* arg1 = strdup("ElVis");
        char* arg2 = strdup(geometryFile.string().c_str());
        char* arg3 = strdup(fieldFile.string().c_str());
        char* argv[] = {arg1, arg2, arg3};
        int argc = 3;
        m_session = LibUtilities::SessionReader::CreateInstance(argc, argv);
        free(arg1);
        free(arg2);
        free(arg3);
        arg1 = arg2 = arg3 = 0;

        // Create MeshGraph object
        m_graph = SpatialDomains::MeshGraph::Read(m_session);

        if (m_graph->GetMeshDimension() != 3)
        {
            throw std::runtime_error("Only 3D supported for Nektar++");
        }

        // Set up expansion lists
        int nvariables = m_session->GetVariables().size();
        m_fields.resize(nvariables);

        for (i = 0; i < m_fields.size(); i++)
        {
            m_fields[i] = MemoryManager<MultiRegions::ExpList3D>
                ::AllocateSharedPtr(
                    m_session, m_graph, 
                    m_session->GetVariable(i));
        }

        // Load field data
        LoadFields(fieldFile);

        // Set up map which takes element global IDs to expansion index
        for (i = 0; i < m_fields[0]->GetExpSize(); ++i)
        {
            m_idToElmt[m_fields[0]->GetExp(i)->GetGeom()->GetGlobalID()] = i;
        }

        // Populate list of faces
        SpatialDomains::QuadGeomMap::const_iterator qIt;
        const SpatialDomains::QuadGeomMap &quadMap = m_graph->GetAllQuadGeoms();
        m_faces.resize(quadMap.size());

        for (i = 0, qIt = quadMap.begin(); qIt != quadMap.end(); ++qIt)
        {
            m_faces[i++] = qIt->second;
        }

        m_fields[0]->SetUpPhysNormals();

        // Initialise with large values (fix me later)
        ElVisFloat extentMaxX = -ELVIS_FLOAT_MAX;
        ElVisFloat extentMaxY = -ELVIS_FLOAT_MAX;
        ElVisFloat extentMaxZ = -ELVIS_FLOAT_MAX;
        ElVisFloat extentMinX =  ELVIS_FLOAT_MAX;
        ElVisFloat extentMinY =  ELVIS_FLOAT_MAX;
        ElVisFloat extentMinZ =  ELVIS_FLOAT_MAX;

        set<int> seenPlanarVerts;
        
        for (i = 0; i < m_faces.size(); ++i)
        {
            SpatialDomains::Geometry2DSharedPtr face = m_faces[i];
            SpatialDomains::ElementFaceVectorSharedPtr connectedElmt =
                boost::dynamic_pointer_cast<SpatialDomains::MeshGraph3D>(
                    m_graph)->GetElementsFromFace(face);

            FaceInfo fInfo;
            fInfo.CommonElements[1] = ElementId(-1,-1);
            for (int i = 0; i < connectedElmt->size(); ++i)
            {
                fInfo.CommonElements[i] = ElementId(
                    m_idToElmt[connectedElmt->at(i)->m_Element->GetGlobalID()],
                    (int)connectedElmt->at(i)->m_Element->GetShapeType());
            }

            if (face->GetGeomFactors()->GetGtype() == SpatialDomains::eDeformed)
            {
                fInfo.Type = eCurved;
            }
            else
            {
                fInfo.Type = ePlanar;

                // Construct face normal
                m_planarFaces.push_back(face);

                const Array<Nektar::OneD, const Array<Nektar::OneD, NekDouble> > &normal =
                    boost::dynamic_pointer_cast<LocalRegions::HexExp>(
                        m_fields[0]->GetExp(
                            m_idToElmt[
                                connectedElmt->at(0)->m_Element->GetGlobalID()]))
                    ->GetFaceNormal(connectedElmt->at(0)->m_FaceIndx);

                m_planarFaceNormals.push_back(
                    WorldVector(normal[0][0], normal[1][0], normal[2][0]));

                for (j = 0; j < face->GetNumVerts(); ++j)
                {
                    int vId = face->GetVid(j);

                    if (seenPlanarVerts.count(vId) > 0)
                    {
                        continue;
                    }

                    SpatialDomains::PointGeomSharedPtr v = face->GetVertex(j);
                    m_planarVerts.push_back(
                        WorldPoint((*v)(0), (*v)(1), (*v)(2)));
                    m_planarVertIdMap[vId] = m_planarVerts.size() - 1;

                    seenPlanarVerts.insert(vId);
                }
            }

            // Calculate estimate of face extent
            LocalRegions::Expansion2DSharedPtr faceExp;
            StdRegions::StdExpansionSharedPtr xmap = face->GetXmap();
            if (face->GetShapeType() == LibUtilities::eQuadrilateral)
            {
                faceExp = MemoryManager<LocalRegions::QuadExp>::
                    AllocateSharedPtr(xmap->GetBasis(0)->GetBasisKey(),
                                      xmap->GetBasis(1)->GetBasisKey(),
                                      boost::dynamic_pointer_cast<SpatialDomains::QuadGeom>(face));
            }
            else if (face->GetShapeType() == LibUtilities::eTriangle)
            {
                faceExp = MemoryManager<LocalRegions::TriExp>::
                    AllocateSharedPtr(xmap->GetBasis(0)->GetBasisKey(),
                                      xmap->GetBasis(1)->GetBasisKey(),
                                      boost::dynamic_pointer_cast<SpatialDomains::TriGeom>(face));
            }

            const int nPts = faceExp->GetTotPoints();
            Array<Nektar::OneD, NekDouble> x(nPts), y(nPts), z(nPts);
            faceExp->GetCoords(x, y, z);

            //WorldVector minExt(x[0], y[0], z[0]), maxExt(x[0], y[0], z[0]);
            ElVisFloat minX = x[0], minY = y[0], minZ = z[0];
            ElVisFloat maxX = x[0], maxY = y[0], maxZ = z[0];
            cout << x[0] << " " << y[0] << " " << z[0] << endl;
            for (j = 1; j < nPts; ++j)
            {
                cout << x[j] << " " << y[j] << " " << z[j] << endl;
                minX = std::min((ElVisFloat)x[j], minX);
                minY = std::min((ElVisFloat)y[j], minY);
                minZ = std::min((ElVisFloat)z[j], minZ);
                maxX = std::max((ElVisFloat)x[j], maxX);
                maxY = std::max((ElVisFloat)y[j], maxY);
                maxZ = std::max((ElVisFloat)z[j], maxZ);
            }

            //WorldVector minExt(minX, minY, minZ), maxExt(maxX, maxY, maxZ);
            //cout << "MINEXT " << minExt << "   MAXEXT = " << maxExt << endl;

            fInfo.MinExtent.x = minX;
            fInfo.MinExtent.y = minY;
            fInfo.MinExtent.z = minZ;
            fInfo.MaxExtent.x = maxX;
            fInfo.MaxExtent.y = maxY;
            fInfo.MaxExtent.z = maxZ;

            cout << "MINEXT " << fInfo.MinExtent.x << " "  << fInfo.MinExtent.y << " " << fInfo.MinExtent.z
                 << "   MAXEXT = " << fInfo.MaxExtent.x << " " << fInfo.MaxExtent.y << " " << fInfo.MaxExtent.z << endl;

            m_faceInfo.push_back(fInfo);

            extentMinX = std::min(minX, extentMinX);
            extentMinY = std::min(minY, extentMinY);
            extentMinZ = std::min(minZ, extentMinZ);
            extentMaxX = std::max(maxX, extentMaxX);
            extentMaxY = std::max(maxY, extentMaxY);
            extentMaxZ = std::max(maxZ, extentMaxZ);
        }

        m_extentMin = WorldPoint(extentMinX, extentMinY, extentMinZ);
        m_extentMax = WorldPoint(extentMaxX, extentMaxY, extentMaxZ);

        cout << "Read " << m_fields[0]->GetExpSize() << " elements" << endl;
        cout << "Extent = ["
             << extentMinX << ", " << extentMinY << ", " << extentMinZ << "] <-> ["
             << extentMaxX << ", " << extentMaxY << ", " << extentMaxZ << "]" << endl;
        cout << "Mesh has " << m_planarFaces.size() << " planar face(s)" << endl;
    }

    void NektarModel::LoadFields(const boost::filesystem::path& fieldFile)
    {
        std::vector<std::vector<NekDouble> > fieldData;
        LibUtilities::Import(fieldFile.string(), m_fieldDef, fieldData);

        // Copy FieldData into m_fields
        for(int j = 0; j < m_fields.size(); ++j)
        {
            Vmath::Zero(m_fields[j]->GetNcoeffs(),
                        m_fields[j]->UpdateCoeffs(),1);
                
            for(int i = 0; i < m_fieldDef.size(); ++i)
            {
                ASSERTL1(m_fieldDef[i]->m_fields[j] ==
                         m_session->GetVariable(j),
                         std::string("Order of ") + fieldFile.string()
                         + std::string(" data and that defined in "
                                       "field data differs"));

                m_fields[j]->ExtractDataToCoeffs(m_fieldDef[i], fieldData[i],
                                                 m_fieldDef[i]->m_fields[j],
                                                 m_fields[j]->UpdateCoeffs());
            }
        }
    }

    int NektarModel::DoGetNumFields() const
    {
        return m_fields.size();
    }

    int NektarModel::DoGetModelDimension() const
    {
        return m_graph->GetMeshDimension();
    }

    FieldInfo NektarModel::DoGetFieldInfo(unsigned int index) const
    {
        FieldInfo returnval;
        returnval.Name = m_session->GetVariable(index);
        returnval.Id = index;
        returnval.Shortcut = m_session->GetVariable(index);
        return returnval;
    }

    int NektarModel::DoGetNumberOfBoundarySurfaces() const
    {
        return 0;
    }

    void NektarModel::DoGetBoundarySurface(
        int surfaceIndex, std::string& name, std::vector<int>& faceIds)
    {
        
    }

    void NektarModel::DoCalculateExtents(WorldPoint& min, WorldPoint& max)
    {
        min = m_extentMin;
        max = m_extentMax;
    }

    const std::string& NektarModel::DoGetPTXPrefix() const
    {
        static std::string extensionName("NektarPlusPlusExtension");
        return extensionName;
    }

    unsigned int NektarModel::DoGetNumberOfElements() const
    {
        cout << "num elements = " << m_fields[0]->GetExpSize() << endl;
        return m_fields[0]->GetExpSize();
    }

    std::vector<optixu::GeometryInstance> NektarModel::DoGet2DPrimaryGeometry(
        boost::shared_ptr<Scene> scene, optixu::Context context)
    {
        return std::vector<optixu::GeometryInstance>();
    }

    optixu::Material NektarModel::DoGet2DPrimaryGeometryMaterial(SceneView* view)
    {
        return optixu::Material();
    }

    size_t NektarModel::DoGetNumberOfFaces() const
    {
        return m_faces.size();
    }

    FaceInfo NektarModel::DoGetFaceDefinition(size_t globalFaceId) const
    {
        FaceInfo f = m_faceInfo[globalFaceId];
        cout << "Face " << globalFaceId << ": left = "
             << f.CommonElements[0].Id << " " << f.CommonElements[0].Type << " "
             << f.CommonElements[1].Id << " " << f.CommonElements[1].Type
             << "   minext = " << f.MinExtent.x << " " << f.MinExtent.y << " " << f.MinExtent.z
             << "   maxext = " << f.MaxExtent.x << " " << f.MaxExtent.y << " " << f.MaxExtent.z << endl;
        return m_faceInfo[globalFaceId];
    }

    size_t NektarModel::DoGetNumberOfPlanarFaceVertices() const
    {
        cout << "num planar verts = " << m_planarVerts.size() << endl;
        return m_planarVerts.size();
    }

    WorldPoint NektarModel::DoGetPlanarFaceVertex(size_t vertexIdx) const
    {
        cout << "planar vert " << vertexIdx << " = " << m_planarVerts[vertexIdx] << endl;
        return m_planarVerts[vertexIdx];
    }

    size_t NektarModel::DoGetNumberOfVerticesForPlanarFace(size_t localFaceIdx) const
    {
        cout << "face " << localFaceIdx << " has " << m_planarFaces[localFaceIdx]->GetNumVerts() << " verts" << endl;
        return m_planarFaces[localFaceIdx]->GetNumVerts();
    }

    size_t NektarModel::DoGetPlanarFaceVertexIndex(size_t localFaceIdx, size_t vertexId)
    {
        cout << "face " << localFaceIdx << " vId " << vertexId << " has index " << m_planarVertIdMap[m_planarFaces[localFaceIdx]->GetVid(vertexId)] << endl;
        return m_planarVertIdMap[m_planarFaces[localFaceIdx]->GetVid(vertexId)];
    }

    WorldVector NektarModel::DoGetPlanarFaceNormal(size_t localFaceIdx) const
    {
        cout << "face " << localFaceIdx << " normal " << m_planarFaceNormals[localFaceIdx] << endl;
        return m_planarFaceNormals[localFaceIdx];
    }

    void NektarModel::DoCopyExtensionSpecificDataToOptiX(optixu::Context context)
    {
        
    }
}
}
