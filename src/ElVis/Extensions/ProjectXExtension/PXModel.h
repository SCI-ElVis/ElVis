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

#ifndef ELVIS_PX_MODEL_H
#define ELVIS_PX_MODEL_H

#ifdef ELVIS_USE_PROJECTX

#include <ElVis/Extensions/ProjectXExtension/PXStructDefinitions.h>

#define LEDA_EXACT
#include <Grid/CutCell3D/PXNumberType.h>
#include <Grid/CutCell3D/PXCutCell3d.h>
#include <Grid/CutCell3D/PXConicStruct.h>
#include <Grid/CutCell3D/PXCutCell3dStruct.h>
#include <Grid/CutCell3D/PXIntersect3dStruct.h>
#include <Grid/CutCell3D/PXReadWriteCutCell3d.h>


extern "C"{
#include <Fundamentals/PX.h>
#include <Fundamentals/PXError.h>
#include <Reference/PXOrder.h>
#include <PXAll.h>
#include <PXCDAll.h>
#include <Grid/PXCoordinates.h>
#include <Grid/PXGridAttachments.h>
}

#include <ElVis/Core/Model.h>
#include <optixu/optixpp.h>
#include <boost/foreach.hpp>
#include <boost/utility.hpp>
#include <ElVis/Core/OptiXBuffer.hpp>
#include <ElVis/Core/Float.h>

namespace ElVis
{
  class PXModel : public Model
  {
  public:
    PXModel(const std::string& modelPath);
    PXModel(const PXModel& rhs);
    virtual ~PXModel();

    static const std::string prefix;

    void LoadVolume(const std::string& filePath);

  protected:
    virtual std::vector<optixu::GeometryGroup> DoGetPointLocationGeometry(Scene* scene, optixu::Context context);
    virtual void DoGetFaceGeometry(Scene* scene, optixu::Context context, optixu::Geometry& faces);
    virtual unsigned int DoGetNumberOfPoints() const;
    virtual WorldPoint DoGetPoint(unsigned int id) const;
    virtual std::vector<optixu::GeometryInstance> DoGet2DPrimaryGeometry(Scene* scene, optixu::Context context);
    virtual optixu::Material DoGet2DPrimaryGeometryMaterial(SceneView* view);
    virtual int DoGetModelDimension() const { return 3; }
    virtual const std::string& DoGetPTXPrefix() const;	    

    virtual unsigned int DoGetNumberOfElements() const;

    virtual void DoCalculateExtents(WorldPoint& min, WorldPoint& max);

    virtual int DoGetNumFields() const;
    virtual FieldInfo DoGetFieldInfo(unsigned int index) const;

    virtual int DoGetNumberOfBoundarySurfaces() const;
    virtual void DoGetBoundarySurface(int surfaceIndex, std::string& name, std::vector<int>& faceIds);

  private:
    static const std::string PXSimplexPtxFileName;
    static const std::string PXSimplexIntersectionProgramName;
    static const std::string PXSimplexBoundingProgramName;
    static const std::string PXSimplexClosestHitProgramName;

    PXModel& operator=(const PXModel& rhs);



    ElVis::OptiXBuffer<ElVisFloat> m_solutionBuffer; //for State_0 GRE values (solution)
    ElVis::OptiXBuffer<ElVisFloat> m_coordinateBuffer; //for coordinates of elements of computational mesh
    ElVis::OptiXBuffer<ElVisFloat> m_boundingBoxBuffer; //for coordinates of bounding boxes of the elements in computational mesh
    ElVis::OptiXBuffer<PX_EgrpData> m_egrpDataBuffer; //data about each element group
    ElVis::OptiXBuffer<unsigned int> m_globalElemToEgrpElemBuffer; //mapping from global element number to (egrp,elem)

    ElVis::OptiXBuffer<PX_SolutionOrderData> m_attachDataBuffer; //solutionorder info about a single attachment (for now, distance function)
    ElVis::OptiXBuffer<ElVisFloat> m_attachmentBuffer; //values from a single attachment (for now, distance function)

    ElVis::OptiXBuffer<ElVisFloat> m_shadowCoordinateBuffer; //coordinates of shadow elements for cut cells; NOTHING stored for non-cut elem!
    ElVis::OptiXBuffer<unsigned int> m_egrpToShadowIndexBuffer; //maps from egrp to appropriate position in m_shadowCoordinateBuffer
    //DO NOT access if this egrp has no cut elements!
    //for some egrp w/cut elements, ShadowCoordinateBuffer[egrpToShadowIndex[egrp]] will be the first coord of the first element in egrp

    ElVis::OptiXBuffer<ElVisFloat> m_patchCoordinateBuffer; //must index using the patch indexes from PX_PatchGroup

    ElVis::OptiXBuffer<PX_REAL> m_knownPointBuffer; //indexed by threeDId; this is a field in PX_PatchGroup
    ElVis::OptiXBuffer<PX_REAL> m_backgroundCoordinateBuffer; //indexed by threeDId; this is a field in PX_PatchGroup

    ElVis::OptiXBuffer<char> m_cutCellBuffer; //data about cut cells, indexed by a global cut cell number
    ElVis::OptiXBuffer<unsigned int> m_globalElemToCutCellBuffer; //mapping from global element number to  global cut cell number

    ElVis::OptiXBuffer<ElVisFloat> m_faceCoordinateBuffer; //for coordinates of element faces of computational mesh
    ElVis::OptiXBuffer<PX_FaceData> m_faceDataBuffer; //for coordinates of element faces of computational mesh

    PX_All *m_pxa;
    unsigned int m_numFieldsToPlot;
    char m_cutCellFlag; //1 if cut cells are present
  };
}

#endif //ELVIS_USE_PROJECTX

#endif //ELVIS_PX_MODEL_H
