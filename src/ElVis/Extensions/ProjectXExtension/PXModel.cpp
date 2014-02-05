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

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <ElVis/Extensions/ProjectXExtension/PXModel.h>
#include <ElVis/Core/PtxManager.h>
#include <ElVis/Core/Util.hpp>
#include <ElVis/Core/Scene.h>
#include <list>
#include <boost/typeof/typeof.hpp>


#include <ElVis/Extensions/ProjectXExtension/PXStructDefinitions.h>

extern "C"{
#include <Fundamentals/PX.h>
#include <Grid/PXGridStruct.h>
#include <Grid/PXMeshStruct.h>
#include <Grid/PXAttachmentType.h>

#include <Fundamentals/PXParameter.h>
#include <PXUnsteady.h>
#include <Grid/PXGrid.h>
#include <Grid/PXGridAttachments.h>
#include <Grid/PXAttachmentMath.h>
#include <Grid/PXMeshSize.h>
#include <Reference/PXCoordinatesReference.h>
#include <Grid/PXCoordinates.h>
#include <Grid/PXNormal.h>

#include <Grid/PXReadWriteGrid.h>
}



//   #include <LEDA/numbers/real.h>
//   #include <LEDA/core/array.h>
//   #include <LEDA/numbers/polynomial.h> 
//   #include <LEDA/numbers/isolating.h>
//   #include <LEDA/numbers/rational.h>

// #define EX_REAL leda::real

extern "C"{
  void PadBoundingBox(int dim, ElVisFloat padFactor, ElVisFloat* bBox);
  std::string GetPluginName();
  ElVis::Model* LoadModel(const char* path);
  std::string GetVolumeFileFilter();
  bool CanLoadModel(const char* path);

  void PadBoundingBox(int dim, ElVisFloat padFactor, ElVisFloat* bBox){
    //inputs
    //dim: spatial dimension (bBox must be n array of size 2*dim)
    //padFactor: (relative) amount to pad bounding box by
    //outputs
    //bBox: array ordered [xmin,ymin,zmin,xmax,ymax,zmax] representing
    //      the 2 corners of the bounding box

    ElVisFloat center, delta, minval;
    int d;
    /* Pad the domain bounding box (cheap and much safer) */
    //padFactor = 0.1+eps;
    for (d=0; d<dim; d++){
      center = 0.5*( bBox[dim+d] + bBox[d] );
      delta  = 0.5*( bBox[dim+d] - bBox[d] );

      if( delta < 100*MEPS ) // degenerate bounding box
      {
          delta = 100*MEPS;
      }
      minval  = center - (1.0+padFactor)*delta;
      bBox[d  ] = minval;
      bBox[dim+d] = center + (1.0+padFactor)*delta;
    }
  }
}

std::string GetPluginName()
{
  std::string pluginName("ProjectXPlugin");
  return pluginName;
}

ElVis::Model* LoadModel(const char* path)
{
  ElVis::PXModel * result = new ElVis::PXModel(path);
  result->LoadVolume(std::string(path));
  return result;
}

std::string GetVolumeFileFilter()
{
  std::string fileExtension("ProjectX PXA (*.pxa)");
  return fileExtension;
}


namespace ElVis
{
  const std::string PXModel::prefix = "PXSimplex";

  int PXModel::DoGetNumFields() const
  {
    return m_numFieldsToPlot;
  }

  int PXModel::DoGetModelDimension() const
  {
    return m_pxa->pg->Dim;
  }

  FieldInfo PXModel::DoGetFieldInfo(unsigned int index) const
  {
    PX_Parameter *Parameter = m_pxa->Parameter;
    if(Parameter == NULL){
      printf("Parameter is NULL :( wtf?\n");
      exit(1);
    }
    /* WARNING this is NOT safe.  But I'm too lazy to give
       ElVis access to PXCompressible.h */

    char svalue[200];
    int turbulenceModel = 1;

    PXError( PXGetKeyValue(Parameter, "TurbulenceModel", svalue) );
    turbulenceModel = (strcmp("SA",svalue) == 0);

    PX_AttachmentGlobRealElem *State;
    int currentIndex = 0;
    PXError( PXRetrieveTimeStepState( m_pxa, currentIndex, -1, NULL,
              &State, NULL ) );
    int StateRank = State->StateRank;


    FieldInfo result;
    result.Name = "UNINITIALIZED";
    result.Shortcut = (char) 255;
    switch(index){
    case 0:
      result.Name = "density";
      result.Id = 0;
      result.Shortcut = "r";
      return result;
    case 1:
      result.Name = "x-mom";
      result.Id = 1;
      result.Shortcut = "u";
      return result;
    case 2:
      result.Name = "y-mom";
      result.Id = 2;
      result.Shortcut = "v";
      return result;
    case 3:
      result.Name = "z-mom";
      result.Id = 3;
      result.Shortcut = "w";
      return result;
    case 4:
      result.Name = "rho E";
      result.Id = 4;
      result.Shortcut = "e";
      return result;
    case 5:
      result.Id = 5;
      if(StateRank >= 6){
        if(StateRank == 7 || turbulenceModel == 1){
          result.Name = "rho nu-tilde";
          result.Shortcut = "n";
        }else{
          result.Name = "dissipation";
          result.Shortcut = "d";
        }
      }
      return result;
    case 6:
      result.Id = 6;
      if(StateRank == 7){
        result.Name = "dissipation";
        result.Shortcut = "d";
      }
      return result;
    case 7:
      result.Name = "mach number";
      result.Id = 7;
      result.Shortcut = "m";
      return result;
    case 8:
      result.Name = "velocity magnitude";
      result.Id = 8;
      result.Shortcut = "q";
      return result;
    case 9:
      result.Name = "pressure";
      result.Id = 9;
      result.Shortcut = "p";
      return result;
    case 10:
      result.Name = "Geometric Jacobian";
      result.Id = 10;
      result.Shortcut = "j";
      return result;
    case 11:
      result.Id = -1;
      if(turbulenceModel == 1){
        result.Name = "Distance";
        result.Shortcut = "[";
      }
      return result;
    default:
      printf("ERROR: invalid field index=%d\n",index);
      result.Name = "OUT_OF_BOUNDS";
      result.Id = -1000;
      result.Shortcut = "";
      return result;
    }

  }

  int PXModel::DoGetNumberOfBoundarySurfaces() const
  {
    PX_Grid *pg = m_pxa->pg;

    int fgrp;

    //COUNTING boundary face groups
    int numBoundaryFaceGroup = 0;
    for(fgrp=0; fgrp<pg->nFaceGroup; fgrp++){
      if ( (pg->FaceGroup[fgrp].FaceGroupFlag==PXE_BoundaryFG) || (pg->FaceGroup[fgrp].FaceGroupFlag==PXE_EmbeddedBoundaryFG) ){
        numBoundaryFaceGroup++; //number of boundary face groups
      }
    }
    return numBoundaryFaceGroup;
  }

  void PXModel::DoGetBoundarySurface(int boundaryFaceGroupNum, std::string& name, std::vector<int>& boundaryFaceList)
  {
      name = std::string("Domain Surf") + boost::lexical_cast<std::string>(boundaryFaceGroupNum+1);

      PX_Grid *pg = m_pxa->pg;
      int *fgrp2GlobalFaceIndex = (int *)malloc((pg->nFaceGroup+1)*sizeof(int));

      fgrp2GlobalFaceIndex[0] = 0;
      for(int fgrp=1; fgrp<=pg->nFaceGroup; fgrp++){
        fgrp2GlobalFaceIndex[fgrp] = fgrp2GlobalFaceIndex[fgrp-1] + pg->FaceGroup[fgrp-1].nFace;
      }

      ///identify which fgrp corresponds to our boundaryFaceGroup
      int nfgrp, fgrpMatch;
      PXError(PXbfgrp2fgrp(pg, boundaryFaceGroupNum, &nfgrp, &fgrpMatch));
      //protect against a lazy assumption.  segfault might already occurred if nfgrp != 1
      if(nfgrp != 1){
        printf("FAIL, 1 boundary face group maps to multiple face groups!\n");
        exit(1);
      }

      for(int face=0; face<pg->FaceGroup[fgrpMatch].nFace; face++){
          boundaryFaceList.push_back(fgrp2GlobalFaceIndex[fgrpMatch] + face);
      }

      free(fgrp2GlobalFaceIndex);
  }

  void PXModel::DoCalculateExtents(WorldPoint& min, WorldPoint& max)
  {
    PX_Grid *pg = m_pxa->pg;
    PX_REAL bBoxDomain[6], domainSize;
    int d;

    PXError(PXGetDomainBoundingBox(pg, pg->Dim, bBoxDomain, &domainSize));

    for(d=0; d<pg->Dim; d++){
      min.SetValue(d, bBoxDomain[2*d+0]);
      max.SetValue(d, bBoxDomain[2*d+1]);
    }
  }

  unsigned int PXModel::DoGetNumberOfElements() const
  {
    PX_Grid *pg = m_pxa->pg;
    int nElementTotal = 0;

    for(int egrp = 0; egrp<pg->nElementGroup; egrp++)
      nElementTotal += pg->ElementGroup[egrp].nElement; //total # elements

    return nElementTotal;
  }

  const std::string& PXModel::DoGetPTXPrefix() const
  {
    static std::string extensionName("ProjectXExtension");
    return extensionName;
  }

  std::vector<optixu::GeometryInstance> PXModel::DoGet2DPrimaryGeometry(boost::shared_ptr<Scene> scene, optixu::Context context)
  {
    return std::vector<optixu::GeometryInstance>();
  }

  optixu::Material PXModel::DoGet2DPrimaryGeometryMaterial(SceneView* view)
  {
    return optixu::Material();
  }

  size_t PXModel::DoGetNumberOfFaces() const
  {
    return m_PlanarFaces.size();
  }

  FaceInfo PXModel::DoGetFaceDefinition(size_t globalFaceId) const
  {
    return m_PlanarFaces[globalFaceId].info;
  }

  size_t PXModel::DoGetNumberOfPlanarFaceVertices() const
  {
    return m_pxa->pg->nNode; //m_vertices.size();
  }

  size_t PXModel::DoGetNumberOfVerticesForPlanarFace(size_t localFaceIdx) const
  {
    if( m_PlanarFaces[localFaceIdx].planarInfo.Type == eTriangle )
    {
      return 3;
    }
    else if( m_PlanarFaces[localFaceIdx].planarInfo.Type == eQuad )
    {
      return 4;
    }
    else
      assert(0); //Really should throw an error here instead

    return -1; //Just so compiler will not complain
  }

  WorldPoint PXModel::DoGetPlanarFaceVertex(size_t vertexIdx) const
  {
    //return m_vertices[vertexIdx];
    WorldPoint FaceNode( m_pxa->pg->coordinate[vertexIdx][0],
                         m_pxa->pg->coordinate[vertexIdx][1],
                         m_pxa->pg->coordinate[vertexIdx][2] );
    return FaceNode;
  }

  size_t PXModel::DoGetPlanarFaceVertexIndex(size_t localFaceIdx, size_t vertexId)
  {
    return m_PlanarFaces[localFaceIdx].planarInfo.vertexIdx[vertexId];
  }

  WorldVector PXModel::DoGetPlanarFaceNormal(size_t localFaceId) const
  {
    return m_PlanarFaces[localFaceId].normal;
  }

  void PXModel::DoCopyExtensionSpecificDataToOptiX(optixu::Context context)
  {
    PX_Grid *pg = m_pxa->pg;
    const int Dim = pg->Dim;
    context["Dim"]->setInt(Dim);


    PX_AttachmentGlobRealElem *State = NULL;
    int currentIndex = 0;
    PXError( PXRetrieveTimeStepState( m_pxa, currentIndex, -1, NULL, &State, NULL ) );

    int StateRank = State->StateRank;
    context["StateRank"]->setInt(StateRank);

    int nbfS = 0, nbfQ = 0;
    int nSolnCoeffTotal = 0;
    int nGeomCoeffTotal = 0;
    for(int egrp = 0; egrp<pg->nElementGroup; egrp++){
//        nElemTotal += pg->ElementGroup[egrp].nElement; //total # elements

        PXOrder2nbf(State->order[egrp], &nbfS);
        PXType2nbf(pg->ElementGroup[egrp].type, &nbfQ);

        printf("egrp=%d, nbfS=%d, nbfQ=%d\n", egrp, nbfS, nbfQ);

        nSolnCoeffTotal += nbfS*StateRank*pg->ElementGroup[egrp].nElement;
        nGeomCoeffTotal += nbfQ*Dim*pg->ElementGroup[egrp].nElement;
        // nAttachCoeffTotal += nbfA*pg->ElementGroup[egrp].nElement;

//        if(pg->ElementGroup[egrp].type == PXE_TetCut){
//            nCutCellTotal += pg->ElementGroup[egrp].nElement;
//        }
    }


    ElVis::OptiXBuffer<int> egrp2GlobalElemIndex("egrp2GlobalElemIndex");
    egrp2GlobalElemIndex.SetContext(context);
    egrp2GlobalElemIndex.SetDimensions(pg->nElementGroup+1);
    BOOST_AUTO(egrp2GlobalElemIndexMap, egrp2GlobalElemIndex.map());

    m_coordinateBuffer.SetContext(context);
    m_coordinateBuffer.SetDimensions(nGeomCoeffTotal);
    BOOST_AUTO(coordinateData, m_coordinateBuffer.Map());

    m_solutionBuffer.SetContext(context);
    m_solutionBuffer.SetDimensions(nSolnCoeffTotal);
    BOOST_AUTO(solution, m_solutionBuffer.map());

    ElVis::OptiXBuffer<PX_EgrpData> egrpDataBuffer(prefix + "EgrpDataBuffer"); //data about each element group
    egrpDataBuffer.SetContext(context);
    egrpDataBuffer.SetDimensions(pg->nElementGroup);
    BOOST_AUTO(egrpDataBufferMap, egrpDataBuffer.map());

    int S = 0, G = 0, qorder=0, porder=0;
    enum PXE_Shape elemShape;
    enum PXE_SolutionOrder orderQ;
    for(int egrp = 0; egrp<pg->nElementGroup; egrp++){
        PXE_ElementType elemType = pg->ElementGroup[egrp].type;

        PXError( PXType2nbf(elemType, &nbfQ) );
        PXError( PXType2Interpolation(elemType, &orderQ));
        PXError( PXType2qorder(elemType, &qorder));
        PXError( PXType2Shape(elemType, &elemShape) );

        printf("egrp=%d, elemType=%d, elemShape=%d, qorder=%d\n", egrp, elemType, elemShape, qorder);

        PXError( PXOrder2nbf(State->order[egrp], &nbfS) );
        PXError( PXOrder2porder(State->order[egrp], &porder));

        egrp2GlobalElemIndexMap[egrp] = m_egrp2GlobalElemIndex[egrp];

        egrpDataBufferMap[egrp].cutCellFlag = 0;

        egrpDataBufferMap[egrp].elemData.type = elemType;
        egrpDataBufferMap[egrp].elemData.nbf = nbfQ;
        egrpDataBufferMap[egrp].elemData.order = orderQ;
        egrpDataBufferMap[egrp].elemData.qorder = qorder;
        egrpDataBufferMap[egrp].elemData.shape = elemShape;

        egrpDataBufferMap[egrp].solData.nbf = nbfS;
        egrpDataBufferMap[egrp].solData.order = State->order[egrp];
        egrpDataBufferMap[egrp].solData.porder = porder;

        egrpDataBufferMap[egrp].egrpSolnCoeffStartIndex = S;
        egrpDataBufferMap[egrp].egrpGeomCoeffStartIndex = G;

        int solnRank = StateRank*nbfS;

        //PXError(PXOrder2nbf(QnDistance->order[egrp],&nbfA));
        //for(int j=0; j<solnRank; j++)
        //  std::cout << egrp << " " << S << " " << pg->ElementGroup[egrp].nElement*StateRank << " " << nbfS << " " << State->value[egrp][0][j] << std::endl;

        for(int elem=0; elem<pg->ElementGroup[egrp].nElement; elem++){

            // fill solution array for this element
            for(int j=0; j<solnRank; j++){
              solution[S++] = State->value[egrp][elem][j];
            }

            for(int i=0; i<nbfQ; i++)
              for( int d = 0; d < Dim; d++ )
                coordinateData[G++] = pg->coordinate[pg->ElementGroup[egrp].Node[elem][i]][d];

            // fill in attachment values
            //for(j=0; j<QnDistance->StateRank*nbfA; j++){
            //    attachmentPtr[j] = QnDistance->value[egrp][elem][j];
            //}
        }
    }

    int egrp = pg->nElementGroup-1;
    egrp2GlobalElemIndexMap[egrp+1] = egrp2GlobalElemIndexMap[egrp] + pg->ElementGroup[egrp].nElement;

    std::cout << "nSolnCoeffTotal = " << nSolnCoeffTotal << " : S = " << S << std::endl;
    std::cout << "nGeomCoeffTotal = " << nGeomCoeffTotal << " : G = " << G << std::endl;
  }

  const std::string PXModel::PXSimplexPtxFileName("PXSimplex.cu.ptx_generated_ElVis.cu.ptx");
  const std::string PXModel::PXSimplexIntersectionProgramName("PXSimplexContainsOriginByCheckingPoint");
  const std::string PXModel::PXSimplexBoundingProgramName("PXSimplex_bounding");
  const std::string PXModel::PXSimplexClosestHitProgramName("EvaluatePXSimplexScalarValueArrayVersion");

  // const std::string PXModel::PrismPtxFileName("prism.cu.ptx");
  // const std::string PXModel::PrismIntersectionProgramName("PrismContainsOriginByCheckingPoint");
  // const std::string PXModel::PrismBoundingProgramName("PrismBounding");
  // const std::string PXModel::PrismClosestHitProgramName("EvaluatePrismScalarValueArrayVersion");



  PXModel::PXModel(const std::string& modelPath) :
    Model(modelPath),
    m_solutionBuffer(prefix + "SolutionBuffer"),
    m_coordinateBuffer(prefix + "CoordinateBuffer")
    //m_globalElemToEgrpElemBuffer(prefix + "GlobalElemToEgrpElemBuffer"),
    //m_attachDataBuffer(prefix + "AttachDataBuffer"),
    //m_attachmentBuffer(prefix + "AttachmentBuffer"),
    //m_shadowCoordinateBuffer(prefix + "ShadowCoordinateBuffer"),
    //m_egrpToShadowIndexBuffer(prefix + "EgrpToShadowIndexBuffer"),
    //m_patchCoordinateBuffer(prefix + "PatchCoordinateBuffer"),
    //m_knownPointBuffer(prefix + "KnownPointBuffer"),
    //m_backgroundCoordinateBuffer(prefix + "BackgroundCoordinateBuffer"),
    //m_cutCellBuffer(prefix + "CutCellBuffer"),
    //m_globalElemToCutCellBuffer(prefix + "GlobalElemToCutCellBuffer"),
    //m_faceCoordinateBuffer(prefix + "FaceCoordinateBuffer"),
    //m_faceDataBuffer(prefix + "FaceDataBuffer")
  {
    m_pxa = NULL;
    m_cutCellFlag = 0;
  }

  // PXModel::PXModel(const PXModel& rhs) :
  // m_volume(rhs.m_volume),
  // m_numberOfCopies(rhs.m_numberOfCopies),
  // m_numberOfModes(rhs.m_numberOfModes)
  // {
  // }

  PXModel::~PXModel()
  {}

  void PXModel::LoadVolume(const std::string& filePath)
  {
    PXError(PXCreateAll(&m_pxa));
    printf("m_pxa->nbf = %d\n",m_pxa->nbc);

    PXError(PXSetDefaults( m_pxa) );

    char *mypath = const_cast <char *>(filePath.c_str());
    printf("Reading %s\n", mypath);
    PXError(PXReadInputFile( m_pxa, mypath));
    PXError(PXReadPxaFile( m_pxa, mypath, "PXParameters"));

    PX_Grid *pg;
    pg = m_pxa->pg;
    printf("Dim = %d\n",pg->Dim);     

    m_cutCellFlag = pg->CC3D != NULL;
    FILE *fil;
    if(m_cutCellFlag && pg->CC3D->CutCellFile != NULL){
      fil = fopen(pg->CC3D->CutCellFile, "rb");
      PXError(PXReadCutCell3d(pg->CC3D, fil));
      fclose(fil);

      fil = fopen(pg->CC3D->CutQuadFile, "rb");
      PXError(PXReadGridQuadRule(pg, fil));
      fclose(fil);
    }


    PX_AttachmentGlobRealElem *State = 0;
    int currentIndex = 0;
    PXError( PXRetrieveTimeStepState( m_pxa, currentIndex, -1, NULL, &State, NULL ) );

    //State can be null if the file was not found
    int StateRank = State->StateRank;
    printf("StateRank = %d\n", StateRank);
    m_numFieldsToPlot = StateRank ; //+ 7;


    printf("00000000000\n");

    //egrp2GlobalElemIndex array
    m_egrp2GlobalElemIndex.resize(pg->nElementGroup);
    m_egrp2GlobalElemIndex[0] = 0;
    for(int egrp=1; egrp<pg->nElementGroup; egrp++){
      m_egrp2GlobalElemIndex[egrp] = m_egrp2GlobalElemIndex[egrp-1] + pg->ElementGroup[egrp-1].nElement;
    }



    ElVisFloat bBoxTemp[BBOX_SIZE];
    ElVisFloat bBoxTempElVis[BBOX_SIZE];

    int geomRank;
    int fgrp, face;
    int egrp, elem, lface;
    int egrpR, elemR;
    int i;
    int qorder;
    int nbfQ, nbfQFace;
    int nFaceTotal = 0;

    int d;
    int Dim = pg->Dim;

    int orientation;
    enum PXE_Shape elemShape, faceShape;
    enum PXE_ElementType FaceType, elemType, elemTypeR, baseFaceType, localFaceType;
    //int maxnbfQ;

    //PX_REAL *nodeCoord;
    //nodeCoord = (PX_REAL*) malloc(maxnbfQ*(Dim+1)*sizeof(PX_REAL));
    //PX_REAL *phiQ = nodeCoord + maxnbfQ*Dim;


    int nodesOnFace[36];
    int nNodesOnFace;
    PX_REAL nvec[3] = {0,0,0};
    PX_REAL xface[2] = {0,0};

    for(fgrp=0; fgrp<pg->nFaceGroup; fgrp++){
      for(face=0; face<pg->FaceGroup[fgrp].nFace; face++){
        // for now, ALWAYS use LEFT face
        egrp = pg->FaceGroup[fgrp].FaceL[face].ElementGroup;
        elem = pg->FaceGroup[fgrp].FaceL[face].Element;
        lface = pg->FaceGroup[fgrp].FaceL[face].Face;
        elemType = pg->ElementGroup[egrp].type;

        //for curved faces
        PXNodesOnFace(elemType, lface, nodesOnFace, &nNodesOnFace);
        //PXType2nbf(FaceType, &nbfQFace);
        PXType2qorder(elemType, &qorder);
        PXElemType2FaceType(elemType, 0, &FaceType );

        PXType2Shape(elemType, &elemShape);
        PXElemShape2FaceShape(elemShape, lface, &faceShape);
        PXElementCentroidReference(faceShape, xface);

        FaceInfo info;
        FaceNodeInfo planarInfo(nNodesOnFace);

        for( d = 0; d < Dim; d++)
        {
          bBoxTemp[2*d+0] =  std::numeric_limits<double>::max();
          bBoxTemp[2*d+1] = -std::numeric_limits<double>::max();
        }

        for(i=0; i < nNodesOnFace; i++){
          WorldPoint FaceNode( pg->coordinate[pg->ElementGroup[egrp].Node[elem][nodesOnFace[i]]][0],
                               pg->coordinate[pg->ElementGroup[egrp].Node[elem][nodesOnFace[i]]][1],
                               pg->coordinate[pg->ElementGroup[egrp].Node[elem][nodesOnFace[i]]][2] );


          planarInfo.vertexIdx[i] = pg->ElementGroup[egrp].Node[elem][nodesOnFace[i]];

          for( d = 0; d < Dim; d++)
          {
            bBoxTemp[2*d+0] = MIN(bBoxTemp[2*d+0], FaceNode[d]);
            bBoxTemp[2*d+1] = MAX(bBoxTemp[2*d+1], FaceNode[d]);
          }
        }

        if( PXE_UniformTriangleQ1 <= FaceType && FaceType <= PXE_UniformTriangleQ5 ) planarInfo.Type = eTriangle;
        else if( (PXE_UniformQuadQ1  <= FaceType && FaceType <= PXE_UniformQuadQ5 ) ||
                 (PXE_SpectralQuadQ1 <= FaceType && FaceType <= PXE_SpectralQuadQ5)) planarInfo.Type = eQuad;
        else
          printf("UNKNOWN FaceType=%d\n", FaceType);

        info.Type = qorder == 1 ? ePlanar : eCurved;

        //PXComputeFaceBoundingBox(pg, fgrp, face, PXE_Left, nodeCoord, phiQ, bBoxTemp);

        // reorder from PX bounding box ordering to ordering used on GPU
        info.MinExtent.x = bBoxTemp[2*0+0];
        info.MinExtent.y = bBoxTemp[2*1+0];
        info.MinExtent.z = bBoxTemp[2*2+0];

        info.MaxExtent.x = bBoxTemp[2*0+1];
        info.MaxExtent.y = bBoxTemp[2*1+1];
        info.MaxExtent.z = bBoxTemp[2*2+1];


        info.CommonElements[0].Id = m_egrp2GlobalElemIndex[egrp] + elem;
        info.CommonElements[0].Type = (int) elemType;
        if ( (pg->FaceGroup[fgrp].FaceGroupFlag!=PXE_BoundaryFG)&& (pg->FaceGroup[fgrp].FaceGroupFlag!=PXE_EmbeddedBoundaryFG) )
        {
          egrpR = pg->FaceGroup[fgrp].FaceR[face].ElementGroup;
          elemR = pg->FaceGroup[fgrp].FaceR[face].Element;
          elemTypeR = pg->ElementGroup[egrpR].type;

          info.CommonElements[1].Id = m_egrp2GlobalElemIndex[egrpR] + elemR;
          info.CommonElements[1].Type = (int) elemTypeR;

        }else{
          info.CommonElements[1].Id = -1;
          info.CommonElements[1].Type = -1;
        }

        PXFaceNormal(pg, fgrp, face, xface, nvec, NULL);

        WorldVector normal(nvec[0], nvec[1], nvec[2]);
        normal /= normal.Magnitude();

        //printf("elem0=%d, elem1=%d\n", info.CommonElements[0].Id, info.CommonElements[1].Id);

        m_PlanarFaces.push_back( PXPlanarFace(normal, info, planarInfo ) );


      }
    }

  }


/*
  std::vector<optixu::GeometryGroup> PXModel::DoGetPointLocationGeometry(Scene* scene, optixu::Context context)
  {
      try
      {        
          std::vector<optixu::GeometryGroup> result;
          if( m_pxa == NULL ) return result;

          PX_Grid *pg = m_pxa->pg;
          PX_AttachmentGlobRealElem *State;
          ElVisFloat * coordPtr;
          ElVisFloat * bBoxPtr;
          ElVisFloat * solutionPtr;
          ElVisFloat * attachmentPtr;
          unsigned int* globalToLocalPtr;

          PX_REAL bBoxTemp[BBOX_SIZE];

          int solnRank, geomRank;
          int egrp, elem;
          int i,j,k;
          int qorder;
          int nbf, nbfQ, nbfA;
          int nElemTotal = 0;
          int nCutCellTotal = 0;
          int nSolnCoeffTotal = 0;
          int nGeomCoeffTotal = 0;
          int nAttachCoeffTotal = 0;

          int currentIndex = 0;
          PXPrintAllAttachment(pg);
          PXError( PXRetrieveTimeStepState( m_pxa, currentIndex, -1, NULL,
                                           &State, NULL ) );

          int d;
          int Dim = pg->Dim;
          int StateRank = State->StateRank;

          printf("Dim=%d, StateRank=%d\n",Dim,StateRank);

          // Get physical constants
          PX_Parameter *Parameter = m_pxa->Parameter;
          if(Parameter == NULL){
              printf("Parameter is NULL :( wtf?\n");
              exit(1);
          }
          PX_REAL SpecificHeatRatio = 0.0;
          PX_REAL GasConstant = 0.0;
          PXError( PXGetKeyValueReal(Parameter, "SpecificHeatRatio", &SpecificHeatRatio) );
          PXError( PXGetKeyValueReal(Parameter, "GasConstant", &GasConstant) );
          printf("gamma = %.8E, R = %.8E\n",SpecificHeatRatio,GasConstant);


          char svalue[200];
          PXError(PXGetKeyValue(Parameter, "SolutionOrder", svalue));
          printf("%s\n",svalue);

          PXError(PXGetKeyValue(Parameter, "BasisType", svalue));
          printf("%s\n",svalue);
          PXError(PXGetKeyValue(Parameter, "EquationSet", svalue));
          printf("%s\n",svalue);

          PXError( PXGetKeyValue(Parameter, "TurbulenceModel", svalue) );
          printf("%s\n",svalue);



          // declare global constants for device (OptiX)
          context["Dim"]->setInt(Dim);
          context["StateRank"]->setInt(StateRank);
          context["SpecificHeatRatio"]->setUserData(sizeof(SpecificHeatRatio),&SpecificHeatRatio);
          context["GasConstant"]->setUserData(sizeof(GasConstant), &GasConstant);


          // declare global constants for device (CUDA)
//          std::string stateRankName = "StateRank";
//          CudaGlobalVariable<int> stateRankVariable((stateRankName), (module));
//          stateRankVariable.WriteToDevice(StateRank);

//          std::string DimName = "Dim";
//          CudaGlobalVariable<int> DimVariable((DimName), (module));
//          DimVariable.WriteToDevice(Dim);

//          std::string SpecificHeatRatioName = "SpecificHeatRatio";
//          CudaGlobalVariable<PX_REAL> SpecificHeatRatioVariable((SpecificHeatRatioName), (module));
//          SpecificHeatRatioVariable.WriteToDevice(SpecificHeatRatio);

//          std::string GasConstantName = "GasConstant";
//          CudaGlobalVariable<PX_REAL> GasConstantVariable((GasConstantName), (module));
//          GasConstantVariable.WriteToDevice(GasConstant);

          // set up attachment visualization
          char distanceName[] = "QnDistanceFunction";
          int ierr;
          PX_AttachmentGlobRealElem *QnDistance = NULL;
          ierr = PXAttachSearch( pg, "QnDistanceFunction", NULL, (void**)&QnDistance);
          if(ierr == PX_SEARCH_NOT_FOUND){
              // distance function doesn't exist; create a dummy one and set it to all 0s
              enum PXE_BasisShape BasisShape;
              enum PXE_BasisNodeDistribution NodeDistribution;
              enum PXE_SolutionOrder *attachOrder = (enum PXE_SolutionOrder*) malloc(pg->nElementGroup*sizeof(int));
              for(egrp = 0; egrp<pg->nElementGroup; egrp++){
                  // get basis shape
                  PXError( PXGetTypeBasisShape(pg->ElementGroup[egrp].type, &BasisShape) );
                  // get node distribution
                  PXError( PXGetTypeBasisNodeDistribution( pg->ElementGroup[egrp].type, &NodeDistribution) );
                  // get enumerated order
                  PXError( PXEnumeratedOrder(2, Dim, PXE_Lagrange, NodeDistribution, BasisShape, &(attachOrder[egrp]) ) );
              }

              PXError( PXCreateAttachGlobRealElem(pg, distanceName, attachOrder, 1, &QnDistance) );
              PXError( PXAttachGRESetZero(pg, QnDistance) );

              free(attachOrder);
          }
          // enforce that solutionorder of ALL egrp in attachment are the same
          // enforce that StateRank of attachemnt is 1
          enum PXE_SolutionOrder attachBaseOrder = QnDistance->order[0];
          if(QnDistance->StateRank != 1){
              printf("ERROR: distance function attachment staterank is %d != 1\n",QnDistance->StateRank);
              exit(1);
          }
          for(egrp=1; egrp<pg->nElementGroup; egrp++){
              if(QnDistance->order[egrp] != attachBaseOrder){
                  printf("ERROR: QnDistance->order[%d] = %d != baseOrder (%d)\n",egrp, QnDistance->order[egrp],attachBaseOrder);
                  exit(1);
              }
          }


          // gather some grid info
          for(egrp = 0; egrp<pg->nElementGroup; egrp++){
              nElemTotal += pg->ElementGroup[egrp].nElement; //total # elements

              PXOrder2nbf(State->order[egrp], &nbf);
              //printf("order=%d, nbf=%d, on egrp %d\n",State->order[egrp],nbf,egrp);
              PXType2nbf(pg->ElementGroup[egrp].type, &nbfQ);
              PXError(PXOrder2nbf(QnDistance->order[egrp],&nbfA));

              nSolnCoeffTotal += nbf*pg->ElementGroup[egrp].nElement;
              nGeomCoeffTotal += nbfQ*pg->ElementGroup[egrp].nElement;
              nAttachCoeffTotal += nbfA*pg->ElementGroup[egrp].nElement;

              if(pg->ElementGroup[egrp].type == PXE_TetCut){
                  nCutCellTotal += pg->ElementGroup[egrp].nElement;
              }
          }
          printf("nCutCellTotal = %d\n",nCutCellTotal);

          // create material
          optixu::Material m_PXSimplexCutSurfaceMaterial = context->createMaterial();

          // set optix user programs
          optixu::Program PXSimplexBoundingProgram = PtxManager::LoadProgram(context, GetPTXPrefix(), PXSimplexBoundingProgramName);
          optixu::Program PXSimplexIntersectionProgram = PtxManager::LoadProgram(context, GetPTXPrefix(), PXSimplexIntersectionProgramName);


          //set optix geometry groups
          optixu::Geometry geometry = context->createGeometry();
          geometry->setPrimitiveCount(nElemTotal);
          optixu::GeometryInstance PXSimplexInstance = context->createGeometryInstance();
          PXSimplexInstance->setGeometry(geometry);
          PXSimplexInstance->setMaterialCount(1);
          PXSimplexInstance->setMaterial(0, m_PXSimplexCutSurfaceMaterial);

          optixu::Geometry PXSimplexGeometry = PXSimplexInstance->getGeometry();
          PXSimplexGeometry->setBoundingBoxProgram( PXSimplexBoundingProgram );
          PXSimplexGeometry->setIntersectionProgram( PXSimplexIntersectionProgram );


          optixu::GeometryGroup group = context->createGeometryGroup();
          group->setChildCount(1);
          group->setChild(0, PXSimplexInstance);
          //group->setChild(1, prismInstance);

          // std::string seedName = "PXSimplexSeed";
          // optixu::Buffer seedBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, 1);
          // context[seedName.c_str()]->set(seedBuffer);
          // int* seedData = static_cast<int*>(seedBuffer->map());

          // seedData[0] = 3873;
          // seedBuffer->unmap();




          // convert to buffers...
          m_solutionBuffer.SetContext(context);
          m_solutionBuffer.SetDimensions(nSolnCoeffTotal*StateRank);
          BOOST_AUTO(solution, m_solutionBuffer.map());

          m_coordinateBuffer.SetContext(context);
          m_coordinateBuffer.SetDimensions(nGeomCoeffTotal*Dim);
          BOOST_AUTO(coordinate, m_coordinateBuffer.map());

          // could fill this with boundingbox program once optix starts
          // BUT currently filling it on the CPU for convenience
          m_boundingBoxBuffer.SetContext(context);
          m_boundingBoxBuffer.SetDimensions(nElemTotal*BBOX_SIZE);
          BOOST_AUTO(boundingBox, m_boundingBoxBuffer.map());

          m_egrpDataBuffer.SetContext(context);
          m_egrpDataBuffer.SetDimensions(pg->nElementGroup);
          BOOST_AUTO(egrpData, m_egrpDataBuffer.map());

          m_attachDataBuffer.SetContext(context);
          m_attachDataBuffer.SetDimensions(pg->nElementGroup);
          BOOST_AUTO(attachData, m_attachDataBuffer.map());

          m_attachmentBuffer.SetContext(context);
          m_attachmentBuffer.SetDimensions(nAttachCoeffTotal*QnDistance->StateRank);
          BOOST_AUTO(attachment, m_attachmentBuffer.map());

          m_globalElemToEgrpElemBuffer.SetContext(context);
          m_globalElemToEgrpElemBuffer.SetDimensions(2*nElemTotal);
          BOOST_AUTO(globalElemToEgrpElem, m_globalElemToEgrpElemBuffer.map());

          // Cut Cell magic
          if(m_cutCellFlag == 1){
              PX_Mesh *meshSurf = pg->CC3D->meshsurf;
              PX_Mesh *meshBack = pg->CC3D->meshback;
              PX_ThreeD_Intersect const *Intersect = pg->CC3D->Intersect;
              PX_MeshElement *Element;
              int const* egrpelem2ThreeD;
              ptrdiff_t lengthCheck;

              // assert that all elements in meshsurf are Q2 tets
              enum PXE_ElementType truthSurfType = PXE_UniformTriangleQ2;
              int nCorrect=0, nCorrect2=0, nCorrect3=0;
              for(elem=0; elem<meshSurf->nElement; elem++){
                  nCorrect += meshSurf->Element[elem].MasterGridElement->Type == truthSurfType;
              }
              if(nCorrect != meshSurf->nElement){
                  const std::string message = "Some elements of pg->CC3D->meshsurf have elements that are not UniformTriangleQ2!\n";
                  printf("%s",message.c_str());
                  std::runtime_error e(message);
                  throw e;
              }

              // assert that all shadow types are Q1 Tet
              enum PXE_ElementType truthShadowType = PXE_UniformTetQ1;
              int nThreeDTotal = 0;
              int kThreeD;
              int nCompThreeD;
              int const* threeDList;
              ThreeD_ThreeD const* ThreeD;

              for(egrp = 0; egrp<pg->nElementGroup; egrp++){
                  if(pg->ElementGroup[egrp].type == PXE_TetCut){
                      nCorrect = 0; nCorrect2 = 0; nCorrect3 = 0;
                      nThreeDTotal = 0;
                      for(elem=0; elem<pg->ElementGroup[egrp].nElement; elem++){
                          nCorrect += pg->ElementGroup[egrp].ShadowType[elem] == truthShadowType;
                          nCorrect2 += pg->IGM->egrpelem2nThreeD[egrp][elem] == 1;

                          egrpelem2ThreeD = pg->IGM->egrpelem2ThreeD[egrp][elem];

                          //kThreeD = egrpelem2ThreeD[jThreeD];
                          kThreeD = egrpelem2ThreeD[0]; //already asserted that nThreeD is 1 for
                          //every element
                          ThreeD = Intersect->ThreeD + kThreeD;

                          //Test for each component threeD separately, if this is merged
                          nCompThreeD = ThreeD->nCompThreeD;
                          threeDList = ThreeD->threeDList;

                          nThreeDTotal += nCompThreeD;
                          for(k = 0; k < nCompThreeD; k++){
                              ThreeD = Intersect->ThreeD + threeDList[k];
                              Element = meshBack->Element + ThreeD->backElem;
                              nCorrect3 += Element->MasterGridElement->Type == truthShadowType;
                          }
                      }
                      if(nCorrect != pg->ElementGroup[egrp].nElement){
                          const std::string message = "Some cut cells have types that are not UniformTetQ1!\n";
                          printf("%s",message.c_str());
                          std::runtime_error e(message);
                          throw e;
                      }
                      if(nCorrect2 != pg->ElementGroup[egrp].nElement){
                          const std::string message = "Some cut cells have nThreeD != 1!\n";
                          printf("%s",message.c_str());
                          std::runtime_error e(message);
                          throw e;
                      }
                      if(nCorrect3 != nThreeDTotal){
                          const std::string message = "Some cut cells have background elements that are not Q1 Tet!\n";
                          printf("%s",message.c_str());
                          std::runtime_error e(message);
                          throw e;
                      }
                      // if(nCorrect4 != pg->ElementGroup[egrp].nElement){
                      // 	const std::string message = "Some cut cells have WholeElement type\n";
                      // 	printf("%s",message.c_str());
                      // 	std::runtime_error e(message);
                      // 	throw e;
                      // }
                  }
              }

              puts("Assertions passed.");

              //Load surface mesh coordinates (quadratic patches)
              m_patchCoordinateBuffer.SetContext(context);
              m_patchCoordinateBuffer.SetDimensions(meshSurf->nElement*PATCH_NBF*Dim);
              BOOST_AUTO(patchCoordinate, m_patchCoordinateBuffer.map());

              geomRank = PATCH_NBF*Dim;
              ElVisFloat* patchCoordPtr = patchCoordinate.get();
              for(elem=0; elem<meshSurf->nElement; elem++){
                  // for(i=0; i<nbfQ; i++){
                  //   patchCoordPtr[i*Dim +0] = meshSurf->Element[elem].coordinate[i*Dim+0];
                  //   patchCoordPtr[i*Dim +1] =
                  //   patchCoordPtr[i*Dim +2] =
                  // }
                  for(i=0; i<geomRank; i++){
                      patchCoordPtr[i] = meshSurf->Element[elem].coordinate[i];
                  }
                  patchCoordPtr += geomRank;
              }
              //sanity check that pointers are where they should be
              lengthCheck = (char *)patchCoordPtr - (char *)patchCoordinate.get();
              printf("Length check: %ld\n",lengthCheck - (ptrdiff_t)(meshSurf->nElement*geomRank*sizeof(*patchCoordPtr)));

              puts("PatchCoordinateBuffer complete");

              // Load shadow element coordinates
              m_shadowCoordinateBuffer.SetContext(context);
              m_shadowCoordinateBuffer.SetDimensions(nCutCellTotal*SHADOW_NBF*Dim);
              BOOST_AUTO(shadowCoordinate, m_shadowCoordinateBuffer.map());

              m_egrpToShadowIndexBuffer.SetContext(context);
              m_egrpToShadowIndexBuffer.SetDimensions(pg->nElementGroup);
              BOOST_AUTO(EgrpToShadowIndex, m_egrpToShadowIndexBuffer.map());

              geomRank = SHADOW_NBF*Dim;
              ElVisFloat* shadowCoordPtr = shadowCoordinate.get();
              int tempCutCellCtr = 0;
              for(egrp = 0; egrp<pg->nElementGroup; egrp++){

                  if(pg->ElementGroup[egrp].type == PXE_TetCut){
                      EgrpToShadowIndex[egrp] = tempCutCellCtr;
                      tempCutCellCtr += pg->ElementGroup[egrp].nElement*geomRank;

                      for(elem=0; elem<pg->ElementGroup[egrp].nElement; elem++){
                          for(i=0; i<geomRank; i++){
                              shadowCoordPtr[i] = pg->ElementGroup[egrp].ShadowVert[elem][i];
                          }
                          shadowCoordPtr += geomRank;
                      }
                  }else
                      EgrpToShadowIndex[egrp] = (unsigned int) -1;
              }

              puts("ShadowCoordinateBuffer complete.");

              // Compute size of cutcell buffer
              // size (in bytes): nCutCellTotal*sizeof(PX_CutCellElVis) + nThreeDTotal*sizeof(PX_PatchGroup) + nPatchIndexesTotal*sizeof(int)
              nThreeDTotal = 0;
              int nPatchIndexesTotal = 0;
              int nLinkedTwoD;
              int kTwoD, jTwoD;
              int *linkedTwoD;
              ThreeD_TwoD* TwoD;
              ThreeD_TwoD* allTwoD = Intersect->TwoD;
              PX_REAL const* knownPoint;
              for(egrp = 0; egrp<pg->nElementGroup; egrp++){
                  if(pg->ElementGroup[egrp].type == PXE_TetCut){
                      for(elem=0; elem<pg->ElementGroup[egrp].nElement; elem++){
                          egrpelem2ThreeD = pg->IGM->egrpelem2ThreeD[egrp][elem];

                          //kThreeD = egrpelem2ThreeD[jThreeD];
                          kThreeD = egrpelem2ThreeD[0]; //already asserted that nThreeD is 1 for
                          //every element
                          ThreeD = Intersect->ThreeD + kThreeD;

                          //Test for each component threeD separately, if this is merged
                          nCompThreeD = ThreeD->nCompThreeD;
                          threeDList = ThreeD->threeDList;

                          nThreeDTotal += nCompThreeD;
                          for(k = 0; k < nCompThreeD; k++){
                              ThreeD = Intersect->ThreeD + threeDList[k];
                              //knownPoint = ThreeD->knownPoint;

                              nLinkedTwoD = ThreeD->nLinkedTwoD;
                              linkedTwoD = ThreeD->linkedTwoD;

                              for(kTwoD = 0; kTwoD < nLinkedTwoD; kTwoD ++){
                                  jTwoD = linkedTwoD[kTwoD];
                                  TwoD = allTwoD + jTwoD;

                                  //If it is on patch face
                                  if(TwoD->TypeOnPatch == PXE_3D_2DPatchFace){
                                      nPatchIndexesTotal += 1;
                                  }
                              }//kTwoD over nLinkedTwoD
                          }//k over nCompThreeD

                      }//elem
                  }
              }//egrp
              puts("CutCellBuffer sizing data gathered.");
              printf("nThreeDTotal = %d, nPatchIndexesTotal = %d\n", nThreeDTotal, nPatchIndexesTotal);

              m_knownPointBuffer.SetContext(context);
              m_knownPointBuffer.SetDimensions(nThreeDTotal*Dim);
              BOOST_AUTO(knownPointBase, m_knownPointBuffer.map());

              m_backgroundCoordinateBuffer.SetContext(context);
              m_backgroundCoordinateBuffer.SetDimensions(nThreeDTotal*BACK_NBF*Dim);
              BOOST_AUTO(backgroundCoordinateBase, m_backgroundCoordinateBuffer.map());

              m_cutCellBuffer.SetContext(context);
              m_cutCellBuffer.SetDimensions(nCutCellTotal*sizeof(PX_CutCellElVis) +
                                          nThreeDTotal*sizeof(PX_PatchGroup) +
                                          nPatchIndexesTotal*sizeof(int));
              BOOST_AUTO(CutCellBase, m_cutCellBuffer.map());

              m_globalElemToCutCellBuffer.SetContext(context);
              m_globalElemToCutCellBuffer.SetDimensions(nElemTotal);
              BOOST_AUTO(GlobalElemToCutCell, m_globalElemToCutCellBuffer.map());

              int threeDCtr = 0;
              int nPatch;
              int totalPatchGroupLength; //cummulative length of patch groups over 1 cut cell
              int *patchList;
              PX_PatchGroup *patchGroup;
              PX_CutCellElVis *cutCell = (PX_CutCellElVis *)CutCellBase.get();
              unsigned int* globalToCutPtr = GlobalElemToCutCell.get();
              geomRank = BACK_NBF*Dim;
              PX_REAL *backgroundCoordPtr = backgroundCoordinateBase.get();
              PX_REAL *knownPointPtr = knownPointBase.get();

              for(egrp = 0; egrp<pg->nElementGroup; egrp++){
                  if(pg->ElementGroup[egrp].type == PXE_TetCut){
                      for(elem=0; elem<pg->ElementGroup[egrp].nElement; elem++){
                          egrpelem2ThreeD = pg->IGM->egrpelem2ThreeD[egrp][elem];

                          //kThreeD = egrpelem2ThreeD[jThreeD];
                          kThreeD = egrpelem2ThreeD[0]; //already asserted that nThreeD is 1 for
                          //every element
                          ThreeD = Intersect->ThreeD + kThreeD;

                          //Test for each component threeD separately, if this is merged
                          nCompThreeD = ThreeD->nCompThreeD;
                          threeDList = ThreeD->threeDList;

                          patchGroup = GetFirstPatchGroup(cutCell);
                          totalPatchGroupLength = 0;
                          for(k = 0; k < nCompThreeD; k++){
                              ThreeD = Intersect->ThreeD + threeDList[k];

                              //setup coordinates of background element for this patch group
                              Element = meshBack->Element + ThreeD->backElem;

                              for(i=0; i<geomRank; i++){
                                  backgroundCoordPtr[i] = Element->coordinate[i];
                              }
                              // for(i=0; i<geomRank; i++){
                              //   patchGroup->backgroundElemCoord[i] = Element->coordinate[i];
                              // }

                              //set up knownpoint for this patch group
                              knownPoint = ThreeD->knownPoint;
                              knownPointPtr[0] = knownPoint[0];
                              knownPointPtr[1] = knownPoint[1];
                              knownPointPtr[2] = knownPoint[2];
                              // patchGroup->knownPoint[0] = knownPoint[0];
                              // patchGroup->knownPoint[1] = knownPoint[1];
                              // patchGroup->knownPoint[2] = knownPoint[2];
                              // patchGroup->knownPointx = knownPoint[0];
                              // patchGroup->knownPointy = knownPoint[1];
                              // patchGroup->knownPointz = knownPoint[2];

                              patchGroup->knownPointFlag = (unsigned char) ThreeD->KnownPointType;

                              // Build list of quadratic patches tied to this ThreeD
                              nLinkedTwoD = ThreeD->nLinkedTwoD;
                              linkedTwoD = ThreeD->linkedTwoD;
                              // set up patchlist for this patchgroup
                              patchList = GetPatchList(patchGroup);
                              nPatch = 0;
                              for(kTwoD = 0; kTwoD < nLinkedTwoD; kTwoD ++){
                                  jTwoD = linkedTwoD[kTwoD];
                                  TwoD = allTwoD + jTwoD;

                                  //If it is on patch face
                                  if(TwoD->TypeOnPatch == PXE_3D_2DPatchFace){
                                      //fill patchList entry
                                      patchList[nPatch] = TwoD->patchFace;
                                      //increment npatch counter
                                      nPatch += 1;
                                  }
                              }//kTwoD over nLinkedTwoD

                              patchGroup->length = sizeof(*patchGroup) + nPatch*sizeof(int);
                              patchGroup->nPatch = nPatch;
                              patchGroup->threeDId = threeDCtr;
                              totalPatchGroupLength += patchGroup->length;
                              patchGroup = GetNextPatchGroup(patchGroup);

                              threeDCtr += 1;
                              knownPointPtr += Dim;
                              backgroundCoordPtr += geomRank;
                          }//k over nCompThreeD

                          globalToCutPtr[elem] = (unsigned int)((char*)cutCell - (char*)CutCellBase.get());;

                          cutCell->length = sizeof(*cutCell)+totalPatchGroupLength;
                          cutCell->nPatchGroup = nCompThreeD;
                          cutCell = GetNextCutCell(cutCell);
                      }//elem

                  }else{
                      for(elem=0; elem<pg->ElementGroup[egrp].nElement; elem++){
                          globalToCutPtr[elem] = (unsigned int) -1;
                      }
                  }
                  globalToCutPtr += pg->ElementGroup[egrp].nElement;
              }//egrp

              if(threeDCtr != nThreeDTotal)
                  printf("MISMATCH: threeDCtr = %d, nThreeDTotal = %d\n",threeDCtr,nThreeDTotal);

              //sanity check: make sure cutCell pointer has exactly traversed the space
              //allocated to CutCellBuffer
              lengthCheck = (char *)cutCell - (char *)CutCellBase.get();
              printf("CutCell length check: %ld\n",lengthCheck - (ptrdiff_t)(nCutCellTotal*sizeof(PX_CutCellElVis) + nThreeDTotal*sizeof(PX_PatchGroup) + nPatchIndexesTotal*sizeof(int)));

              //sanity check: make sure the length values in CutCell & PatchGroup types
              //are at least consistent
              ptrdiff_t patchGroupCheck;
              unsigned int* cutCellSizeBase = (unsigned int*)malloc(nCutCellTotal*sizeof(unsigned int));
              unsigned int* cutCellSizePtr = cutCellSizeBase;
              cutCell = (PX_CutCellElVis *)CutCellBase.get();
              for(i=0; i<nCutCellTotal; i++){
                  patchGroup = GetFirstPatchGroup(cutCell);
                  for(j=0; j<cutCell->nPatchGroup; j++){
                      patchGroup = GetNextPatchGroup(patchGroup);
                  }
                  patchGroupCheck = (char *)patchGroup - (char*)GetNextCutCell(cutCell);
                  if(patchGroupCheck != (ptrdiff_t)0){
                      printf("length check FAILED on CutCell %d!\n",i);
                      printf("length difference: %ld\n",patchGroupCheck);
                  }

                  cutCellSizePtr[i] = cutCell->length;
                  cutCell = GetNextCutCell(cutCell);
              }
              lengthCheck = (char *)cutCell - (char *)CutCellBase.get();
              printf("CutCell length check: %ld\n",lengthCheck - (ptrdiff_t)(nCutCellTotal*sizeof(PX_CutCellElVis) + nThreeDTotal*sizeof(PX_PatchGroup) + nPatchIndexesTotal*sizeof(int)));

              // Check that GlobalElemToCutCellBuffer is set up correctly
              cutCellSizePtr = cutCellSizeBase;
              globalToCutPtr = GlobalElemToCutCell.get();
              //cutCell = (PX_CutCellElVis *)CutCellBase;
              int globalElemCtr = 0;
              int cutCellCtr = 0;
              char passTestFlag = 1;
              for(egrp=0; egrp<pg->nElementGroup; egrp++){
                  if(pg->ElementGroup[egrp].type == PXE_TetCut){
                      for(elem=0; elem<pg->ElementGroup[egrp].nElement; elem++){
                          cutCell = (PX_CutCellElVis*)((char*)CutCellBase.get() + GlobalElemToCutCell[globalElemCtr]);
                          if(cutCell->length != cutCellSizePtr[cutCellCtr]){
                              printf("on global elem %d, cutCell->length = %d, cutCellSizePtr[%d] = %d\n",globalElemCtr, cutCell->length, cutCellCtr, cutCellSizePtr[cutCellCtr]);
                              passTestFlag = 0;
                          }
                          globalElemCtr += 1;
                          cutCellCtr += 1;
                      }
                  }else{
                      globalElemCtr += pg->ElementGroup[egrp].nElement;
                  }
              }
              if(passTestFlag == 1)
                  puts("GlobalElemToCutCellBuffer appears consistent.");

              globalToCutPtr = GlobalElemToCutCell.get();
              globalElemCtr = 0;
              for(egrp=0; egrp<pg->nElementGroup; egrp++){
                  for(elem=0; elem<pg->ElementGroup[egrp].nElement; elem++){
                      printf("globalToCut[%d] = %d\n",globalElemCtr,globalToCutPtr[globalElemCtr]);
                      globalElemCtr += 1;
                  }
              }

              puts("CutCellBuffer complete");

              free(cutCellSizeBase);
          }else{
              m_shadowCoordinateBuffer.SetContext(context);
              m_shadowCoordinateBuffer.SetDimensions(1);

              m_egrpToShadowIndexBuffer.SetContext(context);
              m_egrpToShadowIndexBuffer.SetDimensions(1);

              m_patchCoordinateBuffer.SetContext(context);
              m_patchCoordinateBuffer.SetDimensions(1);

              m_knownPointBuffer.SetContext(context);
              m_knownPointBuffer.SetDimensions(1);

              m_backgroundCoordinateBuffer.SetContext(context);
              m_backgroundCoordinateBuffer.SetDimensions(1);

              m_cutCellBuffer.SetContext(context);
              m_cutCellBuffer.SetDimensions(1);

              m_globalElemToCutCellBuffer.SetContext(context);
              m_globalElemToCutCellBuffer.SetDimensions(1);
          }

          PX_REAL centroid[DIM3D];
          int porder;
          enum PXE_SolutionOrder orderQ;
          enum PXE_Shape shape;

          egrpData[0].egrpStartIndex = 0;
          egrpData[0].egrpGeomCoeffStartIndex = 0;
          egrpData[0].egrpSolnCoeffStartIndex = 0;
          //egrpData[0].typeData.type = (unsigned int) pg->ElementGroup[0].type;
          //egrpData[0].orderData.order = (unsigned int) State->order[0];

          for(egrp=0; egrp<pg->nElementGroup; egrp++){
              // fill in attachment data
              PXError(PXOrder2nbf(QnDistance->order[egrp], &nbf));
              PXError(PXOrder2porder(QnDistance->order[egrp], &porder));
              attachData[egrp].order = (unsigned short) QnDistance->order[egrp];
              attachData[egrp].nbf = (unsigned short) nbf;
              attachData[egrp].porder = (unsigned char) porder;

              // fill in egrp data
              egrpData[egrp].typeData.type = pg->ElementGroup[egrp].type;
              PXError(PXType2Interpolation(pg->ElementGroup[egrp].type, &orderQ));
              PXError(PXType2nbf(pg->ElementGroup[egrp].type, &nbfQ));
              PXError(PXType2qorder(pg->ElementGroup[egrp].type, &qorder));
              PXError(PXType2Shape(pg->ElementGroup[egrp].type, &shape));

              egrpData[egrp].typeData.order = (unsigned short) orderQ;
              egrpData[egrp].typeData.nbf = (unsigned short) nbfQ;
              egrpData[egrp].typeData.qorder = (unsigned char) qorder;
              egrpData[egrp].typeData.shape = (unsigned char) shape;

              PXError(PXElementCentroidReference(shape, centroid));
              for(d=0; d<Dim; d++){
                  egrpData[egrp].typeData.centroidCoord[d] = centroid[d];
              }

              egrpData[egrp].orderData.order = (unsigned short) State->order[egrp];
              PXError(PXOrder2nbf(State->order[egrp], &nbf));
              egrpData[egrp].orderData.nbf = (unsigned short) nbf;
              PXError(PXOrder2porder(State->order[egrp], &porder));
              egrpData[egrp].orderData.porder = (unsigned char) porder;

              egrpData[egrp].cutCellFlag = (char) pg->ElementGroup[egrp].type == PXE_TetCut;

              if(egrp == 0)
                  continue;

              PXOrder2nbf(State->order[egrp-1], &nbf);
              PXType2nbf(pg->ElementGroup[egrp-1].type, &nbfQ);

              egrpData[egrp].egrpStartIndex = egrpData[egrp-1].egrpStartIndex + pg->ElementGroup[egrp-1].nElement;
              egrpData[egrp].egrpGeomCoeffStartIndex = egrpData[egrp-1].egrpGeomCoeffStartIndex + nbfQ*pg->ElementGroup[egrp-1].nElement;
              egrpData[egrp].egrpSolnCoeffStartIndex = egrpData[egrp-1].egrpSolnCoeffStartIndex + nbf*pg->ElementGroup[egrp-1].nElement;
          }

          coordPtr = coordinate.get();
          bBoxPtr = boundingBox.get();
          solutionPtr = solution.get();
          attachmentPtr = attachment.get();
          globalToLocalPtr = globalElemToEgrpElem.get();
          char cutCellFlag = 0;
          for(egrp = 0; egrp<pg->nElementGroup; egrp++){
              PXOrder2nbf(State->order[egrp], &nbf);
              PXType2nbf(pg->ElementGroup[egrp].type, &nbfQ);
              PXError(PXType2qorder(pg->ElementGroup[egrp].type,&qorder));
              cutCellFlag = (char) pg->ElementGroup[egrp].type == PXE_TetCut;
              solnRank = StateRank*nbf;
              geomRank = Dim*nbfQ;

              PXError(PXOrder2nbf(QnDistance->order[egrp],&nbfA));

              for(elem=0; elem<pg->ElementGroup[egrp].nElement; elem++){
                  globalToLocalPtr[2*elem+0] = (unsigned int) egrp;
                  globalToLocalPtr[2*elem+1] = (unsigned int) elem;

                  // fill coordinates for this element
                  for(i=0; i<nbfQ; i++){
                      coordPtr[i*Dim +0] = pg->coordinate[pg->ElementGroup[egrp].Node[elem][i]][0];
                      coordPtr[i*Dim +1] = pg->coordinate[pg->ElementGroup[egrp].Node[elem][i]][1];
                      coordPtr[i*Dim +2] = pg->coordinate[pg->ElementGroup[egrp].Node[elem][i]][2];
                  }

                  // fill bounding box for this element

                  PXError(PXComputeElementBoundingBox(pg, egrp, elem, bBoxTemp));
                  // reorder from PX bounding box ordering to ordering used on GPU
                  for(d=0; d<Dim; d++){
                      bBoxPtr[d] = (ElVisFloat)bBoxTemp[2*d];
                      bBoxPtr[Dim+d] = (ElVisFloat)bBoxTemp[2*d+1];
                  }

                  if(qorder != 1 || cutCellFlag == (char)1)
                      PadBoundingBox(Dim, 0.1, bBoxPtr);

                  // fill solution array for this element
                  for(j=0; j<solnRank; j++){
                      solutionPtr[j] = State->value[egrp][elem][j];
                  }

                  // fill in attachment values
                  for(j=0; j<QnDistance->StateRank*nbfA; j++){
                      attachmentPtr[j] = QnDistance->value[egrp][elem][j];
                  }

                  coordPtr += geomRank;
                  bBoxPtr += BBOX_SIZE;
                  solutionPtr += solnRank;
                  attachmentPtr += QnDistance->StateRank*nbfA;
              }
              globalToLocalPtr += pg->ElementGroup[egrp].nElement*2;
          }

          group->setAcceleration( context->createAcceleration("Sbvh","Bvh") );
          //group->setAcceleration( context->createAcceleration("MedianBvh","Bvh") );
          //group->setAcceleration( context->createAcceleration("NoAccel","NoAccel") );
          result.push_back(group);

          puts("PX LoadVolume complete");

          return result;
      }
      catch(optixu::Exception& e)
      {
          std::cerr << e.getErrorString() << std::endl;
          throw;
      }
      catch(std::exception& f)
      {
          std::cerr << f.what() << std::endl;
          throw;
      }
  }
*/

/*
  void PXModel::DoGetFaceGeometry(Scene* scene, optixu::Context context, optixu::Geometry& faces)
  {
      PX_Grid *pg = m_pxa->pg;
      ElVisFloat *faceCoordPtr;
      ElVisFloat * bBoxMinPtr;
      ElVisFloat * bBoxMaxPtr;
      PX_FaceData* faceDataPtr;
      FaceDef* faceDefsPtr;

      PX_REAL bBoxTemp[BBOX_SIZE];
      ElVisFloat bBoxTempElVis[BBOX_SIZE];

      int geomRank;
      int fgrp, face;
      int egrp, elem, lface;
      int egrpR, elemR;
      int i;
      int qorder;
      int nbfQ, nbfQFace;
      int nFaceTotal = 0;

      int d;
      int Dim = pg->Dim;

      int orientation;
      enum PXE_Shape elemShape, faceShape;
      enum PXE_ElementType elemType, elemTypeR, baseFaceType, localFaceType;
      int maxnbfQ;

      //egrp2GlobalElemIndex array
      int *egrp2GlobalElemIndex = (int *)malloc(pg->nElementGroup*sizeof(int));
      egrp2GlobalElemIndex[0] = 0;
      for(egrp=1; egrp<pg->nElementGroup; egrp++){
          egrp2GlobalElemIndex[egrp] = egrp2GlobalElemIndex[egrp-1] + pg->ElementGroup[egrp-1].nElement;
      }

      //count number of faces
      for(fgrp=0; fgrp<pg->nFaceGroup; fgrp++){
          nFaceTotal += pg->FaceGroup[fgrp].nFace;
      }

      //assert that all face types are the same
      PXElemType2FaceType(pg->ElementGroup[0].type, 0, &baseFaceType );
      PXType2nbf(pg->ElementGroup[0].type, &nbfQ);
      maxnbfQ = nbfQ;
      for(egrp=0; egrp<pg->nElementGroup; egrp++){
          PXType2nbf(pg->ElementGroup[egrp].type, &nbfQ);
          if(nbfQ > maxnbfQ)
              maxnbfQ = nbfQ;

          for(lface=0; lface<pg->ElementGroup[egrp].nFace; lface++){
              if(pg->ElementGroup[egrp].type == PXE_TetCut){
                  localFaceType = PXE_UniformTetQ1;
              }else{
                  PXElemType2FaceType(pg->ElementGroup[egrp].type, lface, &localFaceType );
              }

              if(localFaceType != baseFaceType){
                  const std::string message = "Some faces do not have the same type!\n";
                  printf("%s",message.c_str());
                  std::runtime_error e(message);
                  throw e;
              }
          }
      }
      PXType2nbf(baseFaceType, &nbfQFace);
      geomRank = nbfQFace*Dim;

      m_faceCoordinateBuffer.SetContext(context);
      m_faceCoordinateBuffer.SetDimensions(nFaceTotal*geomRank);
      BOOST_AUTO(faceCoordinate, m_faceCoordinateBuffer.map());


      m_faceDataBuffer.SetContext(context);
      m_faceDataBuffer.SetDimensions(nFaceTotal);
      BOOST_AUTO(faceData, m_faceDataBuffer.map());


      scene->GetFaceMinExtentBuffer().SetDimensions(nFaceTotal);
      scene->GetFaceMaxExtentBuffer().SetDimensions(nFaceTotal);

      BOOST_AUTO(minBuffer, scene->GetFaceMinExtentBuffer().map());
      BOOST_AUTO(maxBuffer, scene->GetFaceMaxExtentBuffer().map());

      ////////////////////////////////////////
      // BLAKE - TODO
      // Fill in the FaceDef struct for each face.  This struct tells ElVis
      // which elements are in/out and if the face is curved or planar.
      ////////////////////////////////////////

      scene->GetFaceIdBuffer().SetDimensions(nFaceTotal);
      BOOST_AUTO(faceDefs, scene->GetFaceIdBuffer().map());

      PX_REAL *nodeCoord;
      nodeCoord = (PX_REAL*) malloc(maxnbfQ*(Dim+1)*sizeof(PX_REAL));
      PX_REAL *phiQ = nodeCoord + maxnbfQ*Dim;


      int nodesOnFace[36];
      int nNodesOnFace;

      faceDefsPtr = faceDefs.get();
      faceCoordPtr = faceCoordinate.get();
      faceDataPtr = faceData.get();
      bBoxMinPtr = reinterpret_cast<ElVisFloat*>(minBuffer.get());
      bBoxMaxPtr = reinterpret_cast<ElVisFloat*>(maxBuffer.get());
      for(fgrp=0; fgrp<pg->nFaceGroup; fgrp++){
          for(face=0; face<pg->FaceGroup[fgrp].nFace; face++){
              // for now, ALWAYS use LEFT face
              egrp = pg->FaceGroup[fgrp].FaceL[face].ElementGroup;
              elem = pg->FaceGroup[fgrp].FaceL[face].Element;
              lface = pg->FaceGroup[fgrp].FaceL[face].Face;
              elemType = pg->ElementGroup[egrp].type;

//              if ( (pg->FaceGroup[fgrp].FaceGroupFlag!=PXE_BoundaryFG)&& (pg->FaceGroup[fgrp].FaceGroupFlag!=PXE_EmbeddedBoundaryFG) ){
//                  egrpR = pg->FaceGroup[fgrp].FaceR[face].ElementGroup;
//                  elemR = pg->FaceGroup[fgrp].FaceR[face].Element;
//                  //lfaceR = pg->FaceGroup[fgrp].FaceR[face].Face;
//                  elemTypeR = pg->ElementGroup[egrpR].type;
//              }else{
//                  egrpR = egrp;
//                  elemR = elem;
//                  //lfaceR = lface;
//                  elemTypeR = elemType;
//              }

	      //for curved faces
              PXNodesOnFace(elemType, lface, nodesOnFace, &nNodesOnFace);
              if(nNodesOnFace != nbfQFace)
                  puts("FAIL!!!!!!!!!!");

	      // to force linear faces
	      //PXVertexNodesOnFace(elemType, lface, nodesOnFace, nNodesOnFace);

              PXType2nbf(elemType, &nbfQ);
              // fill coordinates for this element
              for(i=0; i<nbfQ; i++){
                  nodeCoord[i*Dim +0] = pg->coordinate[pg->ElementGroup[egrp].Node[elem][i]][0];
                  nodeCoord[i*Dim +1] = pg->coordinate[pg->ElementGroup[egrp].Node[elem][i]][1];
                  nodeCoord[i*Dim +2] = pg->coordinate[pg->ElementGroup[egrp].Node[elem][i]][2];
              }

              //use nbfQFace b/c we assume ALL FACES ARE THE SAME TYPE
              for(i=0; i<nNodesOnFace; i++){
                  faceCoordPtr[i*Dim + 0] = pg->coordinate[pg->ElementGroup[egrp].Node[elem][nodesOnFace[i]]][0];
                  faceCoordPtr[i*Dim + 1] = pg->coordinate[pg->ElementGroup[egrp].Node[elem][nodesOnFace[i]]][1];
                  faceCoordPtr[i*Dim + 2] = pg->coordinate[pg->ElementGroup[egrp].Node[elem][nodesOnFace[i]]][2];
              }

              PXComputeFaceBoundingBox(pg, fgrp, face, PXE_Left, nodeCoord, phiQ, bBoxTemp);
              // reorder from PX bounding box ordering to ordering used on GPU
              for(d=0; d<Dim; d++){
                  bBoxTempElVis[d] = (ElVisFloat)bBoxTemp[2*d];
                  bBoxTempElVis[Dim+d] = (ElVisFloat)bBoxTemp[2*d+1];
              }

              PXType2qorder(elemType,&qorder);
              //if(qorder != 1)
                  PadBoundingBox(Dim, 0.1, bBoxTempElVis);

              // bBoxTemp[0] = 0.0; bBoxTemp[1] = 0.0;
              // bBoxTemp[2] = 0.0; bBoxTemp[3] = 0.0;
              // bBoxTemp[4] = 0.0; bBoxTemp[5] = 0.0;
              for(d=0; d<Dim; d++){
                  bBoxMinPtr[d] = bBoxTempElVis[d];
                  bBoxMaxPtr[d] = bBoxTempElVis[Dim+d];
              }

              bBoxMinPtr += Dim;
              bBoxMaxPtr += Dim;
	      //for curved faces
              faceCoordPtr += geomRank;
	      //force linear faces
	      //faceCoordPtr += Dim*nNodesOnFace;

              // set up face data
              PXError( PXFaceOrientation(pg, egrp, elem, lface, &orientation) );
              PXError(PXType2Shape(elemType, &elemShape));
              PXError(PXElemShape2FaceShape(elemShape, lface, &faceShape));

              faceDataPtr[face].orientation = (unsigned char) orientation;
              faceDataPtr[face].side = (unsigned char) 0; //ALL faces are LEFT for now
              faceDataPtr[face].shape = (unsigned char) faceShape;

              faceDefsPtr[face].CommonElements[0].Id = egrp2GlobalElemIndex[egrp] + elem;
              faceDefsPtr[face].CommonElements[0].Type = (int) elemType;

   

              if ( (pg->FaceGroup[fgrp].FaceGroupFlag!=PXE_BoundaryFG)&& (pg->FaceGroup[fgrp].FaceGroupFlag!=PXE_EmbeddedBoundaryFG) ){
                  egrpR = pg->FaceGroup[fgrp].FaceR[face].ElementGroup;
                  elemR = pg->FaceGroup[fgrp].FaceR[face].Element;
                  elemTypeR = pg->ElementGroup[egrpR].type;

                  faceDefsPtr[face].CommonElements[1].Id = egrp2GlobalElemIndex[egrpR] + elemR;
                  faceDefsPtr[face].CommonElements[1].Type = (int) elemTypeR;

              }else{
                  faceDefsPtr[face].CommonElements[1].Id = -1;
                  faceDefsPtr[face].CommonElements[1].Type = -1;
              }

              if(qorder == 1)
                  faceDefsPtr[face].Type = ePlanar;
              else
                  faceDefsPtr[face].Type = eCurved;

              // if ( (pg->FaceGroup[fgrp].FaceGroupFlag==PXE_BoundaryFG)|| (pg->FaceGroup[fgrp].FaceGroupFlag==PXE_EmbeddedBoundaryFG) ){
              // 	printf("Boundary Face %d\n",globalFaceIndex);
              // }
              // globalFaceIndex += 1;
          }

          faceDataPtr += pg->FaceGroup[fgrp].nFace;
          faceDefsPtr += pg->FaceGroup[fgrp].nFace;
      }


      free(nodeCoord);
      free(egrp2GlobalElemIndex);

      faces->setPrimitiveCount(nFaceTotal);

      int nFaceVertex = 3;
      context["nFaceVertex"]->setInt(nFaceVertex);
      context["nbfQFace"]->setInt(nbfQFace);


      context["faceType"]->setInt((int) baseFaceType);
      enum PXE_SolutionOrder faceOrder;
      int porderFace;
      PXType2Interpolation(baseFaceType,&faceOrder);
      PXOrder2porder(faceOrder, &porderFace);
      context["faceOrder"]->setInt(faceOrder);
      context["porderFace"]->setInt(porderFace);

      //to force linear faces
      // context["nFaceVertex"]->setInt(3);
      // context["nbfQFace"]->setInt(3);
      // context["faceType"]->setInt((int) PXE_UniformTriangleQ1);
      // context["faceOrder"]->setInt((int) PXE_LagrangeP1);
      // context["porderFace"]->setInt(1);

//      std::string nFaceVertexName = "nFaceVertex";
//      CudaGlobalVariable<int> nFaceVertexVariable((nFaceVertexName), (module));
//      nFaceVertexVariable.WriteToDevice(nFaceVertex);

//      std::string nbfQFaceName = "nbfQFace";
//      CudaGlobalVariable<int> nbfQFaceVariable((nbfQFaceName), (module));
//      nbfQFaceVariable.WriteToDevice(nbfQFace);

//      std::string faceTypeName = "faceType";
//      CudaGlobalVariable<int> faceTypeVariable((faceTypeName), (module));
//      faceTypeVariable.WriteToDevice((int) baseFaceType);

//      std::string faceOrderName = "faceOrder";
//      CudaGlobalVariable<int> faceOrderVariable((faceOrderName), (module));
//      faceOrderVariable.WriteToDevice(faceOrder);

//      std::string porderFaceName = "porderFace";
//      CudaGlobalVariable<int> porderFaceVariable((porderFaceName), (module));
//      porderFaceVariable.WriteToDevice(porderFace);


      printf("baseFaceType = %d\n", baseFaceType);
      printf("faceOrder = %d\n", faceOrder);
      printf("nbfQFace = %d\n",nbfQFace);
      printf("porderFace = %d\n",porderFace);
  }
  */



}

