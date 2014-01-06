#ifndef _PXSTRUCTDEFINITIONS_H
#define _PXSTRUCTDEFINITIONS_H

extern "C"{
#include <Fundamentals/PX.h>
}

#define PX_DEBUG_MODE 0

#define BBOX_SIZE 6 //number of doubles used to describe a bounding box
#define MAX_NBF 10 //p=2 simplex, 3d
#define MAX_NBF_FACE 6 //p=2 simplex, 2d

#define SOLN_MAX_NBF 20 //p=2 simplex, 3d
#define SOLN_MAX_NBF_FACE 10 //p=2 simplex, 2d

#define DIM3D 3
#define DIM4D 4

#define FLOW_RANK 5 //rho, rho*u, rho*v, rho*w, rho*E
#define MAX_STATERANK 7 //rho,rho*u,rho*v,rho*w,rho*E,rho*nutilde,rho*dissipation

/* basis function counts for geometry related to cut cells */
#define PATCH_NBF 6 //Q2 triangle 
#define SHADOW_NBF 4 //Q1 Tet
#define BACK_NBF 4 //Q1 Tet, nbf for background element

#define MAX_NQUAD_2DSIMPLEX 42
#define MAX_NQUAD_LINE 12

#define KNOWNPT_INSIDE 0
#define KNOWNPT_OUTSIDE 1

#ifdef __CUDACC__
#define GEN_PRINTF ELVIS_PRINTF
#else
#define GEN_PRINTF printf
#endif

#ifdef ELVIS_OPTIX_MODULE
#define ALWAYS_PRINTF rtPrintf
#else
#define ALWAYS_PRINTF printf
#endif


typedef struct{
  enum PXE_ElementType type; //PXE_ElementType
  enum PXE_SolutionOrder order; //PXE_SolutionOrder
  int nbf;
  int qorder; //polynomial order
  enum PXE_Shape shape; //PXE_Shape
  //PX_REAL centroidCoord[DIM3D]; //coordinates of element centroid
  
} PX_ElementTypeData;

typedef struct{
  unsigned char orientation;
  unsigned char side; //0: LEFT, 1: RIGHT.  If side is 1, computed normal must be flipped
  enum PXE_Shape shape; //PXE_Shape
} PX_FaceData;


typedef struct{
  enum PXE_SolutionOrder order; //PXE_SolutionOrder
  int porder;
  int nbf;
} PX_SolutionOrderData;


typedef struct{
  unsigned int egrpStartIndex; //global element number of element 0
  unsigned int egrpGeomCoeffStartIndex; //index of element 0 in an array over coordinates
  unsigned int egrpSolnCoeffStartIndex; //index of element 0 in an array over solution
  char cutCellFlag; //flag for whether this egrp is for cut cells
  //enum PXE_SolutionOrder order; //solution order of an egrp
  //enum PXE_ElementType type; //element type of an egrp

  PX_ElementTypeData elemData; //element type of an egrp
  PX_SolutionOrderData solData; //element type of an egrp
} PX_EgrpData;


typedef struct {
  unsigned int length;
  int nPatchGroup;
} PX_CutCellElVis;

typedef struct { //2 just to differentiate it from the example above
  int length;  //serves the purpose of a "next" pointer since I can't actually make one of those
  int nPatch; //number of quadratic patches 
  int threeDId;
  unsigned char knownPointFlag; //type of known: inside (=0)/outside (=1)
} PX_PatchGroup;


#ifdef __CUDACC__
ELVIS_DEVICE
#else
static inline
#endif
PX_PatchGroup* GetFirstPatchGroup(PX_CutCellElVis *cutCell){
  return (PX_PatchGroup *)(cutCell + 1);
}

#ifdef __CUDACC__
ELVIS_DEVICE
#else
static inline
#endif
PX_CutCellElVis* GetNextCutCell(PX_CutCellElVis *cutCell){
  return (PX_CutCellElVis*)((char*)cutCell + cutCell->length);
}

#ifdef __CUDACC__
ELVIS_DEVICE
#else
static inline
#endif
PX_PatchGroup* GetNextPatchGroup(PX_PatchGroup *patchGroup){
  return (PX_PatchGroup *)((char*)patchGroup + patchGroup->length);
}

#ifdef __CUDACC__
ELVIS_DEVICE
#else
static inline
#endif
int* GetPatchList(PX_PatchGroup *patchGroup){
  return (int *)(patchGroup + 1);
}


#ifdef __CUDACC__
ELVIS_DEVICE
#endif
void PrintPatchGroup(PX_PatchGroup *patchGroup, PX_REAL *backgroundCoordBase, PX_REAL *knownPointBase){
  int i;

  GEN_PRINTF("patchGroup length = %d, nPatch = %d\n",patchGroup->length, patchGroup->nPatch);
  GEN_PRINTF("patchGroup threeDId = %d\n",patchGroup->threeDId);

  for(i=0; i<DIM3D*BACK_NBF; i++){
    GEN_PRINTF("bgElem[%d] = %.8E, ",i,backgroundCoordBase[patchGroup->threeDId*BACK_NBF*DIM3D+i]);
  }
  GEN_PRINTF("\n");

  for(i=0; i<DIM3D; i++){
    GEN_PRINTF("known[%d] = %.8E, ",i,knownPointBase[patchGroup->threeDId*DIM3D+i]);
  }
  GEN_PRINTF("knownType = %d\n",patchGroup->knownPointFlag);
  
  int *patchList = GetPatchList(patchGroup);
  for(i=0; i<patchGroup->nPatch; i++){
    GEN_PRINTF("patchList[%d] = %d, ",i,patchList[i]);
  }
  GEN_PRINTF("\n");
}


#ifdef __CUDACC__
ELVIS_DEVICE
#endif
void PrintCutCellElVis(PX_CutCellElVis* cutCell, PX_REAL *backgroundCoordBase, PX_REAL *knownPointBase){
  PX_PatchGroup *patchGroup = GetFirstPatchGroup(cutCell);
  int i;

  GEN_PRINTF("cutCell length = %d, nPatchGroup = %d\n",cutCell->length, cutCell->nPatchGroup);

  for(i=0; i<cutCell->nPatchGroup; i++){
    PrintPatchGroup(patchGroup, backgroundCoordBase, knownPointBase);
    patchGroup = GetNextPatchGroup(patchGroup);
  }
  GEN_PRINTF("\n");
}



#ifdef __CUDACC__
ELVIS_DEVICE
#endif
//void PrintTest_NoDoubles(PX_CutCellElVis* cutCellBase){
void PrintCutCellData(PX_CutCellElVis* cutCellBase, PX_REAL *backgroundCoordBase, PX_REAL *knownPointBase){

      PX_CutCellElVis* cutCellDummy2 = (PX_CutCellElVis*)((char*)(&cutCellBase[0]) +0); 
      
      PrintCutCellElVis(cutCellDummy2, backgroundCoordBase, knownPointBase);

}

#endif //_PXSTRUCTDEFINITIONS_H
