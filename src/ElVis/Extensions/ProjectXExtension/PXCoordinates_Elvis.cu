/*

LICENSE NOTICE

This source code is a part of the Project X software library.  Project X
solves partial differential equations
in multiple dimensions using an adaptive, discontinuous Galerkin finite
element method, and has been
specifically designed to solve the compressible Euler and Navier-Stokes
equations.

Copyright © 2003-2007 Massachusetts Institute of Technology

 

This library is free software; you can redistribute it and/or modify it
under the terms of the GNU Lesser
General Public License as published by the Free Software Foundation;
either version 2.1 of the License,
or (at your option) any later version.

 

This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser
General Public License in lgpl.txt for more details.

 

You should have received a copy of the GNU Lesser General Public License
along with this library; if not, write
to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
Boston, MA 02111-1307 USA.

This software was developed with the assistance of Government
sponsorship related to Government Contract
Number F33615-03-D-3306.  The Government retains certain rights to this
software under that Contract.


For more information about Project X, please contact:


David L. Darmofal
Department of Aeronautics & Astronautics
Massachusetts Institute of Technology
77 Massachusetts Ave, Room 37-401
Cambridge, MA 02139

Phone: (617) 258-0743
FAX:   (617) 258-5143

E-MAIL: darmofal@mit.edu
URL:    http://raphael.mit.edu

*/



/*!
  \file   PXShape.c

  Contains hardcoded basis functions, without PX-specific definitions.

*/

#ifndef PXCOORDINATES_ELVIS_C 
#define PXCOORDINATES_ELVIS_C 

#include <stdio.h>
#include <Fundamentals/PX.h>

#define PX_REF_OUTSIDE_ELEMENT -18

/******************************************************************/
//  FUNCTION Definition: PXProject2RefElement_Edge
ELVIS_DEVICE int
PXProject2RefElement_Edge( PX_REAL * RESTRICT xref, int * RESTRICT pnFaceSearch, int * RESTRICT FaceSearch )
{
  PX_REAL x;
  int nFaceSearch = 0;

  x = xref[0];

  if      ( x < 0.0 ) {
    x = 0.0;
    FaceSearch[nFaceSearch] = 0;
    nFaceSearch++;
  }
  else if ( x > 1.0 ){
    x = 1.0;
    FaceSearch[nFaceSearch] = 1;
    nFaceSearch++;
  }

  xref[0] = x;
  (*pnFaceSearch) = nFaceSearch;

  return PX_NO_ERROR;
}

/******************************************************************/
//  FUNCTION Definition: PXProject2RefElement_Triangle
ELVIS_DEVICE int
PXProject2RefElement_Triangle( PX_REAL * RESTRICT xref, int * RESTRICT pnFaceSearch, int * RESTRICT FaceSearch )
{
  PX_REAL x, y;
  PX_REAL alpha;
  int nFaceSearch = 0;

  x = xref[0];
  y = xref[1];

  /* project to faces 1 & 2 */
  if  ( x < 0.0 ) { FaceSearch[nFaceSearch] = 1; nFaceSearch++; x = 0.0; }
  if  ( y < 0.0 ) { FaceSearch[nFaceSearch] = 2; nFaceSearch++; y = 0.0; }

  /* project to face 0 */
  if ( x + y > 1 ) {
    FaceSearch[0] = 0;
    nFaceSearch   = 1;

    alpha = (1-x-y)/2.0;
    x = x+alpha;
    y = y+alpha;

    /* Project to corners */
    if ( x < 0 ) {
      FaceSearch[nFaceSearch] = 1;
      nFaceSearch++;
      x = 0.0;
      y = 1.0;
    }
    else if ( y < 0 ) {
      FaceSearch[nFaceSearch] = 2;
      nFaceSearch++;
      x = 1.0;
      y = 0.0;
    }
  }

  xref[0] = x;
  xref[1] = y;
  (*pnFaceSearch) = nFaceSearch;

  return PX_NO_ERROR;
}


/******************************************************************/
//  FUNCTION Definition: PXProject2RefElement_Quad
ELVIS_DEVICE int
PXProject2RefElement_Quad( PX_REAL * RESTRICT xref, int * RESTRICT pnFaceSearch, int * RESTRICT FaceSearch )
{
  PX_REAL x,y;
  int nFaceSearch = 0;

  x = xref[0];
  y = xref[1];

  if      ( x < 0.0 ) { FaceSearch[nFaceSearch] = 0; nFaceSearch++; x = 0.0; }
  else if ( x > 1.0 ) { FaceSearch[nFaceSearch] = 1; nFaceSearch++; x = 1.0; }

  if      ( y < 0.0 ) { FaceSearch[nFaceSearch] = 2; nFaceSearch++; y = 0.0; }
  else if ( y > 1.0 ) { FaceSearch[nFaceSearch] = 3; nFaceSearch++; y = 1.0; }

  xref[0] = x;
  xref[1] = y;
  (*pnFaceSearch) = nFaceSearch;

  return PX_NO_ERROR;
}

/******************************************************************/
//  FUNCTION Definition: PXProject2RefElement_Tet
ELVIS_DEVICE int
PXProject2RefElement_Tet( PX_REAL * RESTRICT xref, int * RESTRICT pnFaceSearch, int * RESTRICT FaceSearch )
{
  PX_REAL x, y, z;
  PX_REAL alpha;
  int nFaceSearch = 0;

  x = xref[0];
  y = xref[1];
  z = xref[2];

  /* project to faces 1-3 */
  if ( x < 0.0 ) { FaceSearch[nFaceSearch] = 1; nFaceSearch++; x = 0.0; }
  if ( y < 0.0 ) { FaceSearch[nFaceSearch] = 2; nFaceSearch++; y = 0.0; }
  if ( z < 0.0 ) { FaceSearch[nFaceSearch] = 3; nFaceSearch++; z = 0.0; }

  /* project to face 0 */
  if ( x + y + z > 1.0 ) {
    FaceSearch[0] = 0;
    nFaceSearch   = 1;

    alpha = (1-x-y-z)/3.0;
    x = x+alpha;
    y = y+alpha;
    z = z+alpha;

    /* project to edges of face 1 */
    if ( x < 0 ) {
      FaceSearch[nFaceSearch] = 1;
      nFaceSearch++;
      alpha = (1-y-z)/2.0;
      x = 0.0;
      y = MIN(y+alpha,1.0);
      z = MIN(z+alpha,1.0);
    }
    if ( y < 0 ) {
      FaceSearch[nFaceSearch] = 2;
      nFaceSearch++;
      alpha = (1-x-z)/2.0;
      x = MIN(x+alpha,1.0);
      y = 0.0;
      z = MIN(z+alpha,1.0);
    }
    if ( z < 0 ) {
      FaceSearch[nFaceSearch] = 3;
      nFaceSearch++;
      alpha = (1-x-y)/2.0;
      x = MIN(x+alpha,1.0);
      y = MIN(y+alpha,1.0);
      z = 0.0;
    }
  }

  xref[0] = x;
  xref[1] = y;
  xref[2] = z;
  (*pnFaceSearch) = nFaceSearch;

  return PX_NO_ERROR;
}


/******************************************************************/
//  FUNCTION Definition: PXProject2RefElement_Quad
ELVIS_DEVICE int
PXProject2RefElement_Hex( PX_REAL * RESTRICT xref, int * RESTRICT pnFaceSearch, int * RESTRICT FaceSearch )
{
  PX_REAL x, y, z;
  int nFaceSearch = 0;

  x = xref[0];
  y = xref[1];
  z = xref[2];

  if      ( x < 0.0 ) { FaceSearch[nFaceSearch] = 0; nFaceSearch++; x = 0.0; }
  else if ( x > 1.0 ) { FaceSearch[nFaceSearch] = 1; nFaceSearch++; x = 1.0; }

  if      ( y < 0.0 ) { FaceSearch[nFaceSearch] = 2; nFaceSearch++; y = 0.0; }
  else if ( y > 1.0 ) { FaceSearch[nFaceSearch] = 3; nFaceSearch++; y = 1.0; }

  if      ( z < 0.0 ) { FaceSearch[nFaceSearch] = 4; nFaceSearch++; z = 0.0; }
  else if ( z > 1.0 ) { FaceSearch[nFaceSearch] = 5; nFaceSearch++; z = 1.0; }

  xref[0] = x;
  xref[1] = y;
  xref[2] = z;
  (*pnFaceSearch) = nFaceSearch;

  return PX_NO_ERROR;
}

#define SEARCH_EPS 1e-1

/******************************************************************/
//  FUNCTION Definition: PXRefElement2SearchParameters_Hex
ELVIS_DEVICE int
PXRefElement2SearchParameters_Hex( PX_REAL const * RESTRICT xref, int * RESTRICT pnFaceSearch, int * RESTRICT FaceSearch)
{
  PX_REAL x, y, z;
  int nFaceSearch = 0;

  x = xref[0];
  y = xref[1];
  z = xref[2];

  if      ( x < 0.0 + SEARCH_EPS ) { FaceSearch[nFaceSearch] = 0; nFaceSearch++; }
  else if ( x > 1.0 - SEARCH_EPS ) { FaceSearch[nFaceSearch] = 1; nFaceSearch++; }

  if      ( y < 0.0 + SEARCH_EPS ) { FaceSearch[nFaceSearch] = 2; nFaceSearch++; }
  else if ( y > 1.0 - SEARCH_EPS ) { FaceSearch[nFaceSearch] = 3; nFaceSearch++; }

  if      ( z < 0.0 + SEARCH_EPS ) { FaceSearch[nFaceSearch] = 4; nFaceSearch++; }
  else if ( z > 1.0 - SEARCH_EPS ) { FaceSearch[nFaceSearch] = 5; nFaceSearch++; }

  (*pnFaceSearch) = nFaceSearch;

  return PX_NO_ERROR;
}


/******************************************************************/
//  FUNCTION Definition: PXProject2RefElement
ELVIS_DEVICE int
PXProject2RefElement( enum PXE_Shape Shape, PX_REAL * RESTRICT xref, int * RESTRICT nFaceSearch, int * RESTRICT FaceSearch )
{
  switch( Shape ) {
  case PXE_Shape_Edge:
    PXErrorReturn( PXProject2RefElement_Edge( xref, nFaceSearch, FaceSearch) );
    break;
  case PXE_Shape_Triangle:
    PXErrorReturn( PXProject2RefElement_Triangle( xref, nFaceSearch, FaceSearch ) );
    break;
  case PXE_Shape_Quad:
    PXErrorReturn( PXProject2RefElement_Quad( xref, nFaceSearch, FaceSearch ) );
    break;
  case PXE_Shape_Tet:
    PXErrorReturn( PXProject2RefElement_Tet( xref, nFaceSearch, FaceSearch ) );
    break;
  case PXE_Shape_Hex:
    PXErrorReturn( PXProject2RefElement_Hex( xref, nFaceSearch, FaceSearch ) );
    break;
  default:
    return PXError(PX_CODE_FLOW_ERROR);
  }

  return PX_NO_ERROR;
}

/******************************************************************/
//  FUNCTION Definition: PXProject2RefElement
ELVIS_DEVICE int
PXProject2RefElement( enum PXE_Shape Shape, PX_REAL * RESTRICT xref )
{
  int nFaceSearch = 0;
  int FaceSearch[6] = {0, 0, 0, 0, 0, 0};

  PXProject2RefElement(Shape, xref, &nFaceSearch, FaceSearch);

  if ( nFaceSearch > 0 )
    return PX_REF_OUTSIDE_ELEMENT;
  else
    return PX_NO_ERROR;
}


/******************************************************************/
//   FUNCTION Definition: PXMatrixDetInverse2
template <typename DT> ELVIS_DEVICE void
PXMatrixDetInverse2(DT const * RESTRICT jac, DT * RESTRICT J, DT * RESTRICT ijac)
{
  DT JJ;

  /* Compute Determinant */
  JJ = jac[0]*jac[3] - jac[2]*jac[1];

  /* Set Determinant */
  if (J != NULL)
    *J = JJ;

  /* Set inverse */
  if (ijac != NULL){
    ijac[0] =  jac[3]/JJ;
    ijac[1] = -jac[1]/JJ;
    ijac[2] = -jac[2]/JJ;
    ijac[3] =  jac[0]/JJ;
  }
}


/******************************************************************/
//   FUNCTION Definition: PXMatrixDetInverse3
template <typename DT> ELVIS_DEVICE void
PXMatrixDetInverse3(DT const * RESTRICT jac, DT * RESTRICT J, DT * RESTRICT ijac)
{

  DT JJ, JJ1;

  JJ = jac[0]*jac[4]*jac[8]
      +jac[1]*jac[5]*jac[6]
      +jac[2]*jac[3]*jac[7]
      -jac[6]*jac[4]*jac[2]
      -jac[7]*jac[5]*jac[0]
      -jac[8]*jac[3]*jac[1];

  if (J != NULL)
    *J = JJ;

  if (ijac != NULL){
    JJ1 = 1.0/JJ;
    ijac[0] = (jac[4]*jac[8]-jac[5]*jac[7])*JJ1;
    ijac[1] = (jac[7]*jac[2]-jac[8]*jac[1])*JJ1;
    ijac[2] = (jac[1]*jac[5]-jac[2]*jac[4])*JJ1;
    ijac[3] = (jac[5]*jac[6]-jac[3]*jac[8])*JJ1;
    ijac[4] = (jac[8]*jac[0]-jac[6]*jac[2])*JJ1;
    ijac[5] = (jac[2]*jac[3]-jac[0]*jac[5])*JJ1;
    ijac[6] = (jac[3]*jac[7]-jac[4]*jac[6])*JJ1;
    ijac[7] = (jac[6]*jac[1]-jac[7]*jac[0])*JJ1;
    ijac[8] = (jac[0]*jac[4]-jac[1]*jac[3])*JJ1;
  }
}


/******************************************************************/
//   FUNCTION Definition: PXRef2GlobFromCoordinates
ELVIS_DEVICE int
PXRef2GlobFromCoordinatesGivenShape2(int nbf, int globDim, PX_REAL const * nodeCoordinates,
				     PX_REAL * xglobal, PX_REAL const * phi)
{
  int node;
  PX_REAL temp1;
  PX_REAL temp2;
  PX_REAL temp3;  

  switch(globDim){
  case 1:
    temp1 = 0.0;
    for (node=0; node<nbf; node++){
      temp1 += nodeCoordinates[node] * phi[node];
    }
    xglobal[0] = temp1;
    return PX_NO_ERROR;
  case 2:
    temp1 = 0.0;
    temp2 = 0.0;
    for (node=0; node<nbf; node++){
      temp1 += nodeCoordinates[0] * phi[node];
      temp2 += nodeCoordinates[1] * phi[node];
      nodeCoordinates += 2;
    }
    xglobal[0] = temp1;
    xglobal[1] = temp2;
    return PX_NO_ERROR;
  case 3:
    temp1 = 0.0;
    temp2 = 0.0;
    temp3 = 0.0;
    for (node=0; node<nbf; node++){
      temp1 += nodeCoordinates[0] * phi[node];
      temp2 += nodeCoordinates[1] * phi[node];
      temp3 += nodeCoordinates[2] * phi[node];
      nodeCoordinates += 3;
    }
    xglobal[0] = temp1;
    xglobal[1] = temp2;
    xglobal[2] = temp3;
    return PX_NO_ERROR;
  default:
    return PX_CODE_FLOW_ERROR;
  }  
  
} /* end PXRef2GlobFromCoordinatesGivenShape */


/******************************************************************/
//   FUNCTION Definition: PXJacobianElementFromCoordinatesGivenGradient2
ELVIS_DEVICE int
PXJacobianElementFromCoordinatesGivenGradient2(enum PXE_ElementType const& type, int nbf, PX_REAL const *nodeCoordinates,
                                               PX_REAL const *xref,
                                               PX_REAL *pJ, PX_REAL *ijac,
                                               PX_REAL const *gphi, enum PXE_Boolean const& CoordinateVerbosity)
{
  int i,j,k;
  PX_REAL J=0.0;
  PX_REAL temp1 = 0.0;
  PX_REAL temp2 = 0.0;
  PX_REAL temp3 = 0.0;
  PX_REAL temp4 = 0.0;

  PX_REAL const *gphix;
  PX_REAL const *gphiy;
  PX_REAL const *gphiz;

  PX_REAL jacT[9] = {0,0,0, 0,0,0, 0,0,0};

  
  /* Get Jacobian Determinant and Inverse */
  switch (Dim) {
  case 1:

    // Evaluate Jacobian
    temp1 = 0.0;
    for( k = 0; k < nbf; k++)
      temp1 += nodeCoordinates[k]*gphi[k];

    J = temp1;
    jacT[0] = temp1;
    if ( fabs(J) > DBL_MIN ){
      ijac[0] = 1.0/jacT[0];
    }
    else {
      ijac[0] = 0.0;
      J = 0.0;
      jacT[0] = 0.0;
    }
    break;
  case 2:
    /* Use an unrolled version of the following code: */
    /* Idea is to compute a matrix-matrix product C=A*B, 
       C is 2x2, A is 2xnbf (col maj), B is nbfx2 (col maj) */

    /* initialize jacT */
/*     for(i = 0; i < 4; i++) */
/*       jacT[i] = 0.0; */
    
    /* Evaluate Jacobian */
/*     for( j = 0; j < 2; j++) */
/*       for( k = 0; k < nbf; k++) */
/* 	for( i = 0; i < 2; i++) */
/* 	  jacT[2*i+j] += ps[k][i]*gphi[j*nbf+k]; */

    // unrolled:
    gphix = gphi;
    gphiy = gphi + nbf;
    temp1 = 0.0;
    temp2 = 0.0;
    temp3 = 0.0;
    temp4 = 0.0;
    for( k = 0; k < nbf; k++){
      temp1 += nodeCoordinates[0]*gphix[k];
      temp2 += nodeCoordinates[0]*gphiy[k];

      temp3 += nodeCoordinates[1]*gphix[k];
      temp4 += nodeCoordinates[1]*gphiy[k];
      nodeCoordinates += 2;
    }
    jacT[0] = temp1;
    jacT[1] = temp2;
    jacT[2] = temp3;
    jacT[3] = temp4;
 
    PXMatrixDetInverse2<PX_REAL>(jacT, &J, ijac);
    break;
  case 3:
    // partially unrolled code, see case2 for details

    gphix = gphi;
    gphiy = gphi + nbf;
    gphiz = gphi + 2*nbf;

    // initialize jacT
    //for(i = 0; i < 9; i++)
    //  jacT[i] = 0.0;

    // Evaluate Jacobian
    for( k = 0; k < nbf; k++){
      for( i = 0; i < 3; i++){
        jacT[3*i+0] += nodeCoordinates[i]*gphix[k];
        jacT[3*i+1] += nodeCoordinates[i]*gphiy[k];
        jacT[3*i+2] += nodeCoordinates[i]*gphiz[k];
      }
      nodeCoordinates += 3;
    }

    PXMatrixDetInverse3<PX_REAL>(jacT, &J, ijac);
    break;
  default:
    //ELVIS_PRINTF("Dim = %d not supported in ProjectX\n", Dim);
    return PXErrorDebug(PX_CODE_FLOW_ERROR);
  }

  /* set pointer */
  if (pJ!=NULL)
    (*pJ) = J;

  if ( unlikely(J<=0.0) ){

//    if (CoordinateVerbosity==PXE_True){

      //ALWAYS_PRINTF("Negative Jacobian found: %e\n",J);

      nodeCoordinates -= Dim*nbf; //reset to beginning of coordinates

      /* printf("ERROR: jacobian = %22.15e < 0 \n",J); */
      //ALWAYS_PRINTF("       type = %s\n",PXE_ElementTypeName[type]);
      //ALWAYS_PRINTF("xref = [");
       for (j=0; j<Dim; j++){
         //ALWAYS_PRINTF(" %22.15e",xref[j]);
       }
       //ALWAYS_PRINTF("];\n");
//       ALWAYS_PRINTF("ps = [\n");
//       for (i=0; i<nbf; i++){
//       	for (j=0; j<Dim; j++){
//       	 ALWAYS_PRINTF(" %22.15e",nodeCoordinates[i*Dim+j]);
//       	}
//       	ALWAYS_PRINTF("\n");
//       }
//       ALWAYS_PRINTF("  ];\n");
       //printf("plot(ps(:,1),ps(:,2),'g^');\n");


    return PX_NON_PHYSICAL;
  }



  return PX_NO_ERROR;
}

/******************************************************************/
//   FUNCTION Definition: PXFaceRef2ElemFaceRefEdge
template <typename T> ELVIS_DEVICE int
PXFaceRef2ElemFaceRefEdge( int orientation, PX_REAL const * RESTRICT xface,
             PX_REAL * RESTRICT xfacelocal )
{
  xfacelocal[0] = xface[0];

  if (orientation != 0){
    xfacelocal[0] = 1.0-xfacelocal[0];
  }

  return PX_NO_ERROR;
}

/******************************************************************/
//   FUNCTION Definition: PXFaceRef2ElemFaceRefTriangle
template <typename T> ELVIS_DEVICE int 
PXFaceRef2ElemFaceRefTriangle( int orientation, T const * RESTRICT xface,
				   T * RESTRICT xfacelocal )
{
  int flip;  // flip
  int rot;   // rotation
  T temp;

  /* Get groupNum and minInd */
  flip = orientation%2;
  rot  = orientation/2; // c will naturally round down

  /* Set xfacelocal */
  xfacelocal[0] = xface[0];
  xfacelocal[1] = xface[1];
  
  /* Make sure face normal matches element's outward normal */
  if ( flip == 1 ) {
    temp          = xfacelocal[0];
    xfacelocal[0] = xfacelocal[1];
    xfacelocal[1] = temp;
  }

  switch ( rot ) {
  case 0:
    /* do nothing */
    break;
  case 1:
    temp          = xfacelocal[0];
    xfacelocal[0] = 1.0-temp-xfacelocal[1];
    xfacelocal[1] = temp;
    break;
  case 2:
    temp          = xfacelocal[1];
    xfacelocal[1] = 1.0-xfacelocal[0]-temp;
    xfacelocal[0] = temp;
    break;
  default:
    //ALWAYS_PRINTF("PXFaceRef2ElemFaceRefTriangle: Invalid rotation: %d\n", rot );
    return PXErrorDebug(PX_BAD_INPUT);
  }

  return PX_NO_ERROR;
}


/******************************************************************/
//   FUNCTION Definition: PXFaceRef2ElemFaceRefQuad
template <typename T> ELVIS_DEVICE int 
PXFaceRef2ElemFaceRefQuad( int orientation, T const * RESTRICT xface,
			       T * RESTRICT xfacelocal )
{
  int rot;    // rotation to get node0 on face ref to match node0 on element's face ref
  int flip;   // flip required if face normal is opposite element outward normal
  T temp;
  
  /* In order to be consistent with quads we should have
     orientation 0 ==>  outward normal starting starting at local node 0
     orientation 1 ==>  inward normal starting at local node 0
     orientation 2 ==>  outward normal starting starting at local node 1
     orientation 3 ==>  inward normal starting at local node 1
     orientation 4 ==>  outward normal starting starting at local node 2
     orientation 5 ==>  inward normal starting at local node 2
     orientation 6 ==>  outward normal starting starting at local node 3
     orientation 7 ==>  inward normal starting at local node 3
  */

  /* Get groupNum and minInd */
  flip = orientation%2;
  rot  = orientation/2; // c will naturally round down

  /* Set xfacelocal to xface */
  xfacelocal[0] = xface[0];
  xfacelocal[1] = xface[1];

  /* Make sure face normal matches element's outward normal */
  if ( flip == 1 ) {
    temp          = xfacelocal[0];
    xfacelocal[0] = xfacelocal[1];
    xfacelocal[1] = temp;
  }

  /* Rotate so that first node on faceref matches element face ref */
  switch (rot) {
  case 0:
    /* Do Nothing */
    break;
  case 1:
    temp          = xfacelocal[0];
    xfacelocal[0] = 1.0-xfacelocal[1];
    xfacelocal[1] = temp;
    break;
  case 2:
    xfacelocal[0] = 1.0-xfacelocal[0];
    xfacelocal[1] = 1.0-xfacelocal[1];
    break;
  case 3:
    temp          = xfacelocal[0];
    xfacelocal[0] = xfacelocal[1];
    xfacelocal[1] = 1.0-temp;
    break;
  default:
    //ALWAYS_PRINTF("PXFaceRef2ElemFaceRefQuad: Invalid orientation: %d\n",orientation);
    return PXErrorDebug(PX_BAD_INPUT);
  }

  return PX_NO_ERROR;
}


/******************************************************************/
//   FUNCTION Definition: PXFaceRef2ElemFaceRef
template <typename T> ELVIS_DEVICE int 
PXFaceRef2ElemFaceRef( enum PXE_Shape faceShape, int orientation, 
			   T const * RESTRICT xface, T * RESTRICT xfacelocal )
{
  switch (faceShape) {
  
   case PXE_Shape_Node:
     xfacelocal[0] = xface[0];
     break;

   case PXE_Shape_Edge:
     ( PXFaceRef2ElemFaceRefEdge<T>( orientation, xface, xfacelocal ) );
     break;

  case PXE_Shape_Triangle:
    ( PXFaceRef2ElemFaceRefTriangle<T>( orientation, xface, xfacelocal ) );
    break;

  case PXE_Shape_Quad:
    ( PXFaceRef2ElemFaceRefQuad<T>( orientation, xface, xfacelocal ) );
    break;

  default:
    //ALWAYS_PRINTF("PXFaceRef2ElemFaceRef: Unkown faceShape = %d", faceShape);
    return PXErrorDebug(PX_CODE_FLOW_ERROR);
  }

  return PX_NO_ERROR;
}


/******************************************************************/
//   FUNCTION Definition PXFace2Ref3dSimplex
ELVIS_DEVICE int 
PXFace2RefTetOrientation0(int lface, PX_REAL const * RESTRICT xfacelocal, PX_REAL * RESTRICT xref)
{
/*
 PURPOSE:
    Converts reference coordinates on a face to reference
    coordinates on an element in 3d. Accounts for face orientation;
    find "orientation" via PXFaceOrientation
 INPUTS:
   lface:    local face number of the face in question
   orientation: orientation of the local face
   xface:    coordinates in face reference space   

 OUTPUTS:
   xref:     coordinates in element reference space

 RETURNS:
   Error Code
*/
  //PX_REAL temp;

  /* xface held face reference coordinates on the reference triangle
     {(0,0),(1,0),(0,1)} in some orientation (so the basis vectors
     may not have been <1,0> and <0,1>

     Now xfacelocal holds the coordinates in the orientation 0 basis
     (i.e., <1,0> and <0,1>) of the same spatial point referred to
     by xface. */

  /* Now calculate the element reference coordinates (note that the
     tetrahedron only has ONE orientation).
     The following code ASSUMES that xfacelocal is in the
     orientation 0 basis. Good thing we made the transformation. */
  switch (lface){
  case 0:
    xref[0] = 1.0 - xfacelocal[0] - xfacelocal[1];
    xref[1] =       xfacelocal[0];
    xref[2] =                       xfacelocal[1];
    return PX_NO_ERROR;
  case 1:
    xref[0] = 0.0;
    xref[1] =                       xfacelocal[1];
    xref[2] =       xfacelocal[0];
    return PX_NO_ERROR;
  case 2:
    xref[0] =       xfacelocal[0];
    xref[1] = 0.0;
    xref[2] =                       xfacelocal[1];
    return PX_NO_ERROR;
  case 3:
    xref[0] =                       xfacelocal[1];
    xref[1] =       xfacelocal[0];
    xref[2] = 0.0;
    return PX_NO_ERROR;
  default:
    //ALWAYS_PRINTF("PXFace2RefTetOrientation0: Invalid lface choice: %d\n",lface);
    return PX_BAD_INPUT;
  }
}


/******************************************************************/
//   FUNCTION Definition PXFace2RefTet
ELVIS_DEVICE int  
PXFace2RefTet(int lface, int orientation, PX_REAL const * RESTRICT xface, PX_REAL * RESTRICT xref)
{
  PX_REAL xfacelocal[2];

  ( PXFaceRef2ElemFaceRefTriangle( orientation, xface, xfacelocal ) );
  ( PXFace2RefTetOrientation0( lface, xfacelocal, xref) );

  return PX_NO_ERROR;
}


/******************************************************************/
//   FUNCTION Definition PXFace2Ref
ELVIS_DEVICE 
int PXFace2RefReference( enum PXE_Shape shape, int lface, int orientation, 
			 PX_REAL const * RESTRICT xface, PX_REAL * RESTRICT xref) 
{
  switch ( shape ) {
  case PXE_Shape_Edge:
    //( PXFace2RefEdge(lface, orientation, xface, xref) );
    break;
    
  case PXE_Shape_Triangle:
    //( PXFace2RefTriangle(lface, orientation, xface, xref) );
    break;

  case PXE_Shape_Quad:
    //( PXFace2RefQuad(lface, orientation, xface, xref) );
    break;

  case PXE_Shape_Tet:
    ( PXFace2RefTet(lface, orientation, xface, xref) );
    break;

  case PXE_Shape_Hex:
    //( PXFace2RefHex(lface, orientation, xface, xref) );
    break;
    
  default:
    return PXErrorDebug(PX_CODE_FLOW_ERROR);
  }

  return PX_NO_ERROR;
}



/******************************************************************/
//   FUNCTION Definition: PXFace2Ref
/* ELVIS_DEVICE int  */
/* PXFace2Ref( PX_Grid const * RESTRICT pg, int egrp, int elem, int lface, PX_REAL * RESTRICT xface, */
/* 		PX_REAL * RESTRICT xref) */
/* { */
/*   enum PXE_BasisShape basisshape; */
  
/*   PXErrorReturn( PXType2Shape(type, &shape) ); */
  
/*   switch (refDim){ */
/*   case 1: */
/*     switch (lface) { */
/*     case 0: xref[0] = 0.0; return PX_NO_ERROR; */
/*     case 1: xref[0] = 1.0; return PX_NO_ERROR; */
/*     default: */
/*       rtPrintf("Invalid lface number = %d\n", lface); */
/*       return PXError(PX_BAD_INPUT); */
/*     } */
/*   case 2: */
/*   case 3: */
/*     //PXErrorReturn( PXFaceOrientation(pg, egrp, elem, lface, &orientation) ); */
/*     PXErrorReturn( PXFace2RefReference( shape, lface, 0, xface, xref) ); */
/*     return PX_NO_ERROR; */
/*   default: */
/*     printf("Dim = %d not supported in ProjectX\n", Dim); */
/*     return PXError(PX_CODE_FLOW_ERROR); */
/*   } */

/* } */

/* /\******************************************************************\/ */
/* //   FUNCTION Definition: PXQuadFace */
/* ELVIS_DEVICE int */
/* PXQuadFace(enum PXE_ElementType type, int fgrp, int face, int quad_order, int *pnquad, enum PXE_QuadratureRule *pquad_rule,  */
/* 	   PX_REAL **pxface, PX_REAL **pxrefL, PX_REAL **pxrefR, PX_REAL **pwquad, PX_REAL **pnvec) */
/* { */
/*   int ierr;                     // error code */
/*   int d, Dim;                   // Dimension */
/*   int egrpL, egrpR = -1;        // element group number */
/*   int elemL, elemR = -1;        // element number */
/*   int lfaceL, lfaceR = -1;      // local face number */
/*   enum PXE_ElementType typeL, typeR;        // element type */
/*   int QorderL, QorderR;         // element numerical order (Q) */
/*   int nquad;                    // number of quad points */
/*   int iquad;                    // index over quad points */
/*   int CutFace_Flag = PXE_False; // flag indicating boundary face is cut */
/*   PX_REAL *xrefL = NULL;        // local (*pxrefL) array */
/*   PX_REAL *xrefR = NULL;        // local (*pxrefR) array */
/*   PX_REAL *nvec  = NULL;        // local (*pnvec ) array */
/*   PX_REAL *xquad = NULL;        // quad points in face reference coordinates */
/*   PX_REAL *wquad = NULL;        // quad weights */


/*   /\* Set pointers *\/ */
/*   if (pxface != NULL) { */
/*     xquad = (*pxface); */
/*   } */
/*   if (pxrefL != NULL) { */
/*     xrefL = (*pxrefL); */
/*   } */
/*   if (pxrefR != NULL) { */
/*     xrefR = (*pxrefR); */
/*   } */
/*   if (pwquad != NULL) { */
/*     wquad  = (*pwquad ); */
/*   } */
/*   if (pnvec != NULL) { */
/*     nvec  = (*pnvec ); */
/*   } */


/*   /\* /\\* Get Element Qorder *\\/ *\/ */
/*   /\* PXErrorReturn( PXType2qorder(typeL, &QorderL) ); *\/ */
/*   /\* PXErrorReturn( PXType2qorder(typeR, &QorderR) ); *\/ */

/*   /\****************************************************************************************\/ */
/*   /\*                                          Not Cut                                     *\/ */
/*   /\****************************************************************************************\/ */

/*     enum PXE_Shape shape, faceShape; */
/*     /\* get element group type *\/ */
/*     //typeL = pg->ElementGroup[egrpL].type; */
/*     //typeR = pg->ElementGroup[egrpR].type; */
/*     PXErrorReturn(PXType2Shape(type, &shape)); */
/*     PXErrorReturn(PXElemShape2FaceShape(shape, lfaceL, &faceShape)); */
    
/*     /\* Add Qorder to quad_order *\/ */
/*     //if (quad_order != PXE_MaxQuad) quad_order += MIN(QorderL, QorderR)-1; */
    
/*     /\* Get Quad points on face *\/ */
/*     PXErrorReturn( PXQuadReference(faceShape, quad_order, &nquad, pquad_rule, &xquad, pwquad) ); */
    
    
/*     /\* Allocate xref and nvec *\/ */
/*     if (pxrefL != NULL) { */
/*       //PXErrorReturn( PXReAllocate( Dim*nquad, sizeof(PX_REAL), (void **)&(xrefL) ) );  */
/*     } */
/*     if (pxrefR != NULL) { */
/*       //PXErrorReturn( PXReAllocate( Dim*nquad, sizeof(PX_REAL), (void **)&(xrefR) ) );  */
/*     } */
/*     if (pnvec != NULL) { */
/*       //PXErrorReturn( PXReAllocate( Dim*nquad, sizeof(PX_REAL), (void **)&(nvec) ) );  */
/*     } */
    
/*     /\* Convert xquad to element reference coordinates and compute nvec *\/ */
/*     for (iquad = 0; iquad < nquad; iquad++) { */
/*       if (pnvec != NULL) { */
/* 	/\* calculate outward pointing normal (also returns xglobal) *\/ */
/* 	//PXErrorReturn( PXFaceNormal(pg, fgrp, face, xquad + (Dim-1)*iquad, nvec+Dim*iquad, NULL) ); */
	
/*       } */
      
/*       /\* map Face2Elem: xface -> xref *\/ */
/*       if (pxrefL != NULL) { */
/* 	PXErrorReturn( PXFace2Ref(pg, egrpL, elemL, lfaceL, xquad+(Dim-1)*iquad, xrefL+Dim*iquad) ); */
/*       }     */
/*       if (pxrefR != NULL) { */
/* 	PXErrorReturn( PXFace2Ref(pg, egrpR, elemR, lfaceR, xquad+(Dim-1)*iquad, xrefR+Dim*iquad) ); */
/*       } */
/*     }//iquad */

  
/*   /\* Set pointers *\/ */
/*   (*pnquad) = nquad; */
  
/*   if (pxface != NULL) { */
/*     (*pxface) = xquad; */
/*   } */
/*   if (pxrefL != NULL) { */
/*     (*pxrefL) = xrefL; */
/*   } */
/*   if (pxrefR != NULL) { */
/*     (*pxrefR) = xrefR; */
/*   } */
/*   if (pnvec != NULL) { */
/*     (*pnvec ) = nvec ; */
/*   } */

    
/*   /\* Release quad point in face reference space if not needed *\/ */
/*   if (pxface == NULL) PXRelease( xquad ); */

 
/*   return PX_NO_ERROR; */
/* } */

/******************************************************************/
//   FUNCTION Definition PXElementCentroidReference
ELVIS_DEVICE int 
PXElementCentroidReference( enum PXE_Shape Shape, PX_REAL * RESTRICT xref )
{
  switch ( Shape ) {
  case PXE_Shape_Edge:
    xref[0] = 0.5;
    break;

  case PXE_Shape_Triangle:
    xref[0] = 0.333333333333333333;
    xref[1] = 0.333333333333333333;
    break;

  case PXE_Shape_Quad:
    xref[0] = 0.5;
    xref[1] = 0.5;
    break;

  case PXE_Shape_Tet:
    xref[0] = 0.25;
    xref[1] = 0.25;
    xref[2] = 0.25;
    break;
    
  case PXE_Shape_Hex:
    xref[0] = 0.5;
    xref[1] = 0.5;
    xref[2] = 0.5;
    break;
  
  default:
    return PXErrorDebug(PX_CODE_FLOW_ERROR);
  }

  return PX_NO_ERROR;
}

/******************************************************************/
//   FUNCTION Definition: LinearSimplexGlob2Ref
ELVIS_DEVICE int
LinearSimplexGlob2Ref(enum PXE_Shape Shape, ElVisFloat const * RESTRICT vertices, PX_REAL const * RESTRICT xglobal, PX_REAL * RESTRICT xref)
{
  PX_REAL Jac[9];    // Transformation Jacobian
  //PX_REAL J;         // Jacobian Determinant
  PX_REAL iJac[9];   // Inverse of Jacobian
  PX_REAL xrefLin[3] = {0,0,0};
  ElVisFloat const * x0 = NULL;       // Coordinates on Node0

  /* Set x0 to node 0 */
  x0 = vertices;

  /* /\* Form Jacobian Matrix *\/ */
  /* for (i=0; i<Dim; i++) */
  /*   for (j=0; j<Dim; j++) */
  /*     Jac[i*Dim+j] = vertices[(1+j)*Dim+i]-x0[i]; */

  /* Get inverse of Jacobian Matrix */
  switch (Dim) {
  case 1:
    iJac[0] = 1.0/(vertices[1]-vertices[0]);
    xref[0] = iJac[0]*(xglobal[0] - vertices[0]);
    break;
  case 2:
    Jac[0] = vertices[2 + 0] - x0[0];
    Jac[1] = vertices[4 + 0] - x0[0];
    Jac[2] = vertices[2 + 1] - x0[1];
    Jac[3] = vertices[4 + 1] - x0[1];

    PXMatrixDetInverse2<PX_REAL>(Jac, NULL, iJac);

    /* Compute Reference Coordinates */
    /* matrix-vec product: iJac*(xglobal - x0) */
    //j=0;
    xref[0] = iJac[0*2 + 0]*(xglobal[0] - x0[0]) + iJac[0*2 + 1]*(xglobal[1] - x0[1]);
    xref[1] = iJac[1*2 + 0]*(xglobal[0] - x0[0]) + iJac[1*2 + 1]*(xglobal[1] - x0[1]);
    break;
  case 3:
    Jac[0] = vertices[3 + 0] - x0[0];
    Jac[1] = vertices[6 + 0] - x0[0];
    Jac[2] = vertices[9 + 0] - x0[0];
    Jac[3] = vertices[3 + 1] - x0[1];
    Jac[4] = vertices[6 + 1] - x0[1];
    Jac[5] = vertices[9 + 1] - x0[1];
    Jac[6] = vertices[3 + 2] - x0[2];
    Jac[7] = vertices[6 + 2] - x0[2];
    Jac[8] = vertices[9 + 2] - x0[2];

    PXMatrixDetInverse3<PX_REAL>(Jac, NULL, iJac);

    /* Compute Reference Coordinates */
    /* matrix-vec product: iJac*(xglobal - x0) */
    //j=0;
    xref[0] = iJac[0*3 + 0]*(xglobal[0] - x0[0]) +
              iJac[0*3 + 1]*(xglobal[1] - x0[1]) +
              iJac[0*3 + 2]*(xglobal[2] - x0[2]);
    xref[1] = iJac[1*3 + 0]*(xglobal[0] - x0[0]) +
              iJac[1*3 + 1]*(xglobal[1] - x0[1]) +
              iJac[1*3 + 2]*(xglobal[2] - x0[2]);
    xref[2] = iJac[2*3 + 0]*(xglobal[0] - x0[0]) +
              iJac[2*3 + 1]*(xglobal[1] - x0[1]) +
              iJac[2*3 + 2]*(xglobal[2] - x0[2]);
    break;
  default:
    //ALWAYS_PRINTF("Dim = %d not supported in ProjectX\n", Dim);
    return PXErrorDebug(PX_BAD_INPUT);
  }

  for (int d=0; d<Dim; d++)
    xrefLin[d] = xref[d];

  //Make sure the reference coordinates are inside the element
  PXErrorReturn( PXProject2RefElement( Shape, xref ) );

  //Compute a residual to see if xref moved back inside the element
  PX_REAL Residual = 0.0;
  for(int d=0; d<Dim; d++)
    Residual += (xrefLin[d]-xref[d])*(xrefLin[d]-xref[d]);
  Residual = sqrt(Residual);

//  ELVIS_PRINTF("PX LinearSimplexGlob2Ref: xrefLin (%2.15f,%2.15f,%2.15f)\n", xrefLin[0], xrefLin[1], xrefLin[2]);
//  ELVIS_PRINTF("PX LinearSimplexGlob2Ref: xref    (%2.15f,%2.15f,%2.15f)\n", xref[0], xref[1], xref[2]);
//  ELVIS_PRINTF("PX LinearSimplexGlob2Ref: Linear Element Residual %2.15f\n", Residual);

  if ( Residual > 1.0E-9)
    return PX_NOT_CONVERGED;

  return PX_NO_ERROR;
}

/******************************************************************/
//   FUNCTION Definition: PXGlob2RefFromCoordinates
ELVIS_DEVICE int
PXCurvedGlob2Ref(PX_ElementTypeData const& elemData, PX_REAL const *xnodes, PX_REAL const * RESTRICT xglobal, PX_REAL * RESTRICT xref, enum PXE_Boolean initialGuessProvided, enum PXE_Boolean CoordinateVerbosity)
{
  int ierr = PX_NO_ERROR;              // error code
  int d,d2;
  int node;              // index over the nodes of an element
  int qorder;            // polynomial order of the element
  int nbf;               // number of basis function in the element
  int iter;              // current iteration of the newton solve
  int nLimitedIter = 5;  // number of iterations we limit the update in the newton solve
  const int maxIter = 200;
  PX_REAL Residual;     // Residual of Newton Solve
  PX_REAL lim = 1.0;    // limit on the newton update - so the first step doesn't take us way out of the reference element
  //PX_REAL Jac[9];       // Transformation Jacobian
  PX_REAL iJac[9] = {0,0,0,0,0,0,0,0,0};      // Inverse of Jacobian
  PX_REAL RHS[3] = {0,0,0};       // right hand side vector for Newton Solve
  PX_REAL xrefold[3] = {0,0,0};     // Previous in xref for Newton Solve
  PX_REAL dxref[3] = {0,0,0};     // Update in xref for Newton Solve
  PX_REAL phi[MAX_NBF] = {0};  // Basis Functions
  PX_REAL gphi[DIM3D*MAX_NBF] = {0}; // Derivative of Basis Functions wrt reference coordinates
  PX_REAL xg[3] = {0,0,0};
  enum PXE_Boolean ConvergedFlag; // flag indicating if the newton solve has converged
  enum PXE_Shape Shape;           // shape of the basis in this element
  enum PXE_SolutionOrder order;   // interpolation order of a given element type

  // get qorder, shape
  qorder = elemData.qorder;
  Shape = elemData.shape;

  // Get type and order
  nbf = elemData.nbf;
  order = elemData.order;

  // Initialize Newton Iteration
  ConvergedFlag = PXE_False;
  Residual = 1.0;
  iter = 0;

  // Initialize xref to centroid
  PXErrorReturn( PXElementCentroidReference( Shape, xref ) );

  // Starting Newton iteration
  for ( iter = 0; iter < maxIter; iter++) {

    // Get Shape functions and gradients
    PXErrorReturn( PXGradientsElem(order, qorder, xref, gphi ) );


    // Jacobian element for current xref
    ierr = PXJacobianElementFromCoordinatesGivenGradient2(elemData.type, nbf, xnodes, xref, NULL, iJac, gphi, CoordinateVerbosity);

    if(ierr != PX_NO_ERROR)
      break;

    PXErrorReturn( PXShapeElem(order, qorder, xref, phi ) );

    // Initialize RHS and Jacobian
    for (d=0; d<Dim; d++)
      RHS[d] = xglobal[d];
    //RHS[0] = xglobal[0];
    //RHS[1] = xglobal[1];
    //RHS[2] = xglobal[2];

    // Evaluate RHS
    for (node=0; node<nbf; node++){
      // NOTE: we are actually accessing phi here!
      // The data is just stored in gphi array!!
      for (d=0; d<Dim; d++)
        RHS[d] -= xnodes[Dim*node+d]*phi[node];
      //RHS[0] -= xnodes[Dim*node+0]*phi[node];
      //RHS[1] -= xnodes[Dim*node+1]*phi[node];
      //RHS[2] -= xnodes[Dim*node+2]*phi[node];
    }

    // find the residual
    Residual = 0.0;
    for(d=0; d<Dim; d++)
      Residual += RHS[d]*RHS[d];
    //Residual += RHS[0]*RHS[0] + RHS[1]*RHS[1] + RHS[2]*RHS[2];
    Residual = sqrt(Residual);

    //ELVIS_PRINTF("PX PXCurvedGlob2Ref xref = %f, %f, %f | residual = %f\n", xref[0], xref[1], xref[2], Residual);

    // Check Residual Tolerance
    if ( ( Residual < 1.0E-10) && (iter>nLimitedIter) ) {
      ConvergedFlag = PXE_True;
      break;
    }

    // Compute State
    //( PXMat_Vec_Mult_Set( iJac, RHS, dxref, Dim, Dim ) );
//    dxref[0] = iJac[0*DIM3D+0]*RHS[0] + iJac[0*DIM3D+1]*RHS[1] + iJac[0*DIM3D+2]*RHS[2];
//    dxref[1] = iJac[1*DIM3D+0]*RHS[0] + iJac[1*DIM3D+1]*RHS[1] + iJac[1*DIM3D+2]*RHS[2];
//    dxref[2] = iJac[2*DIM3D+0]*RHS[0] + iJac[2*DIM3D+1]*RHS[1] + iJac[2*DIM3D+2]*RHS[2];

    for(d=0; d<Dim; d++)
      dxref[d] = 0;

    for(d=0; d<Dim; d++)
      for(d2=0; d2<Dim; d2++)
        dxref[d] += iJac[d*Dim+d2]*RHS[d2];

    // limiting - for the first iteration we don't want to take the full step
    if (iter<nLimitedIter){
      //Residual = 1.0;
      lim = 0.1;
    }
    else{
      lim = 1.0;
    }

    // Update State
    for (d=0; d<Dim; d++)
      xref[d] += lim*dxref[d];

    //Make sure the reference coordinates are inside the element
    ierr = PXProject2RefElement( Shape, xref );

    //Compute the difference between the old and projected xref
    Residual = 0.0;
    for(d=0; d<Dim; d++)
      Residual += (xrefold[d]-xref[d])*(xrefold[d]-xref[d]);
    Residual = sqrt(Residual);

    //Abort if the point is constantly hitting the same point on the edge of the element
    if ( ( Residual < 1.0E-10) && ierr == PX_REF_OUTSIDE_ELEMENT ) {
      break;
    }

    for (d=0; d<Dim; d++)
      xrefold[d] = xref[d];

  }// for iter

  //we may have broken out of while loop b/c ierr != PX_NO_ERROR
  //check what happened
//  if(ierr!=PX_NO_ERROR){
//    ConvergedFlag=PXE_False;
//  }
//  else{
//    PXErrorReturn(ierr);
//  }

  // Check Convergence
  if (ConvergedFlag == PXE_False) {
    //if (CoordinateVerbosity==PXE_True){

    // ref 2 glob for current xref
    PXShapeElem(order, qorder, xref, phi );
    PXRef2GlobFromCoordinatesGivenShape2(nbf, Dim, xnodes, xg, phi);

//      ELVIS_PRINTF("ERROR: Unable to Converge PXCurvedGlob2Ref: xref = %f, %f, %f\n", xref[0], xref[1], xref[2] );
//      ELVIS_PRINTF("ERROR: Residual = %e, ierr = %d, iter = %d\n", Residual, ierr, iter );
//      ELVIS_PRINTF("ERROR: xglobal = %f, %f, %f\n", xglobal[0], xglobal[1], xglobal[2] );
//      ELVIS_PRINTF("ERROR: xg@xref = %f, %f, %f\n", xg[0], xg[1], xg[2] );
    //}
    return PX_NOT_CONVERGED;
  }

  return PX_NO_ERROR;
}

/******************************************************************/
//   FUNCTION Definition: PXGlob2RefFromCoordinates
ELVIS_DEVICE int
PXGlob2RefFromCoordinates2(PX_ElementTypeData const& elemData, PX_REAL const *xnodes, PX_REAL const * RESTRICT xglobal, PX_REAL * RESTRICT xref, enum PXE_Boolean initialGuessProvided, enum PXE_Boolean CoordinateVerbosity)
{
  int qorder;            // polynomial order of the element
  enum PXE_Shape Shape;           // shape of the basis in this element
  
  /* get qorder, shape */
  qorder = elemData.qorder;
  Shape = elemData.shape;

  /*----------------------------------------------------------------------------------------*/
  /*                             Evaluate Based on linear Element                           */
  /*----------------------------------------------------------------------------------------*/
  if ((qorder ==  1) && ( (Shape == PXE_Shape_Edge) || (Shape == PXE_Shape_Triangle) || (Shape == PXE_Shape_Tet))) {
    //ELVIS_PRINTF("MCG: PXGlob2RefFromCoordinates2: Linear Element\n");
    return LinearSimplexGlob2Ref(Shape, xnodes, xglobal, xref);
  }

  //return LinearSimplexGlob2Ref(xnodes, xglobal, xref);
  //ELVIS_PRINTF("PXGlob2RefFromCoordinates2: Dim=%d, qorder=%d, Shape=%d\n", Dim, qorder, Shape);
  //return PX_NO_ERROR;
  /*----------------------------------------------------------------------------------------*/
  /*                         The Hard Way - A Higher Order Element                          */
  /*----------------------------------------------------------------------------------------*/

  return PXCurvedGlob2Ref(elemData, xnodes, xglobal, xref, initialGuessProvided, CoordinateVerbosity);
}

#endif //PXCOORDINATES_ELVIS_C 
