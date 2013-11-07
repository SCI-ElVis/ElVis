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

#ifndef PXNORMAL_ELVIS_C 
#define PXNORMAL_ELVIS_C 

/******************************************************************/
//   FUNCTION Definition: PXCrossProduct
ELVIS_DEVICE void 
PXCrossProduct(PX_REAL const *u, PX_REAL const *v, PX_REAL *w)
{
  w[0] = u[1]*v[2] - u[2]*v[1];
  w[1] = u[2]*v[0] - u[0]*v[2];
  w[2] = u[0]*v[1] - u[1]*v[0];
}


/******************************************************************/
//   FUNCTION Definition: PXFaceNormalGivenGradients
ELVIS_DEVICE int 
PXFaceNormalReferenceGivenGradients( int Dim, int nnode, PX_REAL const *gphi,
					 PX_REAL const *xnodes, PX_REAL const *xface, PX_REAL *nvec)
{
  /* 
     PURPOSE: 

     Computes the outward pointing normal for a face of a 2D or 3D element 

     INPUTS: 
     
     Dim:        Dimension of problem
     nnode:      number of nodal basis functions on face
     gphi:       gradient of nodal basis functions on face
     xnodes:     Nodal locations on face
     xface:      face reference coordinates
     
     OUTPUTS:
     
     nvec:       Normal

  */
  
  int d;          // index over dimension
  int t;          // index over tangent vectors
  int k;          // number of basis functions
  PX_REAL x_u[6] = {0., 0., 0., 0., 0., 0.}; // tangent vectors

  /* Compute tangent vectors */
  for ( t = 0; t < DIM3D-1; t++)
    for ( k = 0; k < nnode; k++)
      for ( d = 0; d < DIM3D; d++)
        x_u[t*DIM3D+d] += xnodes[k*DIM3D+d]*gphi[t*nnode+k];
  
  /* Compute normal */

    /* Compute normal as cross product */
  PXCrossProduct( x_u, x_u+DIM3D, nvec);

  return PX_NO_ERROR;
}


/******************************************************************/
//   FUNCTION Definition: PXOutwardNormal
ELVIS_DEVICE int
PXOutwardNormal( enum PXE_SolutionOrder orderQ, int qorder, int nbfQ, PX_FaceData const *faceData, PX_REAL const * xnodes,
		 PX_REAL const *xface, PX_REAL *nvec)
{
/*
  PURPOSE:
    Computes the outward-pointing normal for a canonical face.

  INPUTS:
    nbfQ: number of basis functions for geometry of this face
    faceData: faceData structure for the face you want normal on
    xnodes: list of coordinates for this face
    xface:   reference coordinates on the face. The face reference
             triangle is formed using globally-ordered vertices adjacent
             to the face.

  OUTPUTS:
    nvec: returned normal

  RETURNS:
    Error code
*/

  int d;                 // dimension

  PX_REAL xfacelocal[2];         // reference coordinates on element face
  PX_REAL gphi[DIM3D*MAX_NBF_FACE];

  
  /* correct for orientation */
  PXErrorDebug( PXFaceRef2ElemFaceRef<PX_REAL>( (enum PXE_Shape) faceData->shape, (int) faceData->orientation, xface, xfacelocal) );

  /* Get gradients */
  ( PXGradientsFace<PX_REAL>( orderQ, qorder, xfacelocal, gphi) );

  PXFaceNormalReferenceGivenGradients( DIM3D, nbfQ, gphi,
						     xnodes, xfacelocal, nvec);

  if(faceData->side == 1){
    for (d=0; d<DIM3D; d++)
      nvec[d] = -nvec[d];
  }

  return PX_NO_ERROR;
}


#endif //PXNORMAL_ELVIS_C 
