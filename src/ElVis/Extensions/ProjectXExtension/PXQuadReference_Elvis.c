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

#ifndef PXQUADREFERENCE_ELVIS_C 
#define PXQUADREFERENCE_ELVIS_C 

/******************************************************************/
//   FUNCTION Definition: PXQuadLine
ELVIS_DEVICE int 
PXQuadLine(int quad_order, int *pnquad, enum PXE_QuadratureRule *pquad_rule, double **pxquad, double **pwquad)
{
  int ierr;  // error code
  enum PXE_QuadratureRule quad_rule; // index for quadrature rules
  int iquad;
  int nquad;     // number of quad points

  double *xquad = NULL; // quad points
  double *wquad = NULL; // quad weights

  /* original quad_order to nquad mapping arrays */
/*   int QO[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 39, -1};  // order of quad rules */
/*   int QN[] = {1, 2, 3, 4, 5,  6,  7,  8,  9, 10, 11, 12, 20, -1};  // number of points for quad rules */

  /* Data is ordered in pairs of (nquad, quad_rule) */
  const int PXquad_orderTonquadLineLookupTable[2*24] = {
    1, 0, //quad_order == 0, quad_rule = 0
    1, 0, //quad_order == 1, quad_rule = 0
    2, 1, //quad_order == 2, quad_rule = 1
    2, 1, //quad_order == 3, quad_rule = 1
    3, 2, //quad_order == 4, quad_rule = 2
    3, 2, //quad_order == 5, quad_rule = 2
    4, 3, //quad_order == 6, quad_rule = 3
    4, 3, //quad_order == 7, quad_rule = 3
    5, 4, //quad_order == 8, quad_rule = 4
    5, 4, //quad_order == 9, quad_rule = 4
    6, 5, //quad_order == 10, quad_rule = 5
    6, 5, //quad_order == 11, quad_rule = 5
    7, 6, //quad_order == 12, quad_rule = 6
    7, 6, //quad_order == 13, quad_rule = 6
    8, 7, //quad_order == 14, quad_rule = 7
    8, 7, //quad_order == 15, quad_rule = 7
    9, 8, //quad_order == 16, quad_rule = 8
    9, 8, //quad_order == 17, quad_rule = 8
    10, 9, //quad_order == 18, quad_rule = 9
    10, 9, //quad_order == 19, quad_rule = 9
    11, 10, //quad_order == 20, quad_rule = 10
    11, 10, //quad_order == 21, quad_rule = 10
    12, 11, //quad_order == 22, quad_rule = 11
    12, 11};//quad_order == 23, quad_rule = 11

  /* quad_order = 39, nquad = 20 is quad_rule 12 */
  /* This last 'dense' quad_rule is not included in the lookup table 
     to decrease the table size */

  /*---------------------*/
  /* Determine quad rule */
  /*---------------------*/

  /* Check quad_flag and set quad order to max order (note, not using 39 order rule) */
  if (quad_order == PXE_MaxQuad) quad_order = 23;


  /* loop through quad rules until sufficient order is achieved */
  /* maps quad_rule to nquad using QO (quad_order) array */
/*   quad_rule = 0; */
/*   while (QO[quad_rule] < quad_order) { */
/*     quad_rule++; */
/*     if (QO[quad_rule] < 0){ */
/*       printf("Order %d integration not supported in PXQuadLine.\n", quad_order); */
/*       return PXError(PX_BAD_INPUT); */
/*     } */
/*   } */

  /*----------------*/
  /* Allocate space */
  /*----------------*/

  /* Set number of quad points */
  //nquad = (*pnquad) = QN[quad_rule];
    nquad = PXquad_orderTonquadLineLookupTable[quad_order<<1]; //2*quad_order
    (*pnquad) = PXquad_orderTonquadLineLookupTable[quad_order<<1];
    quad_rule = (enum PXE_QuadratureRule) PXquad_orderTonquadLineLookupTable[(quad_order<<1) | 1]; //2*quad_order+1
  

  if(pquad_rule != NULL)
    *pquad_rule = quad_rule;

  /* If user only wants nquad */
  if(pxquad == NULL && pwquad == NULL)
    return PX_NO_ERROR;

  /* Allocate space */
  if (pxquad!=NULL){
    //PXErrorReturn( PXReAllocate( nquad, sizeof(double), (void **)pxquad ) ); 
    xquad = *pxquad;
  }
  if (pwquad != NULL) {
    //PXErrorReturn( PXReAllocate( nquad, sizeof(double), (void **)pwquad ) ); 
    wquad = *pwquad;
  }

  /* Set value for coordinates (abscisses) and weightings */
  switch (quad_rule) {

  case PXE_QuadRule0:
    if (xquad != NULL){
      xquad[0] =  0.0;
    }
    if (wquad != NULL) {
      wquad[0] =  2.0;
    }
    break;

  case PXE_QuadRule1:
    if (xquad != NULL){
      xquad[0] = -0.577350269189626;
      xquad[1] =  0.577350269189626;
    }
    if (wquad != NULL) {
      wquad[0] =  1.0;
      wquad[1] =  1.0;
    }
    break;

  case PXE_QuadRule2:
    if (xquad != NULL){
      xquad[0] = -0.7745966692414833770358531;
      xquad[1] =  0.0000000000000000000000000;
      xquad[2] =  0.7745966692414833770358531;
    }
    if (wquad != NULL) {
      wquad[0] =  0.5555555555555555555555556;
      wquad[1] =  0.8888888888888888888888889;
      wquad[2] =  0.5555555555555555555555556;
    }
    break;

  case PXE_QuadRule3:
    if (xquad != NULL){
      xquad[0] = -0.8611363115940525752239465;
      xquad[1] = -0.3399810435848562648026658;
      xquad[2] =  0.3399810435848562648026658;
      xquad[3] =  0.8611363115940525752239465;
    }
    if (wquad != NULL) {
      wquad[0] =  0.3478548451374538573730639;
      wquad[1] =  0.6521451548625461426269361;
      wquad[2] =  0.6521451548625461426269361;
      wquad[3] =  0.3478548451374538573730639;
    }
    break;

  case PXE_QuadRule4:
    if (xquad != NULL){
      xquad[0] = -0.9061798459386639927976269;
      xquad[1] = -0.5384693101056830910363144;
      xquad[2] =  0.0000000000000000000000000;
      xquad[3] =  0.5384693101056830910363144;
      xquad[4] =  0.9061798459386639927976269;
    }
    if (wquad != NULL) {
      wquad[0] =  0.2369268850561890875142640;
      wquad[1] =  0.4786286704993664680412915;
      wquad[2] =  0.5688888888888888888888889;
      wquad[3] =  0.4786286704993664680412915;
      wquad[4] =  0.2369268850561890875142640;
    }
    break;

  case PXE_QuadRule5:
    if (xquad != NULL){
      xquad[0] = -0.9324695142031520278123016;
      xquad[1] = -0.6612093864662645136613996;
      xquad[2] = -0.2386191860831969086305017;
      xquad[3] = 0.2386191860831969086305017 ;
      xquad[4] = 0.6612093864662645136613996 ;
      xquad[5] = 0.9324695142031520278123016 ;
    }
    if (wquad != NULL) {
      wquad[0] =   0.1713244923791703450402961;
      wquad[1] =   0.3607615730481386075698335;
      wquad[2] =   0.4679139345726910473898703;
      wquad[3] =  0.4679139345726910473898703 ;
      wquad[4] =  0.3607615730481386075698335 ;
      wquad[5] =  0.1713244923791703450402961 ;
    }
    break;

  case PXE_QuadRule6:
    if (xquad != NULL){
      xquad[0] = -0.9491079123427585245261897;
      xquad[1] = -0.7415311855993944398638648;
      xquad[2] = -0.4058451513773971669066064;
      xquad[3] = 0.0000000000000000000000000 ;
      xquad[4] = 0.4058451513773971669066064 ;
      xquad[5] = 0.7415311855993944398638648 ;
      xquad[6] = 0.9491079123427585245261897 ;
    }
    if (wquad != NULL) {
      wquad[0] =  0.1294849661688696932706114;
      wquad[1] =  0.2797053914892766679014678;
      wquad[2] =  0.3818300505051189449503698;
      wquad[3] = 0.4179591836734693877551020 ;
      wquad[4] = 0.3818300505051189449503698 ;
      wquad[5] = 0.2797053914892766679014678 ;
      wquad[6] = 0.1294849661688696932706114 ;
    }
    break;

  case PXE_QuadRule7:
    if (xquad != NULL){
      xquad[0] = -0.9602898564975362316835609;
      xquad[1] = -0.7966664774136267395915539;
      xquad[2] = -0.5255324099163289858177390;
      xquad[3] = -0.1834346424956498049394761;
      xquad[4] = 0.1834346424956498049394761 ;
      xquad[5] = 0.5255324099163289858177390 ;
      xquad[6] = 0.7966664774136267395915539 ;
      xquad[7] = 0.9602898564975362316835609 ;
    }
    if (wquad != NULL) {
      wquad[0] =   0.1012285362903762591525314;
      wquad[1] =   0.2223810344533744705443560;
      wquad[2] =   0.3137066458778872873379622;
      wquad[3] =   0.3626837833783619829651504;
      wquad[4] =  0.3626837833783619829651504 ;
      wquad[5] =  0.3137066458778872873379622 ;
      wquad[6] =  0.2223810344533744705443560 ;
      wquad[7] =  0.1012285362903762591525314 ;
    }
    break;

  case PXE_QuadRule8:
    if (xquad != NULL){
      xquad[0] = -0.9681602395076260898355762;
      xquad[1] = -0.8360311073266357942994297;
      xquad[2] = -0.6133714327005903973087020;
      xquad[3] = -0.3242534234038089290385380;
      xquad[4] = 0.0000000000000000000000000 ;
      xquad[5] = 0.3242534234038089290385380 ;
      xquad[6] = 0.6133714327005903973087020 ;
      xquad[7] = 0.8360311073266357942994298 ;
      xquad[8] = 0.9681602395076260898355762 ;
    }
    if (wquad != NULL) {
      wquad[0] =   0.0812743883615744119718922;
      wquad[1] =   0.1806481606948574040584720;
      wquad[2] =   0.2606106964029354623187429;
      wquad[3] =   0.3123470770400028400686304;
      wquad[4] =  0.3302393550012597631645251 ;
      wquad[5] =  0.3123470770400028400686304 ;
      wquad[6] =  0.2606106964029354623187429 ;
      wquad[7] =  0.1806481606948574040584720 ;
      wquad[8] =  0.0812743883615744119718922 ;
    }
    break;

  case PXE_QuadRule9:
    if (xquad != NULL){
      xquad[0] = -0.9739065285171717200779640;
      xquad[1] = -0.8650633666889845107320967;
      xquad[2] = -0.6794095682990244062343274;
      xquad[3] = -0.4333953941292471907992659;
      xquad[4] = -0.1488743389816312108848260;
      xquad[5] = 0.1488743389816312108848260 ;
      xquad[6] = 0.4333953941292471907992659 ;
      xquad[7] = 0.6794095682990244062343274 ;
      xquad[8] = 0.8650633666889845107320967 ;
      xquad[9] = 0.9739065285171717200779640 ;
    }
    if (wquad != NULL) {
      wquad[0] =    0.0666713443086881375935688;
      wquad[1] =   0.1494513491505805931457763 ;
      wquad[2] =   0.2190863625159820439955349 ;
      wquad[3] =   0.2692667193099963550912269 ;
      wquad[4] =   0.2955242247147528701738930 ;
      wquad[5] =  0.2955242247147528701738930  ;
      wquad[6] =  0.2692667193099963550912269  ;
      wquad[7] =  0.2190863625159820439955349  ;
      wquad[8] =  0.1494513491505805931457763  ;
      wquad[9] =  0.0666713443086881375935688  ;
    }
    break;

  case PXE_QuadRule10:
    if (xquad != NULL){
      xquad[0]  = -0.9782286581460569928039380;
      xquad[1]  = -0.8870625997680952990751578;
      xquad[2]  = -0.7301520055740493240934162;
      xquad[3]  = -0.5190961292068118159257257;
      xquad[4]  = -0.2695431559523449723315320;
      xquad[5]  = 0.0000000000000000000000000 ;
      xquad[6]  = 0.2695431559523449723315320 ;
      xquad[7]  = 0.5190961292068118159257257 ;
      xquad[8]  = 0.7301520055740493240934163 ;
      xquad[9]  = 0.8870625997680952990751578 ;
      xquad[10] = 0.9782286581460569928039380 ;
    }
    if (wquad != NULL) {
      wquad[0]  =   0.0556685671161736664827537;
      wquad[1]  =   0.1255803694649046246346943;
      wquad[2]  =   0.1862902109277342514260976;
      wquad[3]  =   0.2331937645919904799185237;
      wquad[4]  =   0.2628045445102466621806889;
      wquad[5]  =   0.2729250867779006307144835 ;
      wquad[6]  =   0.2628045445102466621806889 ;
      wquad[7]  =   0.2331937645919904799185237 ;
      wquad[8]  =   0.1862902109277342514260980 ;
      wquad[9]  =   0.1255803694649046246346940 ;
      wquad[10] =   0.0556685671161736664827537;
    }
    break;

  case PXE_QuadRule11:
    if (xquad != NULL){
      xquad[0] = -0.9815606342467192506905491;
      xquad[1] = -0.9041172563704748566784659;
      xquad[2] = -0.7699026741943046870368938;
      xquad[3] = -0.5873179542866174472967024;
      xquad[4] = -0.3678314989981801937526915;
      xquad[5] = -0.1252334085114689154724414;
      xquad[6] = 0.1252334085114689154724414 ;
      xquad[7] = 0.3678314989981801937526915 ;
      xquad[8] = 0.5873179542866174472967024 ;
      xquad[9] = 0.7699026741943046870368938 ;
      xquad[10] =0.9041172563704748566784659 ;
      xquad[11] =0.9815606342467192506905491 ;
    }
    if (wquad != NULL) {
      wquad[0] =  0.0471753363865118271946160;
      wquad[1] =  0.1069393259953184309602547;
      wquad[2] =  0.1600783285433462263346525;
      wquad[3] =  0.2031674267230659217490645;
      wquad[4] =  0.2334925365383548087608499;
      wquad[5] =  0.2491470458134027850005624;
      wquad[6] = 0.2491470458134027850005624 ;
      wquad[7] = 0.2334925365383548087608499 ;
      wquad[8] = 0.2031674267230659217490645 ;
      wquad[9] = 0.1600783285433462263346525 ;
      wquad[10] =0.1069393259953184309602547 ;
      wquad[11] =0.0471753363865118271946160 ; 
    }
    break;


  default:
    printf("Unknown quadrature rule %d\n", quad_rule);
    return PX_BAD_INPUT;
    break;
  }

  /* scale into reference element 0-1 */
  if (xquad!=NULL){
    for (iquad=0; iquad<nquad; iquad++){
      xquad[iquad] = 0.5*xquad[iquad]+0.5;
    }
  }
  
  if (wquad!=NULL){
    for (iquad=0; iquad<nquad; iquad++){
      wquad[iquad]*=0.5;
    }
  }
  
  return PX_NO_ERROR;
}


/******************************************************************/
//   FUNCTION Definition: PXQuadTriangle
ELVIS_DEVICE int
PXQuadTriangle(int quad_order, int *pnquad, enum PXE_QuadratureRule *pquad_rule, double **pxquad, double **pwquad)
{
  int ierr;  // error code
  enum PXE_QuadratureRule quad_rule; // index for quadrature rules
  int iquad;
  int nquad;     // number of quad points

  double *xquad = NULL;        // quad points
  double *wquad = NULL; // quad weights

  /* original quad_order to nquad mapping arrays */
/*   int QO[] = {1, 2, 3, 4, 5,  6,  7,  8,  9, 10, 12, 13, 14,  30, -1};  // order of quad rules */
/*   int QN[] = {1, 3, 4, 6, 7, 12, 13, 16, 19, 25, 33, 37, 42, 175, -1};  // number of points for quad rules */

  /* Data is ordered in pairs of (nquad, quad_rule) */
  const int PXquad_orderTonquadTriangleLookupTable[2*15] = {
    1, 0, //quad_order == 0, quad_rule = 0
    1, 0, //quad_order == 1, quad_rule = 0
    3, 1, //quad_order == 2, quad_rule = 1
    4, 2, //quad_order == 3, quad_rule = 2
    6, 3, //quad_order == 4, quad_rule = 3
    7, 4, //quad_order == 5, quad_rule = 4
    12, 5, //quad_order == 6, quad_rule = 5
    13, 6, //quad_order == 7, quad_rule = 6
    16, 7, //quad_order == 8, quad_rule = 7
    19, 8, //quad_order == 9, quad_rule = 8
    25, 9, //quad_order == 10, quad_rule = 9
    33, 10, //quad_order == 11, quad_rule = 10
    33, 10, //quad_order == 12, quad_rule = 10
    37, 11, //quad_order == 13, quad_rule = 11
    42, 12};//quad_order == 14, quad_rule = 12

  /* quad_order = 30, nquad = 175 is quad_rule 13 */
  /* This last 'dense' quad_rule is not included in the lookup table 
     to decrease the table size */

  /*---------------------*/
  /* Determine quad rule */
  /*---------------------*/

  /* Check quad_flag and set quad order to max order (note, not using 20) */
  if (quad_order == PXE_MaxQuad) quad_order = 14;
  
  /* loop through quad rules until sufficient order is achieved */
  /* maps quad_rule to nquad using QO (quad_order) array */
/*   quad_rule = 0; */
/*   while (QO[quad_rule] < quad_order) { */
/*     quad_rule++; */
/*     if (QO[quad_rule] < 0){ */
/*       printf("Order %d integration not supported in PXQuadLine.\n", quad_order); */
/*       return PXError(PX_BAD_INPUT); */
/*     } */
/*   } */

  /*----------------*/
  /* Allocate space */
  /*----------------*/

  /* Set number of quad points */
  //nquad = (*pnquad) = QN[quad_rule];
    nquad = PXquad_orderTonquadTriangleLookupTable[quad_order<<1];//2*quad_order
    (*pnquad) = PXquad_orderTonquadTriangleLookupTable[quad_order<<1];
    quad_rule = (enum PXE_QuadratureRule) PXquad_orderTonquadTriangleLookupTable[(quad_order<<1) | 1];//2*quad_order + 1

  if(pquad_rule != NULL)
    *pquad_rule = quad_rule;

  /* If user only wants nquad */
  if(pxquad == NULL && pwquad == NULL)
    return PX_NO_ERROR;

  /* Allocate space */
  if(pxquad != NULL){
    //PXErrorReturn( PXReAllocate( 2*nquad, sizeof(double), (void **)pxquad ) ); 
    xquad = *pxquad;
  }
  
  if (pwquad != NULL) {
    //PXErrorReturn( PXReAllocate( nquad, sizeof(double), (void **)pwquad ) ); 
    wquad = *pwquad;
  }

  /* Set value for coordinates (abscisses) and weightings */
  switch (quad_rule) {

  case PXE_QuadRule0:
    /* Rule 0: 1 point, Order 1 */
    if(xquad != NULL){
      xquad[  0] =  0.333333333333333; xquad[  1] =  0.333333333333333;
    }

    if (wquad != NULL) {
      wquad[  0] =  1.000000000000000;
    }
    break;

  case PXE_QuadRule1:
    /* Strang and Fix formula #1, Order 2 */
    if(xquad != NULL){
      xquad[  0] =  0.666666666666667; xquad[  1] =  0.166666666666667;
      xquad[  2] =  0.166666666666667; xquad[  3] =  0.666666666666667;
      xquad[  4] =  0.166666666666667; xquad[  5] =  0.166666666666667;
    }

    if (wquad != NULL) {
      wquad[  0] =  0.333333333333333;
      wquad[  1] =  0.333333333333333;
      wquad[  2] =  0.333333333333333;
    }
    break;

  case PXE_QuadRule2:
    /* Rule 2 Strang and Fix formula #3, Zienkiewicz #3, Order 3 */
    if(xquad != NULL){
      xquad[  0] =  0.333333333333333; xquad[  1] =  0.333333333333333;
      xquad[  2] =  0.600000000000000; xquad[  3] =  0.200000000000000;
      xquad[  4] =  0.200000000000000; xquad[  5] =  0.600000000000000;
      xquad[  6] =  0.200000000000000; xquad[  7] =  0.200000000000000;
    }

    if (wquad != NULL) {
      wquad[  0] = -0.562500000000000;
      wquad[  1] =  0.520833333333333;
      wquad[  2] =  0.520833333333333;
      wquad[  3] =  0.520833333333333;
    }
    break;

  case PXE_QuadRule3:
    /* Rule 3 Strang and Fix formula #5, Order 4 */
    if(xquad != NULL){
      xquad[  0] =  0.816847572980459; xquad[  1] =  0.091576213509771;
      xquad[  2] =  0.091576213509771; xquad[  3] =  0.816847572980459;
      xquad[  4] =  0.091576213509771; xquad[  5] =  0.091576213509771;
      xquad[  6] =  0.108103018168070; xquad[  7] =  0.445948490915965;
      xquad[  8] =  0.445948490915965; xquad[  9] =  0.108103018168070;
      xquad[ 10] =  0.445948490915965; xquad[ 11] =  0.445948490915965;
    }

    if (wquad != NULL) {
      wquad[  0] =  0.109951743655322;
      wquad[  1] =  0.109951743655322;
      wquad[  2] =  0.109951743655322;
      wquad[  3] =  0.223381589678011;
      wquad[  4] =  0.223381589678011;
      wquad[  5] =  0.223381589678011;
    }
    break;

  case PXE_QuadRule4:
    /* Rule 4 Strang and Fix formula #7, Stroud T2:5-1, Order 5 */
    if(xquad != NULL){
      xquad[  0] =  0.333333333333333; xquad[  1] =  0.333333333333333;
      xquad[  2] =  0.797426985353087; xquad[  3] =  0.101286507323456;
      xquad[  4] =  0.101286507323456; xquad[  5] =  0.797426985353087;
      xquad[  6] =  0.101286507323456; xquad[  7] =  0.101286507323456;
      xquad[  8] =  0.059715871789770; xquad[  9] =  0.470142064105115;
      xquad[ 10] =  0.470142064105115; xquad[ 11] =  0.059715871789770;
      xquad[ 12] =  0.470142064105115; xquad[ 13] =  0.470142064105115;
    }

    if (wquad != NULL) {
      wquad[  0] =  0.225000000000000;
      wquad[  1] =  0.125939180544827;
      wquad[  2] =  0.125939180544827;
      wquad[  3] =  0.125939180544827;
      wquad[  4] =  0.132394152788506;
      wquad[  5] =  0.132394152788506;
      wquad[  6] =  0.132394152788506;
    }
    break;

  case PXE_QuadRule5:
    /* Rule 5 Solin, Segeth and Dolezel (SSD): order 6 */
    if(xquad != NULL){
      xquad[  0] =  0.249286745170910; xquad[  1] =  0.249286745170910;
      xquad[  2] =  0.249286745170910; xquad[  3] =  0.501426509658179;
      xquad[  4] =  0.501426509658179; xquad[  5] =  0.249286745170910;
      xquad[  6] =  0.063089014491502; xquad[  7] =  0.063089014491502;
      xquad[  8] =  0.063089014491502; xquad[  9] =  0.873821971016996;
      xquad[ 10] =  0.873821971016996; xquad[ 11] =  0.063089014491502;
      xquad[ 12] =  0.310352451033784; xquad[ 13] =  0.636502499121399;
      xquad[ 14] =  0.636502499121399; xquad[ 15] =  0.053145049844817;
      xquad[ 16] =  0.053145049844817; xquad[ 17] =  0.310352451033784;
      xquad[ 18] =  0.310352451033784; xquad[ 19] =  0.053145049844817;
      xquad[ 20] =  0.636502499121399; xquad[ 21] =  0.310352451033784;
      xquad[ 22] =  0.053145049844817; xquad[ 23] =  0.636502499121399;
    }

    if (wquad != NULL) {
      wquad[  0] =  0.116786275726379;
      wquad[  1] =  0.116786275726379;
      wquad[  2] =  0.116786275726379;
      wquad[  3] =  0.050844906370207;
      wquad[  4] =  0.050844906370207;
      wquad[  5] =  0.050844906370207;
      wquad[  6] =  0.082851075618374;
      wquad[  7] =  0.082851075618374;
      wquad[  8] =  0.082851075618374;
      wquad[  9] =  0.082851075618374;
      wquad[ 10] =  0.082851075618374;
      wquad[ 11] =  0.082851075618374;
    }
    break;

  case PXE_QuadRule6:
    /* Rule 6 Solin, Segeth and Dolezel (SSD): order 7 */
    if(xquad != NULL){
      xquad[  0] =  0.333333333333333; xquad[  1] =  0.333333333333333;
      xquad[  2] =  0.260345966079040; xquad[  3] =  0.260345966079040;
      xquad[  4] =  0.260345966079040; xquad[  5] =  0.479308067841920;
      xquad[  6] =  0.479308067841920; xquad[  7] =  0.260345966079040;
      xquad[  8] =  0.065130102902216; xquad[  9] =  0.065130102902216;
      xquad[ 10] =  0.065130102902216; xquad[ 11] =  0.869739794195568;
      xquad[ 12] =  0.869739794195568; xquad[ 13] =  0.065130102902216;
      xquad[ 14] =  0.312865496004874; xquad[ 15] =  0.638444188569810;
      xquad[ 16] =  0.638444188569810; xquad[ 17] =  0.048690315425316;
      xquad[ 18] =  0.048690315425316; xquad[ 19] =  0.312865496004874;
      xquad[ 20] =  0.312865496004874; xquad[ 21] =  0.048690315425316;
      xquad[ 22] =  0.638444188569810; xquad[ 23] =  0.312865496004874;
      xquad[ 24] =  0.048690315425316; xquad[ 25] =  0.638444188569810;
    }

    if (wquad != NULL) {
      wquad[  0] = -0.149570044467682;
      wquad[  1] =  0.175615257433208;
      wquad[  2] =  0.175615257433208;
      wquad[  3] =  0.175615257433208;
      wquad[  4] =  0.053347235608838;
      wquad[  5] =  0.053347235608838;
      wquad[  6] =  0.053347235608838;
      wquad[  7] =  0.077113760890257;
      wquad[  8] =  0.077113760890257;
      wquad[  9] =  0.077113760890257;
      wquad[ 10] =  0.077113760890257;
      wquad[ 11] =  0.077113760890257;
      wquad[ 12] =  0.077113760890257;
    }
    break;

  case PXE_QuadRule7:
    /* Rule 7 Solin, Segeth and Dolezel (SSD): order 8 */
    if(xquad != NULL){
      xquad[  0] =  0.333333333333333; xquad[  1] =  0.333333333333333;
      xquad[  2] =  0.459292588292723; xquad[  3] =  0.459292588292723;
      xquad[  4] =  0.459292588292723; xquad[  5] =  0.081414823414554;
      xquad[  6] =  0.081414823414554; xquad[  7] =  0.459292588292723;
      xquad[  8] =  0.170569307751760; xquad[  9] =  0.170569307751760;
      xquad[ 10] =  0.170569307751760; xquad[ 11] =  0.658861384496480;
      xquad[ 12] =  0.658861384496480; xquad[ 13] =  0.170569307751760;
      xquad[ 14] =  0.050547228317031; xquad[ 15] =  0.050547228317031;
      xquad[ 16] =  0.050547228317031; xquad[ 17] =  0.898905543365938;
      xquad[ 18] =  0.898905543365938; xquad[ 19] =  0.050547228317031;
      xquad[ 20] =  0.263112829634638; xquad[ 21] =  0.728492392955404;
      xquad[ 22] =  0.728492392955404; xquad[ 23] =  0.008394777409958;
      xquad[ 24] =  0.008394777409958; xquad[ 25] =  0.263112829634638;
      xquad[ 26] =  0.263112829634638; xquad[ 27] =  0.008394777409958;
      xquad[ 28] =  0.728492392955404; xquad[ 29] =  0.263112829634638;
      xquad[ 30] =  0.008394777409958; xquad[ 31] =  0.728492392955404;
    }
    if (wquad != NULL) {
      wquad[  0] =  0.144315607677787;
      wquad[  1] =  0.095091634267285;
      wquad[  2] =  0.095091634267285;
      wquad[  3] =  0.095091634267285;
      wquad[  4] =  0.103217370534718;
      wquad[  5] =  0.103217370534718;
      wquad[  6] =  0.103217370534718;
      wquad[  7] =  0.032458497623198;
      wquad[  8] =  0.032458497623198;
      wquad[  9] =  0.032458497623198;
      wquad[ 10] =  0.027230314174435;
      wquad[ 11] =  0.027230314174435;
      wquad[ 12] =  0.027230314174435;
      wquad[ 13] =  0.027230314174435;
      wquad[ 14] =  0.027230314174435;
      wquad[ 15] =  0.027230314174435;
    }
    break;

  case PXE_QuadRule8:
    /* Rule 8 Solin, Segeth and Dolezel (SSD): order 9 */
    if(xquad != NULL){
      xquad[  0] =  0.333333333333333; xquad[  1] =  0.333333333333333;
      xquad[  2] =  0.489682519198738; xquad[  3] =  0.489682519198738;
      xquad[  4] =  0.489682519198738; xquad[  5] =  0.020634961602525;
      xquad[  6] =  0.020634961602525; xquad[  7] =  0.489682519198738;
      xquad[  8] =  0.437089591492937; xquad[  9] =  0.437089591492937;
      xquad[ 10] =  0.437089591492937; xquad[ 11] =  0.125820817014127;
      xquad[ 12] =  0.125820817014127; xquad[ 13] =  0.437089591492937;
      xquad[ 14] =  0.188203535619033; xquad[ 15] =  0.188203535619033;
      xquad[ 16] =  0.188203535619033; xquad[ 17] =  0.623592928761935;
      xquad[ 18] =  0.623592928761935; xquad[ 19] =  0.188203535619033;
      xquad[ 20] =  0.044729513394453; xquad[ 21] =  0.044729513394453;
      xquad[ 22] =  0.044729513394453; xquad[ 23] =  0.910540973211095;
      xquad[ 24] =  0.910540973211095; xquad[ 25] =  0.044729513394453;
      xquad[ 26] =  0.221962989160766; xquad[ 27] =  0.741198598784498;
      xquad[ 28] =  0.741198598784498; xquad[ 29] =  0.036838412054736;
      xquad[ 30] =  0.036838412054736; xquad[ 31] =  0.221962989160766;
      xquad[ 32] =  0.221962989160766; xquad[ 33] =  0.036838412054736;
      xquad[ 34] =  0.741198598784498; xquad[ 35] =  0.221962989160766;
      xquad[ 36] =  0.036838412054736; xquad[ 37] =  0.741198598784498;
    }

    if (wquad != NULL) {
      wquad[  0] =  0.097135796282799;
      wquad[  1] =  0.031334700227139;
      wquad[  2] =  0.031334700227139;
      wquad[  3] =  0.031334700227139;
      wquad[  4] =  0.077827541004774;
      wquad[  5] =  0.077827541004774;
      wquad[  6] =  0.077827541004774;
      wquad[  7] =  0.079647738927210;
      wquad[  8] =  0.079647738927210;
      wquad[  9] =  0.079647738927210;
      wquad[ 10] =  0.025577675658698;
      wquad[ 11] =  0.025577675658698;
      wquad[ 12] =  0.025577675658698;
      wquad[ 13] =  0.043283539377289;
      wquad[ 14] =  0.043283539377289;
      wquad[ 15] =  0.043283539377289;
      wquad[ 16] =  0.043283539377289;
      wquad[ 17] =  0.043283539377289;
      wquad[ 18] =  0.043283539377289;
    }
    break;

  case PXE_QuadRule9:
    /* Rule 9 Solin, Segeth and Dolezel (SSD): order 10 */
    if(xquad != NULL){
      xquad[  0] =  0.333333333333333; xquad[  1] =  0.333333333333333;
      xquad[  2] =  0.485577633383657; xquad[  3] =  0.485577633383657;
      xquad[  4] =  0.485577633383657; xquad[  5] =  0.028844733232685;
      xquad[  6] =  0.028844733232685; xquad[  7] =  0.485577633383657;
      xquad[  8] =  0.109481575485037; xquad[  9] =  0.109481575485037;
      xquad[ 10] =  0.109481575485037; xquad[ 11] =  0.781036849029926;
      xquad[ 12] =  0.781036849029926; xquad[ 13] =  0.109481575485037;
      xquad[ 14] =  0.307939838764121; xquad[ 15] =  0.550352941820999;
      xquad[ 16] =  0.550352941820999; xquad[ 17] =  0.141707219414880;
      xquad[ 18] =  0.141707219414880; xquad[ 19] =  0.307939838764121;
      xquad[ 20] =  0.307939838764121; xquad[ 21] =  0.141707219414880;
      xquad[ 22] =  0.550352941820999; xquad[ 23] =  0.307939838764121;
      xquad[ 24] =  0.141707219414880; xquad[ 25] =  0.550352941820999;
      xquad[ 26] =  0.246672560639903; xquad[ 27] =  0.728323904597411;
      xquad[ 28] =  0.728323904597411; xquad[ 29] =  0.025003534762686;
      xquad[ 30] =  0.025003534762686; xquad[ 31] =  0.246672560639903;
      xquad[ 32] =  0.246672560639903; xquad[ 33] =  0.025003534762686;
      xquad[ 34] =  0.728323904597411; xquad[ 35] =  0.246672560639903;
      xquad[ 36] =  0.025003534762686; xquad[ 37] =  0.728323904597411;
      xquad[ 38] =  0.066803251012200; xquad[ 39] =  0.923655933587500;
      xquad[ 40] =  0.923655933587500; xquad[ 41] =  0.009540815400299;
      xquad[ 42] =  0.009540815400299; xquad[ 43] =  0.066803251012200;
      xquad[ 44] =  0.066803251012200; xquad[ 45] =  0.009540815400299;
      xquad[ 46] =  0.923655933587500; xquad[ 47] =  0.066803251012200;
      xquad[ 48] =  0.009540815400299; xquad[ 49] =  0.923655933587500;
    }
  
    if (wquad != NULL) {
      wquad[  0] =  0.090817990382754;
      wquad[  1] =  0.036725957756467;
      wquad[  2] =  0.036725957756467;
      wquad[  3] =  0.036725957756467;
      wquad[  4] =  0.045321059435528;
      wquad[  5] =  0.045321059435528;
      wquad[  6] =  0.045321059435528;
      wquad[  7] =  0.072757916845420;
      wquad[  8] =  0.072757916845420;
      wquad[  9] =  0.072757916845420;
      wquad[ 10] =  0.072757916845420;
      wquad[ 11] =  0.072757916845420;
      wquad[ 12] =  0.072757916845420;
      wquad[ 13] =  0.028327242531057;
      wquad[ 14] =  0.028327242531057;
      wquad[ 15] =  0.028327242531057;
      wquad[ 16] =  0.028327242531057;
      wquad[ 17] =  0.028327242531057;
      wquad[ 18] =  0.028327242531057;
      wquad[ 19] =  0.009421666963733;
      wquad[ 20] =  0.009421666963733;
      wquad[ 21] =  0.009421666963733;
      wquad[ 22] =  0.009421666963733;
      wquad[ 23] =  0.009421666963733;
      wquad[ 24] =  0.009421666963733;
    }
    break;

  case PXE_QuadRule10:
    /* Rule 10 Solin, Segeth and Dolezel (SSD): order 12 */
    if(xquad != NULL){
      xquad[  0] =  0.488217389773805; xquad[  1] =  0.488217389773805;
      xquad[  2] =  0.488217389773805; xquad[  3] =  0.023565220452390;
      xquad[  4] =  0.023565220452390; xquad[  5] =  0.488217389773805;
      xquad[  6] =  0.439724392294460; xquad[  7] =  0.439724392294460;
      xquad[  8] =  0.439724392294460; xquad[  9] =  0.120551215411079;
      xquad[ 10] =  0.120551215411079; xquad[ 11] =  0.439724392294460;
      xquad[ 12] =  0.271210385012116; xquad[ 13] =  0.271210385012116;
      xquad[ 14] =  0.271210385012116; xquad[ 15] =  0.457579229975768;
      xquad[ 16] =  0.457579229975768; xquad[ 17] =  0.271210385012116;
      xquad[ 18] =  0.127576145541586; xquad[ 19] =  0.127576145541586;
      xquad[ 20] =  0.127576145541586; xquad[ 21] =  0.744847708916828;
      xquad[ 22] =  0.744847708916828; xquad[ 23] =  0.127576145541586;
      xquad[ 24] =  0.021317350453210; xquad[ 25] =  0.021317350453210;
      xquad[ 26] =  0.021317350453210; xquad[ 27] =  0.957365299093579;
      xquad[ 28] =  0.957365299093579; xquad[ 29] =  0.021317350453210;
      xquad[ 30] =  0.275713269685514; xquad[ 31] =  0.608943235779788;
      xquad[ 32] =  0.608943235779788; xquad[ 33] =  0.115343494534698;
      xquad[ 34] =  0.115343494534698; xquad[ 35] =  0.275713269685514;
      xquad[ 36] =  0.275713269685514; xquad[ 37] =  0.115343494534698;
      xquad[ 38] =  0.608943235779788; xquad[ 39] =  0.275713269685514;
      xquad[ 40] =  0.115343494534698; xquad[ 41] =  0.608943235779788;
      xquad[ 42] =  0.281325580989940; xquad[ 43] =  0.695836086787803;
      xquad[ 44] =  0.695836086787803; xquad[ 45] =  0.022838332222257;
      xquad[ 46] =  0.022838332222257; xquad[ 47] =  0.281325580989940;
      xquad[ 48] =  0.281325580989940; xquad[ 49] =  0.022838332222257;
      xquad[ 50] =  0.695836086787803; xquad[ 51] =  0.281325580989940;
      xquad[ 52] =  0.022838332222257; xquad[ 53] =  0.695836086787803;
      xquad[ 54] =  0.116251915907597; xquad[ 55] =  0.858014033544073;
      xquad[ 56] =  0.858014033544073; xquad[ 57] =  0.025734050548330;
      xquad[ 58] =  0.025734050548330; xquad[ 59] =  0.116251915907597;
      xquad[ 60] =  0.116251915907597; xquad[ 61] =  0.025734050548330;
      xquad[ 62] =  0.858014033544073; xquad[ 63] =  0.116251915907597;
      xquad[ 64] =  0.025734050548330; xquad[ 65] =  0.858014033544073;
    }

    if (wquad != NULL) {
      wquad[  0] =  0.025731066440455;
      wquad[  1] =  0.025731066440455;
      wquad[  2] =  0.025731066440455;
      wquad[  3] =  0.043692544538038;
      wquad[  4] =  0.043692544538038;
      wquad[  5] =  0.043692544538038;
      wquad[  6] =  0.062858224217885;
      wquad[  7] =  0.062858224217885;
      wquad[  8] =  0.062858224217885;
      wquad[  9] =  0.034796112930709;
      wquad[ 10] =  0.034796112930709;
      wquad[ 11] =  0.034796112930709;
      wquad[ 12] =  0.006166261051559;
      wquad[ 13] =  0.006166261051559;
      wquad[ 14] =  0.006166261051559;
      wquad[ 15] =  0.040371557766381;
      wquad[ 16] =  0.040371557766381;
      wquad[ 17] =  0.040371557766381;
      wquad[ 18] =  0.040371557766381;
      wquad[ 19] =  0.040371557766381;
      wquad[ 20] =  0.040371557766381;
      wquad[ 21] =  0.022356773202303;
      wquad[ 22] =  0.022356773202303;
      wquad[ 23] =  0.022356773202303;
      wquad[ 24] =  0.022356773202303;
      wquad[ 25] =  0.022356773202303;
      wquad[ 26] =  0.022356773202303;
      wquad[ 27] =  0.017316231108659;
      wquad[ 28] =  0.017316231108659;
      wquad[ 29] =  0.017316231108659;
      wquad[ 30] =  0.017316231108659;
      wquad[ 31] =  0.017316231108659;
      wquad[ 32] =  0.017316231108659;
    }
    break;
  
  case PXE_QuadRule11:
    /* Rule 11 Solin, Segeth and Dolezel (SSD): order 13 */
    if(xquad != NULL){
      xquad[  0] =  0.333333333333333; xquad[  1] =  0.333333333333333;
      xquad[  2] =  0.495048184939705; xquad[  3] =  0.495048184939705;
      xquad[  4] =  0.495048184939705; xquad[  5] =  0.009903630120591;
      xquad[  6] =  0.009903630120591; xquad[  7] =  0.495048184939705;
      xquad[  8] =  0.468716635109574; xquad[  9] =  0.468716635109574;
      xquad[ 10] =  0.468716635109574; xquad[ 11] =  0.062566729780852;
      xquad[ 12] =  0.062566729780852; xquad[ 13] =  0.468716635109574;
      xquad[ 14] =  0.414521336801277; xquad[ 15] =  0.414521336801277;
      xquad[ 16] =  0.414521336801277; xquad[ 17] =  0.170957326397447;
      xquad[ 18] =  0.170957326397447; xquad[ 19] =  0.414521336801277;
      xquad[ 20] =  0.229399572042831; xquad[ 21] =  0.229399572042831;
      xquad[ 22] =  0.229399572042831; xquad[ 23] =  0.541200855914337;
      xquad[ 24] =  0.541200855914337; xquad[ 25] =  0.229399572042831;
      xquad[ 26] =  0.114424495196330; xquad[ 27] =  0.114424495196330;
      xquad[ 28] =  0.114424495196330; xquad[ 29] =  0.771151009607340;
      xquad[ 30] =  0.771151009607340; xquad[ 31] =  0.114424495196330;
      xquad[ 32] =  0.024811391363459; xquad[ 33] =  0.024811391363459;
      xquad[ 34] =  0.024811391363459; xquad[ 35] =  0.950377217273082;
      xquad[ 36] =  0.950377217273082; xquad[ 37] =  0.024811391363459;
      xquad[ 38] =  0.268794997058761; xquad[ 39] =  0.636351174561660;
      xquad[ 40] =  0.636351174561660; xquad[ 41] =  0.094853828379579;
      xquad[ 42] =  0.094853828379579; xquad[ 43] =  0.268794997058761;
      xquad[ 44] =  0.268794997058761; xquad[ 45] =  0.094853828379579;
      xquad[ 46] =  0.636351174561660; xquad[ 47] =  0.268794997058761;
      xquad[ 48] =  0.094853828379579; xquad[ 49] =  0.636351174561660;
      xquad[ 50] =  0.291730066734288; xquad[ 51] =  0.690169159986905;
      xquad[ 52] =  0.690169159986905; xquad[ 53] =  0.018100773278807;
      xquad[ 54] =  0.018100773278807; xquad[ 55] =  0.291730066734288;
      xquad[ 56] =  0.291730066734288; xquad[ 57] =  0.018100773278807;
      xquad[ 58] =  0.690169159986905; xquad[ 59] =  0.291730066734288;
      xquad[ 60] =  0.018100773278807; xquad[ 61] =  0.690169159986905;
      xquad[ 62] =  0.126357385491669; xquad[ 63] =  0.851409537834241;
      xquad[ 64] =  0.851409537834241; xquad[ 65] =  0.022233076674090;
      xquad[ 66] =  0.022233076674090; xquad[ 67] =  0.126357385491669;
      xquad[ 68] =  0.126357385491669; xquad[ 69] =  0.022233076674090;
      xquad[ 70] =  0.851409537834241; xquad[ 71] =  0.126357385491669;
      xquad[ 72] =  0.022233076674090; xquad[ 73] =  0.851409537834241;
    }

    if (wquad != NULL) {
      wquad[  0] =  0.052520923400802;
      wquad[  1] =  0.011280145209330;
      wquad[  2] =  0.011280145209330;
      wquad[  3] =  0.011280145209330;
      wquad[  4] =  0.031423518362454;
      wquad[  5] =  0.031423518362454;
      wquad[  6] =  0.031423518362454;
      wquad[  7] =  0.047072502504194;
      wquad[  8] =  0.047072502504194;
      wquad[  9] =  0.047072502504194;
      wquad[ 10] =  0.047363586536355;
      wquad[ 11] =  0.047363586536355;
      wquad[ 12] =  0.047363586536355;
      wquad[ 13] =  0.031167529045794;
      wquad[ 14] =  0.031167529045794;
      wquad[ 15] =  0.031167529045794;
      wquad[ 16] =  0.007975771465074;
      wquad[ 17] =  0.007975771465074;
      wquad[ 18] =  0.007975771465074;
      wquad[ 19] =  0.036848402728732;
      wquad[ 20] =  0.036848402728732;
      wquad[ 21] =  0.036848402728732;
      wquad[ 22] =  0.036848402728732;
      wquad[ 23] =  0.036848402728732;
      wquad[ 24] =  0.036848402728732;
      wquad[ 25] =  0.017401463303822;
      wquad[ 26] =  0.017401463303822;
      wquad[ 27] =  0.017401463303822;
      wquad[ 28] =  0.017401463303822;
      wquad[ 29] =  0.017401463303822;
      wquad[ 30] =  0.017401463303822;
      wquad[ 31] =  0.015521786839045;
      wquad[ 32] =  0.015521786839045;
      wquad[ 33] =  0.015521786839045;
      wquad[ 34] =  0.015521786839045;
      wquad[ 35] =  0.015521786839045;
      wquad[ 36] =  0.015521786839045;
    }
    break;
  
  case PXE_QuadRule12:
    /* Rule 12 Solin, Segeth and Dolezel (SSD): order 14 */
    if(xquad != NULL){
      xquad[  0] =  0.488963910362179; xquad[  1] =  0.488963910362179;
      xquad[  2] =  0.488963910362179; xquad[  3] =  0.022072179275643;
      xquad[  4] =  0.022072179275643; xquad[  5] =  0.488963910362179;
      xquad[  6] =  0.417644719340454; xquad[  7] =  0.417644719340454;
      xquad[  8] =  0.417644719340454; xquad[  9] =  0.164710561319092;
      xquad[ 10] =  0.164710561319092; xquad[ 11] =  0.417644719340454;
      xquad[ 12] =  0.273477528308839; xquad[ 13] =  0.273477528308839;
      xquad[ 14] =  0.273477528308839; xquad[ 15] =  0.453044943382323;
      xquad[ 16] =  0.453044943382323; xquad[ 17] =  0.273477528308839;
      xquad[ 18] =  0.177205532412543; xquad[ 19] =  0.177205532412543;
      xquad[ 20] =  0.177205532412543; xquad[ 21] =  0.645588935174913;
      xquad[ 22] =  0.645588935174913; xquad[ 23] =  0.177205532412543;
      xquad[ 24] =  0.061799883090873; xquad[ 25] =  0.061799883090873;
      xquad[ 26] =  0.061799883090873; xquad[ 27] =  0.876400233818255;
      xquad[ 28] =  0.876400233818255; xquad[ 29] =  0.061799883090873;
      xquad[ 30] =  0.019390961248701; xquad[ 31] =  0.019390961248701;
      xquad[ 32] =  0.019390961248701; xquad[ 33] =  0.961218077502598;
      xquad[ 34] =  0.961218077502598; xquad[ 35] =  0.019390961248701;
      xquad[ 36] =  0.172266687821356; xquad[ 37] =  0.770608554774996;
      xquad[ 38] =  0.770608554774996; xquad[ 39] =  0.057124757403648;
      xquad[ 40] =  0.057124757403648; xquad[ 41] =  0.172266687821356;
      xquad[ 42] =  0.172266687821356; xquad[ 43] =  0.057124757403648;
      xquad[ 44] =  0.770608554774996; xquad[ 45] =  0.172266687821356;
      xquad[ 46] =  0.057124757403648; xquad[ 47] =  0.770608554774996;
      xquad[ 48] =  0.336861459796345; xquad[ 49] =  0.570222290846683;
      xquad[ 50] =  0.570222290846683; xquad[ 51] =  0.092916249356972;
      xquad[ 52] =  0.092916249356972; xquad[ 53] =  0.336861459796345;
      xquad[ 54] =  0.336861459796345; xquad[ 55] =  0.092916249356972;
      xquad[ 56] =  0.570222290846683; xquad[ 57] =  0.336861459796345;
      xquad[ 58] =  0.092916249356972; xquad[ 59] =  0.570222290846683;
      xquad[ 60] =  0.298372882136258; xquad[ 61] =  0.686980167808088;
      xquad[ 62] =  0.686980167808088; xquad[ 63] =  0.014646950055654;
      xquad[ 64] =  0.014646950055654; xquad[ 65] =  0.298372882136258;
      xquad[ 66] =  0.298372882136258; xquad[ 67] =  0.014646950055654;
      xquad[ 68] =  0.686980167808088; xquad[ 69] =  0.298372882136258;
      xquad[ 70] =  0.014646950055654; xquad[ 71] =  0.686980167808088;
      xquad[ 72] =  0.118974497696957; xquad[ 73] =  0.879757171370171;
      xquad[ 74] =  0.879757171370171; xquad[ 75] =  0.001268330932872;
      xquad[ 76] =  0.001268330932872; xquad[ 77] =  0.118974497696957;
      xquad[ 78] =  0.118974497696957; xquad[ 79] =  0.001268330932872;
      xquad[ 80] =  0.879757171370171; xquad[ 81] =  0.118974497696957;
      xquad[ 82] =  0.001268330932872; xquad[ 83] =  0.879757171370171;
    }

    if (wquad != NULL) {
      wquad[  0] =  0.021883581369429;
      wquad[  1] =  0.021883581369429;
      wquad[  2] =  0.021883581369429;
      wquad[  3] =  0.032788353544125;
      wquad[  4] =  0.032788353544125;
      wquad[  5] =  0.032788353544125;
      wquad[  6] =  0.051774104507292;
      wquad[  7] =  0.051774104507292;
      wquad[  8] =  0.051774104507292;
      wquad[  9] =  0.042162588736993;
      wquad[ 10] =  0.042162588736993;
      wquad[ 11] =  0.042162588736993;
      wquad[ 12] =  0.014433699669777;
      wquad[ 13] =  0.014433699669777;
      wquad[ 14] =  0.014433699669777;
      wquad[ 15] =  0.004923403602400;
      wquad[ 16] =  0.004923403602400;
      wquad[ 17] =  0.004923403602400;
      wquad[ 18] =  0.024665753212564;
      wquad[ 19] =  0.024665753212564;
      wquad[ 20] =  0.024665753212564;
      wquad[ 21] =  0.024665753212564;
      wquad[ 22] =  0.024665753212564;
      wquad[ 23] =  0.024665753212564;
      wquad[ 24] =  0.038571510787061;
      wquad[ 25] =  0.038571510787061;
      wquad[ 26] =  0.038571510787061;
      wquad[ 27] =  0.038571510787061;
      wquad[ 28] =  0.038571510787061;
      wquad[ 29] =  0.038571510787061;
      wquad[ 30] =  0.014436308113534;
      wquad[ 31] =  0.014436308113534;
      wquad[ 32] =  0.014436308113534;
      wquad[ 33] =  0.014436308113534;
      wquad[ 34] =  0.014436308113534;
      wquad[ 35] =  0.014436308113534;
      wquad[ 36] =  0.005010228838501;
      wquad[ 37] =  0.005010228838501;
      wquad[ 38] =  0.005010228838501;
      wquad[ 39] =  0.005010228838501;
      wquad[ 40] =  0.005010228838501;
      wquad[ 41] =  0.005010228838501;
    }  
    break;
    
  default:
    printf("Unknown quadrature rule %d\n", quad_rule);
    return PX_BAD_INPUT;
    break;
  }

  /* scale into reference element */
  if (wquad!=NULL){
    for (iquad=0; iquad<nquad; iquad++){
      wquad[iquad]*=0.5;
    }
  }

  return PX_NO_ERROR;
}

/******************************************************************/
//   FUNCTION Definition: PXQuadQuad
/* ELVIS_DEVICE int  */
/* PXQuadQuad(int quad_order, int *pnquad, enum PXE_QuadratureRule *pquad_rule, double **pxquad, double **pwquad) */
/* { */
/*   int ierr;  // error code */
/*   int i, j, k; */
/*   int iquad; */
/*   int nquad;     // number of quad points */
/*   int nquadLine; */
/*   double *xquadLine = NULL; // quad line points */
/*   double *wquadLine = NULL; // quad line weights */
/*   double *xquad = NULL; // quad points */
/*   double *wquad = NULL; // quad weights */

/*   /\* Get Quad Line Rule *\/ */
/*   PXErrorReturn( PXQuadLine(quad_order, &nquadLine, pquad_rule, &xquadLine, &wquadLine) ); */

/*   /\* Allocate space *\/ */
/*   nquad = nquadLine*nquadLine; */
/*   (*pnquad) = nquad; */

/*   if (pxquad!=NULL){ */
/*     PXErrorReturn( PXReAllocate( 2*nquad, sizeof(double), (void **)pxquad ) );  */
/*     xquad = (*pxquad); */
/*   } */
/*   if (pwquad != NULL) { */
/*     PXErrorReturn( PXReAllocate( nquad, sizeof(double), (void **)pwquad ) );  */
/*     wquad = (*pwquad); */
/*   } */

/* /\*   for (iquad=0; iquad<nquadLine; iquad++){ *\/ */
/* /\*     printf("iquad = %d, xquadLine = %.15e, wquadLine = %.15e\n",iquad,xquadLine[iquad],wquadLine[iquad]); *\/ */
/* /\*   } *\/ */

/*   /\* the quad quad rules are just the product of the quad line rules *\/ */
/*   k = 0; */
/*   for (j=0; j<nquadLine; j++){ */
/*     for (i=0; i<nquadLine; i++){ */
/*       if (pxquad!=NULL){ */
/* 	xquad[k*2+0] = xquadLine[i]; */
/* 	xquad[k*2+1] = xquadLine[j]; */
/*       } */
/*       if (pwquad != NULL) { */
/* 	wquad[k] = wquadLine[i]*wquadLine[j]; */
/*       } */
/*       k++; */
/*     } */
/*   } */

/*   /\* release memory *\/ */
/*   PXRelease( xquadLine ); */
/*   PXRelease( wquadLine ); */

/*   return PX_NO_ERROR; */

/* } */


/******************************************************************/
//   FUNCTION Definition: PXQuadElemReference
ELVIS_DEVICE int
PXQuadReference(enum PXE_Shape Shape, int quad_order, int *pnquad, enum PXE_Quadrat
ureRule *pquad_rule, double **pxquad, double **pwquad)
{
  
  switch (Shape) {

  case PXE_Shape_Node:
    //PXErrorReturn( PXQuadNode(quad_order, pnquad, pquad_rule, pxquad, pwquad) );
    break;

  case PXE_Shape_Edge:
    PXErrorReturn( PXQuadLine(quad_order, pnquad, pquad_rule, pxquad, pwquad) );
    break;

  case PXE_Shape_Triangle:
    PXErrorReturn( PXQuadTriangle(quad_order, pnquad, pquad_rule, pxquad, pwquad) );
    break;
 
  case PXE_Shape_Quad:
    //PXErrorReturn( PXQuadQuad(quad_order, pnquad, pquad_rule, pxquad, pwquad) );
    break;
       
  case PXE_Shape_Tet:
    //PXErrorReturn( PXQuadTet(quad_order, pnquad, pquad_rule, pxquad, pwquad) );
    break;

  case PXE_Shape_Hex:
    //PXErrorReturn( PXQuadHex(quad_order, pnquad, pquad_rule, pxquad, pwquad) );
    break;

  default:
    return PXError(PX_CODE_FLOW_ERROR);
  }
  
  return PX_NO_ERROR;
}


#endif //PXQUADREFERENCE_ELVIS_C
