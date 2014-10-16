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

#ifndef PXSHAPE_ELVIS_C
#define PXSHAPE_ELVIS_C

//#include <math.h>

//#include "PXShape.h"

/*----------------------------------------------------------------*/
/* WARNING: max p level for SOLN must be >= max p level for GEOM! */
#define GEOM_USE_P0 1
#define GEOM_USE_P1 1
#define GEOM_USE_P2 1
#define GEOM_USE_P3 1
#define GEOM_USE_P4 0
#define GEOM_USE_P5 0

#define SOLN_USE_P0 1
#define SOLN_USE_P1 1
#define SOLN_USE_P2 1
#define SOLN_USE_P3 1
#define SOLN_USE_P4 0
#define SOLN_USE_P5 0

/* check sizing for geometry basis fcns */
#if GEOM_USE_P5 == 1 && (MAX_NBF < 56 || MAX_NBF_FACE < 21)
#error MAX_NBF and/or MAX_NBF_FACE are too small for P5
#endif

#if GEOM_USE_P4 == 1 && (MAX_NBF < 35 || MAX_NBF_FACE < 15)
#error MAX_NBF and/or MAX_NBF_FACE are too small for P4
#endif

#if GEOM_USE_P3 == 1 && (MAX_NBF < 20 || MAX_NBF_FACE < 10)
#error MAX_NBF and/or MAX_NBF_FACE are too small for P3
#endif

#if GEOM_USE_P2 == 1 && (MAX_NBF < 10 || MAX_NBF_FACE < 6)
#error MAX_NBF and/or MAX_NBF_FACE are too small for P2
#endif

#if GEOM_USE_P1 == 1 && (MAX_NBF < 4 || MAX_NBF_FACE < 3)
#error MAX_NBF and/or MAX_NBF_FACE are too small for P1
#endif

#if GEOM_USE_P0 == 1 && (MAX_NBF < 1 || MAX_NBF_FACE < 1)
#error MAX_NBF and/or MAX_NBF_FACE are too small for P0
#endif

/* check sizing for solution basis fcns */
#if SOLN_USE_P5 == 1 && (SOLN_MAX_NBF < 56 || SOLN_MAX_NBF_FACE < 21)
#error SOLN_MAX_NBF and/or SOLN_MAX_NBF_FACE are too small for P5
#endif

#if SOLN_USE_P4 == 1 && (SOLN_MAX_NBF < 35 || SOLN_MAX_NBF_FACE < 15)
#error SOLN_MAX_NBF and/or SOLN_MAX_NBF_FACE are too small for P4
#endif

#if SOLN_USE_P3 == 1 && (SOLN_MAX_NBF < 20 || SOLN_MAX_NBF_FACE < 10)
#error SOLN_MAX_NBF and/or SOLN_MAX_NBF_FACE are too small for P3
#endif

#if SOLN_USE_P2 == 1 && (SOLN_MAX_NBF < 10 || SOLN_MAX_NBF_FACE < 6)
#error SOLN_MAX_NBF and/or SOLN_MAX_NBF_FACE are too small for P2
#endif

#if SOLN_USE_P1 == 1 && (SOLN_MAX_NBF < 4 || SOLN_MAX_NBF_FACE < 3)
#error SOLN_MAX_NBF and/or SOLN_MAX_NBF_FACE are too small for P1
#endif

#if SOLN_USE_P0 == 1 && (SOLN_MAX_NBF < 1 || SOLN_MAX_NBF_FACE < 1)
#error SOLN_MAX_NBF and/or SOLN_MAX_NBF_FACE are too small for P0
#endif


#ifndef ONESIXTEENTH
#define ONESIXTEENTH 0.0625
#endif

#ifndef ONESIXTH
#define ONESIXTH 0.166666666666666666666666667
#endif

#ifndef ONETHIRD
#define ONETHIRD 0.333333333333333333333333333e+00
#endif

#ifndef ONETWENTYFOURTH
#define ONETWENTYFOURTH 4.1666666666666666666666666666e-02
#endif

#ifndef ONETWOFIFTYSIXTH
#define ONETWOFIFTYSIXTH 0.00390625e+00
#endif

#ifndef ONESEVENSIXTYEIGHTH
#define ONESEVENSIXTYEIGHTH 0.001302083333333333333333333333e+00
#endif

#ifndef SQUAREROOT2
#define SQUAREROOT2  1.41421356237309505e+00
#endif

#ifndef SQUAREROOT3
#define SQUAREROOT3  1.73205080756887729e+00
#endif

#ifndef SQUAREROOT5
#define SQUAREROOT5  2.23606797749978970e+00
#endif

#ifndef SQUAREROOT6
#define SQUAREROOT6  2.44948974278317810e+00
#endif

#ifndef SQUAREROOT7
#define SQUAREROOT7  2.64575131106459059e+00
#endif

#ifndef SQUAREROOT10
#define SQUAREROOT10 3.16227766016837933e+00
#endif

#ifndef SQUAREROOT14
#define SQUAREROOT14 3.74165738677394139e+00
#endif

#ifndef SQUAREROOT15
#define SQUAREROOT15 3.87298334620741689e+00
#endif

#ifndef SQUAREROOT21
#define SQUAREROOT21 4.58257569495583983e+00
#endif

/******************************************************************/
//   FUNCTION Definition: PXShapeHierarch1d
template <typename DT> ELVIS_DEVICE int
PXShapeHierarch1d(const int porder, const DT * RESTRICT xref, DT * RESTRICT phi)
{
  DT x;

  /* Rules were originally from -1 to 1,
     and the easiest way to change the rules 
     (and cheapest operation-wise) was to do this transform */
  x = 2.0*xref[0] - 1.0;

  switch ( porder ) {

    /*--------------------------------------------------------------------*/
    /* Note: For porder > 0, we deliberately do not have break statements */
    /* to allow the case to fall through to lower p for hierarch basis    */
    /*--------------------------------------------------------------------*/
#if GEOM_USE_P5 && GEOM_USE_P4 && GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1 && GEOM_USE_P0
  case 5:
    phi[5] = sqrt(11.0)*0.125*(63.0*x*x*x*x*x - 70.0*x*x*x + 15.0*x);
#endif
#if GEOM_USE_P4 && GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1 && GEOM_USE_P0
  case 4:
    phi[4] = 0.375*(35.0*x*x*x*x - 30.0*x*x + 3.0);
#endif
#if GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1 && GEOM_USE_P0
  case 3:
    phi[3] = sqrt(7.0)*0.5*(5.0*x*x*x - 3.0*x);
#endif
#if GEOM_USE_P2 && GEOM_USE_P1 && GEOM_USE_P0
  case 2:
    phi[2] = sqrt(5.0)*0.5*(3.0*x*x - 1.0);
#endif
#if GEOM_USE_P1 && GEOM_USE_P0
  case 1:
    phi[1] = sqrt(3.0)*x;
#endif
#if GEOM_USE_P0
  case 0:
    phi[0] = 1.0;
    return 0;
#endif
  default:
    ALWAYS_PRINTF("PXShapeHierarch1d: Unknown order $d ", porder);
    return -1;
  }
} // ShapeHierarch1d


/******************************************************************/
//   FUNCTION Definition: PXLagrange1d
template <typename DT> ELVIS_DEVICE int
PXShapeUniformLagrange1d(const int porder, const DT * RESTRICT xref, DT * RESTRICT phi)
{

  DT x = xref[0];

  switch ( porder ) {
#if GEOM_USE_P0  
  case 0:
    phi[0] = 1.0;
    return 0;
#endif
#if GEOM_USE_P1
  case 1:
  {
    phi[0] = 1.0 - x;
    phi[1] = x;
    return 0;
  }
#endif
#if GEOM_USE_P2
  case 2:
  {
    DT x2=x*x;
    phi[0] = 1.0 - 3.0*x + 2.0*x2;
    phi[1] =       4.0*x - 4.0*x2;
    phi[2] =     - 1.0*x + 2.0*x2;
    return 0;
  }
#endif
#if GEOM_USE_P3
  case 3:
  {
    DT x2=x*x;
    DT x3=x2*x;
    phi[0] =  1.0 - 5.5*x +  9.0*x2 -  4.5*x3;
    phi[1] =        9.0*x - 22.5*x2 + 13.5*x3;
    phi[2] =      - 4.5*x + 18.0*x2 - 13.5*x3;
    phi[3] =        1.0*x -  4.5*x2 +  4.5*x3;
    return 0;
  }
#endif
#if GEOM_USE_P4
  case 4:
    DT x2=x*x;
    DT x3=x2*x;
    DT x4=x3*x;
    phi[0] = ( 3.0 - 25.0*x +  70.0*x2 -  80.0*x3 +  32.0*x4 )*ONETHIRD;
    phi[1] = (       48.0*x - 208.0*x2 + 288.0*x3 - 128.0*x4 )*ONETHIRD;
    phi[2] = (     - 36.0*x + 228.0*x2 - 384.0*x3 + 192.0*x4 )*ONETHIRD;
    phi[3] = (       16.0*x - 112.0*x2 + 224.0*x3 - 128.0*x4 )*ONETHIRD;
    phi[4] = (     -  3.0*x +  22.0*x2 -  48.0*x3 +  32.0*x4 )*ONETHIRD;
    return 0;
#endif
#if GEOM_USE_P5
  case 5:
    DT x2=x*x;
    DT x3=x2*x;
    DT x4=x3*x;
    DT x5=x4*x;
    phi[0] = ( 24.0 - 274.0*x + 1125.0*x2 -  2125.0*x3 +  1875.0*x4 -  625.0*x5 ) * ONETWENTYFOURTH;
    phi[1] = (        600.0*x - 3850.0*x2 +  8875.0*x3 -  8750.0*x4 + 3125.0*x5 ) * ONETWENTYFOURTH;
    phi[2] = (      - 600.0*x + 5350.0*x2 - 14750.0*x3 + 16250.0*x4 - 6250.0*x5 ) * ONETWENTYFOURTH;
    phi[3] = (        400.0*x - 3900.0*x2 + 12250.0*x3 - 15000.0*x4 + 6250.0*x5 ) * ONETWENTYFOURTH;
    phi[4] = (      - 150.0*x + 1525.0*x2 -  5125.0*x3 +  6875.0*x4 - 3125.0*x5 ) * ONETWENTYFOURTH;
    phi[5] = (         24.0*x -  250.0*x2 +   875.0*x3 -  1250.0*x4 +  625.0*x5 ) * ONETWENTYFOURTH;
    return 0;
#endif
  default:
    ALWAYS_PRINTF("PXShapeUniformLagrange1d: Unknown order $d ", porder);
    return -1;
  }
}
  

/******************************************************************/
//   FUNCTION Definition: PXShapeSpectralLagrange1d
template <typename DT> ELVIS_DEVICE int
PXShapeSpectralLagrange1d(const int porder, const DT * RESTRICT xref, DT * RESTRICT phi)
{
  DT x;
  
  x = xref[0];

  switch ( porder ) {
#if GEOM_USE_P0
  case 0:
  {
    phi[ 0] = 1.0;
    return 0;
  }
#endif
#if GEOM_USE_P1
  case 1:
  {
    phi[ 0] = 1.0-x;
    phi[ 1] = x;
    return 0;
  }
#endif
#if GEOM_USE_P2
  case 2:
  {
    DT xx = x*x;
    phi[0] = 1.0 - 3.0*x + 2.0*xx;
    phi[1] =       4.0*x - 4.0*xx;
    phi[2] =     - 1.0*x + 2.0*xx;    
    return 0;
  }
#endif
#if GEOM_USE_P3
  case 3:
  {
    DT xx = x*x;
    DT xxx = xx*x;
    phi[0] = ( 3.0 - 19.0*x + 32.0*xx - 16.0*xxx)*ONETHIRD;
    phi[1] = (       24.0*x - 56.0*xx + 32.0*xxx)*ONETHIRD;
    phi[2] = (     -  8.0*x + 40.0*xx - 32.0*xxx)*ONETHIRD;
    phi[3] = (        3.0*x - 16.0*xx + 16.0*xxx)*ONETHIRD;
    return 0;
  }
#endif
#if GEOM_USE_P4
  case 4:
    DT xx = x*x;
    DT xxx = xx*x;
    DT xxxx = xxx*x;
    
    phi[ 0] = 1.0 - 11.0*x + 34.0*xx - 40.0*xxx + 16.0*xxxx;
    phi[ 1] =  4.0*x*( 2.0+SQUAREROOT2-4.0*x )*(1.0-3.0*x+2.0*xx );
    phi[ 2] = -4.0*x + 36.0*xx - 64.0*xxx + 32.0*xxxx;
    phi[ 3] = -4.0*x*(-2.0+SQUAREROOT2+4.0*x )*(1.0-3.0*x+2.0*xx);
    phi[ 4] = -x + 10.0*xx - 24.0*xxx + 16.0*xxxx;
    return 0;
#endif
#if GEOM_USE_P5
  case 5:
    DT xx = x*x;
    DT xxx = xx*x;
    DT xxxx = xxx*x;
    DT xxxxx = xxxx*x;

    
    phi[ 0] = 1.0 - 17.0*x + 83.2*xx - 169.6*xxx + 153.6*xxxx - 51.2*xxxxx;
    phi[ 1] = -0.8*x*( 3.0+SQUAREROOT5-8.0*x)*(-5.0+25.0*x-36.0*xx+16.0*xxx);
    phi[ 2] =  0.8*x*( 5.0+SQUAREROOT5-8.0*x)*(-1.0+13.0*x-28.0*xx+16.0*xxx);
    phi[ 3] =  0.8*x*(-3.0+SQUAREROOT5+8.0*x)*(-5.0+25.0*x-36.0*xx+16.0*xxx);
    phi[ 4] = -0.8*x*(-5.0+SQUAREROOT5+8.0*x)*(-1.0+13.0*x-28.0*xx+16.0*xxx);
    phi[ 5] = x - 16.0*xx + 67.2*xxx - 102.4*xxxx + 51.2*xxxxx;
    return 0;
#endif
  default:
    ALWAYS_PRINTF("PXShapeSpectralLagrange1d: Unknown order $d ", porder);
    return -1;
  }
} 



/******************************************************************/
//   FUNCTION Definition: PXShapeHierarch2d
template <typename DT> ELVIS_DEVICE int
PXShapeHierarch2d(const int porder, const DT * RESTRICT xref, DT * RESTRICT phi)
{
  DT x, y;

  x = xref[0];
  y = xref[1];

  if (porder <= 0)
    return -1;
#if GEOM_USE_P1
  if (porder >= 1){
    phi[0] = 1.0-x-y;
    phi[1] = x;
    phi[2] = y;
    return 0;
  }
#endif
#if GEOM_USE_P2 && GEOM_USE_P1
  if (porder >= 2){
    phi[3] = -x*y*SQUAREROOT6;
    phi[4] = (-1.0+x+y)*y*SQUAREROOT6;
    phi[5] = (-1.0+x+y)*x*SQUAREROOT6;
    return 0;
  }
#endif
#if GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1  
  if (porder >= 3){
    DT xy=x*y;
    
    phi[6] = -xy*SQUAREROOT10*(y-x);
    phi[7] = -(-1.0+x+y)*y*SQUAREROOT10*(-1.0+x+2.0*y);
    phi[8] = (-1.0+x+y)*x*SQUAREROOT10*(2.0*x-1.0+y); 
    phi[9] = -6.0*(-1.0+x+y)*xy;
    return 0;
  }
#endif
#if GEOM_USE_P4 && GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1
  if (porder >= 4){
    DT yy=y*y;
    DT xy=x*y;
    DT xx=x*x;
    
    phi[10] = -xy*SQUAREROOT14*(5.0*yy-10.0*xy+5.0*xx-1.0)*0.25;
    phi[11] = (-1.0+x+y)*y*SQUAREROOT14*(4.0-10.0*x-20.0*y+5.0*xx+20.0*xy+20.0*yy)*0.25;
    phi[12] = (-1.0+x+y)*x*SQUAREROOT14*(20.0*xx-20.0*x+20.0*xy+4.0-10.0*y+5.0*yy)*0.25;
    phi[13] = -2.0*(-1.0+x+y)*xy*(y-x)*SQUAREROOT15;
    phi[14] = -2.0*(-1.0+x+y)*xy*SQUAREROOT15*(2.0*x-1.0+y);
    return 0;
  }
#endif
#if GEOM_USE_P5 && GEOM_USE_P4 && GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1
  if (porder >= 5){
    DT yy=y*y;
    DT xy=x*y;
    DT xx=x*x;

    phi[15] = -0.75*xy*SQUAREROOT2*(7.0*yy-14.0*xy+7.0*xx-3.0)*(y-x);
    phi[16] = -0.75*(-1.0+x+y)*y*SQUAREROOT2*(4.0-14.0*x-28.0*y+7.0*xx+28.0*xy+28.0*yy)*(-1.0+x+2.0*y);
    phi[17] = 0.75*(-1.0+x+y)*x*SQUAREROOT2*(28.0*xx-28.0*x+28.0*xy+4.0-14.0*y+7.0*yy)*(2.0*x-1.0+y);
    phi[18] = -(-1.0+x+y)*xy*(5.0*yy-10.0*xy+5.0*xx-1.0)*SQUAREROOT21*0.5;
    phi[19] = -10.0*(-1.0+x+y)*xy*(y-x)*(2.0*x-1.0+y);
    phi[20] = -(-1.0+x+y)*xy*SQUAREROOT21*(20.0*xx-20.0*x+20.0*xy+4.0-10.0*y+5.0*yy)*0.5;
    return 0;
  }
#endif

  ALWAYS_PRINTF("PXShapeHierarch2d: Unknown order $d ", porder);
  return -1;

} // PXShapeHierarch2d


/******************************************************************/
//   FUNCTION Definition: PXShapeLagrange2d
template <typename DT> ELVIS_DEVICE int
PXShapeLagrange2d(const int porder, const DT * RESTRICT xref, DT * RESTRICT phi)
{
  DT     x,         y;
  //DT    xx,    xy,    yy;
  //DT   xxx,   xxy,   xyy,   yyy;
  //DT  xxxx,  xxxy,  xxyy,  xyyy,  yyyy;
  //DT xxxxx, xxxxy, xxxyy, xxyyy, xyyyy;

  x = xref[0];
  y = xref[1];

  //ELVIS_PRINTF("MCG PXShapeLagrange2d: x=%f, y=%f\n", x, y);

  switch (porder) {
#if GEOM_USE_P0    
  case 0:
  {
    phi[0] = 1.0;
    return 0;
  }
#endif
#if GEOM_USE_P1
  case 1:
  {
    phi[0] = 1-x-y;
    phi[1] =   x  ;
    phi[2] =     y;
    return 0;
  }
#endif
#if GEOM_USE_P2
  case 2:
  {
    DT xx=x*x;
    DT xy=x*y;
    DT yy=y*y;
    
    phi[0] = 1.0-3.0*x-3.0*y+2.0*xx+4.0*xy+2.0*yy;
    phi[1] = -x+2.0*xx;
    phi[2] = -y+2.0*yy;
    phi[3] = 4.0*xy;
    phi[4] = 4.0*y-4.0*xy-4.0*yy;
    phi[5] = 4.0*x-4.0*xx-4.0*xy;
    return 0;
  }
#endif
#if GEOM_USE_P3
  case 3:
  {
    DT xx=x*x;
    DT xy=x*y;
    DT yy=y*y;
    DT xxx=xx*x;
    DT xxy=xx*y;
    DT xyy=xy*y;
    DT yyy=yy*y;
    
    phi[0] = 1.0-5.5*x-5.5*y +9.0*xx+18.0*xy +9.0*yy -4.5*xxx-13.5*xxy-13.5*xyy -4.5*yyy;
    phi[1] =         x       -4.5*xx                 +4.5*xxx;
    phi[2] =               y                -4.5*yy                  +4.5*yyy;
    phi[3] =                         -4.5*xy                 +13.5*xxy;
    phi[4] =                         -4.5*xy                          +13.5*xyy;
    phi[5] =          -4.5*y         +4.5*xy+18.0*yy                  -13.5*xyy-13.5*yyy;
    phi[6] =           9.0*y        -22.5*xy-22.5*yy         +13.5*xxy+27.0*xyy+13.5*yyy;
    phi[7] =     9.0*x      -22.5*xx-22.5*xy        +13.5*xxx+27.0*xxy+13.5*xyy;
    phi[8] =    -4.5*x      +18.0*xx +4.5*xy        -13.5*xxx-13.5*xxy;
    phi[9] =                         27.0*xy                 -27.0*xxy-27.0*xyy;
    return 0;
  }
#endif
#if GEOM_USE_P4	   
  case 4:
    xx=x*x;
    xy=x*y;
    yy=y*y;
    xxx=xx*x;
    xxy=xx*y;
    xyy=xy*y;
    yyy=yy*y;
    xxxx=xxx*x;
    xxxy=xxx*y;
    xxyy=xxy*y;
    xyyy=xyy*y;
    DT yyyy=yyy*y;

    phi[0] = 1.0-25.0*ONETHIRD*x-25.0*ONETHIRD*y+70.0*ONETHIRD*xx+140.0*ONETHIRD*xy+70.0*ONETHIRD*yy-
      80.0*ONETHIRD*xxx-80.0*xxy-80.0*xyy-80.0*ONETHIRD*yyy+32.0*ONETHIRD*xxxx+
      128.0*ONETHIRD*xxxy+64.0*xxyy+128.0*ONETHIRD*xyyy+32.0*ONETHIRD*yyyy; 
    phi[1] = -x+22.0*ONETHIRD*xx-16.0*xxx+32.0*ONETHIRD*xxxx; 
    phi[2] = -y+22.0*ONETHIRD*yy-16.0*yyy+32.0*ONETHIRD*yyyy; 
    phi[3] = 16.0*ONETHIRD*xy-32.0*xxy+128.0*ONETHIRD*xxxy; 
    phi[4] = 4.0*xy-16.0*xxy-16.0*xyy+64.0*xxyy; 
    phi[5] = 16.0*ONETHIRD*xy-32.0*xyy+128.0*ONETHIRD*xyyy; 
    phi[6] = 16.0*ONETHIRD*y-16.0*ONETHIRD*xy-112.0*ONETHIRD*yy+32.0*xyy+224.0*ONETHIRD*yyy-128.0*ONETHIRD*xyyy-128.0*ONETHIRD*yyyy; 
    phi[7] = -12.0*y+28.0*xy+76.0*yy-16.0*xxy-144.0*xyy-128.0*yyy+64.0*xxyy+128.0*xyyy+64.0*yyyy; 
    phi[8] = 16.0*y-208.0*ONETHIRD*xy-208.0*ONETHIRD*yy+96.0*xxy+192.0*xyy+96.0*yyy-128.0*ONETHIRD*xxxy-
      128.0*xxyy-128.0*xyyy-128.0*ONETHIRD*yyyy; 
    phi[9] = 16.0*x-208.0*ONETHIRD*xx-208.0*ONETHIRD*xy+96.0*xxx+192.0*xxy+96.0*xyy-128.0*ONETHIRD*xxxx-
      128.0*xxxy-128.0*xxyy-128.0*ONETHIRD*xyyy; 
    phi[10] = -12.0*x+76.0*xx+28.0*xy-128.0*xxx-144.0*xxy-16.0*xyy+64.0*xxxx+128.0*xxxy+64.0*xxyy; 
    phi[11] = 16.0*ONETHIRD*x-112.0*ONETHIRD*xx-16.0*ONETHIRD*xy+224.0*ONETHIRD*xxx+32.0*xxy-128.0*ONETHIRD*xxxx-128.0*ONETHIRD*xxxy; 
    phi[12] = 96.0*xy-224.0*xxy-224.0*xyy+128.0*xxxy+256.0*xxyy+128.0*xyyy; 
    phi[13] = -32.0*xy+160.0*xxy+32.0*xyy-128.0*xxxy-128.0*xxyy; 
    phi[14] = -32.0*xy+32.0*xxy+160.0*xyy-128.0*xxyy-128.0*xyyy;
    return 0;
#endif

#if GEOM_USE_P5
  case 5:
    xx=x*x;
    xy=x*y;
    yy=y*y;
    xxx=xx*x;
    xxy=xx*y;
    xyy=xy*y;
    yyy=yy*y;
    xxxx=xxx*x;
    xxxy=xxx*y;
    xxyy=xxy*y;
    xyyy=xyy*y;
    yyyy=yyy*y;
    xxxxx=xxxx*x;
    xxxxy=xxxx*y;
    xxxyy=xxxy*y;
    xxyyy=xxyy*y;
    xyyyy=xyyy*y;
    DT yyyyy=yyyy*y;

    phi[0]  = 1.0-137.0/12.0*x-137.0/12.0*y+375.0/8.0*xx+375.0*0.25*xy+375.0/8.0*yy-
                 2125.0/24.0*xxx-2125.0/8.0*xxy-2125.0/8.0*xyy-2125.0/24.0*yyy+
                 625.0/8.0*xxxx+312.5*xxxy+1875.0*0.25*xxyy+312.5*xyyy+
                 625.0/8.0*yyyy-625.0/24.0*xxxxx-3125.0/24.0*xxxxy-3125.0/12.0*xxxyy-
                 3125.0/12.0*xxyyy-3125.0/24.0*xyyyy-625.0/24.0*yyyyy;
    phi[1]  = x-125.0/12.0*xx+875.0/24.0*xxx-625.0/12.0*xxxx+625.0/24.0*xxxxx;
    phi[2]  = y-125.0/12.0*yy+875.0/24.0*yyy-625.0/12.0*yyyy+625.0/24.0*yyyyy;
    phi[3]  = -25.0*0.25*xy+1375.0/24.0*xxy-625.0*0.25*xxxy+3125.0/24.0*xxxxy;
    phi[4]  = -25.0*ONESIXTH*xy+125.0*0.25*xxy+125.0*ONESIXTH*xyy-625.0/12.0*xxxy-625.0*0.25*xxyy+3125.0/12.0*xxxyy;
    phi[5]  = -25.0*ONESIXTH*xy+125.0*ONESIXTH*xxy+125.0*0.25*xyy-625.0*0.25*xxyy-625.0/12.0*xyyy+3125.0/12.0*xxyyy;
    phi[6]  = -25.0*0.25*xy+1375.0/24.0*xyy-625.0*0.25*xyyy+3125.0/24.0*xyyyy;
    phi[7]  = -25.0*0.25*y+25.0*0.25*xy+1525.0/24.0*yy-1375.0/24.0*xyy-5125.0/24.0*yyy+
                625.0*0.25*xyyy+6875.0/24.0*yyyy-3125.0/24.0*xyyyy-3125.0/24.0*yyyyy;
    phi[8]  = 50.0*ONETHIRD*y-37.5*xy-162.5*yy+125.0*ONESIXTH*xxy+3875.0/12.0*xyy+6125.0/12.0*yyy-
                625.0*0.25*xxyy-3125.0*0.25*xyyy-625.0*yyyy+3125.0/12.0*xxyyy+3125.0*ONESIXTH*xyyyy+3125.0/12.0*yyyyy;
    phi[9]  = -25.0*y+1175.0/12.0*xy+2675.0/12.0*yy-125.0*xxy-8875.0/12.0*xyy-7375.0/12.0*yyy+
                625.0/12.0*xxxy+3125.0*0.25*xxyy+5625.0*0.25*xyyy+8125.0/12.0*yyyy-3125.0/12.0*xxxyy-
                3125.0*0.25*xxyyy-3125.0*0.25*xyyyy-3125.0/12.0*yyyyy; 
    phi[10] = 25.0*y-1925.0/12.0*xy-1925.0/12.0*yy+8875.0/24.0*xxy+8875.0/12.0*xyy+8875.0/24.0*yyy-
                4375.0/12.0*xxxy-4375.0*0.25*xxyy-4375.0*0.25*xyyy-4375.0/12.0*yyyy+3125.0/24.0*xxxxy+
                3125.0*ONESIXTH*xxxyy+3125.0*0.25*xxyyy+3125.0*ONESIXTH*xyyyy+3125.0/24.0*yyyyy;
    phi[11] = 25.0*x-1925.0/12.0*xx-1925.0/12.0*xy+8875.0/24.0*xxx+8875.0/12.0*xxy+8875.0/24.0*xyy-
                4375.0/12.0*xxxx-4375.0*0.25*xxxy-4375.0*0.25*xxyy-4375.0/12.0*xyyy+3125.0/24.0*xxxxx+
                3125.0*ONESIXTH*xxxxy+3125.0*0.25*xxxyy+3125.0*ONESIXTH*xxyyy+3125.0/24.0*xyyyy;
    phi[12] = -25.0*x+2675.0/12.0*xx+1175.0/12.0*xy-7375.0/12.0*xxx-8875.0/12.0*xxy-125.0*xyy+
                 8125.0/12.0*xxxx+5625.0*0.25*xxxy+3125.0*0.25*xxyy+625.0/12.0*xyyy-3125.0/12.0*xxxxx-
                 3125.0*0.25*xxxxy-3125.0*0.25*xxxyy-3125.0/12.0*xxyyy;
    phi[13] = 50.0*ONETHIRD*x-162.5*xx-37.5*xy+6125.0/12.0*xxx+3875.0/12.0*xxy+125.0*ONESIXTH*xyy-625.0*xxxx-
                 3125.0*0.25*xxxy-625.0*0.25*xxyy+3125.0/12.0*xxxxx+3125.0*ONESIXTH*xxxxy+3125.0/12.0*xxxyy;
    phi[14] = -25.0*0.25*x+1525.0/24.0*xx+25.0*0.25*xy-5125.0/24.0*xxx-1375.0/24.0*xxy+6875.0/24.0*xxxx+
                 625.0*0.25*xxxy-3125.0/24.0*xxxxx-3125.0/24.0*xxxxy;
    phi[15] = 250.0*xy-5875.0*ONESIXTH*xxy-5875.0*ONESIXTH*xyy+1250.0*xxxy+2500.0*xxyy+1250.0*xyyy-3125.0*ONESIXTH*xxxxy-
                 1562.5*xxxyy-1562.5*xxyyy-3125.0*ONESIXTH*xyyyy;
    phi[16] = -125.0*xy+3625.0*0.25*xxy+1125.0*0.25*xyy-1562.5*xxxy-6875.0*0.25*xxyy-625.0*0.25*xyyy+
                 3125.0*0.25*xxxxy+1562.5*xxxyy+3125.0*0.25*xxyyy;
    phi[17] = 125.0*ONETHIRD*xy-2125.0*ONESIXTH*xxy-125.0*ONETHIRD*xyy+2500.0*ONETHIRD*xxxy+312.5*xxyy-3125.0*ONESIXTH*xxxxy-3125.0*ONESIXTH*xxxyy;
    phi[18] = -125.0*xy+1125.0*0.25*xxy+3625.0*0.25*xyy-625.0*0.25*xxxy-6875.0*0.25*xxyy-1562.5*xyyy+
                 3125.0*0.25*xxxyy+1562.5*xxyyy+3125.0*0.25*xyyyy;
    phi[19] = 125.0*0.25*xy-187.5*xxy-187.5*xyy+625.0*0.25*xxxy+4375.0*0.25*xxyy+
                 625.0*0.25*xyyy-3125.0*0.25*xxxyy-3125.0*0.25*xxyyy;
    phi[20] = 125.0*ONETHIRD*xy-125.0*ONETHIRD*xxy-2125.0*ONESIXTH*xyy+312.5*xxyy+2500.0*ONETHIRD*xyyy-
                 3125.0*ONESIXTH*xxyyy-3125.0*ONESIXTH*xyyyy;
    return 0;
#endif
  default:
    ALWAYS_PRINTF("PXShapeLagrange2d: Unknown order $d ", porder);
    return -1;
  }
}

/******************************************************************/
//   FUNCTION Definition: PXShapeQuadUniformLagrange2d
template <typename DT> ELVIS_DEVICE int
PXShapeQuadUniformLagrange2d(const int porder, const DT * RESTRICT xref, DT * RESTRICT phi)
{
  int ierr;
  int i,j;
  DT xi[2];
  DT eta[2];
  DT phi_i[6]; // 1d uniform lagrange basis functions
  DT phi_j[6]; // 1d uniform lagrange basis functions

  /* move coordinates */
  xi[0]  = xref[0];
  eta[0] = xref[1];

  ierr = PXShapeUniformLagrange1d<DT>( porder, xi , phi_i );
  if (ierr!=0) return ierr;
  ierr = PXShapeUniformLagrange1d<DT>( porder, eta, phi_j );
  if (ierr!=0) return ierr;

  for (j=0; j<(porder+1); j++){
    for (i=0; i<(porder+1); i++){
      phi[j*(porder+1)+i] = phi_i[i]*phi_j[j];
    }
  }
  
  return 0;
}

/******************************************************************/
//   FUNCTION Definition: PXShapeQuadSpectralLagrange2d
template <typename DT> ELVIS_DEVICE int
PXShapeQuadSpectralLagrange2d(const int porder, const DT * RESTRICT xref, DT * RESTRICT phi)
{
  int ierr;
  int i,j;
  DT xi[1];
  DT eta[1];
  DT phi_i[6]; // 1d spectral lagrange basis functions
  DT phi_j[6]; // 1d spectral lagrange basis functions

  /* move coordinates */
  xi[0]  = xref[0];
  eta[0] = xref[1];

  ierr = PXShapeSpectralLagrange1d<DT>( porder, xi , phi_i );
  if (ierr!=0) return ierr;
  ierr = PXShapeSpectralLagrange1d<DT>( porder, eta, phi_j );
  if (ierr!=0) return ierr;

  for (j=0; j<(porder+1); j++){
    for (i=0; i<(porder+1); i++){
      phi[j*(porder+1)+i] = phi_i[i]*phi_j[j];
    }
  }
  
  return 0;
}

/******************************************************************/
//   FUNCTION Definition: PXShapeHierarch3d
template <typename DT> ELVIS_DEVICE int
PXShapeHierarch3d(const int porder, const DT * RESTRICT xref, DT * RESTRICT phi)
{
  DT x , y , z ;

  x = xref[0];
  y = xref[1];
  z = xref[2];

  if (porder <= 0)
    return -1;
#if GEOM_USE_P1
  if (porder >= 1){
    phi[0] = 1.0-x-y-z;
    phi[1] = x;
    phi[2] = y;
    phi[3] = z;
    return 0;
  }
#endif
#if GEOM_USE_P2 && GEOM_USE_P1
  if (porder >= 2){
    phi[4] = -y*z*SQUAREROOT6;
    phi[5] = -x*z*SQUAREROOT6;
    phi[6] = (-1.0+x+y+z)*z*SQUAREROOT6;
    phi[7] = -x*y*SQUAREROOT6;
    phi[8] = (-1.0+x+y+z)*y*SQUAREROOT6;
    phi[9] = (-1.0+x+y+z)*x*SQUAREROOT6;
    return 0;
  }
#endif
#if GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1  
  if (porder >= 3){
    phi[10] = y*z*SQUAREROOT10*(-z+y);
    phi[11] = x*z*SQUAREROOT10*(x-z);
    phi[12] = (-1.0+x+y+z)*z*SQUAREROOT10*(2.0*z-1.0+x+y);
    phi[13] = x*y*SQUAREROOT10*(-y+x);
    phi[14] = (-1.0+x+y+z)*y*SQUAREROOT10*(2.0*y-1.0+x+z);
    phi[15] = (-1.0+x+y+z)*x*SQUAREROOT10*(2.0*x-1.0+y+z);
    phi[16] = 6.0*x*y*z;
    phi[17] = -6.0*(-1.0+x+y+z)*y*z;
    phi[18] = -6.0*(-1.0+x+y+z)*x*z;
    phi[19] = -6.0*(-1.0+x+y+z)*x*y;
    return 0;
  }
#endif
#if GEOM_USE_P4 && GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1  
  if (porder >= 4){
    phi[20] = -y*z*SQUAREROOT14*(5.0*z*z-10.0*y*z+5.0*y*y-1.0)*0.25;
    phi[21] = -x*z*SQUAREROOT14*(5.0*z*z-10.0*x*z+5.0*x*x-1.0)*0.25;
    phi[22] = (-1.0+x+y+z)*z*SQUAREROOT14*(20.0*z*z-20.0*z+20.0*x*z+20.0*y*z+4.0-10.0*x-10.0*y+5.0*x*x+10.0*x*y+5.0*y*y)*0.25;
    phi[23] = -x*y*SQUAREROOT14*(5.0*y*y-10.0*x*y+5.0*x*x-1.0)*0.25;
    phi[24] = (-1.0+x+y+z)*y*SQUAREROOT14*(20.0*y*y-20.0*y+20.0*x*y+20.0*y*z+4.0-10.0*x-10.0*z+5.0*x*x+10.0*x*z+5.0*z*z)*0.25;
    phi[25] = (-1.0+x+y+z)*x*SQUAREROOT14*(20.0*x*x-20.0*x+20.0*x*y+20.0*x*z+4.0-10.0*y-10.0*z+5.0*y*y+10.0*y*z+5.0*z*z)*0.25;
    phi[26] = 2.0*x*y*z*SQUAREROOT15*(x-z);
    phi[27] = 2.0*(-1.0+x+y+z)*y*z*SQUAREROOT15*(2.0*z-1.0+x+y);
    phi[28] = 2.0*(-1.0+x+y+z)*x*z*SQUAREROOT15*(2.0*z-1.0+x+y);
    phi[29] = 2.0*(-1.0+x+y+z)*x*y*SQUAREROOT15*(2.0*y-1.0+x+z);
    phi[30] = -2.0*x*y*z*(-y+x)*SQUAREROOT15;
    phi[31] = -2.0*(-1.0+x+y+z)*y*z*(2.0*y-1.0+x+z)*SQUAREROOT15;
    phi[32] = -2.0*(-1.0+x+y+z)*x*z*(2.0*x-1.0+y+z)*SQUAREROOT15;
    phi[33] = -2.0*(-1.0+x+y+z)*x*y*(2.0*x-1.0+y+z)*SQUAREROOT15;
    phi[34] = 6.0*(-1.0+x+y+z)*x*y*SQUAREROOT6*z;
    return 0;
  }
#endif
#if GEOM_USE_P5 && GEOM_USE_P4 && GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1  
  if (porder >= 5){
    phi[35] = 0.75*y*z*SQUAREROOT2*(7.0*z*z-14.0*y*z+7.0*y*y-3.0)*(-z+y);
    phi[36] = 0.75*x*z*SQUAREROOT2*(7.0*z*z-14.0*x*z+7.0*x*x-3.0)*(x-z);
    phi[37] = 0.75*(-1.0+x+y+z)*z*SQUAREROOT2*(28.0*z*z-28.0*z+28.0*x*z+28.0*y*z+4.0-14.0*x-14.0*y+7.0*x*x+14.0*x*y+7.0*y*y)*(2.0*z-1.0+x+y);
    phi[38] = 0.75*x*y*SQUAREROOT2*(7.0*y*y-14.0*x*y+7.0*x*x-3.0)*(-y+x);
    phi[39] = 0.75*(-1.0+x+y+z)*y*SQUAREROOT2*(28.0*y*y-28.0*y+28.0*x*y+28.0*y*z+4.0-14.0*x-14.0*z+7.0*x*x+14.0*x*z+7.0*z*z)*(2.0*y-1.0+x+z);
    phi[40] = 0.75*(-1.0+x+y+z)*x*SQUAREROOT2*(28.0*x*x-28.0*x+28.0*x*y+28.0*x*z+4.0-14.0*y-14.0*z+7.0*y*y+14.0*y*z+7.0*z*z)*(2.0*x-1.0+y+z);
    phi[41] = x*y*z*SQUAREROOT21*(5.0*z*z-10.0*x*z+5.0*x*x-1.0)*0.5;
    phi[42] = -(-1.0+x+y+z)*y*z*SQUAREROOT21*(20.0*z*z-20.0*z+20.0*x*z+20.0*y*z+4.0-10.0*x-10.0*y+5.0*x*x+10.0*x*y+5.0*y*y)*0.5;
    phi[43] = -(-1.0+x+y+z)*x*z*SQUAREROOT21*(20.0*z*z-20.0*z+20.0*x*z+20.0*y*z+4.0-10.0*x-10.0*y+5.0*x*x+10.0*x*y+5.0*y*y)*0.5;
    phi[44] = -(-1.0+x+y+z)*x*y*SQUAREROOT21*(20.0*y*y-20.0*y+20.0*x*y+20.0*y*z+4.0-10.0*x-10.0*z+5.0*x*x+10.0*x*z+5.0*z*z)*0.5;
    phi[45] = -10.0*x*y*z*(-y+x)*(x-z);
    phi[46] = 10.0*(-1.0+x+y+z)*y*z*(2.0*y-1.0+x+z)*(2.0*z-1.0+x+y);
    phi[47] = 10.0*(-1.0+x+y+z)*x*z*(2.0*x-1.0+y+z)*(2.0*z-1.0+x+y);
    phi[48] = 10.0*(-1.0+x+y+z)*x*y*(2.0*x-1.0+y+z)*(2.0*y-1.0+x+z);
    phi[49] = x*y*z*(5.0*y*y-10.0*x*y+5.0*x*x-1.0)*SQUAREROOT21*0.5;
    phi[50] = -(-1.0+x+y+z)*y*z*(20.0*y*y-20.0*y+20.0*x*y+20.0*y*z+4.0-10.0*x-10.0*z+5.0*x*x+10.0*x*z+5.0*z*z)*SQUAREROOT21*0.5;
    phi[51] = -(-1.0+x+y+z)*x*z*(20.0*x*x-20.0*x+20.0*x*y+20.0*x*z+4.0-10.0*y-10.0*z+5.0*y*y+10.0*y*z+5.0*z*z)*SQUAREROOT21*0.5;
    phi[52] = -(-1.0+x+y+z)*x*y*(20.0*x*x-20.0*x+20.0*x*y+20.0*x*z+4.0-10.0*y-10.0*z+5.0*y*y+10.0*y*z+5.0*z*z)*SQUAREROOT21*0.5;
    phi[53] = -6.0*SQUAREROOT10*(-z+y)*(-1.0+x+y+z)*x*y*z;
    phi[54] = -6.0*SQUAREROOT10*(-y+x)*(-1.0+x+y+z)*x*y*z;
    phi[55] = -6.0*SQUAREROOT10*(2.0*x-1.0+y+z)*(-1.0+x+y+z)*x*y*z;
    return 0;
  }
#endif

  //if (porder >= 6)
  return -1;
} // PXShapeHierarch3d


/******************************************************************/
//   FUNCTION Definition: PXShapeLagrange3d
template <typename DT> ELVIS_DEVICE int
PXShapeLagrange3d(const int porder, const DT * RESTRICT xref, DT * RESTRICT phi)
{
  DT x, y, z;
  //DT MapleGenVar1, x3, x4, y3, y4, z3, z4;

  x = xref[0];
  y = xref[1];
  z = xref[2];

  switch (porder){
#if GEOM_USE_P0    
  case 0:
    phi[0] = 1.0;
    return 0;
#endif
#if GEOM_USE_P1
  case 1:
    phi[0] = 1.0-z-y-x;
    phi[1] = x;
    phi[2] = y;
    phi[3] = z;
    
    return 0;
#endif
#if GEOM_USE_P2
  case 2:
    phi[0] = 1.0-3.0*z-3.0*y-3.0*x+2.0*z*z+4.0*y*z+4.0*x*z+2.0*y*y+4.0*x*y+2.0*x*x;
    phi[1] = 4.0*x-4.0*x*z-4.0*x*y-4.0*x*x;
    phi[2] = -x+2.0*x*x;
    phi[3] = 4.0*y-4.0*y*z-4.0*y*y-4.0*x*y;
    phi[4] = 4.0*x*y;
    phi[5] = -y+2.0*y*y;
    phi[6] = 4.0*z-4.0*z*z-4.0*y*z-4.0*x*z;
    phi[7] = 4.0*x*z;
    phi[8] = 4.0*y*z;
    phi[9] = -z+2.0*z*z;

    return 0;
#endif
#if GEOM_USE_P3
  case 3:

    phi[0] = 1.0-5.5*z-5.5*y-5.5*x+9.0*z*z+18.0*z*y+18.0*z*x+9.0*y*y+18.0*y*x+9.0*x*x-4.5*z*z*z-13.5*z*z*y-13.5*z*z*x-13.5*z*y*y-27.0*z*y*x-13.5*z*x*x-4.5*y*y*y-13.5*y*y*x-13.5*y*x*x-4.5*x*x*x;
    phi[1] = 9.0*x-22.5*z*x-22.5*y*x-22.5*x*x+13.5*z*z*x+27.0*z*y*x+27.0*z*x*x+13.5*y*y*x+27.0*y*x*x+13.5*x*x*x;
    phi[2] = -4.5*x+4.5*z*x+4.5*y*x+18.0*x*x-13.5*z*x*x-13.5*y*x*x-13.5*x*x*x;
    phi[3] = x-4.5*x*x+4.5*x*x*x;
    phi[4] = 9.0*y-22.5*z*y-22.5*y*y-22.5*y*x+13.5*z*z*y+27.0*z*y*y+27.0*z*y*x+13.5*y*y*y+27.0*y*y*x+13.5*y*x*x;
    phi[5] = 27.0*y*x-27.0*z*y*x-27.0*y*y*x-27.0*y*x*x;
    phi[6] = -4.5*y*x+13.5*y*x*x;
    phi[7] = -4.5*y+4.5*z*y+18.0*y*y+4.5*y*x-13.5*z*y*y-13.5*y*y*y-13.5*y*y*x;
    phi[8] = -4.5*y*x+13.5*y*y*x;
    phi[9] = y-4.5*y*y+4.5*y*y*y;
    phi[10] = 9.0*z-22.5*z*z-22.5*z*y-22.5*z*x+13.5*z*z*z+27.0*z*z*y+27.0*z*z*x+13.5*z*y*y+27.0*z*y*x+13.5*z*x*x;
    phi[11] = 27.0*z*x-27.0*z*z*x-27.0*z*y*x-27.0*z*x*x;
    phi[12] = -4.5*z*x+13.5*z*x*x;
    phi[13] = 27.0*z*y-27.0*z*z*y-27.0*z*y*y-27.0*z*y*x;
    phi[14] = 27.0*z*y*x;
    phi[15] = -4.5*z*y+13.5*z*y*y;
    phi[16] = -4.5*z+18.0*z*z+4.5*z*y+4.5*z*x-13.5*z*z*z-13.5*z*z*y-13.5*z*z*x;
    phi[17] = -4.5*z*x+13.5*z*z*x;
    phi[18] = -4.5*z*y+13.5*z*z*y;
    phi[19] = z-4.5*z*z+4.5*z*z*z;

    return 0;
#endif
#if GEOM_USE_P4
  case 4:

    phi[0] = 1.0+128.0*z*y*x*x+128.0*z*z*y*x-160.0*z*y*x+128.0*ONETHIRD*z*z*z*y-25.0*ONETHIRD*x-25.0*ONETHIRD*y+128.0*z*y*y*x-80.0*y*x*x-80.0*z*y*y-80.0*y*y*x+140.0*ONETHIRD*y*x-80.0*z*z*y-25.0*ONETHIRD*z+32.0*ONETHIRD*y*y*y*y+32.0*ONETHIRD*x*x*x*x+32.0*ONETHIRD*z*z*z*z+70.0*ONETHIRD*z*z+70.0*ONETHIRD*y*y-80.0*ONETHIRD*z*z*z-80.0*ONETHIRD*y*y*y+140.0*ONETHIRD*z*y+70.0*ONETHIRD*x*x-80.0*ONETHIRD*x*x*x+128.0*ONETHIRD*y*x*x*x+64.0*y*y*x*x+128.0*ONETHIRD*y*y*y*x+64.0*z*z*x*x+128.0*ONETHIRD*z*x*x*x+128.0*ONETHIRD*z*z*z*x-80.0*z*z*x+140.0*ONETHIRD*z*x+128.0*ONETHIRD*z*y*y*y-80.0*z*x*x+64.0*z*z*y*y;
    phi[1] = 192.0*z*y*x+16.0*x-128.0*z*y*y*x-128.0*z*z*y*x-208.0*ONETHIRD*y*x-208.0*ONETHIRD*x*x-128.0*z*x*x*x-128.0*ONETHIRD*x*x*x*x-208.0*ONETHIRD*z*x-128.0*z*z*x*x-256.0*z*y*x*x+96.0*x*x*x-128.0*ONETHIRD*z*z*z*x+96.0*z*z*x-128.0*ONETHIRD*y*y*y*x-128.0*y*y*x*x-128.0*y*x*x*x+192.0*y*x*x+96.0*y*y*x+192.0*z*x*x;
    phi[2] = -32.0*z*y*x-12.0*x+28.0*y*x+76.0*x*x+128.0*z*x*x*x+64.0*x*x*x*x+28.0*z*x+64.0*z*z*x*x+128.0*z*y*x*x-128.0*x*x*x-16.0*z*z*x+64.0*y*y*x*x+128.0*y*x*x*x-144.0*y*x*x-16.0*y*y*x-144.0*z*x*x;
    phi[3] = 16.0*ONETHIRD*x-16.0*ONETHIRD*y*x-112.0*ONETHIRD*x*x-128.0*ONETHIRD*z*x*x*x-128.0*ONETHIRD*x*x*x*x-16.0*ONETHIRD*z*x+224.0*ONETHIRD*x*x*x-128.0*ONETHIRD*y*x*x*x+32.0*y*x*x+32.0*z*x*x;
    phi[4] = -x+22.0*ONETHIRD*x*x+32.0*ONETHIRD*x*x*x*x-16.0*x*x*x;
    phi[5] = 192.0*z*y*x+16.0*y-256.0*z*y*y*x-128.0*z*z*y*x-208.0*ONETHIRD*y*x-128.0*z*y*y*y-208.0*ONETHIRD*y*y+96.0*y*y*y-128.0*ONETHIRD*y*y*y*y+96.0*z*z*y-128.0*z*z*y*y-128.0*z*y*x*x+192.0*z*y*y-128.0*y*y*y*x-128.0*y*y*x*x-128.0*ONETHIRD*y*x*x*x-128.0*ONETHIRD*z*z*z*y+96.0*y*x*x+192.0*y*y*x-208.0*ONETHIRD*z*y;
    phi[6] = -224.0*z*y*x+256.0*z*y*y*x+128.0*z*z*y*x+96.0*y*x+256.0*z*y*x*x+128.0*y*y*y*x+256.0*y*y*x*x+128.0*y*x*x*x-224.0*y*x*x-224.0*y*y*x;
    phi[7] = 32.0*z*y*x-32.0*y*x-128.0*z*y*x*x-128.0*y*y*x*x-128.0*y*x*x*x+160.0*y*x*x+32.0*y*y*x;
    phi[8] = 16.0*ONETHIRD*y*x+128.0*ONETHIRD*y*x*x*x-32.0*y*x*x;
    phi[9] = -32.0*z*y*x-12.0*y+128.0*z*y*y*x+28.0*y*x+128.0*z*y*y*y+76.0*y*y-128.0*y*y*y+64.0*y*y*y*y-16.0*z*z*y+64.0*z*z*y*y-144.0*z*y*y+128.0*y*y*y*x+64.0*y*y*x*x-16.0*y*x*x-144.0*y*y*x+28.0*z*y;
    phi[10] = 32.0*z*y*x-128.0*z*y*y*x-32.0*y*x-128.0*y*y*y*x-128.0*y*y*x*x+32.0*y*x*x+160.0*y*y*x;
    phi[11] = 4.0*y*x+64.0*y*y*x*x-16.0*y*x*x-16.0*y*y*x;
    phi[12] = 16.0*ONETHIRD*y-16.0*ONETHIRD*y*x-128.0*ONETHIRD*z*y*y*y-112.0*ONETHIRD*y*y+224.0*ONETHIRD*y*y*y-128.0*ONETHIRD*y*y*y*y+32.0*z*y*y-128.0*ONETHIRD*y*y*y*x+32.0*y*y*x-16.0*ONETHIRD*z*y;
    phi[13] = 16.0*ONETHIRD*y*x+128.0*ONETHIRD*y*y*y*x-32.0*y*y*x;
    phi[14] = -y+22.0*ONETHIRD*y*y-16.0*y*y*y+32.0*ONETHIRD*y*y*y*y;
    phi[15] = -128.0*z*y*x*x-256.0*z*z*y*x+192.0*z*y*x-128.0*z*y*y*x+16.0*z-128.0*ONETHIRD*z*y*y*y+192.0*z*z*y-128.0*z*z*y*y+96.0*z*y*y-128.0*z*z*x*x-128.0*ONETHIRD*z*x*x*x-128.0*z*z*z*y-128.0*z*z*z*x+192.0*z*z*x-208.0*ONETHIRD*z*z+96.0*z*z*z-208.0*ONETHIRD*z*x+96.0*z*x*x-128.0*ONETHIRD*z*z*z*z-208.0*ONETHIRD*z*y;
    phi[16] = 256.0*z*y*x*x+256.0*z*z*y*x-224.0*z*y*x+128.0*z*y*y*x+256.0*z*z*x*x+128.0*z*x*x*x+128.0*z*z*z*x-224.0*z*z*x+96.0*z*x-224.0*z*x*x;
    phi[17] = -128.0*z*y*x*x+32.0*z*y*x-128.0*z*z*x*x-128.0*z*x*x*x+32.0*z*z*x-32.0*z*x+160.0*z*x*x;
    phi[18] = 128.0*ONETHIRD*z*x*x*x+16.0*ONETHIRD*z*x-32.0*z*x*x;
    phi[19] = 128.0*z*y*x*x+256.0*z*z*y*x-224.0*z*y*x+128.0*z*z*z*y+256.0*z*y*y*x-224.0*z*y*y-224.0*z*z*y+96.0*z*y+128.0*z*y*y*y+256.0*z*z*y*y;
    phi[20] = -256.0*z*y*x*x-256.0*z*z*y*x+256.0*z*y*x-256.0*z*y*y*x;
    phi[21] = 128.0*z*y*x*x-32.0*z*y*x;
    phi[22] = 32.0*z*y*x-128.0*z*y*y*x+160.0*z*y*y+32.0*z*z*y-32.0*z*y-128.0*z*y*y*y-128.0*z*z*y*y;
    phi[23] = -32.0*z*y*x+128.0*z*y*y*x;
    phi[24] = -32.0*z*y*y+16.0*ONETHIRD*z*y+128.0*ONETHIRD*z*y*y*y;
    phi[25] = 128.0*z*z*y*x-32.0*z*y*x+128.0*z*z*z*y-16.0*z*y*y-144.0*z*z*y-12.0*z+64.0*z*z*z*z+76.0*z*z-128.0*z*z*z+28.0*z*y+64.0*z*z*x*x+128.0*z*z*z*x-144.0*z*z*x+28.0*z*x-16.0*z*x*x+64.0*z*z*y*y;
    phi[26] = -128.0*z*z*y*x+32.0*z*y*x-128.0*z*z*x*x-128.0*z*z*z*x+160.0*z*z*x-32.0*z*x+32.0*z*x*x;
    phi[27] = 64.0*z*z*x*x-16.0*z*z*x+4.0*z*x-16.0*z*x*x;
    phi[28] = -128.0*z*z*y*x+32.0*z*y*x-128.0*z*z*z*y+32.0*z*y*y+160.0*z*z*y-32.0*z*y-128.0*z*z*y*y;
    phi[29] = 128.0*z*z*y*x-32.0*z*y*x;
    phi[30] = -16.0*z*y*y-16.0*z*z*y+4.0*z*y+64.0*z*z*y*y;
    phi[31] = -128.0*ONETHIRD*z*z*z*y+32.0*z*z*y+16.0*ONETHIRD*z-128.0*ONETHIRD*z*z*z*z-112.0*ONETHIRD*z*z+224.0*ONETHIRD*z*z*z-16.0*ONETHIRD*z*y-128.0*ONETHIRD*z*z*z*x+32.0*z*z*x-16.0*ONETHIRD*z*x;
    phi[32] = 128.0*ONETHIRD*z*z*z*x-32.0*z*z*x+16.0*ONETHIRD*z*x;
    phi[33] = 128.0*ONETHIRD*z*z*z*y-32.0*z*z*y+16.0*ONETHIRD*z*y;
    phi[34] = -z+32.0*ONETHIRD*z*z*z*z+22.0*ONETHIRD*z*z-16.0*z*z*z;
  
    return 0;
#endif
#if GEOM_USE_P5
  case 5:
    DT x4 = x*x*x*x;
    DT x3 = x*x*x;
    DT y4 = y*y*y*y;
    DT y3 = y*y*y;
    DT z4 = z*z*z*z;
    DT z3 = z*z*z;
    
    DT MapleGenVar1 = 1.0-3125.0*ONESIXTH*z*y*x*x*x-137.0/12.0*x-137.0/12.0*y-3125.0*ONESIXTH*z*y*y*y*x-2125.0*0.25*z*y*x+1875.0*0.25*y*y*x*x-137.0/12.0*z-3125.0/24.0*z*z*z*z*y-3125.0/12.0*z*z*z*y*y+375.0/8.0*x*x+312.5*y*y*y*x+312.5*z*x*x*x+1875.0*0.25*z*z*y*y+1875.0*0.25*z*z*x*x+312.5*z*y*y*y+312.5*z*z*z*y+375.0*0.25*z*y-2125.0/8.0*z*z*y-2125.0/24.0*x*x*x-3125.0/24.0*z*x*x*x*x-3125.0/12.0*y*y*x*x*x-3125.0/24.0*z*y*y*y*y-2125.0/8.0*y*y*x-3125.0/12.0*z*z*x*x*x-2125.0/8.0*y*x*x-2125.0/8.0*z*x*x-3125.0/12.0*z*z*y*y*y;
    phi[0] = -2125.0/8.0*z*y*y-3125.0/24.0*z*z*z*z*x+312.5*z*z*z*x-3125.0*0.25*z*y*y*x*x-3125.0*0.25*z*z*y*x*x-3125.0/12.0*z*z*z*x*x+937.5*z*y*x*x-3125.0*0.25*z*z*y*y*x+MapleGenVar1-2125.0/8.0*z*z*x+312.5*y*x*x*x-3125.0/24.0*y*x*x*x*x+937.5*z*z*y*x+937.5*z*y*y*x+375.0*0.25*z*x-3125.0/12.0*y*y*y*x*x+375.0/8.0*z*z+375.0/8.0*y*y-2125.0/24.0*z*z*z-2125.0/24.0*y*y*y+625.0/8.0*z*z*z*z+625.0/8.0*y*y*y*y-625.0/24.0*z*z*z*z*z-625.0/24.0*y*y*y*y*y-625.0/24.0*x*x*x*x*x-3125.0*ONESIXTH*z*z*z*y*x+375.0*0.25*y*x+625.0/8.0*x*x*x*x-3125.0/24.0*y*y*y*y*x;
    phi[1] = 1562.5*z*y*x*x*x+25.0*x+3125.0*ONESIXTH*z*y*y*y*x+8875.0/12.0*z*y*x-4375.0*0.25*y*y*x*x-1925.0/12.0*x*x-4375.0/12.0*y*y*y*x-4375.0*0.25*z*x*x*x-4375.0*0.25*z*z*x*x+8875.0/24.0*x*x*x+3125.0*ONESIXTH*z*x*x*x*x+3125.0*0.25*y*y*x*x*x+8875.0/24.0*y*y*x+3125.0*0.25*z*z*x*x*x+8875.0/12.0*y*x*x+8875.0/12.0*z*x*x+3125.0/24.0*z*z*z*z*x-4375.0/12.0*z*z*z*x+1562.5*z*y*y*x*x+1562.5*z*z*y*x*x+3125.0*ONESIXTH*z*z*z*x*x-2187.5*z*y*x*x+3125.0*0.25*z*z*y*y*x+8875.0/24.0*z*z*x-4375.0*0.25*y*x*x*x+3125.0*ONESIXTH*y*x*x*x*x-4375.0*0.25*z*z*y*x-4375.0*0.25*z*y*y*x-1925.0/12.0*z*x+3125.0*ONESIXTH*y*y*y*x*x+3125.0/24.0*x*x*x*x*x+3125.0*ONESIXTH*z*z*z*y*x-1925.0/12.0*y*x-4375.0/12.0*x*x*x*x+3125.0/24.0*y*y*y*y*x;
    phi[2] = -1562.5*z*y*x*x*x-25.0*x-250.0*z*y*x+3125.0*0.25*y*y*x*x+2675.0/12.0*x*x+625.0/12.0*y*y*y*x+5625.0*0.25*z*x*x*x+3125.0*0.25*z*z*x*x-7375.0/12.0*x*x*x-3125.0*0.25*z*x*x*x*x-3125.0*0.25*y*y*x*x*x-125.0*y*y*x-3125.0*0.25*z*z*x*x*x-8875.0/12.0*y*x*x-8875.0/12.0*z*x*x+625.0/12.0*z*z*z*x-3125.0*0.25*z*y*y*x*x-3125.0*0.25*z*z*y*x*x-3125.0/12.0*z*z*z*x*x+1562.5*z*y*x*x-125.0*z*z*x+5625.0*0.25*y*x*x*x-3125.0*0.25*y*x*x*x*x+625.0*0.25*z*z*y*x+625.0*0.25*z*y*y*x+1175.0/12.0*z*x-3125.0/12.0*y*y*y*x*x-3125.0/12.0*x*x*x*x*x+1175.0/12.0*y*x+8125.0/12.0*x*x*x*x;
    phi[3] = 125.0*ONETHIRD*z*y*x+50.0*ONETHIRD*x-312.5*z*y*x*x-162.5*x*x+6125.0/12.0*x*x*x+3125.0*ONESIXTH*z*y*x*x*x-625.0*x*x*x*x+3875.0/12.0*y*x*x+125.0*ONESIXTH*y*y*x-37.5*y*x-37.5*z*x-625.0*0.25*z*z*x*x+3125.0/12.0*x*x*x*x*x+3125.0*ONESIXTH*y*x*x*x*x+3125.0/12.0*y*y*x*x*x-3125.0*0.25*y*x*x*x-625.0*0.25*y*y*x*x-3125.0*0.25*z*x*x*x+3875.0/12.0*z*x*x+125.0*ONESIXTH*z*z*x+3125.0*ONESIXTH*z*x*x*x*x+3125.0/12.0*z*z*x*x*x;
    phi[4] = -25.0*0.25*x+1525.0/24.0*x*x-5125.0/24.0*x*x*x+6875.0/24.0*x*x*x*x-1375.0/24.0*y*x*x+25.0*0.25*y*x+25.0*0.25*z*x-3125.0/24.0*x*x*x*x*x-3125.0/24.0*y*x*x*x*x+625.0*0.25*y*x*x*x+625.0*0.25*z*x*x*x-1375.0/24.0*z*x*x-3125.0/24.0*z*x*x*x*x;
    phi[5] = x-125.0/12.0*x*x+875.0/24.0*x*x*x-625.0/12.0*x*x*x*x+625.0/24.0*x*x*x*x*x;
    phi[6] = 3125.0*ONESIXTH*z*y*x*x*x+25.0*y+1562.5*z*y*y*y*x+8875.0/12.0*z*y*x-4375.0*0.25*y*y*x*x+3125.0/24.0*z*z*z*z*y+3125.0*ONESIXTH*z*z*z*y*y-4375.0*0.25*y*y*y*x-4375.0*0.25*z*z*y*y-4375.0*0.25*z*y*y*y-4375.0/12.0*z*z*z*y-1925.0/12.0*z*y+8875.0/24.0*z*z*y+3125.0*ONESIXTH*y*y*x*x*x+3125.0*ONESIXTH*z*y*y*y*y+8875.0/12.0*y*y*x+8875.0/24.0*y*x*x+3125.0*0.25*z*z*y*y*y+8875.0/12.0*z*y*y+1562.5*z*y*y*x*x+3125.0*0.25*z*z*y*x*x-4375.0*0.25*z*y*x*x+1562.5*z*z*y*y*x-4375.0/12.0*y*x*x*x+3125.0/24.0*y*x*x*x*x-4375.0*0.25*z*z*y*x-2187.5*z*y*y*x+3125.0*0.25*y*y*y*x*x-1925.0/12.0*y*y+8875.0/24.0*y*y*y-4375.0/12.0*y*y*y*y+3125.0/24.0*y*y*y*y*y+3125.0*ONESIXTH*z*z*z*y*x-1925.0/12.0*y*x+3125.0*ONESIXTH*y*y*y*y*x;
    phi[7] = -5875.0*ONESIXTH*z*y*x-3125.0*ONESIXTH*z*z*z*y*x-1562.5*z*z*y*y*x-1562.5*z*y*y*y*x-1562.5*z*z*y*x*x+2500.0*z*y*y*x+1250.0*z*z*y*x+2500.0*z*y*x*x-3125.0*z*y*y*x*x-3125.0*ONESIXTH*y*y*y*y*x-1562.5*z*y*x*x*x-5875.0*ONESIXTH*y*x*x-5875.0*ONESIXTH*y*y*x+250.0*y*x-3125.0*ONESIXTH*y*x*x*x*x-1562.5*y*y*y*x*x-1562.5*y*y*x*x*x+1250.0*y*x*x*x+2500.0*y*y*x*x+1250.0*y*y*y*x;
    phi[8] = 1125.0*0.25*z*y*x+3125.0*0.25*z*z*y*x*x-312.5*z*y*y*x-625.0*0.25*z*z*y*x-6875.0*0.25*z*y*x*x+1562.5*z*y*y*x*x+1562.5*z*y*x*x*x+3625.0*0.25*y*x*x+1125.0*0.25*y*y*x-125.0*y*x+3125.0*0.25*y*x*x*x*x+3125.0*0.25*y*y*y*x*x+1562.5*y*y*x*x*x-1562.5*y*x*x*x-6875.0*0.25*y*y*x*x-625.0*0.25*y*y*y*x;
    phi[9] = -125.0*ONETHIRD*z*y*x+312.5*z*y*x*x-3125.0*ONESIXTH*z*y*x*x*x-2125.0*ONESIXTH*y*x*x-125.0*ONETHIRD*y*y*x+125.0*ONETHIRD*y*x-3125.0*ONESIXTH*y*x*x*x*x-3125.0*ONESIXTH*y*y*x*x*x+2500.0*ONETHIRD*y*x*x*x+312.5*y*y*x*x;
    phi[10] = 1375.0/24.0*y*x*x-25.0*0.25*y*x+3125.0/24.0*y*x*x*x*x-625.0*0.25*y*x*x*x;
    phi[11] = -25.0*y-1562.5*z*y*y*y*x-250.0*z*y*x+3125.0*0.25*y*y*x*x-3125.0/12.0*z*z*z*y*y+5625.0*0.25*y*y*y*x+3125.0*0.25*z*z*y*y+5625.0*0.25*z*y*y*y+625.0/12.0*z*z*z*y+1175.0/12.0*z*y-125.0*z*z*y-3125.0/12.0*y*y*x*x*x-3125.0*0.25*z*y*y*y*y-8875.0/12.0*y*y*x-125.0*y*x*x-3125.0*0.25*z*z*y*y*y-8875.0/12.0*z*y*y-3125.0*0.25*z*y*y*x*x+625.0*0.25*z*y*x*x-3125.0*0.25*z*z*y*y*x+625.0/12.0*y*x*x*x+625.0*0.25*z*z*y*x+1562.5*z*y*y*x-3125.0*0.25*y*y*y*x*x+2675.0/12.0*y*y-7375.0/12.0*y*y*y+8125.0/12.0*y*y*y*y-3125.0/12.0*y*y*y*y*y+1175.0/12.0*y*x-3125.0*0.25*y*y*y*y*x;
    phi[12] = 1125.0*0.25*z*y*x+1562.5*z*y*y*y*x+3125.0*0.25*z*z*y*y*x+3125.0*0.25*y*y*y*y*x-6875.0*0.25*z*y*y*x-625.0*0.25*z*z*y*x-312.5*z*y*x*x+1562.5*z*y*y*x*x+1125.0*0.25*y*x*x+3625.0*0.25*y*y*x-125.0*y*x+1562.5*y*y*y*x*x+3125.0*0.25*y*y*x*x*x-625.0*0.25*y*x*x*x-6875.0*0.25*y*y*x*x-1562.5*y*y*y*x;
    phi[13] = -125.0*0.25*z*y*x+625.0*0.25*z*y*y*x+625.0*0.25*z*y*x*x-3125.0*0.25*z*y*y*x*x-187.5*y*x*x-187.5*y*y*x+125.0*0.25*y*x-3125.0*0.25*y*y*y*x*x-3125.0*0.25*y*y*x*x*x+625.0*0.25*y*x*x*x+4375.0*0.25*y*y*x*x+625.0*0.25*y*y*y*x;
    phi[14] = 125.0*0.25*y*x*x+125.0*ONESIXTH*y*y*x-25.0*ONESIXTH*y*x+3125.0/12.0*y*y*x*x*x-625.0/12.0*y*x*x*x-625.0*0.25*y*y*x*x;
    phi[15] = 125.0*ONETHIRD*z*y*x+3125.0*ONESIXTH*z*y*y*y*x+3125.0*ONESIXTH*y*y*y*y*x+50.0*ONETHIRD*y-312.5*z*y*y*x+3875.0/12.0*z*y*y-162.5*y*y+6125.0/12.0*y*y*y-625.0*y*y*y*y+3125.0/12.0*y*y*y*y*y+3125.0*ONESIXTH*z*y*y*y*y+3125.0/12.0*z*z*y*y*y-3125.0*0.25*z*y*y*y+125.0*ONESIXTH*z*z*y-625.0*0.25*z*z*y*y-37.5*z*y+125.0*ONESIXTH*y*x*x+3875.0/12.0*y*y*x-37.5*y*x+3125.0/12.0*y*y*y*x*x-625.0*0.25*y*y*x*x-3125.0*0.25*y*y*y*x;
    phi[16] = -125.0*ONETHIRD*z*y*x-3125.0*ONESIXTH*z*y*y*y*x-3125.0*ONESIXTH*y*y*y*y*x+312.5*z*y*y*x-125.0*ONETHIRD*y*x*x-2125.0*ONESIXTH*y*y*x+125.0*ONETHIRD*y*x-3125.0*ONESIXTH*y*y*y*x*x+312.5*y*y*x*x+2500.0*ONETHIRD*y*y*y*x;
    phi[17] = 125.0*ONESIXTH*y*x*x+125.0*0.25*y*y*x-25.0*ONESIXTH*y*x+3125.0/12.0*y*y*y*x*x-625.0*0.25*y*y*x*x-625.0/12.0*y*y*y*x;
    phi[18] = -3125.0/24.0*y*y*y*y*x-25.0*0.25*y-1375.0/24.0*z*y*y+1525.0/24.0*y*y-5125.0/24.0*y*y*y+6875.0/24.0*y*y*y*y-3125.0/24.0*y*y*y*y*y-3125.0/24.0*z*y*y*y*y+625.0*0.25*z*y*y*y+25.0*0.25*z*y-1375.0/24.0*y*y*x+25.0*0.25*y*x+625.0*0.25*y*y*y*x;
    phi[19] = 3125.0/24.0*y*y*y*y*x+1375.0/24.0*y*y*x-25.0*0.25*y*x-625.0*0.25*y*y*y*x;
    phi[20] = y-125.0/12.0*y*y+875.0/24.0*y*y*y-625.0/12.0*y*y*y*y+625.0/24.0*y*y*y*y*y;PXShapeElem_Solution
    phi[21] = 3125.0*ONESIXTH*z*y*x*x*x+3125.0*ONESIXTH*z*y*y*y*x+8875.0/12.0*z*y*x+25.0*z+3125.0*ONESIXTH*z*z*z*z*y+3125.0*0.25*z*z*z*y*y-4375.0/12.0*z*x*x*x-4375.0*0.25*z*z*y*y-4375.0*0.25*z*z*x*x-4375.0/12.0*z*y*y*y-4375.0*0.25*z*z*z*y-1925.0/12.0*z*y+8875.0/12.0*z*z*y+3125.0/24.0*z*x*x*x*x+3125.0/24.0*z*y*y*y*y+3125.0*ONESIXTH*z*z*x*x*x+8875.0/24.0*z*x*x+3125.0*ONESIXTH*z*z*y*y*y+8875.0/24.0*z*y*y+3125.0*ONESIXTH*z*z*z*z*x-4375.0*0.25*z*z*z*x+3125.0*0.25*z*y*y*x*x+1562.5*z*z*y*x*x+3125.0*0.25*z*z*z*x*x-4375.0*0.25*z*y*x*x+1562.5*z*z*y*y*x+8875.0/12.0*z*z*x-2187.5*z*z*y*x-4375.0*0.25*z*y*y*x-1925.0/12.0*z*x-1925.0/12.0*z*z+8875.0/24.0*z*z*z-4375.0/12.0*z*z*z*z+3125.0/24.0*z*z*z*z*z+1562.5*z*z*z*y*x;
    phi[22] = -5875.0*ONESIXTH*z*y*x-1562.5*z*y*x*x*x-1562.5*z*y*y*x*x-3125.0*z*z*y*x*x-1562.5*z*z*y*y*x-1562.5*z*z*z*y*x+2500.0*z*y*x*x+2500.0*z*z*y*x-3125.0*ONESIXTH*z*y*y*y*x-5875.0*ONESIXTH*z*z*x+1250.0*z*y*y*x+250.0*z*x+2500.0*z*z*x*x-1562.5*z*z*x*x*x-3125.0*ONESIXTH*z*x*x*x*x-1562.5*z*z*z*x*x-3125.0*ONESIXTH*z*z*z*z*x+1250.0*z*x*x*x+1250.0*z*z*z*x-5875.0*ONESIXTH*z*x*x;
    phi[23] = 1125.0*0.25*z*y*x+1562.5*z*y*x*x*x+3125.0*0.25*z*y*y*x*x+1562.5*z*z*y*x*x-6875.0*0.25*z*y*x*x-312.5*z*z*y*x+1125.0*0.25*z*z*x-625.0*0.25*z*y*y*x-125.0*z*x-6875.0*0.25*z*z*x*x+1562.5*z*z*x*x*x+3125.0*0.25*z*x*x*x*x+3125.0*0.25*z*z*z*x*x-1562.5*z*x*x*x-625.0*0.25*z*z*z*x+3625.0*0.25*z*x*x;
    phi[24] = -125.0*ONETHIRD*z*y*x-3125.0*ONESIXTH*z*y*x*x*x+312.5*z*y*x*x-125.0*ONETHIRD*z*z*x+125.0*ONETHIRD*z*x+312.5*z*z*x*x-3125.0*ONESIXTH*z*z*x*x*x-3125.0*ONESIXTH*z*x*x*x*x+2500.0*ONETHIRD*z*x*x*x-2125.0*ONESIXTH*z*x*x;
    phi[25] = -25.0*0.25*z*x+3125.0/24.0*z*x*x*x*x-625.0*0.25*z*x*x*x+1375.0/24.0*z*x*x;
    phi[26] = -5875.0*ONESIXTH*z*y*x-3125.0*ONESIXTH*z*y*x*x*x-1562.5*z*y*y*x*x-1562.5*z*z*y*x*x-3125.0*z*z*y*y*x-1562.5*z*z*z*y*x+1250.0*z*y*x*x+2500.0*z*z*y*x-1562.5*z*y*y*y*x+2500.0*z*y*y*x+1250.0*z*z*z*y-1562.5*z*z*y*y*y+250.0*z*y-1562.5*z*z*z*y*y-5875.0*ONESIXTH*z*y*y-3125.0*ONESIXTH*z*z*z*z*y-5875.0*ONESIXTH*z*z*y-3125.0*ONESIXTH*z*y*y*y*y+1250.0*z*y*y*y+2500.0*z*z*y*y;
    phi[27] = 1250.0*z*y*x+1562.5*z*y*x*x*x+3125.0*z*y*y*x*x+3125.0*z*z*y*x*x+3125.0*z*z*y*y*x+1562.5*z*z*z*y*x-2812.5*z*y*x*x-2812.5*z*z*y*x+1562.5*z*y*y*y*x-2812.5*z*y*y*x;
    phi[28] = -312.5*z*y*x-1562.5*z*y*x*x*x-1562.5*z*y*y*x*x-1562.5*z*z*y*x*x+1875.0*z*y*x*x+312.5*z*z*y*x+312.5*z*y*y*x;
    phi[29] = 125.0*ONETHIRD*z*y*x+3125.0*ONESIXTH*z*y*x*x*x-312.5*z*y*x*x;
    phi[30] = 1125.0*0.25*z*y*x+3125.0*0.25*z*y*y*x*x+1562.5*z*z*y*y*x-625.0*0.25*z*y*x*x-312.5*z*z*y*x+1562.5*z*y*y*y*x-6875.0*0.25*z*y*y*x-625.0*0.25*z*z*z*y+1562.5*z*z*y*y*y-125.0*z*y+3125.0*0.25*z*z*z*y*y+3625.0*0.25*z*y*y+1125.0*0.25*z*z*y+3125.0*0.25*z*y*y*y*y-1562.5*z*y*y*y-6875.0*0.25*z*z*y*y;
    phi[31] = -312.5*z*y*x-1562.5*z*y*y*x*x-1562.5*z*z*y*y*x+312.5*z*y*x*x+312.5*z*z*y*x-1562.5*z*y*y*y*x+1875.0*z*y*y*x;
    phi[32] = 125.0*0.25*z*y*x+3125.0*0.25*z*y*y*x*x-625.0*0.25*z*y*x*x-625.0*0.25*z*y*y*x;
    phi[33] = -125.0*ONETHIRD*z*y*x-3125.0*ONESIXTH*z*y*y*y*x+312.5*z*y*y*x-3125.0*ONESIXTH*z*z*y*y*y+125.0*ONETHIRD*z*y-2125.0*ONESIXTH*z*y*y-125.0*ONETHIRD*z*z*y-3125.0*ONESIXTH*z*y*y*y*y+2500.0*ONETHIRD*z*y*y*y+312.5*z*z*y*y;
    phi[34] = 125.0*ONETHIRD*z*y*x+3125.0*ONESIXTH*z*y*y*y*x-312.5*z*y*y*x;
    phi[35] = -25.0*0.25*z*y+1375.0/24.0*z*y*y+3125.0/24.0*z*y*y*y*y-625.0*0.25*z*y*y*y;
    phi[36] = -250.0*z*y*x-25.0*z-3125.0*0.25*z*z*z*z*y-3125.0*0.25*z*z*z*y*y+625.0/12.0*z*x*x*x+3125.0*0.25*z*z*y*y+3125.0*0.25*z*z*x*x+625.0/12.0*z*y*y*y+5625.0*0.25*z*z*z*y+1175.0/12.0*z*y-8875.0/12.0*z*z*y-3125.0/12.0*z*z*x*x*x-125.0*z*x*x-3125.0/12.0*z*z*y*y*y-125.0*z*y*y-3125.0*0.25*z*z*z*z*x+5625.0*0.25*z*z*z*x-3125.0*0.25*z*z*y*x*x-3125.0*0.25*z*z*z*x*x+625.0*0.25*z*y*x*x-3125.0*0.25*z*z*y*y*x-8875.0/12.0*z*z*x+1562.5*z*z*y*x+625.0*0.25*z*y*y*x+1175.0/12.0*z*x+2675.0/12.0*z*z-7375.0/12.0*z*z*z+8125.0/12.0*z*z*z*z-3125.0/12.0*z*z*z*z*z-1562.5*z*z*z*y*x;
    phi[37] = 1562.5*z*z*y*x*x+3125.0*0.25*z*z*y*y*x+1562.5*z*z*z*y*x-6875.0*0.25*z*z*y*x+1125.0*0.25*z*y*x-312.5*z*y*x*x-625.0*0.25*z*x*x*x-625.0*0.25*z*y*y*x-6875.0*0.25*z*z*x*x-1562.5*z*z*z*x+3125.0*0.25*z*z*x*x*x+1125.0*0.25*z*x*x-125.0*z*x+3625.0*0.25*z*z*x+1562.5*z*z*z*x*x+3125.0*0.25*z*z*z*z*x;
    phi[38] = -3125.0*0.25*z*z*y*x*x+625.0*0.25*z*z*y*x-125.0*0.25*z*y*x+625.0*0.25*z*y*x*x+625.0*0.25*z*x*x*x+4375.0*0.25*z*z*x*x+625.0*0.25*z*z*z*x-3125.0*0.25*z*z*x*x*x-187.5*z*x*x+125.0*0.25*z*x-187.5*z*z*x-3125.0*0.25*z*z*z*x*x;
    phi[39] = -625.0/12.0*z*x*x*x-625.0*0.25*z*z*x*x+3125.0/12.0*z*z*x*x*x+125.0*0.25*z*x*x-25.0*ONESIXTH*z*x+125.0*ONESIXTH*z*z*x;
    phi[40] = 3125.0*0.25*z*z*y*x*x+1562.5*z*z*y*y*x+1562.5*z*z*z*y*x-6875.0*0.25*z*z*y*x+1125.0*0.25*z*y*x-625.0*0.25*z*y*x*x-312.5*z*y*y*x-1562.5*z*z*z*y+3125.0*0.25*z*z*y*y*y-125.0*z*y+1125.0*0.25*z*y*y+3625.0*0.25*z*z*y+1562.5*z*z*z*y*y+3125.0*0.25*z*z*z*z*y-625.0*0.25*z*y*y*y-6875.0*0.25*z*z*y*y;
    phi[41] = -1562.5*z*z*y*x*x-1562.5*z*z*y*y*x-1562.5*z*z*z*y*x+1875.0*z*z*y*x-312.5*z*y*x+312.5*z*y*x*x+312.5*z*y*y*x;
    phi[42] = 3125.0*0.25*z*z*y*x*x-625.0*0.25*z*z*y*x+125.0*0.25*z*y*x-625.0*0.25*z*y*x*x;
    phi[43] = -3125.0*0.25*z*z*y*y*x+625.0*0.25*z*z*y*x-125.0*0.25*z*y*x+625.0*0.25*z*y*y*x+625.0*0.25*z*z*z*y-3125.0*0.25*z*z*y*y*y+125.0*0.25*z*y-187.5*z*y*y-187.5*z*z*y-3125.0*0.25*z*z*z*y*y+625.0*0.25*z*y*y*y+4375.0*0.25*z*z*y*y;
    phi[44] = 3125.0*0.25*z*z*y*y*x-625.0*0.25*z*z*y*x+125.0*0.25*z*y*x-625.0*0.25*z*y*y*x;
    phi[45] = 3125.0/12.0*z*z*y*y*y-25.0*ONESIXTH*z*y+125.0*0.25*z*y*y+125.0*ONESIXTH*z*z*y-625.0/12.0*z*y*y*y-625.0*0.25*z*z*y*y;
    phi[46] = -312.5*z*z*y*x+125.0*ONETHIRD*z*y*x+3125.0*ONESIXTH*z*z*z*y*x+3125.0/12.0*z*z*z*x*x+50.0*ONETHIRD*z+3125.0*ONESIXTH*z*z*z*z*y+3125.0*ONESIXTH*z*z*z*z*x-3125.0*0.25*z*z*z*y-162.5*z*z+6125.0/12.0*z*z*z-625.0*z*z*z*z-37.5*z*y+125.0*ONESIXTH*z*y*y+3125.0/12.0*z*z*z*z*z-625.0*0.25*z*z*x*x+3875.0/12.0*z*z*y-3125.0*0.25*z*z*z*x+3125.0/12.0*z*z*z*y*y+125.0*ONESIXTH*z*x*x-625.0*0.25*z*z*y*y+3875.0/12.0*z*z*x-37.5*z*x;
    phi[47] = 312.5*z*z*y*x-125.0*ONETHIRD*z*y*x-3125.0*ONESIXTH*z*z*z*y*x-3125.0*ONESIXTH*z*z*z*x*x-3125.0*ONESIXTH*z*z*z*z*x+312.5*z*z*x*x+2500.0*ONETHIRD*z*z*z*x-125.0*ONETHIRD*z*x*x-2125.0*ONESIXTH*z*z*x+125.0*ONETHIRD*z*x;
    phi[48] = 3125.0/12.0*z*z*z*x*x-625.0*0.25*z*z*x*x-625.0/12.0*z*z*z*x+125.0*ONESIXTH*z*x*x+125.0*0.25*z*z*x-25.0*ONESIXTH*z*x;
    phi[49] = 312.5*z*z*y*x-125.0*ONETHIRD*z*y*x-3125.0*ONESIXTH*z*z*z*y*x-3125.0*ONESIXTH*z*z*z*z*y+2500.0*ONETHIRD*z*z*z*y+125.0*ONETHIRD*z*y-125.0*ONETHIRD*z*y*y-2125.0*ONESIXTH*z*z*y-3125.0*ONESIXTH*z*z*z*y*y+312.5*z*z*y*y;
    phi[50] = -312.5*z*z*y*x+125.0*ONETHIRD*z*y*x+3125.0*ONESIXTH*z*z*z*y*x;
    phi[51] = -625.0/12.0*z*z*z*y-25.0*ONESIXTH*z*y+125.0*ONESIXTH*z*y*y+125.0*0.25*z*z*y+3125.0/12.0*z*z*z*y*y-625.0*0.25*z*z*y*y;
    phi[52] = -3125.0/24.0*z*z*z*z*z-25.0*0.25*z-3125.0/24.0*z*z*z*z*y+625.0*0.25*z*z*z*y+25.0*0.25*z*y-1375.0/24.0*z*z*y-3125.0/24.0*z*z*z*z*x+625.0*0.25*z*z*z*x-1375.0/24.0*z*z*x+1525.0/24.0*z*z+25.0*0.25*z*x+6875.0/24.0*z*z*z*z-5125.0/24.0*z*z*z;
    phi[53] = 3125.0/24.0*z*z*z*z*x-625.0*0.25*z*z*z*x+1375.0/24.0*z*z*x-25.0*0.25*z*x;
    phi[54] = 3125.0/24.0*z*z*z*z*y-625.0*0.25*z*z*z*y-25.0*0.25*z*y+1375.0/24.0*z*z*y;
    phi[55] = z-125.0/12.0*z*z+875.0/24.0*z*z*z-625.0/12.0*z*z*z*z+625.0/24.0*z*z*z*z*z;

    return 0;
#endif
  default:
    return -1;
  }
}


/******************************************************************/
//   FUNCTION Definition: PXShapeLagrange3d
template <typename DT> ELVIS_DEVICE int
PXShapeLagrange3d_Solution(const int porder, const DT * RESTRICT xref, DT * RESTRICT phi)
{
  DT x, y, z;
  //DT MapleGenVar1, x3, x4, y3, y4, z3, z4;

  x = xref[0];
  y = xref[1];
  z = xref[2];

  switch (porder){
#if SOLN_USE_P0    
  case 0:
    phi[0] = 1.0;
    return 0;
#endif
#if SOLN_USE_P1
  case 1:
    phi[0] = 1.0-z-y-x;
    phi[1] = x;
    phi[2] = y;
    phi[3] = z;
    
    return 0;
#endif
#if SOLN_USE_P2
  case 2:
    phi[0] = 1.0-3.0*z-3.0*y-3.0*x+2.0*z*z+4.0*y*z+4.0*x*z+2.0*y*y+4.0*x*y+2.0*x*x;
    phi[1] = 4.0*x-4.0*x*z-4.0*x*y-4.0*x*x;
    phi[2] = -x+2.0*x*x;
    phi[3] = 4.0*y-4.0*y*z-4.0*y*y-4.0*x*y;
    phi[4] = 4.0*x*y;
    phi[5] = -y+2.0*y*y;
    phi[6] = 4.0*z-4.0*z*z-4.0*y*z-4.0*x*z;
    phi[7] = 4.0*x*z;
    phi[8] = 4.0*y*z;
    phi[9] = -z+2.0*z*z;

    return 0;
#endif
#if SOLN_USE_P3
  case 3:

    phi[0] = 1.0-5.5*z-5.5*y-5.5*x+9.0*z*z+18.0*z*y+18.0*z*x+9.0*y*y+18.0*y*x+9.0*x*x-4.5*z*z*z-13.5*z*z*y-13.5*z*z*x-13.5*z*y*y-27.0*z*y*x-13.5*z*x*x-4.5*y*y*y-13.5*y*y*x-13.5*y*x*x-4.5*x*x*x;
    phi[1] = 9.0*x-22.5*z*x-22.5*y*x-22.5*x*x+13.5*z*z*x+27.0*z*y*x+27.0*z*x*x+13.5*y*y*x+27.0*y*x*x+13.5*x*x*x;
    phi[2] = -4.5*x+4.5*z*x+4.5*y*x+18.0*x*x-13.5*z*x*x-13.5*y*x*x-13.5*x*x*x;
    phi[3] = x-4.5*x*x+4.5*x*x*x;
    phi[4] = 9.0*y-22.5*z*y-22.5*y*y-22.5*y*x+13.5*z*z*y+27.0*z*y*y+27.0*z*y*x+13.5*y*y*y+27.0*y*y*x+13.5*y*x*x;
    phi[5] = 27.0*y*x-27.0*z*y*x-27.0*y*y*x-27.0*y*x*x;
    phi[6] = -4.5*y*x+13.5*y*x*x;
    phi[7] = -4.5*y+4.5*z*y+18.0*y*y+4.5*y*x-13.5*z*y*y-13.5*y*y*y-13.5*y*y*x;
    phi[8] = -4.5*y*x+13.5*y*y*x;
    phi[9] = y-4.5*y*y+4.5*y*y*y;
    phi[10] = 9.0*z-22.5*z*z-22.5*z*y-22.5*z*x+13.5*z*z*z+27.0*z*z*y+27.0*z*z*x+13.5*z*y*y+27.0*z*y*x+13.5*z*x*x;
    phi[11] = 27.0*z*x-27.0*z*z*x-27.0*z*y*x-27.0*z*x*x;
    phi[12] = -4.5*z*x+13.5*z*x*x;
    phi[13] = 27.0*z*y-27.0*z*z*y-27.0*z*y*y-27.0*z*y*x;
    phi[14] = 27.0*z*y*x;
    phi[15] = -4.5*z*y+13.5*z*y*y;
    phi[16] = -4.5*z+18.0*z*z+4.5*z*y+4.5*z*x-13.5*z*z*z-13.5*z*z*y-13.5*z*z*x;
    phi[17] = -4.5*z*x+13.5*z*z*x;
    phi[18] = -4.5*z*y+13.5*z*z*y;
    phi[19] = z-4.5*z*z+4.5*z*z*z;

    return 0;
#endif
#if SOLN_USE_P4
  case 4:

    phi[0] = 1.0+128.0*z*y*x*x+128.0*z*z*y*x-160.0*z*y*x+128.0*ONETHIRD*z*z*z*y-25.0*ONETHIRD*x-25.0*ONETHIRD*y+128.0*z*y*y*x-80.0*y*x*x-80.0*z*y*y-80.0*y*y*x+140.0*ONETHIRD*y*x-80.0*z*z*y-25.0*ONETHIRD*z+32.0*ONETHIRD*y*y*y*y+32.0*ONETHIRD*x*x*x*x+32.0*ONETHIRD*z*z*z*z+70.0*ONETHIRD*z*z+70.0*ONETHIRD*y*y-80.0*ONETHIRD*z*z*z-80.0*ONETHIRD*y*y*y+140.0*ONETHIRD*z*y+70.0*ONETHIRD*x*x-80.0*ONETHIRD*x*x*x+128.0*ONETHIRD*y*x*x*x+64.0*y*y*x*x+128.0*ONETHIRD*y*y*y*x+64.0*z*z*x*x+128.0*ONETHIRD*z*x*x*x+128.0*ONETHIRD*z*z*z*x-80.0*z*z*x+140.0*ONETHIRD*z*x+128.0*ONETHIRD*z*y*y*y-80.0*z*x*x+64.0*z*z*y*y;
    phi[1] = 192.0*z*y*x+16.0*x-128.0*z*y*y*x-128.0*z*z*y*x-208.0*ONETHIRD*y*x-208.0*ONETHIRD*x*x-128.0*z*x*x*x-128.0*ONETHIRD*x*x*x*x-208.0*ONETHIRD*z*x-128.0*z*z*x*x-256.0*z*y*x*x+96.0*x*x*x-128.0*ONETHIRD*z*z*z*x+96.0*z*z*x-128.0*ONETHIRD*y*y*y*x-128.0*y*y*x*x-128.0*y*x*x*x+192.0*y*x*x+96.0*y*y*x+192.0*z*x*x;
    phi[2] = -32.0*z*y*x-12.0*x+28.0*y*x+76.0*x*x+128.0*z*x*x*x+64.0*x*x*x*x+28.0*z*x+64.0*z*z*x*x+128.0*z*y*x*x-128.0*x*x*x-16.0*z*z*x+64.0*y*y*x*x+128.0*y*x*x*x-144.0*y*x*x-16.0*y*y*x-144.0*z*x*x;
    phi[3] = 16.0*ONETHIRD*x-16.0*ONETHIRD*y*x-112.0*ONETHIRD*x*x-128.0*ONETHIRD*z*x*x*x-128.0*ONETHIRD*x*x*x*x-16.0*ONETHIRD*z*x+224.0*ONETHIRD*x*x*x-128.0*ONETHIRD*y*x*x*x+32.0*y*x*x+32.0*z*x*x;
    phi[4] = -x+22.0*ONETHIRD*x*x+32.0*ONETHIRD*x*x*x*x-16.0*x*x*x;
    phi[5] = 192.0*z*y*x+16.0*y-256.0*z*y*y*x-128.0*z*z*y*x-208.0*ONETHIRD*y*x-128.0*z*y*y*y-208.0*ONETHIRD*y*y+96.0*y*y*y-128.0*ONETHIRD*y*y*y*y+96.0*z*z*y-128.0*z*z*y*y-128.0*z*y*x*x+192.0*z*y*y-128.0*y*y*y*x-128.0*y*y*x*x-128.0*ONETHIRD*y*x*x*x-128.0*ONETHIRD*z*z*z*y+96.0*y*x*x+192.0*y*y*x-208.0*ONETHIRD*z*y;
    phi[6] = -224.0*z*y*x+256.0*z*y*y*x+128.0*z*z*y*x+96.0*y*x+256.0*z*y*x*x+128.0*y*y*y*x+256.0*y*y*x*x+128.0*y*x*x*x-224.0*y*x*x-224.0*y*y*x;
    phi[7] = 32.0*z*y*x-32.0*y*x-128.0*z*y*x*x-128.0*y*y*x*x-128.0*y*x*x*x+160.0*y*x*x+32.0*y*y*x;
    phi[8] = 16.0*ONETHIRD*y*x+128.0*ONETHIRD*y*x*x*x-32.0*y*x*x;
    phi[9] = -32.0*z*y*x-12.0*y+128.0*z*y*y*x+28.0*y*x+128.0*z*y*y*y+76.0*y*y-128.0*y*y*y+64.0*y*y*y*y-16.0*z*z*y+64.0*z*z*y*y-144.0*z*y*y+128.0*y*y*y*x+64.0*y*y*x*x-16.0*y*x*x-144.0*y*y*x+28.0*z*y;
    phi[10] = 32.0*z*y*x-128.0*z*y*y*x-32.0*y*x-128.0*y*y*y*x-128.0*y*y*x*x+32.0*y*x*x+160.0*y*y*x;
    phi[11] = 4.0*y*x+64.0*y*y*x*x-16.0*y*x*x-16.0*y*y*x;
    phi[12] = 16.0*ONETHIRD*y-16.0*ONETHIRD*y*x-128.0*ONETHIRD*z*y*y*y-112.0*ONETHIRD*y*y+224.0*ONETHIRD*y*y*y-128.0*ONETHIRD*y*y*y*y+32.0*z*y*y-128.0*ONETHIRD*y*y*y*x+32.0*y*y*x-16.0*ONETHIRD*z*y;
    phi[13] = 16.0*ONETHIRD*y*x+128.0*ONETHIRD*y*y*y*x-32.0*y*y*x;
    phi[14] = -y+22.0*ONETHIRD*y*y-16.0*y*y*y+32.0*ONETHIRD*y*y*y*y;
    phi[15] = -128.0*z*y*x*x-256.0*z*z*y*x+192.0*z*y*x-128.0*z*y*y*x+16.0*z-128.0*ONETHIRD*z*y*y*y+192.0*z*z*y-128.0*z*z*y*y+96.0*z*y*y-128.0*z*z*x*x-128.0*ONETHIRD*z*x*x*x-128.0*z*z*z*y-128.0*z*z*z*x+192.0*z*z*x-208.0*ONETHIRD*z*z+96.0*z*z*z-208.0*ONETHIRD*z*x+96.0*z*x*x-128.0*ONETHIRD*z*z*z*z-208.0*ONETHIRD*z*y;
    phi[16] = 256.0*z*y*x*x+256.0*z*z*y*x-224.0*z*y*x+128.0*z*y*y*x+256.0*z*z*x*x+128.0*z*x*x*x+128.0*z*z*z*x-224.0*z*z*x+96.0*z*x-224.0*z*x*x;
    phi[17] = -128.0*z*y*x*x+32.0*z*y*x-128.0*z*z*x*x-128.0*z*x*x*x+32.0*z*z*x-32.0*z*x+160.0*z*x*x;
    phi[18] = 128.0*ONETHIRD*z*x*x*x+16.0*ONETHIRD*z*x-32.0*z*x*x;
    phi[19] = 128.0*z*y*x*x+256.0*z*z*y*x-224.0*z*y*x+128.0*z*z*z*y+256.0*z*y*y*x-224.0*z*y*y-224.0*z*z*y+96.0*z*y+128.0*z*y*y*y+256.0*z*z*y*y;
    phi[20] = -256.0*z*y*x*x-256.0*z*z*y*x+256.0*z*y*x-256.0*z*y*y*x;
    phi[21] = 128.0*z*y*x*x-32.0*z*y*x;
    phi[22] = 32.0*z*y*x-128.0*z*y*y*x+160.0*z*y*y+32.0*z*z*y-32.0*z*y-128.0*z*y*y*y-128.0*z*z*y*y;
    phi[23] = -32.0*z*y*x+128.0*z*y*y*x;
    phi[24] = -32.0*z*y*y+16.0*ONETHIRD*z*y+128.0*ONETHIRD*z*y*y*y;
    phi[25] = 128.0*z*z*y*x-32.0*z*y*x+128.0*z*z*z*y-16.0*z*y*y-144.0*z*z*y-12.0*z+64.0*z*z*z*z+76.0*z*z-128.0*z*z*z+28.0*z*y+64.0*z*z*x*x+128.0*z*z*z*x-144.0*z*z*x+28.0*z*x-16.0*z*x*x+64.0*z*z*y*y;
    phi[26] = -128.0*z*z*y*x+32.0*z*y*x-128.0*z*z*x*x-128.0*z*z*z*x+160.0*z*z*x-32.0*z*x+32.0*z*x*x;
    phi[27] = 64.0*z*z*x*x-16.0*z*z*x+4.0*z*x-16.0*z*x*x;
    phi[28] = -128.0*z*z*y*x+32.0*z*y*x-128.0*z*z*z*y+32.0*z*y*y+160.0*z*z*y-32.0*z*y-128.0*z*z*y*y;
    phi[29] = 128.0*z*z*y*x-32.0*z*y*x;
    phi[30] = -16.0*z*y*y-16.0*z*z*y+4.0*z*y+64.0*z*z*y*y;
    phi[31] = -128.0*ONETHIRD*z*z*z*y+32.0*z*z*y+16.0*ONETHIRD*z-128.0*ONETHIRD*z*z*z*z-112.0*ONETHIRD*z*z+224.0*ONETHIRD*z*z*z-16.0*ONETHIRD*z*y-128.0*ONETHIRD*z*z*z*x+32.0*z*z*x-16.0*ONETHIRD*z*x;
    phi[32] = 128.0*ONETHIRD*z*z*z*x-32.0*z*z*x+16.0*ONETHIRD*z*x;
    phi[33] = 128.0*ONETHIRD*z*z*z*y-32.0*z*z*y+16.0*ONETHIRD*z*y;
    phi[34] = -z+32.0*ONETHIRD*z*z*z*z+22.0*ONETHIRD*z*z-16.0*z*z*z;
  
    return 0;
#endif
#if SOLN_USE_P5
  case 5:
    DT x4 = x*x*x*x;
    DT x3 = x*x*x;
    DT y4 = y*y*y*y;
    DT y3 = y*y*y;
    DT z4 = z*z*z*z;
    DT z3 = z*z*z;
    
    DT MapleGenVar1 = 1.0-3125.0*ONESIXTH*z*y*x*x*x-137.0/12.0*x-137.0/12.0*y-3125.0*ONESIXTH*z*y*y*y*x-2125.0*0.25*z*y*x+1875.0*0.25*y*y*x*x-137.0/12.0*z-3125.0/24.0*z*z*z*z*y-3125.0/12.0*z*z*z*y*y+375.0/8.0*x*x+312.5*y*y*y*x+312.5*z*x*x*x+1875.0*0.25*z*z*y*y+1875.0*0.25*z*z*x*x+312.5*z*y*y*y+312.5*z*z*z*y+375.0*0.25*z*y-2125.0/8.0*z*z*y-2125.0/24.0*x*x*x-3125.0/24.0*z*x*x*x*x-3125.0/12.0*y*y*x*x*x-3125.0/24.0*z*y*y*y*y-2125.0/8.0*y*y*x-3125.0/12.0*z*z*x*x*x-2125.0/8.0*y*x*x-2125.0/8.0*z*x*x-3125.0/12.0*z*z*y*y*y;
    phi[0] = -2125.0/8.0*z*y*y-3125.0/24.0*z*z*z*z*x+312.5*z*z*z*x-3125.0*0.25*z*y*y*x*x-3125.0*0.25*z*z*y*x*x-3125.0/12.0*z*z*z*x*x+937.5*z*y*x*x-3125.0*0.25*z*z*y*y*x+MapleGenVar1-2125.0/8.0*z*z*x+312.5*y*x*x*x-3125.0/24.0*y*x*x*x*x+937.5*z*z*y*x+937.5*z*y*y*x+375.0*0.25*z*x-3125.0/12.0*y*y*y*x*x+375.0/8.0*z*z+375.0/8.0*y*y-2125.0/24.0*z*z*z-2125.0/24.0*y*y*y+625.0/8.0*z*z*z*z+625.0/8.0*y*y*y*y-625.0/24.0*z*z*z*z*z-625.0/24.0*y*y*y*y*y-625.0/24.0*x*x*x*x*x-3125.0*ONESIXTH*z*z*z*y*x+375.0*0.25*y*x+625.0/8.0*x*x*x*x-3125.0/24.0*y*y*y*y*x;
    phi[1] = 1562.5*z*y*x*x*x+25.0*x+3125.0*ONESIXTH*z*y*y*y*x+8875.0/12.0*z*y*x-4375.0*0.25*y*y*x*x-1925.0/12.0*x*x-4375.0/12.0*y*y*y*x-4375.0*0.25*z*x*x*x-4375.0*0.25*z*z*x*x+8875.0/24.0*x*x*x+3125.0*ONESIXTH*z*x*x*x*x+3125.0*0.25*y*y*x*x*x+8875.0/24.0*y*y*x+3125.0*0.25*z*z*x*x*x+8875.0/12.0*y*x*x+8875.0/12.0*z*x*x+3125.0/24.0*z*z*z*z*x-4375.0/12.0*z*z*z*x+1562.5*z*y*y*x*x+1562.5*z*z*y*x*x+3125.0*ONESIXTH*z*z*z*x*x-2187.5*z*y*x*x+3125.0*0.25*z*z*y*y*x+8875.0/24.0*z*z*x-4375.0*0.25*y*x*x*x+3125.0*ONESIXTH*y*x*x*x*x-4375.0*0.25*z*z*y*x-4375.0*0.25*z*y*y*x-1925.0/12.0*z*x+3125.0*ONESIXTH*y*y*y*x*x+3125.0/24.0*x*x*x*x*x+3125.0*ONESIXTH*z*z*z*y*x-1925.0/12.0*y*x-4375.0/12.0*x*x*x*x+3125.0/24.0*y*y*y*y*x;
    phi[2] = -1562.5*z*y*x*x*x-25.0*x-250.0*z*y*x+3125.0*0.25*y*y*x*x+2675.0/12.0*x*x+625.0/12.0*y*y*y*x+5625.0*0.25*z*x*x*x+3125.0*0.25*z*z*x*x-7375.0/12.0*x*x*x-3125.0*0.25*z*x*x*x*x-3125.0*0.25*y*y*x*x*x-125.0*y*y*x-3125.0*0.25*z*z*x*x*x-8875.0/12.0*y*x*x-8875.0/12.0*z*x*x+625.0/12.0*z*z*z*x-3125.0*0.25*z*y*y*x*x-3125.0*0.25*z*z*y*x*x-3125.0/12.0*z*z*z*x*x+1562.5*z*y*x*x-125.0*z*z*x+5625.0*0.25*y*x*x*x-3125.0*0.25*y*x*x*x*x+625.0*0.25*z*z*y*x+625.0*0.25*z*y*y*x+1175.0/12.0*z*x-3125.0/12.0*y*y*y*x*x-3125.0/12.0*x*x*x*x*x+1175.0/12.0*y*x+8125.0/12.0*x*x*x*x;
    phi[3] = 125.0*ONETHIRD*z*y*x+50.0*ONETHIRD*x-312.5*z*y*x*x-162.5*x*x+6125.0/12.0*x*x*x+3125.0*ONESIXTH*z*y*x*x*x-625.0*x*x*x*x+3875.0/12.0*y*x*x+125.0*ONESIXTH*y*y*x-37.5*y*x-37.5*z*x-625.0*0.25*z*z*x*x+3125.0/12.0*x*x*x*x*x+3125.0*ONESIXTH*y*x*x*x*x+3125.0/12.0*y*y*x*x*x-3125.0*0.25*y*x*x*x-625.0*0.25*y*y*x*x-3125.0*0.25*z*x*x*x+3875.0/12.0*z*x*x+125.0*ONESIXTH*z*z*x+3125.0*ONESIXTH*z*x*x*x*x+3125.0/12.0*z*z*x*x*x;
    phi[4] = -25.0*0.25*x+1525.0/24.0*x*x-5125.0/24.0*x*x*x+6875.0/24.0*x*x*x*x-1375.0/24.0*y*x*x+25.0*0.25*y*x+25.0*0.25*z*x-3125.0/24.0*x*x*x*x*x-3125.0/24.0*y*x*x*x*x+625.0*0.25*y*x*x*x+625.0*0.25*z*x*x*x-1375.0/24.0*z*x*x-3125.0/24.0*z*x*x*x*x;
    phi[5] = x-125.0/12.0*x*x+875.0/24.0*x*x*x-625.0/12.0*x*x*x*x+625.0/24.0*x*x*x*x*x;
    phi[6] = 3125.0*ONESIXTH*z*y*x*x*x+25.0*y+1562.5*z*y*y*y*x+8875.0/12.0*z*y*x-4375.0*0.25*y*y*x*x+3125.0/24.0*z*z*z*z*y+3125.0*ONESIXTH*z*z*z*y*y-4375.0*0.25*y*y*y*x-4375.0*0.25*z*z*y*y-4375.0*0.25*z*y*y*y-4375.0/12.0*z*z*z*y-1925.0/12.0*z*y+8875.0/24.0*z*z*y+3125.0*ONESIXTH*y*y*x*x*x+3125.0*ONESIXTH*z*y*y*y*y+8875.0/12.0*y*y*x+8875.0/24.0*y*x*x+3125.0*0.25*z*z*y*y*y+8875.0/12.0*z*y*y+1562.5*z*y*y*x*x+3125.0*0.25*z*z*y*x*x-4375.0*0.25*z*y*x*x+1562.5*z*z*y*y*x-4375.0/12.0*y*x*x*x+3125.0/24.0*y*x*x*x*x-4375.0*0.25*z*z*y*x-2187.5*z*y*y*x+3125.0*0.25*y*y*y*x*x-1925.0/12.0*y*y+8875.0/24.0*y*y*y-4375.0/12.0*y*y*y*y+3125.0/24.0*y*y*y*y*y+3125.0*ONESIXTH*z*z*z*y*x-1925.0/12.0*y*x+3125.0*ONESIXTH*y*y*y*y*x;
    phi[7] = -5875.0*ONESIXTH*z*y*x-3125.0*ONESIXTH*z*z*z*y*x-1562.5*z*z*y*y*x-1562.5*z*y*y*y*x-1562.5*z*z*y*x*x+2500.0*z*y*y*x+1250.0*z*z*y*x+2500.0*z*y*x*x-3125.0*z*y*y*x*x-3125.0*ONESIXTH*y*y*y*y*x-1562.5*z*y*x*x*x-5875.0*ONESIXTH*y*x*x-5875.0*ONESIXTH*y*y*x+250.0*y*x-3125.0*ONESIXTH*y*x*x*x*x-1562.5*y*y*y*x*x-1562.5*y*y*x*x*x+1250.0*y*x*x*x+2500.0*y*y*x*x+1250.0*y*y*y*x;
    phi[8] = 1125.0*0.25*z*y*x+3125.0*0.25*z*z*y*x*x-312.5*z*y*y*x-625.0*0.25*z*z*y*x-6875.0*0.25*z*y*x*x+1562.5*z*y*y*x*x+1562.5*z*y*x*x*x+3625.0*0.25*y*x*x+1125.0*0.25*y*y*x-125.0*y*x+3125.0*0.25*y*x*x*x*x+3125.0*0.25*y*y*y*x*x+1562.5*y*y*x*x*x-1562.5*y*x*x*x-6875.0*0.25*y*y*x*x-625.0*0.25*y*y*y*x;
    phi[9] = -125.0*ONETHIRD*z*y*x+312.5*z*y*x*x-3125.0*ONESIXTH*z*y*x*x*x-2125.0*ONESIXTH*y*x*x-125.0*ONETHIRD*y*y*x+125.0*ONETHIRD*y*x-3125.0*ONESIXTH*y*x*x*x*x-3125.0*ONESIXTH*y*y*x*x*x+2500.0*ONETHIRD*y*x*x*x+312.5*y*y*x*x;
    phi[10] = 1375.0/24.0*y*x*x-25.0*0.25*y*x+3125.0/24.0*y*x*x*x*x-625.0*0.25*y*x*x*x;
    phi[11] = -25.0*y-1562.5*z*y*y*y*x-250.0*z*y*x+3125.0*0.25*y*y*x*x-3125.0/12.0*z*z*z*y*y+5625.0*0.25*y*y*y*x+3125.0*0.25*z*z*y*y+5625.0*0.25*z*y*y*y+625.0/12.0*z*z*z*y+1175.0/12.0*z*y-125.0*z*z*y-3125.0/12.0*y*y*x*x*x-3125.0*0.25*z*y*y*y*y-8875.0/12.0*y*y*x-125.0*y*x*x-3125.0*0.25*z*z*y*y*y-8875.0/12.0*z*y*y-3125.0*0.25*z*y*y*x*x+625.0*0.25*z*y*x*x-3125.0*0.25*z*z*y*y*x+625.0/12.0*y*x*x*x+625.0*0.25*z*z*y*x+1562.5*z*y*y*x-3125.0*0.25*y*y*y*x*x+2675.0/12.0*y*y-7375.0/12.0*y*y*y+8125.0/12.0*y*y*y*y-3125.0/12.0*y*y*y*y*y+1175.0/12.0*y*x-3125.0*0.25*y*y*y*y*x;
    phi[12] = 1125.0*0.25*z*y*x+1562.5*z*y*y*y*x+3125.0*0.25*z*z*y*y*x+3125.0*0.25*y*y*y*y*x-6875.0*0.25*z*y*y*x-625.0*0.25*z*z*y*x-312.5*z*y*x*x+1562.5*z*y*y*x*x+1125.0*0.25*y*x*x+3625.0*0.25*y*y*x-125.0*y*x+1562.5*y*y*y*x*x+3125.0*0.25*y*y*x*x*x-625.0*0.25*y*x*x*x-6875.0*0.25*y*y*x*x-1562.5*y*y*y*x;
    phi[13] = -125.0*0.25*z*y*x+625.0*0.25*z*y*y*x+625.0*0.25*z*y*x*x-3125.0*0.25*z*y*y*x*x-187.5*y*x*x-187.5*y*y*x+125.0*0.25*y*x-3125.0*0.25*y*y*y*x*x-3125.0*0.25*y*y*x*x*x+625.0*0.25*y*x*x*x+4375.0*0.25*y*y*x*x+625.0*0.25*y*y*y*x;
    phi[14] = 125.0*0.25*y*x*x+125.0*ONESIXTH*y*y*x-25.0*ONESIXTH*y*x+3125.0/12.0*y*y*x*x*x-625.0/12.0*y*x*x*x-625.0*0.25*y*y*x*x;
    phi[15] = 125.0*ONETHIRD*z*y*x+3125.0*ONESIXTH*z*y*y*y*x+3125.0*ONESIXTH*y*y*y*y*x+50.0*ONETHIRD*y-312.5*z*y*y*x+3875.0/12.0*z*y*y-162.5*y*y+6125.0/12.0*y*y*y-625.0*y*y*y*y+3125.0/12.0*y*y*y*y*y+3125.0*ONESIXTH*z*y*y*y*y+3125.0/12.0*z*z*y*y*y-3125.0*0.25*z*y*y*y+125.0*ONESIXTH*z*z*y-625.0*0.25*z*z*y*y-37.5*z*y+125.0*ONESIXTH*y*x*x+3875.0/12.0*y*y*x-37.5*y*x+3125.0/12.0*y*y*y*x*x-625.0*0.25*y*y*x*x-3125.0*0.25*y*y*y*x;
    phi[16] = -125.0*ONETHIRD*z*y*x-3125.0*ONESIXTH*z*y*y*y*x-3125.0*ONESIXTH*y*y*y*y*x+312.5*z*y*y*x-125.0*ONETHIRD*y*x*x-2125.0*ONESIXTH*y*y*x+125.0*ONETHIRD*y*x-3125.0*ONESIXTH*y*y*y*x*x+312.5*y*y*x*x+2500.0*ONETHIRD*y*y*y*x;
    phi[17] = 125.0*ONESIXTH*y*x*x+125.0*0.25*y*y*x-25.0*ONESIXTH*y*x+3125.0/12.0*y*y*y*x*x-625.0*0.25*y*y*x*x-625.0/12.0*y*y*y*x;
    phi[18] = -3125.0/24.0*y*y*y*y*x-25.0*0.25*y-1375.0/24.0*z*y*y+1525.0/24.0*y*y-5125.0/24.0*y*y*y+6875.0/24.0*y*y*y*y-3125.0/24.0*y*y*y*y*y-3125.0/24.0*z*y*y*y*y+625.0*0.25*z*y*y*y+25.0*0.25*z*y-1375.0/24.0*y*y*x+25.0*0.25*y*x+625.0*0.25*y*y*y*x;
    phi[19] = 3125.0/24.0*y*y*y*y*x+1375.0/24.0*y*y*x-25.0*0.25*y*x-625.0*0.25*y*y*y*x;
    phi[20] = y-125.0/12.0*y*y+875.0/24.0*y*y*y-625.0/12.0*y*y*y*y+625.0/24.0*y*y*y*y*y;
    phi[21] = 3125.0*ONESIXTH*z*y*x*x*x+3125.0*ONESIXTH*z*y*y*y*x+8875.0/12.0*z*y*x+25.0*z+3125.0*ONESIXTH*z*z*z*z*y+3125.0*0.25*z*z*z*y*y-4375.0/12.0*z*x*x*x-4375.0*0.25*z*z*y*y-4375.0*0.25*z*z*x*x-4375.0/12.0*z*y*y*y-4375.0*0.25*z*z*z*y-1925.0/12.0*z*y+8875.0/12.0*z*z*y+3125.0/24.0*z*x*x*x*x+3125.0/24.0*z*y*y*y*y+3125.0*ONESIXTH*z*z*x*x*x+8875.0/24.0*z*x*x+3125.0*ONESIXTH*z*z*y*y*y+8875.0/24.0*z*y*y+3125.0*ONESIXTH*z*z*z*z*x-4375.0*0.25*z*z*z*x+3125.0*0.25*z*y*y*x*x+1562.5*z*z*y*x*x+3125.0*0.25*z*z*z*x*x-4375.0*0.25*z*y*x*x+1562.5*z*z*y*y*x+8875.0/12.0*z*z*x-2187.5*z*z*y*x-4375.0*0.25*z*y*y*x-1925.0/12.0*z*x-1925.0/12.0*z*z+8875.0/24.0*z*z*z-4375.0/12.0*z*z*z*z+3125.0/24.0*z*z*z*z*z+1562.5*z*z*z*y*x;
    phi[22] = -5875.0*ONESIXTH*z*y*x-1562.5*z*y*x*x*x-1562.5*z*y*y*x*x-3125.0*z*z*y*x*x-1562.5*z*z*y*y*x-1562.5*z*z*z*y*x+2500.0*z*y*x*x+2500.0*z*z*y*x-3125.0*ONESIXTH*z*y*y*y*x-5875.0*ONESIXTH*z*z*x+1250.0*z*y*y*x+250.0*z*x+2500.0*z*z*x*x-1562.5*z*z*x*x*x-3125.0*ONESIXTH*z*x*x*x*x-1562.5*z*z*z*x*x-3125.0*ONESIXTH*z*z*z*z*x+1250.0*z*x*x*x+1250.0*z*z*z*x-5875.0*ONESIXTH*z*x*x;
    phi[23] = 1125.0*0.25*z*y*x+1562.5*z*y*x*x*x+3125.0*0.25*z*y*y*x*x+1562.5*z*z*y*x*x-6875.0*0.25*z*y*x*x-312.5*z*z*y*x+1125.0*0.25*z*z*x-625.0*0.25*z*y*y*x-125.0*z*x-6875.0*0.25*z*z*x*x+1562.5*z*z*x*x*x+3125.0*0.25*z*x*x*x*x+3125.0*0.25*z*z*z*x*x-1562.5*z*x*x*x-625.0*0.25*z*z*z*x+3625.0*0.25*z*x*x;
    phi[24] = -125.0*ONETHIRD*z*y*x-3125.0*ONESIXTH*z*y*x*x*x+312.5*z*y*x*x-125.0*ONETHIRD*z*z*x+125.0*ONETHIRD*z*x+312.5*z*z*x*x-3125.0*ONESIXTH*z*z*x*x*x-3125.0*ONESIXTH*z*x*x*x*x+2500.0*ONETHIRD*z*x*x*x-2125.0*ONESIXTH*z*x*x;
    phi[25] = -25.0*0.25*z*x+3125.0/24.0*z*x*x*x*x-625.0*0.25*z*x*x*x+1375.0/24.0*z*x*x;
    phi[26] = -5875.0*ONESIXTH*z*y*x-3125.0*ONESIXTH*z*y*x*x*x-1562.5*z*y*y*x*x-1562.5*z*z*y*x*x-3125.0*z*z*y*y*x-1562.5*z*z*z*y*x+1250.0*z*y*x*x+2500.0*z*z*y*x-1562.5*z*y*y*y*x+2500.0*z*y*y*x+1250.0*z*z*z*y-1562.5*z*z*y*y*y+250.0*z*y-1562.5*z*z*z*y*y-5875.0*ONESIXTH*z*y*y-3125.0*ONESIXTH*z*z*z*z*y-5875.0*ONESIXTH*z*z*y-3125.0*ONESIXTH*z*y*y*y*y+1250.0*z*y*y*y+2500.0*z*z*y*y;
    phi[27] = 1250.0*z*y*x+1562.5*z*y*x*x*x+3125.0*z*y*y*x*x+3125.0*z*z*y*x*x+3125.0*z*z*y*y*x+1562.5*z*z*z*y*x-2812.5*z*y*x*x-2812.5*z*z*y*x+1562.5*z*y*y*y*x-2812.5*z*y*y*x;
    phi[28] = -312.5*z*y*x-1562.5*z*y*x*x*x-1562.5*z*y*y*x*x-1562.5*z*z*y*x*x+1875.0*z*y*x*x+312.5*z*z*y*x+312.5*z*y*y*x;
    phi[29] = 125.0*ONETHIRD*z*y*x+3125.0*ONESIXTH*z*y*x*x*x-312.5*z*y*x*x;
    phi[30] = 1125.0*0.25*z*y*x+3125.0*0.25*z*y*y*x*x+1562.5*z*z*y*y*x-625.0*0.25*z*y*x*x-312.5*z*z*y*x+1562.5*z*y*y*y*x-6875.0*0.25*z*y*y*x-625.0*0.25*z*z*z*y+1562.5*z*z*y*y*y-125.0*z*y+3125.0*0.25*z*z*z*y*y+3625.0*0.25*z*y*y+1125.0*0.25*z*z*y+3125.0*0.25*z*y*y*y*y-1562.5*z*y*y*y-6875.0*0.25*z*z*y*y;
    phi[31] = -312.5*z*y*x-1562.5*z*y*y*x*x-1562.5*z*z*y*y*x+312.5*z*y*x*x+312.5*z*z*y*x-1562.5*z*y*y*y*x+1875.0*z*y*y*x;
    phi[32] = 125.0*0.25*z*y*x+3125.0*0.25*z*y*y*x*x-625.0*0.25*z*y*x*x-625.0*0.25*z*y*y*x;
    phi[33] = -125.0*ONETHIRD*z*y*x-3125.0*ONESIXTH*z*y*y*y*x+312.5*z*y*y*x-3125.0*ONESIXTH*z*z*y*y*y+125.0*ONETHIRD*z*y-2125.0*ONESIXTH*z*y*y-125.0*ONETHIRD*z*z*y-3125.0*ONESIXTH*z*y*y*y*y+2500.0*ONETHIRD*z*y*y*y+312.5*z*z*y*y;
    phi[34] = 125.0*ONETHIRD*z*y*x+3125.0*ONESIXTH*z*y*y*y*x-312.5*z*y*y*x;
    phi[35] = -25.0*0.25*z*y+1375.0/24.0*z*y*y+3125.0/24.0*z*y*y*y*y-625.0*0.25*z*y*y*y;
    phi[36] = -250.0*z*y*x-25.0*z-3125.0*0.25*z*z*z*z*y-3125.0*0.25*z*z*z*y*y+625.0/12.0*z*x*x*x+3125.0*0.25*z*z*y*y+3125.0*0.25*z*z*x*x+625.0/12.0*z*y*y*y+5625.0*0.25*z*z*z*y+1175.0/12.0*z*y-8875.0/12.0*z*z*y-3125.0/12.0*z*z*x*x*x-125.0*z*x*x-3125.0/12.0*z*z*y*y*y-125.0*z*y*y-3125.0*0.25*z*z*z*z*x+5625.0*0.25*z*z*z*x-3125.0*0.25*z*z*y*x*x-3125.0*0.25*z*z*z*x*x+625.0*0.25*z*y*x*x-3125.0*0.25*z*z*y*y*x-8875.0/12.0*z*z*x+1562.5*z*z*y*x+625.0*0.25*z*y*y*x+1175.0/12.0*z*x+2675.0/12.0*z*z-7375.0/12.0*z*z*z+8125.0/12.0*z*z*z*z-3125.0/12.0*z*z*z*z*z-1562.5*z*z*z*y*x;
    phi[37] = 1562.5*z*z*y*x*x+3125.0*0.25*z*z*y*y*x+1562.5*z*z*z*y*x-6875.0*0.25*z*z*y*x+1125.0*0.25*z*y*x-312.5*z*y*x*x-625.0*0.25*z*x*x*x-625.0*0.25*z*y*y*x-6875.0*0.25*z*z*x*x-1562.5*z*z*z*x+3125.0*0.25*z*z*x*x*x+1125.0*0.25*z*x*x-125.0*z*x+3625.0*0.25*z*z*x+1562.5*z*z*z*x*x+3125.0*0.25*z*z*z*z*x;
    phi[38] = -3125.0*0.25*z*z*y*x*x+625.0*0.25*z*z*y*x-125.0*0.25*z*y*x+625.0*0.25*z*y*x*x+625.0*0.25*z*x*x*x+4375.0*0.25*z*z*x*x+625.0*0.25*z*z*z*x-3125.0*0.25*z*z*x*x*x-187.5*z*x*x+125.0*0.25*z*x-187.5*z*z*x-3125.0*0.25*z*z*z*x*x;
    phi[39] = -625.0/12.0*z*x*x*x-625.0*0.25*z*z*x*x+3125.0/12.0*z*z*x*x*x+125.0*0.25*z*x*x-25.0*ONESIXTH*z*x+125.0*ONESIXTH*z*z*x;
    phi[40] = 3125.0*0.25*z*z*y*x*x+1562.5*z*z*y*y*x+1562.5*z*z*z*y*x-6875.0*0.25*z*z*y*x+1125.0*0.25*z*y*x-625.0*0.25*z*y*x*x-312.5*z*y*y*x-1562.5*z*z*z*y+3125.0*0.25*z*z*y*y*y-125.0*z*y+1125.0*0.25*z*y*y+3625.0*0.25*z*z*y+1562.5*z*z*z*y*y+3125.0*0.25*z*z*z*z*y-625.0*0.25*z*y*y*y-6875.0*0.25*z*z*y*y;
    phi[41] = -1562.5*z*z*y*x*x-1562.5*z*z*y*y*x-1562.5*z*z*z*y*x+1875.0*z*z*y*x-312.5*z*y*x+312.5*z*y*x*x+312.5*z*y*y*x;
    phi[42] = 3125.0*0.25*z*z*y*x*x-625.0*0.25*z*z*y*x+125.0*0.25*z*y*x-625.0*0.25*z*y*x*x;
    phi[43] = -3125.0*0.25*z*z*y*y*x+625.0*0.25*z*z*y*x-125.0*0.25*z*y*x+625.0*0.25*z*y*y*x+625.0*0.25*z*z*z*y-3125.0*0.25*z*z*y*y*y+125.0*0.25*z*y-187.5*z*y*y-187.5*z*z*y-3125.0*0.25*z*z*z*y*y+625.0*0.25*z*y*y*y+4375.0*0.25*z*z*y*y;
    phi[44] = 3125.0*0.25*z*z*y*y*x-625.0*0.25*z*z*y*x+125.0*0.25*z*y*x-625.0*0.25*z*y*y*x;
    phi[45] = 3125.0/12.0*z*z*y*y*y-25.0*ONESIXTH*z*y+125.0*0.25*z*y*y+125.0*ONESIXTH*z*z*y-625.0/12.0*z*y*y*y-625.0*0.25*z*z*y*y;
    phi[46] = -312.5*z*z*y*x+125.0*ONETHIRD*z*y*x+3125.0*ONESIXTH*z*z*z*y*x+3125.0/12.0*z*z*z*x*x+50.0*ONETHIRD*z+3125.0*ONESIXTH*z*z*z*z*y+3125.0*ONESIXTH*z*z*z*z*x-3125.0*0.25*z*z*z*y-162.5*z*z+6125.0/12.0*z*z*z-625.0*z*z*z*z-37.5*z*y+125.0*ONESIXTH*z*y*y+3125.0/12.0*z*z*z*z*z-625.0*0.25*z*z*x*x+3875.0/12.0*z*z*y-3125.0*0.25*z*z*z*x+3125.0/12.0*z*z*z*y*y+125.0*ONESIXTH*z*x*x-625.0*0.25*z*z*y*y+3875.0/12.0*z*z*x-37.5*z*x;
    phi[47] = 312.5*z*z*y*x-125.0*ONETHIRD*z*y*x-3125.0*ONESIXTH*z*z*z*y*x-3125.0*ONESIXTH*z*z*z*x*x-3125.0*ONESIXTH*z*z*z*z*x+312.5*z*z*x*x+2500.0*ONETHIRD*z*z*z*x-125.0*ONETHIRD*z*x*x-2125.0*ONESIXTH*z*z*x+125.0*ONETHIRD*z*x;
    phi[48] = 3125.0/12.0*z*z*z*x*x-625.0*0.25*z*z*x*x-625.0/12.0*z*z*z*x+125.0*ONESIXTH*z*x*x+125.0*0.25*z*z*x-25.0*ONESIXTH*z*x;
    phi[49] = 312.5*z*z*y*x-125.0*ONETHIRD*z*y*x-3125.0*ONESIXTH*z*z*z*y*x-3125.0*ONESIXTH*z*z*z*z*y+2500.0*ONETHIRD*z*z*z*y+125.0*ONETHIRD*z*y-125.0*ONETHIRD*z*y*y-2125.0*ONESIXTH*z*z*y-3125.0*ONESIXTH*z*z*z*y*y+312.5*z*z*y*y;
    phi[50] = -312.5*z*z*y*x+125.0*ONETHIRD*z*y*x+3125.0*ONESIXTH*z*z*z*y*x;
    phi[51] = -625.0/12.0*z*z*z*y-25.0*ONESIXTH*z*y+125.0*ONESIXTH*z*y*y+125.0*0.25*z*z*y+3125.0/12.0*z*z*z*y*y-625.0*0.25*z*z*y*y;
    phi[52] = -3125.0/24.0*z*z*z*z*z-25.0*0.25*z-3125.0/24.0*z*z*z*z*y+625.0*0.25*z*z*z*y+25.0*0.25*z*y-1375.0/24.0*z*z*y-3125.0/24.0*z*z*z*z*x+625.0*0.25*z*z*z*x-1375.0/24.0*z*z*x+1525.0/24.0*z*z+25.0*0.25*z*x+6875.0/24.0*z*z*z*z-5125.0/24.0*z*z*z;
    phi[53] = 3125.0/24.0*z*z*z*z*x-625.0*0.25*z*z*z*x+1375.0/24.0*z*z*x-25.0*0.25*z*x;
    phi[54] = 3125.0/24.0*z*z*z*z*y-625.0*0.25*z*z*z*y-25.0*0.25*z*y+1375.0/24.0*z*z*y;
    phi[55] = z-125.0/12.0*z*z+875.0/24.0*z*z*z-625.0/12.0*z*z*z*z+625.0/24.0*z*z*z*z*z;

    return 0;
#endif
  default:
    return -1;
  }
}


/******************************************************************/
//   FUNCTION Definition: PXShapeQuadUniformLagrange2d
template <typename DT> ELVIS_DEVICE int
PXShapeHexUniformLagrange3d(const int porder, const DT * RESTRICT xref, DT * RESTRICT phi)
{
  int i,j,k,l;
  DT phi_i[6]; // 1d uniform lagrange basis functions
  DT phi_j[6]; // 1d uniform lagrange basis functions
  DT phi_k[6]; // 1d uniform lagrange basis functions

  /* move coordinates */
  (PXShapeUniformLagrange1d<DT>( porder, xref  , phi_i ) );
  (PXShapeUniformLagrange1d<DT>( porder, xref+1, phi_j ) );
  (PXShapeUniformLagrange1d<DT>( porder, xref+2, phi_k ) );

  l = 0;
  for (k=0; k<(porder+1); k++){
    for (j=0; j<(porder+1); j++){
      for (i=0; i<(porder+1); i++){
	phi[l] = phi_i[i]*phi_j[j]*phi_k[k];
	l++;
      }
    }
  }

  return 0;
}

/******************************************************************/
//   FUNCTION Definition: PXShapeQuadSpectralLagrange2d
template <typename DT> ELVIS_DEVICE int
PXShapeHexSpectralLagrange3d(const int porder, const DT * RESTRICT xref, DT * RESTRICT phi)
{
  int i,j,k,l;
  DT phi_i[6]; // 1d spectral lagrange basis functions
  DT phi_j[6]; // 1d spectral lagrange basis functions
  DT phi_k[6]; // 1d spectral lagrange basis functions

  /* move coordinates */
  (PXShapeSpectralLagrange1d<DT>( porder, xref  , phi_i ) );
  (PXShapeSpectralLagrange1d<DT>( porder, xref+1, phi_j ) );
  (PXShapeSpectralLagrange1d<DT>( porder, xref+2, phi_k ) );

  l = 0;
  for (k=0; k<(porder+1); k++){
    for (j=0; j<(porder+1); j++){
      for (i=0; i<(porder+1); i++){
        phi[l] = phi_i[i]*phi_j[j]*phi_k[k];
        l++;
      }
    }
  }

  return 0;
}


/******************************************************************/
//   FUNCTION Definition: PXGradientsHierarch1d
template <typename DT> ELVIS_DEVICE int
PXGradientsHierarch1d(const int porder, const DT * RESTRICT xref, DT * RESTRICT gphi)
{ 
  DT x;
  
  x = 2.0*xref[0] - 1.0;
  
  switch ( porder ) {

    /*--------------------------------------------------------------------*/
    /* Note: For porder > 0, we deliberately do not have break statements */
    /* to allow the case to fall through to lower p for hierarch basis    */
    /*--------------------------------------------------------------------*/
#if GEOM_USE_P5 && GEOM_USE_P4 && GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1 && GEOM_USE_P0
  case 5:
    gphi[5] = 2.0*sqrt(11.0)*0.125*(315.0*x*x*x*x - 210.0*x*x + 15.0);
#endif
#if GEOM_USE_P4 && GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1 && GEOM_USE_P0
  case 4:
    gphi[4] = 2.0*1.5*(35.0*x*x*x - 15.0*x);
#endif
#if GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1 && GEOM_USE_P0
  case 3:
    gphi[3] = 2.0*sqrt(7.0)*0.5*(15.0*x*x - 3.0);
#endif
#if GEOM_USE_P2 && GEOM_USE_P1 && GEOM_USE_P0
  case 2:
    gphi[2] = 2.0*sqrt(5.0)*0.5*(6.0*x);
#endif
#if GEOM_USE_P1 && GEOM_USE_P0
  case 1:
    gphi[1] = 2.0*sqrt(3.0);
#endif
#if GEOM_USE_P0
  case 0:
    gphi[0] = 2.0*0.0;
    return 0;
#endif
   default:
    return -1;
  }
}

/******************************************************************/
//   FUNCTION Definition: PXGradientsUniformLagrange1d
template <typename DT> ELVIS_DEVICE int
PXGradientsUniformLagrange1d(const int porder, const DT * RESTRICT xref, DT * RESTRICT gphi)
{
  DT x;

  x = xref[0];

  switch ( porder ) {
#if GEOM_USE_P0    
  case 0:
  {
    gphi[0] = 0.0;
    return 0;
  }
#endif
#if GEOM_USE_P1
  case 1:
  {
    gphi[0] = -1.0;
    gphi[1] = 1.0;
    return 0;
  }
#endif
#if GEOM_USE_P2
  case 2:
  {
    gphi[0] = -3.0 + 4.0*x;
    gphi[1] =  4.0 - 8.0*x;
    gphi[2] = -1.0 + 4.0*x;
    return 0;
  }
#endif
#if GEOM_USE_P3
  case 3:
  {
    DT xx = x*x;
    gphi[0] = -5.5 + 18.0*x - 13.5*xx;
    gphi[1] =  9.0 - 45.0*x + 40.5*xx;
    gphi[2] = -4.5 + 36.0*x - 40.5*xx;
    gphi[3] =  1.0 -  9.0*x + 13.5*xx;
    return 0;
  }
#endif
#if GEOM_USE_P4
  case 4:
  {
    DT xx = x*x;
    DT xxx = xx*x;
    gphi[0] = ( - 25.0 + 140.0*x - 240.0*xx + 128.0*xxx )*ONETHIRD;
    gphi[1] = (   48.0 - 416.0*x + 864.0*xx - 512.0*xxx )*ONETHIRD;
    gphi[2] = ( - 36.0 + 456.0*x -1152.0*xx + 768.0*xxx )*ONETHIRD;
    gphi[3] = (   16.0 - 224.0*x + 672.0*xx - 512.0*xxx )*ONETHIRD;
    gphi[4] = ( -  3.0 +  44.0*x - 144.0*xx + 128.0*xxx )*ONETHIRD;   
    return 0;
  }
#endif
#if GEOM_USE_P5
  case 5:
  {
    DT xx = x*x;
    DT xxx = xx*x;
    DT xxxx = xxx*x;
    gphi[0] = ( - 274.0 + 2250.0*x -  6375.0*xx +  7500.0*xxx - 3125.0*xxxx ) * ONETWENTYFOURTH;
    gphi[1] = (   600.0 - 7700.0*x + 26625.0*xx - 35000.0*xxx +15625.0*xxxx ) * ONETWENTYFOURTH;
    gphi[2] = ( - 600.0 +10700.0*x - 44250.0*xx + 65000.0*xxx -31250.0*xxxx ) * ONETWENTYFOURTH;
    gphi[3] = (   400.0 - 7800.0*x + 36750.0*xx - 60000.0*xxx +31250.0*xxxx ) * ONETWENTYFOURTH;
    gphi[4] = ( - 150.0 + 3050.0*x - 15375.0*xx + 27500.0*xxx -15625.0*xxxx ) * ONETWENTYFOURTH;
    gphi[5] = (    24.0 -  500.0*x +  2625.0*xx -  5000.0*xxx + 3125.0*xxxx ) * ONETWENTYFOURTH;
    return 0;
  }
#endif
  default:
    ALWAYS_PRINTF("PXGradientsUniformLagrange1d: Unknown order $d ", porder);
    return -1;
  }
}

/******************************************************************/
//   FUNCTION Definition: PXGradientsSpectralLagrange1d
template <typename DT> ELVIS_DEVICE int
PXGradientsSpectralLagrange1d(const int porder, const DT * RESTRICT xref, DT * RESTRICT gphi)
{
  DT x;
  //DT xx, xxx, xxxx;
  
  x = xref[0];

  switch ( porder ) {
#if GEOM_USE_P0
  case 0:
  {
    gphi[0] = 0.0;
    return 0;
  }
#endif
#if GEOM_USE_P1
  case 1:
  {
    gphi[0] = -1.0;
    gphi[1] = 1.0;
    return 0;
  }
#endif
#if GEOM_USE_P2
  case 2:
  {
    gphi[0] = -3.0 + 4.0*x;
    gphi[1] =  4.0 - 8.0*x;
    gphi[2] = -1.0 + 4.0*x;
    return 0;
  }
#endif
#if GEOM_USE_P3
  case 3:
  {
    DT xx=x*x;
    gphi[0] = ( - 19.0 + 64.0*x - 48.0*xx)*ONETHIRD;
    gphi[1] = (   24.0 -112.0*x + 96.0*xx)*ONETHIRD;
    gphi[2] = ( -  8.0 + 80.0*x - 96.0*xx)*ONETHIRD;
    gphi[3] = (    3.0 - 32.0*x + 48.0*xx)*ONETHIRD;
    return 0;
  }
#endif
#if GEOM_USE_P4
  case 4:
  {
    DT xx=x*x;
    DT xxx=xx*x;
    gphi[ 0] = - 11.0 + 68.0*x - 120.0*xx + 64.0*xxx;
    gphi[ 1] = 4.0*(2.0+SQUAREROOT2) - (24.0*SQUAREROOT2+80.0)*x + ( 24.0*SQUAREROOT2+192.0)*xx - 128.0*xxx;
    gphi[ 2] = -4.0 + 72.0*x - 192.0*xx + 128.0*xxx;
    gphi[ 3] = 4.0*(2.0-SQUAREROOT2) + (24.0*SQUAREROOT2-80.0)*x + (-24.0*SQUAREROOT2+192.0)*xx - 128.0*xxx;
    gphi[ 4] = -1.0 + 20.0*x - 72.0*xx + 64.0*xxx;
 
    return 0;
  }
#endif
#if GEOM_USE_P5
  case 5:
  {
    DT xx=x*x;
    DT xxx=xx*x;
    DT xxxx=xxx*x;
    
    gphi[0] = - 17.0 + 166.4*x - 508.8*xx + 614.4*xxx - 256.0*xxxx;
    gphi[1] = 4.0*(SQUAREROOT5+3.0) - (40.0*SQUAREROOT5+184.0)*x + ( 86.4*SQUAREROOT5+739.2)*xx - (51.2*SQUAREROOT5+1075.2)*xxx + 512.0*xxxx;
    gphi[2] =-0.8*(SQUAREROOT5+5.0) + (20.8*SQUAREROOT5+116.8)*x - ( 67.2*SQUAREROOT5+585.6)*xx + (51.2*SQUAREROOT5+ 972.8)*xxx - 512.0*xxxx;
    gphi[3] =-4.0*(SQUAREROOT5-3.0) + (40.0*SQUAREROOT5-184.0)*x + (-86.4*SQUAREROOT5+739.2)*xx + (51.2*SQUAREROOT5-1075.2)*xxx + 512.0*xxxx;
    gphi[4] = 0.8*(SQUAREROOT5-5.0) - (20.8*SQUAREROOT5-116.8)*x - (-67.2*SQUAREROOT5+585.6)*xx - (51.2*SQUAREROOT5- 972.8)*xxx - 512.0*xxxx;
    gphi[5] =  1.0 - 32.0*x + 201.6*xx - 409.6*xxx + 256.0*xxxx;
    return 0;
  }
#endif

  default:
    ALWAYS_PRINTF("PXGradientsSpectralLagrange1d: Unknown order $d ", porder);
    return -1;
  }
}

/******************************************************************/
//   FUNCTION Definition: PXGradientsHierarch2d
template <typename DT> ELVIS_DEVICE int
PXGradientsHierarch2d(const int porder, const DT * RESTRICT xref, DT * RESTRICT gphi)
{ 
  int n;
  DT x , y ;

  n = ((porder+1)*(porder+2))/2;
  
  x = xref[0];
  y = xref[1];

  if (porder <= 0)
    return -1;

#if GEOM_USE_P1
  if (porder >= 1){
    gphi[0] =  -1.0;
    gphi[1] =  1.0;
    gphi[2] =  0.0;
    gphi[n+0] =  -1.0;
    gphi[n+1] =  0.0;
    gphi[n+2] =  1.0;
  }
#endif
#if GEOM_USE_P2 && GEOM_USE_P1
  if (porder >= 2){
    gphi[3] =  -y*SQUAREROOT6;
    gphi[4] =  y*SQUAREROOT6;
    gphi[5] =  2.0*x*SQUAREROOT6-SQUAREROOT6+y*SQUAREROOT6;
    gphi[n+3] =  -x*SQUAREROOT6;
    gphi[n+4] =  2.0*y*SQUAREROOT6-SQUAREROOT6+x*SQUAREROOT6;
    gphi[n+5] =  x*SQUAREROOT6;
  }
#endif
#if GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1  
  if (porder >= 3){
    gphi[6] =  -y*y*SQUAREROOT10+2.0*x*y*SQUAREROOT10;
    gphi[7] =  2.0*y*SQUAREROOT10-2.0*x*y*SQUAREROOT10-3.0*y*y*SQUAREROOT10;
    gphi[8] =  6.0*x*x*SQUAREROOT10-6.0*x*SQUAREROOT10+6.0*x*y*SQUAREROOT10+SQUAREROOT10-2.0*y*SQUAREROOT10+y*y*SQUAREROOT10;
    gphi[9] =  -12.0*x*y+6.0*y-6.0*y*y;
    gphi[n+6] =  -2.0*x*y*SQUAREROOT10+x*x*SQUAREROOT10;
    gphi[n+7] =  6.0*y*SQUAREROOT10-6.0*x*y*SQUAREROOT10-6.0*y*y*SQUAREROOT10-SQUAREROOT10+2.0*x*SQUAREROOT10-x*x*SQUAREROOT10;
    gphi[n+8] =  3.0*x*x*SQUAREROOT10-2.0*x*SQUAREROOT10+2.0*x*y*SQUAREROOT10;
    gphi[n+9] =  -12.0*x*y+6.0*x-6.0*x*x;
  }
#endif
#if GEOM_USE_P4 && GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1
  if (porder >= 4){
    gphi[10] =  -5.0*0.25*y*y*y*SQUAREROOT14+5.0*y*y*SQUAREROOT14*x-15.0*0.25*y*SQUAREROOT14*x*x+y*SQUAREROOT14*0.25;
    gphi[11] =  3.5*y*SQUAREROOT14-7.5*x*y*SQUAREROOT14-12.5*y*y*SQUAREROOT14+15.0*0.25*y*SQUAREROOT14*x*x+12.5*y*y*SQUAREROOT14*x+10.0*y*y*y*SQUAREROOT14;
    gphi[12] =  20.0*x*x*x*SQUAREROOT14-30.0*x*x*SQUAREROOT14+30.0*y*SQUAREROOT14*x*x+12.0*x*SQUAREROOT14-25.0*x*y*SQUAREROOT14+12.5*y*y*SQUAREROOT14*x-SQUAREROOT14+3.5*y*SQUAREROOT14-15.0*0.25*y*y*SQUAREROOT14+5.0*0.25*y*y*y*SQUAREROOT14;
    gphi[13] =  6.0*x*x*y*SQUAREROOT5*SQUAREROOT3+2.0*y*y*SQUAREROOT5*SQUAREROOT3-4.0*x*y*SQUAREROOT5*SQUAREROOT3-2.0*y*y*y*SQUAREROOT5*SQUAREROOT3;
    gphi[14] =  -12.0*x*x*y*SQUAREROOT5*SQUAREROOT3+12.0*x*y*SQUAREROOT5*SQUAREROOT3-12.0*x*y*y*SQUAREROOT5*SQUAREROOT3-2.0*y*SQUAREROOT5*SQUAREROOT3+4.0*y*y*SQUAREROOT5*SQUAREROOT3-2.0*y*y*y*SQUAREROOT5*SQUAREROOT3;
    gphi[n+10] =  -15.0*0.25*y*y*SQUAREROOT14*x+5.0*y*SQUAREROOT14*x*x-5.0*0.25*x*x*x*SQUAREROOT14+x*SQUAREROOT14*0.25;
    gphi[n+11] =  12.0*y*SQUAREROOT14-25.0*x*y*SQUAREROOT14-30.0*y*y*SQUAREROOT14+12.5*y*SQUAREROOT14*x*x+30.0*y*y*SQUAREROOT14*x+20.0*y*y*y*SQUAREROOT14-SQUAREROOT14+3.5*x*SQUAREROOT14-15.0*0.25*x*x*SQUAREROOT14+5.0*0.25*x*x*x*SQUAREROOT14;
    gphi[n+12] =  10.0*x*x*x*SQUAREROOT14-12.5*x*x*SQUAREROOT14+12.5*y*SQUAREROOT14*x*x+3.5*x*SQUAREROOT14-7.5*x*y*SQUAREROOT14+15.0*0.25*y*y*SQUAREROOT14*x;
    gphi[n+13] =  -6.0*x*y*y*SQUAREROOT5*SQUAREROOT3+4.0*x*y*SQUAREROOT5*SQUAREROOT3-2.0*x*x*SQUAREROOT5*SQUAREROOT3+2.0*x*x*x*SQUAREROOT5*SQUAREROOT3;
    gphi[n+14] =  -12.0*x*x*y*SQUAREROOT5*SQUAREROOT3+8.0*x*y*SQUAREROOT5*SQUAREROOT3-6.0*x*y*y*SQUAREROOT5*SQUAREROOT3+6.0*x*x*SQUAREROOT5*SQUAREROOT3-2.0*x*SQUAREROOT5*SQUAREROOT3-4.0*x*x*x*SQUAREROOT5*SQUAREROOT3;
  }
#endif

#if GEOM_USE_P5 && GEOM_USE_P4 && GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1
  if (porder >= 5){
    gphi[15] =  -21.0*0.25*y*y*y*y*SQUAREROOT2+31.5*y*y*y*SQUAREROOT2*x-189.0*0.25*y*y*SQUAREROOT2*x*x+21.0*y*SQUAREROOT2*x*x*x+9.0*0.25*y*y*SQUAREROOT2-4.5*y*SQUAREROOT2*x;
    gphi[16] =  16.5*y*SQUAREROOT2-189.0*y*y*y*SQUAREROOT2*x-441.0*0.25*y*y*SQUAREROOT2*x*x-21.0*y*SQUAREROOT2*x*x*x-58.5*y*SQUAREROOT2*x-105.0*y*y*y*y*SQUAREROOT2-103.5*y*y*SQUAREROOT2+63.0*y*SQUAREROOT2*x*x+220.5*y*y*SQUAREROOT2*x+189.0*y*y*y*SQUAREROOT2;
    gphi[17] =  -16.5*y*SQUAREROOT2+73.5*y*y*y*SQUAREROOT2*x+283.5*y*y*SQUAREROOT2*x*x+420.0*y*SQUAREROOT2*x*x*x+207.0*y*SQUAREROOT2*x+3.0*SQUAREROOT2+21.0*0.25*y*y*y*y*SQUAREROOT2+117.0*0.25*y*y*SQUAREROOT2-567.0*y*SQUAREROOT2*x*x-220.5*y*y*SQUAREROOT2*x-21.0*y*y*y*SQUAREROOT2-60.0*x*SQUAREROOT2+210.0*x*x*x*x*SQUAREROOT2-420.0*x*x*x*SQUAREROOT2+270.0*x*x*SQUAREROOT2;
    gphi[18] =  5.0*x*y*y*y*SQUAREROOT7*SQUAREROOT3+7.5*x*x*y*y*SQUAREROOT7*SQUAREROOT3-10.0*x*x*x*y*SQUAREROOT7*SQUAREROOT3+x*y*SQUAREROOT7*SQUAREROOT3+2.5*y*y*y*SQUAREROOT7*SQUAREROOT3-10.0*y*y*SQUAREROOT7*SQUAREROOT3*x+7.5*y*SQUAREROOT7*SQUAREROOT3*x*x-y*SQUAREROOT7*SQUAREROOT3*0.5-2.5*y*y*y*y*SQUAREROOT7*SQUAREROOT3+y*y*SQUAREROOT7*SQUAREROOT3*0.5;
    gphi[19] =  30.0*y*y*x*x+20.0*x*y*y-40.0*y*y*y*x+80.0*y*x*x*x-90.0*x*x*y-10.0*y*y+20.0*y*y*y+20.0*x*y-10.0*y*y*y*y;
    gphi[20] =  -40.0*x*x*x*y*SQUAREROOT7*SQUAREROOT3+60.0*y*SQUAREROOT7*SQUAREROOT3*x*x-60.0*x*x*y*y*SQUAREROOT7*SQUAREROOT3-24.0*x*y*SQUAREROOT7*SQUAREROOT3+50.0*y*y*SQUAREROOT7*SQUAREROOT3*x-25.0*x*y*y*y*SQUAREROOT7*SQUAREROOT3+2.0*y*SQUAREROOT7*SQUAREROOT3-7.0*y*y*SQUAREROOT7*SQUAREROOT3+7.5*y*y*y*SQUAREROOT7*SQUAREROOT3-2.5*y*y*y*y*SQUAREROOT7*SQUAREROOT3;
    gphi[n+15] =  -21.0*y*y*y*SQUAREROOT2*x+189.0*0.25*y*y*SQUAREROOT2*x*x-31.5*y*SQUAREROOT2*x*x*x+21.0*0.25*x*x*x*x*SQUAREROOT2+4.5*y*SQUAREROOT2*x-9.0*0.25*x*x*SQUAREROOT2;
    gphi[n+16] =  -420.0*y*y*y*SQUAREROOT2*x-283.5*y*y*SQUAREROOT2*x*x-73.5*y*SQUAREROOT2*x*x*x-207.0*y*SQUAREROOT2*x+16.5*x*SQUAREROOT2+220.5*y*SQUAREROOT2*x*x+567.0*y*y*SQUAREROOT2*x+60.0*y*SQUAREROOT2-3.0*SQUAREROOT2-210.0*y*y*y*y*SQUAREROOT2-270.0*y*y*SQUAREROOT2-21.0*0.25*x*x*x*x*SQUAREROOT2-117.0*0.25*x*x*SQUAREROOT2+21.0*x*x*x*SQUAREROOT2+420.0*y*y*y*SQUAREROOT2;
    gphi[n+17] =  21.0*y*y*y*SQUAREROOT2*x+441.0*0.25*y*y*SQUAREROOT2*x*x+189.0*y*SQUAREROOT2*x*x*x+58.5*y*SQUAREROOT2*x-16.5*x*SQUAREROOT2-220.5*y*SQUAREROOT2*x*x-63.0*y*y*SQUAREROOT2*x+105.0*x*x*x*x*SQUAREROOT2+103.5*x*x*SQUAREROOT2-189.0*x*x*x*SQUAREROOT2;
    gphi[n+18] =  -10.0*x*y*y*y*SQUAREROOT7*SQUAREROOT3+7.5*x*x*y*y*SQUAREROOT7*SQUAREROOT3+5.0*x*x*x*y*SQUAREROOT7*SQUAREROOT3+x*y*SQUAREROOT7*SQUAREROOT3+7.5*y*y*SQUAREROOT7*SQUAREROOT3*x-10.0*y*SQUAREROOT7*SQUAREROOT3*x*x+2.5*x*x*x*SQUAREROOT7*SQUAREROOT3-x*SQUAREROOT7*SQUAREROOT3*0.5-2.5*x*x*x*x*SQUAREROOT7*SQUAREROOT3+x*x*SQUAREROOT7*SQUAREROOT3*0.5;
    gphi[n+19] =  -60.0*y*y*x*x+60.0*x*y*y-40.0*y*y*y*x+20.0*y*x*x*x+20.0*x*x*y-20.0*x*y-30.0*x*x*x+10.0*x*x+20.0*x*x*x*x;
    gphi[n+20] =  -40.0*x*x*x*y*SQUAREROOT7*SQUAREROOT3+50.0*y*SQUAREROOT7*SQUAREROOT3*x*x-37.5*x*x*y*y*SQUAREROOT7*SQUAREROOT3-14.0*x*y*SQUAREROOT7*SQUAREROOT3+22.5*y*y*SQUAREROOT7*SQUAREROOT3*x-10.0*x*y*y*y*SQUAREROOT7*SQUAREROOT3+20.0*x*x*x*SQUAREROOT7*SQUAREROOT3-12.0*x*x*SQUAREROOT7*SQUAREROOT3+2.0*x*SQUAREROOT7*SQUAREROOT3-10.0*x*x*x*x*SQUAREROOT7*SQUAREROOT3;
  }
#endif 
  if (porder >= 6) {
    ALWAYS_PRINTF("PXGradientsHierarch2d: Unknown order $d ", porder);
    return -1;
  }

  return 0;

} // PXGradientsHierarch2d


/******************************************************************/
//   FUNCTION Definition: PXGradientsLagrange2d
template <typename DT> ELVIS_DEVICE int
PXGradientsLagrange2d(const int porder, const DT * RESTRICT xref, DT * RESTRICT gphi)
{ 
  DT x, y;
  
  x = xref[0];
  y = xref[1];

  switch (porder){
#if GEOM_USE_P0    
  case 0:
    gphi[0] =  0.0;
    gphi[1] =  0.0; 
    return 0;
#endif
#if GEOM_USE_P1
  case 1:
    gphi[0] =  -1.0;
    gphi[1] =  1.0;
    gphi[2] =  0.0;
    gphi[3] =  -1.0;
    gphi[4] =  0.0;
    gphi[5] =  1.0;

    return 0;
#endif
#if GEOM_USE_P2
  case 2:
    gphi[0] =  -3.0+4.0*x+4.0*y;
    gphi[1] =  -1.0+4.0*x;
    gphi[2] =  0.0;
    gphi[3] =  4.0*y;
    gphi[4] =  -4.0*y;
    gphi[5] =  4.0-8.0*x-4.0*y;
    gphi[6] =  -3.0+4.0*x+4.0*y;
    gphi[7] =  0.0;
    gphi[8] =  -1.0+4.0*y;
    gphi[9] =  4.0*x;
    gphi[10] =  4.0-4.0*x-8.0*y;
    gphi[11] =  -4.0*x;

    return 0;
#endif
#if GEOM_USE_P3
  case 3:
    gphi[0] =  -5.5+18.0*x+18.0*y-13.5*x*x-27.0*x*y-13.5*y*y;
    gphi[1] =  1.0-9.0*x+13.5*x*x;
    gphi[2] =  0.0;
    gphi[3] =  -4.5*y+27.0*x*y;
    gphi[4] =  -4.5*y+13.5*y*y;
    gphi[5] =  4.5*y-13.5*y*y;
    gphi[6] =  -22.5*y+27.0*x*y+27.0*y*y;
    gphi[7] =  9.0-45.0*x-22.5*y+40.5*x*x+54.0*x*y+13.5*y*y;
    gphi[8] =  -4.5+36.0*x+4.5*y-40.5*x*x-27.0*x*y;
    gphi[9] =  27.0*y-54.0*x*y-27.0*y*y;
    gphi[10] =  -5.5+18.0*x+18.0*y-13.5*x*x-27.0*x*y-13.5*y*y;
    gphi[11] =  0.0;
    gphi[12] =  1.0-9.0*y+13.5*y*y;
    gphi[13] =  -4.5*x+13.5*x*x;
    gphi[14] =  -4.5*x+27.0*x*y;
    gphi[15] =  -4.5+4.5*x+36.0*y-27.0*x*y-40.5*y*y;
    gphi[16] =  9.0-22.5*x-45.0*y+13.5*x*x+54.0*x*y+40.5*y*y;
    gphi[17] =  -22.5*x+27.0*x*x+27.0*x*y;
    gphi[18] =  4.5*x-13.5*x*x;
    gphi[19] =  27.0*x-27.0*x*x-54.0*x*y;

    return 0;
#endif
#if GEOM_USE_P4
  case 4:
    gphi[0] =  -25.0*ONETHIRD+140.0*ONETHIRD*x+140.0*ONETHIRD*y-80.0*x*x-160.0*x*y-80.0*y*y+128.0*ONETHIRD*x*x*x+128.0*x*x*y+128.0*x*y*y+128.0*ONETHIRD*y*y*y;
    gphi[1] =  -1.0+44.0*ONETHIRD*x-48.0*x*x+128.0*ONETHIRD*x*x*x;
    gphi[2] =  0.0;
    gphi[3] =  16.0*ONETHIRD*y-64.0*x*y+128.0*x*x*y;
    gphi[4] =  4.0*y-32.0*x*y-16.0*y*y+128.0*x*y*y;
    gphi[5] =  16.0*ONETHIRD*y-32.0*y*y+128.0*ONETHIRD*y*y*y;
    gphi[6] =  -16.0*ONETHIRD*y+32.0*y*y-128.0*ONETHIRD*y*y*y;
    gphi[7] =  28.0*y-32.0*x*y-144.0*y*y+128.0*x*y*y+128.0*y*y*y;
    gphi[8] =  -208.0*ONETHIRD*y+192.0*x*y+192.0*y*y-128.0*x*x*y-256.0*x*y*y-128.0*y*y*y;
    gphi[9] =  16.0-416.0*ONETHIRD*x-208.0*ONETHIRD*y+288.0*x*x+384.0*x*y+96.0*y*y-512.0*ONETHIRD*x*x*x-384.0*x*x*y-256.0*x*y*y-128.0*ONETHIRD*y*y*y;
    gphi[10] =  -12.0+152.0*x+28.0*y-384.0*x*x-288.0*x*y-16.0*y*y+256.0*x*x*x+384.0*x*x*y+128.0*x*y*y;
    gphi[11] =  16.0*ONETHIRD-224.0*ONETHIRD*x-16.0*ONETHIRD*y+224.0*x*x+64.0*x*y-512.0*ONETHIRD*x*x*x-128.0*x*x*y;
    gphi[12] =  96.0*y-448.0*x*y-224.0*y*y+384.0*x*x*y+512.0*x*y*y+128.0*y*y*y;
    gphi[13] =  -32.0*y+320.0*x*y+32.0*y*y-384.0*x*x*y-256.0*x*y*y;
    gphi[14] =  -32.0*y+64.0*x*y+160.0*y*y-256.0*x*y*y-128.0*y*y*y;
    gphi[15] =  -25.0*ONETHIRD+140.0*ONETHIRD*x+140.0*ONETHIRD*y-80.0*x*x-160.0*x*y-80.0*y*y+128.0*ONETHIRD*x*x*x+128.0*x*x*y+128.0*x*y*y+128.0*ONETHIRD*y*y*y;
    gphi[16] =  0.0;
    gphi[17] =  -1.0+44.0*ONETHIRD*y-48.0*y*y+128.0*ONETHIRD*y*y*y;
    gphi[18] =  16.0*ONETHIRD*x-32.0*x*x+128.0*ONETHIRD*x*x*x;
    gphi[19] =  4.0*x-16.0*x*x-32.0*x*y+128.0*x*x*y;
    gphi[20] =  16.0*ONETHIRD*x-64.0*x*y+128.0*x*y*y;
    gphi[21] =  16.0*ONETHIRD-16.0*ONETHIRD*x-224.0*ONETHIRD*y+64.0*x*y+224.0*y*y-128.0*x*y*y-512.0*ONETHIRD*y*y*y;
    gphi[22] =  -12.0+28.0*x+152.0*y-16.0*x*x-288.0*x*y-384.0*y*y+128.0*x*x*y+384.0*x*y*y+256.0*y*y*y;
    gphi[23] =  16.0-208.0*ONETHIRD*x-416.0*ONETHIRD*y+96.0*x*x+384.0*x*y+288.0*y*y-128.0*ONETHIRD*x*x*x-256.0*x*x*y-384.0*x*y*y-512.0*ONETHIRD*y*y*y;
    gphi[24] =  -208.0*ONETHIRD*x+192.0*x*x+192.0*x*y-128.0*x*x*x-256.0*x*x*y-128.0*x*y*y;
    gphi[25] =  28.0*x-144.0*x*x-32.0*x*y+128.0*x*x*x+128.0*x*x*y;
    gphi[26] =  -16.0*ONETHIRD*x+32.0*x*x-128.0*ONETHIRD*x*x*x;
    gphi[27] =  96.0*x-224.0*x*x-448.0*x*y+128.0*x*x*x+512.0*x*x*y+384.0*x*y*y;
    gphi[28] =  -32.0*x+160.0*x*x+64.0*x*y-128.0*x*x*x-256.0*x*x*y;
    gphi[29] =  -32.0*x+32.0*x*x+320.0*x*y-256.0*x*x*y-384.0*x*y*y;
    return 0;
#endif
#if GEOM_USE_P5
  case 5:
    gphi[0] =  -137.0/12.0+375.0*0.25*x+375.0*0.25*y-2125.0/8.0*x*x-2125.0*0.25*x*y-2125.0/8.0*y*y+312.5*x*x*x+937.5*x*x*y+937.5*x*y*y+312.5*y*y*y-3125.0/24.0*x*x*x*x-3125.0*ONESIXTH*x*x*x*y-3125.0*0.25*x*x*y*y-3125.0*ONESIXTH*x*y*y*y-3125.0/24.0*y*y*y*y;
    gphi[1] =  1.0-125.0*ONESIXTH*x+875.0/8.0*x*x-625.0*ONETHIRD*x*x*x+3125.0/24.0*x*x*x*x;
    gphi[2] =  0.0;
    gphi[3] =  -25.0*0.25*y+1375.0/12.0*x*y-1875.0*0.25*x*x*y+3125.0*ONESIXTH*x*x*x*y;
    gphi[4] =  -25.0*ONESIXTH*y+62.5*x*y+125.0*ONESIXTH*y*y-625.0*0.25*x*x*y-312.5*x*y*y+3125.0*0.25*x*x*y*y;
    gphi[5] =  -25.0*ONESIXTH*y+125.0*ONETHIRD*x*y+125.0*0.25*y*y-312.5*x*y*y-625.0/12.0*y*y*y+3125.0*ONESIXTH*x*y*y*y;
    gphi[6] =  -25.0*0.25*y+1375.0/24.0*y*y-625.0*0.25*y*y*y+3125.0/24.0*y*y*y*y;
    gphi[7] =  25.0*0.25*y-1375.0/24.0*y*y+625.0*0.25*y*y*y-3125.0/24.0*y*y*y*y;
    gphi[8] =  -37.5*y+125.0*ONETHIRD*x*y+3875.0/12.0*y*y-312.5*x*y*y-3125.0*0.25*y*y*y+3125.0*ONESIXTH*x*y*y*y+3125.0*ONESIXTH*y*y*y*y;
    gphi[9] =  1175.0/12.0*y-250.0*x*y-8875.0/12.0*y*y+625.0*0.25*x*x*y+1562.5*x*y*y+5625.0*0.25*y*y*y-3125.0*0.25*x*x*y*y-1562.5*x*y*y*y-3125.0*0.25*y*y*y*y;
    gphi[10] =  -1925.0/12.0*y+8875.0/12.0*x*y+8875.0/12.0*y*y-4375.0*0.25*x*x*y-2187.5*x*y*y-4375.0*0.25*y*y*y+3125.0*ONESIXTH*x*x*x*y+1562.5*x*x*y*y+1562.5*x*y*y*y+3125.0*ONESIXTH*y*y*y*y;
    gphi[11] =  25.0-1925.0*ONESIXTH*x-1925.0/12.0*y+8875.0/8.0*x*x+8875.0*ONESIXTH*x*y+8875.0/24.0*y*y-4375.0*ONETHIRD*x*x*x-13125.0*0.25*x*x*y-2187.5*x*y*y-4375.0/12.0*y*y*y+15625.0/24.0*x*x*x*x+6250.0*ONETHIRD*x*x*x*y+9375.0*0.25*x*x*y*y+3125.0*ONETHIRD*x*y*y*y+3125.0/24.0*y*y*y*y;
    gphi[12] =  -25.0+2675.0*ONESIXTH*x+1175.0/12.0*y-7375.0*0.25*x*x-8875.0*ONESIXTH*x*y-125.0*y*y+8125.0*ONETHIRD*x*x*x+16875.0*0.25*x*x*y+1562.5*x*y*y+625.0/12.0*y*y*y-15625.0/12.0*x*x*x*x-3125.0*x*x*x*y-9375.0*0.25*x*x*y*y-3125.0*ONESIXTH*x*y*y*y;
    gphi[13] =  50.0*ONETHIRD-325.0*x-37.5*y+6125.0*0.25*x*x+3875.0*ONESIXTH*x*y+125.0*ONESIXTH*y*y-2500.0*x*x*x-9375.0*0.25*x*x*y-312.5*x*y*y+15625.0/12.0*x*x*x*x+6250.0*ONETHIRD*x*x*x*y+3125.0*0.25*x*x*y*y;
    gphi[14] =  -25.0*0.25+1525.0/12.0*x+25.0*0.25*y-5125.0/8.0*x*x-1375.0/12.0*x*y+6875.0*ONESIXTH*x*x*x+1875.0*0.25*x*x*y-15625.0/24.0*x*x*x*x-3125.0*ONESIXTH*x*x*x*y;
    gphi[15] =  250.0*y-5875.0*ONETHIRD*x*y-5875.0*ONESIXTH*y*y+3750.0*x*x*y+5000.0*x*y*y+1250.0*y*y*y-6250.0*ONETHIRD*x*x*x*y-4687.5*x*x*y*y-3125.0*x*y*y*y-3125.0*ONESIXTH*y*y*y*y;
    gphi[16] =  -125.0*y+1812.5*x*y+1125.0*0.25*y*y-4687.5*x*x*y-3437.5*x*y*y-625.0*0.25*y*y*y+3125.0*x*x*x*y+4687.5*x*x*y*y+1562.5*x*y*y*y;
    gphi[17] =  125.0*ONETHIRD*y-2125.0*ONETHIRD*x*y-125.0*ONETHIRD*y*y+2500.0*x*x*y+625.0*x*y*y-6250.0*ONETHIRD*x*x*x*y-1562.5*x*x*y*y;
    gphi[18] =  -125.0*y+562.5*x*y+3625.0*0.25*y*y-1875.0*0.25*x*x*y-3437.5*x*y*y-1562.5*y*y*y+9375.0*0.25*x*x*y*y+3125.0*x*y*y*y+3125.0*0.25*y*y*y*y;
    gphi[19] =  125.0*0.25*y-375.0*x*y-187.5*y*y+1875.0*0.25*x*x*y+2187.5*x*y*y+625.0*0.25*y*y*y-9375.0*0.25*x*x*y*y-1562.5*x*y*y*y;
    gphi[20] =  125.0*ONETHIRD*y-250.0*ONETHIRD*x*y-2125.0*ONESIXTH*y*y+625.0*x*y*y+2500.0*ONETHIRD*y*y*y-3125.0*ONETHIRD*x*y*y*y-3125.0*ONESIXTH*y*y*y*y;
    gphi[21] =  -137.0/12.0+375.0*0.25*x+375.0*0.25*y-2125.0/8.0*x*x-2125.0*0.25*x*y-2125.0/8.0*y*y+312.5*x*x*x+937.5*x*x*y+937.5*x*y*y+312.5*y*y*y-3125.0/24.0*x*x*x*x-3125.0*ONESIXTH*x*x*x*y-3125.0*0.25*x*x*y*y-3125.0*ONESIXTH*x*y*y*y-3125.0/24.0*y*y*y*y;
    gphi[22] =  0.0;
    gphi[23] =  1.0-125.0*ONESIXTH*y+875.0/8.0*y*y-625.0*ONETHIRD*y*y*y+3125.0/24.0*y*y*y*y;
    gphi[24] =  -25.0*0.25*x+1375.0/24.0*x*x-625.0*0.25*x*x*x+3125.0/24.0*x*x*x*x;
    gphi[25] =  -25.0*ONESIXTH*x+125.0*0.25*x*x+125.0*ONETHIRD*x*y-625.0/12.0*x*x*x-312.5*x*x*y+3125.0*ONESIXTH*x*x*x*y;
    gphi[26] =  -25.0*ONESIXTH*x+125.0*ONESIXTH*x*x+62.5*x*y-312.5*x*x*y-625.0*0.25*x*y*y+3125.0*0.25*x*x*y*y;
    gphi[27] =  -25.0*0.25*x+1375.0/12.0*x*y-1875.0*0.25*x*y*y+3125.0*ONESIXTH*x*y*y*y;
    gphi[28] =  -25.0*0.25+25.0*0.25*x+1525.0/12.0*y-1375.0/12.0*x*y-5125.0/8.0*y*y+1875.0*0.25*x*y*y+6875.0*ONESIXTH*y*y*y-3125.0*ONESIXTH*x*y*y*y-15625.0/24.0*y*y*y*y;
    gphi[29] =  50.0*ONETHIRD-37.5*x-325.0*y+125.0*ONESIXTH*x*x+3875.0*ONESIXTH*x*y+6125.0*0.25*y*y-312.5*x*x*y-9375.0*0.25*x*y*y-2500.0*y*y*y+3125.0*0.25*x*x*y*y+6250.0*ONETHIRD*x*y*y*y+15625.0/12.0*y*y*y*y;
    gphi[30] =  -25.0+1175.0/12.0*x+2675.0*ONESIXTH*y-125.0*x*x-8875.0*ONESIXTH*x*y-7375.0*0.25*y*y+625.0/12.0*x*x*x+1562.5*x*x*y+16875.0*0.25*x*y*y+8125.0*ONETHIRD*y*y*y-3125.0*ONESIXTH*x*x*x*y-9375.0*0.25*x*x*y*y-3125.0*x*y*y*y-15625.0/12.0*y*y*y*y;
    gphi[31] =  25.0-1925.0/12.0*x-1925.0*ONESIXTH*y+8875.0/24.0*x*x+8875.0*ONESIXTH*x*y+8875.0/8.0*y*y-4375.0/12.0*x*x*x-2187.5*x*x*y-13125.0*0.25*x*y*y-4375.0*ONETHIRD*y*y*y+3125.0/24.0*x*x*x*x+3125.0*ONETHIRD*x*x*x*y+9375.0*0.25*x*x*y*y+6250.0*ONETHIRD*x*y*y*y+15625.0/24.0*y*y*y*y;
    gphi[32] =  -1925.0/12.0*x+8875.0/12.0*x*x+8875.0/12.0*x*y-4375.0*0.25*x*x*x-2187.5*x*x*y-4375.0*0.25*x*y*y+3125.0*ONESIXTH*x*x*x*x+1562.5*x*x*x*y+1562.5*x*x*y*y+3125.0*ONESIXTH*x*y*y*y;
    gphi[33] =  1175.0/12.0*x-8875.0/12.0*x*x-250.0*x*y+5625.0*0.25*x*x*x+1562.5*x*x*y+625.0*0.25*x*y*y-3125.0*0.25*x*x*x*x-1562.5*x*x*x*y-3125.0*0.25*x*x*y*y;
    gphi[34] =  -37.5*x+3875.0/12.0*x*x+125.0*ONETHIRD*x*y-3125.0*0.25*x*x*x-312.5*x*x*y+3125.0*ONESIXTH*x*x*x*x+3125.0*ONESIXTH*x*x*x*y;
    gphi[35] =  25.0*0.25*x-1375.0/24.0*x*x+625.0*0.25*x*x*x-3125.0/24.0*x*x*x*x;
    gphi[36] =  250.0*x-5875.0*ONESIXTH*x*x-5875.0*ONETHIRD*x*y+1250.0*x*x*x+5000.0*x*x*y+3750.0*x*y*y-3125.0*ONESIXTH*x*x*x*x-3125.0*x*x*x*y-4687.5*x*x*y*y-6250.0*ONETHIRD*x*y*y*y;
    gphi[37] =  -125.0*x+3625.0*0.25*x*x+562.5*x*y-1562.5*x*x*x-3437.5*x*x*y-1875.0*0.25*x*y*y+3125.0*0.25*x*x*x*x+3125.0*x*x*x*y+9375.0*0.25*x*x*y*y;
    gphi[38] =  125.0*ONETHIRD*x-2125.0*ONESIXTH*x*x-250.0*ONETHIRD*x*y+2500.0*ONETHIRD*x*x*x+625.0*x*x*y-3125.0*ONESIXTH*x*x*x*x-3125.0*ONETHIRD*x*x*x*y;
    gphi[39] =  -125.0*x+1125.0*0.25*x*x+1812.5*x*y-625.0*0.25*x*x*x-3437.5*x*x*y-4687.5*x*y*y+1562.5*x*x*x*y+4687.5*x*x*y*y+3125.0*x*y*y*y;
    gphi[40] =  125.0*0.25*x-187.5*x*x-375.0*x*y+625.0*0.25*x*x*x+2187.5*x*x*y+1875.0*0.25*x*y*y-1562.5*x*x*x*y-9375.0*0.25*x*x*y*y;
    gphi[41] =  125.0*ONETHIRD*x-125.0*ONETHIRD*x*x-2125.0*ONETHIRD*x*y+625.0*x*x*y+2500.0*x*y*y-1562.5*x*x*y*y-6250.0*ONETHIRD*x*y*y*y;
    return 0;
#endif
  default:
    ALWAYS_PRINTF("PXGradientsLagrange2d: Unknown order $d ", porder);
    return -1;
  }
} // PXGradientsLagrange2d


/******************************************************************/
//   FUNCTION Definition: PXGradientsQuadUniformLagrange2d
template <typename DT> ELVIS_DEVICE int
PXGradientsQuadUniformLagrange2d(const int porder, const DT * RESTRICT xref, DT * RESTRICT gphi)
{ 
  int ierr;
  int i,j;
  int nDOF;
  DT xi[2];
  DT eta[2];
  DT  phi_i[6]; // 1d uniform lagrange basis functions
  DT  phi_j[6]; // 1d uniform lagrange basis functions
  DT gphi_xi_i[6]; // gradient of 1d uniform lagrange basis functions with respect to xi
  DT gphi_eta_j[6]; // gradient of 1d uniform lagrange basis functions with respect to eta

  /* get number of degrees of freedom */
  nDOF = (porder+1)*(porder+1);

  /* move coordinates */
  xi[0]  = xref[0];
  eta[0] = xref[1];

  ierr = PXShapeUniformLagrange1d<DT>( porder, xi , phi_i );
  if (ierr!=0) return ierr;
  ierr = PXShapeUniformLagrange1d<DT>( porder, eta, phi_j );
  if (ierr!=0) return ierr;
  ierr = PXGradientsUniformLagrange1d<DT>( porder, xi , gphi_xi_i );
  if (ierr!=0) return ierr;
  ierr = PXGradientsUniformLagrange1d<DT>( porder, eta, gphi_eta_j );
  if (ierr!=0) return ierr;

  /* compute dphi/dxi */
  for (j=0; j<(porder+1); j++){
    for (i=0; i<(porder+1); i++){
      gphi[j*(porder+1)+i] = gphi_xi_i[i]*phi_j[j];
    }
  }
  /* compute dphi/deta */
  for (j=0; j<(porder+1); j++){
    for (i=0; i<(porder+1); i++){
      gphi[nDOF+j*(porder+1)+i] = phi_i[i]*gphi_eta_j[j];
    }
  }

  return 0;
}


/******************************************************************/
//   FUNCTION Definition: PXGradientsQuadSpectralLagrange2d
template <typename DT> ELVIS_DEVICE int
PXGradientsQuadSpectralLagrange2d(const int porder, const DT * RESTRICT xref, DT * RESTRICT gphi)
{ 
  int ierr;
  int i,j;
  int nDOF;
  DT xi[2];
  DT eta[2];
  DT  phi_i[6]; // 1d spectral lagrange basis functions
  DT  phi_j[6]; // 1d spectral lagrange basis functions
  DT gphi_xi_i[6]; // gradient of 1d spectral lagrange basis functions with respect to xi
  DT gphi_eta_j[6]; // gradient of 1d spectral lagrange basis functions with respect to eta

  /* get number of degrees of freedom */
  nDOF = (porder+1)*(porder+1);

  /* move coordinates */
  xi[0]  = xref[0];
  eta[0] = xref[1];

  ierr = PXShapeSpectralLagrange1d<DT>( porder, xi , phi_i );
  if (ierr!=0) return ierr;
  ierr = PXShapeSpectralLagrange1d<DT>( porder, eta, phi_j );
  if (ierr!=0) return ierr;
  ierr = PXGradientsSpectralLagrange1d<DT>( porder, xi , gphi_xi_i );
  if (ierr!=0) return ierr;
  ierr = PXGradientsSpectralLagrange1d<DT>( porder, eta, gphi_eta_j );
  if (ierr!=0) return ierr;

  /* compute dphi/dxi */
  for (j=0; j<(porder+1); j++){
    for (i=0; i<(porder+1); i++){
      gphi[j*(porder+1)+i] = gphi_xi_i[i]*phi_j[j];
    }
  }
  /* compute dphi/deta */
  for (j=0; j<(porder+1); j++){
    for (i=0; i<(porder+1); i++){
      gphi[nDOF+j*(porder+1)+i] = phi_i[i]*gphi_eta_j[j];
    }
  }

  return 0;
}


/******************************************************************/
//   FUNCTION Definition: PXGradientsHierarch3d
template <typename DT> ELVIS_DEVICE int
PXGradientsHierarch3d(const int porder, const DT * RESTRICT xref, DT * RESTRICT gphi)
{
  int n, nn;
  DT x , y , z ;

  n = ((porder+1)*(porder+2)*(porder+3)) / 6;
  nn = 2*n;
  
  x = xref[0]; 
  y = xref[1]; 
  z = xref[2];

  if (porder <= 0)
    return -1;
#if GEOM_USE_P1
  if (porder >= 1){
    gphi[0] = -1.0;
    gphi[1] = 1.0;
    gphi[2] = 0.0;
    gphi[3] = 0.0;
    gphi[n+0] = -1.0;
    gphi[n+1] = 0.0;
    gphi[n+2] = 1.0;
    gphi[n+3] = 0.0;
    gphi[nn+0] = -1.0;
    gphi[nn+1] = 0.0;
    gphi[nn+2] = 0.0;
    gphi[nn+3] = 1.0;
   
  }
#endif
#if GEOM_USE_P2 && GEOM_USE_P1
  if (porder >= 2){
   gphi[4] = 0.0;
    gphi[5] = -z*SQUAREROOT6;
    gphi[6] = z*SQUAREROOT6;
    gphi[7] = -y*SQUAREROOT6;
    gphi[8] = y*SQUAREROOT6;
    gphi[9] = (2.0*x-1.0+y+z)*SQUAREROOT6;
    gphi[n+4] = -z*SQUAREROOT6;
    gphi[n+5] = 0.0;
    gphi[n+6] = z*SQUAREROOT6;
    gphi[n+7] = -x*SQUAREROOT6;
    gphi[n+8] = (2.0*y-1.0+x+z)*SQUAREROOT6;
    gphi[n+9] = x*SQUAREROOT6;
    gphi[nn+4] = -y*SQUAREROOT6;
    gphi[nn+5] = -x*SQUAREROOT6;
    gphi[nn+6] = (2.0*z-1.0+x+y)*SQUAREROOT6;
    gphi[nn+7] = 0.0;
    gphi[nn+8] = y*SQUAREROOT6;
    gphi[nn+9] = x*SQUAREROOT6;
  }
#endif
#if GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1
  if (porder >= 3){
    gphi[10] = 0.0;
    gphi[11] = -z*z*SQUAREROOT10+2.0*x*z*SQUAREROOT10;
    gphi[12] = 3.0*z*z*SQUAREROOT10-2.0*z*SQUAREROOT10+2.0*x*z*SQUAREROOT10+2.0*z*SQUAREROOT10*y;
    gphi[13] = -y*y*SQUAREROOT10+2.0*x*y*SQUAREROOT10;
    gphi[14] = 3.0*y*y*SQUAREROOT10-2.0*y*SQUAREROOT10+2.0*x*y*SQUAREROOT10+2.0*z*SQUAREROOT10*y;
    gphi[15] = 6.0*x*x*SQUAREROOT10-6.0*x*SQUAREROOT10+6.0*x*y*SQUAREROOT10+6.0*x*z*SQUAREROOT10+SQUAREROOT10-2.0*y*SQUAREROOT10-2.0*z*SQUAREROOT10+y*y*SQUAREROOT10+2.0*z*SQUAREROOT10*y+z*z*SQUAREROOT10;
    gphi[16] = 6.0*y*z;
    gphi[17] = -6.0*y*z;
    gphi[18] = -12.0*x*z+6.0*z-6.0*y*z-6.0*z*z;
    gphi[19] = -12.0*x*y+6.0*y-6.0*y*y-6.0*y*z;
    gphi[n+10] = -z*z*SQUAREROOT10+2.0*z*SQUAREROOT10*y;
    gphi[n+11] = 0.0;
    gphi[n+12] = 3.0*z*z*SQUAREROOT10-2.0*z*SQUAREROOT10+2.0*x*z*SQUAREROOT10+2.0*z*SQUAREROOT10*y;
    gphi[n+13] = -2.0*x*y*SQUAREROOT10+x*x*SQUAREROOT10;
    gphi[n+14] = 6.0*y*y*SQUAREROOT10-6.0*y*SQUAREROOT10+6.0*x*y*SQUAREROOT10+6.0*z*SQUAREROOT10*y+SQUAREROOT10-2.0*x*SQUAREROOT10-2.0*z*SQUAREROOT10+x*x*SQUAREROOT10+2.0*x*z*SQUAREROOT10+z*z*SQUAREROOT10;
    gphi[n+15] = 3.0*x*x*SQUAREROOT10-2.0*x*SQUAREROOT10+2.0*x*y*SQUAREROOT10+2.0*x*z*SQUAREROOT10;
    gphi[n+16] = 6.0*x*z;
    gphi[n+17] = -12.0*y*z+6.0*z-6.0*x*z-6.0*z*z;
    gphi[n+18] = -6.0*x*z;
    gphi[n+19] = -12.0*x*y+6.0*x-6.0*x*x-6.0*x*z;
    gphi[nn+10] = -2.0*z*SQUAREROOT10*y+y*y*SQUAREROOT10;
    gphi[nn+11] = -2.0*x*z*SQUAREROOT10+x*x*SQUAREROOT10;
    gphi[nn+12] = 6.0*z*z*SQUAREROOT10-6.0*z*SQUAREROOT10+6.0*x*z*SQUAREROOT10+6.0*z*SQUAREROOT10*y+SQUAREROOT10-2.0*x*SQUAREROOT10-2.0*y*SQUAREROOT10+x*x*SQUAREROOT10+2.0*x*y*SQUAREROOT10+y*y*SQUAREROOT10;
    gphi[nn+13] = 0.0;
    gphi[nn+14] = 3.0*y*y*SQUAREROOT10-2.0*y*SQUAREROOT10+2.0*x*y*SQUAREROOT10+2.0*z*SQUAREROOT10*y;
    gphi[nn+15] = 3.0*x*x*SQUAREROOT10-2.0*x*SQUAREROOT10+2.0*x*y*SQUAREROOT10+2.0*x*z*SQUAREROOT10;
    gphi[nn+16] = 6.0*x*y;
    gphi[nn+17] = -12.0*y*z+6.0*y-6.0*x*y-6.0*y*y;
    gphi[nn+18] = -12.0*x*z+6.0*x-6.0*x*x-6.0*x*y;
    gphi[nn+19] = -6.0*x*y;
  }
#endif
#if GEOM_USE_P4 && GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1  
  if (porder >= 4){
    gphi[20] = 0.0;
    gphi[21] = -5.0*0.25*z*z*z*SQUAREROOT14+5.0*z*z*SQUAREROOT14*x-15.0*0.25*z*SQUAREROOT14*x*x+z*SQUAREROOT14*0.25;
    gphi[22] = 10.0*z*z*z*SQUAREROOT14-12.5*z*z*SQUAREROOT14+12.5*z*z*SQUAREROOT14*x+12.5*z*z*SQUAREROOT14*y+3.5*z*SQUAREROOT14-7.5*x*z*SQUAREROOT14-7.5*z*SQUAREROOT14*y+15.0*0.25*z*SQUAREROOT14*x*x+7.5*z*SQUAREROOT14*x*y+15.0*0.25*z*SQUAREROOT14*y*y;
    gphi[23] = -5.0*0.25*y*y*y*SQUAREROOT14+5.0*y*y*SQUAREROOT14*x-15.0*0.25*y*SQUAREROOT14*x*x+y*SQUAREROOT14*0.25;
    gphi[24] = 10.0*y*y*y*SQUAREROOT14-12.5*y*y*SQUAREROOT14+12.5*y*y*SQUAREROOT14*x+12.5*z*SQUAREROOT14*y*y+3.5*y*SQUAREROOT14-7.5*x*y*SQUAREROOT14-7.5*z*SQUAREROOT14*y+15.0*0.25*y*SQUAREROOT14*x*x+7.5*z*SQUAREROOT14*x*y+15.0*0.25*z*z*SQUAREROOT14*y;
    gphi[25] = 30.0*y*SQUAREROOT14*x*x-7.5*z*SQUAREROOT14*y-25.0*x*y*SQUAREROOT14+15.0*0.25*z*SQUAREROOT14*y*y+12.5*y*y*SQUAREROOT14*x-25.0*x*z*SQUAREROOT14+12.5*z*z*SQUAREROOT14*x+30.0*z*SQUAREROOT14*x*x+12.0*x*SQUAREROOT14+3.5*y*SQUAREROOT14-15.0*0.25*y*y*SQUAREROOT14+5.0*0.25*y*y*y*SQUAREROOT14-30.0*x*x*SQUAREROOT14+20.0*x*x*x*SQUAREROOT14-15.0*0.25*z*z*SQUAREROOT14+3.5*z*SQUAREROOT14+5.0*0.25*z*z*z*SQUAREROOT14+25.0*z*SQUAREROOT14*x*y-SQUAREROOT14+15.0*0.25*z*z*SQUAREROOT14*y;
    gphi[26] = 4.0*x*y*z*SQUAREROOT15-2.0*y*z*z*SQUAREROOT15;
    gphi[27] = -4.0*y*z*SQUAREROOT15+4.0*x*y*z*SQUAREROOT15+4.0*y*y*z*SQUAREROOT15+6.0*y*z*z*SQUAREROOT15;
    gphi[28] = -8.0*x*z*SQUAREROOT15+6.0*x*x*z*SQUAREROOT15+8.0*x*y*z*SQUAREROOT15+12.0*x*z*z*SQUAREROOT15+2.0*z*SQUAREROOT15-4.0*y*z*SQUAREROOT15-6.0*z*z*SQUAREROOT15+2.0*y*y*z*SQUAREROOT15+6.0*y*z*z*SQUAREROOT15+4.0*z*z*z*SQUAREROOT15;
    gphi[29] = -8.0*x*y*SQUAREROOT15+6.0*x*x*y*SQUAREROOT15+12.0*x*y*y*SQUAREROOT15+8.0*x*y*z*SQUAREROOT15+2.0*y*SQUAREROOT15-6.0*y*y*SQUAREROOT15-4.0*y*z*SQUAREROOT15+4.0*y*y*y*SQUAREROOT15+6.0*y*y*z*SQUAREROOT15+2.0*y*z*z*SQUAREROOT15;
    gphi[30] = 2.0*y*y*z*SQUAREROOT15-4.0*x*y*z*SQUAREROOT15;
    gphi[31] = -6.0*y*y*z*SQUAREROOT15+4.0*y*z*SQUAREROOT15-4.0*x*y*z*SQUAREROOT15-4.0*y*z*z*SQUAREROOT15;
    gphi[32] = -12.0*x*x*z*SQUAREROOT15+12.0*x*z*SQUAREROOT15-12.0*x*y*z*SQUAREROOT15-12.0*x*z*z*SQUAREROOT15-2.0*z*SQUAREROOT15+4.0*y*z*SQUAREROOT15+4.0*z*z*SQUAREROOT15-2.0*y*y*z*SQUAREROOT15-4.0*y*z*z*SQUAREROOT15-2.0*z*z*z*SQUAREROOT15;
    gphi[33] = -12.0*x*x*y*SQUAREROOT15+12.0*x*y*SQUAREROOT15-12.0*x*y*y*SQUAREROOT15-12.0*x*y*z*SQUAREROOT15-2.0*y*SQUAREROOT15+4.0*y*y*SQUAREROOT15+4.0*y*z*SQUAREROOT15-2.0*y*y*y*SQUAREROOT15-4.0*y*y*z*SQUAREROOT15-2.0*y*z*z*SQUAREROOT15;
    gphi[34] = 12.0*x*y*SQUAREROOT6*z-6.0*y*z*SQUAREROOT6+6.0*y*y*SQUAREROOT6*z+6.0*y*SQUAREROOT6*z*z;
    gphi[n+20] = -5.0*0.25*z*z*z*SQUAREROOT14+5.0*z*z*SQUAREROOT14*y-15.0*0.25*z*SQUAREROOT14*y*y+z*SQUAREROOT14*0.25;
    gphi[n+21] = 0.0;
    gphi[n+22] = 10.0*z*z*z*SQUAREROOT14-12.5*z*z*SQUAREROOT14+12.5*z*z*SQUAREROOT14*x+12.5*z*z*SQUAREROOT14*y+3.5*z*SQUAREROOT14-7.5*x*z*SQUAREROOT14-7.5*z*SQUAREROOT14*y+15.0*0.25*z*SQUAREROOT14*x*x+7.5*z*SQUAREROOT14*x*y+15.0*0.25*z*SQUAREROOT14*y*y;
    gphi[n+23] = -15.0*0.25*y*y*SQUAREROOT14*x+5.0*y*SQUAREROOT14*x*x-5.0*0.25*x*x*x*SQUAREROOT14+x*SQUAREROOT14*0.25;
    gphi[n+24] = 30.0*y*y*SQUAREROOT14*x+12.5*y*SQUAREROOT14*x*x+15.0*0.25*z*SQUAREROOT14*x*x+15.0*0.25*z*z*SQUAREROOT14*x-7.5*x*z*SQUAREROOT14-25.0*z*SQUAREROOT14*y-25.0*x*y*SQUAREROOT14+12.5*z*z*SQUAREROOT14*y+30.0*z*SQUAREROOT14*y*y+12.0*y*SQUAREROOT14+5.0*0.25*x*x*x*SQUAREROOT14-SQUAREROOT14-30.0*y*y*SQUAREROOT14+20.0*y*y*y*SQUAREROOT14-15.0*0.25*x*x*SQUAREROOT14-15.0*0.25*z*z*SQUAREROOT14+3.5*z*SQUAREROOT14+5.0*0.25*z*z*z*SQUAREROOT14+25.0*z*SQUAREROOT14*x*y+3.5*x*SQUAREROOT14;
    gphi[n+25] = 10.0*x*x*x*SQUAREROOT14-12.5*x*x*SQUAREROOT14+12.5*y*SQUAREROOT14*x*x+12.5*z*SQUAREROOT14*x*x+3.5*x*SQUAREROOT14-7.5*x*y*SQUAREROOT14-7.5*x*z*SQUAREROOT14+15.0*0.25*y*y*SQUAREROOT14*x+7.5*z*SQUAREROOT14*x*y+15.0*0.25*z*z*SQUAREROOT14*x;
    gphi[n+26] = 2.0*x*z*SQUAREROOT15*(x-z);
    gphi[n+27] = -8.0*y*z*SQUAREROOT15+8.0*x*y*z*SQUAREROOT15+6.0*y*y*z*SQUAREROOT15+12.0*y*z*z*SQUAREROOT15+2.0*z*SQUAREROOT15-4.0*x*z*SQUAREROOT15-6.0*z*z*SQUAREROOT15+2.0*x*x*z*SQUAREROOT15+6.0*x*z*z*SQUAREROOT15+4.0*z*z*z*SQUAREROOT15;
    gphi[n+28] = -4.0*x*z*SQUAREROOT15+4.0*x*x*z*SQUAREROOT15+4.0*x*y*z*SQUAREROOT15+6.0*x*z*z*SQUAREROOT15;
    gphi[n+29] = -12.0*x*y*SQUAREROOT15+12.0*x*x*y*SQUAREROOT15+12.0*x*y*y*SQUAREROOT15+12.0*x*y*z*SQUAREROOT15+2.0*x*SQUAREROOT15-4.0*x*x*SQUAREROOT15-4.0*x*z*SQUAREROOT15+2.0*x*x*x*SQUAREROOT15+4.0*x*x*z*SQUAREROOT15+2.0*x*z*z*SQUAREROOT15;
    gphi[n+30] = 4.0*x*y*z*SQUAREROOT15-2.0*x*x*z*SQUAREROOT15;
    gphi[n+31] = -12.0*y*y*z*SQUAREROOT15+12.0*y*z*SQUAREROOT15-12.0*x*y*z*SQUAREROOT15-12.0*y*z*z*SQUAREROOT15-2.0*z*SQUAREROOT15+4.0*x*z*SQUAREROOT15+4.0*z*z*SQUAREROOT15-2.0*x*x*z*SQUAREROOT15-4.0*x*z*z*SQUAREROOT15-2.0*z*z*z*SQUAREROOT15;
    gphi[n+32] = -6.0*x*x*z*SQUAREROOT15+4.0*x*z*SQUAREROOT15-4.0*x*y*z*SQUAREROOT15-4.0*x*z*z*SQUAREROOT15;
    gphi[n+33] = -12.0*x*x*y*SQUAREROOT15+8.0*x*y*SQUAREROOT15-6.0*x*y*y*SQUAREROOT15-8.0*x*y*z*SQUAREROOT15+6.0*x*x*SQUAREROOT15-2.0*x*SQUAREROOT15+4.0*x*z*SQUAREROOT15-4.0*x*x*x*SQUAREROOT15-6.0*x*x*z*SQUAREROOT15-2.0*x*z*z*SQUAREROOT15;
    gphi[n+34] = 12.0*x*y*SQUAREROOT6*z-6.0*x*SQUAREROOT6*z+6.0*x*x*SQUAREROOT6*z+6.0*x*SQUAREROOT6*z*z;
    gphi[nn+20] = -15.0*0.25*z*z*SQUAREROOT14*y+5.0*z*SQUAREROOT14*y*y-5.0*0.25*y*y*y*SQUAREROOT14+y*SQUAREROOT14*0.25;
    gphi[nn+21] = -5.0*0.25*x*x*x*SQUAREROOT14+5.0*z*SQUAREROOT14*x*x-15.0*0.25*z*z*SQUAREROOT14*x+x*SQUAREROOT14*0.25;
    gphi[nn+22] = 3.5*x*SQUAREROOT14+5.0*0.25*y*y*y*SQUAREROOT14+5.0*0.25*x*x*x*SQUAREROOT14+12.5*z*SQUAREROOT14*x*x+30.0*z*z*SQUAREROOT14*x+25.0*z*SQUAREROOT14*x*y-25.0*z*SQUAREROOT14*y+12.0*z*SQUAREROOT14-25.0*x*z*SQUAREROOT14+3.5*y*SQUAREROOT14+30.0*z*z*SQUAREROOT14*y+12.5*z*SQUAREROOT14*y*y-SQUAREROOT14-7.5*x*y*SQUAREROOT14+15.0*0.25*y*y*SQUAREROOT14*x+15.0*0.25*y*SQUAREROOT14*x*x+20.0*z*z*z*SQUAREROOT14-30.0*z*z*SQUAREROOT14-15.0*0.25*x*x*SQUAREROOT14-15.0*0.25*y*y*SQUAREROOT14;
    gphi[nn+23] = 0.0;
    gphi[nn+24] = 10.0*y*y*y*SQUAREROOT14-12.5*y*y*SQUAREROOT14+12.5*y*y*SQUAREROOT14*x+12.5*z*SQUAREROOT14*y*y+3.5*y*SQUAREROOT14-7.5*x*y*SQUAREROOT14-7.5*z*SQUAREROOT14*y+15.0*0.25*y*SQUAREROOT14*x*x+7.5*z*SQUAREROOT14*x*y+15.0*0.25*z*z*SQUAREROOT14*y;
    gphi[nn+25] = 10.0*x*x*x*SQUAREROOT14-12.5*x*x*SQUAREROOT14+12.5*y*SQUAREROOT14*x*x+12.5*z*SQUAREROOT14*x*x+3.5*x*SQUAREROOT14-7.5*x*y*SQUAREROOT14-7.5*x*z*SQUAREROOT14+15.0*0.25*y*y*SQUAREROOT14*x+7.5*z*SQUAREROOT14*x*y+15.0*0.25*z*z*SQUAREROOT14*x;
    gphi[nn+26] = 2.0*x*x*y*SQUAREROOT15-4.0*x*y*z*SQUAREROOT15;
    gphi[nn+27] = -12.0*y*z*SQUAREROOT15+12.0*x*y*z*SQUAREROOT15+12.0*y*y*z*SQUAREROOT15+12.0*y*z*z*SQUAREROOT15+2.0*y*SQUAREROOT15-4.0*x*y*SQUAREROOT15-4.0*y*y*SQUAREROOT15+2.0*x*x*y*SQUAREROOT15+4.0*x*y*y*SQUAREROOT15+2.0*y*y*y*SQUAREROOT15;
    gphi[nn+28] = -12.0*x*z*SQUAREROOT15+12.0*x*x*z*SQUAREROOT15+12.0*x*y*z*SQUAREROOT15+12.0*x*z*z*SQUAREROOT15+2.0*x*SQUAREROOT15-4.0*x*x*SQUAREROOT15-4.0*x*y*SQUAREROOT15+2.0*x*x*x*SQUAREROOT15+4.0*x*x*y*SQUAREROOT15+2.0*x*y*y*SQUAREROOT15;
    gphi[nn+29] = -4.0*x*y*SQUAREROOT15+4.0*x*x*y*SQUAREROOT15+6.0*x*y*y*SQUAREROOT15+4.0*x*y*z*SQUAREROOT15;
    gphi[nn+30] = -2.0*x*y*SQUAREROOT5*(-y+x)*SQUAREROOT3;
    gphi[nn+31] = -12.0*y*y*z*SQUAREROOT15+8.0*y*z*SQUAREROOT15-8.0*x*y*z*SQUAREROOT15-6.0*y*z*z*SQUAREROOT15+6.0*y*y*SQUAREROOT15-2.0*y*SQUAREROOT15+4.0*x*y*SQUAREROOT15-6.0*x*y*y*SQUAREROOT15-2.0*x*x*y*SQUAREROOT15-4.0*y*y*y*SQUAREROOT15;
    gphi[nn+32] = -12.0*x*x*z*SQUAREROOT15+8.0*x*z*SQUAREROOT15-8.0*x*y*z*SQUAREROOT15-6.0*x*z*z*SQUAREROOT15+6.0*x*x*SQUAREROOT15-2.0*x*SQUAREROOT15+4.0*x*y*SQUAREROOT15-4.0*x*x*x*SQUAREROOT15-6.0*x*x*y*SQUAREROOT15-2.0*x*y*y*SQUAREROOT15;
    gphi[nn+33] = -6.0*x*x*y*SQUAREROOT15+4.0*x*y*SQUAREROOT15-4.0*x*y*y*SQUAREROOT15-4.0*x*y*z*SQUAREROOT15;
    gphi[nn+34] = 12.0*x*y*SQUAREROOT6*z-6.0*x*y*SQUAREROOT6+6.0*x*x*y*SQUAREROOT6+6.0*x*y*y*SQUAREROOT6;
  }
#endif
#if GEOM_USE_P5 && GEOM_USE_P4 && GEOM_USE_P3 && GEOM_USE_P2 && GEOM_USE_P1  
  if (porder >= 5){
    gphi[35] = 0.0;
    gphi[36] = -21.0*0.25*z*z*z*z*SQUAREROOT2+31.5*z*z*z*SQUAREROOT2*x-189.0*0.25*z*z*SQUAREROOT2*x*x+21.0*z*SQUAREROOT2*x*x*x+9.0*0.25*z*z*SQUAREROOT2-4.5*z*SQUAREROOT2*x;
    gphi[37] = 105.0*z*z*z*z*SQUAREROOT2+103.5*z*z*SQUAREROOT2-16.5*z*SQUAREROOT2+21.0*z*SQUAREROOT2*y*y*y-63.0*z*SQUAREROOT2*x*x-220.5*z*z*SQUAREROOT2*y-220.5*z*z*SQUAREROOT2*x+441.0*0.25*z*z*SQUAREROOT2*y*y+189.0*z*z*z*SQUAREROOT2*y-63.0*z*SQUAREROOT2*y*y+58.5*z*SQUAREROOT2*y-189.0*z*z*z*SQUAREROOT2-126.0*z*SQUAREROOT2*x*y+220.5*z*z*SQUAREROOT2*x*y+63.0*z*SQUAREROOT2*x*y*y+63.0*z*SQUAREROOT2*x*x*y+189.0*z*z*z*SQUAREROOT2*x+441.0*0.25*z*z*SQUAREROOT2*x*x+21.0*z*SQUAREROOT2*x*x*x+58.5*z*SQUAREROOT2*x;
    gphi[38] = -21.0*0.25*y*y*y*y*SQUAREROOT2+31.5*y*y*y*SQUAREROOT2*x-189.0*0.25*y*y*SQUAREROOT2*x*x+21.0*y*SQUAREROOT2*x*x*x+9.0*0.25*y*y*SQUAREROOT2-4.5*y*SQUAREROOT2*x;
    gphi[39] = -63.0*y*SQUAREROOT2*x*x-16.5*y*SQUAREROOT2-220.5*y*y*SQUAREROOT2*x+103.5*y*y*SQUAREROOT2+441.0*0.25*z*z*SQUAREROOT2*y*y-63.0*z*z*SQUAREROOT2*y+58.5*z*SQUAREROOT2*y-220.5*z*SQUAREROOT2*y*y+21.0*z*z*z*SQUAREROOT2*y+189.0*z*SQUAREROOT2*y*y*y-189.0*y*y*y*SQUAREROOT2+220.5*z*SQUAREROOT2*x*y*y-126.0*z*SQUAREROOT2*x*y+63.0*z*z*SQUAREROOT2*x*y+105.0*y*y*y*y*SQUAREROOT2+21.0*y*SQUAREROOT2*x*x*x+441.0*0.25*y*y*SQUAREROOT2*x*x+189.0*y*y*y*SQUAREROOT2*x+58.5*y*SQUAREROOT2*x+63.0*z*SQUAREROOT2*x*x*y;
    gphi[40] = 207.0*y*SQUAREROOT2*x+420.0*y*SQUAREROOT2*x*x*x+283.5*y*y*SQUAREROOT2*x*x+73.5*y*y*y*SQUAREROOT2*x+117.0*0.25*y*y*SQUAREROOT2-567.0*y*SQUAREROOT2*x*x-220.5*y*y*SQUAREROOT2*x+3.0*SQUAREROOT2+73.5*z*z*z*SQUAREROOT2*x-16.5*y*SQUAREROOT2-21.0*y*y*y*SQUAREROOT2+21.0*0.25*y*y*y*y*SQUAREROOT2+21.0*0.25*z*z*z*z*SQUAREROOT2+117.0*0.25*z*z*SQUAREROOT2-16.5*z*SQUAREROOT2-21.0*z*z*z*SQUAREROOT2-567.0*z*SQUAREROOT2*x*x-63.0*z*z*SQUAREROOT2*y-220.5*z*z*SQUAREROOT2*x+31.5*z*z*SQUAREROOT2*y*y+21.0*z*z*z*SQUAREROOT2*y-63.0*z*SQUAREROOT2*y*y+58.5*z*SQUAREROOT2*y+21.0*z*SQUAREROOT2*y*y*y-441.0*z*SQUAREROOT2*x*y+220.5*z*z*SQUAREROOT2*x*y+220.5*z*SQUAREROOT2*x*y*y+567.0*z*SQUAREROOT2*x*x*y+270.0*x*x*SQUAREROOT2-420.0*x*x*x*SQUAREROOT2+210.0*x*x*x*x*SQUAREROOT2-60.0*x*SQUAREROOT2+283.5*z*z*SQUAREROOT2*x*x+420.0*z*SQUAREROOT2*x*x*x+207.0*z*SQUAREROOT2*x;
    gphi[41] = 2.5*y*z*z*z*SQUAREROOT21-10.0*y*z*z*SQUAREROOT21*x+7.5*y*z*SQUAREROOT21*x*x-y*z*SQUAREROOT21*0.5;
    gphi[42] = -20.0*y*z*z*z*SQUAREROOT21+25.0*y*z*z*SQUAREROOT21-25.0*y*z*z*SQUAREROOT21*x-25.0*y*y*z*z*SQUAREROOT21-7.0*y*z*SQUAREROOT21+15.0*x*y*z*SQUAREROOT21+15.0*y*y*z*SQUAREROOT21-7.5*y*z*SQUAREROOT21*x*x-15.0*y*y*z*SQUAREROOT21*x-7.5*y*y*y*z*SQUAREROOT21;
    gphi[43] = -12.0*z*z*SQUAREROOT21+20.0*z*z*z*SQUAREROOT21-10.0*z*z*z*z*SQUAREROOT21-50.0*y*z*z*SQUAREROOT21*x-22.5*y*z*SQUAREROOT21*x*x+30.0*x*y*z*SQUAREROOT21-20.0*y*z*z*z*SQUAREROOT21-15.0*y*y*z*SQUAREROOT21*x+25.0*y*z*z*SQUAREROOT21-12.5*y*y*z*z*SQUAREROOT21+7.5*y*y*z*SQUAREROOT21-2.5*y*y*y*z*SQUAREROOT21-7.0*y*z*SQUAREROOT21-40.0*x*z*z*z*SQUAREROOT21+50.0*x*z*z*SQUAREROOT21-37.5*x*x*z*z*SQUAREROOT21+22.5*x*x*z*SQUAREROOT21-10.0*x*x*x*z*SQUAREROOT21-14.0*x*z*SQUAREROOT21+2.0*z*SQUAREROOT21;
    gphi[44] = -37.5*x*x*y*y*SQUAREROOT21-12.0*y*y*SQUAREROOT21-10.0*y*y*y*y*SQUAREROOT21+20.0*y*y*y*SQUAREROOT21-14.0*x*y*SQUAREROOT21+22.5*x*x*y*SQUAREROOT21-10.0*x*x*x*y*SQUAREROOT21+2.0*y*SQUAREROOT21-15.0*y*z*z*SQUAREROOT21*x-22.5*y*z*SQUAREROOT21*x*x+30.0*x*y*z*SQUAREROOT21-2.5*y*z*z*z*SQUAREROOT21-50.0*y*y*z*SQUAREROOT21*x+7.5*y*z*z*SQUAREROOT21-12.5*y*y*z*z*SQUAREROOT21+25.0*y*y*z*SQUAREROOT21-20.0*y*y*y*z*SQUAREROOT21-7.0*y*z*SQUAREROOT21-40.0*x*y*y*y*SQUAREROOT21+50.0*x*y*y*SQUAREROOT21;
    gphi[45] = 20.0*x*z*y*y-10.0*z*z*y*y-30.0*x*x*z*y+20.0*x*z*z*y;
    gphi[46] = 80.0*x*z*z*y+50.0*z*y*y*y+110.0*z*z*y*y+80.0*x*z*y*y+30.0*x*x*z*y+50.0*z*z*z*y+30.0*y*z-60.0*x*y*z-80.0*y*z*z-80.0*y*y*z;
    gphi[47] = -10.0*z+220.0*x*z*z*y+10.0*z*y*y*y+40.0*z*z*y*y+80.0*x*z*y*y+150.0*x*x*z*y+50.0*z*z*z*y+40.0*z*z+80.0*x*x*x*z+30.0*y*z+80.0*x*z+210.0*x*x*z*z-160.0*x*y*z+140.0*x*z*z*z-220.0*z*z*x-150.0*z*x*x+20.0*z*z*z*z-50.0*z*z*z-80.0*y*z*z-30.0*y*y*z;
    gphi[48] = -10.0*y+20.0*y*y*y*y+80.0*y*x*x*x+80.0*x*z*z*y+210.0*y*y*x*x+140.0*y*y*y*x+50.0*z*y*y*y+40.0*z*z*y*y+220.0*x*z*y*y+150.0*x*x*z*y+10.0*z*z*z*y-50.0*y*y*y+30.0*y*z+80.0*x*y-160.0*x*y*z-220.0*x*y*y+40.0*y*y-150.0*x*x*y-30.0*y*z*z-80.0*y*y*z;
    gphi[49] = 2.5*y*y*y*z*SQUAREROOT21-10.0*y*y*z*SQUAREROOT21*x+7.5*y*z*SQUAREROOT21*x*x-y*z*SQUAREROOT21*0.5;
    gphi[50] = -20.0*y*y*y*z*SQUAREROOT21+25.0*y*y*z*SQUAREROOT21-25.0*y*y*z*SQUAREROOT21*x-25.0*y*y*z*z*SQUAREROOT21-7.0*y*z*SQUAREROOT21+15.0*x*y*z*SQUAREROOT21+15.0*y*z*z*SQUAREROOT21-7.5*y*z*SQUAREROOT21*x*x-15.0*y*z*z*SQUAREROOT21*x-7.5*y*z*z*z*SQUAREROOT21;
    gphi[51] = 50.0*x*y*z*SQUAREROOT21-2.5*z*z*z*z*SQUAREROOT21+7.5*z*z*z*SQUAREROOT21-7.0*z*z*SQUAREROOT21-24.0*x*z*SQUAREROOT21-50.0*y*z*z*SQUAREROOT21*x-25.0*y*y*z*SQUAREROOT21*x+15.0*y*z*z*SQUAREROOT21-2.5*y*y*y*z*SQUAREROOT21-7.5*y*z*z*z*SQUAREROOT21-7.5*y*y*z*z*SQUAREROOT21+7.5*y*y*z*SQUAREROOT21+2.0*z*SQUAREROOT21-7.0*y*z*SQUAREROOT21-40.0*x*x*x*z*SQUAREROOT21+60.0*x*x*z*SQUAREROOT21-60.0*x*x*z*z*SQUAREROOT21+50.0*x*z*z*SQUAREROOT21-25.0*x*z*z*z*SQUAREROOT21-60.0*y*z*SQUAREROOT21*x*x;
    gphi[52] = 50.0*x*y*z*SQUAREROOT21+50.0*x*y*y*SQUAREROOT21-60.0*x*x*y*y*SQUAREROOT21-7.0*y*y*SQUAREROOT21-2.5*y*y*y*y*SQUAREROOT21+7.5*y*y*y*SQUAREROOT21-24.0*x*y*SQUAREROOT21+60.0*x*x*y*SQUAREROOT21-40.0*x*x*x*y*SQUAREROOT21+2.0*y*SQUAREROOT21-25.0*y*z*z*SQUAREROOT21*x-50.0*y*y*z*SQUAREROOT21*x+7.5*y*z*z*SQUAREROOT21-7.5*y*y*y*z*SQUAREROOT21-2.5*y*z*z*z*SQUAREROOT21-25.0*x*y*y*y*SQUAREROOT21-7.5*y*y*z*z*SQUAREROOT21+15.0*y*y*z*SQUAREROOT21-7.0*y*z*SQUAREROOT21-60.0*y*z*SQUAREROOT21*x*x;
    gphi[53] = 12.0*SQUAREROOT10*x*y*z*z-12.0*SQUAREROOT10*x*y*y*z-6.0*SQUAREROOT10*y*z*z+6.0*SQUAREROOT10*y*z*z*z+6.0*SQUAREROOT10*y*y*z-6.0*SQUAREROOT10*y*y*y*z;
    gphi[54] = 12.0*SQUAREROOT10*x*y*z-18.0*SQUAREROOT10*x*x*y*z-12.0*SQUAREROOT10*x*y*z*z-6.0*SQUAREROOT10*y*y*z+6.0*SQUAREROOT10*y*y*y*z+6.0*SQUAREROOT10*y*y*z*z;
    gphi[55] = 36.0*SQUAREROOT10*x*y*z-36.0*SQUAREROOT10*x*x*y*z-36.0*SQUAREROOT10*x*y*y*z-36.0*SQUAREROOT10*x*y*z*z-6.0*z*SQUAREROOT10*y+12.0*SQUAREROOT10*y*y*z+12.0*SQUAREROOT10*y*z*z-6.0*SQUAREROOT10*y*y*y*z-12.0*SQUAREROOT10*y*y*z*z-6.0*SQUAREROOT10*y*z*z*z;
    gphi[n+35] = -21.0*0.25*z*z*z*z*SQUAREROOT2+31.5*z*z*z*SQUAREROOT2*y-189.0*0.25*z*z*SQUAREROOT2*y*y+21.0*z*SQUAREROOT2*y*y*y+9.0*0.25*z*z*SQUAREROOT2-4.5*z*SQUAREROOT2*y;
    gphi[n+36] = 0.0;
    gphi[n+37] = 105.0*z*z*z*z*SQUAREROOT2+103.5*z*z*SQUAREROOT2-16.5*z*SQUAREROOT2+21.0*z*SQUAREROOT2*y*y*y-63.0*z*SQUAREROOT2*x*x-220.5*z*z*SQUAREROOT2*y-220.5*z*z*SQUAREROOT2*x+441.0*0.25*z*z*SQUAREROOT2*y*y+189.0*z*z*z*SQUAREROOT2*y-63.0*z*SQUAREROOT2*y*y+58.5*z*SQUAREROOT2*y-189.0*z*z*z*SQUAREROOT2-126.0*z*SQUAREROOT2*x*y+220.5*z*z*SQUAREROOT2*x*y+63.0*z*SQUAREROOT2*x*y*y+63.0*z*SQUAREROOT2*x*x*y+189.0*z*z*z*SQUAREROOT2*x+441.0*0.25*z*z*SQUAREROOT2*x*x+21.0*z*SQUAREROOT2*x*x*x+58.5*z*SQUAREROOT2*x;
    gphi[n+38] = -21.0*y*y*y*SQUAREROOT2*x+189.0*0.25*y*y*SQUAREROOT2*x*x-31.5*y*SQUAREROOT2*x*x*x+21.0*0.25*x*x*x*x*SQUAREROOT2+4.5*y*SQUAREROOT2*x-9.0*0.25*x*x*SQUAREROOT2;
    gphi[n+39] = 207.0*y*SQUAREROOT2*x+73.5*y*SQUAREROOT2*x*x*x+283.5*y*y*SQUAREROOT2*x*x+420.0*y*y*y*SQUAREROOT2*x+270.0*y*y*SQUAREROOT2-220.5*y*SQUAREROOT2*x*x-567.0*y*y*SQUAREROOT2*x+3.0*SQUAREROOT2+21.0*z*z*z*SQUAREROOT2*x-60.0*y*SQUAREROOT2-420.0*y*y*y*SQUAREROOT2+210.0*y*y*y*y*SQUAREROOT2+21.0*0.25*z*z*z*z*SQUAREROOT2+117.0*0.25*z*z*SQUAREROOT2-16.5*z*SQUAREROOT2-21.0*z*z*z*SQUAREROOT2-63.0*z*SQUAREROOT2*x*x-220.5*z*z*SQUAREROOT2*y-63.0*z*z*SQUAREROOT2*x+283.5*z*z*SQUAREROOT2*y*y+73.5*z*z*z*SQUAREROOT2*y-567.0*z*SQUAREROOT2*y*y+207.0*z*SQUAREROOT2*y+420.0*z*SQUAREROOT2*y*y*y-441.0*z*SQUAREROOT2*x*y+220.5*z*z*SQUAREROOT2*x*y+567.0*z*SQUAREROOT2*x*y*y+220.5*z*SQUAREROOT2*x*x*y+117.0*0.25*x*x*SQUAREROOT2-21.0*x*x*x*SQUAREROOT2+21.0*0.25*x*x*x*x*SQUAREROOT2-16.5*x*SQUAREROOT2+31.5*z*z*SQUAREROOT2*x*x+21.0*z*SQUAREROOT2*x*x*x+58.5*z*SQUAREROOT2*x;
    gphi[n+40] = -220.5*z*SQUAREROOT2*x*x-63.0*z*z*SQUAREROOT2*x+21.0*z*z*z*SQUAREROOT2*x+441.0*0.25*z*z*SQUAREROOT2*x*x+58.5*z*SQUAREROOT2*x+189.0*z*SQUAREROOT2*x*x*x+58.5*y*SQUAREROOT2*x+441.0*0.25*y*y*SQUAREROOT2*x*x+189.0*y*SQUAREROOT2*x*x*x+21.0*y*y*y*SQUAREROOT2*x-16.5*x*SQUAREROOT2-63.0*y*y*SQUAREROOT2*x+103.5*x*x*SQUAREROOT2-220.5*y*SQUAREROOT2*x*x-189.0*x*x*x*SQUAREROOT2+105.0*x*x*x*x*SQUAREROOT2-126.0*z*SQUAREROOT2*x*y+63.0*z*SQUAREROOT2*x*y*y+63.0*z*z*SQUAREROOT2*x*y+220.5*z*SQUAREROOT2*x*x*y;
    gphi[n+41] = x*z*SQUAREROOT21*(5.0*x*x-10.0*x*z+5.0*z*z-1.0)*0.5;
    gphi[n+42] = -14.0*y*z*SQUAREROOT21-15.0*y*z*SQUAREROOT21*x*x-22.5*y*y*z*SQUAREROOT21*x-50.0*y*z*z*SQUAREROOT21*x+30.0*x*y*z*SQUAREROOT21-12.5*x*x*z*z*SQUAREROOT21-2.5*x*x*x*z*SQUAREROOT21-7.0*x*z*SQUAREROOT21+22.5*y*y*z*SQUAREROOT21+50.0*y*z*z*SQUAREROOT21-10.0*y*y*y*z*SQUAREROOT21-37.5*y*y*z*z*SQUAREROOT21-40.0*y*z*z*z*SQUAREROOT21+7.5*x*x*z*SQUAREROOT21+25.0*x*z*z*SQUAREROOT21-20.0*x*z*z*z*SQUAREROOT21+2.0*z*SQUAREROOT21-12.0*z*z*SQUAREROOT21+20.0*z*z*z*SQUAREROOT21-10.0*z*z*z*z*SQUAREROOT21;
    gphi[n+43] = -7.0*x*z*SQUAREROOT21+15.0*x*x*z*SQUAREROOT21+15.0*x*y*z*SQUAREROOT21+25.0*x*z*z*SQUAREROOT21-7.5*x*x*x*z*SQUAREROOT21-15.0*y*z*SQUAREROOT21*x*x-25.0*x*x*z*z*SQUAREROOT21-7.5*y*y*z*SQUAREROOT21*x-25.0*y*z*z*SQUAREROOT21*x-20.0*x*z*z*z*SQUAREROOT21;
    gphi[n+44] = -24.0*x*y*SQUAREROOT21-50.0*y*z*SQUAREROOT21*x*x-60.0*y*y*z*SQUAREROOT21*x-25.0*y*z*z*SQUAREROOT21*x+50.0*x*y*z*SQUAREROOT21-7.5*x*x*z*z*SQUAREROOT21-7.5*x*x*x*z*SQUAREROOT21-7.0*x*z*SQUAREROOT21-40.0*x*y*y*y*SQUAREROOT21+60.0*x*y*y*SQUAREROOT21-60.0*x*x*y*y*SQUAREROOT21+50.0*x*x*y*SQUAREROOT21-25.0*x*x*x*y*SQUAREROOT21+15.0*x*x*z*SQUAREROOT21+7.5*x*z*z*SQUAREROOT21-2.5*x*z*z*z*SQUAREROOT21+2.0*x*SQUAREROOT21-7.0*x*x*SQUAREROOT21+7.5*x*x*x*SQUAREROOT21-2.5*x*x*x*x*SQUAREROOT21;
    gphi[n+45] = 20.0*x*x*z*y-20.0*x*z*z*y-10.0*x*x*x*z+10.0*x*x*z*z;
    gphi[n+46] = -10.0*z+220.0*x*z*z*y+80.0*z*y*y*y+210.0*z*z*y*y+150.0*x*z*y*y+80.0*x*x*z*y+140.0*z*z*z*y+40.0*z*z+10.0*x*x*x*z+80.0*y*z+30.0*x*z+40.0*x*x*z*z-160.0*x*y*z+50.0*x*z*z*z-80.0*z*z*x-30.0*z*x*x+20.0*z*z*z*z-50.0*z*z*z-220.0*y*z*z-150.0*y*y*z;
    gphi[n+47] = 80.0*x*z*z*y+30.0*x*z*y*y+80.0*x*x*z*y+50.0*x*x*x*z+30.0*x*z+110.0*x*x*z*z-60.0*x*y*z+50.0*x*z*z*z-80.0*z*z*x-80.0*z*x*x;
    gphi[n+48] = -10.0*x+140.0*y*x*x*x+80.0*x*z*z*y+210.0*y*y*x*x+80.0*y*y*y*x+150.0*x*z*y*y+220.0*x*x*z*y+40.0*x*x+50.0*x*x*x*z+30.0*x*z+80.0*x*y+40.0*x*x*z*z-160.0*x*y*z+10.0*x*z*z*z-30.0*z*z*x-80.0*z*x*x-150.0*x*y*y-50.0*x*x*x+20.0*x*x*x*x-220.0*x*x*y;
    gphi[n+49] = 7.5*y*y*z*SQUAREROOT21*x-10.0*y*z*SQUAREROOT21*x*x+2.5*x*x*x*z*SQUAREROOT21-x*z*SQUAREROOT21*0.5;
    gphi[n+50] = -7.0*x*z*SQUAREROOT21+50.0*y*z*z*SQUAREROOT21-25.0*y*z*z*z*SQUAREROOT21+7.5*x*x*z*SQUAREROOT21-2.5*z*z*z*z*SQUAREROOT21+7.5*z*z*z*SQUAREROOT21-7.0*z*z*SQUAREROOT21-24.0*y*z*SQUAREROOT21+2.0*z*SQUAREROOT21-50.0*y*z*z*SQUAREROOT21*x-25.0*y*z*SQUAREROOT21*x*x-60.0*y*y*z*SQUAREROOT21*x-2.5*x*x*x*z*SQUAREROOT21+50.0*x*y*z*SQUAREROOT21+60.0*y*y*z*SQUAREROOT21-60.0*y*y*z*z*SQUAREROOT21-40.0*y*y*y*z*SQUAREROOT21-7.5*x*z*z*z*SQUAREROOT21-7.5*x*x*z*z*SQUAREROOT21+15.0*x*z*z*SQUAREROOT21;
    gphi[n+51] = -20.0*x*x*x*z*SQUAREROOT21+25.0*x*x*z*SQUAREROOT21-25.0*y*z*SQUAREROOT21*x*x-25.0*x*x*z*z*SQUAREROOT21-7.0*x*z*SQUAREROOT21+15.0*x*y*z*SQUAREROOT21+15.0*x*z*z*SQUAREROOT21-7.5*y*y*z*SQUAREROOT21*x-15.0*y*z*z*SQUAREROOT21*x-7.5*x*z*z*z*SQUAREROOT21;
    gphi[n+52] = -7.0*x*z*SQUAREROOT21+25.0*x*x*z*SQUAREROOT21-15.0*y*z*z*SQUAREROOT21*x-50.0*y*z*SQUAREROOT21*x*x-22.5*y*y*z*SQUAREROOT21*x-10.0*x*x*x*x*SQUAREROOT21+20.0*x*x*x*SQUAREROOT21-12.0*x*x*SQUAREROOT21-20.0*x*x*x*z*SQUAREROOT21+30.0*x*y*z*SQUAREROOT21+2.0*x*SQUAREROOT21-2.5*x*z*z*z*SQUAREROOT21-12.5*x*x*z*z*SQUAREROOT21+7.5*x*z*z*SQUAREROOT21-40.0*x*x*x*y*SQUAREROOT21+50.0*x*x*y*SQUAREROOT21-37.5*x*x*y*y*SQUAREROOT21+22.5*x*y*y*SQUAREROOT21-14.0*x*y*SQUAREROOT21-10.0*x*y*y*y*SQUAREROOT21;
    gphi[n+53] = 12.0*SQUAREROOT10*x*y*z-12.0*SQUAREROOT10*x*x*y*z-18.0*SQUAREROOT10*x*y*y*z-6.0*SQUAREROOT10*x*z*z+6.0*SQUAREROOT10*x*x*z*z+6.0*SQUAREROOT10*x*z*z*z;
    gphi[n+54] = -12.0*SQUAREROOT10*x*y*z+18.0*SQUAREROOT10*x*y*y*z+12.0*SQUAREROOT10*x*y*z*z+6.0*SQUAREROOT10*x*x*z-6.0*SQUAREROOT10*x*x*x*z-6.0*SQUAREROOT10*x*x*z*z;
    gphi[n+55] = 24.0*SQUAREROOT10*x*y*z-36.0*SQUAREROOT10*x*x*y*z-18.0*SQUAREROOT10*x*y*y*z-24.0*SQUAREROOT10*x*y*z*z-6.0*x*z*SQUAREROOT10+18.0*SQUAREROOT10*x*x*z+12.0*SQUAREROOT10*x*z*z-12.0*SQUAREROOT10*x*x*x*z-18.0*SQUAREROOT10*x*x*z*z-6.0*SQUAREROOT10*x*z*z*z;
    gphi[nn+35] = -21.0*z*z*z*SQUAREROOT2*y+189.0*0.25*z*z*SQUAREROOT2*y*y-31.5*z*SQUAREROOT2*y*y*y+21.0*0.25*y*y*y*y*SQUAREROOT2+4.5*z*SQUAREROOT2*y-9.0*0.25*y*y*SQUAREROOT2;
    gphi[nn+36] = -21.0*z*z*z*SQUAREROOT2*x+189.0*0.25*z*z*SQUAREROOT2*x*x-31.5*z*SQUAREROOT2*x*x*x+21.0*0.25*x*x*x*x*SQUAREROOT2+4.5*z*SQUAREROOT2*x-9.0*0.25*x*x*SQUAREROOT2;
    gphi[nn+37] = 58.5*y*SQUAREROOT2*x+21.0*y*SQUAREROOT2*x*x*x+31.5*y*y*SQUAREROOT2*x*x+21.0*y*y*y*SQUAREROOT2*x+117.0*0.25*y*y*SQUAREROOT2-63.0*y*SQUAREROOT2*x*x-63.0*y*y*SQUAREROOT2*x+3.0*SQUAREROOT2+420.0*z*z*z*SQUAREROOT2*x-16.5*y*SQUAREROOT2-21.0*y*y*y*SQUAREROOT2+21.0*0.25*y*y*y*y*SQUAREROOT2+210.0*z*z*z*z*SQUAREROOT2+270.0*z*z*SQUAREROOT2-60.0*z*SQUAREROOT2-420.0*z*z*z*SQUAREROOT2-220.5*z*SQUAREROOT2*x*x-567.0*z*z*SQUAREROOT2*y-567.0*z*z*SQUAREROOT2*x+283.5*z*z*SQUAREROOT2*y*y+420.0*z*z*z*SQUAREROOT2*y-220.5*z*SQUAREROOT2*y*y+207.0*z*SQUAREROOT2*y+73.5*z*SQUAREROOT2*y*y*y-441.0*z*SQUAREROOT2*x*y+567.0*z*z*SQUAREROOT2*x*y+220.5*z*SQUAREROOT2*x*y*y+220.5*z*SQUAREROOT2*x*x*y+117.0*0.25*x*x*SQUAREROOT2-21.0*x*x*x*SQUAREROOT2+21.0*0.25*x*x*x*x*SQUAREROOT2-16.5*x*SQUAREROOT2+283.5*z*z*SQUAREROOT2*x*x+73.5*z*SQUAREROOT2*x*x*x+207.0*z*SQUAREROOT2*x;
    gphi[nn+38] = 0.0;
    gphi[nn+39] = -63.0*y*SQUAREROOT2*x*x-16.5*y*SQUAREROOT2-220.5*y*y*SQUAREROOT2*x+103.5*y*y*SQUAREROOT2+441.0*0.25*z*z*SQUAREROOT2*y*y-63.0*z*z*SQUAREROOT2*y+58.5*z*SQUAREROOT2*y-220.5*z*SQUAREROOT2*y*y+21.0*z*z*z*SQUAREROOT2*y+189.0*z*SQUAREROOT2*y*y*y-189.0*y*y*y*SQUAREROOT2+220.5*z*SQUAREROOT2*x*y*y-126.0*z*SQUAREROOT2*x*y+63.0*z*z*SQUAREROOT2*x*y+105.0*y*y*y*y*SQUAREROOT2+21.0*y*SQUAREROOT2*x*x*x+441.0*0.25*y*y*SQUAREROOT2*x*x+189.0*y*y*y*SQUAREROOT2*x+58.5*y*SQUAREROOT2*x+63.0*z*SQUAREROOT2*x*x*y;
    gphi[nn+40] = -220.5*z*SQUAREROOT2*x*x-63.0*z*z*SQUAREROOT2*x+21.0*z*z*z*SQUAREROOT2*x+441.0*0.25*z*z*SQUAREROOT2*x*x+58.5*z*SQUAREROOT2*x+189.0*z*SQUAREROOT2*x*x*x+58.5*y*SQUAREROOT2*x+441.0*0.25*y*y*SQUAREROOT2*x*x+189.0*y*SQUAREROOT2*x*x*x+21.0*y*y*y*SQUAREROOT2*x-16.5*x*SQUAREROOT2-63.0*y*y*SQUAREROOT2*x+103.5*x*x*SQUAREROOT2-220.5*y*SQUAREROOT2*x*x-189.0*x*x*x*SQUAREROOT2+105.0*x*x*x*x*SQUAREROOT2-126.0*z*SQUAREROOT2*x*y+63.0*z*SQUAREROOT2*x*y*y+63.0*z*z*SQUAREROOT2*x*y+220.5*z*SQUAREROOT2*x*x*y;
    gphi[nn+41] = 2.5*x*x*x*y*SQUAREROOT21-10.0*y*z*SQUAREROOT21*x*x+7.5*y*z*z*SQUAREROOT21*x-x*y*SQUAREROOT21*0.5;
    gphi[nn+42] = 50.0*y*y*z*SQUAREROOT21+60.0*y*z*z*SQUAREROOT21-25.0*y*y*y*z*SQUAREROOT21-60.0*y*y*z*z*SQUAREROOT21-40.0*y*z*z*z*SQUAREROOT21+7.5*x*x*y*SQUAREROOT21+15.0*x*y*y*SQUAREROOT21-7.5*x*y*y*y*SQUAREROOT21-7.5*x*x*y*y*SQUAREROOT21-7.0*y*y*SQUAREROOT21-2.5*y*y*y*y*SQUAREROOT21+7.5*y*y*y*SQUAREROOT21-7.0*x*y*SQUAREROOT21-2.5*x*x*x*y*SQUAREROOT21-50.0*y*y*z*SQUAREROOT21*x-25.0*y*z*SQUAREROOT21*x*x-60.0*y*z*z*SQUAREROOT21*x+50.0*x*y*z*SQUAREROOT21-24.0*y*z*SQUAREROOT21+2.0*y*SQUAREROOT21;
    gphi[nn+43] = 15.0*x*x*y*SQUAREROOT21+7.5*x*y*y*SQUAREROOT21-2.5*x*y*y*y*SQUAREROOT21-7.5*x*x*y*y*SQUAREROOT21-24.0*x*z*SQUAREROOT21-7.0*x*y*SQUAREROOT21-7.5*x*x*x*y*SQUAREROOT21-25.0*y*y*z*SQUAREROOT21*x-50.0*y*z*SQUAREROOT21*x*x-60.0*y*z*z*SQUAREROOT21*x-7.0*x*x*SQUAREROOT21+7.5*x*x*x*SQUAREROOT21-2.5*x*x*x*x*SQUAREROOT21+50.0*x*y*z*SQUAREROOT21+2.0*x*SQUAREROOT21+50.0*x*x*z*SQUAREROOT21+60.0*x*z*z*SQUAREROOT21-25.0*x*x*x*z*SQUAREROOT21-60.0*x*x*z*z*SQUAREROOT21-40.0*x*z*z*z*SQUAREROOT21;
    gphi[nn+44] = -20.0*x*y*y*y*SQUAREROOT21+25.0*x*y*y*SQUAREROOT21-25.0*x*x*y*y*SQUAREROOT21-25.0*y*y*z*SQUAREROOT21*x-7.0*x*y*SQUAREROOT21+15.0*x*x*y*SQUAREROOT21+15.0*x*y*z*SQUAREROOT21-7.5*x*x*x*y*SQUAREROOT21-15.0*y*z*SQUAREROOT21*x*x-7.5*y*z*z*SQUAREROOT21*x;
    gphi[nn+45] = 10.0*y*y*x*x-20.0*x*z*y*y-10.0*y*x*x*x+20.0*x*x*z*y;
    gphi[nn+46] = -10.0*y+20.0*y*y*y*y+10.0*y*x*x*x+150.0*x*z*z*y+40.0*y*y*x*x+50.0*y*y*y*x+140.0*z*y*y*y+210.0*z*z*y*y+220.0*x*z*y*y+80.0*x*x*z*y+80.0*z*z*z*y-50.0*y*y*y+80.0*y*z+30.0*x*y-160.0*x*y*z-80.0*x*y*y+40.0*y*y-30.0*x*x*y-150.0*y*z*z-220.0*y*y*z;
    gphi[nn+47] = -10.0*x+50.0*y*x*x*x+150.0*x*z*z*y+40.0*y*y*x*x+10.0*y*y*y*x+80.0*x*z*y*y+220.0*x*x*z*y+40.0*x*x+140.0*x*x*x*z+80.0*x*z+30.0*x*y+210.0*x*x*z*z-160.0*x*y*z+80.0*x*z*z*z-150.0*z*z*x-220.0*z*x*x-30.0*x*y*y-50.0*x*x*x+20.0*x*x*x*x-80.0*x*x*y;
    gphi[nn+48] = 50.0*y*x*x*x+30.0*x*z*z*y+110.0*y*y*x*x+50.0*y*y*y*x+80.0*x*z*y*y+80.0*x*x*z*y+30.0*x*y-60.0*x*y*z-80.0*x*y*y-80.0*x*x*y;
    gphi[nn+49] = x*y*SQUAREROOT7*(5.0*y*y-10.0*x*y+5.0*x*x-1.0)*SQUAREROOT3*0.5;
    gphi[nn+50] = 50.0*y*y*z*SQUAREROOT21+22.5*y*z*z*SQUAREROOT21-40.0*y*y*y*z*SQUAREROOT21-37.5*y*y*z*z*SQUAREROOT21-10.0*y*z*z*z*SQUAREROOT21+7.5*x*x*y*SQUAREROOT21+25.0*x*y*y*SQUAREROOT21-20.0*x*y*y*y*SQUAREROOT21-12.5*x*x*y*y*SQUAREROOT21-12.0*y*y*SQUAREROOT21-10.0*y*y*y*y*SQUAREROOT21+20.0*y*y*y*SQUAREROOT21-7.0*x*y*SQUAREROOT21-2.5*x*x*x*y*SQUAREROOT21-50.0*y*y*z*SQUAREROOT21*x-15.0*y*z*SQUAREROOT21*x*x-22.5*y*z*z*SQUAREROOT21*x+30.0*x*y*z*SQUAREROOT21-14.0*y*z*SQUAREROOT21+2.0*y*SQUAREROOT21;
    gphi[nn+51] = -22.5*y*z*z*SQUAREROOT21*x-10.0*x*x*x*x*SQUAREROOT21+20.0*x*x*x*SQUAREROOT21-12.0*x*x*SQUAREROOT21+50.0*x*x*z*SQUAREROOT21-37.5*x*x*z*z*SQUAREROOT21+22.5*x*z*z*SQUAREROOT21-40.0*x*x*x*z*SQUAREROOT21+30.0*x*y*z*SQUAREROOT21-14.0*x*z*SQUAREROOT21+2.0*x*SQUAREROOT21-10.0*x*z*z*z*SQUAREROOT21+25.0*x*x*y*SQUAREROOT21+7.5*x*y*y*SQUAREROOT21-2.5*x*y*y*y*SQUAREROOT21-7.0*x*y*SQUAREROOT21-20.0*x*x*x*y*SQUAREROOT21-15.0*y*y*z*SQUAREROOT21*x-50.0*y*z*SQUAREROOT21*x*x-12.5*x*x*y*y*SQUAREROOT21;
    gphi[nn+52] = -20.0*x*x*x*y*SQUAREROOT21+25.0*x*x*y*SQUAREROOT21-25.0*x*x*y*y*SQUAREROOT21-25.0*y*z*SQUAREROOT21*x*x-7.0*x*y*SQUAREROOT21+15.0*x*y*y*SQUAREROOT21+15.0*x*y*z*SQUAREROOT21-7.5*x*y*y*y*SQUAREROOT21-15.0*y*y*z*SQUAREROOT21*x-7.5*y*z*z*SQUAREROOT21*x;
    gphi[nn+53] = -12.0*SQUAREROOT10*x*y*z+12.0*SQUAREROOT10*x*x*y*z+18.0*SQUAREROOT10*x*y*z*z+6.0*SQUAREROOT10*x*y*y-6.0*SQUAREROOT10*x*x*y*y-6.0*SQUAREROOT10*x*y*y*y;
    gphi[nn+54] = 12.0*SQUAREROOT10*x*y*y*z-12.0*SQUAREROOT10*x*x*y*z-6.0*SQUAREROOT10*x*y*y+6.0*SQUAREROOT10*x*y*y*y+6.0*SQUAREROOT10*x*x*y-6.0*SQUAREROOT10*x*x*x*y;
    gphi[nn+55] = 24.0*SQUAREROOT10*x*y*z-36.0*SQUAREROOT10*x*x*y*z-24.0*SQUAREROOT10*x*y*y*z-18.0*SQUAREROOT10*x*y*z*z-6.0*x*y*SQUAREROOT10+18.0*SQUAREROOT10*x*x*y+12.0*SQUAREROOT10*x*y*y-12.0*SQUAREROOT10*x*x*x*y-18.0*SQUAREROOT10*x*x*y*y-6.0*SQUAREROOT10*x*y*y*y;
  }
#endif
  if (porder >= 6)
    return -1;

  return 0;
}


/******************************************************************/
//   FUNCTION Definition: PXGradientsLagrange3d
template <typename DT> ELVIS_DEVICE int
PXGradientsLagrange3d(const int porder, const DT * RESTRICT xref, DT * RESTRICT gphi)
{
  DT x, y, z;
  
  x = xref[0]; 
  y = xref[1]; 
  z = xref[2];
  
  switch (porder){
#if GEOM_USE_P0    
  case 0:
    gphi[0] = 0.0;
    gphi[1] = 0.0;
    gphi[2] = 0.0;
    return 0;
#endif
#if GEOM_USE_P1
  case 1:
    gphi[0] = -1.0;
    gphi[1] = 1.0;
    gphi[2] = 0.0;
    gphi[3] = 0.0;
    gphi[4] = -1.0;
    gphi[5] = 0.0;
    gphi[6] = 1.0;
    gphi[7] = 0.0;
    gphi[8] = -1.0;
    gphi[9] = 0.0;
    gphi[10] = 0.0;
    gphi[11] = 1.0;
    return 0;
#endif
#if GEOM_USE_P2
  case 2:
    gphi[0] = -3.0+4.0*z+4.0*y+4.0*x;
    gphi[1] = 4.0-4.0*z-4.0*y-8.0*x;
    gphi[2] = -1.0+4.0*x;
    gphi[3] = -4.0*y;
    gphi[4] = 4.0*y;
    gphi[5] = 0.0;
    gphi[6] = -4.0*z;
    gphi[7] = 4.0*z;
    gphi[8] = 0.0;
    gphi[9] = 0.0;
    gphi[10] = -3.0+4.0*z+4.0*y+4.0*x;
    gphi[11] = -4.0*x;
    gphi[12] = 0.0;
    gphi[13] = 4.0-4.0*z-8.0*y-4.0*x;
    gphi[14] = 4.0*x;
    gphi[15] = -1.0+4.0*y;
    gphi[16] = -4.0*z;
    gphi[17] = 0.0;
    gphi[18] = 4.0*z;
    gphi[19] = 0.0;
    gphi[20] = -3.0+4.0*z+4.0*y+4.0*x;
    gphi[21] = -4.0*x;
    gphi[22] = 0.0;
    gphi[23] = -4.0*y;
    gphi[24] = 0.0;
    gphi[25] = 0.0;
    gphi[26] = 4.0-8.0*z-4.0*y-4.0*x;
    gphi[27] = 4.0*x;
    gphi[28] = 4.0*y;
    gphi[29] = -1.0+4.0*z;
    
    return 0;
#endif
#if GEOM_USE_P3    
  case 3:
    gphi[0] = -5.5+18.0*z+18.0*y+18.0*x-13.5*z*z-27.0*z*y-27.0*z*x-13.5*y*y-27.0*y*x-13.5*x*x;
    gphi[1] = 9.0-22.5*z-22.5*y-45.0*x+13.5*z*z+27.0*z*y+54.0*z*x+13.5*y*y+54.0*y*x+40.5*x*x;
    gphi[2] = -4.5+4.5*z+4.5*y+36.0*x-27.0*z*x-27.0*y*x-40.5*x*x;
    gphi[3] = 1.0-9.0*x+13.5*x*x;
    gphi[4] = -22.5*y+27.0*z*y+27.0*y*y+27.0*y*x;
    gphi[5] = 27.0*y-27.0*z*y-27.0*y*y-54.0*y*x;
    gphi[6] = -4.5*y+27.0*y*x;
    gphi[7] = 4.5*y-13.5*y*y;
    gphi[8] = -4.5*y+13.5*y*y;
    gphi[9] = 0.0;
    gphi[10] = -22.5*z+27.0*z*z+27.0*z*y+27.0*z*x;
    gphi[11] = 27.0*z-27.0*z*z-27.0*z*y-54.0*z*x;
    gphi[12] = -4.5*z+27.0*z*x;
    gphi[13] = -27.0*z*y;
    gphi[14] = 27.0*z*y;
    gphi[15] = 0.0;
    gphi[16] = 4.5*z-13.5*z*z;
    gphi[17] = -4.5*z+13.5*z*z;
    gphi[18] = 0.0;
    gphi[19] = 0.0;
    gphi[20] = -5.5+18.0*z+18.0*y+18.0*x-13.5*z*z-27.0*z*y-27.0*z*x-13.5*y*y-27.0*y*x-13.5*x*x;
    gphi[21] = -22.5*x+27.0*z*x+27.0*y*x+27.0*x*x;
    gphi[22] = 4.5*x-13.5*x*x;
    gphi[23] = 0.0;
    gphi[24] = 9.0-22.5*z-45.0*y-22.5*x+13.5*z*z+54.0*z*y+27.0*z*x+40.5*y*y+54.0*y*x+13.5*x*x;
    gphi[25] = 27.0*x-27.0*z*x-54.0*y*x-27.0*x*x;
    gphi[26] = -4.5*x+13.5*x*x;
    gphi[27] = -4.5+4.5*z+36.0*y+4.5*x-27.0*z*y-40.5*y*y-27.0*y*x;
    gphi[28] = -4.5*x+27.0*y*x;
    gphi[29] = 1.0-9.0*y+13.5*y*y;
    gphi[30] = -22.5*z+27.0*z*z+27.0*z*y+27.0*z*x;
    gphi[31] = -27.0*z*x;
    gphi[32] = 0.0;
    gphi[33] = 27.0*z-27.0*z*z-54.0*z*y-27.0*z*x;
    gphi[34] = 27.0*z*x;
    gphi[35] = -4.5*z+27.0*z*y;
    gphi[36] = 4.5*z-13.5*z*z;
    gphi[37] = 0.0;
    gphi[38] = -4.5*z+13.5*z*z;
    gphi[39] = 0.0;
    gphi[40] = -5.5+18.0*z+18.0*y+18.0*x-13.5*z*z-27.0*z*y-27.0*z*x-13.5*y*y-27.0*y*x-13.5*x*x;
    gphi[41] = -22.5*x+27.0*z*x+27.0*y*x+27.0*x*x;
    gphi[42] = 4.5*x-13.5*x*x;
    gphi[43] = 0.0;
    gphi[44] = -22.5*y+27.0*z*y+27.0*y*y+27.0*y*x;
    gphi[45] = -27.0*y*x;
    gphi[46] = 0.0;
    gphi[47] = 4.5*y-13.5*y*y;
    gphi[48] = 0.0;
    gphi[49] = 0.0;
    gphi[50] = 9.0-45.0*z-22.5*y-22.5*x+40.5*z*z+54.0*z*y+54.0*z*x+13.5*y*y+27.0*y*x+13.5*x*x;
    gphi[51] = 27.0*x-54.0*z*x-27.0*y*x-27.0*x*x;
    gphi[52] = -4.5*x+13.5*x*x;
    gphi[53] = 27.0*y-54.0*z*y-27.0*y*y-27.0*y*x;
    gphi[54] = 27.0*y*x;
    gphi[55] = -4.5*y+13.5*y*y;
    gphi[56] = -4.5+36.0*z+4.5*y+4.5*x-40.5*z*z-27.0*z*y-27.0*z*x;
    gphi[57] = -4.5*x+27.0*z*x;
    gphi[58] = -4.5*y+27.0*z*y;
    gphi[59] = 1.0-9.0*z+13.5*z*z;
    return 0;
#endif
#if GEOM_USE_P4    
  case 4:
      gphi[0] = 128.0*z*y*y-25.0*ONETHIRD+256.0*z*y*x+128.0*z*z*y-160.0*z*y+140.0*ONETHIRD*x+128.0*ONETHIRD*y*y*y+128.0*y*y*x+128.0*y*x*x+128.0*z*z*x+128.0*z*x*x+128.0*ONETHIRD*z*z*z+140.0*ONETHIRD*y-80.0*z*z-80.0*y*y-160.0*y*x+128.0*ONETHIRD*x*x*x-160.0*z*x+140.0*ONETHIRD*z-80.0*x*x;
  gphi[1] = -128.0*z*y*y-512.0*z*y*x+16.0-128.0*z*z*y+192.0*z*y+384.0*y*x-416.0*ONETHIRD*x+96.0*y*y+384.0*z*x+96.0*z*z-128.0*ONETHIRD*z*z*z-256.0*z*z*x-384.0*z*x*x-128.0*ONETHIRD*y*y*y-256.0*y*y*x-384.0*y*x*x-208.0*ONETHIRD*y-208.0*ONETHIRD*z+288.0*x*x-512.0*ONETHIRD*x*x*x;
  gphi[2] = 256.0*z*y*x-12.0-32.0*z*y-288.0*y*x+152.0*x-16.0*y*y-288.0*z*x-16.0*z*z+128.0*z*z*x+384.0*z*x*x+128.0*y*y*x+384.0*y*x*x+28.0*y+28.0*z-384.0*x*x+256.0*x*x*x;
  gphi[3] = 16.0*ONETHIRD+64.0*y*x-224.0*ONETHIRD*x+64.0*z*x-128.0*z*x*x-128.0*y*x*x-16.0*ONETHIRD*y-16.0*ONETHIRD*z+224.0*x*x-512.0*ONETHIRD*x*x*x;
  gphi[4] = -1.0+44.0*ONETHIRD*x-48.0*x*x+128.0*ONETHIRD*x*x*x;
  gphi[5] = -256.0*z*y*y-256.0*z*y*x-128.0*z*z*y+192.0*z*y+192.0*y*x+192.0*y*y-128.0*y*y*y-256.0*y*y*x-128.0*y*x*x-208.0*ONETHIRD*y;
  gphi[6] = 256.0*z*y*y+512.0*z*y*x+128.0*z*z*y-224.0*z*y-448.0*y*x-224.0*y*y+128.0*y*y*y+512.0*y*y*x+384.0*y*x*x+96.0*y;
  gphi[7] = -256.0*z*y*x+32.0*z*y+320.0*y*x+32.0*y*y-256.0*y*y*x-384.0*y*x*x-32.0*y;
  gphi[8] = -64.0*y*x+128.0*y*x*x+16.0*ONETHIRD*y;
  gphi[9] = 128.0*z*y*y-32.0*z*y-32.0*y*x-144.0*y*y+128.0*y*y*y+128.0*y*y*x+28.0*y;
  gphi[10] = -128.0*z*y*y+32.0*z*y+64.0*y*x+160.0*y*y-128.0*y*y*y-256.0*y*y*x-32.0*y;
  gphi[11] = -32.0*y*x-16.0*y*y+128.0*y*y*x+4.0*y;
  gphi[12] = 32.0*y*y-128.0*ONETHIRD*y*y*y-16.0*ONETHIRD*y;
  gphi[13] = -32.0*y*y+128.0*ONETHIRD*y*y*y+16.0*ONETHIRD*y;
  gphi[14] = 0.0;
  gphi[15] = -128.0*z*y*y-256.0*z*z*y-256.0*z*y*x+192.0*z*y-128.0*z*x*x-256.0*z*z*x-128.0*z*z*z+192.0*z*x+192.0*z*z-208.0*ONETHIRD*z;
  gphi[16] = 128.0*z*y*y+256.0*z*z*y+512.0*z*y*x-224.0*z*y+384.0*z*x*x+512.0*z*z*x+128.0*z*z*z-448.0*z*x-224.0*z*z+96.0*z;
  gphi[17] = -256.0*z*y*x+32.0*z*y-384.0*z*x*x-256.0*z*z*x+320.0*z*x+32.0*z*z-32.0*z;
  gphi[18] = 128.0*z*x*x-64.0*z*x+16.0*ONETHIRD*z;
  gphi[19] = 256.0*z*y*y+256.0*z*z*y+256.0*z*y*x-224.0*z*y;
  gphi[20] = -256.0*z*y*y-256.0*z*z*y-512.0*z*y*x+256.0*z*y;
  gphi[21] = 256.0*z*y*x-32.0*z*y;
  gphi[22] = -128.0*z*y*y+32.0*z*y;
  gphi[23] = 128.0*z*y*y-32.0*z*y;
  gphi[24] = 0.0;
  gphi[25] = 128.0*z*z*y-32.0*z*y+128.0*z*z*x+128.0*z*z*z-144.0*z*z-32.0*z*x+28.0*z;
  gphi[26] = -128.0*z*z*y+32.0*z*y-256.0*z*z*x-128.0*z*z*z+160.0*z*z+64.0*z*x-32.0*z;
  gphi[27] = 128.0*z*z*x-16.0*z*z-32.0*z*x+4.0*z;
  gphi[28] = -128.0*z*z*y+32.0*z*y;
  gphi[29] = 128.0*z*z*y-32.0*z*y;
  gphi[30] = 0.0;
  gphi[31] = -128.0*ONETHIRD*z*z*z+32.0*z*z-16.0*ONETHIRD*z;
  gphi[32] = 128.0*ONETHIRD*z*z*z-32.0*z*z+16.0*ONETHIRD*z;
  gphi[33] = 0.0;
  gphi[34] = 0.0;
  gphi[35] = 128.0*z*y*y-25.0*ONETHIRD+256.0*z*y*x+128.0*z*z*y-160.0*z*y+140.0*ONETHIRD*x+128.0*ONETHIRD*y*y*y+128.0*y*y*x+128.0*y*x*x+128.0*z*z*x+128.0*z*x*x+128.0*ONETHIRD*z*z*z+140.0*ONETHIRD*y-80.0*z*z-80.0*y*y-160.0*y*x+128.0*ONETHIRD*x*x*x-160.0*z*x+140.0*ONETHIRD*z-80.0*x*x;
  gphi[36] = -256.0*z*y*x-256.0*z*x*x-128.0*z*z*x+192.0*z*x+192.0*x*x+192.0*y*x-128.0*y*y*x-256.0*y*x*x-128.0*x*x*x-208.0*ONETHIRD*x;
  gphi[37] = 128.0*z*x*x-32.0*z*x-144.0*x*x-32.0*y*x+128.0*y*x*x+128.0*x*x*x+28.0*x;
  gphi[38] = 32.0*x*x-128.0*ONETHIRD*x*x*x-16.0*ONETHIRD*x;
  gphi[39] = 0.0;
  gphi[40] = -512.0*z*y*x-128.0*z*x*x+16.0-128.0*z*z*x+192.0*z*x+96.0*x*x+384.0*y*x+384.0*z*y-128.0*ONETHIRD*z*z*z-256.0*z*z*y-384.0*z*y*y-384.0*y*y*x-256.0*y*x*x-128.0*ONETHIRD*x*x*x+96.0*z*z-208.0*ONETHIRD*x-208.0*ONETHIRD*z-416.0*ONETHIRD*y+288.0*y*y-512.0*ONETHIRD*y*y*y;
  gphi[41] = 512.0*z*y*x+256.0*z*x*x+128.0*z*z*x-224.0*z*x-224.0*x*x-448.0*y*x+384.0*y*y*x+512.0*y*x*x+128.0*x*x*x+96.0*x;
  gphi[42] = -128.0*z*x*x+32.0*z*x+160.0*x*x+64.0*y*x-256.0*y*x*x-128.0*x*x*x-32.0*x;
  gphi[43] = -32.0*x*x+128.0*ONETHIRD*x*x*x+16.0*ONETHIRD*x;
  gphi[44] = 256.0*z*y*x-12.0-32.0*z*x-16.0*x*x-288.0*y*x-288.0*z*y+128.0*z*z*y+384.0*z*y*y+384.0*y*y*x+128.0*y*x*x-16.0*z*z+28.0*x+28.0*z+152.0*y-384.0*y*y+256.0*y*y*y;
  gphi[45] = -256.0*z*y*x+32.0*z*x+32.0*x*x+320.0*y*x-384.0*y*y*x-256.0*y*x*x-32.0*x;
  gphi[46] = -16.0*x*x-32.0*y*x+128.0*y*x*x+4.0*x;
  gphi[47] = 16.0*ONETHIRD+64.0*y*x+64.0*z*y-128.0*z*y*y-128.0*y*y*x-16.0*ONETHIRD*x-16.0*ONETHIRD*z-224.0*ONETHIRD*y+224.0*y*y-512.0*ONETHIRD*y*y*y;
  gphi[48] = -64.0*y*x+128.0*y*y*x+16.0*ONETHIRD*x;
  gphi[49] = -1.0+44.0*ONETHIRD*y-48.0*y*y+128.0*ONETHIRD*y*y*y;
  gphi[50] = -128.0*z*y*y-256.0*z*z*y-256.0*z*y*x+192.0*z*y-128.0*z*x*x-256.0*z*z*x-128.0*z*z*z+192.0*z*x+192.0*z*z-208.0*ONETHIRD*z;
  gphi[51] = 256.0*z*y*x+256.0*z*z*x+256.0*z*x*x-224.0*z*x;
  gphi[52] = -128.0*z*x*x+32.0*z*x;
  gphi[53] = 0.0;
  gphi[54] = 512.0*z*y*x+256.0*z*z*x+128.0*z*x*x-224.0*z*x-448.0*z*y+128.0*z*z*z+512.0*z*z*y+384.0*z*y*y-224.0*z*z+96.0*z;
  gphi[55] = -512.0*z*y*x-256.0*z*z*x-256.0*z*x*x+256.0*z*x;
  gphi[56] = 128.0*z*x*x-32.0*z*x;
  gphi[57] = -256.0*z*y*x+32.0*z*x+320.0*z*y-256.0*z*z*y-384.0*z*y*y+32.0*z*z-32.0*z;
  gphi[58] = 256.0*z*y*x-32.0*z*x;
  gphi[59] = -64.0*z*y+128.0*z*y*y+16.0*ONETHIRD*z;
  gphi[60] = 128.0*z*z*y-32.0*z*y+128.0*z*z*x+128.0*z*z*z-144.0*z*z-32.0*z*x+28.0*z;
  gphi[61] = -128.0*z*z*x+32.0*z*x;
  gphi[62] = 0.0;
  gphi[63] = -128.0*z*z*x+32.0*z*x+64.0*z*y-256.0*z*z*y-128.0*z*z*z+160.0*z*z-32.0*z;
  gphi[64] = 128.0*z*z*x-32.0*z*x;
  gphi[65] = -32.0*z*y+128.0*z*z*y-16.0*z*z+4.0*z;
  gphi[66] = -128.0*ONETHIRD*z*z*z+32.0*z*z-16.0*ONETHIRD*z;
  gphi[67] = 0.0;
  gphi[68] = 128.0*ONETHIRD*z*z*z-32.0*z*z+16.0*ONETHIRD*z;
  gphi[69] = 0.0;
  gphi[70] = 128.0*z*y*y-25.0*ONETHIRD+256.0*z*y*x+128.0*z*z*y-160.0*z*y+140.0*ONETHIRD*x+128.0*ONETHIRD*y*y*y+128.0*y*y*x+128.0*y*x*x+128.0*z*z*x+128.0*z*x*x+128.0*ONETHIRD*z*z*z+140.0*ONETHIRD*y-80.0*z*z-80.0*y*y-160.0*y*x+128.0*ONETHIRD*x*x*x-160.0*z*x+140.0*ONETHIRD*z-80.0*x*x;
  gphi[71] = -256.0*z*y*x-256.0*z*x*x-128.0*z*z*x+192.0*z*x+192.0*x*x+192.0*y*x-128.0*y*y*x-256.0*y*x*x-128.0*x*x*x-208.0*ONETHIRD*x;
  gphi[72] = 128.0*z*x*x-32.0*z*x-144.0*x*x-32.0*y*x+128.0*y*x*x+128.0*x*x*x+28.0*x;
  gphi[73] = 32.0*x*x-128.0*ONETHIRD*x*x*x-16.0*ONETHIRD*x;
  gphi[74] = 0.0;
  gphi[75] = -256.0*z*y*y-256.0*z*y*x-128.0*z*z*y+192.0*z*y+192.0*y*x+192.0*y*y-128.0*y*y*y-256.0*y*y*x-128.0*y*x*x-208.0*ONETHIRD*y;
  gphi[76] = 256.0*y*y*x+256.0*y*x*x+256.0*z*y*x-224.0*y*x;
  gphi[77] = -128.0*y*x*x+32.0*y*x;
  gphi[78] = 0.0;
  gphi[79] = 128.0*z*y*y-32.0*z*y-32.0*y*x-144.0*y*y+128.0*y*y*y+128.0*y*y*x+28.0*y;
  gphi[80] = -128.0*y*y*x+32.0*y*x;
  gphi[81] = 0.0;
  gphi[82] = 32.0*y*y-128.0*ONETHIRD*y*y*y-16.0*ONETHIRD*y;
  gphi[83] = 0.0;
  gphi[84] = 0.0;
  gphi[85] = -128.0*y*y*x-416.0*ONETHIRD*z+288.0*z*z-512.0*ONETHIRD*z*z*z+16.0-512.0*z*y*x-128.0*y*x*x+192.0*y*x-128.0*ONETHIRD*x*x*x-256.0*z*x*x-384.0*z*z*x+96.0*y*y-384.0*z*z*y-256.0*z*y*y-128.0*ONETHIRD*y*y*y+384.0*z*y+96.0*x*x-208.0*ONETHIRD*y+384.0*z*x-208.0*ONETHIRD*x;
  gphi[86] = 128.0*y*y*x+512.0*z*y*x+256.0*y*x*x-224.0*y*x+128.0*x*x*x+512.0*z*x*x+384.0*z*z*x-224.0*x*x-448.0*z*x+96.0*x;
  gphi[87] = -128.0*y*x*x+32.0*y*x-128.0*x*x*x-256.0*z*x*x+160.0*x*x+64.0*z*x-32.0*x;
  gphi[88] = -32.0*x*x+128.0*ONETHIRD*x*x*x+16.0*ONETHIRD*x;
  gphi[89] = 256.0*y*y*x+512.0*z*y*x+128.0*y*x*x-224.0*y*x-224.0*y*y+384.0*z*z*y+512.0*z*y*y+128.0*y*y*y-448.0*z*y+96.0*y;
  gphi[90] = -256.0*y*y*x-512.0*z*y*x-256.0*y*x*x+256.0*y*x;
  gphi[91] = 128.0*y*x*x-32.0*y*x;
  gphi[92] = -128.0*y*y*x+32.0*y*x+160.0*y*y-256.0*z*y*y-128.0*y*y*y+64.0*z*y-32.0*y;
  gphi[93] = 128.0*y*y*x-32.0*y*x;
  gphi[94] = -32.0*y*y+128.0*ONETHIRD*y*y*y+16.0*ONETHIRD*y;
  gphi[95] = -12.0+256.0*z*y*x-32.0*y*x-16.0*y*y+128.0*z*y*y+128.0*z*x*x+384.0*z*z*y+384.0*z*z*x-288.0*z*y-288.0*z*x-16.0*x*x+256.0*z*z*z+152.0*z-384.0*z*z+28.0*x+28.0*y;
  gphi[96] = -256.0*z*y*x+32.0*y*x-256.0*z*x*x-384.0*z*z*x+320.0*z*x+32.0*x*x-32.0*x;
  gphi[97] = 128.0*z*x*x-32.0*z*x-16.0*x*x+4.0*x;
  gphi[98] = -256.0*z*y*x+32.0*y*x+32.0*y*y-256.0*z*y*y-384.0*z*z*y+320.0*z*y-32.0*y;
  gphi[99] = 256.0*z*y*x-32.0*y*x;
  gphi[100] = -16.0*y*y+128.0*z*y*y-32.0*z*y+4.0*y;
  gphi[101] = 16.0*ONETHIRD-128.0*z*z*y-128.0*z*z*x+64.0*z*y+64.0*z*x-512.0*ONETHIRD*z*z*z-224.0*ONETHIRD*z+224.0*z*z-16.0*ONETHIRD*x-16.0*ONETHIRD*y;
  gphi[102] = 128.0*z*z*x-64.0*z*x+16.0*ONETHIRD*x;
  gphi[103] = 128.0*z*z*y-64.0*z*y+16.0*ONETHIRD*y;
  gphi[104] = -1.0+128.0*ONETHIRD*z*z*z+44.0*ONETHIRD*z-48.0*z*z;

    return 0;
#endif
#if GEOM_USE_P5
  case 5:
     gphi[0] = 375.0*0.25*z+375.0*0.25*x+375.0*0.25*y-1562.5*z*y*x*x-1562.5*z*y*y*x-1562.5*z*z*y*x+1875.0*z*y*x-137.0/12.0-2125.0/8.0*x*x+937.5*z*z*y-3125.0/24.0*x*x*x*x-3125.0*ONESIXTH*z*z*z*x-2125.0/8.0*y*y+312.5*y*y*y+312.5*x*x*x-3125.0*0.25*y*y*x*x-3125.0*ONESIXTH*y*x*x*x-3125.0*ONESIXTH*y*y*y*x-3125.0*ONESIXTH*z*x*x*x-3125.0*ONESIXTH*z*y*y*y-3125.0*0.25*z*z*x*x+937.5*z*z*x-2125.0*0.25*z*x-3125.0*0.25*z*z*y*y-2125.0/8.0*z*z+312.5*z*z*z-3125.0/24.0*y*y*y*y-2125.0*0.25*z*y-3125.0/24.0*z*z*z*z+937.5*y*x*x+937.5*z*y*y+937.5*z*x*x-3125.0*ONESIXTH*z*z*z*y+937.5*y*y*x-2125.0*0.25*y*x;
  gphi[1] = 25.0-1925.0/12.0*z-1925.0*ONESIXTH*x-1925.0/12.0*y+4687.5*z*y*x*x+3125.0*z*y*y*x+3125.0*z*z*y*x-4375.0*z*y*x+8875.0/8.0*x*x-4375.0*0.25*z*z*y+15625.0/24.0*x*x*x*x+3125.0*ONETHIRD*z*z*z*x+8875.0/24.0*y*y-4375.0/12.0*y*y*y-4375.0*ONETHIRD*x*x*x+9375.0*0.25*y*y*x*x+6250.0*ONETHIRD*y*x*x*x+3125.0*ONETHIRD*y*y*y*x+6250.0*ONETHIRD*z*x*x*x+3125.0*ONESIXTH*z*y*y*y+9375.0*0.25*z*z*x*x-2187.5*z*z*x+8875.0*ONESIXTH*z*x+3125.0*0.25*z*z*y*y+8875.0/24.0*z*z-4375.0/12.0*z*z*z+3125.0/24.0*y*y*y*y+8875.0/12.0*z*y+3125.0/24.0*z*z*z*z-13125.0*0.25*y*x*x-4375.0*0.25*z*y*y-13125.0*0.25*z*x*x+3125.0*ONESIXTH*z*z*z*y-2187.5*y*y*x+8875.0*ONESIXTH*y*x;
  gphi[2] = -25.0+1175.0/12.0*z+2675.0*ONESIXTH*x+1175.0/12.0*y-4687.5*z*y*x*x-1562.5*z*y*y*x-1562.5*z*z*y*x+3125.0*z*y*x-7375.0*0.25*x*x+625.0*0.25*z*z*y-15625.0/12.0*x*x*x*x-3125.0*ONESIXTH*z*z*z*x-125.0*y*y+625.0/12.0*y*y*y+8125.0*ONETHIRD*x*x*x-9375.0*0.25*y*y*x*x-3125.0*y*x*x*x-3125.0*ONESIXTH*y*y*y*x-3125.0*z*x*x*x-9375.0*0.25*z*z*x*x+1562.5*z*z*x-8875.0*ONESIXTH*z*x-125.0*z*z+625.0/12.0*z*z*z-250.0*z*y+16875.0*0.25*y*x*x+625.0*0.25*z*y*y+16875.0*0.25*z*x*x+1562.5*y*y*x-8875.0*ONESIXTH*y*x;
  gphi[3] = 15625.0/12.0*x*x*x*x+3125.0*0.25*z*z*x*x+50.0*ONETHIRD-9375.0*0.25*y*x*x-312.5*y*y*x-9375.0*0.25*z*x*x-312.5*z*z*x+3875.0*ONESIXTH*y*x+125.0*ONESIXTH*y*y+3875.0*ONESIXTH*z*x+125.0*ONESIXTH*z*z+1562.5*z*y*x*x+6250.0*ONETHIRD*y*x*x*x+3125.0*0.25*y*y*x*x+6250.0*ONETHIRD*z*x*x*x-37.5*y-625.0*z*y*x+125.0*ONETHIRD*z*y-37.5*z-325.0*x-2500.0*x*x*x+6125.0*0.25*x*x;
  gphi[4] = -15625.0/24.0*x*x*x*x-25.0*0.25+1875.0*0.25*y*x*x+1875.0*0.25*z*x*x-1375.0/12.0*y*x-1375.0/12.0*z*x-3125.0*ONESIXTH*y*x*x*x-3125.0*ONESIXTH*z*x*x*x+25.0*0.25*y+25.0*0.25*z+1525.0/12.0*x+6875.0*ONESIXTH*x*x*x-5125.0/8.0*x*x;
  gphi[5] = 3125.0/24.0*x*x*x*x+1.0-125.0*ONESIXTH*x-625.0*ONETHIRD*x*x*x+875.0/8.0*x*x;
  gphi[6] = 1562.5*z*y*y*y+1562.5*z*z*y*x+1562.5*z*z*y*y+3125.0*ONESIXTH*z*z*z*y-2187.5*z*y*x-2187.5*z*y*y-4375.0*0.25*z*z*y+8875.0/12.0*z*y+1562.5*z*y*x*x+3125.0*z*y*y*x+3125.0*ONESIXTH*y*y*y*y+1562.5*y*y*y*x+1562.5*y*y*x*x+3125.0*ONESIXTH*y*x*x*x-2187.5*y*y*x-4375.0*0.25*y*x*x-4375.0*0.25*y*y*y+8875.0/12.0*y*x+8875.0/12.0*y*y-1925.0/12.0*y;
  gphi[7] = 3750.0*y*x*x+5000.0*y*y*x+1250.0*y*y*y-5875.0*ONETHIRD*y*x-5875.0*ONESIXTH*y*y-4687.5*z*y*x*x-6250.0*z*y*y*x-3125.0*z*z*y*x-6250.0*ONETHIRD*y*x*x*x-4687.5*y*y*x*x-3125.0*y*y*y*x-1562.5*z*z*y*y+250.0*y+5000.0*z*y*x+2500.0*z*y*y+1250.0*z*z*y-5875.0*ONESIXTH*z*y-1562.5*z*y*y*y-3125.0*ONESIXTH*z*z*z*y-3125.0*ONESIXTH*y*y*y*y;
  gphi[8] = -4687.5*y*x*x-3437.5*y*y*x-625.0*0.25*y*y*y+1812.5*y*x+1125.0*0.25*y*y+4687.5*z*y*x*x+3125.0*z*y*y*x+1562.5*z*z*y*x+3125.0*y*x*x*x+4687.5*y*y*x*x+1562.5*y*y*y*x-125.0*y-3437.5*z*y*x-312.5*z*y*y-625.0*0.25*z*z*y+1125.0*0.25*z*y;
  gphi[9] = 2500.0*y*x*x+625.0*y*y*x-2125.0*ONETHIRD*y*x-125.0*ONETHIRD*y*y-1562.5*z*y*x*x-6250.0*ONETHIRD*y*x*x*x-1562.5*y*y*x*x+125.0*ONETHIRD*y+625.0*z*y*x-125.0*ONETHIRD*z*y;
  gphi[10] = -1875.0*0.25*y*x*x+1375.0/12.0*y*x+3125.0*ONESIXTH*y*x*x*x-25.0*0.25*y;
  gphi[11] = -1562.5*z*y*y*y-3125.0*0.25*z*z*y*y+312.5*z*y*x+1562.5*z*y*y+625.0*0.25*z*z*y-250.0*z*y-1562.5*z*y*y*x-3125.0*0.25*y*y*y*y-1562.5*y*y*y*x-3125.0*0.25*y*y*x*x+1562.5*y*y*x+625.0*0.25*y*x*x+5625.0*0.25*y*y*y-250.0*y*x-8875.0/12.0*y*y+1175.0/12.0*y;
  gphi[12] = -1875.0*0.25*y*x*x-3437.5*y*y*x-1562.5*y*y*y+562.5*y*x+3625.0*0.25*y*y+3125.0*z*y*y*x+9375.0*0.25*y*y*x*x+3125.0*y*y*y*x+1562.5*z*y*y*y+3125.0*0.25*z*z*y*y-125.0*y-625.0*z*y*x-6875.0*0.25*z*y*y-625.0*0.25*z*z*y+1125.0*0.25*z*y+3125.0*0.25*y*y*y*y;
  gphi[13] = 1875.0*0.25*y*x*x+2187.5*y*y*x+625.0*0.25*y*y*y-375.0*y*x-187.5*y*y-1562.5*z*y*y*x-9375.0*0.25*y*y*x*x-1562.5*y*y*y*x+125.0*0.25*y+312.5*z*y*x+625.0*0.25*z*y*y-125.0*0.25*z*y;
  gphi[14] = -625.0*0.25*y*x*x-312.5*y*y*x+62.5*y*x+125.0*ONESIXTH*y*y+3125.0*0.25*y*y*x*x-25.0*ONESIXTH*y;
  gphi[15] = -312.5*y*y*x-3125.0*0.25*y*y*y+125.0*ONETHIRD*y*x+3875.0/12.0*y*y+3125.0*ONESIXTH*y*y*y*x+3125.0*ONESIXTH*z*y*y*y-37.5*y-312.5*z*y*y+125.0*ONETHIRD*z*y+3125.0*ONESIXTH*y*y*y*y;
  gphi[16] = 625.0*y*y*x+2500.0*ONETHIRD*y*y*y-250.0*ONETHIRD*y*x-2125.0*ONESIXTH*y*y-3125.0*ONETHIRD*y*y*y*x-3125.0*ONESIXTH*z*y*y*y+125.0*ONETHIRD*y+312.5*z*y*y-125.0*ONETHIRD*z*y-3125.0*ONESIXTH*y*y*y*y;
  gphi[17] = -312.5*y*y*x-625.0/12.0*y*y*y+125.0*ONETHIRD*y*x+125.0*0.25*y*y+3125.0*ONESIXTH*y*y*y*x-25.0*ONESIXTH*y;
  gphi[18] = 625.0*0.25*y*y*y-1375.0/24.0*y*y+25.0*0.25*y-3125.0/24.0*y*y*y*y;
  gphi[19] = -625.0*0.25*y*y*y+1375.0/24.0*y*y-25.0*0.25*y+3125.0/24.0*y*y*y*y;
  gphi[20] = 0.0;
  gphi[21] = 3125.0*ONESIXTH*z*z*z*z+3125.0*ONESIXTH*z*y*y*y+3125.0*z*z*y*x+1562.5*z*z*y*y+1562.5*z*z*z*y-2187.5*z*y*x-4375.0*0.25*z*y*y-2187.5*z*z*y+8875.0/12.0*z*y+1562.5*z*y*x*x+1562.5*z*y*y*x-4375.0*0.25*z*z*z+3125.0*ONESIXTH*z*x*x*x+1562.5*z*z*x*x+1562.5*z*z*z*x-4375.0*0.25*z*x*x-2187.5*z*z*x+8875.0/12.0*z*z-1925.0/12.0*z+8875.0/12.0*z*x;
  gphi[22] = -4687.5*z*y*x*x-3125.0*z*y*y*x-3125.0*ONESIXTH*z*y*y*y-6250.0*z*z*y*x-1562.5*z*z*y*y-1562.5*z*z*z*y+5000.0*z*y*x+1250.0*z*y*y+2500.0*z*z*y-5875.0*ONESIXTH*z*y-6250.0*ONETHIRD*z*x*x*x-3125.0*z*z*z*x-4687.5*z*z*x*x-3125.0*ONESIXTH*z*z*z*z+5000.0*z*z*x+3750.0*z*x*x+1250.0*z*z*z-5875.0*ONESIXTH*z*z-5875.0*ONETHIRD*z*x+250.0*z;
  gphi[23] = 4687.5*z*y*x*x+1562.5*z*y*y*x+3125.0*z*z*y*x-3437.5*z*y*x-625.0*0.25*z*y*y-312.5*z*z*y+1125.0*0.25*z*y+3125.0*z*x*x*x+1562.5*z*z*z*x+4687.5*z*z*x*x-3437.5*z*z*x-4687.5*z*x*x-625.0*0.25*z*z*z+1125.0*0.25*z*z+1812.5*z*x-125.0*z;
  gphi[24] = -1562.5*z*y*x*x+625.0*z*y*x-125.0*ONETHIRD*z*y-6250.0*ONETHIRD*z*x*x*x-1562.5*z*z*x*x+625.0*z*z*x+2500.0*z*x*x-125.0*ONETHIRD*z*z-2125.0*ONETHIRD*z*x+125.0*ONETHIRD*z;
  gphi[25] = 3125.0*ONESIXTH*z*x*x*x-1875.0*0.25*z*x*x+1375.0/12.0*z*x-25.0*0.25*z;
  gphi[26] = -1562.5*z*y*x*x-3125.0*z*y*y*x-1562.5*z*y*y*y-3125.0*z*z*y*x-3125.0*z*z*y*y-1562.5*z*z*z*y+2500.0*z*y*x+2500.0*z*y*y+2500.0*z*z*y-5875.0*ONESIXTH*z*y;
  gphi[27] = 4687.5*z*y*x*x+6250.0*z*y*y*x+1562.5*z*y*y*y+6250.0*z*z*y*x+3125.0*z*z*y*y+1562.5*z*z*z*y-5625.0*z*y*x-2812.5*z*y*y-2812.5*z*z*y+1250.0*z*y;
  gphi[28] = -4687.5*z*y*x*x-3125.0*z*y*y*x-3125.0*z*z*y*x+3750.0*z*y*x+312.5*z*y*y+312.5*z*z*y-312.5*z*y;
  gphi[29] = 1562.5*z*y*x*x-625.0*z*y*x+125.0*ONETHIRD*z*y;
  gphi[30] = 1562.5*z*y*y*x+1562.5*z*y*y*y+1562.5*z*z*y*y-312.5*z*y*x-6875.0*0.25*z*y*y-312.5*z*z*y+1125.0*0.25*z*y;
  gphi[31] = -3125.0*z*y*y*x-1562.5*z*y*y*y-1562.5*z*z*y*y+625.0*z*y*x+1875.0*z*y*y+312.5*z*z*y-312.5*z*y;
  gphi[32] = 1562.5*z*y*y*x-312.5*z*y*x-625.0*0.25*z*y*y+125.0*0.25*z*y;
  gphi[33] = -3125.0*ONESIXTH*z*y*y*y+312.5*z*y*y-125.0*ONETHIRD*z*y;
  gphi[34] = 3125.0*ONESIXTH*z*y*y*y-312.5*z*y*y+125.0*ONETHIRD*z*y;
  gphi[35] = 0.0;
  gphi[36] = -3125.0*0.25*z*z*z*z-1562.5*z*z*y*x-3125.0*0.25*z*z*y*y-1562.5*z*z*z*y+312.5*z*y*x+625.0*0.25*z*y*y+1562.5*z*z*y-250.0*z*y+5625.0*0.25*z*z*z-3125.0*0.25*z*z*x*x-1562.5*z*z*z*x+625.0*0.25*z*x*x+1562.5*z*z*x-8875.0/12.0*z*z+1175.0/12.0*z-250.0*z*x;
  gphi[37] = -625.0*z*y*x-625.0*0.25*z*y*y+1125.0*0.25*z*y+1562.5*z*z*z*y+3125.0*0.25*z*z*y*y+3125.0*z*z*y*x-6875.0*0.25*z*z*y+9375.0*0.25*z*z*x*x-1875.0*0.25*z*x*x+3125.0*z*z*z*x+3125.0*0.25*z*z*z*z-125.0*z-3437.5*z*z*x-1562.5*z*z*z+562.5*z*x+3625.0*0.25*z*z;
  gphi[38] = 312.5*z*y*x-125.0*0.25*z*y-1562.5*z*z*y*x+625.0*0.25*z*z*y-9375.0*0.25*z*z*x*x+1875.0*0.25*z*x*x-1562.5*z*z*z*x+125.0*0.25*z+2187.5*z*z*x+625.0*0.25*z*z*z-375.0*z*x-187.5*z*z;
  gphi[39] = 3125.0*0.25*z*z*x*x-625.0*0.25*z*x*x-25.0*ONESIXTH*z-312.5*z*z*x+62.5*z*x+125.0*ONESIXTH*z*z;
  gphi[40] = -312.5*z*y*x-312.5*z*y*y+1125.0*0.25*z*y+1562.5*z*z*z*y+1562.5*z*z*y*y+1562.5*z*z*y*x-6875.0*0.25*z*z*y;
  gphi[41] = 625.0*z*y*x+312.5*z*y*y-312.5*z*y-1562.5*z*z*z*y-1562.5*z*z*y*y-3125.0*z*z*y*x+1875.0*z*z*y;
  gphi[42] = -312.5*z*y*x+125.0*0.25*z*y+1562.5*z*z*y*x-625.0*0.25*z*z*y;
  gphi[43] = 625.0*0.25*z*y*y-125.0*0.25*z*y-3125.0*0.25*z*z*y*y+625.0*0.25*z*z*y;
  gphi[44] = -625.0*0.25*z*y*y+125.0*0.25*z*y+3125.0*0.25*z*z*y*y-625.0*0.25*z*z*y;
  gphi[45] = 0.0;
  gphi[46] = 125.0*ONETHIRD*z*y+3125.0*ONESIXTH*z*z*z*y-312.5*z*z*y-312.5*z*z*x-3125.0*0.25*z*z*z+125.0*ONETHIRD*z*x+3875.0/12.0*z*z+3125.0*ONESIXTH*z*z*z*x-37.5*z+3125.0*ONESIXTH*z*z*z*z;
  gphi[47] = -125.0*ONETHIRD*z*y-3125.0*ONESIXTH*z*z*z*y+312.5*z*z*y+625.0*z*z*x+2500.0*ONETHIRD*z*z*z-250.0*ONETHIRD*z*x-2125.0*ONESIXTH*z*z-3125.0*ONETHIRD*z*z*z*x+125.0*ONETHIRD*z-3125.0*ONESIXTH*z*z*z*z;
  gphi[48] = -312.5*z*z*x-625.0/12.0*z*z*z+125.0*ONETHIRD*z*x+125.0*0.25*z*z+3125.0*ONESIXTH*z*z*z*x-25.0*ONESIXTH*z;
  gphi[49] = -125.0*ONETHIRD*z*y-3125.0*ONESIXTH*z*z*z*y+312.5*z*z*y;
  gphi[50] = 125.0*ONETHIRD*z*y+3125.0*ONESIXTH*z*z*z*y-312.5*z*z*y;
  gphi[51] = 0.0;
  gphi[52] = -3125.0/24.0*z*z*z*z+625.0*0.25*z*z*z-1375.0/24.0*z*z+25.0*0.25*z;
  gphi[53] = 3125.0/24.0*z*z*z*z-625.0*0.25*z*z*z+1375.0/24.0*z*z-25.0*0.25*z;
  gphi[54] = 0.0;
  gphi[55] = 0.0;
  gphi[56] = 375.0*0.25*z+375.0*0.25*x+375.0*0.25*y-1562.5*z*y*x*x-1562.5*z*y*y*x-1562.5*z*z*y*x+1875.0*z*y*x-137.0/12.0-2125.0/8.0*x*x+937.5*z*z*y-3125.0/24.0*x*x*x*x-3125.0*ONESIXTH*z*z*z*x-2125.0/8.0*y*y+312.5*y*y*y+312.5*x*x*x-3125.0*0.25*y*y*x*x-3125.0*ONESIXTH*y*x*x*x-3125.0*ONESIXTH*y*y*y*x-3125.0*ONESIXTH*z*x*x*x-3125.0*ONESIXTH*z*y*y*y-3125.0*0.25*z*z*x*x+937.5*z*z*x-2125.0*0.25*z*x-3125.0*0.25*z*z*y*y-2125.0/8.0*z*z+312.5*z*z*z-3125.0/24.0*y*y*y*y-2125.0*0.25*z*y-3125.0/24.0*z*z*z*z+937.5*y*x*x+937.5*z*y*y+937.5*z*x*x-3125.0*ONESIXTH*z*z*z*y+937.5*y*y*x-2125.0*0.25*y*x;
  gphi[57] = 1562.5*z*y*y*x+1562.5*z*z*x*x+1562.5*z*z*y*x+3125.0*ONESIXTH*z*z*z*x-2187.5*z*x*x-2187.5*z*y*x-4375.0*0.25*z*z*x+8875.0/12.0*z*x+1562.5*z*x*x*x+3125.0*z*y*x*x+3125.0*ONESIXTH*y*y*y*x+1562.5*y*y*x*x+1562.5*y*x*x*x+3125.0*ONESIXTH*x*x*x*x-2187.5*y*x*x-4375.0*0.25*x*x*x-4375.0*0.25*y*y*x+8875.0/12.0*x*x+8875.0/12.0*y*x-1925.0/12.0*x;
  gphi[58] = -3125.0*0.25*z*z*x*x+1562.5*z*x*x+312.5*z*y*x+625.0*0.25*z*z*x-250.0*z*x-1562.5*z*x*x*x-1562.5*z*y*x*x-3125.0*0.25*y*y*x*x-1562.5*y*x*x*x-3125.0*0.25*x*x*x*x+1562.5*y*x*x+5625.0*0.25*x*x*x+625.0*0.25*y*y*x-8875.0/12.0*x*x-250.0*y*x+1175.0/12.0*x;
  gphi[59] = -3125.0*0.25*x*x*x-312.5*y*x*x+3875.0/12.0*x*x+125.0*ONETHIRD*y*x+3125.0*ONESIXTH*z*x*x*x+3125.0*ONESIXTH*x*x*x*x+3125.0*ONESIXTH*y*x*x*x-37.5*x-312.5*z*x*x+125.0*ONETHIRD*z*x;
  gphi[60] = 625.0*0.25*x*x*x-1375.0/24.0*x*x-3125.0/24.0*x*x*x*x+25.0*0.25*x;
  gphi[61] = 0.0;
  gphi[62] = 25.0-1925.0/12.0*z-1925.0/12.0*x-1925.0*ONESIXTH*y+3125.0*z*y*x*x+4687.5*z*y*y*x+3125.0*z*z*y*x-4375.0*z*y*x+8875.0/24.0*x*x-2187.5*z*z*y+3125.0/24.0*x*x*x*x+3125.0*ONESIXTH*z*z*z*x+8875.0/8.0*y*y-4375.0*ONETHIRD*y*y*y-4375.0/12.0*x*x*x+9375.0*0.25*y*y*x*x+3125.0*ONETHIRD*y*x*x*x+6250.0*ONETHIRD*y*y*y*x+3125.0*ONESIXTH*z*x*x*x+6250.0*ONETHIRD*z*y*y*y+3125.0*0.25*z*z*x*x-4375.0*0.25*z*z*x+8875.0/12.0*z*x+9375.0*0.25*z*z*y*y+8875.0/24.0*z*z-4375.0/12.0*z*z*z+15625.0/24.0*y*y*y*y+8875.0*ONESIXTH*z*y+3125.0/24.0*z*z*z*z-2187.5*y*x*x-13125.0*0.25*z*y*y-4375.0*0.25*z*x*x+3125.0*ONETHIRD*z*z*z*y-13125.0*0.25*y*y*x+8875.0*ONESIXTH*y*x;
  gphi[63] = 1250.0*x*x*x+5000.0*y*x*x+3750.0*y*y*x-5875.0*ONESIXTH*x*x-5875.0*ONETHIRD*y*x-1562.5*z*x*x*x-6250.0*z*y*x*x-1562.5*z*z*x*x-3125.0*ONESIXTH*x*x*x*x-3125.0*y*x*x*x-4687.5*y*y*x*x-3125.0*z*z*y*x+250.0*x+2500.0*z*x*x+5000.0*z*y*x+1250.0*z*z*x-5875.0*ONESIXTH*z*x-4687.5*z*y*y*x-3125.0*ONESIXTH*z*z*z*x-6250.0*ONETHIRD*y*y*y*x;
  gphi[64] = -1562.5*x*x*x-3437.5*y*x*x-1875.0*0.25*y*y*x+3625.0*0.25*x*x+562.5*y*x+1562.5*z*x*x*x+3125.0*z*y*x*x+3125.0*0.25*z*z*x*x+3125.0*0.25*x*x*x*x+3125.0*y*x*x*x+9375.0*0.25*y*y*x*x-125.0*x-6875.0*0.25*z*x*x-625.0*z*y*x-625.0*0.25*z*z*x+1125.0*0.25*z*x;
  gphi[65] = 2500.0*ONETHIRD*x*x*x+625.0*y*x*x-2125.0*ONESIXTH*x*x-250.0*ONETHIRD*y*x-3125.0*ONESIXTH*z*x*x*x-3125.0*ONESIXTH*x*x*x*x-3125.0*ONETHIRD*y*x*x*x+125.0*ONETHIRD*x+312.5*z*x*x-125.0*ONETHIRD*z*x;
  gphi[66] = -625.0*0.25*x*x*x+1375.0/24.0*x*x+3125.0/24.0*x*x*x*x-25.0*0.25*x;
  gphi[67] = -25.0+1175.0/12.0*z+1175.0/12.0*x+2675.0*ONESIXTH*y-1562.5*z*y*x*x-4687.5*z*y*y*x-1562.5*z*z*y*x+3125.0*z*y*x-125.0*x*x+1562.5*z*z*y-7375.0*0.25*y*y+8125.0*ONETHIRD*y*y*y+625.0/12.0*x*x*x-9375.0*0.25*y*y*x*x-3125.0*ONESIXTH*y*x*x*x-3125.0*y*y*y*x-3125.0*z*y*y*y+625.0*0.25*z*z*x-250.0*z*x-9375.0*0.25*z*z*y*y-125.0*z*z+625.0/12.0*z*z*z-15625.0/12.0*y*y*y*y-8875.0*ONESIXTH*z*y+1562.5*y*x*x+16875.0*0.25*z*y*y+625.0*0.25*z*x*x-3125.0*ONESIXTH*z*z*z*y+16875.0*0.25*y*y*x-8875.0*ONESIXTH*y*x;
  gphi[68] = -625.0*0.25*x*x*x-3437.5*y*x*x-4687.5*y*y*x+1125.0*0.25*x*x+1812.5*y*x+3125.0*z*y*x*x+1562.5*y*x*x*x+4687.5*y*y*x*x+4687.5*z*y*y*x+1562.5*z*z*y*x-125.0*x-312.5*z*x*x-3437.5*z*y*x-625.0*0.25*z*z*x+1125.0*0.25*z*x+3125.0*y*y*y*x;
  gphi[69] = 625.0*0.25*x*x*x+2187.5*y*x*x+1875.0*0.25*y*y*x-187.5*x*x-375.0*y*x-1562.5*z*y*x*x-1562.5*y*x*x*x-9375.0*0.25*y*y*x*x+125.0*0.25*x+625.0*0.25*z*x*x+312.5*z*y*x-125.0*0.25*z*x;
  gphi[70] = -625.0/12.0*x*x*x-312.5*y*x*x+125.0*0.25*x*x+125.0*ONETHIRD*y*x+3125.0*ONESIXTH*y*x*x*x-25.0*ONESIXTH*x;
  gphi[71] = -9375.0*0.25*z*y*y+50.0*ONETHIRD-312.5*y*x*x-9375.0*0.25*y*y*x+125.0*ONESIXTH*x*x+3875.0*ONESIXTH*y*x+3125.0*0.25*y*y*x*x+1562.5*z*y*y*x-37.5*x-625.0*z*y*x+125.0*ONETHIRD*z*x-312.5*z*z*y+6250.0*ONETHIRD*y*y*y*x+3875.0*ONESIXTH*z*y+6250.0*ONETHIRD*z*y*y*y+125.0*ONESIXTH*z*z+3125.0*0.25*z*z*y*y-325.0*y+6125.0*0.25*y*y-2500.0*y*y*y+15625.0/12.0*y*y*y*y-37.5*z;
  gphi[72] = 625.0*y*x*x+2500.0*y*y*x-125.0*ONETHIRD*x*x-2125.0*ONETHIRD*y*x-1562.5*y*y*x*x-1562.5*z*y*y*x+125.0*ONETHIRD*x+625.0*z*y*x-125.0*ONETHIRD*z*x-6250.0*ONETHIRD*y*y*y*x;
  gphi[73] = -312.5*y*x*x-625.0*0.25*y*y*x+125.0*ONESIXTH*x*x+62.5*y*x+3125.0*0.25*y*y*x*x-25.0*ONESIXTH*x;
  gphi[74] = 1875.0*0.25*z*y*y-25.0*0.25+1875.0*0.25*y*y*x-1375.0/12.0*y*x+25.0*0.25*x-3125.0*ONESIXTH*y*y*y*x-1375.0/12.0*z*y-3125.0*ONESIXTH*z*y*y*y+1525.0/12.0*y-5125.0/8.0*y*y+6875.0*ONESIXTH*y*y*y-15625.0/24.0*y*y*y*y+25.0*0.25*z;
  gphi[75] = -1875.0*0.25*y*y*x+1375.0/12.0*y*x-25.0*0.25*x+3125.0*ONESIXTH*y*y*y*x;
  gphi[76] = 1.0-125.0*ONESIXTH*y+875.0/8.0*y*y-625.0*ONETHIRD*y*y*y+3125.0/24.0*y*y*y*y;
  gphi[77] = 3125.0*ONESIXTH*z*z*z*z+3125.0*ONESIXTH*z*y*y*y+3125.0*z*z*y*x+1562.5*z*z*y*y+1562.5*z*z*z*y-2187.5*z*y*x-4375.0*0.25*z*y*y-2187.5*z*z*y+8875.0/12.0*z*y+1562.5*z*y*x*x+1562.5*z*y*y*x-4375.0*0.25*z*z*z+3125.0*ONESIXTH*z*x*x*x+1562.5*z*z*x*x+1562.5*z*z*z*x-4375.0*0.25*z*x*x-2187.5*z*z*x+8875.0/12.0*z*z-1925.0/12.0*z+8875.0/12.0*z*x;
  gphi[78] = -1562.5*z*x*x*x-3125.0*z*y*x*x-1562.5*z*y*y*x-3125.0*z*z*x*x-3125.0*z*z*y*x-1562.5*z*z*z*x+2500.0*z*x*x+2500.0*z*y*x+2500.0*z*z*x-5875.0*ONESIXTH*z*x;
  gphi[79] = 1562.5*z*x*x*x+1562.5*z*y*x*x+1562.5*z*z*x*x-6875.0*0.25*z*x*x-312.5*z*y*x-312.5*z*z*x+1125.0*0.25*z*x;
  gphi[80] = -3125.0*ONESIXTH*z*x*x*x+312.5*z*x*x-125.0*ONETHIRD*z*x;
  gphi[81] = 0.0;
  gphi[82] = -3125.0*ONESIXTH*z*x*x*x-3125.0*z*y*x*x-4687.5*z*y*y*x-1562.5*z*z*x*x-6250.0*z*z*y*x-1562.5*z*z*z*x+1250.0*z*x*x+5000.0*z*y*x+2500.0*z*z*x-5875.0*ONESIXTH*z*x-5875.0*ONETHIRD*z*y-5875.0*ONESIXTH*z*z-4687.5*z*z*y*y-3125.0*ONESIXTH*z*z*z*z+5000.0*z*z*y+3750.0*z*y*y+1250.0*z*z*z+250.0*z-6250.0*ONETHIRD*z*y*y*y-3125.0*z*z*z*y;
  gphi[83] = 1562.5*z*x*x*x+6250.0*z*y*x*x+4687.5*z*y*y*x+3125.0*z*z*x*x+6250.0*z*z*y*x+1562.5*z*z*z*x-2812.5*z*x*x-5625.0*z*y*x-2812.5*z*z*x+1250.0*z*x;
  gphi[84] = -1562.5*z*x*x*x-3125.0*z*y*x*x-1562.5*z*z*x*x+1875.0*z*x*x+625.0*z*y*x+312.5*z*z*x-312.5*z*x;
  gphi[85] = 3125.0*ONESIXTH*z*x*x*x-312.5*z*x*x+125.0*ONETHIRD*z*x;
  gphi[86] = 1562.5*z*y*x*x+4687.5*z*y*y*x+3125.0*z*z*y*x-625.0*0.25*z*x*x-3437.5*z*y*x-312.5*z*z*x+1125.0*0.25*z*x+1812.5*z*y+1125.0*0.25*z*z+4687.5*z*z*y*y-3437.5*z*z*y-4687.5*z*y*y-625.0*0.25*z*z*z-125.0*z+3125.0*z*y*y*y+1562.5*z*z*z*y;
  gphi[87] = -3125.0*z*y*x*x-4687.5*z*y*y*x-3125.0*z*z*y*x+312.5*z*x*x+3750.0*z*y*x+312.5*z*z*x-312.5*z*x;
  gphi[88] = 1562.5*z*y*x*x-625.0*0.25*z*x*x-312.5*z*y*x+125.0*0.25*z*x;
  gphi[89] = -1562.5*z*y*y*x+625.0*z*y*x-125.0*ONETHIRD*z*x-2125.0*ONETHIRD*z*y-125.0*ONETHIRD*z*z-1562.5*z*z*y*y+625.0*z*z*y+2500.0*z*y*y+125.0*ONETHIRD*z-6250.0*ONETHIRD*z*y*y*y;
  gphi[90] = 1562.5*z*y*y*x-625.0*z*y*x+125.0*ONETHIRD*z*x;
  gphi[91] = 1375.0/12.0*z*y-1875.0*0.25*z*y*y-25.0*0.25*z+3125.0*ONESIXTH*z*y*y*y;
  gphi[92] = -3125.0*0.25*z*z*z*z-1562.5*z*z*y*x-3125.0*0.25*z*z*y*y-1562.5*z*z*z*y+312.5*z*y*x+625.0*0.25*z*y*y+1562.5*z*z*y-250.0*z*y+5625.0*0.25*z*z*z-3125.0*0.25*z*z*x*x-1562.5*z*z*z*x+625.0*0.25*z*x*x+1562.5*z*z*x-8875.0/12.0*z*z+1175.0/12.0*z-250.0*z*x;
  gphi[93] = -312.5*z*x*x-312.5*z*y*x+1125.0*0.25*z*x+1562.5*z*z*z*x+1562.5*z*z*y*x+1562.5*z*z*x*x-6875.0*0.25*z*z*x;
  gphi[94] = 625.0*0.25*z*x*x-125.0*0.25*z*x-3125.0*0.25*z*z*x*x+625.0*0.25*z*z*x;
  gphi[95] = 0.0;
  gphi[96] = -625.0*0.25*z*x*x-625.0*z*y*x+1125.0*0.25*z*x+1562.5*z*z*z*x+3125.0*z*z*y*x+3125.0*0.25*z*z*x*x-6875.0*0.25*z*z*x+562.5*z*y+3625.0*0.25*z*z+9375.0*0.25*z*z*y*y-3437.5*z*z*y-1875.0*0.25*z*y*y+3125.0*z*z*z*y+3125.0*0.25*z*z*z*z-125.0*z-1562.5*z*z*z;
  gphi[97] = 312.5*z*x*x+625.0*z*y*x-312.5*z*x-1562.5*z*z*z*x-3125.0*z*z*y*x-1562.5*z*z*x*x+1875.0*z*z*x;
  gphi[98] = -625.0*0.25*z*x*x+125.0*0.25*z*x+3125.0*0.25*z*z*x*x-625.0*0.25*z*z*x;
  gphi[99] = 312.5*z*y*x-125.0*0.25*z*x-1562.5*z*z*y*x+625.0*0.25*z*z*x-375.0*z*y-187.5*z*z-9375.0*0.25*z*z*y*y+2187.5*z*z*y+1875.0*0.25*z*y*y-1562.5*z*z*z*y+125.0*0.25*z+625.0*0.25*z*z*z;
  gphi[100] = -312.5*z*y*x+125.0*0.25*z*x+1562.5*z*z*y*x-625.0*0.25*z*z*x;
  gphi[101] = 62.5*z*y+125.0*ONESIXTH*z*z+3125.0*0.25*z*z*y*y-312.5*z*z*y-625.0*0.25*z*y*y-25.0*ONESIXTH*z;
  gphi[102] = 125.0*ONETHIRD*z*y+3125.0*ONESIXTH*z*z*z*y-312.5*z*z*y-312.5*z*z*x-3125.0*0.25*z*z*z+125.0*ONETHIRD*z*x+3875.0/12.0*z*z+3125.0*ONESIXTH*z*z*z*x-37.5*z+3125.0*ONESIXTH*z*z*z*z;
  gphi[103] = -125.0*ONETHIRD*z*x-3125.0*ONESIXTH*z*z*z*x+312.5*z*z*x;
  gphi[104] = 0.0;
  gphi[105] = -3125.0*ONESIXTH*z*z*z*z-125.0*ONETHIRD*z*x-3125.0*ONESIXTH*z*z*z*x+312.5*z*z*x-250.0*ONETHIRD*z*y-2125.0*ONESIXTH*z*z+625.0*z*z*y-3125.0*ONETHIRD*z*z*z*y+125.0*ONETHIRD*z+2500.0*ONETHIRD*z*z*z;
  gphi[106] = 125.0*ONETHIRD*z*x+3125.0*ONESIXTH*z*z*z*x-312.5*z*z*x;
  gphi[107] = 125.0*ONETHIRD*z*y+125.0*0.25*z*z-312.5*z*z*y+3125.0*ONESIXTH*z*z*z*y-25.0*ONESIXTH*z-625.0/12.0*z*z*z;
  gphi[108] = -3125.0/24.0*z*z*z*z+625.0*0.25*z*z*z-1375.0/24.0*z*z+25.0*0.25*z;
  gphi[109] = 0.0;
  gphi[110] = 3125.0/24.0*z*z*z*z-625.0*0.25*z*z*z+1375.0/24.0*z*z-25.0*0.25*z;
  gphi[111] = 0.0;
  gphi[112] = 375.0*0.25*z+375.0*0.25*x+375.0*0.25*y-1562.5*z*y*x*x-1562.5*z*y*y*x-1562.5*z*z*y*x+1875.0*z*y*x-137.0/12.0-2125.0/8.0*x*x+937.5*z*z*y-3125.0/24.0*x*x*x*x-3125.0*ONESIXTH*z*z*z*x-2125.0/8.0*y*y+312.5*y*y*y+312.5*x*x*x-3125.0*0.25*y*y*x*x-3125.0*ONESIXTH*y*x*x*x-3125.0*ONESIXTH*y*y*y*x-3125.0*ONESIXTH*z*x*x*x-3125.0*ONESIXTH*z*y*y*y-3125.0*0.25*z*z*x*x+937.5*z*z*x-2125.0*0.25*z*x-3125.0*0.25*z*z*y*y-2125.0/8.0*z*z+312.5*z*z*z-3125.0/24.0*y*y*y*y-2125.0*0.25*z*y-3125.0/24.0*z*z*z*z+937.5*y*x*x+937.5*z*y*y+937.5*z*x*x-3125.0*ONESIXTH*z*z*z*y+937.5*y*y*x-2125.0*0.25*y*x;
  gphi[113] = 1562.5*z*y*y*x+1562.5*z*z*x*x+1562.5*z*z*y*x+3125.0*ONESIXTH*z*z*z*x-2187.5*z*x*x-2187.5*z*y*x-4375.0*0.25*z*z*x+8875.0/12.0*z*x+1562.5*z*x*x*x+3125.0*z*y*x*x+3125.0*ONESIXTH*y*y*y*x+1562.5*y*y*x*x+1562.5*y*x*x*x+3125.0*ONESIXTH*x*x*x*x-2187.5*y*x*x-4375.0*0.25*x*x*x-4375.0*0.25*y*y*x+8875.0/12.0*x*x+8875.0/12.0*y*x-1925.0/12.0*x;
  gphi[114] = -3125.0*0.25*z*z*x*x+1562.5*z*x*x+312.5*z*y*x+625.0*0.25*z*z*x-250.0*z*x-1562.5*z*x*x*x-1562.5*z*y*x*x-3125.0*0.25*y*y*x*x-1562.5*y*x*x*x-3125.0*0.25*x*x*x*x+1562.5*y*x*x+5625.0*0.25*x*x*x+625.0*0.25*y*y*x-8875.0/12.0*x*x-250.0*y*x+1175.0/12.0*x;
  gphi[115] = -3125.0*0.25*x*x*x-312.5*y*x*x+3875.0/12.0*x*x+125.0*ONETHIRD*y*x+3125.0*ONESIXTH*z*x*x*x+3125.0*ONESIXTH*x*x*x*x+3125.0*ONESIXTH*y*x*x*x-37.5*x-312.5*z*x*x+125.0*ONETHIRD*z*x;
  gphi[116] = 625.0*0.25*x*x*x-1375.0/24.0*x*x-3125.0/24.0*x*x*x*x+25.0*0.25*x;
  gphi[117] = 0.0;
  gphi[118] = 1562.5*z*y*y*y+1562.5*z*z*y*x+1562.5*z*z*y*y+3125.0*ONESIXTH*z*z*z*y-2187.5*z*y*x-2187.5*z*y*y-4375.0*0.25*z*z*y+8875.0/12.0*z*y+1562.5*z*y*x*x+3125.0*z*y*y*x+3125.0*ONESIXTH*y*y*y*y+1562.5*y*y*y*x+1562.5*y*y*x*x+3125.0*ONESIXTH*y*x*x*x-2187.5*y*y*x-4375.0*0.25*y*x*x-4375.0*0.25*y*y*y+8875.0/12.0*y*x+8875.0/12.0*y*y-1925.0/12.0*y;
  gphi[119] = -1562.5*y*x*x*x-3125.0*y*y*x*x-3125.0*z*y*x*x-3125.0*z*y*y*x+2500.0*y*x*x+2500.0*y*y*x+2500.0*z*y*x-5875.0*ONESIXTH*y*x-1562.5*y*y*y*x-1562.5*z*z*y*x;
  gphi[120] = 1562.5*y*x*x*x+1562.5*y*y*x*x+1562.5*z*y*x*x-6875.0*0.25*y*x*x-312.5*y*y*x-312.5*z*y*x+1125.0*0.25*y*x;
  gphi[121] = -3125.0*ONESIXTH*y*x*x*x+312.5*y*x*x-125.0*ONETHIRD*y*x;
  gphi[122] = 0.0;
  gphi[123] = -1562.5*z*y*y*y-3125.0*0.25*z*z*y*y+312.5*z*y*x+1562.5*z*y*y+625.0*0.25*z*z*y-250.0*z*y-1562.5*z*y*y*x-3125.0*0.25*y*y*y*y-1562.5*y*y*y*x-3125.0*0.25*y*y*x*x+1562.5*y*y*x+625.0*0.25*y*x*x+5625.0*0.25*y*y*y-250.0*y*x-8875.0/12.0*y*y+1175.0/12.0*y;
  gphi[124] = 1562.5*y*y*x*x+1562.5*y*y*y*x+1562.5*z*y*y*x-312.5*y*x*x-6875.0*0.25*y*y*x-312.5*z*y*x+1125.0*0.25*y*x;
  gphi[125] = -3125.0*0.25*y*y*x*x+625.0*0.25*y*x*x+625.0*0.25*y*y*x-125.0*0.25*y*x;
  gphi[126] = 0.0;
  gphi[127] = -312.5*y*y*x-3125.0*0.25*y*y*y+125.0*ONETHIRD*y*x+3875.0/12.0*y*y+3125.0*ONESIXTH*y*y*y*x+3125.0*ONESIXTH*z*y*y*y-37.5*y-312.5*z*y*y+125.0*ONETHIRD*z*y+3125.0*ONESIXTH*y*y*y*y;
  gphi[128] = -3125.0*ONESIXTH*y*y*y*x+312.5*y*y*x-125.0*ONETHIRD*y*x;
  gphi[129] = 0.0;
  gphi[130] = 625.0*0.25*y*y*y-1375.0/24.0*y*y+25.0*0.25*y-3125.0/24.0*y*y*y*y;
  gphi[131] = 0.0;
  gphi[132] = 0.0;
  gphi[133] = 25.0-1925.0*ONESIXTH*z-1925.0/12.0*x-1925.0/12.0*y+3125.0*z*y*x*x+3125.0*z*y*y*x+4687.5*z*z*y*x-4375.0*z*y*x+8875.0/24.0*x*x-13125.0*0.25*z*z*y+3125.0/24.0*x*x*x*x+6250.0*ONETHIRD*z*z*z*x+8875.0/24.0*y*y-4375.0/12.0*y*y*y-4375.0/12.0*x*x*x+3125.0*0.25*y*y*x*x+3125.0*ONESIXTH*y*x*x*x+3125.0*ONESIXTH*y*y*y*x+3125.0*ONETHIRD*z*x*x*x+3125.0*ONETHIRD*z*y*y*y+9375.0*0.25*z*z*x*x-13125.0*0.25*z*z*x+8875.0*ONESIXTH*z*x+9375.0*0.25*z*z*y*y+8875.0/8.0*z*z-4375.0*ONETHIRD*z*z*z+3125.0/24.0*y*y*y*y+8875.0*ONESIXTH*z*y+15625.0/24.0*z*z*z*z-4375.0*0.25*y*x*x-2187.5*z*y*y-2187.5*z*x*x+6250.0*ONETHIRD*z*z*z*y-4375.0*0.25*y*y*x+8875.0/12.0*y*x;
  gphi[134] = -1562.5*y*x*x*x-1562.5*y*y*x*x-3125.0*ONESIXTH*y*y*y*x-6250.0*z*y*x*x-3125.0*z*y*y*x-4687.5*z*z*y*x+2500.0*y*x*x+1250.0*y*y*x+5000.0*z*y*x-5875.0*ONESIXTH*y*x-3125.0*ONESIXTH*x*x*x*x-4687.5*z*z*x*x-3125.0*z*x*x*x-6250.0*ONETHIRD*z*z*z*x+5000.0*z*x*x+1250.0*x*x*x+3750.0*z*z*x-5875.0*ONETHIRD*z*x-5875.0*ONESIXTH*x*x+250.0*x;
  gphi[135] = 1562.5*y*x*x*x+3125.0*0.25*y*y*x*x+3125.0*z*y*x*x-6875.0*0.25*y*x*x-625.0*0.25*y*y*x-625.0*z*y*x+1125.0*0.25*y*x+3125.0*0.25*x*x*x*x+9375.0*0.25*z*z*x*x+3125.0*z*x*x*x-3437.5*z*x*x-1562.5*x*x*x-1875.0*0.25*z*z*x+562.5*z*x+3625.0*0.25*x*x-125.0*x;
  gphi[136] = -3125.0*ONESIXTH*y*x*x*x+312.5*y*x*x-125.0*ONETHIRD*y*x-3125.0*ONESIXTH*x*x*x*x-3125.0*ONETHIRD*z*x*x*x+625.0*z*x*x+2500.0*ONETHIRD*x*x*x-250.0*ONETHIRD*z*x-2125.0*ONESIXTH*x*x+125.0*ONETHIRD*x;
  gphi[137] = -625.0*0.25*x*x*x+1375.0/24.0*x*x+3125.0/24.0*x*x*x*x-25.0*0.25*x;
  gphi[138] = -3125.0*ONESIXTH*y*x*x*x-1562.5*y*y*x*x-1562.5*y*y*y*x-3125.0*z*y*x*x-6250.0*z*y*y*x-4687.5*z*z*y*x+1250.0*y*x*x+2500.0*y*y*x+5000.0*z*y*x-5875.0*ONESIXTH*y*x-5875.0*ONESIXTH*y*y-5875.0*ONETHIRD*z*y-3125.0*z*y*y*y-6250.0*ONETHIRD*z*z*z*y+5000.0*z*y*y+1250.0*y*y*y+3750.0*z*z*y+250.0*y-3125.0*ONESIXTH*y*y*y*y-4687.5*z*z*y*y;
  gphi[139] = 1562.5*y*x*x*x+3125.0*y*y*x*x+1562.5*y*y*y*x+6250.0*z*y*x*x+6250.0*z*y*y*x+4687.5*z*z*y*x-2812.5*y*x*x-2812.5*y*y*x-5625.0*z*y*x+1250.0*y*x;
  gphi[140] = -1562.5*y*x*x*x-1562.5*y*y*x*x-3125.0*z*y*x*x+1875.0*y*x*x+312.5*y*y*x+625.0*z*y*x-312.5*y*x;
  gphi[141] = 3125.0*ONESIXTH*y*x*x*x-312.5*y*x*x+125.0*ONETHIRD*y*x;
  gphi[142] = 3125.0*0.25*y*y*x*x+1562.5*y*y*y*x+3125.0*z*y*y*x-625.0*0.25*y*x*x-6875.0*0.25*y*y*x-625.0*z*y*x+1125.0*0.25*y*x+3625.0*0.25*y*y+562.5*z*y+3125.0*z*y*y*y-3437.5*z*y*y-1562.5*y*y*y-1875.0*0.25*z*z*y-125.0*y+3125.0*0.25*y*y*y*y+9375.0*0.25*z*z*y*y;
  gphi[143] = -1562.5*y*y*x*x-1562.5*y*y*y*x-3125.0*z*y*y*x+312.5*y*x*x+1875.0*y*y*x+625.0*z*y*x-312.5*y*x;
  gphi[144] = 3125.0*0.25*y*y*x*x-625.0*0.25*y*x*x-625.0*0.25*y*y*x+125.0*0.25*y*x;
  gphi[145] = -3125.0*ONESIXTH*y*y*y*x+312.5*y*y*x-125.0*ONETHIRD*y*x-2125.0*ONESIXTH*y*y-250.0*ONETHIRD*z*y-3125.0*ONETHIRD*z*y*y*y+625.0*z*y*y+2500.0*ONETHIRD*y*y*y+125.0*ONETHIRD*y-3125.0*ONESIXTH*y*y*y*y;
  gphi[146] = 3125.0*ONESIXTH*y*y*y*x-312.5*y*y*x+125.0*ONETHIRD*y*x;
  gphi[147] = -625.0*0.25*y*y*y+1375.0/24.0*y*y-25.0*0.25*y+3125.0/24.0*y*y*y*y;
  gphi[148] = -25.0+2675.0*ONESIXTH*z+1175.0/12.0*x+1175.0/12.0*y-1562.5*z*y*x*x-1562.5*z*y*y*x-4687.5*z*z*y*x+3125.0*z*y*x-125.0*x*x+16875.0*0.25*z*z*y-3125.0*z*z*z*x-125.0*y*y+625.0/12.0*y*y*y+625.0/12.0*x*x*x-3125.0*ONESIXTH*z*x*x*x-3125.0*ONESIXTH*z*y*y*y-9375.0*0.25*z*z*x*x+16875.0*0.25*z*z*x-8875.0*ONESIXTH*z*x-9375.0*0.25*z*z*y*y-7375.0*0.25*z*z+8125.0*ONETHIRD*z*z*z-8875.0*ONESIXTH*z*y-15625.0/12.0*z*z*z*z+625.0*0.25*y*x*x+1562.5*z*y*y+1562.5*z*x*x-3125.0*z*z*z*y+625.0*0.25*y*y*x-250.0*y*x;
  gphi[149] = -312.5*y*x*x-625.0*0.25*y*y*x+1125.0*0.25*y*x+4687.5*z*z*y*x+1562.5*z*y*y*x+3125.0*z*y*x*x-3437.5*z*y*x+1562.5*z*x*x*x-625.0*0.25*x*x*x+4687.5*z*z*x*x+3125.0*z*z*z*x-125.0*x-3437.5*z*x*x-4687.5*z*z*x+1125.0*0.25*x*x+1812.5*z*x;
  gphi[150] = 625.0*0.25*y*x*x-125.0*0.25*y*x-1562.5*z*y*x*x+312.5*z*y*x-1562.5*z*x*x*x+625.0*0.25*x*x*x-9375.0*0.25*z*z*x*x+125.0*0.25*x+2187.5*z*x*x+1875.0*0.25*z*z*x-187.5*x*x-375.0*z*x;
  gphi[151] = 3125.0*ONESIXTH*z*x*x*x-625.0/12.0*x*x*x-25.0*ONESIXTH*x-312.5*z*x*x+125.0*0.25*x*x+125.0*ONETHIRD*z*x;
  gphi[152] = -625.0*0.25*y*x*x-312.5*y*y*x+1125.0*0.25*y*x+4687.5*z*z*y*x+3125.0*z*y*y*x+1562.5*z*y*x*x-3437.5*z*y*x+1125.0*0.25*y*y+1812.5*z*y+1562.5*z*y*y*y-3437.5*z*y*y-625.0*0.25*y*y*y+4687.5*z*z*y*y+3125.0*z*z*z*y-125.0*y-4687.5*z*z*y;
  gphi[153] = 312.5*y*x*x+312.5*y*y*x-312.5*y*x-4687.5*z*z*y*x-3125.0*z*y*y*x-3125.0*z*y*x*x+3750.0*z*y*x;
  gphi[154] = -625.0*0.25*y*x*x+125.0*0.25*y*x+1562.5*z*y*x*x-312.5*z*y*x;
  gphi[155] = 625.0*0.25*y*y*x-125.0*0.25*y*x-1562.5*z*y*y*x+312.5*z*y*x-187.5*y*y-375.0*z*y-1562.5*z*y*y*y+2187.5*z*y*y+625.0*0.25*y*y*y-9375.0*0.25*z*z*y*y+125.0*0.25*y+1875.0*0.25*z*z*y;
  gphi[156] = -625.0*0.25*y*y*x+125.0*0.25*y*x+1562.5*z*y*y*x-312.5*z*y*x;
  gphi[157] = 125.0*0.25*y*y+125.0*ONETHIRD*z*y+3125.0*ONESIXTH*z*y*y*y-312.5*z*y*y-625.0/12.0*y*y*y-25.0*ONESIXTH*y;
  gphi[158] = 50.0*ONETHIRD+6250.0*ONETHIRD*z*z*z*y+125.0*ONETHIRD*y*x+1562.5*z*z*y*x-625.0*z*y*x-312.5*z*x*x-9375.0*0.25*z*z*x+125.0*ONESIXTH*y*y+3875.0*ONESIXTH*z*y+125.0*ONESIXTH*x*x+3875.0*ONESIXTH*z*x+3125.0*0.25*z*z*x*x-312.5*z*y*y-37.5*x-325.0*z+6125.0*0.25*z*z-2500.0*z*z*z+6250.0*ONETHIRD*z*z*z*x+15625.0/12.0*z*z*z*z+3125.0*0.25*z*z*y*y-37.5*y-9375.0*0.25*z*z*y;
  gphi[159] = -125.0*ONETHIRD*y*x-1562.5*z*z*y*x+625.0*z*y*x+625.0*z*x*x+2500.0*z*z*x-125.0*ONETHIRD*x*x-2125.0*ONETHIRD*z*x-1562.5*z*z*x*x+125.0*ONETHIRD*x-6250.0*ONETHIRD*z*z*z*x;
  gphi[160] = -312.5*z*x*x-625.0*0.25*z*z*x+125.0*ONESIXTH*x*x+62.5*z*x+3125.0*0.25*z*z*x*x-25.0*ONESIXTH*x;
  gphi[161] = -6250.0*ONETHIRD*z*z*z*y-125.0*ONETHIRD*y*x-1562.5*z*z*y*x+625.0*z*y*x-125.0*ONETHIRD*y*y-2125.0*ONETHIRD*z*y+625.0*z*y*y-1562.5*z*z*y*y+125.0*ONETHIRD*y+2500.0*z*z*y;
  gphi[162] = 125.0*ONETHIRD*y*x+1562.5*z*z*y*x-625.0*z*y*x;
  gphi[163] = 125.0*ONESIXTH*y*y+62.5*z*y-312.5*z*y*y+3125.0*0.25*z*z*y*y-25.0*ONESIXTH*y-625.0*0.25*z*z*y;
  gphi[164] = -25.0*0.25-3125.0*ONESIXTH*z*z*z*x-3125.0*ONESIXTH*z*z*z*y-1375.0/12.0*z*y+1875.0*0.25*z*z*x-1375.0/12.0*z*x+25.0*0.25*x+1525.0/12.0*z-5125.0/8.0*z*z+25.0*0.25*y+6875.0*ONESIXTH*z*z*z-15625.0/24.0*z*z*z*z+1875.0*0.25*z*z*y;
  gphi[165] = 3125.0*ONESIXTH*z*z*z*x-1875.0*0.25*z*z*x+1375.0/12.0*z*x-25.0*0.25*x;
  gphi[166] = 3125.0*ONESIXTH*z*z*z*y+1375.0/12.0*z*y-25.0*0.25*y-1875.0*0.25*z*z*y;
  gphi[167] = 1.0-125.0*ONESIXTH*z+875.0/8.0*z*z-625.0*ONETHIRD*z*z*z+3125.0/24.0*z*z*z*z;
    return 0;
#endif    
  default:
    return -1;
  }
}



/******************************************************************/
//   FUNCTION Definition: PXGradientsLagrange3d
template <typename DT> ELVIS_DEVICE int
PXGradientsLagrange3d_Solution(const int porder, const DT * RESTRICT xref, DT * RESTRICT gphi)
{
  DT x, y, z;
  
  x = xref[0]; 
  y = xref[1]; 
  z = xref[2];
  
  switch (porder){
#if SOLN_USE_P0    
  case 0:
    gphi[0] = 0.0;
    gphi[1] = 0.0;
    gphi[2] = 0.0;
    return 0;
#endif
#if SOLN_USE_P1
  case 1:
    gphi[0] = -1.0;
    gphi[1] = 1.0;
    gphi[2] = 0.0;
    gphi[3] = 0.0;
    gphi[4] = -1.0;
    gphi[5] = 0.0;
    gphi[6] = 1.0;
    gphi[7] = 0.0;
    gphi[8] = -1.0;
    gphi[9] = 0.0;
    gphi[10] = 0.0;
    gphi[11] = 1.0;
    return 0;
#endif
#if SOLN_USE_P2
  case 2:
    gphi[0] = -3.0+4.0*z+4.0*y+4.0*x;
    gphi[1] = 4.0-4.0*z-4.0*y-8.0*x;
    gphi[2] = -1.0+4.0*x;
    gphi[3] = -4.0*y;
    gphi[4] = 4.0*y;
    gphi[5] = 0.0;
    gphi[6] = -4.0*z;
    gphi[7] = 4.0*z;
    gphi[8] = 0.0;
    gphi[9] = 0.0;
    gphi[10] = -3.0+4.0*z+4.0*y+4.0*x;
    gphi[11] = -4.0*x;
    gphi[12] = 0.0;
    gphi[13] = 4.0-4.0*z-8.0*y-4.0*x;
    gphi[14] = 4.0*x;
    gphi[15] = -1.0+4.0*y;
    gphi[16] = -4.0*z;
    gphi[17] = 0.0;
    gphi[18] = 4.0*z;
    gphi[19] = 0.0;
    gphi[20] = -3.0+4.0*z+4.0*y+4.0*x;
    gphi[21] = -4.0*x;
    gphi[22] = 0.0;
    gphi[23] = -4.0*y;
    gphi[24] = 0.0;
    gphi[25] = 0.0;
    gphi[26] = 4.0-8.0*z-4.0*y-4.0*x;
    gphi[27] = 4.0*x;
    gphi[28] = 4.0*y;
    gphi[29] = -1.0+4.0*z;
    
    return 0;
#endif
#if SOLN_USE_P3    
  case 3:
    gphi[0] = -5.5+18.0*z+18.0*y+18.0*x-13.5*z*z-27.0*z*y-27.0*z*x-13.5*y*y-27.0*y*x-13.5*x*x;
    gphi[1] = 9.0-22.5*z-22.5*y-45.0*x+13.5*z*z+27.0*z*y+54.0*z*x+13.5*y*y+54.0*y*x+40.5*x*x;
    gphi[2] = -4.5+4.5*z+4.5*y+36.0*x-27.0*z*x-27.0*y*x-40.5*x*x;
    gphi[3] = 1.0-9.0*x+13.5*x*x;
    gphi[4] = -22.5*y+27.0*z*y+27.0*y*y+27.0*y*x;
    gphi[5] = 27.0*y-27.0*z*y-27.0*y*y-54.0*y*x;
    gphi[6] = -4.5*y+27.0*y*x;
    gphi[7] = 4.5*y-13.5*y*y;
    gphi[8] = -4.5*y+13.5*y*y;
    gphi[9] = 0.0;
    gphi[10] = -22.5*z+27.0*z*z+27.0*z*y+27.0*z*x;
    gphi[11] = 27.0*z-27.0*z*z-27.0*z*y-54.0*z*x;
    gphi[12] = -4.5*z+27.0*z*x;
    gphi[13] = -27.0*z*y;
    gphi[14] = 27.0*z*y;
    gphi[15] = 0.0;
    gphi[16] = 4.5*z-13.5*z*z;
    gphi[17] = -4.5*z+13.5*z*z;
    gphi[18] = 0.0;
    gphi[19] = 0.0;
    gphi[20] = -5.5+18.0*z+18.0*y+18.0*x-13.5*z*z-27.0*z*y-27.0*z*x-13.5*y*y-27.0*y*x-13.5*x*x;
    gphi[21] = -22.5*x+27.0*z*x+27.0*y*x+27.0*x*x;
    gphi[22] = 4.5*x-13.5*x*x;
    gphi[23] = 0.0;
    gphi[24] = 9.0-22.5*z-45.0*y-22.5*x+13.5*z*z+54.0*z*y+27.0*z*x+40.5*y*y+54.0*y*x+13.5*x*x;
    gphi[25] = 27.0*x-27.0*z*x-54.0*y*x-27.0*x*x;
    gphi[26] = -4.5*x+13.5*x*x;
    gphi[27] = -4.5+4.5*z+36.0*y+4.5*x-27.0*z*y-40.5*y*y-27.0*y*x;
    gphi[28] = -4.5*x+27.0*y*x;
    gphi[29] = 1.0-9.0*y+13.5*y*y;
    gphi[30] = -22.5*z+27.0*z*z+27.0*z*y+27.0*z*x;
    gphi[31] = -27.0*z*x;
    gphi[32] = 0.0;
    gphi[33] = 27.0*z-27.0*z*z-54.0*z*y-27.0*z*x;
    gphi[34] = 27.0*z*x;
    gphi[35] = -4.5*z+27.0*z*y;
    gphi[36] = 4.5*z-13.5*z*z;
    gphi[37] = 0.0;
    gphi[38] = -4.5*z+13.5*z*z;
    gphi[39] = 0.0;
    gphi[40] = -5.5+18.0*z+18.0*y+18.0*x-13.5*z*z-27.0*z*y-27.0*z*x-13.5*y*y-27.0*y*x-13.5*x*x;
    gphi[41] = -22.5*x+27.0*z*x+27.0*y*x+27.0*x*x;
    gphi[42] = 4.5*x-13.5*x*x;
    gphi[43] = 0.0;
    gphi[44] = -22.5*y+27.0*z*y+27.0*y*y+27.0*y*x;
    gphi[45] = -27.0*y*x;
    gphi[46] = 0.0;
    gphi[47] = 4.5*y-13.5*y*y;
    gphi[48] = 0.0;
    gphi[49] = 0.0;
    gphi[50] = 9.0-45.0*z-22.5*y-22.5*x+40.5*z*z+54.0*z*y+54.0*z*x+13.5*y*y+27.0*y*x+13.5*x*x;
    gphi[51] = 27.0*x-54.0*z*x-27.0*y*x-27.0*x*x;
    gphi[52] = -4.5*x+13.5*x*x;
    gphi[53] = 27.0*y-54.0*z*y-27.0*y*y-27.0*y*x;
    gphi[54] = 27.0*y*x;
    gphi[55] = -4.5*y+13.5*y*y;
    gphi[56] = -4.5+36.0*z+4.5*y+4.5*x-40.5*z*z-27.0*z*y-27.0*z*x;
    gphi[57] = -4.5*x+27.0*z*x;
    gphi[58] = -4.5*y+27.0*z*y;
    gphi[59] = 1.0-9.0*z+13.5*z*z;
    return 0;
#endif
#if SOLN_USE_P4    
  case 4:
      gphi[0] = 128.0*z*y*y-25.0*ONETHIRD+256.0*z*y*x+128.0*z*z*y-160.0*z*y+140.0*ONETHIRD*x+128.0*ONETHIRD*y*y*y+128.0*y*y*x+128.0*y*x*x+128.0*z*z*x+128.0*z*x*x+128.0*ONETHIRD*z*z*z+140.0*ONETHIRD*y-80.0*z*z-80.0*y*y-160.0*y*x+128.0*ONETHIRD*x*x*x-160.0*z*x+140.0*ONETHIRD*z-80.0*x*x;
  gphi[1] = -128.0*z*y*y-512.0*z*y*x+16.0-128.0*z*z*y+192.0*z*y+384.0*y*x-416.0*ONETHIRD*x+96.0*y*y+384.0*z*x+96.0*z*z-128.0*ONETHIRD*z*z*z-256.0*z*z*x-384.0*z*x*x-128.0*ONETHIRD*y*y*y-256.0*y*y*x-384.0*y*x*x-208.0*ONETHIRD*y-208.0*ONETHIRD*z+288.0*x*x-512.0*ONETHIRD*x*x*x;
  gphi[2] = 256.0*z*y*x-12.0-32.0*z*y-288.0*y*x+152.0*x-16.0*y*y-288.0*z*x-16.0*z*z+128.0*z*z*x+384.0*z*x*x+128.0*y*y*x+384.0*y*x*x+28.0*y+28.0*z-384.0*x*x+256.0*x*x*x;
  gphi[3] = 16.0*ONETHIRD+64.0*y*x-224.0*ONETHIRD*x+64.0*z*x-128.0*z*x*x-128.0*y*x*x-16.0*ONETHIRD*y-16.0*ONETHIRD*z+224.0*x*x-512.0*ONETHIRD*x*x*x;
  gphi[4] = -1.0+44.0*ONETHIRD*x-48.0*x*x+128.0*ONETHIRD*x*x*x;
  gphi[5] = -256.0*z*y*y-256.0*z*y*x-128.0*z*z*y+192.0*z*y+192.0*y*x+192.0*y*y-128.0*y*y*y-256.0*y*y*x-128.0*y*x*x-208.0*ONETHIRD*y;
  gphi[6] = 256.0*z*y*y+512.0*z*y*x+128.0*z*z*y-224.0*z*y-448.0*y*x-224.0*y*y+128.0*y*y*y+512.0*y*y*x+384.0*y*x*x+96.0*y;
  gphi[7] = -256.0*z*y*x+32.0*z*y+320.0*y*x+32.0*y*y-256.0*y*y*x-384.0*y*x*x-32.0*y;
  gphi[8] = -64.0*y*x+128.0*y*x*x+16.0*ONETHIRD*y;
  gphi[9] = 128.0*z*y*y-32.0*z*y-32.0*y*x-144.0*y*y+128.0*y*y*y+128.0*y*y*x+28.0*y;
  gphi[10] = -128.0*z*y*y+32.0*z*y+64.0*y*x+160.0*y*y-128.0*y*y*y-256.0*y*y*x-32.0*y;
  gphi[11] = -32.0*y*x-16.0*y*y+128.0*y*y*x+4.0*y;
  gphi[12] = 32.0*y*y-128.0*ONETHIRD*y*y*y-16.0*ONETHIRD*y;
  gphi[13] = -32.0*y*y+128.0*ONETHIRD*y*y*y+16.0*ONETHIRD*y;
  gphi[14] = 0.0;
  gphi[15] = -128.0*z*y*y-256.0*z*z*y-256.0*z*y*x+192.0*z*y-128.0*z*x*x-256.0*z*z*x-128.0*z*z*z+192.0*z*x+192.0*z*z-208.0*ONETHIRD*z;
  gphi[16] = 128.0*z*y*y+256.0*z*z*y+512.0*z*y*x-224.0*z*y+384.0*z*x*x+512.0*z*z*x+128.0*z*z*z-448.0*z*x-224.0*z*z+96.0*z;
  gphi[17] = -256.0*z*y*x+32.0*z*y-384.0*z*x*x-256.0*z*z*x+320.0*z*x+32.0*z*z-32.0*z;
  gphi[18] = 128.0*z*x*x-64.0*z*x+16.0*ONETHIRD*z;
  gphi[19] = 256.0*z*y*y+256.0*z*z*y+256.0*z*y*x-224.0*z*y;
  gphi[20] = -256.0*z*y*y-256.0*z*z*y-512.0*z*y*x+256.0*z*y;
  gphi[21] = 256.0*z*y*x-32.0*z*y;
  gphi[22] = -128.0*z*y*y+32.0*z*y;
  gphi[23] = 128.0*z*y*y-32.0*z*y;
  gphi[24] = 0.0;
  gphi[25] = 128.0*z*z*y-32.0*z*y+128.0*z*z*x+128.0*z*z*z-144.0*z*z-32.0*z*x+28.0*z;
  gphi[26] = -128.0*z*z*y+32.0*z*y-256.0*z*z*x-128.0*z*z*z+160.0*z*z+64.0*z*x-32.0*z;
  gphi[27] = 128.0*z*z*x-16.0*z*z-32.0*z*x+4.0*z;
  gphi[28] = -128.0*z*z*y+32.0*z*y;
  gphi[29] = 128.0*z*z*y-32.0*z*y;
  gphi[30] = 0.0;
  gphi[31] = -128.0*ONETHIRD*z*z*z+32.0*z*z-16.0*ONETHIRD*z;
  gphi[32] = 128.0*ONETHIRD*z*z*z-32.0*z*z+16.0*ONETHIRD*z;
  gphi[33] = 0.0;
  gphi[34] = 0.0;
  gphi[35] = 128.0*z*y*y-25.0*ONETHIRD+256.0*z*y*x+128.0*z*z*y-160.0*z*y+140.0*ONETHIRD*x+128.0*ONETHIRD*y*y*y+128.0*y*y*x+128.0*y*x*x+128.0*z*z*x+128.0*z*x*x+128.0*ONETHIRD*z*z*z+140.0*ONETHIRD*y-80.0*z*z-80.0*y*y-160.0*y*x+128.0*ONETHIRD*x*x*x-160.0*z*x+140.0*ONETHIRD*z-80.0*x*x;
  gphi[36] = -256.0*z*y*x-256.0*z*x*x-128.0*z*z*x+192.0*z*x+192.0*x*x+192.0*y*x-128.0*y*y*x-256.0*y*x*x-128.0*x*x*x-208.0*ONETHIRD*x;
  gphi[37] = 128.0*z*x*x-32.0*z*x-144.0*x*x-32.0*y*x+128.0*y*x*x+128.0*x*x*x+28.0*x;
  gphi[38] = 32.0*x*x-128.0*ONETHIRD*x*x*x-16.0*ONETHIRD*x;
  gphi[39] = 0.0;
  gphi[40] = -512.0*z*y*x-128.0*z*x*x+16.0-128.0*z*z*x+192.0*z*x+96.0*x*x+384.0*y*x+384.0*z*y-128.0*ONETHIRD*z*z*z-256.0*z*z*y-384.0*z*y*y-384.0*y*y*x-256.0*y*x*x-128.0*ONETHIRD*x*x*x+96.0*z*z-208.0*ONETHIRD*x-208.0*ONETHIRD*z-416.0*ONETHIRD*y+288.0*y*y-512.0*ONETHIRD*y*y*y;
  gphi[41] = 512.0*z*y*x+256.0*z*x*x+128.0*z*z*x-224.0*z*x-224.0*x*x-448.0*y*x+384.0*y*y*x+512.0*y*x*x+128.0*x*x*x+96.0*x;
  gphi[42] = -128.0*z*x*x+32.0*z*x+160.0*x*x+64.0*y*x-256.0*y*x*x-128.0*x*x*x-32.0*x;
  gphi[43] = -32.0*x*x+128.0*ONETHIRD*x*x*x+16.0*ONETHIRD*x;
  gphi[44] = 256.0*z*y*x-12.0-32.0*z*x-16.0*x*x-288.0*y*x-288.0*z*y+128.0*z*z*y+384.0*z*y*y+384.0*y*y*x+128.0*y*x*x-16.0*z*z+28.0*x+28.0*z+152.0*y-384.0*y*y+256.0*y*y*y;
  gphi[45] = -256.0*z*y*x+32.0*z*x+32.0*x*x+320.0*y*x-384.0*y*y*x-256.0*y*x*x-32.0*x;
  gphi[46] = -16.0*x*x-32.0*y*x+128.0*y*x*x+4.0*x;
  gphi[47] = 16.0*ONETHIRD+64.0*y*x+64.0*z*y-128.0*z*y*y-128.0*y*y*x-16.0*ONETHIRD*x-16.0*ONETHIRD*z-224.0*ONETHIRD*y+224.0*y*y-512.0*ONETHIRD*y*y*y;
  gphi[48] = -64.0*y*x+128.0*y*y*x+16.0*ONETHIRD*x;
  gphi[49] = -1.0+44.0*ONETHIRD*y-48.0*y*y+128.0*ONETHIRD*y*y*y;
  gphi[50] = -128.0*z*y*y-256.0*z*z*y-256.0*z*y*x+192.0*z*y-128.0*z*x*x-256.0*z*z*x-128.0*z*z*z+192.0*z*x+192.0*z*z-208.0*ONETHIRD*z;
  gphi[51] = 256.0*z*y*x+256.0*z*z*x+256.0*z*x*x-224.0*z*x;
  gphi[52] = -128.0*z*x*x+32.0*z*x;
  gphi[53] = 0.0;
  gphi[54] = 512.0*z*y*x+256.0*z*z*x+128.0*z*x*x-224.0*z*x-448.0*z*y+128.0*z*z*z+512.0*z*z*y+384.0*z*y*y-224.0*z*z+96.0*z;
  gphi[55] = -512.0*z*y*x-256.0*z*z*x-256.0*z*x*x+256.0*z*x;
  gphi[56] = 128.0*z*x*x-32.0*z*x;
  gphi[57] = -256.0*z*y*x+32.0*z*x+320.0*z*y-256.0*z*z*y-384.0*z*y*y+32.0*z*z-32.0*z;
  gphi[58] = 256.0*z*y*x-32.0*z*x;
  gphi[59] = -64.0*z*y+128.0*z*y*y+16.0*ONETHIRD*z;
  gphi[60] = 128.0*z*z*y-32.0*z*y+128.0*z*z*x+128.0*z*z*z-144.0*z*z-32.0*z*x+28.0*z;
  gphi[61] = -128.0*z*z*x+32.0*z*x;
  gphi[62] = 0.0;
  gphi[63] = -128.0*z*z*x+32.0*z*x+64.0*z*y-256.0*z*z*y-128.0*z*z*z+160.0*z*z-32.0*z;
  gphi[64] = 128.0*z*z*x-32.0*z*x;
  gphi[65] = -32.0*z*y+128.0*z*z*y-16.0*z*z+4.0*z;
  gphi[66] = -128.0*ONETHIRD*z*z*z+32.0*z*z-16.0*ONETHIRD*z;
  gphi[67] = 0.0;
  gphi[68] = 128.0*ONETHIRD*z*z*z-32.0*z*z+16.0*ONETHIRD*z;
  gphi[69] = 0.0;
  gphi[70] = 128.0*z*y*y-25.0*ONETHIRD+256.0*z*y*x+128.0*z*z*y-160.0*z*y+140.0*ONETHIRD*x+128.0*ONETHIRD*y*y*y+128.0*y*y*x+128.0*y*x*x+128.0*z*z*x+128.0*z*x*x+128.0*ONETHIRD*z*z*z+140.0*ONETHIRD*y-80.0*z*z-80.0*y*y-160.0*y*x+128.0*ONETHIRD*x*x*x-160.0*z*x+140.0*ONETHIRD*z-80.0*x*x;
  gphi[71] = -256.0*z*y*x-256.0*z*x*x-128.0*z*z*x+192.0*z*x+192.0*x*x+192.0*y*x-128.0*y*y*x-256.0*y*x*x-128.0*x*x*x-208.0*ONETHIRD*x;
  gphi[72] = 128.0*z*x*x-32.0*z*x-144.0*x*x-32.0*y*x+128.0*y*x*x+128.0*x*x*x+28.0*x;
  gphi[73] = 32.0*x*x-128.0*ONETHIRD*x*x*x-16.0*ONETHIRD*x;
  gphi[74] = 0.0;
  gphi[75] = -256.0*z*y*y-256.0*z*y*x-128.0*z*z*y+192.0*z*y+192.0*y*x+192.0*y*y-128.0*y*y*y-256.0*y*y*x-128.0*y*x*x-208.0*ONETHIRD*y;
  gphi[76] = 256.0*y*y*x+256.0*y*x*x+256.0*z*y*x-224.0*y*x;
  gphi[77] = -128.0*y*x*x+32.0*y*x;
  gphi[78] = 0.0;
  gphi[79] = 128.0*z*y*y-32.0*z*y-32.0*y*x-144.0*y*y+128.0*y*y*y+128.0*y*y*x+28.0*y;
  gphi[80] = -128.0*y*y*x+32.0*y*x;
  gphi[81] = 0.0;
  gphi[82] = 32.0*y*y-128.0*ONETHIRD*y*y*y-16.0*ONETHIRD*y;
  gphi[83] = 0.0;
  gphi[84] = 0.0;
  gphi[85] = -128.0*y*y*x-416.0*ONETHIRD*z+288.0*z*z-512.0*ONETHIRD*z*z*z+16.0-512.0*z*y*x-128.0*y*x*x+192.0*y*x-128.0*ONETHIRD*x*x*x-256.0*z*x*x-384.0*z*z*x+96.0*y*y-384.0*z*z*y-256.0*z*y*y-128.0*ONETHIRD*y*y*y+384.0*z*y+96.0*x*x-208.0*ONETHIRD*y+384.0*z*x-208.0*ONETHIRD*x;
  gphi[86] = 128.0*y*y*x+512.0*z*y*x+256.0*y*x*x-224.0*y*x+128.0*x*x*x+512.0*z*x*x+384.0*z*z*x-224.0*x*x-448.0*z*x+96.0*x;
  gphi[87] = -128.0*y*x*x+32.0*y*x-128.0*x*x*x-256.0*z*x*x+160.0*x*x+64.0*z*x-32.0*x;
  gphi[88] = -32.0*x*x+128.0*ONETHIRD*x*x*x+16.0*ONETHIRD*x;
  gphi[89] = 256.0*y*y*x+512.0*z*y*x+128.0*y*x*x-224.0*y*x-224.0*y*y+384.0*z*z*y+512.0*z*y*y+128.0*y*y*y-448.0*z*y+96.0*y;
  gphi[90] = -256.0*y*y*x-512.0*z*y*x-256.0*y*x*x+256.0*y*x;
  gphi[91] = 128.0*y*x*x-32.0*y*x;
  gphi[92] = -128.0*y*y*x+32.0*y*x+160.0*y*y-256.0*z*y*y-128.0*y*y*y+64.0*z*y-32.0*y;
  gphi[93] = 128.0*y*y*x-32.0*y*x;
  gphi[94] = -32.0*y*y+128.0*ONETHIRD*y*y*y+16.0*ONETHIRD*y;
  gphi[95] = -12.0+256.0*z*y*x-32.0*y*x-16.0*y*y+128.0*z*y*y+128.0*z*x*x+384.0*z*z*y+384.0*z*z*x-288.0*z*y-288.0*z*x-16.0*x*x+256.0*z*z*z+152.0*z-384.0*z*z+28.0*x+28.0*y;
  gphi[96] = -256.0*z*y*x+32.0*y*x-256.0*z*x*x-384.0*z*z*x+320.0*z*x+32.0*x*x-32.0*x;
  gphi[97] = 128.0*z*x*x-32.0*z*x-16.0*x*x+4.0*x;
  gphi[98] = -256.0*z*y*x+32.0*y*x+32.0*y*y-256.0*z*y*y-384.0*z*z*y+320.0*z*y-32.0*y;
  gphi[99] = 256.0*z*y*x-32.0*y*x;
  gphi[100] = -16.0*y*y+128.0*z*y*y-32.0*z*y+4.0*y;
  gphi[101] = 16.0*ONETHIRD-128.0*z*z*y-128.0*z*z*x+64.0*z*y+64.0*z*x-512.0*ONETHIRD*z*z*z-224.0*ONETHIRD*z+224.0*z*z-16.0*ONETHIRD*x-16.0*ONETHIRD*y;
  gphi[102] = 128.0*z*z*x-64.0*z*x+16.0*ONETHIRD*x;
  gphi[103] = 128.0*z*z*y-64.0*z*y+16.0*ONETHIRD*y;
  gphi[104] = -1.0+128.0*ONETHIRD*z*z*z+44.0*ONETHIRD*z-48.0*z*z;

    return 0;
#endif
#if SOLN_USE_P5
  case 5:
     gphi[0] = 375.0*0.25*z+375.0*0.25*x+375.0*0.25*y-1562.5*z*y*x*x-1562.5*z*y*y*x-1562.5*z*z*y*x+1875.0*z*y*x-137.0/12.0-2125.0/8.0*x*x+937.5*z*z*y-3125.0/24.0*x*x*x*x-3125.0*ONESIXTH*z*z*z*x-2125.0/8.0*y*y+312.5*y*y*y+312.5*x*x*x-3125.0*0.25*y*y*x*x-3125.0*ONESIXTH*y*x*x*x-3125.0*ONESIXTH*y*y*y*x-3125.0*ONESIXTH*z*x*x*x-3125.0*ONESIXTH*z*y*y*y-3125.0*0.25*z*z*x*x+937.5*z*z*x-2125.0*0.25*z*x-3125.0*0.25*z*z*y*y-2125.0/8.0*z*z+312.5*z*z*z-3125.0/24.0*y*y*y*y-2125.0*0.25*z*y-3125.0/24.0*z*z*z*z+937.5*y*x*x+937.5*z*y*y+937.5*z*x*x-3125.0*ONESIXTH*z*z*z*y+937.5*y*y*x-2125.0*0.25*y*x;
  gphi[1] = 25.0-1925.0/12.0*z-1925.0*ONESIXTH*x-1925.0/12.0*y+4687.5*z*y*x*x+3125.0*z*y*y*x+3125.0*z*z*y*x-4375.0*z*y*x+8875.0/8.0*x*x-4375.0*0.25*z*z*y+15625.0/24.0*x*x*x*x+3125.0*ONETHIRD*z*z*z*x+8875.0/24.0*y*y-4375.0/12.0*y*y*y-4375.0*ONETHIRD*x*x*x+9375.0*0.25*y*y*x*x+6250.0*ONETHIRD*y*x*x*x+3125.0*ONETHIRD*y*y*y*x+6250.0*ONETHIRD*z*x*x*x+3125.0*ONESIXTH*z*y*y*y+9375.0*0.25*z*z*x*x-2187.5*z*z*x+8875.0*ONESIXTH*z*x+3125.0*0.25*z*z*y*y+8875.0/24.0*z*z-4375.0/12.0*z*z*z+3125.0/24.0*y*y*y*y+8875.0/12.0*z*y+3125.0/24.0*z*z*z*z-13125.0*0.25*y*x*x-4375.0*0.25*z*y*y-13125.0*0.25*z*x*x+3125.0*ONESIXTH*z*z*z*y-2187.5*y*y*x+8875.0*ONESIXTH*y*x;
  gphi[2] = -25.0+1175.0/12.0*z+2675.0*ONESIXTH*x+1175.0/12.0*y-4687.5*z*y*x*x-1562.5*z*y*y*x-1562.5*z*z*y*x+3125.0*z*y*x-7375.0*0.25*x*x+625.0*0.25*z*z*y-15625.0/12.0*x*x*x*x-3125.0*ONESIXTH*z*z*z*x-125.0*y*y+625.0/12.0*y*y*y+8125.0*ONETHIRD*x*x*x-9375.0*0.25*y*y*x*x-3125.0*y*x*x*x-3125.0*ONESIXTH*y*y*y*x-3125.0*z*x*x*x-9375.0*0.25*z*z*x*x+1562.5*z*z*x-8875.0*ONESIXTH*z*x-125.0*z*z+625.0/12.0*z*z*z-250.0*z*y+16875.0*0.25*y*x*x+625.0*0.25*z*y*y+16875.0*0.25*z*x*x+1562.5*y*y*x-8875.0*ONESIXTH*y*x;
  gphi[3] = 15625.0/12.0*x*x*x*x+3125.0*0.25*z*z*x*x+50.0*ONETHIRD-9375.0*0.25*y*x*x-312.5*y*y*x-9375.0*0.25*z*x*x-312.5*z*z*x+3875.0*ONESIXTH*y*x+125.0*ONESIXTH*y*y+3875.0*ONESIXTH*z*x+125.0*ONESIXTH*z*z+1562.5*z*y*x*x+6250.0*ONETHIRD*y*x*x*x+3125.0*0.25*y*y*x*x+6250.0*ONETHIRD*z*x*x*x-37.5*y-625.0*z*y*x+125.0*ONETHIRD*z*y-37.5*z-325.0*x-2500.0*x*x*x+6125.0*0.25*x*x;
  gphi[4] = -15625.0/24.0*x*x*x*x-25.0*0.25+1875.0*0.25*y*x*x+1875.0*0.25*z*x*x-1375.0/12.0*y*x-1375.0/12.0*z*x-3125.0*ONESIXTH*y*x*x*x-3125.0*ONESIXTH*z*x*x*x+25.0*0.25*y+25.0*0.25*z+1525.0/12.0*x+6875.0*ONESIXTH*x*x*x-5125.0/8.0*x*x;
  gphi[5] = 3125.0/24.0*x*x*x*x+1.0-125.0*ONESIXTH*x-625.0*ONETHIRD*x*x*x+875.0/8.0*x*x;
  gphi[6] = 1562.5*z*y*y*y+1562.5*z*z*y*x+1562.5*z*z*y*y+3125.0*ONESIXTH*z*z*z*y-2187.5*z*y*x-2187.5*z*y*y-4375.0*0.25*z*z*y+8875.0/12.0*z*y+1562.5*z*y*x*x+3125.0*z*y*y*x+3125.0*ONESIXTH*y*y*y*y+1562.5*y*y*y*x+1562.5*y*y*x*x+3125.0*ONESIXTH*y*x*x*x-2187.5*y*y*x-4375.0*0.25*y*x*x-4375.0*0.25*y*y*y+8875.0/12.0*y*x+8875.0/12.0*y*y-1925.0/12.0*y;
  gphi[7] = 3750.0*y*x*x+5000.0*y*y*x+1250.0*y*y*y-5875.0*ONETHIRD*y*x-5875.0*ONESIXTH*y*y-4687.5*z*y*x*x-6250.0*z*y*y*x-3125.0*z*z*y*x-6250.0*ONETHIRD*y*x*x*x-4687.5*y*y*x*x-3125.0*y*y*y*x-1562.5*z*z*y*y+250.0*y+5000.0*z*y*x+2500.0*z*y*y+1250.0*z*z*y-5875.0*ONESIXTH*z*y-1562.5*z*y*y*y-3125.0*ONESIXTH*z*z*z*y-3125.0*ONESIXTH*y*y*y*y;
  gphi[8] = -4687.5*y*x*x-3437.5*y*y*x-625.0*0.25*y*y*y+1812.5*y*x+1125.0*0.25*y*y+4687.5*z*y*x*x+3125.0*z*y*y*x+1562.5*z*z*y*x+3125.0*y*x*x*x+4687.5*y*y*x*x+1562.5*y*y*y*x-125.0*y-3437.5*z*y*x-312.5*z*y*y-625.0*0.25*z*z*y+1125.0*0.25*z*y;
  gphi[9] = 2500.0*y*x*x+625.0*y*y*x-2125.0*ONETHIRD*y*x-125.0*ONETHIRD*y*y-1562.5*z*y*x*x-6250.0*ONETHIRD*y*x*x*x-1562.5*y*y*x*x+125.0*ONETHIRD*y+625.0*z*y*x-125.0*ONETHIRD*z*y;
  gphi[10] = -1875.0*0.25*y*x*x+1375.0/12.0*y*x+3125.0*ONESIXTH*y*x*x*x-25.0*0.25*y;
  gphi[11] = -1562.5*z*y*y*y-3125.0*0.25*z*z*y*y+312.5*z*y*x+1562.5*z*y*y+625.0*0.25*z*z*y-250.0*z*y-1562.5*z*y*y*x-3125.0*0.25*y*y*y*y-1562.5*y*y*y*x-3125.0*0.25*y*y*x*x+1562.5*y*y*x+625.0*0.25*y*x*x+5625.0*0.25*y*y*y-250.0*y*x-8875.0/12.0*y*y+1175.0/12.0*y;
  gphi[12] = -1875.0*0.25*y*x*x-3437.5*y*y*x-1562.5*y*y*y+562.5*y*x+3625.0*0.25*y*y+3125.0*z*y*y*x+9375.0*0.25*y*y*x*x+3125.0*y*y*y*x+1562.5*z*y*y*y+3125.0*0.25*z*z*y*y-125.0*y-625.0*z*y*x-6875.0*0.25*z*y*y-625.0*0.25*z*z*y+1125.0*0.25*z*y+3125.0*0.25*y*y*y*y;
  gphi[13] = 1875.0*0.25*y*x*x+2187.5*y*y*x+625.0*0.25*y*y*y-375.0*y*x-187.5*y*y-1562.5*z*y*y*x-9375.0*0.25*y*y*x*x-1562.5*y*y*y*x+125.0*0.25*y+312.5*z*y*x+625.0*0.25*z*y*y-125.0*0.25*z*y;
  gphi[14] = -625.0*0.25*y*x*x-312.5*y*y*x+62.5*y*x+125.0*ONESIXTH*y*y+3125.0*0.25*y*y*x*x-25.0*ONESIXTH*y;
  gphi[15] = -312.5*y*y*x-3125.0*0.25*y*y*y+125.0*ONETHIRD*y*x+3875.0/12.0*y*y+3125.0*ONESIXTH*y*y*y*x+3125.0*ONESIXTH*z*y*y*y-37.5*y-312.5*z*y*y+125.0*ONETHIRD*z*y+3125.0*ONESIXTH*y*y*y*y;
  gphi[16] = 625.0*y*y*x+2500.0*ONETHIRD*y*y*y-250.0*ONETHIRD*y*x-2125.0*ONESIXTH*y*y-3125.0*ONETHIRD*y*y*y*x-3125.0*ONESIXTH*z*y*y*y+125.0*ONETHIRD*y+312.5*z*y*y-125.0*ONETHIRD*z*y-3125.0*ONESIXTH*y*y*y*y;
  gphi[17] = -312.5*y*y*x-625.0/12.0*y*y*y+125.0*ONETHIRD*y*x+125.0*0.25*y*y+3125.0*ONESIXTH*y*y*y*x-25.0*ONESIXTH*y;
  gphi[18] = 625.0*0.25*y*y*y-1375.0/24.0*y*y+25.0*0.25*y-3125.0/24.0*y*y*y*y;
  gphi[19] = -625.0*0.25*y*y*y+1375.0/24.0*y*y-25.0*0.25*y+3125.0/24.0*y*y*y*y;
  gphi[20] = 0.0;
  gphi[21] = 3125.0*ONESIXTH*z*z*z*z+3125.0*ONESIXTH*z*y*y*y+3125.0*z*z*y*x+1562.5*z*z*y*y+1562.5*z*z*z*y-2187.5*z*y*x-4375.0*0.25*z*y*y-2187.5*z*z*y+8875.0/12.0*z*y+1562.5*z*y*x*x+1562.5*z*y*y*x-4375.0*0.25*z*z*z+3125.0*ONESIXTH*z*x*x*x+1562.5*z*z*x*x+1562.5*z*z*z*x-4375.0*0.25*z*x*x-2187.5*z*z*x+8875.0/12.0*z*z-1925.0/12.0*z+8875.0/12.0*z*x;
  gphi[22] = -4687.5*z*y*x*x-3125.0*z*y*y*x-3125.0*ONESIXTH*z*y*y*y-6250.0*z*z*y*x-1562.5*z*z*y*y-1562.5*z*z*z*y+5000.0*z*y*x+1250.0*z*y*y+2500.0*z*z*y-5875.0*ONESIXTH*z*y-6250.0*ONETHIRD*z*x*x*x-3125.0*z*z*z*x-4687.5*z*z*x*x-3125.0*ONESIXTH*z*z*z*z+5000.0*z*z*x+3750.0*z*x*x+1250.0*z*z*z-5875.0*ONESIXTH*z*z-5875.0*ONETHIRD*z*x+250.0*z;
  gphi[23] = 4687.5*z*y*x*x+1562.5*z*y*y*x+3125.0*z*z*y*x-3437.5*z*y*x-625.0*0.25*z*y*y-312.5*z*z*y+1125.0*0.25*z*y+3125.0*z*x*x*x+1562.5*z*z*z*x+4687.5*z*z*x*x-3437.5*z*z*x-4687.5*z*x*x-625.0*0.25*z*z*z+1125.0*0.25*z*z+1812.5*z*x-125.0*z;
  gphi[24] = -1562.5*z*y*x*x+625.0*z*y*x-125.0*ONETHIRD*z*y-6250.0*ONETHIRD*z*x*x*x-1562.5*z*z*x*x+625.0*z*z*x+2500.0*z*x*x-125.0*ONETHIRD*z*z-2125.0*ONETHIRD*z*x+125.0*ONETHIRD*z;
  gphi[25] = 3125.0*ONESIXTH*z*x*x*x-1875.0*0.25*z*x*x+1375.0/12.0*z*x-25.0*0.25*z;
  gphi[26] = -1562.5*z*y*x*x-3125.0*z*y*y*x-1562.5*z*y*y*y-3125.0*z*z*y*x-3125.0*z*z*y*y-1562.5*z*z*z*y+2500.0*z*y*x+2500.0*z*y*y+2500.0*z*z*y-5875.0*ONESIXTH*z*y;
  gphi[27] = 4687.5*z*y*x*x+6250.0*z*y*y*x+1562.5*z*y*y*y+6250.0*z*z*y*x+3125.0*z*z*y*y+1562.5*z*z*z*y-5625.0*z*y*x-2812.5*z*y*y-2812.5*z*z*y+1250.0*z*y;
  gphi[28] = -4687.5*z*y*x*x-3125.0*z*y*y*x-3125.0*z*z*y*x+3750.0*z*y*x+312.5*z*y*y+312.5*z*z*y-312.5*z*y;
  gphi[29] = 1562.5*z*y*x*x-625.0*z*y*x+125.0*ONETHIRD*z*y;
  gphi[30] = 1562.5*z*y*y*x+1562.5*z*y*y*y+1562.5*z*z*y*y-312.5*z*y*x-6875.0*0.25*z*y*y-312.5*z*z*y+1125.0*0.25*z*y;
  gphi[31] = -3125.0*z*y*y*x-1562.5*z*y*y*y-1562.5*z*z*y*y+625.0*z*y*x+1875.0*z*y*y+312.5*z*z*y-312.5*z*y;
  gphi[32] = 1562.5*z*y*y*x-312.5*z*y*x-625.0*0.25*z*y*y+125.0*0.25*z*y;
  gphi[33] = -3125.0*ONESIXTH*z*y*y*y+312.5*z*y*y-125.0*ONETHIRD*z*y;
  gphi[34] = 3125.0*ONESIXTH*z*y*y*y-312.5*z*y*y+125.0*ONETHIRD*z*y;
  gphi[35] = 0.0;
  gphi[36] = -3125.0*0.25*z*z*z*z-1562.5*z*z*y*x-3125.0*0.25*z*z*y*y-1562.5*z*z*z*y+312.5*z*y*x+625.0*0.25*z*y*y+1562.5*z*z*y-250.0*z*y+5625.0*0.25*z*z*z-3125.0*0.25*z*z*x*x-1562.5*z*z*z*x+625.0*0.25*z*x*x+1562.5*z*z*x-8875.0/12.0*z*z+1175.0/12.0*z-250.0*z*x;
  gphi[37] = -625.0*z*y*x-625.0*0.25*z*y*y+1125.0*0.25*z*y+1562.5*z*z*z*y+3125.0*0.25*z*z*y*y+3125.0*z*z*y*x-6875.0*0.25*z*z*y+9375.0*0.25*z*z*x*x-1875.0*0.25*z*x*x+3125.0*z*z*z*x+3125.0*0.25*z*z*z*z-125.0*z-3437.5*z*z*x-1562.5*z*z*z+562.5*z*x+3625.0*0.25*z*z;
  gphi[38] = 312.5*z*y*x-125.0*0.25*z*y-1562.5*z*z*y*x+625.0*0.25*z*z*y-9375.0*0.25*z*z*x*x+1875.0*0.25*z*x*x-1562.5*z*z*z*x+125.0*0.25*z+2187.5*z*z*x+625.0*0.25*z*z*z-375.0*z*x-187.5*z*z;
  gphi[39] = 3125.0*0.25*z*z*x*x-625.0*0.25*z*x*x-25.0*ONESIXTH*z-312.5*z*z*x+62.5*z*x+125.0*ONESIXTH*z*z;
  gphi[40] = -312.5*z*y*x-312.5*z*y*y+1125.0*0.25*z*y+1562.5*z*z*z*y+1562.5*z*z*y*y+1562.5*z*z*y*x-6875.0*0.25*z*z*y;
  gphi[41] = 625.0*z*y*x+312.5*z*y*y-312.5*z*y-1562.5*z*z*z*y-1562.5*z*z*y*y-3125.0*z*z*y*x+1875.0*z*z*y;
  gphi[42] = -312.5*z*y*x+125.0*0.25*z*y+1562.5*z*z*y*x-625.0*0.25*z*z*y;
  gphi[43] = 625.0*0.25*z*y*y-125.0*0.25*z*y-3125.0*0.25*z*z*y*y+625.0*0.25*z*z*y;
  gphi[44] = -625.0*0.25*z*y*y+125.0*0.25*z*y+3125.0*0.25*z*z*y*y-625.0*0.25*z*z*y;
  gphi[45] = 0.0;
  gphi[46] = 125.0*ONETHIRD*z*y+3125.0*ONESIXTH*z*z*z*y-312.5*z*z*y-312.5*z*z*x-3125.0*0.25*z*z*z+125.0*ONETHIRD*z*x+3875.0/12.0*z*z+3125.0*ONESIXTH*z*z*z*x-37.5*z+3125.0*ONESIXTH*z*z*z*z;
  gphi[47] = -125.0*ONETHIRD*z*y-3125.0*ONESIXTH*z*z*z*y+312.5*z*z*y+625.0*z*z*x+2500.0*ONETHIRD*z*z*z-250.0*ONETHIRD*z*x-2125.0*ONESIXTH*z*z-3125.0*ONETHIRD*z*z*z*x+125.0*ONETHIRD*z-3125.0*ONESIXTH*z*z*z*z;
  gphi[48] = -312.5*z*z*x-625.0/12.0*z*z*z+125.0*ONETHIRD*z*x+125.0*0.25*z*z+3125.0*ONESIXTH*z*z*z*x-25.0*ONESIXTH*z;
  gphi[49] = -125.0*ONETHIRD*z*y-3125.0*ONESIXTH*z*z*z*y+312.5*z*z*y;
  gphi[50] = 125.0*ONETHIRD*z*y+3125.0*ONESIXTH*z*z*z*y-312.5*z*z*y;
  gphi[51] = 0.0;
  gphi[52] = -3125.0/24.0*z*z*z*z+625.0*0.25*z*z*z-1375.0/24.0*z*z+25.0*0.25*z;
  gphi[53] = 3125.0/24.0*z*z*z*z-625.0*0.25*z*z*z+1375.0/24.0*z*z-25.0*0.25*z;
  gphi[54] = 0.0;
  gphi[55] = 0.0;
  gphi[56] = 375.0*0.25*z+375.0*0.25*x+375.0*0.25*y-1562.5*z*y*x*x-1562.5*z*y*y*x-1562.5*z*z*y*x+1875.0*z*y*x-137.0/12.0-2125.0/8.0*x*x+937.5*z*z*y-3125.0/24.0*x*x*x*x-3125.0*ONESIXTH*z*z*z*x-2125.0/8.0*y*y+312.5*y*y*y+312.5*x*x*x-3125.0*0.25*y*y*x*x-3125.0*ONESIXTH*y*x*x*x-3125.0*ONESIXTH*y*y*y*x-3125.0*ONESIXTH*z*x*x*x-3125.0*ONESIXTH*z*y*y*y-3125.0*0.25*z*z*x*x+937.5*z*z*x-2125.0*0.25*z*x-3125.0*0.25*z*z*y*y-2125.0/8.0*z*z+312.5*z*z*z-3125.0/24.0*y*y*y*y-2125.0*0.25*z*y-3125.0/24.0*z*z*z*z+937.5*y*x*x+937.5*z*y*y+937.5*z*x*x-3125.0*ONESIXTH*z*z*z*y+937.5*y*y*x-2125.0*0.25*y*x;
  gphi[57] = 1562.5*z*y*y*x+1562.5*z*z*x*x+1562.5*z*z*y*x+3125.0*ONESIXTH*z*z*z*x-2187.5*z*x*x-2187.5*z*y*x-4375.0*0.25*z*z*x+8875.0/12.0*z*x+1562.5*z*x*x*x+3125.0*z*y*x*x+3125.0*ONESIXTH*y*y*y*x+1562.5*y*y*x*x+1562.5*y*x*x*x+3125.0*ONESIXTH*x*x*x*x-2187.5*y*x*x-4375.0*0.25*x*x*x-4375.0*0.25*y*y*x+8875.0/12.0*x*x+8875.0/12.0*y*x-1925.0/12.0*x;
  gphi[58] = -3125.0*0.25*z*z*x*x+1562.5*z*x*x+312.5*z*y*x+625.0*0.25*z*z*x-250.0*z*x-1562.5*z*x*x*x-1562.5*z*y*x*x-3125.0*0.25*y*y*x*x-1562.5*y*x*x*x-3125.0*0.25*x*x*x*x+1562.5*y*x*x+5625.0*0.25*x*x*x+625.0*0.25*y*y*x-8875.0/12.0*x*x-250.0*y*x+1175.0/12.0*x;
  gphi[59] = -3125.0*0.25*x*x*x-312.5*y*x*x+3875.0/12.0*x*x+125.0*ONETHIRD*y*x+3125.0*ONESIXTH*z*x*x*x+3125.0*ONESIXTH*x*x*x*x+3125.0*ONESIXTH*y*x*x*x-37.5*x-312.5*z*x*x+125.0*ONETHIRD*z*x;
  gphi[60] = 625.0*0.25*x*x*x-1375.0/24.0*x*x-3125.0/24.0*x*x*x*x+25.0*0.25*x;
  gphi[61] = 0.0;
  gphi[62] = 25.0-1925.0/12.0*z-1925.0/12.0*x-1925.0*ONESIXTH*y+3125.0*z*y*x*x+4687.5*z*y*y*x+3125.0*z*z*y*x-4375.0*z*y*x+8875.0/24.0*x*x-2187.5*z*z*y+3125.0/24.0*x*x*x*x+3125.0*ONESIXTH*z*z*z*x+8875.0/8.0*y*y-4375.0*ONETHIRD*y*y*y-4375.0/12.0*x*x*x+9375.0*0.25*y*y*x*x+3125.0*ONETHIRD*y*x*x*x+6250.0*ONETHIRD*y*y*y*x+3125.0*ONESIXTH*z*x*x*x+6250.0*ONETHIRD*z*y*y*y+3125.0*0.25*z*z*x*x-4375.0*0.25*z*z*x+8875.0/12.0*z*x+9375.0*0.25*z*z*y*y+8875.0/24.0*z*z-4375.0/12.0*z*z*z+15625.0/24.0*y*y*y*y+8875.0*ONESIXTH*z*y+3125.0/24.0*z*z*z*z-2187.5*y*x*x-13125.0*0.25*z*y*y-4375.0*0.25*z*x*x+3125.0*ONETHIRD*z*z*z*y-13125.0*0.25*y*y*x+8875.0*ONESIXTH*y*x;
  gphi[63] = 1250.0*x*x*x+5000.0*y*x*x+3750.0*y*y*x-5875.0*ONESIXTH*x*x-5875.0*ONETHIRD*y*x-1562.5*z*x*x*x-6250.0*z*y*x*x-1562.5*z*z*x*x-3125.0*ONESIXTH*x*x*x*x-3125.0*y*x*x*x-4687.5*y*y*x*x-3125.0*z*z*y*x+250.0*x+2500.0*z*x*x+5000.0*z*y*x+1250.0*z*z*x-5875.0*ONESIXTH*z*x-4687.5*z*y*y*x-3125.0*ONESIXTH*z*z*z*x-6250.0*ONETHIRD*y*y*y*x;
  gphi[64] = -1562.5*x*x*x-3437.5*y*x*x-1875.0*0.25*y*y*x+3625.0*0.25*x*x+562.5*y*x+1562.5*z*x*x*x+3125.0*z*y*x*x+3125.0*0.25*z*z*x*x+3125.0*0.25*x*x*x*x+3125.0*y*x*x*x+9375.0*0.25*y*y*x*x-125.0*x-6875.0*0.25*z*x*x-625.0*z*y*x-625.0*0.25*z*z*x+1125.0*0.25*z*x;
  gphi[65] = 2500.0*ONETHIRD*x*x*x+625.0*y*x*x-2125.0*ONESIXTH*x*x-250.0*ONETHIRD*y*x-3125.0*ONESIXTH*z*x*x*x-3125.0*ONESIXTH*x*x*x*x-3125.0*ONETHIRD*y*x*x*x+125.0*ONETHIRD*x+312.5*z*x*x-125.0*ONETHIRD*z*x;
  gphi[66] = -625.0*0.25*x*x*x+1375.0/24.0*x*x+3125.0/24.0*x*x*x*x-25.0*0.25*x;
  gphi[67] = -25.0+1175.0/12.0*z+1175.0/12.0*x+2675.0*ONESIXTH*y-1562.5*z*y*x*x-4687.5*z*y*y*x-1562.5*z*z*y*x+3125.0*z*y*x-125.0*x*x+1562.5*z*z*y-7375.0*0.25*y*y+8125.0*ONETHIRD*y*y*y+625.0/12.0*x*x*x-9375.0*0.25*y*y*x*x-3125.0*ONESIXTH*y*x*x*x-3125.0*y*y*y*x-3125.0*z*y*y*y+625.0*0.25*z*z*x-250.0*z*x-9375.0*0.25*z*z*y*y-125.0*z*z+625.0/12.0*z*z*z-15625.0/12.0*y*y*y*y-8875.0*ONESIXTH*z*y+1562.5*y*x*x+16875.0*0.25*z*y*y+625.0*0.25*z*x*x-3125.0*ONESIXTH*z*z*z*y+16875.0*0.25*y*y*x-8875.0*ONESIXTH*y*x;
  gphi[68] = -625.0*0.25*x*x*x-3437.5*y*x*x-4687.5*y*y*x+1125.0*0.25*x*x+1812.5*y*x+3125.0*z*y*x*x+1562.5*y*x*x*x+4687.5*y*y*x*x+4687.5*z*y*y*x+1562.5*z*z*y*x-125.0*x-312.5*z*x*x-3437.5*z*y*x-625.0*0.25*z*z*x+1125.0*0.25*z*x+3125.0*y*y*y*x;
  gphi[69] = 625.0*0.25*x*x*x+2187.5*y*x*x+1875.0*0.25*y*y*x-187.5*x*x-375.0*y*x-1562.5*z*y*x*x-1562.5*y*x*x*x-9375.0*0.25*y*y*x*x+125.0*0.25*x+625.0*0.25*z*x*x+312.5*z*y*x-125.0*0.25*z*x;
  gphi[70] = -625.0/12.0*x*x*x-312.5*y*x*x+125.0*0.25*x*x+125.0*ONETHIRD*y*x+3125.0*ONESIXTH*y*x*x*x-25.0*ONESIXTH*x;
  gphi[71] = -9375.0*0.25*z*y*y+50.0*ONETHIRD-312.5*y*x*x-9375.0*0.25*y*y*x+125.0*ONESIXTH*x*x+3875.0*ONESIXTH*y*x+3125.0*0.25*y*y*x*x+1562.5*z*y*y*x-37.5*x-625.0*z*y*x+125.0*ONETHIRD*z*x-312.5*z*z*y+6250.0*ONETHIRD*y*y*y*x+3875.0*ONESIXTH*z*y+6250.0*ONETHIRD*z*y*y*y+125.0*ONESIXTH*z*z+3125.0*0.25*z*z*y*y-325.0*y+6125.0*0.25*y*y-2500.0*y*y*y+15625.0/12.0*y*y*y*y-37.5*z;
  gphi[72] = 625.0*y*x*x+2500.0*y*y*x-125.0*ONETHIRD*x*x-2125.0*ONETHIRD*y*x-1562.5*y*y*x*x-1562.5*z*y*y*x+125.0*ONETHIRD*x+625.0*z*y*x-125.0*ONETHIRD*z*x-6250.0*ONETHIRD*y*y*y*x;
  gphi[73] = -312.5*y*x*x-625.0*0.25*y*y*x+125.0*ONESIXTH*x*x+62.5*y*x+3125.0*0.25*y*y*x*x-25.0*ONESIXTH*x;
  gphi[74] = 1875.0*0.25*z*y*y-25.0*0.25+1875.0*0.25*y*y*x-1375.0/12.0*y*x+25.0*0.25*x-3125.0*ONESIXTH*y*y*y*x-1375.0/12.0*z*y-3125.0*ONESIXTH*z*y*y*y+1525.0/12.0*y-5125.0/8.0*y*y+6875.0*ONESIXTH*y*y*y-15625.0/24.0*y*y*y*y+25.0*0.25*z;
  gphi[75] = -1875.0*0.25*y*y*x+1375.0/12.0*y*x-25.0*0.25*x+3125.0*ONESIXTH*y*y*y*x;
  gphi[76] = 1.0-125.0*ONESIXTH*y+875.0/8.0*y*y-625.0*ONETHIRD*y*y*y+3125.0/24.0*y*y*y*y;
  gphi[77] = 3125.0*ONESIXTH*z*z*z*z+3125.0*ONESIXTH*z*y*y*y+3125.0*z*z*y*x+1562.5*z*z*y*y+1562.5*z*z*z*y-2187.5*z*y*x-4375.0*0.25*z*y*y-2187.5*z*z*y+8875.0/12.0*z*y+1562.5*z*y*x*x+1562.5*z*y*y*x-4375.0*0.25*z*z*z+3125.0*ONESIXTH*z*x*x*x+1562.5*z*z*x*x+1562.5*z*z*z*x-4375.0*0.25*z*x*x-2187.5*z*z*x+8875.0/12.0*z*z-1925.0/12.0*z+8875.0/12.0*z*x;
  gphi[78] = -1562.5*z*x*x*x-3125.0*z*y*x*x-1562.5*z*y*y*x-3125.0*z*z*x*x-3125.0*z*z*y*x-1562.5*z*z*z*x+2500.0*z*x*x+2500.0*z*y*x+2500.0*z*z*x-5875.0*ONESIXTH*z*x;
  gphi[79] = 1562.5*z*x*x*x+1562.5*z*y*x*x+1562.5*z*z*x*x-6875.0*0.25*z*x*x-312.5*z*y*x-312.5*z*z*x+1125.0*0.25*z*x;
  gphi[80] = -3125.0*ONESIXTH*z*x*x*x+312.5*z*x*x-125.0*ONETHIRD*z*x;
  gphi[81] = 0.0;
  gphi[82] = -3125.0*ONESIXTH*z*x*x*x-3125.0*z*y*x*x-4687.5*z*y*y*x-1562.5*z*z*x*x-6250.0*z*z*y*x-1562.5*z*z*z*x+1250.0*z*x*x+5000.0*z*y*x+2500.0*z*z*x-5875.0*ONESIXTH*z*x-5875.0*ONETHIRD*z*y-5875.0*ONESIXTH*z*z-4687.5*z*z*y*y-3125.0*ONESIXTH*z*z*z*z+5000.0*z*z*y+3750.0*z*y*y+1250.0*z*z*z+250.0*z-6250.0*ONETHIRD*z*y*y*y-3125.0*z*z*z*y;
  gphi[83] = 1562.5*z*x*x*x+6250.0*z*y*x*x+4687.5*z*y*y*x+3125.0*z*z*x*x+6250.0*z*z*y*x+1562.5*z*z*z*x-2812.5*z*x*x-5625.0*z*y*x-2812.5*z*z*x+1250.0*z*x;
  gphi[84] = -1562.5*z*x*x*x-3125.0*z*y*x*x-1562.5*z*z*x*x+1875.0*z*x*x+625.0*z*y*x+312.5*z*z*x-312.5*z*x;
  gphi[85] = 3125.0*ONESIXTH*z*x*x*x-312.5*z*x*x+125.0*ONETHIRD*z*x;
  gphi[86] = 1562.5*z*y*x*x+4687.5*z*y*y*x+3125.0*z*z*y*x-625.0*0.25*z*x*x-3437.5*z*y*x-312.5*z*z*x+1125.0*0.25*z*x+1812.5*z*y+1125.0*0.25*z*z+4687.5*z*z*y*y-3437.5*z*z*y-4687.5*z*y*y-625.0*0.25*z*z*z-125.0*z+3125.0*z*y*y*y+1562.5*z*z*z*y;
  gphi[87] = -3125.0*z*y*x*x-4687.5*z*y*y*x-3125.0*z*z*y*x+312.5*z*x*x+3750.0*z*y*x+312.5*z*z*x-312.5*z*x;
  gphi[88] = 1562.5*z*y*x*x-625.0*0.25*z*x*x-312.5*z*y*x+125.0*0.25*z*x;
  gphi[89] = -1562.5*z*y*y*x+625.0*z*y*x-125.0*ONETHIRD*z*x-2125.0*ONETHIRD*z*y-125.0*ONETHIRD*z*z-1562.5*z*z*y*y+625.0*z*z*y+2500.0*z*y*y+125.0*ONETHIRD*z-6250.0*ONETHIRD*z*y*y*y;
  gphi[90] = 1562.5*z*y*y*x-625.0*z*y*x+125.0*ONETHIRD*z*x;
  gphi[91] = 1375.0/12.0*z*y-1875.0*0.25*z*y*y-25.0*0.25*z+3125.0*ONESIXTH*z*y*y*y;
  gphi[92] = -3125.0*0.25*z*z*z*z-1562.5*z*z*y*x-3125.0*0.25*z*z*y*y-1562.5*z*z*z*y+312.5*z*y*x+625.0*0.25*z*y*y+1562.5*z*z*y-250.0*z*y+5625.0*0.25*z*z*z-3125.0*0.25*z*z*x*x-1562.5*z*z*z*x+625.0*0.25*z*x*x+1562.5*z*z*x-8875.0/12.0*z*z+1175.0/12.0*z-250.0*z*x;
  gphi[93] = -312.5*z*x*x-312.5*z*y*x+1125.0*0.25*z*x+1562.5*z*z*z*x+1562.5*z*z*y*x+1562.5*z*z*x*x-6875.0*0.25*z*z*x;
  gphi[94] = 625.0*0.25*z*x*x-125.0*0.25*z*x-3125.0*0.25*z*z*x*x+625.0*0.25*z*z*x;
  gphi[95] = 0.0;
  gphi[96] = -625.0*0.25*z*x*x-625.0*z*y*x+1125.0*0.25*z*x+1562.5*z*z*z*x+3125.0*z*z*y*x+3125.0*0.25*z*z*x*x-6875.0*0.25*z*z*x+562.5*z*y+3625.0*0.25*z*z+9375.0*0.25*z*z*y*y-3437.5*z*z*y-1875.0*0.25*z*y*y+3125.0*z*z*z*y+3125.0*0.25*z*z*z*z-125.0*z-1562.5*z*z*z;
  gphi[97] = 312.5*z*x*x+625.0*z*y*x-312.5*z*x-1562.5*z*z*z*x-3125.0*z*z*y*x-1562.5*z*z*x*x+1875.0*z*z*x;
  gphi[98] = -625.0*0.25*z*x*x+125.0*0.25*z*x+3125.0*0.25*z*z*x*x-625.0*0.25*z*z*x;
  gphi[99] = 312.5*z*y*x-125.0*0.25*z*x-1562.5*z*z*y*x+625.0*0.25*z*z*x-375.0*z*y-187.5*z*z-9375.0*0.25*z*z*y*y+2187.5*z*z*y+1875.0*0.25*z*y*y-1562.5*z*z*z*y+125.0*0.25*z+625.0*0.25*z*z*z;
  gphi[100] = -312.5*z*y*x+125.0*0.25*z*x+1562.5*z*z*y*x-625.0*0.25*z*z*x;
  gphi[101] = 62.5*z*y+125.0*ONESIXTH*z*z+3125.0*0.25*z*z*y*y-312.5*z*z*y-625.0*0.25*z*y*y-25.0*ONESIXTH*z;
  gphi[102] = 125.0*ONETHIRD*z*y+3125.0*ONESIXTH*z*z*z*y-312.5*z*z*y-312.5*z*z*x-3125.0*0.25*z*z*z+125.0*ONETHIRD*z*x+3875.0/12.0*z*z+3125.0*ONESIXTH*z*z*z*x-37.5*z+3125.0*ONESIXTH*z*z*z*z;
  gphi[103] = -125.0*ONETHIRD*z*x-3125.0*ONESIXTH*z*z*z*x+312.5*z*z*x;
  gphi[104] = 0.0;
  gphi[105] = -3125.0*ONESIXTH*z*z*z*z-125.0*ONETHIRD*z*x-3125.0*ONESIXTH*z*z*z*x+312.5*z*z*x-250.0*ONETHIRD*z*y-2125.0*ONESIXTH*z*z+625.0*z*z*y-3125.0*ONETHIRD*z*z*z*y+125.0*ONETHIRD*z+2500.0*ONETHIRD*z*z*z;
  gphi[106] = 125.0*ONETHIRD*z*x+3125.0*ONESIXTH*z*z*z*x-312.5*z*z*x;
  gphi[107] = 125.0*ONETHIRD*z*y+125.0*0.25*z*z-312.5*z*z*y+3125.0*ONESIXTH*z*z*z*y-25.0*ONESIXTH*z-625.0/12.0*z*z*z;
  gphi[108] = -3125.0/24.0*z*z*z*z+625.0*0.25*z*z*z-1375.0/24.0*z*z+25.0*0.25*z;
  gphi[109] = 0.0;
  gphi[110] = 3125.0/24.0*z*z*z*z-625.0*0.25*z*z*z+1375.0/24.0*z*z-25.0*0.25*z;
  gphi[111] = 0.0;
  gphi[112] = 375.0*0.25*z+375.0*0.25*x+375.0*0.25*y-1562.5*z*y*x*x-1562.5*z*y*y*x-1562.5*z*z*y*x+1875.0*z*y*x-137.0/12.0-2125.0/8.0*x*x+937.5*z*z*y-3125.0/24.0*x*x*x*x-3125.0*ONESIXTH*z*z*z*x-2125.0/8.0*y*y+312.5*y*y*y+312.5*x*x*x-3125.0*0.25*y*y*x*x-3125.0*ONESIXTH*y*x*x*x-3125.0*ONESIXTH*y*y*y*x-3125.0*ONESIXTH*z*x*x*x-3125.0*ONESIXTH*z*y*y*y-3125.0*0.25*z*z*x*x+937.5*z*z*x-2125.0*0.25*z*x-3125.0*0.25*z*z*y*y-2125.0/8.0*z*z+312.5*z*z*z-3125.0/24.0*y*y*y*y-2125.0*0.25*z*y-3125.0/24.0*z*z*z*z+937.5*y*x*x+937.5*z*y*y+937.5*z*x*x-3125.0*ONESIXTH*z*z*z*y+937.5*y*y*x-2125.0*0.25*y*x;
  gphi[113] = 1562.5*z*y*y*x+1562.5*z*z*x*x+1562.5*z*z*y*x+3125.0*ONESIXTH*z*z*z*x-2187.5*z*x*x-2187.5*z*y*x-4375.0*0.25*z*z*x+8875.0/12.0*z*x+1562.5*z*x*x*x+3125.0*z*y*x*x+3125.0*ONESIXTH*y*y*y*x+1562.5*y*y*x*x+1562.5*y*x*x*x+3125.0*ONESIXTH*x*x*x*x-2187.5*y*x*x-4375.0*0.25*x*x*x-4375.0*0.25*y*y*x+8875.0/12.0*x*x+8875.0/12.0*y*x-1925.0/12.0*x;
  gphi[114] = -3125.0*0.25*z*z*x*x+1562.5*z*x*x+312.5*z*y*x+625.0*0.25*z*z*x-250.0*z*x-1562.5*z*x*x*x-1562.5*z*y*x*x-3125.0*0.25*y*y*x*x-1562.5*y*x*x*x-3125.0*0.25*x*x*x*x+1562.5*y*x*x+5625.0*0.25*x*x*x+625.0*0.25*y*y*x-8875.0/12.0*x*x-250.0*y*x+1175.0/12.0*x;
  gphi[115] = -3125.0*0.25*x*x*x-312.5*y*x*x+3875.0/12.0*x*x+125.0*ONETHIRD*y*x+3125.0*ONESIXTH*z*x*x*x+3125.0*ONESIXTH*x*x*x*x+3125.0*ONESIXTH*y*x*x*x-37.5*x-312.5*z*x*x+125.0*ONETHIRD*z*x;
  gphi[116] = 625.0*0.25*x*x*x-1375.0/24.0*x*x-3125.0/24.0*x*x*x*x+25.0*0.25*x;
  gphi[117] = 0.0;
  gphi[118] = 1562.5*z*y*y*y+1562.5*z*z*y*x+1562.5*z*z*y*y+3125.0*ONESIXTH*z*z*z*y-2187.5*z*y*x-2187.5*z*y*y-4375.0*0.25*z*z*y+8875.0/12.0*z*y+1562.5*z*y*x*x+3125.0*z*y*y*x+3125.0*ONESIXTH*y*y*y*y+1562.5*y*y*y*x+1562.5*y*y*x*x+3125.0*ONESIXTH*y*x*x*x-2187.5*y*y*x-4375.0*0.25*y*x*x-4375.0*0.25*y*y*y+8875.0/12.0*y*x+8875.0/12.0*y*y-1925.0/12.0*y;
  gphi[119] = -1562.5*y*x*x*x-3125.0*y*y*x*x-3125.0*z*y*x*x-3125.0*z*y*y*x+2500.0*y*x*x+2500.0*y*y*x+2500.0*z*y*x-5875.0*ONESIXTH*y*x-1562.5*y*y*y*x-1562.5*z*z*y*x;
  gphi[120] = 1562.5*y*x*x*x+1562.5*y*y*x*x+1562.5*z*y*x*x-6875.0*0.25*y*x*x-312.5*y*y*x-312.5*z*y*x+1125.0*0.25*y*x;
  gphi[121] = -3125.0*ONESIXTH*y*x*x*x+312.5*y*x*x-125.0*ONETHIRD*y*x;
  gphi[122] = 0.0;
  gphi[123] = -1562.5*z*y*y*y-3125.0*0.25*z*z*y*y+312.5*z*y*x+1562.5*z*y*y+625.0*0.25*z*z*y-250.0*z*y-1562.5*z*y*y*x-3125.0*0.25*y*y*y*y-1562.5*y*y*y*x-3125.0*0.25*y*y*x*x+1562.5*y*y*x+625.0*0.25*y*x*x+5625.0*0.25*y*y*y-250.0*y*x-8875.0/12.0*y*y+1175.0/12.0*y;
  gphi[124] = 1562.5*y*y*x*x+1562.5*y*y*y*x+1562.5*z*y*y*x-312.5*y*x*x-6875.0*0.25*y*y*x-312.5*z*y*x+1125.0*0.25*y*x;
  gphi[125] = -3125.0*0.25*y*y*x*x+625.0*0.25*y*x*x+625.0*0.25*y*y*x-125.0*0.25*y*x;
  gphi[126] = 0.0;
  gphi[127] = -312.5*y*y*x-3125.0*0.25*y*y*y+125.0*ONETHIRD*y*x+3875.0/12.0*y*y+3125.0*ONESIXTH*y*y*y*x+3125.0*ONESIXTH*z*y*y*y-37.5*y-312.5*z*y*y+125.0*ONETHIRD*z*y+3125.0*ONESIXTH*y*y*y*y;
  gphi[128] = -3125.0*ONESIXTH*y*y*y*x+312.5*y*y*x-125.0*ONETHIRD*y*x;
  gphi[129] = 0.0;
  gphi[130] = 625.0*0.25*y*y*y-1375.0/24.0*y*y+25.0*0.25*y-3125.0/24.0*y*y*y*y;
  gphi[131] = 0.0;
  gphi[132] = 0.0;
  gphi[133] = 25.0-1925.0*ONESIXTH*z-1925.0/12.0*x-1925.0/12.0*y+3125.0*z*y*x*x+3125.0*z*y*y*x+4687.5*z*z*y*x-4375.0*z*y*x+8875.0/24.0*x*x-13125.0*0.25*z*z*y+3125.0/24.0*x*x*x*x+6250.0*ONETHIRD*z*z*z*x+8875.0/24.0*y*y-4375.0/12.0*y*y*y-4375.0/12.0*x*x*x+3125.0*0.25*y*y*x*x+3125.0*ONESIXTH*y*x*x*x+3125.0*ONESIXTH*y*y*y*x+3125.0*ONETHIRD*z*x*x*x+3125.0*ONETHIRD*z*y*y*y+9375.0*0.25*z*z*x*x-13125.0*0.25*z*z*x+8875.0*ONESIXTH*z*x+9375.0*0.25*z*z*y*y+8875.0/8.0*z*z-4375.0*ONETHIRD*z*z*z+3125.0/24.0*y*y*y*y+8875.0*ONESIXTH*z*y+15625.0/24.0*z*z*z*z-4375.0*0.25*y*x*x-2187.5*z*y*y-2187.5*z*x*x+6250.0*ONETHIRD*z*z*z*y-4375.0*0.25*y*y*x+8875.0/12.0*y*x;
  gphi[134] = -1562.5*y*x*x*x-1562.5*y*y*x*x-3125.0*ONESIXTH*y*y*y*x-6250.0*z*y*x*x-3125.0*z*y*y*x-4687.5*z*z*y*x+2500.0*y*x*x+1250.0*y*y*x+5000.0*z*y*x-5875.0*ONESIXTH*y*x-3125.0*ONESIXTH*x*x*x*x-4687.5*z*z*x*x-3125.0*z*x*x*x-6250.0*ONETHIRD*z*z*z*x+5000.0*z*x*x+1250.0*x*x*x+3750.0*z*z*x-5875.0*ONETHIRD*z*x-5875.0*ONESIXTH*x*x+250.0*x;
  gphi[135] = 1562.5*y*x*x*x+3125.0*0.25*y*y*x*x+3125.0*z*y*x*x-6875.0*0.25*y*x*x-625.0*0.25*y*y*x-625.0*z*y*x+1125.0*0.25*y*x+3125.0*0.25*x*x*x*x+9375.0*0.25*z*z*x*x+3125.0*z*x*x*x-3437.5*z*x*x-1562.5*x*x*x-1875.0*0.25*z*z*x+562.5*z*x+3625.0*0.25*x*x-125.0*x;
  gphi[136] = -3125.0*ONESIXTH*y*x*x*x+312.5*y*x*x-125.0*ONETHIRD*y*x-3125.0*ONESIXTH*x*x*x*x-3125.0*ONETHIRD*z*x*x*x+625.0*z*x*x+2500.0*ONETHIRD*x*x*x-250.0*ONETHIRD*z*x-2125.0*ONESIXTH*x*x+125.0*ONETHIRD*x;
  gphi[137] = -625.0*0.25*x*x*x+1375.0/24.0*x*x+3125.0/24.0*x*x*x*x-25.0*0.25*x;
  gphi[138] = -3125.0*ONESIXTH*y*x*x*x-1562.5*y*y*x*x-1562.5*y*y*y*x-3125.0*z*y*x*x-6250.0*z*y*y*x-4687.5*z*z*y*x+1250.0*y*x*x+2500.0*y*y*x+5000.0*z*y*x-5875.0*ONESIXTH*y*x-5875.0*ONESIXTH*y*y-5875.0*ONETHIRD*z*y-3125.0*z*y*y*y-6250.0*ONETHIRD*z*z*z*y+5000.0*z*y*y+1250.0*y*y*y+3750.0*z*z*y+250.0*y-3125.0*ONESIXTH*y*y*y*y-4687.5*z*z*y*y;
  gphi[139] = 1562.5*y*x*x*x+3125.0*y*y*x*x+1562.5*y*y*y*x+6250.0*z*y*x*x+6250.0*z*y*y*x+4687.5*z*z*y*x-2812.5*y*x*x-2812.5*y*y*x-5625.0*z*y*x+1250.0*y*x;
  gphi[140] = -1562.5*y*x*x*x-1562.5*y*y*x*x-3125.0*z*y*x*x+1875.0*y*x*x+312.5*y*y*x+625.0*z*y*x-312.5*y*x;
  gphi[141] = 3125.0*ONESIXTH*y*x*x*x-312.5*y*x*x+125.0*ONETHIRD*y*x;
  gphi[142] = 3125.0*0.25*y*y*x*x+1562.5*y*y*y*x+3125.0*z*y*y*x-625.0*0.25*y*x*x-6875.0*0.25*y*y*x-625.0*z*y*x+1125.0*0.25*y*x+3625.0*0.25*y*y+562.5*z*y+3125.0*z*y*y*y-3437.5*z*y*y-1562.5*y*y*y-1875.0*0.25*z*z*y-125.0*y+3125.0*0.25*y*y*y*y+9375.0*0.25*z*z*y*y;
  gphi[143] = -1562.5*y*y*x*x-1562.5*y*y*y*x-3125.0*z*y*y*x+312.5*y*x*x+1875.0*y*y*x+625.0*z*y*x-312.5*y*x;
  gphi[144] = 3125.0*0.25*y*y*x*x-625.0*0.25*y*x*x-625.0*0.25*y*y*x+125.0*0.25*y*x;
  gphi[145] = -3125.0*ONESIXTH*y*y*y*x+312.5*y*y*x-125.0*ONETHIRD*y*x-2125.0*ONESIXTH*y*y-250.0*ONETHIRD*z*y-3125.0*ONETHIRD*z*y*y*y+625.0*z*y*y+2500.0*ONETHIRD*y*y*y+125.0*ONETHIRD*y-3125.0*ONESIXTH*y*y*y*y;
  gphi[146] = 3125.0*ONESIXTH*y*y*y*x-312.5*y*y*x+125.0*ONETHIRD*y*x;
  gphi[147] = -625.0*0.25*y*y*y+1375.0/24.0*y*y-25.0*0.25*y+3125.0/24.0*y*y*y*y;
  gphi[148] = -25.0+2675.0*ONESIXTH*z+1175.0/12.0*x+1175.0/12.0*y-1562.5*z*y*x*x-1562.5*z*y*y*x-4687.5*z*z*y*x+3125.0*z*y*x-125.0*x*x+16875.0*0.25*z*z*y-3125.0*z*z*z*x-125.0*y*y+625.0/12.0*y*y*y+625.0/12.0*x*x*x-3125.0*ONESIXTH*z*x*x*x-3125.0*ONESIXTH*z*y*y*y-9375.0*0.25*z*z*x*x+16875.0*0.25*z*z*x-8875.0*ONESIXTH*z*x-9375.0*0.25*z*z*y*y-7375.0*0.25*z*z+8125.0*ONETHIRD*z*z*z-8875.0*ONESIXTH*z*y-15625.0/12.0*z*z*z*z+625.0*0.25*y*x*x+1562.5*z*y*y+1562.5*z*x*x-3125.0*z*z*z*y+625.0*0.25*y*y*x-250.0*y*x;
  gphi[149] = -312.5*y*x*x-625.0*0.25*y*y*x+1125.0*0.25*y*x+4687.5*z*z*y*x+1562.5*z*y*y*x+3125.0*z*y*x*x-3437.5*z*y*x+1562.5*z*x*x*x-625.0*0.25*x*x*x+4687.5*z*z*x*x+3125.0*z*z*z*x-125.0*x-3437.5*z*x*x-4687.5*z*z*x+1125.0*0.25*x*x+1812.5*z*x;
  gphi[150] = 625.0*0.25*y*x*x-125.0*0.25*y*x-1562.5*z*y*x*x+312.5*z*y*x-1562.5*z*x*x*x+625.0*0.25*x*x*x-9375.0*0.25*z*z*x*x+125.0*0.25*x+2187.5*z*x*x+1875.0*0.25*z*z*x-187.5*x*x-375.0*z*x;
  gphi[151] = 3125.0*ONESIXTH*z*x*x*x-625.0/12.0*x*x*x-25.0*ONESIXTH*x-312.5*z*x*x+125.0*0.25*x*x+125.0*ONETHIRD*z*x;
  gphi[152] = -625.0*0.25*y*x*x-312.5*y*y*x+1125.0*0.25*y*x+4687.5*z*z*y*x+3125.0*z*y*y*x+1562.5*z*y*x*x-3437.5*z*y*x+1125.0*0.25*y*y+1812.5*z*y+1562.5*z*y*y*y-3437.5*z*y*y-625.0*0.25*y*y*y+4687.5*z*z*y*y+3125.0*z*z*z*y-125.0*y-4687.5*z*z*y;
  gphi[153] = 312.5*y*x*x+312.5*y*y*x-312.5*y*x-4687.5*z*z*y*x-3125.0*z*y*y*x-3125.0*z*y*x*x+3750.0*z*y*x;
  gphi[154] = -625.0*0.25*y*x*x+125.0*0.25*y*x+1562.5*z*y*x*x-312.5*z*y*x;
  gphi[155] = 625.0*0.25*y*y*x-125.0*0.25*y*x-1562.5*z*y*y*x+312.5*z*y*x-187.5*y*y-375.0*z*y-1562.5*z*y*y*y+2187.5*z*y*y+625.0*0.25*y*y*y-9375.0*0.25*z*z*y*y+125.0*0.25*y+1875.0*0.25*z*z*y;
  gphi[156] = -625.0*0.25*y*y*x+125.0*0.25*y*x+1562.5*z*y*y*x-312.5*z*y*x;
  gphi[157] = 125.0*0.25*y*y+125.0*ONETHIRD*z*y+3125.0*ONESIXTH*z*y*y*y-312.5*z*y*y-625.0/12.0*y*y*y-25.0*ONESIXTH*y;
  gphi[158] = 50.0*ONETHIRD+6250.0*ONETHIRD*z*z*z*y+125.0*ONETHIRD*y*x+1562.5*z*z*y*x-625.0*z*y*x-312.5*z*x*x-9375.0*0.25*z*z*x+125.0*ONESIXTH*y*y+3875.0*ONESIXTH*z*y+125.0*ONESIXTH*x*x+3875.0*ONESIXTH*z*x+3125.0*0.25*z*z*x*x-312.5*z*y*y-37.5*x-325.0*z+6125.0*0.25*z*z-2500.0*z*z*z+6250.0*ONETHIRD*z*z*z*x+15625.0/12.0*z*z*z*z+3125.0*0.25*z*z*y*y-37.5*y-9375.0*0.25*z*z*y;
  gphi[159] = -125.0*ONETHIRD*y*x-1562.5*z*z*y*x+625.0*z*y*x+625.0*z*x*x+2500.0*z*z*x-125.0*ONETHIRD*x*x-2125.0*ONETHIRD*z*x-1562.5*z*z*x*x+125.0*ONETHIRD*x-6250.0*ONETHIRD*z*z*z*x;
  gphi[160] = -312.5*z*x*x-625.0*0.25*z*z*x+125.0*ONESIXTH*x*x+62.5*z*x+3125.0*0.25*z*z*x*x-25.0*ONESIXTH*x;
  gphi[161] = -6250.0*ONETHIRD*z*z*z*y-125.0*ONETHIRD*y*x-1562.5*z*z*y*x+625.0*z*y*x-125.0*ONETHIRD*y*y-2125.0*ONETHIRD*z*y+625.0*z*y*y-1562.5*z*z*y*y+125.0*ONETHIRD*y+2500.0*z*z*y;
  gphi[162] = 125.0*ONETHIRD*y*x+1562.5*z*z*y*x-625.0*z*y*x;
  gphi[163] = 125.0*ONESIXTH*y*y+62.5*z*y-312.5*z*y*y+3125.0*0.25*z*z*y*y-25.0*ONESIXTH*y-625.0*0.25*z*z*y;
  gphi[164] = -25.0*0.25-3125.0*ONESIXTH*z*z*z*x-3125.0*ONESIXTH*z*z*z*y-1375.0/12.0*z*y+1875.0*0.25*z*z*x-1375.0/12.0*z*x+25.0*0.25*x+1525.0/12.0*z-5125.0/8.0*z*z+25.0*0.25*y+6875.0*ONESIXTH*z*z*z-15625.0/24.0*z*z*z*z+1875.0*0.25*z*z*y;
  gphi[165] = 3125.0*ONESIXTH*z*z*z*x-1875.0*0.25*z*z*x+1375.0/12.0*z*x-25.0*0.25*x;
  gphi[166] = 3125.0*ONESIXTH*z*z*z*y+1375.0/12.0*z*y-25.0*0.25*y-1875.0*0.25*z*z*y;
  gphi[167] = 1.0-125.0*ONESIXTH*z+875.0/8.0*z*z-625.0*ONETHIRD*z*z*z+3125.0/24.0*z*z*z*z;
    return 0;
#endif    
  default:
    return -1;
  }
}




/******************************************************************/
//   FUNCTION Definition: PXGradientsQuadUniformLagrange2d
template <typename DT> ELVIS_DEVICE int
PXGradientsHexUniformLagrange3d(const int porder, const DT * RESTRICT xref, DT * RESTRICT gphi)
{ 
  int i,j,k,l;
  DT  phi_i[6]; // 1d uniform lagrange basis functions
  DT  phi_j[6]; // 1d uniform lagrange basis functions
  DT  phi_k[6]; // 1d uniform lagrange basis functions
  DT gphi_i[6]; // gradient of 1d uniform lagrange basis functions
  DT gphi_j[6]; // gradient of 1d uniform lagrange basis functions
  DT gphi_k[6]; // gradient of 1d uniform lagrange basis functions

  ( PXShapeUniformLagrange1d<DT>( porder, xref  , phi_i ) );
  ( PXShapeUniformLagrange1d<DT>( porder, xref+1, phi_j ) );
  ( PXShapeUniformLagrange1d<DT>( porder, xref+2, phi_k ) );

  ( PXGradientsUniformLagrange1d<DT>( porder, xref  , gphi_i ) );
  ( PXGradientsUniformLagrange1d<DT>( porder, xref+1, gphi_j ) );
  ( PXGradientsUniformLagrange1d<DT>( porder, xref+2, gphi_k ) );
  
  /* compute dphi/dx */
  l = 0;
  for (k=0; k<(porder+1); k++){
    for (j=0; j<(porder+1); j++){
      for (i=0; i<(porder+1); i++){
        gphi[l] = gphi_i[i]*phi_j[j]*phi_k[k];
        l++;
      }
    }
  }

  /* compute dphi/dy */
  for (k=0; k<(porder+1); k++){
    for (j=0; j<(porder+1); j++){
      for (i=0; i<(porder+1); i++){
        gphi[l] = phi_i[i]*gphi_j[j]*phi_k[k];
        l++;
      }
    }
  }

  /* compute dphi/dy */
  for (k=0; k<(porder+1); k++){
    for (j=0; j<(porder+1); j++){
      for (i=0; i<(porder+1); i++){
        gphi[l] = phi_i[i]*phi_j[j]*gphi_k[k];
        l++;
      }
    }
  }

  return 0;
}



/******************************************************************/
//   FUNCTION Definition: PXGradientsQuadSpectralLagrange3d
template <typename DT> ELVIS_DEVICE int
PXGradientsHexSpectralLagrange3d(const int porder, const DT * RESTRICT xref, DT * RESTRICT gphi)
{ 
  int i,j,k,l;
  DT  phi_i[6]; // 1d spectral lagrange basis functions
  DT  phi_j[6]; // 1d spectral lagrange basis functions
  DT  phi_k[6]; // 1d spectral lagrange basis functions
  DT gphi_i[6]; // gradient of 1d spectral lagrange basis functions
  DT gphi_j[6]; // gradient of 1d spectral lagrange basis functions
  DT gphi_k[6]; // gradient of 1d spectral lagrange basis functions

  ( PXShapeSpectralLagrange1d<DT>( porder, xref  , phi_i ) );
  ( PXShapeSpectralLagrange1d<DT>( porder, xref+1, phi_j ) );
  ( PXShapeSpectralLagrange1d<DT>( porder, xref+2, phi_k ) );

  ( PXGradientsSpectralLagrange1d<DT>( porder, xref  , gphi_i ) );
  ( PXGradientsSpectralLagrange1d<DT>( porder, xref+1, gphi_j ) );
  ( PXGradientsSpectralLagrange1d<DT>( porder, xref+2, gphi_k ) );
  
  /* compute dphi/dx */
  l = 0;
  for (k=0; k<(porder+1); k++){
    for (j=0; j<(porder+1); j++){
      for (i=0; i<(porder+1); i++){
        gphi[l] = gphi_i[i]*phi_j[j]*phi_k[k];
        l++;
      }
    }
  }

  /* compute dphi/dy */
  for (k=0; k<(porder+1); k++){
    for (j=0; j<(porder+1); j++){
      for (i=0; i<(porder+1); i++){
        gphi[l] = phi_i[i]*gphi_j[j]*phi_k[k];
        l++;
      }
    }
  }

  /* compute dphi/dy */
  for (k=0; k<(porder+1); k++){
    for (j=0; j<(porder+1); j++){
      for (i=0; i<(porder+1); i++){
        gphi[l] = phi_i[i]*phi_j[j]*gphi_k[k];
        l++;
      }
    }
  }

  return 0;
}


/******************************************************************/
//   FUNCTION Definition: PXShape2
/* template <typename DT> ELVIS_DEVICE int */
/* PXShape2(enum PXE_SolutionOrder order, int porder, DT const * RESTRICT xref, DT * RESTRICT phi) */
/* { */
/*   int ierr;      // error code */
/*   //int porder;    // polynomial interpolation order */

/*   if ( unlikely(order==PXE_OrderDummy) ){ */
/*     phi[0] = 1; */
/*     return PX_NO_ERROR; */
/*   } */

/*   /\* get porder *\/ */
/*   //PXErrorReturn( PXOrder2porder(order, &porder) ); */

/*   /\****************************************************************************************\/ */
/*   /\*                             Switch over Order                                        *\/ */
/*   /\****************************************************************************************\/ */
/*   switch (order) { */
    
/*   case PXE_Lagrange1dP0: */
/*   case PXE_Lagrange1dP1: */
/*   case PXE_Lagrange1dP2: */
/*   case PXE_Lagrange1dP3: */
/*   case PXE_Lagrange1dP4: */
/*   case PXE_Lagrange1dP5: */
/*     ( PXShapeUniformLagrange1d<DT>(porder, xref, phi) ); */
/*     return PX_NO_ERROR; */
/*   case PXE_SpectralLagrange1dP0: */
/*   case PXE_SpectralLagrange1dP1: */
/*   case PXE_SpectralLagrange1dP2: */
/*   case PXE_SpectralLagrange1dP3: */
/*   case PXE_SpectralLagrange1dP4: */
/*   case PXE_SpectralLagrange1dP5: */
/*     ( PXShapeSpectralLagrange1d<DT>(porder, xref, phi) ); */
/*     return PX_NO_ERROR; */
/*   /\* case PXE_Hierarch1dP1: *\/ */
/*   /\* case PXE_Hierarch1dP2: *\/ */
/*   /\* case PXE_Hierarch1dP3: *\/ */
/*   /\* case PXE_Hierarch1dP4: *\/ */
/*   /\* case PXE_Hierarch1dP5: *\/ */
/*   /\*   ( PXShapeHierarch1d(porder, xref, phi) ); *\/ */
/*   /\*   return PX_NO_ERROR; *\/ */
/*   case PXE_LagrangeP0: */
/*   case PXE_LagrangeP1: */
/*   case PXE_LagrangeP2:  */
/*   case PXE_LagrangeP3: */
/*   case PXE_LagrangeP4: */
/*   case PXE_LagrangeP5: */
/*     ( PXShapeLagrange2d<DT>(porder, xref, phi) ); */
/*     return PX_NO_ERROR; */
/*   /\* case PXE_HierarchP1: *\/ */
/*   /\* case PXE_HierarchP2: *\/ */
/*   /\* case PXE_HierarchP3:  *\/ */
/*   /\* case PXE_HierarchP4:  *\/ */
/*   /\* case PXE_HierarchP5:  *\/ */
/*   /\*   ( PXShapeHierarch2d(porder, xref, phi) ); *\/ */
/*   /\*   return PX_NO_ERROR; *\/ */
/*   case PXE_QuadUniformLagrangeP0: */
/*   case PXE_QuadUniformLagrangeP1: */
/*   case PXE_QuadUniformLagrangeP2:  */
/*   case PXE_QuadUniformLagrangeP3: */
/*   case PXE_QuadUniformLagrangeP4: */
/*   case PXE_QuadUniformLagrangeP5: */
/*     ( PXShapeQuadUniformLagrange2d<DT>(porder, xref, phi) ); */
/*     return PX_NO_ERROR; */
/*   case PXE_QuadSpectralLagrangeP0: */
/*   case PXE_QuadSpectralLagrangeP1: */
/*   case PXE_QuadSpectralLagrangeP2:  */
/*   case PXE_QuadSpectralLagrangeP3: */
/*   case PXE_QuadSpectralLagrangeP4: */
/*   case PXE_QuadSpectralLagrangeP5: */
/*     ( PXShapeQuadSpectralLagrange2d<DT>(porder, xref, phi) ); */
/*     return PX_NO_ERROR; */
/*   case PXE_Lagrange3dP0: */
/*   case PXE_Lagrange3dP1:  */
/*   case PXE_Lagrange3dP2:  */
/*   case PXE_Lagrange3dP3: */
/*   case PXE_Lagrange3dP4:  */
/*   case PXE_Lagrange3dP5:  */
/*       ( PXShapeLagrange3d<DT>(porder, xref, phi) ); */
/*       return PX_NO_ERROR; */
/*   /\* case PXE_Hierarch3dP1:  *\/ */
/*   /\* case PXE_Hierarch3dP2:  *\/ */
/*   /\* case PXE_Hierarch3dP3:  *\/ */
/*   /\* case PXE_Hierarch3dP4:  *\/ */
/*   /\* case PXE_Hierarch3dP5: *\/ */
/*   /\*     ( PXShapeHierarch3d(porder, xref, phi) ); *\/ */
/*   /\*     return PX_NO_ERROR; *\/ */
/*   case PXE_HexUniformLagrangeP0: */
/*   case PXE_HexUniformLagrangeP1: */
/*   case PXE_HexUniformLagrangeP2:  */
/*   case PXE_HexUniformLagrangeP3: */
/*   case PXE_HexUniformLagrangeP4: */
/*   case PXE_HexUniformLagrangeP5: */
/*     ( PXShapeHexUniformLagrange3d<DT>(porder, xref, phi) ); */
/*     return PX_NO_ERROR; */
/*   case PXE_HexSpectralLagrangeP0: */
/*   case PXE_HexSpectralLagrangeP1: */
/*   case PXE_HexSpectralLagrangeP2:  */
/*   case PXE_HexSpectralLagrangeP3: */
/*   case PXE_HexSpectralLagrangeP4: */
/*   case PXE_HexSpectralLagrangeP5: */
/*     ( PXShapeHexSpectralLagrange3d<DT>(porder, xref, phi) ); */
/*     return PX_NO_ERROR; */
/*   default: */
/*     //ELVIS_PRINTF("Unknown order = %d\n", order); */
/*     return PXErrorDebug(PX_BAD_INPUT); */
/*   } */
 
/* } */


/******************************************************************/
//   FUNCTION Definition: PXShapeElem
template <typename DT> ELVIS_DEVICE int
PXShapeElem(enum PXE_SolutionOrder order, int porder, DT const * RESTRICT xref, DT * RESTRICT phi)
{
  //int porder;    // polynomial interpolation order

  if ( unlikely(order==PXE_OrderDummy) ){
    phi[0] = 1;
    return PX_NO_ERROR;
  }

  /* get porder */
  //( PXOrder2porder(order, &porder) );

  /****************************************************************************************/
  /*                             Switch over Order                                        */
  /****************************************************************************************/
  switch (order) {   
  case PXE_LagrangeP0:
  case PXE_LagrangeP1:
  case PXE_LagrangeP2:
  case PXE_LagrangeP3:
  case PXE_LagrangeP4:
  case PXE_LagrangeP5:
      ( PXShapeLagrange2d<DT>(porder, xref, phi) );
      return PX_NO_ERROR;
  case PXE_HierarchP1:
  case PXE_HierarchP2:
  case PXE_HierarchP3:
  case PXE_HierarchP4:
  case PXE_HierarchP5:
      ( PXShapeHierarch2d(porder, xref, phi) );
      return PX_NO_ERROR;
  case PXE_Lagrange3dP0:
  case PXE_Lagrange3dP1: 
  case PXE_Lagrange3dP2: 
  case PXE_Lagrange3dP3:
  case PXE_Lagrange3dP4: 
  case PXE_Lagrange3dP5: 
      ( PXShapeLagrange3d<DT>(porder, xref, phi) );
      return PX_NO_ERROR;
  /* case PXE_Hierarch3dP1:  */
  /* case PXE_Hierarch3dP2:  */
  /* case PXE_Hierarch3dP3:  */
  /* case PXE_Hierarch3dP4:  */
  /* case PXE_Hierarch3dP5: */
  /*     ( PXShapeHierarch3d(porder, xref, phi) ); */
  /*     return PX_NO_ERROR; */
  case PXE_HexUniformLagrangeP0:
  case PXE_HexUniformLagrangeP1:
  case PXE_HexUniformLagrangeP2: 
  case PXE_HexUniformLagrangeP3:
  case PXE_HexUniformLagrangeP4:
  case PXE_HexUniformLagrangeP5:
    ( PXShapeHexUniformLagrange3d<DT>(porder, xref, phi) );
    return PX_NO_ERROR;
  case PXE_HexSpectralLagrangeP0:
  case PXE_HexSpectralLagrangeP1:
  case PXE_HexSpectralLagrangeP2: 
  case PXE_HexSpectralLagrangeP3:
  case PXE_HexSpectralLagrangeP4:
  case PXE_HexSpectralLagrangeP5:
    ( PXShapeHexSpectralLagrange3d<DT>(porder, xref, phi) );
    return PX_NO_ERROR;
  default:
    ELVIS_PRINTF("PXShapeElem: Unknown order = %d\n", order);
    return PXErrorDebug(PX_BAD_INPUT);
  }
 
}


/******************************************************************/
//   FUNCTION Definition: PXShapeElem_Solution
template <typename DT> ELVIS_DEVICE int
PXShapeElem_Solution(enum PXE_SolutionOrder order, int porder, DT const * RESTRICT xref, DT * RESTRICT phi)
{
  //int porder;    // polynomial interpolation order

  if ( unlikely(order==PXE_OrderDummy) ){
    phi[0] = 1;
    return PX_NO_ERROR;
  }

  /* get porder */
  //( PXOrder2porder(order, &porder) );

  /****************************************************************************************/
  /*                             Switch over Order                                        */
  /****************************************************************************************/
  switch (order) {
  case PXE_LagrangeP0:
  case PXE_LagrangeP1:
  case PXE_LagrangeP2:
  case PXE_LagrangeP3:
  case PXE_LagrangeP4:
  case PXE_LagrangeP5:
      ( PXShapeLagrange2d<DT>(porder, xref, phi) );
      return PX_NO_ERROR;
   case PXE_HierarchP1:
   case PXE_HierarchP2:
   case PXE_HierarchP3:
   case PXE_HierarchP4:
   case PXE_HierarchP5:
       ( PXShapeHierarch2d(porder, xref, phi) );
       return PX_NO_ERROR;
  case PXE_Lagrange3dP0:
  case PXE_Lagrange3dP1: 
  case PXE_Lagrange3dP2: 
  case PXE_Lagrange3dP3:
  case PXE_Lagrange3dP4: 
  case PXE_Lagrange3dP5: 
      ( PXShapeLagrange3d_Solution<DT>(porder, xref, phi) );
      return PX_NO_ERROR;
  /* case PXE_Hierarch3dP1:  */
  /* case PXE_Hierarch3dP2:  */
  /* case PXE_Hierarch3dP3:  */
  /* case PXE_Hierarch3dP4:  */
  /* case PXE_Hierarch3dP5: */
  /*     ( PXShapeHierarch3d(porder, xref, phi) ); */
  /*     return PX_NO_ERROR; */
  case PXE_HexUniformLagrangeP0:
  case PXE_HexUniformLagrangeP1:
  case PXE_HexUniformLagrangeP2:
  case PXE_HexUniformLagrangeP3:
  case PXE_HexUniformLagrangeP4:
  case PXE_HexUniformLagrangeP5:
    ( PXShapeHexUniformLagrange3d<DT>(porder, xref, phi) );
     return PX_NO_ERROR;
  case PXE_HexSpectralLagrangeP0:
  case PXE_HexSpectralLagrangeP1:
  case PXE_HexSpectralLagrangeP2:
  case PXE_HexSpectralLagrangeP3:
  case PXE_HexSpectralLagrangeP4:
  case PXE_HexSpectralLagrangeP5:
    ( PXShapeHexSpectralLagrange3d<DT>(porder, xref, phi) );
    return PX_NO_ERROR;
  default:
    ELVIS_PRINTF("PXShapeElem_Solution: Unknown order = %d\n", order);
    return PXErrorDebug(PX_BAD_INPUT);
  }
 
}


/******************************************************************/
//   FUNCTION Definition: PXShape2
template <typename DT> ELVIS_DEVICE int
PXShapeFace(enum PXE_SolutionOrder order, int porder, DT const * RESTRICT xref, DT * RESTRICT phi)
{
  //int porder;    // polynomial interpolation order

  if ( unlikely(order==PXE_OrderDummy) ){
    phi[0] = 1;
    return PX_NO_ERROR;
  }

  /* get porder */
  //( PXOrder2porder(order, &porder) );

  /****************************************************************************************/
  /*                             Switch over Order                                        */
  /****************************************************************************************/
  switch (order) {
    
  case PXE_Lagrange1dP0:
  case PXE_Lagrange1dP1:
  case PXE_Lagrange1dP2:
  case PXE_Lagrange1dP3:
  case PXE_Lagrange1dP4:
  case PXE_Lagrange1dP5:
    ( PXShapeUniformLagrange1d<DT>(porder, xref, phi) );
    return PX_NO_ERROR;
  case PXE_SpectralLagrange1dP0:
  case PXE_SpectralLagrange1dP1:
  case PXE_SpectralLagrange1dP2:
  case PXE_SpectralLagrange1dP3:
  case PXE_SpectralLagrange1dP4:
  case PXE_SpectralLagrange1dP5:
    ( PXShapeSpectralLagrange1d<DT>(porder, xref, phi) );
    return PX_NO_ERROR;
   case PXE_Hierarch1dP1:
   case PXE_Hierarch1dP2:
   case PXE_Hierarch1dP3:
   case PXE_Hierarch1dP4:
   case PXE_Hierarch1dP5:
     ( PXShapeHierarch1d(porder, xref, phi) );
     return PX_NO_ERROR;
  case PXE_LagrangeP0:
  case PXE_LagrangeP1:
  case PXE_LagrangeP2: 
  case PXE_LagrangeP3:
  case PXE_LagrangeP4:
  case PXE_LagrangeP5:
    ( PXShapeLagrange2d<DT>(porder, xref, phi) );
    return PX_NO_ERROR;
   case PXE_HierarchP1:
   case PXE_HierarchP2:
   case PXE_HierarchP3:
   case PXE_HierarchP4:
   case PXE_HierarchP5:
     ( PXShapeHierarch2d(porder, xref, phi) );
     return PX_NO_ERROR;
  case PXE_QuadUniformLagrangeP0:
  case PXE_QuadUniformLagrangeP1:
  case PXE_QuadUniformLagrangeP2: 
  case PXE_QuadUniformLagrangeP3:
  case PXE_QuadUniformLagrangeP4:
  case PXE_QuadUniformLagrangeP5:
    ( PXShapeQuadUniformLagrange2d<DT>(porder, xref, phi) );
    return PX_NO_ERROR;
  case PXE_QuadSpectralLagrangeP0:
  case PXE_QuadSpectralLagrangeP1:
  case PXE_QuadSpectralLagrangeP2: 
  case PXE_QuadSpectralLagrangeP3:
  case PXE_QuadSpectralLagrangeP4:
  case PXE_QuadSpectralLagrangeP5:
    ( PXShapeQuadSpectralLagrange2d<DT>(porder, xref, phi) );
    return PX_NO_ERROR;
  default:
    ALWAYS_PRINTF("PXShapeFace: Unknown order = %d\n", order);
    return PXErrorDebug(PX_BAD_INPUT);
  }
 
}

/******************************************************************/
//   FUNCTION Definition: PXGradients2
/* template <typename DT> ELVIS_DEVICE int */
/* PXGradients2(enum PXE_SolutionOrder order, int porder, DT const * RESTRICT xref, DT * RESTRICT gphi) */
/* { */
/*   int ierr;      // error code */
/*   //int porder;    // polynomial interpolation order */

/*   /\* get porder *\/ */
/*   //( PXOrder2porder(order, &porder) ); */

/*   /\****************************************************************************************\/ */
/*   /\*                             Switch over Order                                        *\/ */
/*   /\****************************************************************************************\/ */
/*   switch (order) { */
    
/*   case PXE_Lagrange1dP0: */
/*   case PXE_Lagrange1dP1: */
/*   case PXE_Lagrange1dP2: */
/*   case PXE_Lagrange1dP3: */
/*   case PXE_Lagrange1dP4: */
/*   case PXE_Lagrange1dP5: */
/*     ( PXGradientsUniformLagrange1d<DT>(porder, xref, gphi) ); */
/*     return PX_NO_ERROR; */
/*   case PXE_SpectralLagrange1dP0: */
/*   case PXE_SpectralLagrange1dP1: */
/*   case PXE_SpectralLagrange1dP2: */
/*   case PXE_SpectralLagrange1dP3: */
/*   case PXE_SpectralLagrange1dP4: */
/*   case PXE_SpectralLagrange1dP5: */
/*     ( PXGradientsSpectralLagrange1d<DT>(porder, xref, gphi) ); */
/*     return PX_NO_ERROR; */
/*   /\* case PXE_Hierarch1dP1: *\/ */
/*   /\* case PXE_Hierarch1dP2: *\/ */
/*   /\* case PXE_Hierarch1dP3: *\/ */
/*   /\* case PXE_Hierarch1dP4: *\/ */
/*   /\* case PXE_Hierarch1dP5: *\/ */
/*   /\*   ( PXGradientsHierarch1d(porder, xref, gphi) ); *\/ */
/*   /\*   return PX_NO_ERROR; *\/ */
/*   case PXE_LagrangeP0: */
/*   case PXE_LagrangeP1: */
/*   case PXE_LagrangeP2:  */
/*   case PXE_LagrangeP3: */
/*   case PXE_LagrangeP4: */
/*   case PXE_LagrangeP5: */
/*     ( PXGradientsLagrange2d<DT>(porder, xref, gphi) ); */
/*     return PX_NO_ERROR; */
/*   /\* case PXE_HierarchP1: *\/ */
/*   /\* case PXE_HierarchP2: *\/ */
/*   /\* case PXE_HierarchP3:  *\/ */
/*   /\* case PXE_HierarchP4:  *\/ */
/*   /\* case PXE_HierarchP5:  *\/ */
/*   /\*   ( PXGradientsHierarch2d(porder, xref, gphi) ); *\/ */
/*   /\*   return PX_NO_ERROR; *\/ */
/*   case PXE_QuadUniformLagrangeP0: */
/*   case PXE_QuadUniformLagrangeP1: */
/*   case PXE_QuadUniformLagrangeP2:  */
/*   case PXE_QuadUniformLagrangeP3: */
/*   case PXE_QuadUniformLagrangeP4: */
/*   case PXE_QuadUniformLagrangeP5: */
/*     ( PXGradientsQuadUniformLagrange2d<DT>(porder, xref, gphi) ); */
/*     return PX_NO_ERROR; */
/*   case PXE_QuadSpectralLagrangeP0: */
/*   case PXE_QuadSpectralLagrangeP1: */
/*   case PXE_QuadSpectralLagrangeP2:  */
/*   case PXE_QuadSpectralLagrangeP3: */
/*   case PXE_QuadSpectralLagrangeP4: */
/*   case PXE_QuadSpectralLagrangeP5: */
/*     ( PXGradientsQuadSpectralLagrange2d<DT>(porder, xref, gphi) ); */
/*     return PX_NO_ERROR; */
/*   case PXE_Lagrange3dP0: */
/*   case PXE_Lagrange3dP1:  */
/*   case PXE_Lagrange3dP2:  */
/*   case PXE_Lagrange3dP3: */
/*   case PXE_Lagrange3dP4:  */
/*   case PXE_Lagrange3dP5:  */
/*       ( PXGradientsLagrange3d<DT>(porder, xref, gphi) ); */
/*       return PX_NO_ERROR; */
/*   /\* case PXE_Hierarch3dP1:  *\/ */
/*   /\* case PXE_Hierarch3dP2:  *\/ */
/*   /\* case PXE_Hierarch3dP3:  *\/ */
/*   /\* case PXE_Hierarch3dP4:  *\/ */
/*   /\* case PXE_Hierarch3dP5: *\/ */
/*   /\*     ( PXGradientsHierarch3d(porder, xref, gphi) ); *\/ */
/*   /\*     return PX_NO_ERROR; *\/ */
/*   case PXE_HexUniformLagrangeP0: */
/*   case PXE_HexUniformLagrangeP1: */
/*   case PXE_HexUniformLagrangeP2:  */
/*   case PXE_HexUniformLagrangeP3: */
/*   case PXE_HexUniformLagrangeP4: */
/*   case PXE_HexUniformLagrangeP5: */
/*     ( PXGradientsHexUniformLagrange3d<DT>(porder, xref, gphi) ); */
/*     return PX_NO_ERROR; */
/*   case PXE_HexSpectralLagrangeP0: */
/*   case PXE_HexSpectralLagrangeP1: */
/*   case PXE_HexSpectralLagrangeP2:  */
/*   case PXE_HexSpectralLagrangeP3: */
/*   case PXE_HexSpectralLagrangeP4: */
/*   case PXE_HexSpectralLagrangeP5: */
/*     ( PXGradientsHexSpectralLagrange3d<DT>(porder, xref, gphi) ); */
/*     return PX_NO_ERROR; */
/*    default: */
/*      //ELVIS_PRINTF("Unknown order = %d\n", order); */
/*     return PXErrorDebug(PX_BAD_INPUT); */
/*   } */

/* } */

/******************************************************************/
//   FUNCTION Definition: PXGradientsElem
template <typename DT> ELVIS_DEVICE int
PXGradientsElem(enum PXE_SolutionOrder order, int porder, DT const * RESTRICT xref, DT * RESTRICT gphi)
{
  //int porder;    // polynomial interpolation order

  /* get porder */
  //( PXOrder2porder(order, &porder) );

  /****************************************************************************************/
  /*                             Switch over Order                                        */
  /****************************************************************************************/
  switch (order) {
  case PXE_LagrangeP0:
  case PXE_LagrangeP1:
  case PXE_LagrangeP2:
  case PXE_LagrangeP3:
  case PXE_LagrangeP4:
  case PXE_LagrangeP5:
    ( PXGradientsLagrange2d<DT>(porder, xref, gphi) );
    return PX_NO_ERROR;
  case PXE_QuadUniformLagrangeP0:
  case PXE_QuadUniformLagrangeP1:
  case PXE_QuadUniformLagrangeP2:
  case PXE_QuadUniformLagrangeP3:
  case PXE_QuadUniformLagrangeP4:
  case PXE_QuadUniformLagrangeP5:
    ( PXGradientsQuadUniformLagrange2d<DT>(porder, xref, gphi) );
    return PX_NO_ERROR;
  case PXE_QuadSpectralLagrangeP0:
  case PXE_QuadSpectralLagrangeP1:
  case PXE_QuadSpectralLagrangeP2:
  case PXE_QuadSpectralLagrangeP3:
  case PXE_QuadSpectralLagrangeP4:
  case PXE_QuadSpectralLagrangeP5:
    ( PXGradientsQuadSpectralLagrange2d<DT>(porder, xref, gphi) );
    return PX_NO_ERROR;
  case PXE_Lagrange3dP0:
  case PXE_Lagrange3dP1: 
  case PXE_Lagrange3dP2: 
  case PXE_Lagrange3dP3:
  case PXE_Lagrange3dP4: 
  case PXE_Lagrange3dP5: 
      ( PXGradientsLagrange3d<DT>(porder, xref, gphi) );
      return PX_NO_ERROR;
   case PXE_Hierarch3dP1:
   case PXE_Hierarch3dP2:
   case PXE_Hierarch3dP3:
   case PXE_Hierarch3dP4:
   case PXE_Hierarch3dP5:
       ( PXGradientsHierarch3d(porder, xref, gphi) );
       return PX_NO_ERROR;
  case PXE_HexUniformLagrangeP0:
  case PXE_HexUniformLagrangeP1:
  case PXE_HexUniformLagrangeP2: 
  case PXE_HexUniformLagrangeP3:
  case PXE_HexUniformLagrangeP4:
  case PXE_HexUniformLagrangeP5:
    ( PXGradientsHexUniformLagrange3d<DT>(porder, xref, gphi) );
    return PX_NO_ERROR;
  case PXE_HexSpectralLagrangeP0:
  case PXE_HexSpectralLagrangeP1:
  case PXE_HexSpectralLagrangeP2: 
  case PXE_HexSpectralLagrangeP3:
  case PXE_HexSpectralLagrangeP4:
  case PXE_HexSpectralLagrangeP5:
    ( PXGradientsHexSpectralLagrange3d<DT>(porder, xref, gphi) );
    return PX_NO_ERROR;

   default:
     ALWAYS_PRINTF("PXGradientsElem: Unknown order = %d\n", order);
     return PXErrorDebug(PX_BAD_INPUT);
  }

}


/******************************************************************/
//   FUNCTION Definition: PXGradientsElem_Solution
template <typename DT> ELVIS_DEVICE int
PXGradientsElem_Solution(enum PXE_SolutionOrder order, int porder, DT const * RESTRICT xref, DT * RESTRICT gphi)
{
  //int porder;    // polynomial interpolation order

  /* get porder */
  //( PXOrder2porder(order, &porder) );

  /****************************************************************************************/
  /*                             Switch over Order                                        */
  /****************************************************************************************/
  switch (order) {
  case PXE_Lagrange3dP0:
  case PXE_Lagrange3dP1: 
  case PXE_Lagrange3dP2: 
  case PXE_Lagrange3dP3:
  case PXE_Lagrange3dP4: 
  case PXE_Lagrange3dP5: 
      ( PXGradientsLagrange3d_Solution<DT>(porder, xref, gphi) );
      return PX_NO_ERROR;
   case PXE_Hierarch3dP1:
   case PXE_Hierarch3dP2:
   case PXE_Hierarch3dP3:
   case PXE_Hierarch3dP4:
   case PXE_Hierarch3dP5:
       ( PXGradientsHierarch3d(porder, xref, gphi) );
       return PX_NO_ERROR;
   case PXE_HexUniformLagrangeP0:
   case PXE_HexUniformLagrangeP1:
   case PXE_HexUniformLagrangeP2:
   case PXE_HexUniformLagrangeP3:
   case PXE_HexUniformLagrangeP4:
   case PXE_HexUniformLagrangeP5:
     ( PXGradientsHexUniformLagrange3d<DT>(porder, xref, gphi) );
     return PX_NO_ERROR;
   case PXE_HexSpectralLagrangeP0:
   case PXE_HexSpectralLagrangeP1:
   case PXE_HexSpectralLagrangeP2:
   case PXE_HexSpectralLagrangeP3:
   case PXE_HexSpectralLagrangeP4:
   case PXE_HexSpectralLagrangeP5:
     ( PXGradientsHexSpectralLagrange3d<DT>(porder, xref, gphi) );
     return PX_NO_ERROR;
   default:
     ALWAYS_PRINTF("PXGradientsElem_Solution: Unknown order = %d\n", order);
     return PXErrorDebug(PX_BAD_INPUT);
  }

}



/******************************************************************/
//   FUNCTION Definition: PXGradients2
template <typename DT> ELVIS_DEVICE int
PXGradientsFace(enum PXE_SolutionOrder order, int porder, DT const * RESTRICT xref, DT * RESTRICT gphi)
{
  //int porder;    // polynomial interpolation order

  /* get porder */
  //( PXOrder2porder(order, &porder) );

  /****************************************************************************************/
  /*                             Switch over Order                                        */
  /****************************************************************************************/
  switch (order) {
    
  case PXE_Lagrange1dP0:
  case PXE_Lagrange1dP1:
  case PXE_Lagrange1dP2:
  case PXE_Lagrange1dP3:
  case PXE_Lagrange1dP4:
  case PXE_Lagrange1dP5:
    ( PXGradientsUniformLagrange1d<DT>(porder, xref, gphi) );
    return PX_NO_ERROR;
  case PXE_SpectralLagrange1dP0:
  case PXE_SpectralLagrange1dP1:
  case PXE_SpectralLagrange1dP2:
  case PXE_SpectralLagrange1dP3:
  case PXE_SpectralLagrange1dP4:
  case PXE_SpectralLagrange1dP5:
    ( PXGradientsSpectralLagrange1d<DT>(porder, xref, gphi) );
    return PX_NO_ERROR;
  case PXE_Hierarch1dP1:
  case PXE_Hierarch1dP2:
  case PXE_Hierarch1dP3:
  case PXE_Hierarch1dP4:
  case PXE_Hierarch1dP5:
     ( PXGradientsHierarch1d<DT>(porder, xref, gphi) );
     return PX_NO_ERROR;
  case PXE_LagrangeP0:
  case PXE_LagrangeP1:
  case PXE_LagrangeP2: 
  case PXE_LagrangeP3:
  case PXE_LagrangeP4:
  case PXE_LagrangeP5:
    ( PXGradientsLagrange2d<DT>(porder, xref, gphi) );
    return PX_NO_ERROR;
  case PXE_HierarchP1:
  case PXE_HierarchP2:
  case PXE_HierarchP3:
  case PXE_HierarchP4:
  case PXE_HierarchP5:
     ( PXGradientsHierarch2d<DT>(porder, xref, gphi) );
     return PX_NO_ERROR;
  case PXE_QuadUniformLagrangeP0:
  case PXE_QuadUniformLagrangeP1:
  case PXE_QuadUniformLagrangeP2: 
  case PXE_QuadUniformLagrangeP3:
  case PXE_QuadUniformLagrangeP4:
  case PXE_QuadUniformLagrangeP5:
    ( PXGradientsQuadUniformLagrange2d<DT>(porder, xref, gphi) );
    return PX_NO_ERROR;
  case PXE_QuadSpectralLagrangeP0:
  case PXE_QuadSpectralLagrangeP1:
  case PXE_QuadSpectralLagrangeP2: 
  case PXE_QuadSpectralLagrangeP3:
  case PXE_QuadSpectralLagrangeP4:
  case PXE_QuadSpectralLagrangeP5:
    ( PXGradientsQuadSpectralLagrange2d<DT>(porder, xref, gphi) );
    return PX_NO_ERROR;

   default:
    ALWAYS_PRINTF("PXGradientsFace: Unknown order = %d\n", order);
    return PXErrorDebug(PX_BAD_INPUT);
  }

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
//   FUNCTION Definition: PXPhysicalGradients
template <typename DT> ELVIS_DEVICE int 
PXPhysicalGradientsGivenGradients(enum PXE_SolutionOrder order, int nbf, DT const * RESTRICT iJac, DT const * RESTRICT gphi, DT * RESTRICT phix)
{
  int k;  // number of basis functions
  DT const * RESTRICT gphix;
  DT const * RESTRICT gphiy;
  DT const * RESTRICT gphiz;
  DT * RESTRICT phixx;
  DT * RESTRICT phixy;
  DT * RESTRICT phixz;
  /* Get Dimension */
  //PXOrder2Dim(order, &Dim);

  /* Get number of basis functions */
  //( PXOrder2nbf(order, &nbf) );

  /*transform shape gradients to physical space */
  switch(Dim){
  case 1:
    for(k=0; k<nbf; k++){
      phix[k] = iJac[0]*gphi[k];
    }
    return PX_NO_ERROR;
  case 2:
    /* use an unrolled version of the following loop */
    /* idea is to compute a matrix-matrix product C = A*B where
       C is 2xnbf, A is 2x2 (col. major); B is 2xnbf (row maj) */

/*     for(di = 0; di < 2; di++){ */
/*       for(k=0; k<nbf; k++){ */
/* 	phix[di*nbf+k] = 0.0; */
/* 	for(dj=0; dj<2; dj++){ */
/* 	  phix[di*nbf+k] += iJac[dj*2 + di]*gphi[dj*nbf+k]; */
/* 	} */
/*       } */
/*     } */

    /* unrolled */
    gphix = gphi;
    gphiy = gphi + nbf;
    phixx = phix;
    phixy = phix + nbf;

    for(k=0; k<nbf; k++){
      phixx[k] = iJac[0*2 + 0]*gphix[k] + 
	         iJac[1*2 + 0]*gphiy[k];
      phixy[k] = iJac[0*2 + 1]*gphix[k] +
                 iJac[1*2 + 1]*gphiy[k];
    }
    return PX_NO_ERROR;
  case 3:
    gphix = gphi;
    gphiy = gphi + nbf;
    gphiz = gphi + 2*nbf;
    phixx = phix;
    phixy = phix + nbf;
    phixz = phix + 2*nbf;

    for(k=0; k<nbf; k++){
      phixx[k] = iJac[0*3 + 0]*gphix[k] +
                 iJac[1*3 + 0]*gphiy[k] +
                 iJac[2*3 + 0]*gphiz[k];
      phixy[k] = iJac[0*3 + 1]*gphix[k] +
                 iJac[1*3 + 1]*gphiy[k] +
               	 iJac[2*3 + 1]*gphiz[k];
      phixz[k] = iJac[0*3 + 2]*gphix[k] +
                 iJac[1*3 + 2]*gphiy[k] +
                 iJac[2*3 + 2]*gphiz[k];
    }
    return PX_NO_ERROR;
  default:
    return PX_CODE_FLOW_ERROR;
  }
/*   for (k = 0; k < nbf; k++){ */
/*     //for (d = 0; d < Dim; d++) p[d] = gphi[d*nbf+k]; */
/*     PXMat_Vec_Mult_Set_Tr(iJac, gphi, q, Dim, Dim); */

/*     for (d = 0; d < Dim; d++) phix[d*nbf+k] = q[d]; //gphi[d*nbf+k] = q[d]; */
/*     gphi += Dim; */
/*   } */
/*   return PX_NO_ERROR; */
}


/******************************************************************/
//   FUNCTION Definition: LinearSimplexGlob2Ref
ELVIS_DEVICE int
LinearSimplexGlob2Ref(ElVisFloat const * RESTRICT vertices, PX_REAL const * RESTRICT xglobal, PX_REAL * RESTRICT xref)
{
  PX_REAL Jac[9];    // Transformation Jacobian
  //PX_REAL J;         // Jacobian Determinant
  PX_REAL iJac[9];   // Inverse of Jacobian
  ElVisFloat const * x0;       // Coordinates on Node0

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
    return PX_NO_ERROR;
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
    return PX_NO_ERROR;
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
    return PX_NO_ERROR;
  default:
    ALWAYS_PRINTF("Dim = %d not supported in ProjectX\n", Dim);
    return PXErrorDebug(PX_BAD_INPUT);
  }

  //return PX_NO_ERROR;
}



#endif //PXSHAPE_ELVIS_C
