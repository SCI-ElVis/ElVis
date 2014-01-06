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

#ifndef _PX_CUTCELL_ELIVS_C
#define _PX_CUTCELL_ELIVS_C

#include <Grid/CutCell3D/PXConicStruct.h>
#include <Grid/CutCell3D/PXIntersect3dStruct.h>
#include <Grid/CutCell3D/PXPolynomial.h>
#include <Grid/CutCell3D/PXConic.h>

/******************************************************************/
//   FUNCTION Definition: PXFaceVertex
ELVIS_DEVICE int 
PXFaceVertex(enum PXE_ElementType type, int lface, int *CornerNodeOnFace, int *nCornerNodeOnFace)
{

  /****************************************************************************************/
  /*                      Get the number of corner nodes on the face                      */
  /*                           Get the corner nodes on the face                           */
  /****************************************************************************************/
  switch (type){
  case PXE_UniformTetQ1:
  case PXE_UniformTetQ2:
  case PXE_UniformTetQ3:
  case PXE_UniformTetQ4:
  case PXE_UniformTetQ5:
    (*nCornerNodeOnFace) = 3;
    switch (lface){
    case 0:
      CornerNodeOnFace[0] = 1;
      CornerNodeOnFace[1] = 2;
      CornerNodeOnFace[2] = 3;
      return PX_NO_ERROR;
    case 1:
      CornerNodeOnFace[0] = 0;
      CornerNodeOnFace[1] = 3;
      CornerNodeOnFace[2] = 2;
      return PX_NO_ERROR;
    case 2:
      CornerNodeOnFace[0] = 0;
      CornerNodeOnFace[1] = 1;
      CornerNodeOnFace[2] = 3;
      return PX_NO_ERROR;
    case 3:
      CornerNodeOnFace[0] = 0;
      CornerNodeOnFace[1] = 2;
      CornerNodeOnFace[2] = 1;
      return PX_NO_ERROR;
    default:
      ELVIS_PRINTF("ERROR: we can't handle lface = %d\n",lface);
      return PX_BAD_INPUT;
    }
  default:
    ELVIS_PRINTF("Unknown type = %d\n", type);
    return PXError(PX_BAD_INPUT);
  }
}


/******************************************************************/
//   FUNCTION Prototype: PXComputeTetVolume
template <typename DT> ELVIS_DEVICE int
PXComputeTetVolume(DT const* v1, DT const *v2, DT const* v3,
		   DT& vol)
/*-------------------------------------------------------------------*/
/*   PURPOSE: Compute the signed volume of a tet                     */
/*                                                                   */
/*   INPUTS: v1,v2, v3: define the tet                               */
/*                                                                   */
/*   OUTPUTS: vol: volume of the tet                                 */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/	
{
  vol = v1[0]*(v2[1]*v3[2]-v2[2]*v3[1]) 
    + v1[1]*(v2[2]*v3[0]-v2[0]*v3[2]) 
    + v1[2]*(v2[0]*v3[1]-v2[1]*v3[0]);
  
  vol /= 6.0;
  
  return PX_NO_ERROR;
}

/******************************************************************/
//   FUNCTION Prototype: PXDetermineInsideTet
template <typename DT> ELVIS_DEVICE int
PXDetermineInsideTet(DT const* tetNodes, DT const pt[3], enum PXE_3D_ZeroDTypeOnBack &type)
/*-------------------------------------------------------------------*/
/*   PURPOSE: Determine if a point is inside a tet                   */
/*                                                                   */
/*   INPUTS: tetNodes: the tet vertices                              */
/*           faceNormal: normal of each face                         */
/*           faceOrientation: whether the normal is outward or inward*/
/*                              = 1 if outward, = -1 if inward       */
/*           pt: query point                                         */
/*                                                                   */
/*   OUTPUTS: type: whether in or out or on the face or edge         */
/*            typeIndex: face or edge or vertex number               */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/	
{
  int d;
  const int dim = DIM3D;
  DT vec[BACK_NBF][DIM3D];
  DT vol;
  int nDegenerate;
  int lface;
  int lvertex;
  
  /*Compute vector v_i - p*/
  for(lvertex = 0; lvertex < BACK_NBF; lvertex ++)
    for(d = 0; d < dim; d++)
      vec[lvertex][d] = tetNodes[lvertex*dim + d] - pt[d];

  /*Compute the volume of each sub-tet formed from pt, and the other 3 nodes should
    have positive orientation on the tet face*/
  nDegenerate = 0;
  int faceVertex[3];
  int nFaceVertex;
  for(lface = 0; lface < 4; lface++){
    PXFaceVertex(PXE_UniformTetQ1, lface, faceVertex, &nFaceVertex);
    PXComputeTetVolume<PX_REAL>(vec[faceVertex[0]], vec[faceVertex[1]], vec[faceVertex[2]], vol);
      
    //as long as vol is negtiave, the point is outside
    if(vol < 0){
      type = PXE_3D_0DBackNull;
      return PX_NO_ERROR;
    }
    else if(vol == 0){
      nDegenerate++;
    }
  }//lface  

  /*Now the point is inside or on the tet*/
  switch(nDegenerate){
    case 0:
      type = PXE_3D_0DBackElement;
      return PX_NO_ERROR;
    case 1:
      type = PXE_3D_0DBackFace;
      //the face is the one that leads to zero volume
      return PX_NO_ERROR;
    case 2:
      type = PXE_3D_0DBackEdge;
      //the edge is the one whose common faces lead to zero volume
      //ELVIS_PRINTF("Warning: More implementation needed for determining which edge\n");
      //PXErrorReturn(PX_CODE_FLOW_ERROR);
      return PX_NO_ERROR;
    case 3:
      type = PXE_3D_0DBackVertex;
      //the node is the one whose adjacent three faces lead to zero volume
      //ELVIS_PRINTF("Warning: More implementation needed for determining which vertex\n");
      //PXErrorReturn(PX_CODE_FLOW_ERROR);
      return PX_NO_ERROR;
  default:
    ELVIS_PRINTF("Number of degenerate faces can be at most 3\n");
    return PX_CODE_FLOW_ERROR;
  }
}

/******************************************************************/
//   function definition: PXQuadPatchRef2Glob
template <class DT>  ELVIS_DEVICE int
PXQuadPatchRef2Glob(DT const * const quadNode[6], const DT* xRef, DT* xGlob)
{
/*-------------------------------------------------------------------*/
/*   PURPOSE: find the physical coorindate of a point on a quadratic */
/*            patch given its reference coordinate                   */
/*                                                                   */
/*   INPUTS: qNode:[x1,y1,z1,...,x6,y6,z6] coordinates of the        */
/*           nodes defining the quadratic patch                      */
/*           xRef: reference coordinate of the query point           */
/*                                                                   */
/*   OUTPUTS: xGlob: physical coordinate of the query point [x,y,z]  */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/
  const DT& X = xRef[0]; //for coding easiness
  const DT& Y = xRef[1];
  int d, dim = 3;
  int node;
   DT phi[6]; //basis function
   DT Xsq, Ysq, XY; //temporary variables  
  
  /*Basis*/
  Xsq = X*X; Ysq = Y*Y; XY = X*Y;
  phi[0] = 1-3*X-3*Y+2*Xsq+4*XY+2*Ysq;
  phi[1] = -X+2*Xsq;
  phi[2] = -Y+2*Ysq;
  phi[3] = 4*XY;
  phi[4] = 4*(Y-XY-Ysq);
  phi[5] = 4*(X-Xsq-XY);
  
  
  for(d = 0; d < dim; d++){
    xGlob[d] = 0;
    for(node = 0; node < 6; node++)
      xGlob[d] = xGlob[d] + phi[node]*quadNode[node][d];
  }

  return PX_NO_ERROR;
}

/******************************************************************/
//   FUNCTION Definition: PXDetermineZero
ELVIS_DEVICE bool
PXDetermineZero(PX_REAL const& x, PX_REAL const& eps)
{
  if(fabs(x) <= eps) return true;
  else return false;
}


/******************************************************************/
//   FUNCTION Definition: PXSolveQuadratic
ELVIS_DEVICE  int
PXSolveQuadratic(PX_REAL * RESTRICT A, PX_REAL * RESTRICT xsol, int * RESTRICT nsol)
{
  /*
  PURPOSE: Solves quadratic equation.  Coefficients are assumed to be
  perfectly valid, with A[0] nonzero.

  INPUTS:
    A      vector of 3 coefficients: A[0]x^2 + A[1]x + A[2] = 0

  OUTPUTS: 
    xsol   solution vector
    nsol   number of real solutions

  RETURN: Error Code
  */
  PX_REAL disc;
  
  disc = A[1]*A[1] - 4.0*A[0]*A[2];
  if (fabs(disc) < A[1]*A[1]*MEPS){//because sqrt(disc) will be added to A[1]
    (*nsol) = 1;
    xsol[0] = -0.5*A[1]/A[0];
  }
  else if (disc < 0){
    (*nsol) = 0;
  }
  else{
    (*nsol) = 2;
    disc = sqrt(disc);
    xsol[0] = 0.5*(-A[1] - disc)/A[0];
    xsol[1] = 0.5*(-A[1] + disc)/A[0];
  }


  return PX_NO_ERROR;
}


/******************************************************************/
//   FUNCTION Definition: PXSolveCubic
ELVIS_DEVICE  int
PXSolveCubic(PX_REAL * RESTRICT A, PX_REAL * RESTRICT xsol, int * RESTRICT nsol)
{
  /*
  PURPOSE: Solves cubic equation.  Coefficients are assumed to be
  perfectly valid, with A[0] nonzero.  No Newton fixing is performed.

  INPUTS:
    A      vector of 4 coefficients: A[0]x^3 + A[1]x^2 + A[2]x + A[3] = 0

  OUTPUTS: 
    xsol   solution vector
    nsol   number of real solutions

  RETURN: Error Code
  */

  /* For improved conditioning, we solve for r = A[0]^(1/3)*x:
     r^3 + A[1]*A[0]^(-2/3)*r^2 + A[2]*A[0]^(-1/3)*r + A[3] = 0 */
  int n;
  PX_REAL a0, a1, a2, a3;
  PX_REAL b0, b1, b2;
  PX_REAL t0, t1, t2;
  PX_REAL Q, R, D, S, T;
  PX_REAL rcr_a3, smq, theta; //rcr = reciprocal cube root
  PX_REAL eps;

  eps = 10*MEPS;

  n = -1; 
  t0 = t1 = t2 = 0.0;

  a3 = A[0]; a2 = A[1]; a1 = A[2]; a0 = A[3];

  rcr_a3 = rcbrt(a3);
  
  b2 = a2*rcr_a3*rcr_a3;
  b1 = a1*rcr_a3;
  b0 = a0;
  
  // From Mathworld
  Q = (3.0*b1 - b2*b2)/9.0;
  R = (9.0*b1*b2 - 27.0*b0 - 2.0*b2*b2*b2)/54.0;
    
  D = Q*Q*Q + R*R;

  // double check to see if D is close to zero
  if (D < 0){
    smq = sqrt(-Q); // no other way to get D < 0
    if (((R/(-Q)*smq) >= 1.0) ||
	((R/(-Q)*smq) <= -1.0))
      D = 0.0; // set D to zero
  }
    
  if (fabs(D) < eps){

    // all roots real, at least two equal
    S = cbrt(R);
    T = S;

    if (fabs(R) < eps){ // all three roots same
      n = 1;
      t0 = -1.0/3.0*b2 + 0.0;
    }
    else{ // single root plus one double root
      n = 2;
      t0 = -1.0/3.0*b2 + 2.0*S;
      t1 = -1.0/3.0*b2 - S;
    }
  }
  else if (D > 0){
    // one real root
    S = cbrt(R+sqrt(D));

    T = cbrt(R-sqrt(D));

    n = 1;
    t0 = -1.0/3.0*b2 + (S+T);
  }
  else{
    // three distinct real roots (implies Q necessarily < 0)
    smq = sqrt(-Q);
    theta = acos(R/((-Q)*smq));
    n = 3;
    t0 = 2.0*smq*cos(theta/3.0             ) - 1.0/3.0*b2;
    t1 = 2.0*smq*cos(theta/3.0 + 2.0*PI/3.0) - 1.0/3.0*b2;
    t2 = 2.0*smq*cos(theta/3.0 + 4.0*PI/3.0) - 1.0/3.0*b2;
  }
  
  if (n < 0) return PXError(PX_NOT_CONVERGED);

  (*nsol) = n;

  if (n >= 1) xsol[0] = t0*rcr_a3;
  if (n >= 2) xsol[1] = t1*rcr_a3;
  if (n >= 3) xsol[2] = t2*rcr_a3;  

  return PX_NO_ERROR;
}


/******************************************************************/
//   FUNCTION Definition:  PXRootsPoly
ELVIS_DEVICE int 
PXRootsPoly(PX_REAL * RESTRICT pA, int n, int fixflag, int approxflag,
		PX_REAL * RESTRICT xsol, int * RESTRICT nsol)
{
  int k, isol;
  int converged, inewton, maxnewton;
  PX_REAL AA, eps, tol;
  PX_REAL R, R_U, X, XX;
  PX_REAL R0;
  PX_REAL *A;

  eps = MEPS*100;
  A = pA;

  /* /\* Disregard leading zero coefficients *\/ */
  /* while ((fabs(A[0]) < eps) && (n > 0)){ */
  /*   A = A+1; */
  /*   n--; */
  /* } */
  /* if (n <= 0){					 */
  /*   ELVIS_PRINTF("This case might correspond to infinitely many solutions!!\n"); */
  /*   //PXErrorReturn(PX_BAD_INPUT); */
  /*   *nsol = 0; */
  /*   return PX_NO_ERROR; */
  /* } */


  /* divide coeffs by max abs of coeffs */
  AA = 0.0;
  for (k=0; k<(n+1); k++)
    if (fabs(A[k]) > AA) AA = fabs(A[k]);
  /* if (AA < eps){ */
  /*   *nsol = 0; */ //this doesn't make sense, all coeff can be just scaled to very small
  /*   return PX_NO_ERROR; */
  /* } */
  for (k=0; k<(n+1); k++) A[k] /= AA;

  /* Again, disregard leading zero coefficients that may have arisen
     in division by AA */
  while ((fabs(A[0]) < eps) && (n > 0)){
    A = A+1;
    n--;
  }

  /*No unknown coefficients*/
  if(n == 0){
    if(A[0] == 0){
      ELVIS_PRINTF("This case corresponds to infinitely many solutions!!\n");
      PXErrorReturn(PX_BAD_INPUT);
    }
    else{
      *nsol = 0;
      return PX_NO_ERROR;
    }
  }
  
  if (n == 1){
    (*nsol) = 1;
    xsol[0] = -A[1]/A[0];
  }
  else if (n == 2){ // Quadratic Formula
    PXErrorReturn( PXSolveQuadratic(A, xsol, nsol) );
  }
  else if (n == 3){ // Cubic Formula
    PXErrorReturn( PXSolveCubic(A, xsol, nsol) );
  }
  /* else if (n == 4){ // Quartic Formula */
  /*   PXErrorReturn( PXSolveQuartic(A, approxflag, xsol, nsol) ); */
  /*   fixflag = PXE_False; // Newton already done in SolveQuartic */
  /* } */
  else{
    ELVIS_PRINTF("Order %d root finding not supported.\n", n);
    return PXError(PX_BAD_INPUT);
  }

  if ((fixflag) && (n > 1)){
    eps = 10*MEPS;
    tol = 10*eps;

    /* Improve accuracy via Newton */
    
    // B stores the coefficients of the derivative w.r.t x
    //PXErrorReturn( PXAllocate( n, sizeof(PX_REAL), (void **)&(B) ) ); 
    PX_REAL B[3];

    for (k=0; k<n; k++)
      B[k] = ( (PX_REAL) (n-k))*A[k];

    for (isol=0; isol<(*nsol); isol++){
      X = xsol[isol];
      R = 0; XX = 1;
      for (k=n; k>=0; k--){
	R += A[k]*XX;
	XX *= X;
      }
      R0 = R;
    			
      converged = PXE_False;
      maxnewton = 25;
      for (inewton=0; inewton<maxnewton; inewton++){

	if (fabs(R) < tol) converged = PXE_True;

	R_U = 0; XX = 1;
	for (k=n-1; k>=0; k--){
	  R_U += B[k]*XX;
	  XX *= X;
	}

	if (fabs(R_U) < eps) break; // cannot invert
	X = X - R/R_U;

	R = 0; XX = 1;
	for (k=n; k>=0; k--){
	  R += A[k]*XX;
	  XX *= X;
	}

	if (converged) break;

      } // inewton
      if (!converged){
	if(fabs(R) < fabs(R0))
	  xsol[isol] = X;
	//if R is still large, then just take the root from the polynomial formula
	/* if (fabs(R) > 1e-7){ */ 
	/*   ELVIS_PRINTF("Warning, Newton did not converge in root fixing (R = %.10E).\n", R); */
	/* } */
      }
      else
	xsol[isol] = X;
     
    } // isol
  }

  return PX_NO_ERROR;
}


/******************************************************************/
//   function definition: PXSolveQuadraticRoots
ELVIS_DEVICE int
PXSolveQuadraticRoots(PX_REAL* coeff, 
		      enum PXE_QuadraticEquation_Type& type, int& count, 
		      PX_REAL* sol)
/*-------------------------------------------------------------------*/
/*   PURPOSE: Solve for roots of quadratic with double precision     */
/*            Overloading with the function with exact arithmetics   */
/*                                                                   */
/*   INPUTS: coeff: coeff[i] is the coefficent for x^i               */
/*                                                                   */
/*   OUTPUTS: count: numer of roots                                  */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/
{
  PX_REAL temp;

  /*Reverse coefficient*/
  temp = coeff[0]; coeff[0] = coeff[2]; coeff[2] = temp;
  
  /*Solve, using quadratic equation formula*/
  //ELVIS_PRINTF("%.15e %.15e %.15e\n", coeff[0], coeff[1], coeff[2]);
  int ierr = PXError(PXRootsPoly(coeff, 2, PXE_True, PXE_False,
				 sol, &count));
  //ELVIS_PRINTF("%.15e %.15e %.15e\n", coeff[0], coeff[1], coeff[2]);
  //ELVIS_PRINTF("count = %d\n", count);

  if(ierr != PX_NO_ERROR){
    ELVIS_PRINTF("%.15e %.15e %.15e\n", coeff[0], coeff[1], coeff[2]);
    PXErrorReturn(ierr);
  }

  if(count == 1)
    type = PXE_Quadratic_DuplicateRealRoots; //could be linear root too
  else if(count == 2)
    type = PXE_Quadratic_DistinctRealRoots;
  else //count = 0
    type = PXE_Quadratic_NoRealRoots;

  return PX_NO_ERROR;
}


/******************************************************************/
//   function definition: CountQuadraticRootsbySolve
template <class DT>  ELVIS_DEVICE int 
PXCountQuadraticRootsbySolve(DT* coeff, 
			     enum PXE_QuadraticEquation_Type& type, int& count, 
			     int& boundary, int& rootToSolve, DT* sol)
{
/*-------------------------------------------------------------------*/
/*   PURPOSE: Counts the number of real roots in the range [0,1] for */
/*            a quadratic equation a*x^2+bx+c by solving the roots   */
/*                                                                   */
/*   INPUTS: coeff: coeff[i] is the coefficent for x^i               */
/*                                                                   */
/*   OUTPUTS: count: numer of roots in [0,1]                         */
/*                   count = -1 if there are infinitely many sol     */
/*            type: quadratic equation type                          */
/*            boundary = 0 if x=0 is one solution                    */
/*                     = 1 if x=1 is one solution                    */
/*                     = 2 if x=0 and x=1 are solutions              */
/*                     = -1 if solutions are in (0,1)                */
/*            rootToSolve = 0 if solution in (0,1) is -b+sqrt(d)/2a  */
/*                        = 1 if solution in (0,1) is -b-sqrt(d)/2a  */
/*                        = 2 if both are needed                     */
/*                        = -1 otherwise                             */
/*            sol      solution in the range [0,1]                   */
/*   Note: the reason why we need all the information about solution */
/*         properties is we might need solve the equation again using*/
/*         finite precision later, but don't want to do root         */
/*         classification with finite precision                      */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/
  const DT& a = coeff[2];
  const DT& b = coeff[1];
  const DT& c = coeff[0];
  
   DT temp;

  count = 0;
  boundary = -1;
  rootToSolve = -1;

  int tempcount;
  DT tempsol[2];

  PXErrorReturn(PXSolveQuadraticRoots(coeff, type, tempcount, tempsol));
  
  if(type == PXE_Quadratic_ConstantZero){
    count = -1;
  }
  else if(type == PXE_Quadratic_Linear 
     || type == PXE_Quadratic_DuplicateRealRoots){
    if(tempsol[0] < 1 && tempsol[0] > 0){
      sol[0] = tempsol[0];
      count = 1;
    }
    else if(tempsol[0] == 0){
      sol[0] = 0;
      boundary = 0;
      count = 1;
    }
    else if(tempsol[0] == 1){
      sol[0] = 1;
      boundary = 1;
      count = 1;
    }
  }
  else if(type == PXE_Quadratic_DistinctRealRoots){
    if(c == 0){ //one solution = 0
      boundary = 0;
      if(b == -a){ //solutions are 0, 1
	boundary = 2;
	count = 2;
	sol[0] = 0.0;
	sol[1] = 1.0;
	return PX_NO_ERROR;
      }
      else{ //solutions are 0 and -b/a
	temp = -b/a;
	sol[0] = 0.0;
	if(temp > 0 && temp < 1){
	  count = 2;
	  sol[1] = temp;
	}
	else 
	  count = 1;
	return PX_NO_ERROR;
      }
    }
    else if(a+b+c == 0){ //one solution = 1, the other = c/a
      boundary = 1;
      temp = c/a;
      sol[0] = 1.0;
      if(temp > 0 && temp < 1){
	sol[1] = temp;
	count = 2;
      }
      else 
	count = 1;
      return PX_NO_ERROR;
    }
    else{//two distinct roots, no root = 0 or = 1
      count = 0;
      if(tempsol[0] > 0 && tempsol[0] < 1){
	rootToSolve = 0;
	(count)++;
	sol[0] = tempsol[0];
      }
      
      if(tempsol[1] > 0 && tempsol[1] < 1){
	rootToSolve = 1;
	(count)++;
	sol[0] = tempsol[1];
      }
      
      if(count == 2){
	rootToSolve = 2;
	sol[0] = tempsol[0];
	sol[1] = tempsol[1];
      }
    }
  }
 
  return PX_NO_ERROR;
}


/******************************************************************/
//   function definition: PXSolveOneCubicRoot
ELVIS_DEVICE  int
PXSolveOneCubicRoot(PX_REAL* coeff, enum PXE_CubicEquation_Type& cubicType, 
		    bool& rootExist, PX_REAL& root, enum PXE_Boolean& Inverse, 
		    enum PXE_Boolean &solved)
/*-------------------------------------------------------------------*/
/*   PURPOSE: Solve for one real root of cubic equation with exact   */
/*            arithmetic                                             */
/*                                                                   */
/*   INPUTS: coeff: coeff[i] is the coefficent for x^i               */
/*           might be changed so that leading coeff > 0              */
/*           solved: whether the solution has to be found no matter  */
/*                   how slow the code might be due to precision     */
/*                   = true if the case                              */
/*                                                                   */
/*   OUTPUTS: cubicType: type of the cubic equation                  */
/*            rootExist: whether a real root exists                  */
/*                 could be false when the cubic is a quadratic      */
/*            root: the found real root                              */
/*            Inverse: whether the root found is 1/x                 */
/*            solved: whether the solution has been found            */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/  
{
  PX_REAL temp;
  PX_REAL sol[3];
  int nSol;
  PX_REAL res;
  PX_REAL tol;
  int k1, k2;
  
  /*Reverse coefficient*/
  //ELVIS_PRINTF("coeff = [%.15e %.15e %.15e %.15e];\n",coeff[3], coeff[2], coeff[1], coeff[0]); 
  temp = coeff[0]; coeff[0] = coeff[3]; coeff[3] = temp;
  temp = coeff[1]; coeff[1] = coeff[2]; coeff[2] = temp;

  /*Solve, using cubic equation formula and Newton's method*/
  PXErrorReturn(PXRootsPoly(coeff, 3, PXE_True, PXE_False,
			    sol, &nSol));
  
  /*Classify and copy the root*/
  if(nSol == 0)
    rootExist = PXE_False;
  else{
    root = sol[0];
    rootExist = PXE_True;
  }
  //ELVIS_PRINTF("coeff = [%.15e %.15e %.15e %.15e];\n",coeff[0], coeff[1], coeff[2], coeff[3]); 
  //ELVIS_PRINTF("root = %.15e\n", root);
  //ELVIS_PRINTF("res = %.15e\n", coeff[0]*root*root*root + coeff[1]*root*root + coeff[2]*root + coeff[3]);

  solved = PXE_True;
  cubicType = PXE_Cubic_Last;//not set  
  Inverse = PXE_False;
    
  /*Just making sure*/
  if(rootExist == PXE_True){
    res = coeff[0]*root*root*root + coeff[1]*root*root + coeff[2]*root + coeff[3];
    tol = 0;
    for(k1 = 0; k1 < 4; k1 ++){
      //compute coeff[k]*root^(3-k)
      temp = coeff[k1];
      for(k2 = 0; k2 < 3-k1; k2 ++)
	temp *= root;
      //find the max
      if(tol < fabs(temp))
	tol = fabs(temp);
    }

    //check
    if(fabs(res) > 1e-10*tol){
      ELVIS_PRINTF("Switch to exact precision to solve the cubic equation\n");
    /*   EX_REAL exCoeff[4]; */
    /*   EX_REAL exRoot; */
    
    /*   exCoeff[3] = coeff[0]; exCoeff[2] = coeff[1]; exCoeff[1] = coeff[2]; exCoeff[0] = coeff[3]; */
    /*   PXErrorReturn(PXSolveOneCubicRoot(exCoeff, cubicType, rootExist, exRoot, Inverse, solved)); */
    
    /*   if(solved == PXE_True) */
    /* 	PXErrorReturn(PXConvertExact2Double(&exRoot, &root, 1)); */
    }
  }

  return PX_NO_ERROR;
}


/******************************************************************/
//   function definition: PXBivariateQuadResultant
template <class DT> ELVIS_DEVICE int
PXBivariateQuadResultant(DT const* coeff[2][6], DT Xres[5], DT Yres[5])
/*-------------------------------------------------------------------*/
/*   PURPOSE: Given a system of bivariate quadratic functions,       */
/*            find the X- and Y- resultants                          */
/*                                                                   */
/*   INPUTS: coeff:                                                  */
/*           a1*x^2+2*b1*xy+c1*y^2+2*d1*x+2*e1*y+f1=0                */
/*           a2*x^2+2*b2*xy+c2*y^2+2*d2*x+2*e2*y+f2=0                */
/*           Defined as pointer in order to refine the precsion of   */
/*           of the same variable, when exact datatype is used       */
/*                                                                   */
/*   OUTPUTS: Xres: coeff for y^4,y^3,y^2,y,1, Xres[i] is for y^i    */
/*            Yres: coeff for x^4,x^3,x^2,x,1                        */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/  
{
  DT const &a1 = *(coeff[0][0]); DT const &a2 = *(coeff[1][0]);
  DT const &b1 = *(coeff[0][1]); DT const &b2 = *(coeff[1][1]);
  DT const &c1 = *(coeff[0][2]); DT const &c2 = *(coeff[1][2]);
  DT const &d1 = *(coeff[0][3]); DT const &d2 = *(coeff[1][3]);
  DT const &e1 = *(coeff[0][4]); DT const &e2 = *(coeff[1][4]);
  DT const &f1 = *(coeff[0][5]); DT const &f2 = *(coeff[1][5]);
  
  
   DT ab, ac, ad, ae, af;
   DT bc, bd, be, bf;
   DT cd, ce, cf;
   DT de, df;
   DT ef;

  ab = a1*b2-a2*b1;
  ac = a1*c2-a2*c1;
  ad = a1*d2-a2*d1;
  ae = a1*e2-a2*e1;
  af = a1*f2-a2*f1;
  bc = b1*c2-b2*c1;
  bd = b1*d2-b2*d1;
  be = b1*e2-b2*e1;
  bf = b1*f2-b2*f1;
  cd = c1*d2-c2*d1;
  ce = c1*e2-c2*e1;
  cf = c1*f2-c2*f1;
  de = d1*e2-d2*e1;
  df = d1*f2-d2*f1;
  ef = e1*f2-e2*f1;


  /*Compute X res*/
  if(a1 == 0 && a2 == 0){ //then no x^2 term
    Xres[4] = 0;
    
    Xres[3] = 2*bc;
    
    Xres[2] = 4*be - 2*cd;
    
    Xres[1] = 2*bf + 4*de;
    
    Xres[0] = 2*df;
  }
  else{
    Xres[4] = ac*ac-4*ab*bc;
    
    Xres[3] = 4*(ae*ac-2*ab*be+ab*cd-ad*bc);
    
    Xres[2] = 4*(ae*ae+ad*cd-ab*bf-2*ad*be-2*de*ab)+2*af*ac;
    
    Xres[1] = 4*(-2*de*ad+ae*af-ad*bf-df*ab);
    
    Xres[0] =  af*af-4*df*ad;
  }

  /*Compute Y res*/
  if(c1 == 0 && c2 == 0){ //then no y^2 term
    Yres[4] = 0;
    
    Yres[3] = -2*ab;
    
    Yres[2] = -2*ae + 4*bd;
    
    Yres[1] = 2*bf - 4*de;
    
    Yres[0] = 2*ef;
  }
  else{
    Yres[4] = ac*ac-4*ab*bc;
    
    Yres[3] = 4*(-cd*ac+2*bc*bd-bc*ae+ce*ab);
    
    Yres[2] = 4*(cd*cd+ce*ae+bc*bf-2*ce*bd-2*de*bc)-2*cf*ac;
    
    Yres[1] = 4*(2*de*ce+cd*cf-ce*bf+ef*bc);
    
    Yres[0] = cf*cf-4*ef*ce;
  }
  

  return PX_NO_ERROR;
}


/******************************************************************/
//   function definition: PXCountSignChange
template <class DT>  ELVIS_DEVICE int
PXCountSignChange(DT const* sturmSeq, int const& n, int &numSignChange)
/*-------------------------------------------------------------------*/
/*   PURPOSE: Given a sturm sequence, count the number of sign       */
/*            changes                                                */
/*            Note: +,0,- or -,0,+ is 1 sign change                  */
/*                  +,0,+, or -,0,0 is 0 sign change                 */
/*            Sturm sequence can't have two consecutive 0's          */
/*                                                                   */
/*   INPUTS: Sturm seq: the sequence                                 */
/*           n: length of sequence                                   */
/*                                                                   */
/*   OUTPUTS: numSignChange: number of sign changes                  */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/  
{
  int currk; 
  int prevk;
  int firstk;//first non-zero coefficient
  
  /*Initialize*/
  numSignChange = 0;
  firstk = 0;
  prevk = 0;
  
  /*First non-zero*/
  for(currk = 0; currk < n; currk++){
    if(sturmSeq[currk] != 0){
      firstk = currk;
      prevk = firstk;
      break;
    }
  }
  
  //check (can't have two consecutive 0)
  if(firstk > 1){
    ELVIS_PRINTF("Two consecutive zero's in Sturm sequence?\n");
    PXErrorReturn(PX_BAD_INPUT);
  }
  
  //if there is only one non-zero term
  if(firstk == n-1){
    numSignChange = 0;
    return PX_NO_ERROR;
  }

  /*Count sign changes*/
  for(currk = firstk+1; currk < n; currk++){
    
    //zero
    if(sturmSeq[currk] == 0){
      
      if(prevk != currk-1){
	ELVIS_PRINTF("there are two consecutive zero's!\n");
	ELVIS_PRINTF("n = %d\n", n);
	ELVIS_PRINTF("SturmSeq = :\n");
	//PXPrintCoeffcients(sturmSeq, n);
	PXErrorReturn(PX_BAD_INPUT);
      }
      continue;
    }
    
    //count
    if((sturmSeq[currk] > 0 && sturmSeq[prevk] < 0)
       || (sturmSeq[currk] < 0 && sturmSeq[prevk] > 0))
      numSignChange++;

    prevk = currk;      
  }


  return PX_NO_ERROR;
}


/******************************************************************/
//   FUNCTION Definition: PXRescaleCoefficients
ELVIS_DEVICE  int
PXRescaleCoefficients(PX_REAL *coeff, int const& n)
{
  int k;
  
  //convert to PX_REAL
  PX_REAL maxcoeff = 0;
  for(k = 0; k < n; k++){
    if(maxcoeff < fabs(coeff[k]))
      maxcoeff = fabs(coeff[k]);
  }
  if(maxcoeff == 0)
    PXErrorReturn(PX_BAD_INPUT);
  
  //rescale
  for(k = 0; k < n; k++)
    coeff[k] /= maxcoeff;
  
  return PX_NO_ERROR;

}


/******************************************************************/
//   function definition: PXCountCubicRoots
template <class DT>  ELVIS_DEVICE int
PXCountCubicRoots(DT coeff[5], bool& rootExist, int *pNumRoot)
/*-------------------------------------------------------------------*/
/*   PURPOSE: Given a cubic equation in the form of                  */
/*            coeff[i] is the coeff of x^i                           */
/*            Determine if there are solutions in [0,1]              */
/*                                                                   */
/*   INPUTS: coeff: the coefficients: d,c,b,a                        */
/*                                                                   */
/*   OUTPUTS: rootExist: whethere there is a root in [0,1]           */
/*            numRoot: number of roots in (0,1), only true if 0 or 1 */
/*                     not a root (multiplicity not counted          */
/*            The coefficients are changed in this functions         */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/  
{
  DT& d = coeff[0];
  DT& c = coeff[1];
  DT& b = coeff[2];
  DT& a = coeff[3];
 
   DT W;
   DT d1, d2, d3;
   
   DT sturmSeq[2][5];
  int nSturmSeq; //number of nonzero Sturm terms
   
  int numSignChange[2];
  int nRoot;
  
  /*Trivial case*/
  if(d == 0 || a+b+c+d == 0){
    rootExist = true;
    return PX_NO_ERROR;
  } 

  //quadratic
  if(PXDetermineZero(a,MEPS)){
    enum PXE_QuadraticEquation_Type type;
    int count;
    int boundary; 
    int rootToSolve;
    DT sol[2];

    //equation is now b*x^2+c*x+d = 0
    PXErrorReturn(PXCountQuadraticRootsbySolve<DT>(coeff, type, count, 
						   boundary, rootToSolve, sol));
    
    if(count > 0)
      rootExist = true;
    else
      rootExist = false;
    
    return PX_NO_ERROR;
  }
  //make sure a > 0
  else if(a < 0){
    a = -a; b = -b; c = -c; d = -d; 
  }

  /*Compute all the invariants and Bezout matrix components*/
  d2 = b*b - 3*a*c;
  d3 = c*c - 3*b*d;
  W = b*c - 9*a*d;
  d1 = -(W*W-4*d2*d3)/9;

  /*Compute sturm sequence at x= 0 and x = 1*/
  //S0  = p,  S1 = p'
  //S2 = (2*Delta_2*x+W)/9/a
  //S3 = Delta_1*9*a/4/Delta_2^2

  //first two terms are the same regardless what case
  //Sturm sequenece at x = 0
  sturmSeq[0][0] = d;
  sturmSeq[0][1] = c;

  //Sturm sequenece at x = 1
  sturmSeq[1][0] = a+b+c+d;
  sturmSeq[1][1] = 3*a+2*b+c;

  //the rest terms can be special 
  if(PXDetermineZero(d2,MEPS)){//special cases

    //S2 is a constant
    //at x= 0
    sturmSeq[0][2] = W;
         
    //at x=1
    sturmSeq[1][2] = W;

    nSturmSeq = 3;
  }
  else{//general case
    
    /*Sturm sequenece at x = 0*/
    sturmSeq[0][2] = W;
    sturmSeq[0][3] = d1;
    
    /*Sturm sequenece at x = 1*/
    sturmSeq[1][2] = 2*d2+W;
    sturmSeq[1][3] = d1;
      
    nSturmSeq = 4;
  }

  /*Count sign changes*/
  PXErrorReturn(PXCountSignChange<DT>(sturmSeq[0], nSturmSeq, numSignChange[0]));
  PXErrorReturn(PXCountSignChange<DT>(sturmSeq[1], nSturmSeq, numSignChange[1]));
   
  /*Number of roots in [0,1]*/
  nRoot = numSignChange[0] - numSignChange[1];  
  if(nRoot > 0)
    rootExist = true;
  else if(nRoot == 0)
    rootExist = false;
  else{
    //ELVIS_PRINTF("Can Sturm sequence give negative number of real roots??\n");
    return PX_CODE_FLOW_ERROR;
  }

  if(pNumRoot != NULL)
    *pNumRoot = nRoot;
  
  return PX_NO_ERROR;
}


/******************************************************************/
//   function definition: PXCountQuarticRoots
template <class DT>  ELVIS_DEVICE int
PXCountQuarticRoots(DT coeff[5], bool& rootExist, int *pNumRoot)
/*-------------------------------------------------------------------*/
/*   PURPOSE: Given a quartic equation in the form of                */
/*            coeff[i] is the coeff of x^i                           */
/*            Determine if there are solutions in [0,1]              */
/*                                                                   */
/*   INPUTS: coeff: the coefficients: e,d,c,b,a                      */
/*                                                                   */
/*   OUTPUTS: rootExist: whethere there is a root in [0,1]           */
/*            numRoot: number of roots in (0,1), only true if 0 or 1 */
/*                     not a root (multiplicity not counted          */
/*            The coefficients are changed in this functions         */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/  
{
  DT& e = coeff[0];
  DT& d = coeff[1];
  DT& c = coeff[2];
  DT& b = coeff[3];
  DT& a = coeff[4];

   DT W1, W3; //W2
   DT d1, d2, d3;
   DT A, B;
   DT T1, T2;
  
   DT sturmSeq[2][5];
  int nSturmSeq; //number of nonzero Sturm terms
   
  int numSignChange[2];
  int nRoot;

  int ierr;
  
  /*Infinitely many solutions*/
  if(coeff[0] == 0 && coeff[1] == 0 && coeff[2] == 0
     && coeff[3] == 0 && coeff[4] == 0){
    rootExist = true;
    
    if(pNumRoot != NULL)
      *pNumRoot = -1;
    
    return PX_NO_ERROR;
  }
  //Note: if the resultant is all zero, that means the two equations have a 
  //common factor

  /*Rescale*/
  PXErrorReturn(PXRescaleCoefficients(coeff, 5));
  
  /*Trivial case*/
  if(e == 0 || a+b+c+d+e == 0){
    rootExist = true;
    return PX_NO_ERROR;
  } 

  //cubic
  if(PXDetermineZero(a,MEPS)){
    PXErrorReturn(PXCountCubicRoots<DT>(coeff, rootExist, pNumRoot));
    return PX_NO_ERROR;
  }
  //make sure a > 0
  else if(a < 0){
    a = -a; b = -b; c = -c;
    d = -d; e = -e;
  }
  
  /*Change to the form :
    ax^4-4bx^3+6cx^2-4dx+e
    using notation in Mourrain's paper:
    Algebriac Issues in Computational Geometry*/
  b /= -4.0;
  c /= 6.0;
  d /= -4.0;  

  /*Compute all the invariants and Bezout matrix components*/
  W1 = a*d-b*c;
  //W2 = b*e-c*d;
  W3 = a*e-b*d;
  
  d2 = b*b-a*c;
  d3 = c*c-b*d;
  A = W3+3*d3;
  B = -d*W1-e*d2-c*d3;
  d1 = A*A*A-27*B*B;
  
  T1 = -W3*d2-3*W1*W1+9*d2*d3;
  T2 = A*W1-3*b*B;

  /*Compute sturm sequence at x= 0 and x = 1*/
  //S0  = p,  S1 = p'
  //S2 = (3Delta_2*x^2+3*W_1*x-W_3)/a
  //S3 = (T_1*x+T_2)*4a/3/Delta_2^2
  //S4 = Delta_1*Delta_2^2/a/T_1^2

  //first two terms are the same regardless what case
  //Sturm sequenece at x = 0
  sturmSeq[0][0] = e;
  sturmSeq[0][1] = -d;

  //Sturm sequenece at x = 1
  sturmSeq[1][0] = a-4*b+6*c-4*d+e;
  sturmSeq[1][1] = a-3*b+3*c-d;

  //the rest terms can be special 
  if(PXDetermineZero(d2,0)){//special cases

    if(PXDetermineZero(W1,0)){//stop at S2
      //at x= 0
      sturmSeq[0][2] = -W3;     
      //at x= 1
      sturmSeq[1][2] = -W3; 
      //number of terms
      nSturmSeq = 3;
    }
    else{//S2 is a linear, S3 is constant
      //at x= 0
      sturmSeq[0][2] = -W3;
      sturmSeq[0][3] = -d1*W1;
     
      //at x=1
      sturmSeq[1][2] = 3*W1-W3;  
      sturmSeq[1][3] = -d1*W1;

      nSturmSeq = 4;
    }

  }
  else if(PXDetermineZero(T1,0)){//stop at S3

    /*Sturm sequenece at x = 0*/
    sturmSeq[0][2] = -W3;
    sturmSeq[0][3] = T2;
  
    /*Sturm sequenece at x = 1*/
    sturmSeq[1][2] = 3*d2+3*W1-W3;
    sturmSeq[1][3] = T2;  
    
    nSturmSeq = 4;
  }
  else{//general case
    
    /*Sturm sequenece at x = 0*/
    sturmSeq[0][2] = -W3;
    sturmSeq[0][3] = T2;
    sturmSeq[0][4] = d1;
  
    /*Sturm sequenece at x = 1*/
    sturmSeq[1][2] = 3*d2+3*W1-W3;
    sturmSeq[1][3] = T1+T2;
    sturmSeq[1][4] = d1;   
    
    nSturmSeq = 5;
  }

  /*Count sign changes*/
  ierr = PXError(PXCountSignChange<DT>(sturmSeq[0], nSturmSeq, numSignChange[0]));
  if(ierr != PX_NO_ERROR){
    //PXPrintCoeffcients(coeff, 5);
    PXErrorReturn(ierr);
  }

  PXErrorReturn(PXCountSignChange<DT>(sturmSeq[1], nSturmSeq, numSignChange[1]));
  
  
  /*Number of roots in [0,1]*/
  nRoot = numSignChange[0] - numSignChange[1];  
  if(nRoot > 0)
    rootExist = true;
  else if(nRoot == 0)
    rootExist = false;
  else{
    //ELVIS_PRINTF("Can Sturm sequence give negative number of real roots??\n");
    return PX_CODE_FLOW_ERROR;
  }

  if(pNumRoot != NULL)
    *pNumRoot = nRoot;
  
  return PX_NO_ERROR;

}



/******************************************************************/
//   function definition: PXDetermineBiQuadSystemRoot
template <class DT>  ELVIS_DEVICE int
PXDetermineBiQuadSystemRoot(DT const* coeff[2][6], bool& rootExist)
/*-------------------------------------------------------------------*/
/*   PURPOSE: Given a system of bivariate quadratic functions,       */
/*            determine if there is solution in [0,1]x[0,1]          */
/*                                                                   */
/*   INPUTS: coeff:                                                  */
/*           a1*x^2+2*b1*xy+c1*y^2+2*d1*x+2*e1*y+f1=0                */
/*           a2*x^2+2*b2*xy+c2*y^2+2*d2*x+2*e2*y+f2=0                */
/*           Defined as pointer in order to refine the precsion of   */
/*           of the same variable, when exact datatype is used       */
/*                                                                   */
/*   OUTPUTS: rootExist: whether there is a solution in unit square  */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/ 
{
   DT Xres[5];
   DT Yres[5];
  int ierr;

  //form x-resultant (quartic in y) and y-resultant (quartic in x)
  PXErrorReturn(PXBivariateQuadResultant<DT>(coeff, Xres, Yres));

  //determine number of y roots within [0,1]
  ierr = PXCountQuarticRoots<DT>(Xres, rootExist, NULL);
  if(ierr != PX_NO_ERROR){
    //ELVIS_PRINTF("Warning: Something is wrong when counting number of roots, this might be due to precision issue. \nIf double precision is used here, just solve for the root directly. \nThis should not happen for exact precision\n");
    //ELVIS_PRINTF("Coefficients = \n");
    //PXPrintCoeffcients((*coeff)[0], 6);
    //PXPrintCoeffcients((*coeff)[1], 6);
    //ELVIS_PRINTF("Xres = \n");
    //PXPrintCoeffcients(Xres,5);
    rootExist = PXE_True;
    //PXErrorReturn(ierr);
    return PX_NO_ERROR;
  }
  if(!rootExist)
    return PX_NO_ERROR;

  //determine number of x roots within [0,1]
  ierr = PXCountQuarticRoots<DT>(Yres, rootExist, NULL);
  if(ierr != PX_NO_ERROR){
    //ELVIS_PRINTF("Warning: Something is wrong when counting number of roots, this might be due to precision issue. \nIf double precision is used here, just solve for the root directly. \nThis should not happen for exact precision\n");
    //ELVIS_PRINTF("Coefficients = \n");
    //PXPrintCoeffcients((*coeff)[0], 6);
    //PXPrintCoeffcients((*coeff)[1], 6);
    //ELVIS_PRINTF("Yres = \n");
    //PXPrintCoeffcients(Yres,5);
    rootExist = PXE_True;
    return PX_NO_ERROR;
    //PXErrorReturn(ierr);
  }

  return PX_NO_ERROR;
}


/******************************************************************/
//   FUNCTION Definition: PXSetConicLineMultiplicity
ELVIS_DEVICE  int
PXSetConicLineMultiplicity(enum PXE_QuadraticEquation_Type const& type, 
			   int &multiplicity)
/*-------------------------------------------------------------------*/
/*   PURPOSE: For a conic-line intersection, given the quadratic     */
/*            equation type, set the multiplicity                    */
/*                                                                   */
/*   INPUTS: type: quadratic equation type                           */
/*                                                                   */
/*   OUTPUTS:                                                        */
/*            multiplicity: intersection multiplicity                */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/
{
  switch(type){
    case PXE_Quadratic_DuplicateRealRoots:
      multiplicity = 2;
      break;
    case PXE_Quadratic_DistinctRealRoots:
    case PXE_Quadratic_Linear:
      multiplicity = 1;
      break;
    default: 
      ELVIS_PRINTF("The quadratic equation type is not recognized\n");
      PXErrorReturn(PX_CODE_FLOW_ERROR);
  }
  
  return PX_NO_ERROR;
}

/******************************************************************/
//   FUNCTION Prototype: PXFormConicSystemForTetEdgeQuadFaceIntersect
template <class DT>  ELVIS_DEVICE int
PXFormConicSystemForTetEdgeQuadFaceIntersect(DT Fcoeff[3][8], ConicSection<DT> *conicS1,
					     ConicSection<DT> *conicS2)
/*-------------------------------------------------------------------*/
/*   PURPOSE: For a tet-edge quad-face intersection, it can be posed */
/*            as a system of three quadratic equations:              */
/*     F(i) = F_0X^2+ F_1XY + F_2Y^2 + F_3X + F_4Y + F_5 - F_6t - F_7*/
/*            This can be solved by elimination of the term of t, and*/
/*            gives a system of two conic equations                  */
/*                                                                   */
/*   INPUTS:  Fcoeff: coefficients for these three equations         */
/*                                                                   */
/*   OUTPUTS: conicS1, conicS2: the result two conic equations       */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/
{
   DT alpha1, alpha2;
  int ZeroCoeffEq[3];//equation with 
  int nZeroCoeffEq = 0;
  int NonZeroCoeffEq[3];
  int nNonZeroCoeffEq = 0;
  int kEquation, jEquation;
  int maxEquation; //the equation with max Fcoeff[i][6]
                   //like a pivot
   DT maxCoeff6;
  
  //initialize
  conicS1->type =  PXE_Conic_Undetermined;
  conicS1->degenerate = -1;

  conicS2->type =  PXE_Conic_Undetermined;
  conicS2->degenerate = -1;

  // ELVIS_PRINTF("coeff = \n");
  // for(kEquation = 0; kEquation < 3; kEquation++){
  //   for(int k = 0; k < 8; k ++ )
  //     cout<<Fcoeff[kEquation][k]<<" ";
  //   cout<<endl;
  // }

  //find if any of the Fcoeff[i][6] is 0 and the equation with maxEuqation
  maxCoeff6 = 0; maxEquation = -1;
  for(kEquation = 0; kEquation < 3; kEquation++){
    //if(PXDetermineZero(Fcoeff[kEquation][6])){
    if(Fcoeff[kEquation][6] == 0){
      ZeroCoeffEq[nZeroCoeffEq] = kEquation;
      nZeroCoeffEq++;	
    }
    else{
      NonZeroCoeffEq[nNonZeroCoeffEq] = kEquation;
      nNonZeroCoeffEq++;	
    }

    //find the equation with max Fcoeff[i][6]
    //exact library has abs not fabs
    //this is for better conditioning during elimination
    if(Fcoeff[kEquation][6] > 0 && maxCoeff6 < Fcoeff[kEquation][6]){
      maxCoeff6 = Fcoeff[kEquation][6];
      maxEquation = kEquation;
    }
    else if(Fcoeff[kEquation][6] < 0 && maxCoeff6 < -Fcoeff[kEquation][6]){
      maxCoeff6 = -Fcoeff[kEquation][6];
      maxEquation = kEquation;
    }
  }
  
  //for equations with non-zero Fcoeff[i][6]
  //need to compute the linear combination of the two equations to eliminate Fcoeff[i][6] 
  if(nNonZeroCoeffEq >= 2){
    kEquation = maxEquation; 
    
    if(NonZeroCoeffEq[0] == maxEquation) jEquation = NonZeroCoeffEq[1];
    else jEquation = NonZeroCoeffEq[0];
    //ELVIS_PRINTF("eq %d %d\n", kEquation, jEquation);

    //kEquation = NonZeroCoeffEq[0]; jEquation = NonZeroCoeffEq[1];
    alpha1 = Fcoeff[kEquation][6]; alpha2 =  Fcoeff[jEquation][6];
    conicS1->A = Fcoeff[kEquation][0]*alpha2 - Fcoeff[jEquation][0]*alpha1;
    conicS1->B = (Fcoeff[kEquation][1]*alpha2 - Fcoeff[jEquation][1]*alpha1)/2;
    conicS1->C = Fcoeff[kEquation][2]*alpha2 - Fcoeff[jEquation][2]*alpha1;
    conicS1->D = (Fcoeff[kEquation][3]*alpha2 - Fcoeff[jEquation][3]*alpha1)/2;
    conicS1->E = (Fcoeff[kEquation][4]*alpha2 - Fcoeff[jEquation][4]*alpha1)/2;
    conicS1->F = (Fcoeff[kEquation][5]*alpha2 - Fcoeff[jEquation][5]*alpha1)-(Fcoeff[kEquation][7]*alpha2 - Fcoeff[jEquation][7]*alpha1);
  }    
  if(nNonZeroCoeffEq == 3){
    kEquation = maxEquation; 
    
    if(NonZeroCoeffEq[2] == maxEquation) jEquation = NonZeroCoeffEq[1];
    else jEquation = NonZeroCoeffEq[2];
    //ELVIS_PRINTF("eq %d %d\n", kEquation, jEquation);

    //kEquation = NonZeroCoeffEq[0]; jEquation = NonZeroCoeffEq[2];
    alpha1 = Fcoeff[kEquation][6]; alpha2 =  Fcoeff[jEquation][6];
    conicS2->A = Fcoeff[kEquation][0]*alpha2 - Fcoeff[jEquation][0]*alpha1;
    conicS2->B = (Fcoeff[kEquation][1]*alpha2 - Fcoeff[jEquation][1]*alpha1)/2;
    conicS2->C = Fcoeff[kEquation][2]*alpha2 - Fcoeff[jEquation][2]*alpha1;
    conicS2->D = (Fcoeff[kEquation][3]*alpha2 - Fcoeff[jEquation][3]*alpha1)/2;
    conicS2->E = (Fcoeff[kEquation][4]*alpha2 - Fcoeff[jEquation][4]*alpha1)/2;
    conicS2->F = (Fcoeff[kEquation][5]*alpha2 - Fcoeff[jEquation][5]*alpha1)-(Fcoeff[kEquation][7]*alpha2 - Fcoeff[jEquation][7]*alpha1);
    return PX_NO_ERROR;
  }
  
  //for equtaions with zero Fcoeff[i][6]
  //can use them directly for the conic equations
  if(nZeroCoeffEq >= 1){ //nNonZeroCoeffEq <= 2
    kEquation = ZeroCoeffEq[0];
    conicS2->A = Fcoeff[kEquation][0];
    conicS2->B = Fcoeff[kEquation][1]/2;
    conicS2->C = Fcoeff[kEquation][2];
    conicS2->D = Fcoeff[kEquation][3]/2;
    conicS2->E = Fcoeff[kEquation][4]/2;
    conicS2->F = Fcoeff[kEquation][5] - Fcoeff[kEquation][7];
  }
  if(nZeroCoeffEq >= 2){//nNonZeroCoeffEq <= 1
    kEquation = ZeroCoeffEq[1];
    conicS1->A = Fcoeff[kEquation][0];
    conicS1->B = Fcoeff[kEquation][1]/2;
    conicS1->C = Fcoeff[kEquation][2];
    conicS1->D = Fcoeff[kEquation][3]/2;
    conicS1->E = Fcoeff[kEquation][4]/2;
    conicS1->F = Fcoeff[kEquation][5] - Fcoeff[kEquation][7];
  }
 
  return PX_NO_ERROR;
}

/******************************************************************/
//   FUNCTION Definition: PXInitializeConic
template <class DT> ELVIS_DEVICE int
PXInitializeConic(ConicSection<DT>& conicS)
{
  conicS.type = PXE_Conic_Undetermined;
  conicS.degenerate = -1;
  conicS.A = conicS.B = conicS.C = conicS.D = conicS.E = conicS.F = 0.0;
  conicS.DegComputed = PXE_False;
  conicS.Parametrized = PXE_False;
  conicS.O[0] = conicS.O[1] = 0;
  conicS.X[0] = conicS.X[1] = 0;
  conicS.Y[0] = conicS.Y[1] = 0;
  conicS.aa = conicS.bb = 0;
  return PX_NO_ERROR;
}

/******************************************************************/
//   FUNCTION Definition: PXClassifyConic
ELVIS_DEVICE  int
PXClassifyConic(ConicSection<PX_REAL>& conicS)
{
/*-------------------------------------------------------------------*/
/*   PURPOSE: Determine the type of a conic                          */
/*                                                                   */
/*   INPUTS: conicS, a conic section                                 */
/*                                                                   */
/*   OUTPUTS: conicS with its type determined                        */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/
  
  PX_REAL &AA = conicS.A; //make a reference for coding easiness 
  PX_REAL &BB = conicS.B; 
  PX_REAL &CC = conicS.C;
  PX_REAL &DD = conicS.D;
  PX_REAL &EE = conicS.E;
  PX_REAL &FF = conicS.F;  
  PX_REAL &JJ = conicS.J;
  PX_REAL *coeff[6];
  PX_REAL mincoeff;
  PX_REAL maxcoeff;
  int k;

  /*Return if type already determined*/
  if(conicS.type != PXE_Conic_Undetermined && conicS.type < PXE_Conic_Last)
    return PX_NO_ERROR;
  
  /*Find max coefficient*/
  coeff[0] = &AA; coeff[1] = &BB; coeff[2] = &CC; 
  coeff[3] = &DD; coeff[4] = &EE; coeff[5] = &FF;
  maxcoeff = 0.0;
  mincoeff = fabs(AA);
  for(k = 0; k < 6; k ++){
    maxcoeff = MAX(maxcoeff, fabs(*(coeff[k])));
    mincoeff = MIN(mincoeff, fabs(*(coeff[k])));
  }
  if(maxcoeff == 0){
    conicS.type = PXE_Conic_ZeroConic;
    return PX_NO_ERROR;
  }

  /*Normalize, and ignore the very small coefficients,
    note this can change the type of the conic*/
  for(k = 0; k < 6; k++){
    *(coeff[k]) = *(coeff[k]) / maxcoeff;
    if(fabs(*(coeff[k])) < 100000*MEPS) 
      //note here: if for parametrization, we shouldn't change the coefficients
      //need to change the code structure a little... Huafei
      *coeff[k] = 0.0;
  }
  mincoeff /= maxcoeff;
 
  /*First case*/
  if(AA == 0 && BB == 0 && CC == 0){
    if(DD == 0 && EE == 0 && FF == 0)
      conicS.type = PXE_Conic_ZeroConic;
    else     
      conicS.type = PXE_Conic_OneLine;
    conicS.degenerate = 1;

    return PX_NO_ERROR;
  } 

  /*Compute the discriminant*/
  if(conicS.degenerate == 1)//the input conic has already been determined to be degenerate
    conicS.det = 0;
  else
    conicS.det = AA*CC*FF + 2*BB*DD*EE - DD*DD*CC - BB*BB*FF - EE*EE*AA;

  JJ = AA*CC - BB*BB;

  /*Differentiate degenerate cases and non-degenerate cases*/
  if(fabs(conicS.det) <= MEPS){ //det is a product of three numbers, could be very small
    conicS.degenerate = 1;
    conicS.det = 0.0;

    /*Differentiate degenerate cases*/
    if(JJ < 0 || AA*CC < 0) 
      //it is ok to misrecognize parallel/coincident lines as crossing lines
      conicS.type = PXE_Conic_CrossingLines;
    else if(JJ > 100*MEPS)
      conicS.type = PXE_Conic_OnePoint;
    else{      
      JJ = 0.0;
      conicS.K1 = AA*FF - DD*DD;
      conicS.K2 = CC*FF - EE*EE;
      conicS.K = conicS.K1+conicS.K2;
      
      if(conicS.K < 0)
	//it is ok to misrecognize coincident lines as parallel lines
	conicS.type = PXE_Conic_ParallelLines;
      else if(conicS.K > 100*MEPS)
	conicS.type =  PXE_Conic_ImaginaryLines;
      else{
	conicS.K = 0.0;
	conicS.type =  PXE_Conic_CoincidentLines;
      }
      
      if(fabs(conicS.K1) < 100*MEPS) conicS.K1 = 0;
      if(fabs(conicS.K2) < 100*MEPS) conicS.K2 = 0;
    }
  }
  else{
    /*Differentiate conic sections*/
    conicS.degenerate = 0;

    if(JJ < -100*MEPS)
      conicS.type = PXE_Conic_Hyperbola;
    else if (JJ > 100*MEPS){
      if(conicS.det/(AA+CC) < 0)
	conicS.type = PXE_Conic_Ellipse;
      else
	conicS.type =  PXE_Conic_ImaginaryEllipse;
    }
    else{
      JJ = 0.0;
      conicS.type = PXE_Conic_Parabola;
    }
  }
 
  return PX_NO_ERROR;  
}


/******************************************************************/
//   FUNCTION Definition: PXConic2Lines
ELVIS_DEVICE int
PXCheckConic2Lines(ConicSection<PX_REAL> const& conicS, PX_REAL const& tol, 
		   enum PXE_Boolean &correct)
/*-------------------------------------------------------------------*/
/*   PURPOSE: Verifies if the line equations obtained are correct    */
/*                                                                   */
/*   INPUTS: conicS, a degenerate conic section                      */
/*           tol: tolerance                                          */
/*                                                                   */
/*   OUTPUTS: correct: whether it is correct                         */
/*   Degenerate cases:                                               */
/*                     crossing lines: (Ax+by+c)*(x+ey+f)=0          */
/*                     parallel lines: (ax+by+c)*(ax+by+f)=0         */
/*                     one line: (ax+by+c)^2=0                       */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/
{
  PX_REAL const& AA = conicS.A; //make a reference for coding easiness 
  PX_REAL const& BB = conicS.B; 
  PX_REAL const& CC = conicS.C;
  PX_REAL const& DD = conicS.D;
  PX_REAL const& EE = conicS.E;
  PX_REAL const& FF = conicS.F; 
  PX_REAL coeff[6];
  int k;

  /*Make sure line coefficients have been computed*/
  if(conicS.DegComputed != PXE_True || conicS.degenerate != 1)
    PXErrorReturn(PX_BAD_INPUT);

  switch(conicS.type){
    case PXE_Conic_OneLine:
    case PXE_Conic_ImaginaryLines:
    case PXE_Conic_ZeroConic:
    case PXE_Conic_OnePoint:
      correct = PXE_True;
      return PX_NO_ERROR;
    case PXE_Conic_CrossingLines:
      coeff[0] = AA;
      coeff[1] = AA*conicS.e+conicS.b;
      coeff[2] = conicS.b*conicS.e;
      coeff[3] = AA*conicS.f+conicS.c;
      coeff[4] = conicS.c*conicS.e+conicS.b*conicS.f;
      coeff[5] = conicS.c*conicS.f;
      break;
    case PXE_Conic_ParallelLines:
      coeff[0] = conicS.a*conicS.a;
      coeff[1] = 2*conicS.a*conicS.b;
      coeff[2] = conicS.b*conicS.b;
      coeff[3] = conicS.a*(conicS.c+conicS.f);
      coeff[4] = conicS.b*(conicS.c+conicS.f);
      coeff[5] = conicS.c*conicS.f;
    
      if(AA < 0 || (AA == 0 && CC < 0)){
	for(k = 0; k < 6; k++) coeff[k] = -coeff[k];
      }
      break;
    case PXE_Conic_CoincidentLines:
      coeff[0] = conicS.a*conicS.a;
      coeff[1] = 2*conicS.a*conicS.b;
      coeff[2] = conicS.b*conicS.b;
      coeff[3] = 2*conicS.a*conicS.c;
      coeff[4] = 2*conicS.b*conicS.c;
      coeff[5] = conicS.c*conicS.c;
      
      if(AA < 0 || (AA == 0 && CC < 0)){
	for(k = 0; k < 6; k++) coeff[k] = -coeff[k];
      }
      break;
    default:
      ELVIS_PRINTF("Unrecognized type\n");
      PXErrorReturn(PX_BAD_INPUT);
  }
 
  if(fabs(coeff[0] - AA) > tol*fabs(AA)
     || fabs(coeff[1] - 2*BB) > tol*2*fabs(BB)
     || fabs(coeff[2] - CC) > tol*fabs(CC) 
     || fabs(coeff[3] - 2*DD) > tol*2*fabs(DD)
     || fabs(coeff[4] - 2*EE) > tol*2*fabs(EE) 
     || fabs(coeff[5] - FF) > tol*fabs(FF)){
    
    // ELVIS_PRINTF("type = %s\n",  PXE_ConicSectionTypeName[conicS.type]);
    // ELVIS_PRINTF("A = %.15e, B = %.15e, C = %.15e, D = %.15e, E = %.15e, F = %.15e\n", conicS.A, conicS.B, conicS.C, conicS.D, conicS.E, conicS.F);
    // ELVIS_PRINTF("plotconicsection(A,B,C,D,E,F)\n");
    // switch(conicS.type){
    //   case PXE_Conic_CrossingLines:
    // 	ELVIS_PRINTF("hold on\n");
    // 	ELVIS_PRINTF("ezplot('%.15e*x+%.15e*y+%.15e=0');\n", AA,conicS.b,conicS.c);
    // 	ELVIS_PRINTF("ezplot('x+%.15e*y+%.15e=0');\n",conicS.e,conicS.f);
    // 	break;
    // default:
    //   break;
    // }
    // ELVIS_PRINTF("%.15e %.15e %.15e %.15e %.15e %.15e\n", fabs(coeff[0] - AA),fabs(coeff[1] - 2*BB), fabs(coeff[2] - CC), fabs(coeff[3] - 2*DD), fabs(coeff[4] - 2*EE), fabs(coeff[5] - FF));
   
    correct = PXE_False;
  }
  else correct = PXE_True;

  return PX_NO_ERROR;
}



/******************************************************************/
//   FUNCTION Definition: PXConic2Lines
ELVIS_DEVICE int
PXConic2Lines(ConicSection<PX_REAL>& conicS)
{
/*-------------------------------------------------------------------*/
/*   PURPOSE: Obtains the line equations for a degenerate conic      */
/*            Function overloading for double precisiion             */
/*                                                                   */
/*   INPUTS: conicS, a degenerate conic section                      */
/*                                                                   */
/*   OUTPUTS: conicS with the line equation determined               */
/*   Degenerate cases:                                               */
/*                     crossing lines: (Ax+by+c)*(x+ey+f)=0          */
/*                     parallel lines: (ax+by+c)*(ax+by+f)=0         */
/*                     one line: (ax+by+c)^2=0                       */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/

  PX_REAL &AA = conicS.A; //make a reference for coding easiness 
  PX_REAL &BB = conicS.B; 
  PX_REAL &CC = conicS.C;
  PX_REAL &DD = conicS.D;
  PX_REAL &EE = conicS.E;
  PX_REAL &FF = conicS.F;  
  PX_REAL temp, temp2;
  bool SignFlipped = false;
  enum PXE_Boolean correct;


  /*If the degenerate coefficients have been computed*/
  if(conicS.DegComputed == PXE_True)
    return PX_NO_ERROR;

  /*Only works for degenerate conics*/
  if(conicS.degenerate != 1){
    ELVIS_PRINTF("The conic should be degenerate!\n");
    return PXError(PX_BAD_INPUT);
  }

  /*Find the type if it is undetermined*/
  if(conicS.type == PXE_Conic_Undetermined)
    PXErrorReturn(PXClassifyConic(conicS));
  
  /*Flag*/
  conicS.DegComputed = PXE_True;
  
  /*First case is if A=B=C=0*/
  if(conicS.type == PXE_Conic_OneLine){    
    conicS.a = DD;
    conicS.b = EE;
    conicS.c = FF/2;
    return PX_NO_ERROR;
  }   
  
  /*Distinguish difference cases if at least one of A,B,C is nonzero*/
  /*Note: it is assumed that if type has been determined, 
    the conic determinants must have also been determined*/
  if(conicS.type == PXE_Conic_CrossingLines){//2 intersectin lines
    //(Ax+by+c)(x+ey+f) = Ax^2+2Bxy+Cy^2+2Dx+2Ey+F
    //solve for b,c,e,f       
    if(fabs(AA) <= 100*MEPS){ //one line is vertical
      conicS.b = 2*BB;
      conicS.e = CC/conicS.b;
      conicS.c = 2*DD;
      if(conicS.e > 1e-5 || conicS.e < -1e-5)
	conicS.f = FF/conicS.c;
      else
	conicS.f = (EE-conicS.e*DD)/BB;
    }
    else{
      temp = sqrt(-conicS.J);
      conicS.b = BB-temp;
      conicS.e = (BB+temp)/AA;
      conicS.f = (DD*conicS.e-EE)/temp;
      if(conicS.f > 1e-5 || conicS.f < -1e-5)
	conicS.c = FF/conicS.f; //this line seems to better conditioned than the next
      else
	conicS.c = 2*DD-AA*conicS.f;
    }
  }
  else if(conicS.type == PXE_Conic_OnePoint){//one point    
    conicS.xp = (BB*EE-CC*DD)/conicS.J;
    conicS.yp = (BB*DD-AA*EE)/conicS.J;
  }
  else{ //AA*CC=BB^2 >=0, delta == 0
    //this code works only for A >= 0
    //Note: if AA>0, then CC>=0
    if(AA < 0 || (AA == 0 && CC < 0)){
      AA = -AA; BB = -BB; CC = -CC; DD = -DD; EE = -EE; FF = -FF;
      SignFlipped = true;
      
      if(CC < 0){
	ELVIS_PRINTF("The arithmetic for sign comparison is not right!\n");
	ELVIS_PRINTF("%.15e %.15e %.15e %.15e %.15e %.15e\n", AA, BB, CC, DD, EE, FF);
	return PXError(PX_CODE_FLOW_ERROR);
      }
    }

    if(conicS.type == PXE_Conic_ParallelLines){//parallel lines
      //(ax+by+c)(ax+by+f) = Ax^2+2Bxy+Cy^2+2Dx+2Ey+F for A>=0
      //solve for a,b,c,f    
      //ELVIS_PRINTF("%.15e %.15e %.15e %.15e %.15e %.15e\n", AA, BB, CC, DD, EE, FF);
      conicS.a = sqrt(AA); 
      conicS.b = (BB < 0) ? -sqrt(CC) : sqrt(CC);
      if(conicS.K1<=0 && AA!=0){
	temp2 = sqrt(-conicS.K1);
	conicS.c = (DD+temp2)/conicS.a;
	conicS.f = (DD-temp2)/conicS.a;	
      }
      else if(conicS.K2<=0 && CC!=0){
	temp2 = sqrt(-conicS.K2);
	conicS.c = (EE+temp2)/conicS.b;
	conicS.f = (EE-temp2)/conicS.b;
      }
      else{
	ELVIS_PRINTF("The arithmetic for sign comparison is not right!\n");
	ELVIS_PRINTF("%.15e %.15e %.15e %.15e %.15e %.15e\n", AA, BB, CC, DD, EE, FF);
	ELVIS_PRINTF("%.15e\n", AA*CC*FF + 2*BB*DD*EE - DD*DD*CC - BB*BB*FF - EE*EE*AA);
	return PXError(PX_CODE_FLOW_ERROR);
      }
    }
    else if(conicS.type ==  PXE_Conic_CoincidentLines){//one line, temp == 0
      //(ax+by+c)^2 = Ax^2+2Bxy+Cy^2+2Dx+2Ey+F for A>=0
      //solve for a,b,c
      conicS.a = sqrt(AA);
      conicS.b = (BB < 0) ? (-sqrt(CC)):(sqrt(CC));
      if(AA!=0)
	conicS.c = DD/conicS.a;
      else if(CC!=0)
	conicS.c = EE/conicS.b;
    }
    else if(conicS.type ==  PXE_Conic_ImaginaryLines
	    || conicS.type == PXE_Conic_ZeroConic){
      //don't need to do antyhing
    }
    else{
      ELVIS_PRINTF("type not correct! %d\n", conicS.type);
      return PXError(PX_CODE_FLOW_ERROR);
    }
    
    /*Recover the original coefficients*/
    if(SignFlipped){
      AA = -AA; BB = -BB; CC = -CC; DD = -DD; EE = -EE; FF = -FF;
    }      
  }
  
  /*Verify if it is correct*/
  PXErrorReturn(PXCheckConic2Lines(conicS, 1e0, correct)); 
  //the tolerance isn't set very high, because it is found that
  //even the product of coefficients don't match very well
  //the line close to the reference triangle is still fine 
  //need a better way to solve for line coeff, Huafei
  if(correct != PXE_True){
    //for the cases that fail, it is often the case where A, B, C, are small
    //and when this happens, b is usually small but c is order 1, and making this line 
    //far from referenc triangle
    if(conicS.type == PXE_Conic_CrossingLines &&
       fabs(AA) < 1e-5 && fabs(conicS.b) < 1e-5 && fabs(conicS.c) > 1e-4){
      
      if(fabs(conicS.e)<1e3){
	PX_REAL coeff[3];
	enum PXE_QuadraticEquation_Type type;
	int nsol;
	PX_REAL xsol[2];
	PX_REAL ss, tt;

	//solve for intersection with y = 0
	coeff[2] = AA;
	coeff[1] = 2*DD;
	coeff[0] = FF;
	PXErrorReturn(PXSolveQuadraticRoots(coeff, type, nsol, xsol));
	if(nsol != 2) return PX_CODE_FLOW_ERROR;
	tt = (fabs(xsol[0]) < fabs(xsol[1])) ? xsol[0] : xsol[1];
	
	//solve for intersection with y = 1
	coeff[2] = AA;
	coeff[1] = 2*BB+2*DD;
	coeff[0] = CC+2*EE+FF;
	PXErrorReturn(PXSolveQuadraticRoots(coeff, type, nsol, xsol));
	if(nsol != 2) return PX_CODE_FLOW_ERROR;
	ss = (fabs(xsol[0]) < fabs(xsol[1])) ? xsol[0] : xsol[1];
	
	//equation
	conicS.e = -ss+tt;
	conicS.f = -tt;
	
	//PXErrorReturn(PXCheckConic2Lines(conicS, 1e0, correct)); 
	//if(correct != PXE_True) return PX_CODE_FLOW_ERROR;
      }
      else return PX_CODE_FLOW_ERROR;
    }
    else
      //otherwise, don't know how to make it more correct
      return PX_CODE_FLOW_ERROR;
  }
   
  return PX_NO_ERROR;
}


/******************************************************************/
//   FUNCTION Definition: PXDetermineLineIntersectReferenceTriangle
template <class DT> ELVIS_DEVICE int
PXDetermineLineIntersectReferenceTriangle(const DT& a, const DT& b, const DT& c, 
					  bool& intersect)
{
/*-------------------------------------------------------------------*/
/*   PURPOSE: Determine whether a line intersects with the reference */
/*            triangle                                               */
/*            if the line is the same as one edge of the triangle,   */
/*            this function will return true                         */
/*                                                                   */
/*   INPUTS: a,b,c, the line coefficients: ax+by+c=0                 */
/*                                                                   */
/*   OUTPUTS: intersect: whether intersects or not with the ref tri  */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/

   DT temp;

  /*Determine if the y-intercept (x=0) is within [0,1]*/
  if(b!=0){
    temp = -c/b;
    if(temp <= 1 && temp >= 0){
      intersect = true;
      return PX_NO_ERROR;
    }
  }
  
  /*Determine if the x-intercept (y=0) is within [0,1]*/
  if(a!=0){
    temp = -c/a;
    if(temp <= 1 && temp >= 0){
      intersect = true;
      return PX_NO_ERROR;
    }
  }
  
  /*If the x- and y- intercepts are both outside [0,1], 
    then no intersection with the reference triangle*/
  intersect = false;
  return PX_NO_ERROR;
}



/******************************************************************/
//   FUNCTION Definition: PXLineLineIntersectInReferenceTriangle
template <class DT> ELVIS_DEVICE int
PXLineLineIntersectInReferenceTriangle(const DT& a1, const DT& b1, const DT& c1,
				       const DT& a2, const DT& b2, const DT& c2,
				       int &nIntersect, DT* xIntersect, 
				       enum PXE_RefElemIntersectType &boundaryType, 
				       int &boundaryIndex)
{
/*-------------------------------------------------------------------*/
/*   PURPOSE: Determine if two lines intersect within the reference  */
/*            triangle, if they do, find the intersection coordinate */
/*                                                                   */
/*   INPUTS: a1,b1,c1, the first line coefficients: a1x+b1y+c1=0     */
/*           a2,b2,c2, the first line coefficients: a2x+b2y+c2=0     */
/*           a1,b1 (and a2,b2) should not be both zero               */
/*                                                                   */
/*   OUTPUTS: nIntersect: number of intersections inside the ref     */
/*                        = -1 if infinitely many intersections      */
/*            xIntersect: the coordinate of intersection             */
/*            boundaryType: whether the intersection is at reference */
/*                          triangle boundary                        */
/*                                 = 1 if inside ref triangle        */
/*                                 = 2 if on ref triangle edge       */
/*                                 = 3 if on ref triangle vertex     */
/*            boundaryIndex: the index on the ref triangle boundary  */
/*                           either local face or local vertex       */
/*                           number                                  */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/
  
   DT temp1, temp2, temp3;//for solving for the intersection point
   DT det;//for solving for the intersection point
  bool lineInTriangle;//whether the given line intersects the reference tri

  /*Determine if the two lines intersect with the reference triangle first*/
  PXErrorReturn(PXDetermineLineIntersectReferenceTriangle<DT>(a1, b1, c1, lineInTriangle));
  if(!lineInTriangle){
    nIntersect = 0;
    return PX_NO_ERROR;
  }
  PXErrorReturn(PXDetermineLineIntersectReferenceTriangle<DT>(a2, b2, c2, lineInTriangle));
  if(!lineInTriangle){
    nIntersect = 0;
    return PX_NO_ERROR;
  }
  
  /*Now both lines intersect the reference triangle, 
    determine if their intersection is inside the reference triangle*/
  temp1 = b1*c2 - b2*c1;
  temp2 = a2*c1 - a1*c2;
  det = a1*b2 - a2*b1;
  temp3 = temp1+temp2;
  //the intersection is [x,y] = [temp1/det, temp2/det]
  
  /*Determine if x>=0, y>=0, x+y<=1*/
  /*Note: for exact arithmetics, for the expression whose sign needs to 
    be determined, try to avoid division in the expression */
  if(det == 0){//no intersection or infinitely many solutions
    if(temp1 == 0 && temp2 == 0){ //infinitely many solutions, i.e. two lines are the same
      nIntersect = -1;
      return PX_NO_ERROR;
    }
    else{//no solution
      nIntersect = 0;
      return PX_NO_ERROR;
    }
  }
  else if(det > 0){
    if(temp1<0 || temp2<0 || temp3>det){
      nIntersect = 0;
      return PX_NO_ERROR;
    }
  }
  else{
    if(temp1>0 || temp2>0 || temp3<det){
      nIntersect = 0;
      return PX_NO_ERROR;
    }
  }
  
  /*At this point, there is intersection inside the reference triangle or on its boundary*/
  nIntersect = 1;
  if(temp1 == 0){//x=0
    if(temp2 == 0){//intersection at (0,0)
      boundaryType = PXE_RefElementVertex;
      boundaryIndex = 0;
      xIntersect[0] = 0;
      xIntersect[1] = 0;
    }
    else if(temp2 == det){//intersection at (0,1)
      boundaryType = PXE_RefElementVertex;
      boundaryIndex = 2;
      xIntersect[0] = 0;
      xIntersect[1] = 1;
    }
    else{//intersection on the edge of x=0
      boundaryType = PXE_RefElementEdge;
      boundaryIndex = 1;
      xIntersect[0] = 0;
      xIntersect[1] = temp2/det;
    }
  }
  else if(temp2 == 0){//y=0
    if(temp1 == det){//intersection at (1,0)
      boundaryType = PXE_RefElementVertex;
      boundaryIndex = 1;
      xIntersect[0] = 1;
      xIntersect[1] = 0;
    }
    else{//intersection on the edge of y=0
      boundaryType = PXE_RefElementEdge;
      boundaryIndex = 2;
      xIntersect[0] = temp1/det;
      xIntersect[1] = 0;
    }
  }
  else if(temp3 == det){//intersection on the edge of x+y=1
    boundaryType = PXE_RefElementEdge;
    boundaryIndex = 0;
    xIntersect[0] = temp1/det;
    xIntersect[1] = 1-temp1/det;
  }
  else{//intersection inside the reference triangle
    boundaryType = PXE_RefElementInterior;
    boundaryIndex = -1;
    xIntersect[0] = temp1/det;
    xIntersect[1] = temp2/det;
  }    
  
  return PX_NO_ERROR;
}



/******************************************************************/
//   FUNCTION Definition: PXConicLineIntersectInReferenceTriangle
template <class DT> ELVIS_DEVICE int
PXConicLineIntersectInReferenceTriangle(ConicSection<DT>& conicS, 
					const DT& aa, const DT& bb, const DT& cc, 
					int& nIntersect, DT* xIntersect,
					enum PXE_RefElemIntersectType boundaryType[2], 
					int boundaryIndex[2], int multiplicity[2])
{
/*-------------------------------------------------------------------*/
/*   PURPOSE: Determine whether a conic intersects a given line      */
/*            within the reference triangle                          */
/*                                                                   */
/*   INPUTS: conicS, the input conic section, its type or degeneracy */
/*             might be determined in this function                  */
/*           aa,bb,cc, the input line   aa*x+bb*y+cc=0               */
/*                                                                   */
/*   OUTPUTS: nIntersect: number of intersections                    */
/*                        = -1 if infinitely many intersections      */
/*                        at least one pt inside reference triangle  */
/*                        including the case where only one branch of*/
/*                        each conic is the same (this happens only  */
/*                        when they both are two lines               */
/*            xIntersect: the intersection coorindate                */
/*            boundaryType: whether the intersection is at reference */
/*                                 triangle boundary                 */
/*                                 = 1 if inside ref triangle        */
/*                                 = 2 if on ref triangle edge       */
/*                                 = 3 if on ref triangle vertex     */
/*            boundaryIndex: the index on the ref triangle boundary  */
/*                           either local face or local vertex       */
/*                           number                                  */
/*            multiplicity: intersection multiplicity                */
/*           Note: if ConicS is degenerate to lines, multiplicity is */
/*                 defined to always one                             */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/


  const DT &AA = conicS.A; //make a reference for coding easiness 
  const DT &BB = conicS.B; 
  const DT &CC = conicS.C;
  const DT &DD = conicS.D;
  const DT &EE = conicS.E;
  const DT &FF = conicS.F;  
  bool lineInTriangle; //whether the given line intersects the reference triangle
  int linelineIntersect; //for conicS being degenerate, 
                         //# of intersections between the degenerate line and the given line
   DT coeff[3]; //coefficients for the quadratic equation for the intersection pt
   DT asq, ac, ab; //aa*aa, aa*ac, aa*ab
  
   DT xsol[2], ysol[2]; //solution of x- and y-coordinate from the quadratic equation
  int count; //number of intersection points inside and outside the ref triangle
  enum PXE_QuadraticEquation_Type type; //whether the formed quadratic equation is degenerate
  int kRoot;//index for the quadratic roots
  int boundary;//whether the quadratic root is on the boundary of the ref triangle
  int xsolBoundary[2]; // = 0 if x = 0 is a sol, = 1  if x = 1 is a sol, = -1 otherwise
  int ysolBoundary[2]; // = 0 if y = 0 is a sol, = 1  if y = 1 is a sol, = -1 otherwise
  int rootToSolve;//which root of the quadratic equation is needed
   DT temp, temp2;

  /*Initialize*/
  nIntersect = 0;

  /*If the line doesn't intersect the triangle, then no intersection needs to be solved*/
  PXErrorReturn(PXDetermineLineIntersectReferenceTriangle<DT>(aa, bb, cc, lineInTriangle));
  if(!lineInTriangle)
    return PX_NO_ERROR;
  /*returns true if this line is the same as one of the reference triangle edges*/

  /*If the conic has been determined to be degenerate lines, just use line-line intersection*/
  /*Note: if the conic is degenerate, but its type has not been determiend, then this conic
    is treated as general conic because finding its degenerate line coefficients also need to solve
    quadratic equations*/
  if(conicS.DegComputed == PXE_True){
    
    /*If the conic is two crossing lines*/
    if(conicS.type ==  PXE_Conic_CrossingLines){
      //check intersection with 1st line
      PXErrorReturn(PXLineLineIntersectInReferenceTriangle<DT>(aa, bb, cc,
							       conicS.A, conicS.b, conicS.c,
							       linelineIntersect, xIntersect,
							       boundaryType[0], boundaryIndex[0]));
      if(linelineIntersect > 0){
	multiplicity[nIntersect] = 1;
	nIntersect++;
      }
      else if(linelineIntersect < 0){//infinitely many solutions
	nIntersect = -1;
	return PX_NO_ERROR;
      }
      
      //check intersection with 2nd line
      PXErrorReturn(PXLineLineIntersectInReferenceTriangle<DT>(aa, bb, cc,
							       1, conicS.e, conicS.f,
							       linelineIntersect, xIntersect+2*nIntersect,
							       boundaryType[nIntersect], boundaryIndex[nIntersect]));
      if(linelineIntersect > 0){
	multiplicity[nIntersect] = 1;
	nIntersect++;
	//make sure the two intersections are different
	if(nIntersect == 2){
	  if(xIntersect[0] == xIntersect[2] && xIntersect[1] == xIntersect[3]){
	    nIntersect --;
	    ELVIS_PRINTF("This case should not happen in a real intersection case\n");
	    //because it does not make sense to have crossing lines inside the patch...
	  }
	}
      }
      else if(linelineIntersect < 0){//infinitely many solutions
	nIntersect = -1;
	return PX_NO_ERROR;
      }
    }
    /*If the conic is two parallel lines*/
    else if(conicS.type ==  PXE_Conic_ParallelLines){
      //check intersection with 1st line
      PXErrorReturn(PXLineLineIntersectInReferenceTriangle<DT>(aa, bb, cc,
							       conicS.a, conicS.b, conicS.c,
							       linelineIntersect, xIntersect,
							       boundaryType[0], boundaryIndex[0]));
      if(linelineIntersect > 0){
	multiplicity[nIntersect] = 1;
	nIntersect++;
      }
      else if(linelineIntersect < 0){//infinitely many solutions
	nIntersect = -1;
	return PX_NO_ERROR;
      }
      
      //check intersection with 2nd line
      PXErrorReturn(PXLineLineIntersectInReferenceTriangle<DT>(aa, bb, cc,
							       conicS.a, conicS.b, conicS.f,
							       linelineIntersect, xIntersect+2*nIntersect,
							       boundaryType[nIntersect], boundaryIndex[nIntersect]));

      if(linelineIntersect > 0){
	multiplicity[nIntersect] = 1;
	nIntersect++;
      }
      else if(linelineIntersect < 0){//infinitely many solutions
	nIntersect = -1;
	return PX_NO_ERROR;
      }
    }
    /*If the conic is one line*/
    else if(conicS.type ==  PXE_Conic_OneLine || conicS.type ==  PXE_Conic_CoincidentLines){
      PXErrorReturn(PXLineLineIntersectInReferenceTriangle<DT>(aa, bb, cc,
							       conicS.a, conicS.b, conicS.c,
							       linelineIntersect, xIntersect,
							       boundaryType[0], boundaryIndex[0]));
      if(linelineIntersect > 0){
	multiplicity[nIntersect] = 1;
	nIntersect++;
      }
      else if(linelineIntersect < 0){//infinitely many solutions
	nIntersect = -1;
	return PX_NO_ERROR;
      }
    }
    else{
      ELVIS_PRINTF("conicS type is not right! %d\n", conicS.type);
      return PXError(PX_BAD_INPUT);
    }
  }
  /*For a conic-line intersection, need to solve a quadratic equation*/
  else{
    /*The result quadratic equation is different when a=0 or not*/
    if(aa!=0){          
      /*Compute the coefficients*/
      asq = aa*aa; ab = aa*bb; ac = aa*cc;
      coeff[2] = CC*asq+AA*bb*bb-2*BB*ab;
      coeff[1] = 2*(EE*asq-BB*ac-DD*ab+AA*bb*cc);
      coeff[0] = FF*asq+AA*cc*cc-2*DD*ac;

      //ELVIS_PRINTF("coeff = %.15e %.15e %.15e\n", coeff[2], coeff[1], coeff[0]);
   
      /*Count number of real roots in [0,1]*/
      //actually only need to count number of real roots in [0,\infty) at this stage
      PXErrorReturn(PXCountQuadraticRootsbySolve<DT>(coeff, 
						     type, count, boundary, rootToSolve, ysol));   
 
      /*Count number of intersection points within the reference triangle,
       i.e., x>=0, y>=0, x+y<=1*/
      if(count == 0){
	nIntersect = 0;
	return PX_NO_ERROR;
      }
      else if(count == -1){//infinitely many solutions, i.e., the given line coincides with the conic
	nIntersect = -1;
	return PX_NO_ERROR;
      }
      else if(count == 1 || count == 2){
	nIntersect = 0;

	//determine y = 0 or y = 1
	switch(boundary){
          case -1://set the boundary values anyway even though there might be no solution
	    ysolBoundary[0] = -1;
	    ysolBoundary[1] = -1;
	    break;
          case 0:
	    ysolBoundary[0] = 0;//y = 0
	    ysolBoundary[1] = -1;
	    break;
          case 1:
	    ysolBoundary[0] = 1; //y = 1 
	    ysolBoundary[1] = -1;
	    break;
          case 2:
	    ysolBoundary[0] = 0; //y = 0
	    ysolBoundary[1] = 1; //y = 1
	    break;
          default:
	    ELVIS_PRINTF("boundary of roots is not correct!\n");
	    return PXError(PX_CODE_FLOW_ERROR);
	}
  
	/*Check each root*/
	for(kRoot = 0; kRoot < count; kRoot ++){
	  //if -x>0, then don't need this root
	  temp = aa*(bb*ysol[kRoot] + cc);
	  if(temp >0){
	    continue;
	  }
	  else if(temp == 0){ //x=0
	    xIntersect[2*nIntersect] = 0;
	    
	    if(ysolBoundary[nIntersect] == 0){ //y=0
	      boundaryType[nIntersect] = PXE_RefElementVertex;
	      boundaryIndex[nIntersect] = 0;
	      xIntersect[2*nIntersect+1] = 0;
	    }
	    else if(ysolBoundary[nIntersect] == 1){//y =1
	      boundaryType[nIntersect] = PXE_RefElementVertex;
	      boundaryIndex[nIntersect] = 2;
	      xIntersect[2*nIntersect+1] = 1;
	    }
	    else{ //on the edge of x = 0
	      boundaryType[nIntersect] = PXE_RefElementEdge;
	      boundaryIndex[nIntersect] = 1;
	      xIntersect[2*nIntersect+1] = ysol[kRoot];
	    }
	    
	    //multiplicity
	    PXErrorReturn(PXSetConicLineMultiplicity(type, multiplicity[nIntersect]));
	    
	    nIntersect++;
	    continue;
	  }

	  //if x+y-1>0, then don't need this root
	  temp2 = asq*(ysol[kRoot]-1)-temp;
	  if(temp2>0)
	    continue;
	  else if(temp2 == 0){//x+y = 1
	    if(ysolBoundary[nIntersect] == 0){ //y=0
	      boundaryType[nIntersect] = PXE_RefElementVertex;
	      boundaryIndex[nIntersect] = 1;
	      xIntersect[2*nIntersect] = 1;
	      xIntersect[2*nIntersect+1] = 0;
	    }
	    else{//on the edge of x+y = 1
	         //note: x = 0, y = 1 case should already be handled in the branch temp = 0
	      boundaryType[nIntersect] = PXE_RefElementEdge;
	      boundaryIndex[nIntersect] = 0;
	      xIntersect[2*nIntersect+1] = ysol[kRoot];
	      xIntersect[2*nIntersect] = 1 - xIntersect[2*nIntersect+1];	      
	    }
	    
	    //multiplicity
	    PXErrorReturn(PXSetConicLineMultiplicity(type, multiplicity[nIntersect]));

	    nIntersect++;
	    continue;
	  }
	  
	  //check if y = 0, note: (1,0) and (0,0) should already be handled
	  if(ysolBoundary[nIntersect] == 0){//on the edge of y = 0
	    boundaryType[nIntersect] = PXE_RefElementEdge;
	    boundaryIndex[nIntersect] = 2;
	    xIntersect[2*nIntersect] =  -temp/asq;
	    xIntersect[2*nIntersect+1] = 0;	    
	  }
	  else{
	    //at this point, the solution is inside the reference triangle
	    boundaryType[nIntersect] = PXE_RefElementInterior;
	    xIntersect[2*nIntersect] = -temp/asq;//(-bb*ysol[kRoot] - cc)/aa;
	    xIntersect[2*nIntersect+1] = ysol[kRoot];	
	  }
 	  
	  //multiplicity
	  PXErrorReturn(PXSetConicLineMultiplicity(type, multiplicity[nIntersect]));
	  nIntersect++;
	}
      }
      else{
	ELVIS_PRINTF("Number of quadratic roots is not right!\n");
	return PXError(PX_CODE_FLOW_ERROR);
      }
    }
    else{//when aa = 0, then by+c=0
      ysol[0] = -cc/bb;
      /*needs only y such that 0<=y<=1*/
      if(ysol[0] > 1 || ysol[0] < 0){
	nIntersect = 0;
	return PX_NO_ERROR;
      }
      else if(ysol[0] == 0)
	ysolBoundary[0] = 0;
      else if(ysol[0] == 1)
	ysolBoundary[0] = 1;
      else
	ysolBoundary[0] = -1;
      
      /*compute the coeffcient for the quadratic equation*/
      coeff[2] = AA;
      coeff[1] = 2*(BB*ysol[0]+DD);
      coeff[0] = ysol[0]*(CC*ysol[0]+2*EE)+FF;
      
      /*Count number of real roots in [0,1]*/
      //actually only need to count number of real roots in [0,\infty)
      PXErrorReturn(PXCountQuadraticRootsbySolve<DT>(coeff,
						     type, count, boundary, rootToSolve, xsol)); 
      
      /*Count number of intersection points within the reference triangle,
	i.e., x>=0, y>=0, x+y<=1*/
      if(count == 0){
	nIntersect = 0;
	return PX_NO_ERROR;
      }
      else if(count == -1){//infinitely many solutions, i.e., the given line coincides with the conic
	nIntersect = -1;
	return PX_NO_ERROR;
      }
      else if(count == 1 || count == 2){
	nIntersect = 0;

	//determine x = 0 or x = 1
	switch(boundary){
          case -1://set the boundary values anyway even though there might be no solution
	    xsolBoundary[0] = -1;
	    xsolBoundary[1] = -1;
	    break;
          case 0:
	    xsolBoundary[0] = 0;//x = 0
	    xsolBoundary[1] = -1;
	    break;
          case 1:
	    xsolBoundary[0] = 1; //x = 1 
	    xsolBoundary[1] = -1;
	    break;
          case 2:
	    xsolBoundary[0] = 0; //x = 0
	    xsolBoundary[1] = 1; //x = 1
	    break;
	  default:
	    ELVIS_PRINTF("boundary of roots is not correct!\n");
	    return PXError(PX_CODE_FLOW_ERROR);
	}
		
	for(kRoot = 0; kRoot < count; kRoot ++){
	  //if x+y-1>0, then don't need this root	
	  temp = xsol[kRoot] + ysol[0];
	  if(temp > 1)
	    continue;
	  else if(temp == 1){//x+y=1
	    if(ysolBoundary[0] == 0){//y=0,x=1
	      boundaryType[nIntersect] = PXE_RefElementVertex;
	      boundaryIndex[nIntersect] = 1;
	      xIntersect[2*nIntersect] = 1;
	      xIntersect[2*nIntersect+1] = 0;
	    }
	    else if(ysolBoundary[0] == 1){//y=1,x=0
	      boundaryType[nIntersect] = PXE_RefElementVertex;
	      boundaryIndex[nIntersect] = 2;
	      xIntersect[2*nIntersect] = 0;
	      xIntersect[2*nIntersect+1] = 1;
	    }
	    else{//on the edge of x+y=1
	      boundaryType[nIntersect] = PXE_RefElementEdge;
	      boundaryIndex[nIntersect] = 0;
	      xIntersect[2*nIntersect] = 1.0 - ysol[0];
	      xIntersect[2*nIntersect+1] = ysol[0];
	    }
	    
	    //multiplicity
	    PXErrorReturn(PXSetConicLineMultiplicity(type, multiplicity[nIntersect]));
	  	    
	    nIntersect++;
	    continue;
	  }
	  
	  /*x = 0*/
	  if(xsolBoundary[nIntersect] == 0){//x = 0
	    xIntersect[2*nIntersect] = 0;
	    if(ysolBoundary[0] == 0){
	      boundaryType[nIntersect] = PXE_RefElementVertex;
	      boundaryIndex[nIntersect] = 0;
	      xIntersect[2*nIntersect+1] = 0;
	    }
	    else{ //on the edge of x = 0, note: (x=0,y=1) should already be handled
	      boundaryType[nIntersect] = PXE_RefElementEdge;
	      boundaryIndex[nIntersect] = 1;
	      xIntersect[2*nIntersect+1] = ysol[0];
	    }
	    
	    //multiplicity
	    PXErrorReturn(PXSetConicLineMultiplicity(type, multiplicity[nIntersect]));
	    
	    nIntersect++;
	    continue;
	  }
	  
	  /*y = 0*/
	  if(ysolBoundary[0] == 0){//y = 0
	    //note (x=0,y=0) and (x=1,y=0) should already be handled
	    boundaryType[nIntersect] = PXE_RefElementEdge;
	    boundaryIndex[nIntersect] = 2;
	    xIntersect[2*nIntersect] = xsol[kRoot];
	    xIntersect[2*nIntersect+1] = 0;
	    
	    //multiplicity
	    PXErrorReturn(PXSetConicLineMultiplicity(type, multiplicity[nIntersect]));
	    
	    nIntersect++;
	    continue;
	  }
	  
	  //at this point, the solution is inside the reference triangle
	  boundaryType[nIntersect] = PXE_RefElementInterior;
	  xIntersect[2*nIntersect] = xsol[kRoot];
	  xIntersect[2*nIntersect+1] = ysol[0];	
	  
	  //multiplicity
	  PXErrorReturn(PXSetConicLineMultiplicity(type, multiplicity[nIntersect]));

	  nIntersect++;
	}
      }
      else{
	ELVIS_PRINTF("Number of quadratic roots is not right!\n");
	return PXError(PX_CODE_FLOW_ERROR);
      }
    }//end when aa=0
  }//end when conic is not degenerate
  
  return PX_NO_ERROR;
}



/******************************************************************/
//   FUNCTION Definition: PXConicDegConicIntersectInReferenceTriangle
template <class DT> ELVIS_DEVICE int 
PXConicDegConicIntersectInReferenceTriangle(ConicSection<DT>& conicS1, 
					    ConicSection<DT>& conicS2, 
					    int& nIntersect, DT* XIntersect,
					    enum PXE_RefElemIntersectType boundaryType[4], 
					    int boundaryIndex[4], int multiplicity[4])
{
/*-------------------------------------------------------------------*/
/*   PURPOSE: Determine whether a conic intersects another conic that*/
/*            is degenerate within the reference triangle            */
/*                                                                   */
/*   INPUTS: conicS1, the first input conic section, its type or     */
/*           degeneracy might be determined in this function         */
/*           conicS2, the second input conic section, known to be    */
/*           degenerate, its type  might be determined in this func  */
/*                                                                   */
/*   OUTPUTS: nIntersect: number of intersections                    */
/*                        = -1 if infinitely many intersections      */
/*                        at least one pt inside reference triangle  */
/*                        including the case where only one branch of*/
/*                        each conic is the same (this happens only  */
/*                        when they both are two lines)              */
/*            xIntersect: the intersection coorindate                */
/*            multiplicity: intersection multiplicity                */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/

  int nIntersectBranch;//number of intersection between S1 and one line of S2
   DT XIntersectBranch[4];//intersection coordinates between S1 and one line of S2
  int jRoot, kRoot;//index for intersection
  int multiplicityBranch[2];
 
  /*Initialize*/
  nIntersect = 0;

  /*Check whether S2 is indeed degenerate*/
  if(conicS2.degenerate != 1){
    ELVIS_PRINTF("conicS2 has to be degenerate!\n");
    return PXError(PX_BAD_INPUT);
  }
 
  /*Determine the coefficients of S2*/
  if(conicS2.DegComputed == PXE_False){
    PXErrorReturnSilent(PXConic2Lines(conicS2));  
    //if double precision is used, then just return the error, and exact precision will then be used
  }
  //ELVIS_PRINTF("degenerate type = %s\n", PXE_ConicSectionTypeName[conicS2.type]);

  /*Call ConicLine intersection for each degenerate line in S2*/

  if(conicS2.type == PXE_Conic_ParallelLines ||
     conicS2.type == PXE_Conic_OneLine ||
     conicS2.type == PXE_Conic_CoincidentLines){
    PXErrorReturn(PXConicLineIntersectInReferenceTriangle<DT>(conicS1, conicS2.a, conicS2.b, conicS2.c, 
							      nIntersectBranch, XIntersectBranch,
							      boundaryType, boundaryIndex,
							      multiplicityBranch));
  }

  switch(conicS2.type){
    case PXE_Conic_CrossingLines:{    
      /*Intersection with first branch*/
      //cout<<conicS2.A<<" "<<conicS2.b<<" "<<conicS2.c<<endl;
      PXErrorReturn(PXConicLineIntersectInReferenceTriangle<DT>(conicS1, conicS2.A, conicS2.b, conicS2.c, 
								nIntersectBranch, XIntersectBranch,
								boundaryType, boundaryIndex,
								multiplicityBranch));
      if(nIntersectBranch < 0){//infinitely many intersections
	nIntersect = -1;
	return PX_NO_ERROR;
      }
      //ELVIS_PRINTF("%d\n", nIntersectBranch);

      for(kRoot = 0; kRoot < nIntersectBranch; kRoot++){
	XIntersect[2*kRoot] = XIntersectBranch[2*kRoot];
	XIntersect[2*kRoot+1] = XIntersectBranch[2*kRoot+1];
	multiplicity[kRoot] = multiplicityBranch[kRoot];
      }
      nIntersect += nIntersectBranch;

      /*Intersection with second branch*/
      //cout<<"1"<<" "<<conicS2.e<<" "<<conicS2.f<<endl;
      PXErrorReturn(PXConicLineIntersectInReferenceTriangle<DT>(conicS1, 1, conicS2.e, conicS2.f, 
								nIntersectBranch, XIntersectBranch,
								boundaryType+nIntersect, boundaryIndex+nIntersect,
								multiplicityBranch));
      if(nIntersectBranch < 0){//infinitely many intersections
	nIntersect = -1;
	return PX_NO_ERROR;
      }
      
      //ELVIS_PRINTF("%d\n", nIntersectBranch);

      int nIntersectPrevBranch = nIntersect;
      enum PXE_Boolean sameIntersect;
      for(kRoot = 0; kRoot < nIntersectBranch; kRoot++){
	//make sure the intersection with second branch is not the same as with first branch (i.e. intersection at center)
	sameIntersect = PXE_False;
	for(jRoot = 0; jRoot < nIntersectPrevBranch; jRoot ++){
	  if(XIntersectBranch[2*kRoot] == XIntersect[2*jRoot] 
	     && XIntersectBranch[2*kRoot+1] == XIntersect[2*jRoot+1]){
	    ELVIS_PRINTF("This case should not happen in a real intersection case\n");
	    //because it does not make sense to have crossing lines inside the patch...
	    multiplicity[jRoot] +=  multiplicityBranch[kRoot];
	    sameIntersect = PXE_True;
	  }
	}
      
	if(sameIntersect == PXE_False){
	  XIntersect[2*nIntersect] = XIntersectBranch[2*kRoot];
	  XIntersect[2*nIntersect+1] = XIntersectBranch[2*kRoot+1];
	  multiplicity[nIntersect] = multiplicityBranch[kRoot];
	  nIntersect++;
	}
      }
      break;
    }
    case PXE_Conic_ParallelLines:{
      // ELVIS_PRINTF("A=%.15e;B= %.15e;C= %.15e;D=%.15e;E=%.15e;F=%.15e\n", conicS1.A,conicS1.B,conicS1.C,conicS1.D,conicS1.E,conicS1.F);
      // ELVIS_PRINTF("A=%.15e;B= %.15e;C= %.15e;D=%.15e;E=%.15e;F=%.15e\n", conicS2.A,conicS2.B,conicS2.C,conicS2.D,conicS2.E,conicS2.F);
      //ELVIS_PRINTF("a=%.15e;b=%.15e;c=%.15e;f=%.15e\n", conicS2.a,conicS2.b,conicS2.c,conicS2.f);
      // PXErrorReturn(PXConicLineIntersectInReferenceTriangle<DT>(conicS1, conicS2.a, conicS2.b, conicS2.c, 
      // 								nIntersectBranch, XIntersectBranch,
      // 								boundaryType, boundaryIndex,
      // 								multiplicityBranch));
      

      if(nIntersectBranch < 0){//infinitely many intersections
      	nIntersect = -1;
      	return PX_NO_ERROR;
      }
    
      for(kRoot = 0; kRoot < nIntersectBranch; kRoot++){

      	XIntersect[2*kRoot] = XIntersectBranch[2*kRoot];
      	XIntersect[2*kRoot+1] = XIntersectBranch[2*kRoot+1];
      	multiplicity[kRoot] = multiplicityBranch[kRoot];
      }
      nIntersect += nIntersectBranch;
    
      PXErrorReturn(PXConicLineIntersectInReferenceTriangle<DT>(conicS1, conicS2.a, conicS2.b, conicS2.f, 
      								nIntersectBranch, XIntersectBranch,
      								boundaryType+nIntersect, boundaryIndex+nIntersect,
      								multiplicityBranch));
      if(nIntersectBranch < 0){//infinitely many intersections
      	nIntersect = -1;
      	return PX_NO_ERROR;
      }

      for(kRoot = 0; kRoot < nIntersectBranch; kRoot++){
      	XIntersect[2*(nIntersect+kRoot)] = XIntersectBranch[2*kRoot];
      	XIntersect[2*(nIntersect+kRoot)+1] = XIntersectBranch[2*kRoot+1];
      	multiplicity[nIntersect+kRoot] = multiplicityBranch[kRoot];
      }
      nIntersect += nIntersectBranch;
      break;
    }
    case PXE_Conic_OneLine:
    case PXE_Conic_CoincidentLines:{
      // PXErrorReturn(PXConicLineIntersectInReferenceTriangle<DT>(conicS1, conicS2.a, conicS2.b, conicS2.c, 
      // 								nIntersectBranch, XIntersectBranch,
      // 								boundaryType, boundaryIndex,
      // 								multiplicityBranch));
      if(nIntersectBranch < 0){//infinitely many intersections
    	nIntersect = -1;
    	return PX_NO_ERROR;
      }

      for(kRoot = 0; kRoot < nIntersectBranch; kRoot++){
    	XIntersect[2*kRoot] = XIntersectBranch[2*kRoot];
    	XIntersect[2*kRoot+1] = XIntersectBranch[2*kRoot+1];
    	multiplicity[kRoot] = multiplicityBranch[kRoot];
      }
    
      //double count the multiplicity for coincdient lines
      if(conicS2.type == PXE_Conic_CoincidentLines){
    	for(kRoot = 0; kRoot < nIntersectBranch; kRoot++)
    	  multiplicity[kRoot] *= 2;
      }
    
      nIntersect += nIntersectBranch;
      break;
    }
    case PXE_Conic_OnePoint:{
      //verify if this point is on S1
      DT& xp = conicS2.xp;
      DT& yp = conicS2.yp;
       DT temp;
      temp = conicS1.A*xp*xp+2*conicS1.B*xp*yp+conicS1.C*yp*yp+2*conicS1.D*xp+2*conicS1.E*yp+conicS1.F;
      if(temp == 0){
    	XIntersect[0] = xp;
    	XIntersect[1] = yp;
    	nIntersect ++;
    	multiplicity[0] = 2; //the only case where a degenerate conic is an interseciton point has multiplicity 2
    	                     //see Coffman website
	
	//determine the boundary type
	if(xp == 0){
	  if(yp == 0){
	    boundaryType[0] = PXE_RefElementVertex;
	    boundaryIndex[0] = 0;
	  }
	  else if(yp == 1){
	    boundaryType[0] = PXE_RefElementVertex;
	    boundaryIndex[0] = 2;
	  }
	  else{
	    boundaryType[0] =  PXE_RefElementEdge;
	    boundaryIndex[0] = 1;
	  }
	}
	else if(yp == 0){
	  if(xp == 1){
	    boundaryType[0] = PXE_RefElementVertex;
	    boundaryIndex[0] = 1;
	  }
	  else{
	    boundaryType[0] =  PXE_RefElementEdge;
	    boundaryIndex[0] = 2;
	  }
	}
	else if(xp+yp == 1){
	  boundaryType[0] =  PXE_RefElementEdge;
	  boundaryIndex[0] = 0;
	}
	else{
	  boundaryType[0] =  PXE_RefElementInterior;
	  boundaryIndex[0] = -1;
	}
      }
      break;
    }
    case PXE_Conic_ImaginaryLines:{
      nIntersect = 0;
      break;
    }
    case  PXE_Conic_ZeroConic:{
      //inifinitely many solutions
      //nIntersect = -1;
      
      ELVIS_PRINTF("Need implmenetation to determine whether the other conic has parts in the ref triangle\n");
      PXErrorReturn(PX_BAD_INPUT);

      break;
    }
    default:
      ELVIS_PRINTF("conicS2 type has to be lines! BUT type=%d\n",conicS2.type);
      //ELVIS_PRINTF("type = %s\n", PXE_ConicSectionTypeName[conicS2.type]);
      return PXError(PX_BAD_INPUT);
  }

  return PX_NO_ERROR;
}



/******************************************************************/
//   FUNCTION Definition: PXConicConicIntersectSpecial
template <class DT> ELVIS_DEVICE int 
PXConicConicIntersectSpecial(ConicSection<DT>& conicS1, ConicSection<DT>& conicS2, 
			     int& nIntersect, DT* XIntersect,
			     enum PXE_RefElemIntersectType boundaryType[4], int boundaryIndex[4], 
			     int multiplicity[4], enum PXE_Boolean &SpecialCase)
/*-------------------------------------------------------------------*/
/*   PURPOSE: This function deals with easy special cases for        */
/*            conic-conic intersection                               */
/*                                                                   */
/*   INPUTS: conicS1, the first input conic section                  */
/*           conicS2, the second input conic section                 */
/*                                                                   */
/*   OUTPUTS: nIntersect: number of intersection                     */
/*                        = -1 if infinitely many intersections      */
/*                        at least one of which is in ref triangle   */
/*            Note: if two conics have infinitely many intersections,*/
/*                  then the intersected part must represent a       */
/*                  straight line in physical space, and the         */
/*                  intersected part must be a continuous segment    */
/*            xIntersect: the intersection coorindate                */
/*            multiplicity: the intersection multiplicity            */
/*            SpecialCase: whether the case is special               */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/
{
  /*If one of the conic is zero*/
  if(conicS1.type == PXE_Conic_ZeroConic 
     || conicS2.type == PXE_Conic_ZeroConic){
    ELVIS_PRINTF("This case corresponds to a tet edge entirely on the patch\n");
    ELVIS_PRINTF("Need to implement the function that whether the other conic have parts inside the patch\n");
    PXErrorReturn(PX_BAD_INPUT);
  }

  /*If one of the conics is imaginary*/
  if(conicS1.type == PXE_Conic_ImaginaryEllipse || conicS1.type == PXE_Conic_ImaginaryLines
     || conicS2.type == PXE_Conic_ImaginaryEllipse || conicS2.type == PXE_Conic_ImaginaryLines){
    nIntersect = 0;
    SpecialCase = PXE_True;
    return PX_NO_ERROR;
  }

  /*If one of the conics is already degenerate*/
  if(conicS1.degenerate == 1 || conicS2.degenerate == 1){
    //default: assume conicS1.degenerate == 1 (mu=0).  
    //If not true, then flip tempConic1 & 2

    //case where conicS1.degenerate == 1
    ConicSection<DT>& tempConic1 = conicS2;
    ConicSection<DT>& tempConic2 = conicS1;
    if(conicS2.degenerate == 1){ //corresponds to mu = 1
      tempConic1 = conicS1;
      tempConic2 = conicS2;
    }

    PXErrorReturnSilent(PXConicDegConicIntersectInReferenceTriangle(tempConic1, tempConic2, nIntersect, XIntersect,
							      boundaryType, boundaryIndex, multiplicity));
    SpecialCase = PXE_True;
    return PX_NO_ERROR;
  }  
 
  // if(conicS1.degenerate == 1){//this corresponds to mu = 0
  //   PXErrorReturnSilent(PXConicDegConicIntersectInReferenceTriangle(conicS2, conicS1, nIntersect, XIntersect,
  // 							      boundaryType, boundaryIndex, multiplicity));
  //   SpecialCase = PXE_True;
  //   return PX_NO_ERROR;
  // }  
 
  // if(conicS2.degenerate == 1){//this corresponds to mu = 1
  //   PXErrorReturnSilent(PXConicDegConicIntersectInReferenceTriangle(conicS1, conicS2, nIntersect, XIntersect,
  // 							      boundaryType, boundaryIndex, multiplicity));
  //   SpecialCase = PXE_True;
  //   return PX_NO_ERROR;
  // }
  
  
  SpecialCase = PXE_False;
  return PX_NO_ERROR;
}



/******************************************************************/
//   FUNCTION Definition: PXConicConicIntersectInReferenceTriangle
template <class DT> ELVIS_DEVICE int 
PXConicConicIntersectInReferenceTriangle(ConicSection<DT>& conicS1, ConicSection<DT>& conicS2, 
					 int& nIntersect, DT* XIntersect,
					 enum PXE_RefElemIntersectType boundaryType[4], int boundaryIndex[4], 
					 int multiplicity[4], enum PXE_Boolean &solved)
{
/*-------------------------------------------------------------------*/
/*   PURPOSE: Determine whether a conic intersects another conic that*/
/*            is degenerate within the reference triangle            */
/*                                                                   */
/*   INPUTS: conicS1, the first input conic section, its type or     */
/*           degeneracy might be determined in this function         */
/*           conicS2, the second input conic section its type or     */
/*           degeneracy might be determined in this function         */
/*           solved: whether the solution has to be found no matter  */
/*                   how slow the code might be due to precision     */
/*                   = true if the case                              */
/*                                                                   */
/*   OUTPUTS: nIntersect: number of intersection                     */
/*                        = -1 if infinitely many intersections      */
/*            Note: if two conics have infinitely many intersections,*/
/*                  then the intersected part must represent a       */
/*                  straight line in physical space, and the         */
/*                  intersected part must be a continuous segment    */
/*            xIntersect: the intersection coorindate                */
/*            multiplicity: the intersection multiplicity            */
/*            solved: whether the solution has been found            */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/
  const DT &A1 = conicS1.A; //make a reference for coding easiness 
  const DT &B1 = conicS1.B; 
  const DT &C1 = conicS1.C;
  const DT &D1 = conicS1.D;
  const DT &E1 = conicS1.E;
  const DT &F1 = conicS1.F;  
  const DT &A2 = conicS2.A; //make a reference for coding easiness 
  const DT &B2 = conicS2.B; 
  const DT &C2 = conicS2.C;
  const DT &D2 = conicS2.D;
  const DT &E2 = conicS2.E;
  const DT &F2 = conicS2.F; 

   DT B1sq, B2sq, D1sq, D2sq, E1sq, E2sq; //for computation easiness
   DT B1D1, B2D2;

   ConicSection<DT> conicS3; //S3 = S1+mu*S2 such that S3 degenerates to lines
   DT mu; //as above
   DT coeff[4]; //coefficients for cubic equation to find mu 
  
  enum PXE_CubicEquation_Type cubicType;
  bool rootExist;
  enum PXE_Boolean Inverse;
  int whichConic; //which conic to intersect with S3, either 1 or 2
  
  enum PXE_Boolean SpecialCase;
  
  DT const* BiQuadCoeff[2][6];

  /*Classify the two conics*/
  //PXErrorReturn(PXClassifyConic(conicS1));
  conicS1.type = PXE_Conic_Undetermined;
  PXErrorReturn(PXClassifyConic(conicS1));


  //PXErrorReturn(PXClassifyConic(conicS2));
  conicS2.type = PXE_Conic_Undetermined;
  PXErrorReturn(PXClassifyConic(conicS2));
  //PXPrintConic(conicS1);
  //PXPrintConic(conicS2);

  /*Special cases*/
  PXErrorReturnSilent(PXConicConicIntersectSpecial<DT>(conicS1, conicS2, nIntersect, XIntersect,
						 boundaryType, boundaryIndex, multiplicity, SpecialCase));
  if(SpecialCase == PXE_True){
    solved = PXE_True;
    return PX_NO_ERROR;
  }

  
  /*Using polynomial resultant to determine if there is possible intersection
    in the unit square [0,1]x[0,1]*/
  BiQuadCoeff[0][0] = &A1; BiQuadCoeff[0][1] = &B1; BiQuadCoeff[0][2] = &C1;
  BiQuadCoeff[0][3] = &D1; BiQuadCoeff[0][4] = &E1; BiQuadCoeff[0][5] = &F1;
  
  BiQuadCoeff[1][0] = &A2; BiQuadCoeff[1][1] = &B2; BiQuadCoeff[1][2] = &C2;
  BiQuadCoeff[1][3] = &D2; BiQuadCoeff[1][4] = &E2; BiQuadCoeff[1][5] = &F2;
  PXErrorReturnSilent(PXDetermineBiQuadSystemRoot<DT>(BiQuadCoeff, rootExist));
  if(!rootExist){
    nIntersect = 0;
    solved = PXE_True;
    return PX_NO_ERROR;
  } 
     
  /*If possible intersection exists, define the equations for mu such that det(S1+mu*S2) = 0*/
  B1sq = B1*B1; B2sq = B2*B2; 
  D1sq = D1*D1; D2sq = D2*D2;
  E1sq = E1*E1; E2sq = E2*E2;
  B1D1 = B1*D1; B2D2 = B2*D2;
  coeff[3] = (- F2*B2sq + 2*B2D2*E2 - C2*D2sq - A2*E2sq + A2*C2*F2);
  coeff[2] = (- F1*B2sq + 2*E1*B2D2 + 2*D1*B2*E2 - 2*B1*F2*B2 - C1*D2sq + 2*B1*D2*E2 - 2*C2*D1*D2 - A1*E2sq - 2*A2*E1*E2 + A1*C2*F2 + A2*C1*F2 + A2*C2*F1);
  coeff[1] = (- F2*B1sq + 2*E2*B1D1 + 2*D2*B1*E1 - 2*B2*F1*B1 - C2*D1sq + 2*B2*D1*E1 - 2*C1*D2*D1 - A2*E1sq - 2*A1*E2*E1 + A1*C1*F2 + A1*C2*F1 + A2*C1*F1);
  coeff[0] =  - F1*B1sq + 2*B1D1*E1 - C1*D1sq - A1*E1sq + A1*C1*F1;

  /*Solves for mu*/
  PXErrorReturn(PXSolveOneCubicRoot(coeff, cubicType, rootExist, mu, Inverse, solved));

  if(solved == PXE_False) return PX_NO_ERROR;
  if(cubicType == PXE_Cubic_ConstantZero){
    ELVIS_PRINTF("This case corresponds to both S1 and S2 being degenerate, should already been handled\n");
    PXErrorReturn(PX_CODE_FLOW_ERROR);
  }    

  /*S3*/
  if(rootExist){
    PXErrorReturn(PXInitializeConic<DT>(conicS3));
    if(Inverse == PXE_False){
      conicS3.A = A1+mu*A2;
      conicS3.B = B1+mu*B2;
      conicS3.C = C1+mu*C2;
      conicS3.D = D1+mu*D2;
      conicS3.E = E1+mu*E2;
      conicS3.F = F1+mu*F2;
      
      if(mu < 1e-5 && mu > -1e-5)
	whichConic = 2;
      else
	whichConic = 1;
    }
    else{
      conicS3.A = mu*A1+A2;
      conicS3.B = mu*B1+B2;
      conicS3.C = mu*C1+C2;
      conicS3.D = mu*D1+D2;
      conicS3.E = mu*E1+E2;
      conicS3.F = mu*F1+F2;
      
      if(mu < 1e-5 && mu > -1e-5)
	whichConic = 1;
      else
	whichConic = 2;
    }   
    //ELVIS_PRINTF("conicS3:\n");
    //PXPrintConic<DT>(conicS3);
    //std::cout<<"mu="<<mu<<"Inverse = "<<Inverse<<std::endl;
    
    //This line is very slow!! makes tetedge-patchface intersection 10 times slower
    //But sometimes the exact libarary does not give correct result if the cubic is ill-conditioned without this line
    //maybe the diamond operator is not correctly used???
     // PXErrorReturn(PXDetermineDegenerateConic<DT>(conicS3)); 
     // if(conicS3.degenerate != 1){
     //   ELVIS_PRINTF("Something not right...\n");
     // }    
    conicS3.degenerate = 1;
    conicS3.type = PXE_Conic_Undetermined;
      
    /*Intersection between S1 and S3*/
    //Note: at this point, mu should not be 0, so S3 != S1 or S2
    ConicSection<DT>& tempConic = conicS1;
    if(whichConic != 1)
      tempConic = conicS2;

    PXErrorReturnSilent(PXConicDegConicIntersectInReferenceTriangle<DT>(tempConic, conicS3, nIntersect, XIntersect, boundaryType, boundaryIndex, multiplicity));

    //ELVIS_PRINTF("nIntersect = %d, whichConic = %d\n", nIntersect, whichConic);
    // std::cout<<"mu="<<mu<<"Inverse = "<<Inverse<<std::endl;

  }
  else{
    //probably the two conics are the same
    /*This might happen when the tet edge lies in the patch entirely*/
    bool sameConic = false;

    //A1/A2 = B1/B2 = C1/C2 = D1/D2 = E1/E2 = F1/F2
    if(A1 != 0 || A2 != 0){
      if(A1*B2 == A2*B1 && A1*C2 == A2*C1 && A1*D2 == A2*D1 && A1*E2 == A2*E1 && A1*F2 == A2*F1)
	sameConic = true;
    }
    else if(B1 != 0 || B2 != 0){ //A1=A2=0
      if(B1*C2 == B2*C1 && B1*D2 == B2*D1 && B1*E2 == B2*E1 && B1*F2 == B2*F1)
	sameConic = true;
    }
    else if(C1 != 0 || C2 != 0){ //A1=A2=0, B1=B2=0
      if(C1*D2 == C2*D1 && C1*E2 == C2*E1 && C1*F2 == C2*F1)
	sameConic = true;
    }
    else if(D1 != 0 || D2 != 0){ //A1=A2=0, B1=B2=0, C1=C2=0
      if(D1*E2 == D2*E1 && D1*F2 == D2*F1)
	sameConic = true;
    }
    else if(E1 != 0 || E2 != 0){//A1=A2=0, B1=B2=0, C1=C2=0, D1=D2=0
      if(E1*F2 == E2*F1)
	sameConic = true;
    }
    
    if(sameConic == true){
      ELVIS_PRINTF("Two non-degenerate conics are the same...\n");
      ELVIS_PRINTF("This case should already be handled with mu that makes S1+muS2 = 0 for any x and y\n");
    }
    else{
      ELVIS_PRINTF("No plane that contains the given line is degenerate in the reference space of the patch!\n");
      ELVIS_PRINTF("This should never happen..., but if it does, more development is needed!\n");
      ELVIS_PRINTF("Possibly the two triangles that define the line are in the same plane...\n");
      return PXError(PX_BAD_INPUT);
    }
 
  }

  return PX_NO_ERROR;
}


/******************************************************************/
//   FUNCTION Prototype: PXIntersectTetFaceQuadraticEdge
template <class DT> ELVIS_DEVICE int
PXIntersectTetEdgeQuadraticFacePredicate(DT const * const lineNode[2], 
					 DT const * const quadNode[6],  
					 int& nIntersect, DT* sIntersect, DT* xIntersect,
					 int tetEdgeBoundary[2], enum PXE_RefElemIntersectType quadFaceBoundaryType[2], 
					 int quadFaceBoundaryIndex[2], 
					 int multiplicity[2], enum PXE_Boolean &solved)
{
/*-------------------------------------------------------------------*/
/*   PURPOSE: Determine if a line segment intersects with a          */
/*            quadratic face, and if it does, return the             */
/*            intersection coordinate, in the reference coordinate   */
/*            of the quadratic patch and physical coordiante         */
/*                                                                   */
/*   INPUTS:  lineNode = [x0,y0,z0,x1,y1,z1], defining the line seg  */
/*            quadNode = [x0,y0,z0,...,x5,y5,z5], quad segment       */
/*            conicS1, conicS2: the conics in the patch referece     */
/*                              formed from intersecting a planar    */
/*                              face and a quad patch                */
/*                              if they are null, the method of      */
/*                              directly computing the intersection  */
/*                              will be used                         */
/*            note: no check on the validity of the conics           */
/*            solved: whether the solution has to be found no matter */
/*                   how slow the code might be due to precision     */
/*                   = true if the case                              */
/*                                                                   */
/*   OUTPUTS: nIntersect: number of intersection points              */
/*            = -1 if infinitely many solutions                      */
/*            sIntersect: intersection coorindate in the reference   */
/*                        space of quad segment, at most 2           */
/*                        intersection points                        */
/*            xIntersect: physical coordinate of intersection        */
/*            tetEdgeBoundary: whether the intersection is at quad   */
/*                         edge boundary:                            */
/*			   = 0 if intersection at lineNode[0]        */
/*                         = 1 if intersection at lineNode[1]        */
/*                         = -1 if intersection not at edge vertex   */
/*            quadFaceBoundaryType: whether the intersection is at   */
/*                                 quad face boundary                */
/*            tetFaceBoundaryIndex: the index on the quad face bound */
/*                                 either local face or local vertex */
/*                                 number                            */
/*            solved: whether the solution has been found            */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/
  ConicSection<DT> conicS1;
  ConicSection<DT> conicS2;
  int d;
  const int dim = DIM3D;
   DT refIntersect[4]; //intersection point in ref coordinate, either on the edge or on its extension
   DT globIntersect[3]; //intersection point in glob coordinate, either on the edge or on its extension
  enum PXE_RefElemIntersectType quadFaceBoundaryTypetemp[2];
  int quadFaceBoundaryIndextemp[2];
  int nIntersectTemp; //number of intersection points, inside and outside the patch
  int multiplicitytemp[2];
  int k;
 
  bool intersectionFound;
  
  int compareDim; //the dimension to compare
   DT compareDimDiff;
   DT temp;
  
  /*if conicS's are not provided, compute them based on the quad and the given segment, without
    the knowledge of the adjacent background faces, see Method 2 of report on Apr. 6*/
     DT Fcoeff[3][8];
     DT lineVec[3]; //line vector
    const DT *x0, *x1, *x2, *x3, *x4, *x5;//quad patch nodes, for coding easiness
     
    PXErrorReturn(PXInitializeConic(conicS1));
    PXErrorReturn(PXInitializeConic(conicS2));
 
    /*Get the quad node pointers*/
    x0 = quadNode[0]; x1 = quadNode[1]; x2 = quadNode[2];
    x3 = quadNode[3]; x4 = quadNode[4]; x5 = quadNode[5];
    
    /*Line vector*/
    for(d = 0; d < dim; d++)
      lineVec[d] = lineNode[1][d]-lineNode[0][d]; 

    /*Compute all the three equations for intersection*/
    for(d = 0; d < dim; d++){
      Fcoeff[d][0] = 2*(x0[d]+x1[d]-2*x5[d]);
      Fcoeff[d][1] = 4*(x0[d]+x3[d]-x4[d]-x5[d]);
      Fcoeff[d][2] = 2*(x0[d]+x2[d]-2*x4[d]);
      Fcoeff[d][3] = (4*x5[d]-x1[d]-3*x0[d]);
      Fcoeff[d][4] = (4*x4[d]-x2[d]-3*x0[d]);
      Fcoeff[d][5] = (x0[d]);
      
      Fcoeff[d][6] = lineVec[d];
      Fcoeff[d][7] = lineNode[0][d];
    }
    
    PXErrorReturn(PXFormConicSystemForTetEdgeQuadFaceIntersect<DT>(Fcoeff, &conicS1, &conicS2));   
   
  
  /*Find the intersection of these conics*/
    PXErrorReturn(PXConicConicIntersectInReferenceTriangle<DT>(conicS1, conicS2, nIntersectTemp, refIntersect,
						      quadFaceBoundaryTypetemp, quadFaceBoundaryIndextemp, multiplicitytemp, 
							       solved));
  //PXPrintConic<DT>(*conicS1);
  // PXPrintConic<DT>(*conicS2);
  //ELVIS_PRINTF("nIntersect = %d\n", nIntersectTemp);

  if(solved == PXE_False){
    return PX_NO_ERROR;
  }

  /*Determine if the intersection point is on the edge*/
  /*If nIntersectTemp < 0, that means the entire tet edge is on the patch, 
    we don't need to store anything for that case because the end points of the tet edge
    will be found from other tet edges*/
  nIntersect = 0;
  for(k = 0;k < nIntersectTemp; k++){
    intersectionFound = false;

    /*Map the intersection point to physical space*/
    PXErrorReturn(PXQuadPatchRef2Glob<DT>(quadNode, refIntersect+2*k, globIntersect));
    
    
    /*Determine the dimension with largest distance*/
    compareDim = -1; 
    compareDimDiff = 0;
    for(d = 0; d < dim; d++){
      temp = lineNode[0][d]-lineNode[1][d];
      if(temp < 0) temp = -temp;
      
      if(compareDimDiff < temp){
	compareDimDiff = temp;
	compareDim = d;
      }
    }
    
    if(compareDim == -1){
      ELVIS_PRINTF("The line is zero vector?\n");
      //cout<<lineNode[0][0]<<" "<<lineNode[0][1]<<" "<<lineNode[0][2]<<endl;
      //cout<<lineNode[1][0]<<" "<<lineNode[1][1]<<" "<<lineNode[1][2]<<endl;
      PXErrorReturn(PX_CODE_FLOW_ERROR);
    }

    
    /*Determine if that point in physical space lies on the edge*/
    // for(d = 0; d < dim; d++){
    //   //if the line end points has same coordinate in the dimension d
    //   if(PXDetermineZero(lineNode[0][d]-lineNode[1][d]))
    // 	continue;

    d = compareDim;

      if(lineNode[0][d] < lineNode[1][d]){
	if(globIntersect[d] > lineNode[0][d] && globIntersect[d] < lineNode[1][d]){
	  intersectionFound = true;
	  tetEdgeBoundary[nIntersect] = -1;
	}
	else if(globIntersect[d] == lineNode[0][d]){
	  intersectionFound = true;
	  tetEdgeBoundary[nIntersect] = 0;
	}
	else if(globIntersect[d] == lineNode[1][d]){
	  intersectionFound = true;
	  tetEdgeBoundary[nIntersect] = 1;
	}
      }
      else{//i.e. lineNode[0][d] > lineNode[1][d]
	if(globIntersect[d] < lineNode[0][d] && globIntersect[d] > lineNode[1][d]){
	   intersectionFound = true;
	   tetEdgeBoundary[nIntersect] = -1;
	}
	else if(globIntersect[d] == lineNode[0][d]){
	  intersectionFound = true;
	  tetEdgeBoundary[nIntersect] = 0;
	}
	else if(globIntersect[d] == lineNode[1][d]){
	  intersectionFound = true;
	  tetEdgeBoundary[nIntersect] = 1;
	}
      }
      
      //if intersection point is on the edge, store it
      if(intersectionFound){
	sIntersect[2*nIntersect] = refIntersect[2*k]; 
	sIntersect[2*nIntersect+1] = refIntersect[2*k+1];
	xIntersect[3*nIntersect] = globIntersect[0]; 
	xIntersect[3*nIntersect+1] = globIntersect[1]; 
	xIntersect[3*nIntersect+2] = globIntersect[2]; 
	quadFaceBoundaryType[nIntersect] = quadFaceBoundaryTypetemp[k];
	quadFaceBoundaryIndex[nIntersect] = quadFaceBoundaryIndextemp[k];
	multiplicity[nIntersect] = multiplicitytemp[k];
	nIntersect++;
      }
      //break; //checking one coordinate is enough to detemine if the point is on the edge or not
      //}
  }
    
  return PX_NO_ERROR;
}



/******************************************************************/
//   FUNCTION Definition: PXCountSegmentPatchesIntersect
ELVIS_DEVICE int
PXCountSegmentPatchesIntersect(PX_REAL *patchCoordinates, int const* patchFace, int const& npatchFace,
			       PX_REAL const* const lineNode[2], int &nIntersect)
/*-------------------------------------------------------------------*/
/*   PURPOSE: Given a line segment and a set of patches, count the   */
/*            the total number of intersections                      */
/*                                                                   */
/*   INPUTS: meshsurf: the surface patch mesh                        */
/*           patchFace: patch face list                              */
/*           lineNode: the segment end points                        */
/*                                                                   */
/*   OUTPUTS: nIntersect: total number of intersections              */
/*                                                                   */
/*   RETURNS: error code                                             */
/*-------------------------------------------------------------------*/

{
  int d;
  const int dim = DIM3D;
  int kface;
  int knode;
  PX_REAL const *patchCoordPtr;
  PX_REAL tempQuadNode[PATCH_NBF][DIM3D]; //store quadNode in PX_REAL type, either PX_REAL or EX_REAL
  PX_REAL *ptempQuadNode[PATCH_NBF];
  PX_REAL tempLineNode[2][DIM3D];
  PX_REAL *ptempLineNode[2];
  /* PX_MeshElement *allElement = meshsurf->Element; */
  /* PX_MeshElement *Element; */

  //intersection variables, not really used here
  int nIntersectPerPatch;
  PX_REAL sIntersect[4];
  PX_REAL xIntersect[6];
  int tetEdgeBoundary[2];
  enum PXE_RefElemIntersectType quadFaceBoundaryType[2];
  int quadFaceBoundaryIndex[2]; 
  int multiplicity[2];
  enum PXE_Boolean solved = PXE_True;

  
  /*Initialize*/
  nIntersect = 0;
  for(knode = 0; knode < PATCH_NBF; knode ++)
    ptempQuadNode[knode] = tempQuadNode[knode];

  for(knode = 0; knode < 2; knode ++){
    ptempLineNode[knode] = tempLineNode[knode];
    
    for(d = 0; d < dim; d ++)
      tempLineNode[knode][d] = lineNode[knode][d];
  }

  /*Loop through each patch, and find the intersection with each patch*/
  for(kface = 0; kface < npatchFace; kface ++){
    
    /*Get the nodes defining the patch*/
    //Element = allElement + patchFace[kface]; //assume Q2 patch   
    patchCoordPtr = patchCoordinates + patchFace[kface]*PATCH_NBF*DIM3D;
    for(knode = 0; knode < PATCH_NBF; knode ++){
      //quadNode[knode] = Element->coordinate + knode * dim;
      for(d = 0; d < dim; d++)
	tempQuadNode[knode][d] = patchCoordPtr[knode*dim + d];//convert to PX_REAL, either EX_REAL or PX_REAL
    }   

    /*Determine the number of intersections with this patch*/
    PXErrorReturnSilent(PXIntersectTetEdgeQuadraticFacePredicate<PX_REAL>(ptempLineNode, ptempQuadNode, nIntersectPerPatch, sIntersect, xIntersect, tetEdgeBoundary, quadFaceBoundaryType, quadFaceBoundaryIndex, multiplicity, solved));
    //actually should look at quadfaceboundary
    //because the intersection at quad face boundary might have be counted twice here....
    
    // if(nIntersectPerPatch > 0)
    //   ELVIS_PRINTF("%d %.15e %.15e %.15e\n", nIntersectPerPatch, xIntersect[0], xIntersect[1], xIntersect[2]);
    // if(ierr != PX_NO_ERROR){
    //   PXPrintLineAndPatch<PX_REAL>(lineNode, quadNode);
    //   PXErrorReturn(ierr);
    // }
	
    /*Update*/
    nIntersect += nIntersectPerPatch;
  }


  return PX_NO_ERROR;
}


#endif //_PX_CUTCELL_ELIVS_C
