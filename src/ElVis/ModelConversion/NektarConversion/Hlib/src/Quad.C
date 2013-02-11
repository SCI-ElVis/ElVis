/**************************************************************************/
//                                                                        //
//   Author:    T.Warburton                                               //
//   Design:    T.Warburton && S.Sherwin                                  //
//   Date  :    12/4/96                                                   //
//                                                                        //
//   Copyright notice:  This code shall not be replicated or used without //
//                      the permission f the author.                     //
//                                                                        //
/**************************************************************************/

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <polylib.h>
#include "veclib.h"
#include "hotel.h"
#include "nekstruct.h"

static double Quad_Penalty_Fac = 1.;

Quad::Quad() : Element()
{
	Nverts = 4;
	Nedges = 4;
	Nfaces = 1;
}

Quad::Quad(Element *E)
{

	if(fabs(dparam("DPENFAC")) < FPTOL)
		Quad_Penalty_Fac = dparam("DPENFAC");

	if(!Quad_wk)
		Quad_work();

	id      = E->id;
	type    = E->type;
	state   = 'p';
	Nverts = 4;
	Nedges = 4;
	Nfaces = 1;

	vert    = (Vert *)calloc(Nverts,sizeof(Vert));
	edge    = (Edge *)calloc(Nedges,sizeof(Edge));
	face    = (Face *)calloc(Nfaces,sizeof(Face));
	lmax    = E->lmax;
	interior_l       = E->interior_l;
	Nmodes  = E->Nmodes;
	Nbmodes = E->Nbmodes;
	qa      = E->qa;
	qb      = E->qb;
	qc      = 0;

	qtot    = qa*qb;

	memcpy(vert,E->vert,Nverts*sizeof(Vert));
	memcpy(edge,E->edge,Nedges*sizeof(Edge));
	memcpy(face,E->face,Nfaces*sizeof(Face));

	/* set memory */
	vert[0].hj = (double*)  0;
	face[0].hj = (double**) 0;
	h          = (double**) 0;

	curve  = E->curve;
	curvX  = E->curvX;
	geom   = E->geom;
	dgL    = E->dgL;
	group  = E->group;
}


Quad::Quad(int i_d, char ty, int L, int Qa, int Qb, int Qc, Coord *X)
{
	int i;

	if(!Quad_wk)
		Quad_work();

	Qc = Qc; // compiler fix
	id = i_d;
	type = ty;
	state = 'p';
	Nverts = 4;
	Nedges = 4;
	Nfaces = 1;

	vert = (Vert *)calloc(Nverts,sizeof(Vert));
	edge = (Edge *)calloc(Nedges,sizeof(Edge));
	face = (Face *)calloc(Nfaces,sizeof(Face));
	lmax = L;
	interior_l    = 0;
	Nmodes  = Nverts + Nedges*(L-2) + Nfaces*(L-2)*(L-2);
	Nbmodes = Nmodes - (L-2)*(L-2);
	qa      = Qa;
	qb      = Qb;
	qc      = 0;

	qtot    = qa*qb;

	/* set vertex solve mask to 1 by default */
	for(i = 0; i < Nverts; ++i)
	{
		vert[i].id    = i;
		vert[i].eid   = id;
		vert[i].solve = 1;
		vert[i].x     = X->x[i];
		vert[i].y     = X->y[i];
	}

	/* construct edge system */
	for(i = 0; i < Nedges; ++i)
	{
		edge[i].id  = i;
		edge[i].eid = id;
		edge[i].l   = L-2;
	}

	/* construct face system */
	for(i = 0; i < Nfaces; ++i)
	{
		face[i].id  = i;
		face[i].eid = id;
		face[i].l   = max(0,L-2);
	}

	/* set memory */
	vert[0].hj = (double*)  0;
	face[0].hj = (double**) 0;
	h          = (double**) 0;

	curve  = (Curve*) calloc(1,sizeof(Curve));
	curve->face = -1;
	curve->type = T_Straight;
	curvX  = (Cmodes*)0;
}

Quad::Quad(int , char , int *, int *, Coord *)
{}

/********* Helmholtz matrix generation for parallelograms *********/


	// global variables
static double ***Prllgram_iplap = (double***)0;
static int Prllgram_ipmodes = 0;

	// function prototypes
static void Prllgram_setup_iprodlap();
static double Prllgram_iprodhelm(Element *Q, int i, int j, double lambda);

// Helmholtz matrix generation
void  Prllgram_HelmMat (Element *E, LocMat *helm, double lambda)
{
	if(!Prllgram_iplap)
		Prllgram_setup_iprodlap();

	double **a = helm->a,
	       **b = helm->b,
	       **c = helm->c;
	int    nbl    = E->Nbmodes;
	int    Nmodes = E->Nmodes;
	int    i,j;
	
	/* 'A' matrix */
	for(i = 0; i < nbl; ++i)
		for(j = i; j < nbl; ++j)
			a[i][j] = a[j][i] = Prllgram_iprodhelm(E, i, j, lambda);

	/* 'B' matrix */
	for(i = 0; i < nbl; ++i)
		for(j = nbl; j < Nmodes; ++j)
			b[i][j-nbl] = Prllgram_iprodhelm(E, i, j, lambda);

	/* 'C' matrix */
	for(i = nbl; i < Nmodes; ++i)
		for(j = nbl; j < Nmodes; ++j)
			c[i-nbl][j-nbl] =c[j-nbl][i-nbl] = Prllgram_iprodhelm(E,i,j,lambda);

}


// Calculus of terms that do not depend on the geometry
static void Prllgram_setup_iprodlap()
{

	int qa = QGmax;
	int qb = QGmax;
	int qc = 0;
	int L  = LGmax;

	// Set up dummy element with maximum quadrature/edge order

	Coord X;
	X.x = dvector(0,NQuad_verts-1);
	X.y = dvector(0,NQuad_verts-1);

	X.x[0] = -1.0;
	X.x[1] = 1.0;
	X.x[2] = 1.0;
	X.x[3] = -1.0;
	X.y[0] = -1.0;
	X.y[1] = -1.0;
	X.y[2] = 1.0;
	X.y[3] = 1.0;

	Quad *Q = (Quad*) new Quad(0,'Q', L, qa, qb, qc, &X);

	free(X.x);
	free(X.y);

	int i,j,k,n;
	int facs = Quad_DIM*Quad_DIM;
	Prllgram_ipmodes = Q->Nmodes;
	Prllgram_iplap = (double***) calloc(facs,sizeof(double**));

	for(i = 0; i < facs; ++i)
	{
		Prllgram_iplap[i] = dmatrix(0, Q->Nmodes-1, 0, Q->Nmodes-1);
		dzero(Q->Nmodes*Q->Nmodes, Prllgram_iplap[i][0], 1);
	}

	// Set up memory for gradient basis

	Basis   *B,*DB;
	Mode    *w;			// Integration weights
	double  *z;			// Integration zeros
	Mode **gb;			// Gradient of basis 
	Mode **gb1;			// Gradient of basis times integration weights
	Mode *m,*m1,*md,*md1, *fac;

	B      = Q->getbasis();
	DB     = Q->derbasis();

	fac    = B->vert;
	w      = mvector(0,0);
	gb     = (Mode **) malloc(Q->Nmodes*sizeof(Mode *));
	gb[0]  = mvecset(0,Quad_DIM*Q->Nmodes,qa, qb, qc);
	gb1    = (Mode **) malloc(Q->Nmodes*sizeof(Mode *));
	gb1[0] = mvecset(0,Quad_DIM*Q->Nmodes,qa, qb, qc);

	for(i = 1; i < Q->Nmodes; ++i)
		gb[i]  = gb[i-1]+Quad_DIM;
	for(i = 1; i < Q->Nmodes; ++i)
		gb1[i] = gb1[i-1]+Quad_DIM;

	// Getting zeros and weights for integration
	
	getzw(qa,&z,&w[0].a,'a');
	getzw(qb,&z,&w[0].b,'a');

	/* fill gb with basis info for laplacian calculation */

	// vertex modes
	m  =  B->vert;
	md = DB->vert;
	for(i = 0,n=0; i < Q->Nverts; ++i,++n)
		Q->fill_gradbase(gb[n],m+i,md+i,fac);

	// edge modes
	for(i = 0; i < Q->Nedges; ++i)
	{
		m1  = B ->edge[i];
		md1 = DB->edge[i];
		for(j = 0; j < Q->edge[i].l; ++j,++n)
			Q->fill_gradbase(gb[n],m1+j,md1+j,fac);
	}

	// face modes
	for(i = 0; i < Q->Nfaces; ++i)
		for(j = 0; j < Q->face[0].l; ++j)
		{
			m1  = B ->face[i][j];
			md1 = DB->face[i][j];
			for(k = 0; k < Q->face[0].l; ++k,++n)
				Q->fill_gradbase(gb[n],m1+k,md1+k,fac);
		}

	/* multiply by weights */
	for(i = 0; i < Q->Nmodes; ++i)
	{
		Tri_mvmul2d(qa,qb,qc,gb[i]  ,w,gb1[i]);
		Tri_mvmul2d(qa,qb,qc,gb[i]+1,w,gb1[i]+1);
	}

	// Calculate Laplacian inner products

	double s1, s2, s3, s4;

	fac = B->vert+1;

	for(i = 0; i < Q->Nmodes; ++i)
		for(j = 0; j < Q->Nmodes; ++j)
		{
			// dv/dr*du/dr
			s1  = ddot(qa,gb[i][0].a,1,gb1[j][0].a,1);
			s1 *= ddot(qb,gb[i][0].b,1,gb1[j][0].b,1);

			// dv/dr*du/ds
			s2  = ddot(qa,gb[i][0].a,1,gb1[j][1].a,1);
			s2 *= ddot(qb,gb[i][0].b,1,gb1[j][1].b,1);

			// dv/ds*du/dr
			s3  = ddot(qa,gb[i][1].a,1,gb1[j][0].a,1);
			s3 *= ddot(qb,gb[i][1].b,1,gb1[j][0].b,1);

			// dv/ds*dv/ds
			s4  = ddot(qa,gb[i][1].a,1,gb1[j][1].a,1);
			s4 *= ddot(qb,gb[i][1].b,1,gb1[j][1].b,1);

			Prllgram_iplap[0][i][j] = s1;
			Prllgram_iplap[1][i][j] = s2+s3;
			Prllgram_iplap[2][i][j] = s4;
		}

	/* fill gb with basis info for mass matrix calculation */
	// vertex modes
	m  = B->vert;
	for(i = 0,n=0; i < Q->Nverts; ++i,++n)
	{
		dcopy(qa, m[i].a, 1, gb[n]->a, 1);
		dcopy(qb, m[i].b, 1, gb[n]->b, 1);
	}

	// edge modes
	for(i = 0; i < Q->Nedges; ++i)
	{
		m1 = B ->edge[i];
		for(j = 0; j < Q->edge[i].l; ++j,++n)
		{
			dcopy(qa, m1[j].a, 1, gb[n]->a, 1);
			dcopy(qb, m1[j].b, 1, gb[n]->b, 1);
		}
	}

	// face modes
	for(i = 0; i < Q->Nfaces; ++i)
		for(j = 0; j < Q->face[i].l; ++j)
		{
			m1  = B ->face[i][j];
			for(k = 0; k < Q->face[i].l; ++k,++n)
			{
				dcopy(qa, m1[k].a, 1, gb[n]->a, 1);
				dcopy(qb, m1[k].b, 1, gb[n]->b, 1);
			}
		}

	/* multiply by weights */
	for(i = 0; i < Q->Nmodes; ++i)
		Tri_mvmul2d(qa,qb,qc,gb[i]  ,w,gb1[i]);

	for(i = 0; i < Q->Nmodes; ++i)
		for(j = 0; j < Q->Nmodes; ++j)
		{

			s1  = ddot(qa,gb[i][0].a,1,gb1[j][0].a,1);
			s1 *= ddot(qb,gb[i][0].b,1,gb1[j][0].b,1);

			Prllgram_iplap[3][i][j] = s1;
		}

	free_mvec(gb[0]) ;
	free((char *) gb);
	free_mvec(gb1[0]);
	free((char *) gb1);
	free((char*)w);

	delete (Q);
}


// Multiplying by the geometrical factors
static double Prllgram_iprodhelm(Element *E, int i, int j, double lambda)
{
	double d;
	double jac = E->geom->jac.p[0];
	double rx  = E->geom->rx.p[0], sx = E->geom->sx.p[0];
	double ry  = E->geom->ry.p[0], sy = E->geom->sy.p[0];

	d  = (rx*rx+ry*ry)*Prllgram_iplap[0][i][j];
	d += (rx*sx+ry*sy)*Prllgram_iplap[1][i][j];
	d += (sx*sx+sy*sy)*Prllgram_iplap[2][i][j];
	d +=        lambda*Prllgram_iplap[3][i][j];
	d *= jac;

	return d;
}


// ============================================================================
void Quad::PSE_Mat(Element *E, LocMat *pse, double *DU)
{

	double *save = dvector(0, qtot+Nmodes-1);
	dcopy(qtot, h[0], 1, save, 1);
	dcopy(Nmodes, vert[0].hj, 1, save+qtot, 1);
	char orig_state = state;

	register int i,j,n;
	int nbl, L, N, Nm, qt, asize = pse->asize, csize = pse->csize;
	Basis   *B = E->getbasis();
	double *Eh;
	double **a = pse->a;
	double **b = pse->b;
	double **c = pse->c;
	double **d = pse->d;

	// This routine is placed within the for loop for each element
	// at around line 30 in the code. Therefore, this is within an element

	nbl = E->Nbmodes;
	N   = E->Nmodes - nbl;
	Nm  = E->Nmodes;
	qt  = E->qa*E->qb;

	Eh  = dvector(0, qt - 1);    // Temporary storage for E->h[0] ------
	dcopy(qt, E->h[0], 1, Eh, 1);

	// Fill A and D ----------------------------------------------------

	for(i = 0, n = 0; i < E->Nverts; ++i, ++n)
	{
		E->fillElmt(B->vert + i);

		// ROUTINE THAT DOES INNERPRODUCT AND EVALUATES TERM--------------
		dvmul(qt, DU, 1, E->h[0], 1, E->h[0], 1);
		E->Iprod(E);
		// ---------------------------------------------------------------

		dcopy(nbl, E->vert->hj, 1, *a + n, asize);
		dcopy(N,   E->vert->hj + nbl, 1, d[n], 1);
	}

	dcopy(qt, Eh, 1, E->h[0], 1);

	for(i = 0; i < E->Nedges; ++i)
	{
		for(j = 0; j < (L = E->edge[i].l); ++j, ++n)
		{
			E->fillElmt(B->edge[i] + j);

			// ROUTINE THAT DOES INNERPRODUCT AND EVALUATES TERM------------
			dvmul(qt, DU, 1, E->h[0], 1, E->h[0], 1);
			E->Iprod(E);
			// -------------------------------------------------------------

			dcopy(nbl, E->vert->hj, 1, *a + n, asize);
			dcopy(N,   E->vert->hj + nbl, 1, d[n], 1);
		}
	}

	dcopy(qt, Eh, 1, E->h[0], 1);

	// Fill B and C ----------------------------------------------------

	L = E->face->l;
	for(i = 0, n = 0; i < L; ++i)
	{
		for(j = 0; j < L; ++j, ++n)
		{
			E->fillElmt(B->face[0][i]+j);

			// ROUTINE THAT DOES INNERPRODUCT AND EVALUATES TERM-----------
			dvmul(qt, DU, 1, E->h[0], 1, E->h[0], 1);
			E->Iprod(E);
			// ------------------------------------------------------------

			dcopy(nbl, E->vert->hj    , 1, *b + n, csize);
			dcopy(N, E->vert->hj + nbl, 1, *c + n, csize);
		}
	}

	// -----------------------------------------------------------------

	free(Eh);

	state = orig_state;

	dcopy(qtot, save, 1, h[0], 1);
	dcopy(Nmodes, save+qtot, 1, vert[0].hj, 1);
	free(save);
}

// ============================================================================
void Quad::BET_Mat(Element *P, LocMatDiv *bet, double *beta, double *sigma)
{
	register int i,j,n;
	const    int nbl = Nbmodes, N = Nmodes - Nbmodes;
	int      L;
	Basis   *b = getbasis();
	double **dxb = bet->Dxb,   // MSB: dx corresponds to bar(beta)
	               **dxi = bet->Dxi,
	                       **dyb = bet->Dyb,   // MSB: dy corresponds to sigma
	                               **dyi = bet->Dyi;
	char orig_state = state;

	/* fill boundary systems */
	for(i = 0,n=0; i < Nverts; ++i,++n)
	{
		fillElmt(b->vert+i);

		dvmul(qtot,beta,1,*h,1,*P->h,1);
#ifndef PCONTBASE

		P->Ofwd(*P->h,P->vert->hj,P->dgL);
#else

		P->Iprod(P);
#endif

		dcopy(P->Nmodes,P->vert->hj,1,*dxb + n,nbl);

		dvmul(qtot,sigma,1,*h,1,*P->h,1);
#ifndef PCONTBASE

		P->Ofwd(*P->h,P->vert->hj,P->dgL);
#else

		P->Iprod(P);
#endif

		dcopy(P->Nmodes,P->vert->hj,1,*dyb + n,nbl);
	}

	for(i = 0; i < Nedges; ++i)
		for(j = 0; j < edge[i].l; ++j, ++n)
		{
			fillElmt(b->edge[i]+j);

			dvmul(qtot,beta,1,*h,1,*P->h,1);
#ifndef PCONTBASE

			P->Ofwd(*P->h,P->vert->hj,P->dgL);
#else

			P->Iprod(P);
#endif

			dcopy(P->Nmodes,P->vert->hj,1,*dxb + n,nbl);

			dvmul(qtot,sigma,1,*h,1,*P->h,1);
#ifndef PCONTBASE

			P->Ofwd(*P->h,P->vert->hj,P->dgL);
#else

			P->Iprod(P);
#endif

			dcopy(P->Nmodes,P->vert->hj,1,*dyb + n,nbl);
		}

	L = face->l;
	for(i = 0,n=0; i < L;++i)
		for(j = 0; j < L; ++j,++n)
		{
			fillElmt(b->face[0][i]+j);

			dvmul(qtot,beta,1,*h,1,*P->h,1);
#ifndef PCONTBASE

			P->Ofwd(*P->h,P->vert->hj,P->dgL);
#else

			P->Iprod(P);
#endif

			dcopy(P->Nmodes,P->vert->hj,1,*dxi + n,N);

			dvmul(qtot,sigma,1,*h,1,*P->h,1);
#ifndef PCONTBASE

			P->Ofwd(*P->h,P->vert->hj,P->dgL);
#else

			P->Iprod(P);
#endif

			dcopy(P->Nmodes,P->vert->hj,1,*dyi + n,N);
		}

	state = orig_state;

	/* negate all systems to that the whole operator can be treated
	   as positive when condensing */
	/*
	dneg(nbl*P->Nmodes,*dxb,1);
	dneg(nbl*P->Nmodes,*dyb,1);
	dneg(N  *P->Nmodes,*dxi,1);
	dneg(N  *P->Nmodes,*dyi,1);
	*/
}

// ============================================================================
