// Routines for Multi-RHS iterative solver
//
// Reference: Fischer, P. "Projection techniques for iterative solution of Ax=b
// with successive right-hand sides." Comp. Meth. Appl. Engrg. 163, p.193-204
// 1998.
//
// If ORIG is defined below, the first method of the paper is used, otherwise,
// the second method of the paper is used.

#include <math.h>
#include <veclib.h>
#include "hotel.h"

#if 0
#define MINENERGY
#endif
#define ORIG

static Multi_RHS *Setup_Multi_RHS (int Nrhs, int nsolve, char type);

Multi_RHS *Get_Multi_RHS(Bsystem *Ubsys, int Nrhs, int nsolve, char type)
{
	Multi_RHS *mrhs;

	for(mrhs = Ubsys->mrhs; mrhs; mrhs = mrhs->next)
	{
		if(mrhs->type == type)
			return mrhs;
	}

	// if not previously set up initialise a new stucture
	mrhs = Setup_Multi_RHS(Nrhs,nsolve,type);

	mrhs->next = Ubsys->mrhs;
	Ubsys->mrhs = mrhs;

	return mrhs;
}

static Multi_RHS *Setup_Multi_RHS(int Nrhs, int nsolve, char type)
{
	Multi_RHS *mrhs;

	mrhs = (Multi_RHS*) calloc(1, sizeof(Multi_RHS));
	mrhs->type = type;
	mrhs->step   = 0;

	mrhs->nsolve = nsolve;
	mrhs->Nrhs   = Nrhs;

	mrhs->alpha  = dvector(0, Nrhs-1);
	dzero(Nrhs, mrhs->alpha, 1);
#ifdef ORIG

	mrhs->bt     = dmatrix(0, Nrhs-1, 0, nsolve-1);
	dzero(Nrhs*nsolve, mrhs->bt[0], 1);
#else

	mrhs->xbar   = dvector(0, nsolve-1);
#endif

	mrhs->xt     = dmatrix(0, Nrhs-1, 0, nsolve-1);
	dzero(Nrhs*nsolve, mrhs->xt[0], 1);

	return mrhs;
}

#ifdef ORIG
void Mrhs_rhs(Element_List *, Element_List *, Bsystem *B,
              Multi_RHS *mrhs, double *rhs)
{
	register int i;
	
	// Evaluate projection coefficient alpha_i = <b,bt_i>
	DO_PARALLEL
	{
	    int j;
	    double *wk = dvector(0,mrhs->step);
	    for(i = 0; i < mrhs->step; ++i)
    	{
		    mrhs->alpha[i] = 0.0;
		    for(j = 0; j < mrhs->nsolve; ++j)
			    mrhs->alpha[i] += B->pll->mult[j]*mrhs->bt[i][j]*rhs[j];
	    }
	    gdsum(mrhs->alpha,mrhs->step,wk);
	    free(wk);
	}
	else
		for(i = 0; i < mrhs->step; ++i)
			mrhs->alpha[i] = ddot(mrhs->nsolve, mrhs->bt[i], 1, rhs, 1);

	// Evaluate bt = b - sum(alpha_i * bt_i)
	for(i = 0; i < mrhs->step; ++i)
		daxpy(mrhs->nsolve, -mrhs->alpha[i], mrhs->bt[i], 1, rhs, 1);
}
#else
void Mrhs_rhs(Element_List *U, Element_List *Uf, Bsystem *B,
              Multi_RHS *mrhs, double *rhs)
{
	double *bhat;
	register int i;
	static double *wkrecur = (double*)0;

	bhat = dvector(0, B->nglobal-1);
	dzero(B->nglobal, bhat, 1);
	dzero(B->nglobal,mrhs->xbar,1);
	
#ifndef SPARSEMAT
	if (B->rslv && !wkrecur)
		wkrecur = dvector(0,2*(B->rslv->max_asize)-1);
#endif

	// Evaluate alpha_i = xt_i . b and assign xbar = sum(alpha_i * xt_i)
	DO_PARALLEL
	{
	    int j;
	    double *wk = dvector(0,mrhs->step);
	    for(i = 0; i < mrhs->step; ++i)
    	{
	    	mrhs->alpha[i] = 0.0;
		    for(j = 0; j < mrhs->nsolve; ++j)
			    mrhs->alpha[i] += B->pll->mult[j]*mrhs->xt[i][j]*rhs[j];
	    }
	    gdsum(mrhs->alpha,mrhs->step,wk);

	    for(i = 0; i < mrhs->step; ++i)
	    daxpy(mrhs->nsolve,mrhs->alpha[i],mrhs->xt[i],1,mrhs->xbar,1);

	    free(wk);
	}
	else
		for(i = 0; i < mrhs->step; ++i)
			{
				mrhs->alpha[i] = ddot(mrhs->nsolve, mrhs->xt[i], 1, rhs, 1);
				daxpy(mrhs->nsolve,mrhs->alpha[i],mrhs->xt[i],1,mrhs->xbar,1);
			}

	// bt = b - A*xbar
#ifdef SPARSEMAT
	SparseMat Spa;
	
	if (B->rslv)
		Spa = B->rslv->A.Spa;
	else
		Spa = B->Gmat->Spa;	
			
	int nlines = Spa.nlines;	//  bhat = A * xbar
	int *rpt = Spa.rpt;
	int *ija = Spa.ija;
	double *sa = Spa.sa;
	double *bhati = bhat;
	double *xbari = mrhs->xbar;
	register int k;
	
	for (i = 0; i < nlines; i++)
	{
		for (k = rpt[i]; k < rpt[i+1] - 1; k++)
		{
			(*bhati) += mrhs->xbar[ija[k]] * (*sa);
			bhat[ija[k]] += (*xbari) * (*sa++);
		}
		(*bhati++) += (*xbari++) * (*sa++);
	}
#else
	if (B->rslv)
		Recur_A(B->rslv, mrhs->xbar, bhat, wkrecur);
	else
		A_fast(U,Uf,B,mrhs->xbar,bhat);
#endif
		
	dvsub(mrhs->nsolve,rhs,1,bhat,1,rhs,1);

	free(bhat);
}
#endif

#ifdef MINENERGY
static int ida_min(int n, double *d, int skip)
{
	int i;
	int ida = 0;
	for(i = 1; i < n; i += skip)
		ida = (fabs(d[i]) < fabs(d[ida])) ? i : ida;

	return ida;
}
#endif

#ifdef ORIG
void Update_Mrhs(Element_List *U, Element_List *Uf,
                 Bsystem *B, Multi_RHS *mrhs, double *sol)
{
	int         i, j, k, lt;
	int         nsolve = mrhs->nsolve;
	double     *bhat, *xhat, *wk, norm;
	static double *wkrecur = (double*)0;

	bhat = dvector(0, B->nglobal-1);
	dzero(B->nglobal, bhat, 1);
	xhat = dvector(0, B->nglobal-1);
	dzero(B->nglobal, xhat, 1);
	wk = dvector(0,mrhs->step);
	
#ifndef SPARSEMAT
	if (B->rslv && !wkrecur)
		wkrecur = dvector(0,2*(B->rslv->max_asize)-1);
#endif

	// xhat = xt
	dcopy (nsolve, sol, 1, xhat, 1);

	// Update x^n = sol + sum(alpha_k * xt_k)
	for(i = 0; i < mrhs->step; ++i)
		daxpy(nsolve, mrhs->alpha[i], mrhs->xt[i], 1, sol, 1);

#ifdef MINENERGY

	// find mode with minimum "energy"
	if(mrhs->step == mrhs->Nrhs)
		lt = ida_min(mrhs->Nrhs, mrhs->alpha, 1);
	else
		lt = mrhs->step;
#else

	if(mrhs->step == mrhs->Nrhs)	// Reset projection space
	{
		mrhs->step = 0;
		lt = 0;
	}
	else
		lt = mrhs->step;
#endif

#ifdef SPARSEMAT
	SparseMat Spa;
	
	if (B->rslv)
		Spa = B->rslv->A.Spa;
	else
		Spa = B->Gmat->Spa;	
	
	int nlines = Spa.nlines;	//  bhat = A * xhat
	int *rpt = Spa.rpt;
	int *ija = Spa.ija;
	double *sa = Spa.sa;
	double *bhati = bhat;
	double *xhati = xhat;
	
	for (i = 0; i < nlines; i++)
	{
		for (k = rpt[i]; k < rpt[i+1] - 1; k++)
		{
			(*bhati) += xhat[ija[k]] * (*sa);
			bhat[ija[k]] += (*xhati) * (*sa++);
		}
		(*bhati++) += (*xhati++) * (*sa++);
	}
		
#else
	
	if (B->rslv)
		Recur_A(B->rslv, xhat, bhat, wkrecur);
	else
		A_fast(U,Uf,B,xhat,bhat);
	
#endif

	// alpha_i = <bhat, bt_i>
	DO_PARALLEL
	{
	    parallel_gather(bhat,B);
	    for(i = 0; i < mrhs->step; ++i)
    	{
		    mrhs->alpha[i] = 0.0;
		    for(j = 0; j < nsolve; ++j)
			    mrhs->alpha[i]  += B->pll->mult[j]*mrhs->bt[i][j]*bhat[j];
	    }
	    gdsum(mrhs->alpha, mrhs->step, wk);
	}
	else
		for(i = 0; i < mrhs->step; ++i)
			mrhs->alpha[i] = ddot(nsolve, mrhs->bt[i], 1, bhat, 1);

	mrhs->alpha[lt] = 0.0;

	// bt_(l+1) = bhat - sum(alpha_i * bt_i)
	// xt_(l+1) = xt - sum(alpha_i * xt_i)
	for(i = 0; i < mrhs->step; ++i)
	{
		daxpy(nsolve, -mrhs->alpha[i], mrhs->bt[i], 1, bhat, 1);
		daxpy(nsolve, -mrhs->alpha[i], mrhs->xt[i], 1, xhat, 1);
	}

	// Normalise bt_(l+1) and xt_(l+1)
	DO_PARALLEL
	{
	    norm =  0.0;
	    for(j = 0; j < mrhs->nsolve; ++j)
	    norm += B->pll->mult[j]*bhat[j]*bhat[j];
	    gdsum(&norm, 1, wk);
	}
	else
	{
		norm = ddot(nsolve, bhat, 1, bhat, 1);
	}

	norm = (fabs(norm) > FPTOL) ? 1.0/sqrt(norm): 1.0;

	dsmul(nsolve, norm, bhat, 1, mrhs->bt[lt], 1);
	dsmul(nsolve, norm, xhat, 1, mrhs->xt[lt], 1);

	// Increment size of projection space
	mrhs->step = min(mrhs->step+1,mrhs->Nrhs);

	free(xhat);
	free(bhat);
	free(wk);
}
#else
void Update_Mrhs(Element_List *U, Element_List *Uf,
                 Bsystem *B, Multi_RHS *mrhs, double *sol)
{
	int         i, j, lt;
	int         nsolve = mrhs->nsolve;
	double      *bhat, *wk, norm;
	static double *wkrecur = (double*)0;

	bhat = dvector(0, B->nglobal-1);
	dzero(B->nglobal, bhat, 1);
	wk = dvector(0, mrhs->step);

#ifndef SPARSEMAT
	if (B->rslv && !wkrecur)
		wkrecur = dvector(0,2*(B->rslv->max_asize)-1);
#endif

	if(mrhs->step == mrhs->Nrhs)
	{
		// Update x^n = sol + xbar
		dvadd(nsolve,mrhs->xbar,1,sol,1,sol,1);

		mrhs->step = 0;
		dcopy(mrhs->nsolve,sol,1,mrhs->xt[0],1);
	}
	else
	{
		dcopy(nsolve,sol,1,mrhs->xt[mrhs->step],1);

		//  bhat = A * sol
#ifdef SPARSEMAT
		SparseMat Spa;
	
		if (B->rslv)
			Spa = B->rslv->A.Spa;
		else
			Spa = B->Gmat->Spa;	
		
		int nlines = Spa.nlines;
		int *rpt = Spa.rpt;
		int *ija = Spa.ija;
		double *sa = Spa.sa;
		double *bhati = bhat;
		double *soli = sol;
	
		for (i = 0; i < nlines; i++)
		{
			for (j = rpt[i]; j < rpt[i+1] - 1; j++)
			{
				(*bhati) += sol[ija[j]] * (*sa);
				bhat[ija[j]] += (*soli) * (*sa++);
			}
			(*bhati++) += (*soli++) * (*sa++);
		}
#else
		if (B->rslv)
			Recur_A(B->rslv, sol, bhat, wkrecur);
		else
			A_fast(U,Uf,B,sol,bhat);
#endif		

		// Update x^n = sol + xbar
		dvadd(nsolve,mrhs->xbar,1,sol,1,sol,1);

		// alpha_i = xt_i . bhat   (bhat = A * sol)
		DO_PARALLEL
		{
		    parallel_gather(bhat,B);
		    for(i = 0; i < mrhs->step; ++i)
	    	{
			    mrhs->alpha[i] = 0.0;
			    for(j = 0; j < nsolve; ++j)
				    mrhs->alpha[i]  += B->pll->mult[j]*mrhs->bt[i][j]*bhat[j];
		    }
		    gdsum(mrhs->alpha,mrhs->step, wk);
		}
		else
			for(i = 0; i < mrhs->step; ++i)
				mrhs->alpha[i] = ddot(nsolve, mrhs->xt[i], 1, bhat, 1);

		// xt_(l+1) = sol - sum(alpha_i * xt_i) 
		for(i = 0; i < mrhs->step; ++i)
			daxpy(nsolve, -mrhs->alpha[i], mrhs->xt[i], 1,
			      mrhs->xt[mrhs->step], 1);
	}

	// Normalise xt_(l+1)
	DO_PARALLEL
	{
	    norm =  0.0;
	    for(j = 0; j < mrhs->nsolve; ++j)
	    norm += B->pll->mult[j]*mrhs->xt[mrhs->step][j]*
								mrhs->xt[mrhs->step][j];
	    gdsum(&norm, 1, wk);
	}
	else
		norm = ddot(nsolve, mrhs->xt[mrhs->step], 1, mrhs->xt[mrhs->step], 1);

	norm = (fabs(norm) > FPTOL) ? 1.0/sqrt(norm): 1.0;
	dsmul(nsolve, norm, mrhs->xt[mrhs->step], 1, mrhs->xt[mrhs->step], 1);

	// Increment size of projection space
	mrhs->step = min(mrhs->step+1,mrhs->Nrhs);

	free(bhat);
	free(wk);
}
#endif
