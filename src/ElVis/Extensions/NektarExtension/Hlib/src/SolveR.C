/**************************************************************************/
//                                                                        //
//   Author:    S.Sherwin                                                 //
//   Design:    S.Sherwin                                                 //
//   Date  :    12/4/96                                                   //
//                                                                        //
//   Copyright notice:  This code shall not be replicated or used without //
//                      the permission of the author.                     //
//                                                                        //
/**************************************************************************/

#include <math.h>
#include <veclib.h>
#include <string.h>
#include "hotel.h"
#include "Tri.h"
#include "Quad.h"

static void Precon(Bsystem *B, double *r, double *z);

/* subtract off recursive interior patch coupling */

void Recur_setrhs(Rsolver *R, double *rhs)
{
	register int i,j,n;
	int   nrecur = R->nrecur;
	int   cstart,asize,csize;
	Recur *rdata = R->rdata;
	double *tmp = dvector(0,R->max_asize);

	for(i = 0; i < nrecur; ++i)
	{

		cstart = rdata[i].cstart;

		for(n = 0; n < rdata[i].npatch; ++n)
		{
			asize = rdata[i].patchlen_a[n];
			csize = rdata[i].patchlen_c[n];
			if(asize&&csize)
			{
				dgemv('T', csize, asize, 1., rdata[i].binvc[n], csize,
				      rhs + cstart, 1, 0., tmp,1);

				for(j = 0; j < asize; ++j)
					rhs[rdata[i].map[n][j]] -= tmp[j];
			}
			cstart += csize;
		}
	}

	free(tmp);
}

void Recur_backslv(Rsolver *R, double *rhs, char trip)
{
	register int i,j,n;
	int     nrecur = R->nrecur;
	int     cstart,asize,csize,bw,info;
	Recur   *rdata = R->rdata;
	double  *tmp   = dvector(0,R->max_asize);

	for(i = nrecur-1; i >= 0; --i)
	{
		cstart = rdata[i].cstart;

		for(n = 0; n < rdata[i].npatch; ++n)
		{
			asize = rdata[i].patchlen_a[n];
			csize = rdata[i].patchlen_c[n];
			bw    = rdata[i].bwidth_c[n];

			if(csize)
			{
				if(trip == 'p')
				{
					if(2*bw < csize)    /* banded matrix */
						dpbtrs('L',csize,bw-1,1,rdata[i].invc[n],bw,rhs+cstart,
						       csize, info);
					else                /* symmetric matrix */
						dpptrs('L',csize,1,rdata[i].invc[n],rhs+cstart,csize,
							   info);
				}
				else
				{
					if(2*bw < csize)
					{   				/* banded matrix */
						error_msg(error in H_SolveR.c);
					}
					else                /* symmetric matrix */
						dsptrs('L',csize,1,rdata[i].invc[n],rdata[i].pivotc[n],
						       rhs + cstart, csize, info);
				}

				if(asize)
				{
					for(j = 0; j < asize; ++j)
						tmp[j] = rhs[rdata[i].map[n][j]];

					dgemv('N', csize, asize,-1., rdata[i].binvc[n], csize,
					      tmp, 1, 1.0, rhs + cstart,1);
				}
			}
			cstart += csize;
		}
	}

	free(tmp);
}

#define MAX_ITERATIONS 2*nsolve

void Recur_Bsolve_CG(Bsystem *B, double *p, char type)
{
	Rsolver    *R = B->rslv;
	const  int nsolve = R->rdata[R->nrecur-1].cstart;
	int    iter = 0;
	double tolcg, alpha, beta, eps, rtz, rtz_old, epsfac;
	extern double tol;
	static double *w = (double*)0;
	static double *u = (double*)0;
	static double *r = (double*)0;
	static double *z = (double*)0;
	static double *wk = (double*)0;
	static int nsol = 0;
	int Nrhs = option("MRHS_NRHS");
	Multi_RHS *mrhs;

	if(nsolve > nsol)
	{
		if(nsol)
		{
			free(u);
			free(r);
			free(z);
			free(w);
#ifndef SPARSEMAT			
			free(wk);
#endif
		}	
		
		/* Temporary arrays */
		u  = dvector(0,nsolve-1);          /* Solution              */
		r  = dvector(0,nsolve-1);          /* residual              */
		z  = dvector(0,nsolve-1);          /* precondition solution */
		w  = dvector(0,nsolve-1);          /* A*Search direction    */
#ifndef SPARSEMAT
		wk = dvector(0,2*R->max_asize-1);  /* work space            */
#endif
		nsol = nsolve;
	}

	dzero (nsolve, u, 1);
	dcopy (nsolve, p, 1, r, 1);

#if 0
	// currently this is being down outside multi-level recursion in Solve.C
	if(B->singular) /* take off mean of vertex modes */
		dsadd(nvs, -dsum(nvs, r, 1)/(double)nvs, r, 1, r, 1);
#endif

    if(Nrhs)
   	{
	    mrhs = Get_Multi_RHS(B, Nrhs, nsolve, type);
	    Mrhs_rhs((Element_List *)NULL , (Element_List *)NULL, B, mrhs, r);
    }
	
	tolcg  = (type == 'p')? dparam("TOLCGP"):dparam("TOLCG");
	epsfac = (fabs(tol) > FPTOL) ? 1.0/tol : 1.0;
	eps    = sqrt(ddot(nsolve,r,1,r,1))*epsfac;

	if (option("verbose") > 1)
		ROOTONLY printf("\tInitial eps : %lg\n",eps);
	
	/* =========================================================== *
	 *            ---- Conjugate Gradient Iteration ----           *
	 * =========================================================== */
				
#ifdef SPARSEMAT
		
	int nlines = R->A.Spa.nlines;
	int *rpt = R->A.Spa.rpt;
	int *ija = R->A.Spa.ija;
	double *sa;
	double *wi, *pi;
	register int i, k;

#endif
		
	while (eps > tolcg && iter++ < MAX_ITERATIONS )
	{
		Precon(B,r,z);

		rtz  = ddot (nsolve, r, 1, z, 1);

		if (iter > 1)
		{                         /* Update search direction */
			beta = rtz / rtz_old;
			dsvtvp(nsolve, beta, p, 1, z, 1, p, 1);
		}
		else
			dcopy(nsolve, z, 1, p, 1);

#ifdef SPARSEMAT		
		
		dzero(nsolve, w, 1);				// w(i) = A*p(i)
		sa = R->A.Spa.sa;
		wi = w;
		pi = p;
		for (i = 0; i < nlines; i++)
		{
			for (k = rpt[i]; k < rpt[i+1] - 1; k++)
			{
				(*wi) += p[ija[k]] * (*sa);
				w[ija[k]] += (*pi) * (*sa++);
			}
			(*wi++) += (*pi++) * (*sa++);
		}

#else
				
		Recur_A(R,p,w,wk);

#endif
		
		alpha = rtz/ddot(nsolve, p, 1, w, 1);

		daxpy(nsolve, alpha, p, 1, u, 1);            /* Update solution...   */
		daxpy(nsolve,-alpha, w, 1, r, 1);            /* ...and residual      */

		rtz_old = rtz;
		eps = sqrt(ddot(nsolve, r, 1, r, 1))*epsfac; /* Compute new L2-error */
		
	}
			
	/* =========================================================== *
	 *                        End of Loop                          *
	 * =========================================================== */

	// Update multiple rhs data structure
	if(Nrhs)
		Update_Mrhs((Element_List *)NULL, (Element_List *)NULL, B, mrhs, u);
	
	/* Save solution and clean up */
	dcopy(nsolve,u,1,p,1);

	if (iter > MAX_ITERATIONS)
	{
		error_msg (Recur_Bsolve_CG failed to converge);
	}
	else if (option("verbose") > 1)
		ROOTONLY printf("\tField %c: %3d iterations, error = %#14.6g %lg %lg\n",
		       			type, iter, eps, epsfac, tolcg);
	
	return;
}

void Recur_A(Rsolver *R, double *p, double *w, double *wk)
{
	Recur    *rdata = R->rdata + R->nrecur-1;
	double   **a    = R->A.a;
	int      npatch = rdata->npatch;
	int      *alen  = rdata->patchlen_a;
	int      **map  = rdata->map;
	register int i, k;
	double   *wk1 = wk + R->max_asize;

	memset (w, '\0', rdata->cstart * sizeof(double));

	/* put p boundary modes into U and impose continuity */
	for(k = 0; k < npatch; ++k)
	{
		/* gather in terms for patch k */
		dgathr(alen[k],p,map[k],wk);

		/* multiply by a */
		dspmv('L',alen[k],1.0,a[k],wk,1,0.0,wk1,1);

		/* scatter back terms and put in w */
		for(i = 0; i < alen[k]; ++i)
			w[map[k][i]] += wk1[i];
	}
}


static void Precon(Bsystem *B, double *r, double *z)
{
	switch(B->Precon)
	{
		case Pre_Diag:
			dvmul(B->Pmat->info.diag.ndiag,B->Pmat->info.diag.idiag,1,r,1,z,1);
			break;
		case Pre_Block:
			fprintf(stderr,"Recursive Block precondition not set up\n");
			exit(1);
			break;
		case Pre_None:
			dcopy(B->rslv->rdata[B->rslv->nrecur-1].cstart,r,1,z,1);
			break;
		case Pre_SSOR:
		{
			int i, j;
			int nlines = B->rslv->A.Spa.nlines;
			double *sa = B->rslv->A.Spa.sa;
			int *ija = B->rslv->A.Spa.ija;
			int *rpt = B->rslv->A.Spa.rpt;
			double *tmp = B->Pmat->info.ssor.tmp;
			
			dzero (nlines, tmp, 1);
			
			for (i = 0; i < nlines; i++)
			{
				for (j = rpt[i]; j < rpt[i+1] - 1; j++)
				{
					tmp[i] += sa[j]*tmp[ija[j]];
				}
				tmp[i] = (r[i]-tmp[i])/sa[rpt[i+1]-1];
			}
			
			dzero(nlines,z,1);
			
			for (i = nlines-1; i >= 0; i--)
			{
				z[i] = tmp[i] - z[i]/sa[rpt[i+1]-1];
				for (j = rpt[i+1]-2; j >= rpt[i]; j--)
				{
					z[ija[j]] += sa[j]*z[i];
				}
			}
			
			break;
		}			
		default:
			break;
	}
}
