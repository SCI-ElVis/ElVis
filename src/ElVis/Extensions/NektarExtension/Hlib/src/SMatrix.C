#include <veclib.h>
#include "hotel.h"
#include <math.h>

#ifdef UMFPACKSLV
extern "C"
{
	#include <umfpack.h>
}
#endif

// Build Gathered Global Sparse Matrix from scattered dense matrices taking
// symmetry into account - non-Recursive solver
void BuildSMat(Bsystem *Ubsys, Element_List *U)
{
	register int i, j, k;
	int nel = Ubsys->nel;
	int nbmodes, geomid;
	double val;
	double **a = Ubsys->Gmat->a;
	double *sc = Ubsys->signchange;
	int *bmap;
	int nlines = Ubsys->Gmat->Spa.nlines;
	
	ZeroSMat(Ubsys->Gmat->Spa);
	
	for (k = 0; k < nel; k++)
	{
		nbmodes = U->flist[k]->Nbmodes;
		geomid  = U->flist[k]->geom->id;
		bmap    = Ubsys->bmap[k];
		
		for (j = 0; j < nbmodes; j++)
		{
			if (bmap[j] < nlines)
			{
				for (i = j; i < nbmodes; i++)
				{
					if (bmap[i] < nlines)
					{
						val = a[geomid][i+((2*nbmodes-j-1)*j)/2]*sc[i]*sc[j];
						if (bmap[j] <= bmap[i])
							AddContSMat(val, Ubsys->Gmat->Spa, bmap[i],bmap[j]);
						else 	// bmap[i] < bmap[j]
							AddContSMat(val, Ubsys->Gmat->Spa, bmap[j],bmap[i]);
					}
				}
			}
		}
		sc += nbmodes;
	}
	
	if (Ubsys->Gmat->Spa.status == OPEN)
		CloseSMat(Ubsys->Gmat->Spa);
}

// Build Gathered Global Sparse Matrix from scattered dense matrices taking
// symmetry into account - non-Recursive solver - selective matrix regeneration
void Selec_BuildSMat(Bsystem *Ubsys, Element_List *U)
{
	register int i, j, k;
	int nel = Ubsys->nel;
	int nbmodes, geomid;
	double val;
	double **a = Ubsys->Gmat->a;
	double *sc = Ubsys->signchange;
	int *bmap;
	int nlines = Ubsys->Gmat->Spa.nlines;
	
	for (k = 0; k < nel; k++)
	{
		nbmodes = U->flist[k]->Nbmodes;
		if (U->flist[k]->group < 0)
		{
			geomid  = U->flist[k]->geom->id;
			bmap    = Ubsys->bmap[k];
		
			for (j = 0; j < nbmodes; j++)
			{
				if (bmap[j] < nlines)
				{
					for (i = j; i < nbmodes; i++)
					{
						if (bmap[i] < nlines)
						{
							val = a[geomid][i+((2*nbmodes-j-1)*j)/2]*
								  sc[i]*sc[j];
							if (bmap[j] <= bmap[i])
								AddContSMat(val, Ubsys->Gmat->Spa,
											bmap[i], bmap[j]);
							else 	// bmap[i] < bmap[j]
								AddContSMat(val, Ubsys->Gmat->Spa,
										    bmap[j], bmap[i]);
						}
					}
				}
			}
		}
		sc += nbmodes;
	}
	
	if (Ubsys->Gmat->Spa.status == OPEN)
		CloseSMat(Ubsys->Gmat->Spa);
}


// Build Gathered Global Sparse Matrix from scattered dense matrices taking
// symmetry into account - Recursive solver
void RBuildSMat(Rsolver *rslv)
{
	register int i, j, k;
	Recur *rdata = rslv->rdata + rslv->nrecur-1;
	int npatch = rdata->npatch;
	int *alen  = rdata->patchlen_a;
	double **a = rslv->A.a;
	
	double val;
	int *map;
	
	ZeroSMat(rslv->A.Spa);
	
	for (k = 0; k < npatch; k++)
	{
		map    = rdata->map[k];
		
		for (j = 0; j < alen[k]; j++)
		{
			for (i = j; i < alen[k]; i++)
			{
				val = a[k][i+((2*alen[k]-j-1)*j)/2];
				if (map[j] <= map[i])
					AddContSMat(val, rslv->A.Spa, map[i],map[j]);
				if (map[i] < map[j])
					AddContSMat(val, rslv->A.Spa, map[j],map[i]);
			}
		}
	}
	
	if (rslv->A.Spa.status == OPEN)
		CloseSMat(rslv->A.Spa);
}


// Add contribution (val) to sparse matrix (in Gmat structure) at position
// (row, col)
void AddContSMat(double val, SparseMat & sm, int row, int col)
{
	if (sm.status == OPEN)
	{
		PMatrixElem *pel = findElPMat(sm,row,col);
		pel->val += val;
	}
	else
	{
		register int j;
		
		for (j = sm.rpt[row]; j < sm.rpt[row+1] && sm.ija[j] < col; j++);
		
		if (sm.ija[j] == col && j < sm.rpt[row+1])
			sm.sa[j] += val;
		else
		{
			fprintf(stderr,"\nTrying to access a non-existent element in a "
						   "closed sparse matrix. Aborting.\n");
			exit(-1);
		}
	}	
}


// Routine that searches an element in a pre-matrix (open form of sparse matrix)
PMatrixElem * findElPMat(SparseMat & sm, int row, int col)
{
	PMatrixElem *premat = sm.premat;
	
	if (premat[row].col < 0)
	{
		premat[row].col = col;
		premat[row].val = 0.0;
		sm.nnz++;
		return &(premat[row]);
	}
	else
	{
		PMatrixElem *pel = NULL;
		
		if (premat[row].col > col)
		{
			pel = (PMatrixElem *)malloc(sizeof(PMatrixElem));
			pel->col = premat[row].col;
			pel->val = premat[row].val;
			pel->next = premat[row].next;
			premat[row].col = col;
			premat[row].val = 0.0;
			premat[row].next = pel;
			sm.nnz++;
			pel = &(premat[row]);
		}
		else if (premat[row].col == col)
			pel = &(premat[row]);
		else
		{
			PMatrixElem *pan = &(premat[row]);
		
			while (pan->next != NULL && pan->next->col < col) pan = pan->next;
		
			if (pan->next == NULL)
			{
				pel = (PMatrixElem *)malloc(sizeof(PMatrixElem));
				pel->col = col;
				pel->val = 0.0;
				pel->next = NULL;
				pan->next = pel;
				sm.nnz++;
			}
			else if (pan->next->col == col)
				pel = pan->next;
			else if (pan->next->col > col)
			{
				pel = (PMatrixElem *)malloc(sizeof(PMatrixElem));
				pel->col = col;
				pel->val = 0.0;
				pel->next = pan->next;
				pan->next = pel;
				sm.nnz++;
			}
		}
		return pel;
	}
}


// Put sparse matrix into the closed form (for calculations)
void CloseSMat(SparseMat & sm)
{
	if (sm.status == CLOSED) return;
	
	register int i, k;
	PMatrixElem *pel;
	
	sm.rpt = (int *)malloc((sm.nlines+1)*sizeof(int));
	sm.ija = (int *)malloc(sm.nnz*sizeof(int));
	sm.sa = (double *)malloc(sm.nnz*sizeof(double));
	
	k = 0;
	
	for (i = 0; i < sm.nlines; i++)
	{
		sm.rpt[i] = k;
		for (pel = &(sm.premat[i]); pel != NULL && pel->col >= 0;
			 pel = pel->next)
		{
			sm.ija[k] = pel->col;
			sm.sa[k] = pel->val;
			k++;
		}			
	}
	sm.rpt[sm.nlines] = sm.nnz;
	
	freepremat(sm);
	sm.status = CLOSED;
}


// Free memory alocated for prematrix structure
void freepremat(SparseMat & sm)
{
	if (!(sm.premat)) return;
	
	register int i;
	PMatrixElem *pel, *pan;
	
	for (i = 0; i < sm.nlines; i++)
	{
		pel = sm.premat[i].next;
		while (pel != NULL)
		{
			pan = pel;
			pel = pel->next;
			free(pan);
		}
	}
	free(sm.premat);
	sm.premat = NULL;
}


//Zero all the elements of a matrix, conserving pattern if the matrix is closed
void ZeroSMat(SparseMat & sm)
{
	if (sm.status == OPEN)
	{ 
		register int i;
		freepremat(sm);
		sm.premat = (PMatrixElem *)malloc(sm.nlines*sizeof(PMatrixElem));
					
		for (i = 0; i < sm.nlines; i++)
		{
			sm.premat[i].next = NULL;
			sm.premat[i].col = -1;
			sm.premat[i].val = 0.0;  
		}
	}
	else
		dzero(sm.nnz, sm.sa, 1);
}


//Init SparseMat data structure
void InitSMat(SparseMat & sm, int nl)
{
	int i;
	sm.status = OPEN;
	sm.nnz = 0;
	sm.nlines = nl;
	sm.premat = (PMatrixElem *)malloc(sm.nlines*sizeof(PMatrixElem));
	for (i = 0; i < sm.nlines; i++)
	{
		sm.premat[i].next = NULL;
		sm.premat[i].col = -1;
		sm.premat[i].val = 0.0;  
	}
#ifdef UMFPACKSLV
	sm.symbolic = (void *)NULL;	
	sm.numeric = (void *)NULL;
	sm.tmprhs = (double *)NULL;
#endif
}


// Free memory alocated for SparseMat data structure
void FreeSMat(SparseMat & sm)
{
	if (sm.status == OPEN)
		freepremat(sm);
	else
	{
		free(sm.rpt);
		free(sm.ija);
		free(sm.sa);
		sm.nlines = 0;
		sm.nnz = 0;
		sm.status = OPEN;
#ifdef UMFPACKSLV
		if (sm.symbolic)
			umfpack_di_free_symbolic(&sm.symbolic);
		if (sm.numeric)
			umfpack_di_free_numeric(&sm.numeric);
		if (sm.tmprhs)
			free(sm.tmprhs);
#endif
	}	
}	
