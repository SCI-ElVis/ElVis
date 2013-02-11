/* ------------------------------------------------------------------------- *
 * StokesBC() - Calculate the high-order boundary conditions for Stokes flow *
 *                                                                           *
 * This routine simply sets the non-linear terms to zero.                    *
 *                                                                           *
 * RCS Information                                  
 * ---------------
 * $Author: bscarmo $
 * $Date: 2009-04-09 14:19:37 $
 * $Source: /homedir/cvs/Nektar/Nektar3d/src/stokes.C,v $
 * $Revision: 1.3 $
 * ------------------------------------------------------------------------- */

#include "nektar.h"

void StokesBC(Domain *omega){
  int eDIM = omega->Uf->fhead->dim();
  /* Set the non-linear terms to zero */

  if(eDIM == 2){
    dzero(omega->Uf->htot, omega->Uf->base_h, 1);
    dzero(omega->Vf->htot, omega->Vf->base_h, 1);
    
    dzero(omega->Uf->hjtot, omega->Uf->base_hj, 1);
    dzero(omega->Vf->hjtot, omega->Vf->base_hj, 1);

    omega->Uf->Set_state('p');
    omega->Vf->Set_state('p');
  }
  else{
    dzero(omega->Uf->htot, omega->Uf->base_h, 1);
    dzero(omega->Vf->htot, omega->Vf->base_h, 1);
    dzero(omega->Wf->htot, omega->Vf->base_h, 1);
    
    dzero(omega->Uf->hjtot, omega->Uf->base_hj, 1);
    dzero(omega->Vf->hjtot, omega->Vf->base_hj, 1);
    dzero(omega->Wf->hjtot, omega->Vf->base_hj, 1);
    
    set_state(omega->Wf->fhead,'p');
  }
  return;
}

