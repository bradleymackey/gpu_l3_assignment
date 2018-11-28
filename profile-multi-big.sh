#!/bin/bash

# note that the first run uses -f to clean prior register that may have been in use

echo testing large DG matrices
echo resultc
likwid-perfctr -f -g FLOPS_DP -C S0:1 -m ./sparsemm ./matrix/bigres/resultc.matrix ./matrix/big/DG1-mass-3D.matrix ./matrix/big/DG1-ip-laplace-3D.matrix
echo resultd
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm ./matrix/bigres/resultd.matrix ./matrix/big/DG1-ip-laplace-3D.matrix ./matrix/big/DG1-mass-3D.matrix
echo resulte
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm ./matrix/bigres/resulte.matrix ./matrix/big/DG2-mass-3D.matrix ./matrix/big/DG2-ip-laplace-3D.matrix
echo resultf
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm ./matrix/bigres/resultf.matrix ./matrix/big/DG2-ip-laplace-3D.matrix ./matrix/big/DG2-mass-3D.matrix
echo resultg
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm ./matrix/bigres/resultg.matrix ./matrix/big/DG3-mass-3D.matrix ./matrix/big/DG3-ip-laplace-3D.matrix
echo resulth
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm ./matrix/bigres/resulth.matrix ./matrix/big/DG3-ip-laplace-3D.matrix ./matrix/big/DG3-mass-3D.matrix
echo resulti
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm ./matrix/bigres/resulti.matrix ./matrix/big/DG4-mass-3D.matrix ./matrix/big/DG4-ip-laplace-3D.matrix
echo resultj
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm ./matrix/bigres/resultj.matrix ./matrix/big/DG4-ip-laplace-3D.matrix ./matrix/big/DG4-mass-3D.matrix


echo testing bigger 3D
echo resultk
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm ./matrix/bigres/resultk.matrix ./matrix/big/large-CG1-mass-3D.matrix ./matrix/big/small-CG1-laplace-3D.matrix
echo resultl
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm ./matrix/bigres/resultl.matrix ./matrix/big/small-CG1-laplace-3D.matrix ./matrix/big/large-CG1-mass-3D.matrix
echo resultm
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm ./matrix/bigres/resultm.matrix ./matrix/big/large-CG2-mass-3D.matrix ./matrix/big/large-CG2-laplace-3D.matrix
echo resultn
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm ./matrix/bigres/resultn.matrix ./matrix/big/large-CG2-laplace-3D.matrix ./matrix/big/large-CG2-mass-3D.matrix
