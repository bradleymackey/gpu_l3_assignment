#!/bin/bash

# note that the first run uses -f to clean prior register that may have been in use

echo testing small matricies
echo resulta
likwid-perfctr -f -g FLOPS_DP -C S0:1 -m ./sparsemm resulta.matrix ./matrix/small/testA.matrix ./matrix/small/testB.matrix
echo resultb
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultb.matrix ./matrix/small/testB.matrix ./matrix/small/testA.matrix

echo testing small DG matrices
echo resultc
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultc.matrix ./matrix/small/DG1-mass-2D.matrix ./matrix/small/DG1-ip-laplace-2D.matrix
echo resultd
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultd.matrix ./matrix/small/DG1-ip-laplace-2D.matrix ./matrix/small/DG1-mass-2D.matrix
echo resulte
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resulte.matrix ./matrix/small/DG2-mass-2D.matrix ./matrix/small/DG2-ip-laplace-2D.matrix
echo resultf
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultf.matrix ./matrix/small/DG2-ip-laplace-2D.matrix ./matrix/small/DG2-mass-2D.matrix
echo resultg
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultg.matrix ./matrix/small/DG3-mass-2D.matrix ./matrix/small/DG3-ip-laplace-2D.matrix
echo resulth
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resulth.matrix ./matrix/small/DG3-ip-laplace-2D.matrix ./matrix/small/DG3-mass-2D.matrix
echo resulti
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resulti.matrix ./matrix/small/DG4-mass-2D.matrix ./matrix/small/DG4-ip-laplace-2D.matrix
echo resultj
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultj.matrix ./matrix/small/DG4-ip-laplace-2D.matrix ./matrix/small/DG4-mass-2D.matrix


echo testing bigger 3D
echo resultk
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultk.matrix ./matrix/small/small-CG1-mass-3D.matrix ./matrix/small/small-CG1-laplace-3D.matrix
echo resultl
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultl.matrix ./matrix/small/small-CG1-laplace-3D.matrix ./matrix/small/small-CG1-mass-3D.matrix
echo resultm
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultm.matrix ./matrix/small/small-CG2-mass-3D.matrix ./matrix/small/small-CG2-laplace-3D.matrix
echo resultn
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultn.matrix ./matrix/small/small-CG2-laplace-3D.matrix ./matrix/small/small-CG2-mass-3D.matrix
