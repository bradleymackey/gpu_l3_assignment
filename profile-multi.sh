#!/bin/bash

likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resulta.matrix ./matrix/small/testA.matrix ./matrix/small/testB.matrix
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultb.matrix ./matrix/small/testB.matrix ./matrix/small/testA.matrix

likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultc.matrix ./matrix/small/DG1-mass-2D.matrix ./matrix/small/DG1-ip-laplace-2D.matrix
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultd.matrix ./matrix/small/DG1-ip-laplace-2D.matrix ./matrix/small/DG1-mass-2D.matrix
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resulte.matrix ./matrix/small/DG2-mass-2D.matrix ./matrix/small/DG2-ip-laplace-2D.matrix
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultf.matrix ./matrix/small/DG2-ip-laplace-2D.matrix ./matrix/small/DG2-mass-2D.matrix
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultg.matrix ./matrix/small/DG3-mass-2D.matrix ./matrix/small/DG3-ip-laplace-2D.matrix
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resulth.matrix ./matrix/small/DG3-ip-laplace-2D.matrix ./matrix/small/DG3-mass-2D.matrix
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resulti.matrix ./matrix/small/DG4-mass-2D.matrix ./matrix/small/DG4-ip-laplace-2D.matrix
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultj.matrix ./matrix/small/DG4-ip-laplace-2D.matrix ./matrix/small/DG4-mass-2D.matrix

likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultk.matrix ./matrix/small/small-CG1-mass-3D.matrix ./matrix/small/small-CG1-laplace-3D.matrix
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultl.matrix ./matrix/small/small-CG1-laplace-3D.matrix ./matrix/small/small-CG1-mass-3D.matrix
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultm.matrix ./matrix/small/small-CG2-mass-3D.matrix ./matrix/small/small-CG2-laplace-3D.matrix
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm resultn.matrix ./matrix/small/small-CG2-laplace-3D.matrix ./matrix/small/small-CG2-mass-3D.matrix
