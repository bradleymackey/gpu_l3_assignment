#!/bin/bash

likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm testy.matrix ./matrix/small/testA.matrix ./matrix/small/testB.matrix
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm testy.matrix ./matrix/small/DG1-mass-2D.matrix ./matrix/small/DG1-ip-laplace-2D.matrix
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm testy.matrix ./matrix/small/DG1-ip-laplace-2D.matrix ./matrix/small/DG1-mass-2D.matrix
