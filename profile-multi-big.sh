#!/bin/bash

# note that the first run uses -f to clean prior register that may have been in use

echo testing large DG matrices
echo resultc
./sparsemm --binary OUT.bm ./matrix/big/DG1-mass-3D.bm ./matrix/big/DG1-ip-laplace-3D.bm
likwid-perfctr -f -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG1-mass-3D.bm ./matrix/big/DG1-ip-laplace-3D.bm
echo resultd
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG1-ip-laplace-3D.bm ./matrix/big/DG1-mass-3D.bm
echo resulte
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG2-mass-3D.bm ./matrix/big/DG2-ip-laplace-3D.bm
echo resultf
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG2-ip-laplace-3D.bm ./matrix/big/DG2-mass-3D.bm
echo resultg
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG3-mass-3D.bm ./matrix/big/DG3-ip-laplace-3D.bm
echo resulth
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG3-ip-laplace-3D.bm ./matrix/big/DG3-mass-3D.bm
echo resulti
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG4-mass-3D.bm ./matrix/big/DG4-ip-laplace-3D.bm
echo resultj
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG4-ip-laplace-3D.bm ./matrix/big/DG4-mass-3D.bm


echo testing bigger 3D
echo resultk
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/large-CG1-mass-3D.bm ./matrix/big/large-CG1-laplace-3D.bm
echo resultl
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/large-CG1-laplace-3D.bm ./matrix/big/large-CG1-mass-3D.bm
echo resultm
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/large-CG2-mass-3D.bm ./matrix/big/large-CG2-laplace-3D.bm
echo resultn
likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/large-CG2-laplace-3D.bm ./matrix/big/large-CG2-mass-3D.bm
