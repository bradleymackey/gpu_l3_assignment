#!/bin/bash

# note that the first run uses -f to clean prior register that may have been in use

echo testing large DG matrices
echo resultc
time ./sparsemm --binary OUT.bm ./matrix/big/DG1-mass-3D.bm ./matrix/big/DG1-ip-laplace-3D.bm
# likwid-perfctr -f -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG1-mass-3D.bm ./matrix/big/DG1-ip-laplace-3D.bm
echo resultd
time ./sparsemm --binary OUT.bm ./matrix/big/DG1-ip-laplace-3D.bm ./matrix/big/DG1-mass-3D.bm
# likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG1-ip-laplace-3D.bm ./matrix/big/DG1-mass-3D.bm
echo resulte
time ./sparsemm --binary OUT.bm ./matrix/big/DG2-mass-3D.bm ./matrix/big/DG2-ip-laplace-3D.bm
# likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG2-mass-3D.bm ./matrix/big/DG2-ip-laplace-3D.bm
echo resultf
time ./sparsemm --binary OUT.bm ./matrix/big/DG2-ip-laplace-3D.bm ./matrix/big/DG2-mass-3D.bm
# likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG2-ip-laplace-3D.bm ./matrix/big/DG2-mass-3D.bm
echo resultg
time ./sparsemm --binary OUT.bm ./matrix/big/DG3-mass-3D.bm ./matrix/big/DG3-ip-laplace-3D.bm
# likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG3-mass-3D.bm ./matrix/big/DG3-ip-laplace-3D.bm
echo resulth
time ./sparsemm --binary OUT.bm ./matrix/big/DG3-ip-laplace-3D.bm ./matrix/big/DG3-mass-3D.bm
# likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG3-ip-laplace-3D.bm ./matrix/big/DG3-mass-3D.bm
echo resulti
time ./sparsemm --binary OUT.bm ./matrix/big/DG4-mass-3D.bm ./matrix/big/DG4-ip-laplace-3D.bm
# likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG4-mass-3D.bm ./matrix/big/DG4-ip-laplace-3D.bm
echo resultj
time ./sparsemm --binary OUT.bm ./matrix/big/DG4-ip-laplace-3D.bm ./matrix/big/DG4-mass-3D.bm
# likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/DG4-ip-laplace-3D.bm ./matrix/big/DG4-mass-3D.bm


echo testing bigger 3D
echo resultk
time ./sparsemm --binary OUT.bm ./matrix/big/large-CG1-mass-3D.bm ./matrix/big/large-CG1-laplace-3D.bm
# likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/large-CG1-mass-3D.bm ./matrix/big/large-CG1-laplace-3D.bm
echo resultl
time ./sparsemm --binary OUT.bm ./matrix/big/large-CG1-laplace-3D.bm ./matrix/big/large-CG1-mass-3D.bm
# likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/large-CG1-laplace-3D.bm ./matrix/big/large-CG1-mass-3D.bm
echo resultm
time ./sparsemm --binary OUT.bm ./matrix/big/large-CG2-mass-3D.bm ./matrix/big/large-CG2-laplace-3D.bm
# likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/large-CG2-mass-3D.bm ./matrix/big/large-CG2-laplace-3D.bm
echo resultn
time ./sparsemm --binary OUT.bm ./matrix/big/large-CG2-laplace-3D.bm ./matrix/big/large-CG2-mass-3D.bm
# likwid-perfctr -g FLOPS_DP -C S0:1 -m ./sparsemm --binary OUT.bm ./matrix/big/large-CG2-laplace-3D.bm ./matrix/big/large-CG2-mass-3D.bm
