bias_imex_nout|-conv -apc jacobi_pcg -gd 1 -rtol 1e-8 -cflt 0.5 -mref 1 -nout
bias_imex_dout|-conv -apc jacobi_pcg -gd 1 -rtol 1e-8 -cflt 0.5 -mref 1 -dout
cal_imex_pcg|-conv -apc jacobi_pcg -gd 1 -rtol 1e-8 -cflt 0.5 -mref 1 -dout -tf 4.5
cal_rot_pbj_cc|-rot -pbj -schur cc -gd 1 -rtol 1e-8 -cflt 0.5 -mref 1 -dout -tf 4.5
cal_imex_amg|-conv -apc loramg -gd 1 -rtol 1e-8 -cflt 0.5 -mref 1 -dout -tf 4.5
cal_rot_pbj_auto|-rot -pbj -schur auto -gd 1 -rtol 1e-8 -cflt 0.5 -mref 1 -dout -tf 4.5
cal_imex_cheb|-conv -apc jacobi_chebyshev -gd 1 -rtol 1e-8 -cflt 0.5 -mref 1 -dout -tf 4.5
cal_rot_sym_pcg|-rot -apc jacobi_pcg -schur cc -gd 1 -rtol 1e-8 -cflt 0.5 -mref 1 -dout -tf 4.5
