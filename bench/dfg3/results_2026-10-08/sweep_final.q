m1_imex_c0.3|-conv -apc jacobi_pcg -gd 1 -dout -rtol 1e-8 -ext 3 -mref 1 -cflt 0.3
m1_rot_c0.3|-rot -pbj -schur cc -gd 1 -dout -rtol 1e-8 -ext 3 -mref 1 -cflt 0.3
m1_imex_c0.5|-conv -apc jacobi_pcg -gd 1 -dout -rtol 1e-8 -ext 3 -mref 1 -cflt 0.5
m1_rot_c0.5|-rot -pbj -schur cc -gd 1 -dout -rtol 1e-8 -ext 3 -mref 1 -cflt 0.5
m1_imex_c0.7|-conv -apc jacobi_pcg -gd 1 -dout -rtol 1e-8 -ext 3 -mref 1 -cflt 0.7
m1_rot_c0.7|-rot -pbj -schur cc -gd 1 -dout -rtol 1e-8 -ext 3 -mref 1 -cflt 0.7
m1_imex_c0.9|-conv -apc jacobi_pcg -gd 1 -dout -rtol 1e-8 -ext 3 -mref 1 -cflt 0.9
m1_imex_e2_c0.7|-conv -apc jacobi_pcg -gd 1 -dout -rtol 1e-8 -ext 2 -mref 1 -cflt 0.7
m1_rot_e2_c0.7|-rot -pbj -schur cc -gd 1 -dout -rtol 1e-8 -ext 2 -mref 1 -cflt 0.7
m1_imex_e2_c0.9|-conv -apc jacobi_pcg -gd 1 -dout -rtol 1e-8 -ext 2 -mref 1 -cflt 0.9
m1_rot_e2_c0.9|-rot -pbj -schur cc -gd 1 -dout -rtol 1e-8 -ext 2 -mref 1 -cflt 0.9
m2_imex_e2_c0.7|-conv -apc jacobi_pcg -gd 1 -dout -rtol 1e-8 -ext 2 -mref 2 -cflt 0.7
m2_rot_e2_c0.7|-rot -pbj -schur cc -gd 1 -dout -rtol 1e-8 -ext 2 -mref 2 -cflt 0.7
