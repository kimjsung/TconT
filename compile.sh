rm -rf build
cmake -S . -B build -DCOGENT_CUBIN_COMPILE_OPT=DEFAULT
# cmake -S . -B build -DCOGENT_CUBIN_COMPILE_OPT=A100
cmake --build build --target bench_tccg -j4