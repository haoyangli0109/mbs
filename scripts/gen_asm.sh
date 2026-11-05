hipcc -O3 -std=c++20 --offload-arch=gfx950 -D__HIP_PLATFORM_AMD__=1 -D__Float4_e2m1fn_x2 -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-unused-result -Ikernel kernel/example.cpp -S -o build/example.s 
