# rocprof-compute 3.4.0 (matches ROCm 7.2.3 rocprofiler-sdk); python deps live in a
# --system-site-packages venv so the profiled workload still runs under the system python.
RPC_HOME=/home/hangy/rocprof-compute-build
$RPC_HOME/venv-7.2.3/bin/python $RPC_HOME/install-7.2.3/bin/rocprof-compute profile --name mxfp4_gemm --roof-only -k gemm_a4w4_wg_kernel -- python test_gemm_a4w4.py

$RPC_HOME/venv-7.2.3/bin/python $RPC_HOME/install-7.2.3/bin/rocprof-compute analyze -p workloads/mxfp4_gemm/MI350 -b 4
