# GEMM lab: the harness with every kernel version.
#   make            -> $(BUILD_DIR)/gemm_lab
#   make asm        -> $(BUILD_DIR)/gemm_lab.s   (device ISA with line info)
#   make src        -> $(BUILD_DIR)/gemm_lab_src.txt   (device ISA interleaved with the source lines)
#   make resources  -> VGPR / AGPR / SGPR / scratch / occupancy / LDS of every kernel
#   make debug      -> $(BUILD_DIR)/gemm_lab_dbg and $(BUILD_DIR)/dbg_demo   (-O0 -g, for rocgdb)
HIPCC     ?= hipcc
OBJDUMP   ?= /opt/rocm/llvm/bin/llvm-objdump
ARCH      ?= gfx950
BUILD_DIR ?= build
FLAGS     := -O3 -std=c++20 --offload-arch=$(ARCH) -gline-tables-only
DBG_FLAGS := -O0 -g -std=c++20 --offload-arch=$(ARCH)
SRCS      := gemm_lab.hip $(wildcard *.h)

$(BUILD_DIR)/gemm_lab: $(SRCS) | $(BUILD_DIR)
	$(HIPCC) $(FLAGS) gemm_lab.hip -o $@

asm: $(BUILD_DIR)/gemm_lab.s

$(BUILD_DIR)/gemm_lab.s: $(SRCS) | $(BUILD_DIR)
	$(HIPCC) $(FLAGS) --cuda-device-only -S gemm_lab.hip -o $@

src: $(BUILD_DIR)/gemm_lab_src.txt

$(BUILD_DIR)/gemm_lab.co: $(SRCS) | $(BUILD_DIR)
	$(HIPCC) $(FLAGS) --cuda-device-only --no-gpu-bundle-output -c gemm_lab.hip -o $@

$(BUILD_DIR)/gemm_lab_src.txt: $(BUILD_DIR)/gemm_lab.co
	$(OBJDUMP) -d --source -l --no-show-raw-insn --symbolize-operands $< \
	  | sed -E 's@ +// [0-9A-F]{12}:.*@@' > $@

resources: $(SRCS)
	$(HIPCC) $(FLAGS) --cuda-device-only -c -o /dev/null \
	    -Rpass-analysis=kernel-resource-usage gemm_lab.hip 2>&1 \
	  | grep "remark:" | sed 's/.*remark: //; s/ \[-Rpass-analysis=kernel-resource-usage\]//'

debug: $(BUILD_DIR)/gemm_lab_dbg $(BUILD_DIR)/dbg_demo

$(BUILD_DIR)/gemm_lab_dbg: $(SRCS) | $(BUILD_DIR)
	$(HIPCC) $(DBG_FLAGS) gemm_lab.hip -o $@

$(BUILD_DIR)/dbg_demo: dbg_demo.hip | $(BUILD_DIR)
	$(HIPCC) $(DBG_FLAGS) dbg_demo.hip -o $@

$(BUILD_DIR):
	mkdir -p $@

.PHONY: asm src resources debug
