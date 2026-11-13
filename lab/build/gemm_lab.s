	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_ZN2v013gemm_v0_naiveE7Problem ; -- Begin function _ZN2v013gemm_v0_naiveE7Problem
	.globl	_ZN2v013gemm_v0_naiveE7Problem
	.p2align	8
	.type	_ZN2v013gemm_v0_naiveE7Problem,@function
_ZN2v013gemm_v0_naiveE7Problem:         ; @_ZN2v013gemm_v0_naiveE7Problem
.Lfunc_begin0:
	.file	0 "/home/haoyanli/a121_quark/mbs/lab" "gemm_lab.hip" md5 0x394887fa59d9a0b9e418e5cbe756be2c
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; 
	.cfi_undefined 16
	.file	1 "/opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail" "amd_device_functions.h" md5 0x1a8b7178a036d604cccb3d2554abd5c3
	.loc	1 623 25 prologue_end           ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:623:25 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:634:83 @[ ./v0_naive.h:17:23 ] ]
	s_memtime s[20:21]
.Ltmp0:
	.loc	1 623 25 is_stmt 0              ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:623:25 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:634:83 @[ ./v0_naive.h:26:21 ] ]
	s_memtime s[22:23]
.Ltmp1:
	.file	2 "." "v0_naive.h" md5 0xdb790ced971ad47b552b828380ce9d6c
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	s_load_dwordx2 s[18:19], s[0:1], 0x88
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_mbcnt_hi_u32_b32 v1, -1, v1
	v_mov_b64_e32 v[6:7], 0
	v_readfirstlane_b32 s4, v1
	s_nop 1
	v_cmp_eq_u32_e64 s[4:5], s4, v1
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB0_6
; %bb.1:
	v_mov_b32_e32 v2, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[8:9], v2, s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v2, s[18:19] offset:40
	global_load_dwordx2 v[6:7], v2, s[18:19]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v3, v4, v8
	v_and_b32_e32 v4, v5, v9
	v_mul_lo_u32 v4, v4, 24
	v_mul_hi_u32 v5, v3, 24
	v_add_u32_e32 v5, v5, v4
	v_mul_lo_u32 v4, v3, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[6:7], 0, v[4:5]
	global_load_dwordx2 v[6:7], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v2, v[6:9], s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[6:7], v[8:9]
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_5
; %bb.2:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_mov_b64 s[10:11], 0
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	s_sleep 1
	global_load_dwordx2 v[4:5], v2, s[18:19] offset:40
	global_load_dwordx2 v[10:11], v2, s[18:19]
	v_mov_b64_e32 v[8:9], v[6:7]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v4, v8
	v_and_b32_e32 v3, v5, v9
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[12:13], v4, 24, v[10:11]
	v_mov_b32_e32 v6, v5
	v_mad_u64_u32 v[6:7], s[12:13], v3, 24, v[6:7]
	v_mov_b32_e32 v5, v6
	global_load_dwordx2 v[6:7], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v2, v[6:9], s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]
	s_or_b64 s[10:11], vcc, s[10:11]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB0_3
; %bb.4:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[10:11]
.LBB0_5:
	s_or_b64 exec, exec, s[8:9]
.LBB0_6:
	s_or_b64 exec, exec, s[6:7]
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	v_mov_b32_e32 v25, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[8:9], v25, s[18:19] offset:40
	global_load_dwordx4 v[2:5], v25, s[18:19]
	s_load_dwordx8 s[8:15], s[0:1], 0x10
	s_load_dwordx2 s[16:17], s[0:1], 0x30
	v_readfirstlane_b32 s25, v7
	v_readfirstlane_b32 s24, v6
	s_mov_b64 s[6:7], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s26, v8
	v_readfirstlane_b32 s27, v9
	s_and_b64 s[26:27], s[26:27], s[24:25]
	s_mul_i32 s28, s27, 24
	s_mul_hi_u32 s29, s26, 24
	s_add_i32 s29, s29, s28
	s_mul_i32 s28, s26, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[2:3], 0, s[28:29]
	s_and_saveexec_b64 s[28:29], s[4:5]
	s_cbranch_execz .LBB0_8
; %bb.7:
	v_mov_b64_e32 v[8:9], s[6:7]
	v_mov_b32_e32 v10, 2
	v_mov_b32_e32 v11, 1
	global_store_dwordx4 v[6:7], v[8:11], off offset:8
.LBB0_8:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[28:29]
	s_load_dwordx2 s[6:7], s[0:1], 0x4
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_lshl_b64 s[0:1], s[26:27], 12
	v_lshl_add_u64 v[8:9], v[4:5], 0, s[0:1]
	s_mov_b32 s28, 0
	v_lshlrev_b32_e32 v24, 6, v1
	v_mov_b32_e32 v10, 33
	v_mov_b32_e32 v11, v25
	v_mov_b32_e32 v12, v25
	v_mov_b32_e32 v13, v25
	v_readfirstlane_b32 s0, v8
	v_readfirstlane_b32 s1, v9
	s_mov_b32 s29, s28
	s_mov_b32 s30, s28
	s_mov_b32 s31, s28
	s_nop 1
	global_store_dwordx4 v24, v[10:13], s[0:1]
	s_nop 1
	v_mov_b64_e32 v[10:11], s[28:29]
	v_mov_b64_e32 v[12:13], s[30:31]
	global_store_dwordx4 v24, v[10:13], s[0:1] offset:16
	global_store_dwordx4 v24, v[10:13], s[0:1] offset:32
	global_store_dwordx4 v24, v[10:13], s[0:1] offset:48
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB0_16
; %bb.9:
	v_mov_b32_e32 v12, 0
	global_load_dwordx2 v[16:17], v12, s[18:19] offset:32 sc0 sc1
	global_load_dwordx2 v[4:5], v12, s[18:19] offset:40
	v_mov_b32_e32 v14, s24
	v_mov_b32_e32 v15, s25
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v4, s24, v4
	v_and_b32_e32 v5, s25, v5
	v_mul_lo_u32 v5, v5, 24
	v_mul_hi_u32 v10, v4, 24
	v_mul_lo_u32 v4, v4, 24
	v_add_u32_e32 v5, v10, v5
	v_lshl_add_u64 v[10:11], v[2:3], 0, v[4:5]
	global_store_dwordx2 v[10:11], v[16:17], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v12, v[14:17], s[18:19] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[16:17]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_12
; %bb.10:
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_mov_b64 s[28:29], 0
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	s_sleep 1
	global_store_dwordx2 v[10:11], v[4:5], off
	v_mov_b32_e32 v2, s24
	v_mov_b32_e32 v3, s25
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v12, v[2:5], s[18:19] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]
	s_or_b64 s[28:29], vcc, s[28:29]
	v_mov_b64_e32 v[4:5], v[2:3]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB0_11
.LBB0_12:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[26:27]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_mov_b32_e32 v5, 0
	global_load_dwordx2 v[2:3], v5, s[18:19] offset:16
	s_mov_b64 s[26:27], exec
	v_mbcnt_lo_u32_b32 v4, s26, 0
	v_mbcnt_hi_u32_b32 v4, s27, v4
	v_cmp_eq_u32_e32 vcc, 0, v4
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_bcnt1_i32_b64 s26, s[26:27]
	v_mov_b32_e32 v4, s26
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[2:3], v[4:5], off offset:8 sc1
.LBB0_14:
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[28:29]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[4:5], v[2:3], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[4:5]
	s_cbranch_vccnz .LBB0_16
; %bb.15:
	global_load_dword v2, v[2:3], off offset:24
	v_mov_b32_e32 v3, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s26, v2
	s_and_b32 m0, s26, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[4:5], v[2:3], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_16:
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[0:1]
	v_lshl_add_u64 v[2:3], v[8:9], 0, v[24:25]
	s_branch .LBB0_20
.LBB0_17:                               ;   in Loop: Header=BB0_20 Depth=1
	s_or_b64 exec, exec, s[0:1]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_readfirstlane_b32 s0, v4
	s_cmp_eq_u32 s0, 0
	s_cbranch_scc1 .LBB0_19
; %bb.18:                               ;   in Loop: Header=BB0_20 Depth=1
	s_sleep 1
	s_cbranch_execnz .LBB0_20
	s_branch .LBB0_22
.LBB0_19:
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_branch .LBB0_22
.LBB0_20:                               ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v4, 1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB0_17
; %bb.21:                               ;   in Loop: Header=BB0_20 Depth=1
	global_load_dword v4, v[6:7], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v4, 1, v4
	s_branch .LBB0_17
.LBB0_22:
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	global_load_dwordx2 v[2:3], v[2:3], off
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB0_25
; %bb.23:
	v_mov_b32_e32 v10, 0
	global_load_dwordx2 v[4:5], v10, s[18:19] offset:40
	global_load_dwordx2 v[14:15], v10, s[18:19] offset:24 sc0 sc1
	global_load_dwordx2 v[6:7], v10, s[18:19]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	s_add_u32 s28, s26, 1
	s_addc_u32 s29, s27, 0
	s_add_u32 s4, s28, s24
	s_addc_u32 s5, s29, s25
	s_cmp_eq_u64 s[4:5], 0
	s_cselect_b32 s5, s29, s5
	s_cselect_b32 s4, s28, s4
	s_and_b64 s[24:25], s[4:5], s[26:27]
	s_mul_i32 s25, s25, 24
	s_mul_hi_u32 s26, s24, 24
	s_mul_i32 s24, s24, 24
	s_add_i32 s25, s26, s25
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[8:9], v[6:7], 0, s[24:25]
	v_mov_b32_e32 v12, s4
	global_store_dwordx2 v[8:9], v[14:15], off
	v_mov_b32_e32 v13, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v10, v[12:15], s[18:19] offset:24 sc0 sc1
	s_mov_b64 s[24:25], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_25
.LBB0_24:                               ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[8:9], v[6:7], off
	v_mov_b32_e32 v4, s4
	v_mov_b32_e32 v5, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v10, v[4:7], s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[6:7], v[4:5]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_24
.LBB0_25:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_getpc_b64 s[4:5]
	s_add_u32 s4, s4, .str@rel32@lo+4
	s_addc_u32 s5, s5, .str@rel32@hi+12
	s_cmp_lg_u64 s[4:5], 0
	s_cbranch_scc0 .LBB0_111
; %bb.26:
	v_mov_b32_e32 v29, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v26, 2, v2
	v_and_b32_e32 v4, -3, v2
	v_mov_b32_e32 v5, v3
	s_mov_b64 s[24:25], 17
	v_mov_b32_e32 v25, v29
	v_mov_b32_e32 v34, 2
	v_mov_b32_e32 v35, 1
	s_branch .LBB0_28
.LBB0_27:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[30:31]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_sub_u32 s24, s24, s26
	s_subb_u32 s25, s25, s27
	s_add_u32 s4, s4, s26
	s_addc_u32 s5, s5, s27
	s_cmp_lg_u64 s[24:25], 0
	s_cbranch_scc0 .LBB0_110
.LBB0_28:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_31 Depth 2
                                        ;     Child Loop BB0_39 Depth 2
                                        ;     Child Loop BB0_47 Depth 2
                                        ;     Child Loop BB0_55 Depth 2
                                        ;     Child Loop BB0_63 Depth 2
                                        ;     Child Loop BB0_71 Depth 2
                                        ;     Child Loop BB0_79 Depth 2
                                        ;     Child Loop BB0_87 Depth 2
                                        ;     Child Loop BB0_95 Depth 2
                                        ;     Child Loop BB0_104 Depth 2
                                        ;     Child Loop BB0_109 Depth 2
	v_cmp_lt_u64_e64 s[0:1], s[24:25], 56
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s27, s25, 0
	s_cselect_b32 s26, s24, 56
	v_cmp_gt_u64_e64 s[28:29], s[24:25], 7
	s_add_u32 s0, s4, 8
	s_addc_u32 s1, s5, 0
	s_and_b64 vcc, exec, s[28:29]
	s_cbranch_vccnz .LBB0_32
; %bb.29:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u64 s[24:25], 0
	s_cbranch_scc1 .LBB0_33
; %bb.30:                               ;   in Loop: Header=BB0_28 Depth=1
	s_lshl_b64 s[0:1], s[26:27], 3
	s_mov_b64 s[28:29], 0
	v_mov_b64_e32 v[6:7], 0
	s_mov_b64 s[30:31], s[4:5]
.LBB0_31:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_ubyte v8, v29, s[30:31]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v28, 0xffff, v8
	v_lshlrev_b64 v[8:9], s28, v[28:29]
	s_add_u32 s28, s28, 8
	s_addc_u32 s29, s29, 0
	s_add_u32 s30, s30, 1
	s_addc_u32 s31, s31, 0
	v_or_b32_e32 v6, v8, v6
	s_cmp_lg_u32 s0, s28
	v_or_b32_e32 v7, v9, v7
	s_cbranch_scc1 .LBB0_31
	s_branch .LBB0_34
.LBB0_32:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_mov_b32 s33, 0
	s_branch .LBB0_35
.LBB0_33:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[6:7], 0
.LBB0_34:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[0:1], s[4:5]
	s_mov_b32 s33, 0
	s_cbranch_execnz .LBB0_36
.LBB0_35:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	global_load_dwordx2 v[6:7], v29, s[4:5]
	s_add_i32 s33, s26, -8
.LBB0_36:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_add_u32 s28, s0, 8
	s_addc_u32 s29, s1, 0
	s_cmp_gt_u32 s33, 7
	s_cbranch_scc1 .LBB0_40
; %bb.37:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc1 .LBB0_41
; %bb.38:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_mov_b64 s[28:29], 0
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_mov_b64_e32 v[8:9], 0
	s_mov_b64 s[30:31], 0
.LBB0_39:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s34, s0, s30
	s_addc_u32 s35, s1, s31
	global_load_ubyte v10, v29, s[34:35]
	s_add_u32 s30, s30, 1
	s_addc_u32 s31, s31, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v28, 0xffff, v10
	v_lshlrev_b64 v[10:11], s28, v[28:29]
	s_add_u32 s28, s28, 8
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v8, v10, v8
	s_cmp_lg_u32 s33, s30
	v_or_b32_e32 v9, v11, v9
	s_cbranch_scc1 .LBB0_39
	s_branch .LBB0_42
.LBB0_40:                               ;   in Loop: Header=BB0_28 Depth=1
                                        ; implicit-def: $vgpr8_vgpr9
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_mov_b32 s34, 0
	s_branch .LBB0_43
.LBB0_41:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[8:9], 0
.LBB0_42:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[28:29], s[0:1]
	s_mov_b32 s34, 0
	s_cbranch_execnz .LBB0_44
.LBB0_43:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	global_load_dwordx2 v[8:9], v29, s[0:1]
	s_add_i32 s34, s33, -8
.LBB0_44:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_add_u32 s0, s28, 8
	s_addc_u32 s1, s29, 0
	s_cmp_gt_u32 s34, 7
	s_cbranch_scc1 .LBB0_48
; %bb.45:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s34, 0
	s_cbranch_scc1 .LBB0_49
; %bb.46:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_mov_b64 s[0:1], 0
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_mov_b64_e32 v[10:11], 0
	s_mov_b64 s[30:31], 0
.LBB0_47:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s36, s28, s30
	s_addc_u32 s37, s29, s31
	global_load_ubyte v12, v29, s[36:37]
	s_add_u32 s30, s30, 1
	s_addc_u32 s31, s31, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v28, 0xffff, v12
	v_lshlrev_b64 v[12:13], s0, v[28:29]
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
	v_or_b32_e32 v10, v12, v10
	s_cmp_lg_u32 s34, s30
	v_or_b32_e32 v11, v13, v11
	s_cbranch_scc1 .LBB0_47
	s_branch .LBB0_50
.LBB0_48:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_mov_b32 s33, 0
	s_branch .LBB0_51
.LBB0_49:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[10:11], 0
.LBB0_50:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[0:1], s[28:29]
	s_mov_b32 s33, 0
	s_cbranch_execnz .LBB0_52
.LBB0_51:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	global_load_dwordx2 v[10:11], v29, s[28:29]
	s_add_i32 s33, s34, -8
.LBB0_52:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_add_u32 s28, s0, 8
	s_addc_u32 s29, s1, 0
	s_cmp_gt_u32 s33, 7
	s_cbranch_scc1 .LBB0_56
; %bb.53:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc1 .LBB0_57
; %bb.54:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_mov_b64 s[28:29], 0
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_mov_b64_e32 v[12:13], 0
	s_mov_b64 s[30:31], 0
.LBB0_55:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s34, s0, s30
	s_addc_u32 s35, s1, s31
	global_load_ubyte v14, v29, s[34:35]
	s_add_u32 s30, s30, 1
	s_addc_u32 s31, s31, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v28, 0xffff, v14
	v_lshlrev_b64 v[14:15], s28, v[28:29]
	s_add_u32 s28, s28, 8
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v12, v14, v12
	s_cmp_lg_u32 s33, s30
	v_or_b32_e32 v13, v15, v13
	s_cbranch_scc1 .LBB0_55
	s_branch .LBB0_58
.LBB0_56:                               ;   in Loop: Header=BB0_28 Depth=1
                                        ; implicit-def: $vgpr12_vgpr13
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_mov_b32 s34, 0
	s_branch .LBB0_59
.LBB0_57:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[12:13], 0
.LBB0_58:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[28:29], s[0:1]
	s_mov_b32 s34, 0
	s_cbranch_execnz .LBB0_60
.LBB0_59:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	global_load_dwordx2 v[12:13], v29, s[0:1]
	s_add_i32 s34, s33, -8
.LBB0_60:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_add_u32 s0, s28, 8
	s_addc_u32 s1, s29, 0
	s_cmp_gt_u32 s34, 7
	s_cbranch_scc1 .LBB0_64
; %bb.61:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s34, 0
	s_cbranch_scc1 .LBB0_65
; %bb.62:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_mov_b64 s[0:1], 0
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_mov_b64_e32 v[14:15], 0
	s_mov_b64 s[30:31], 0
.LBB0_63:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s36, s28, s30
	s_addc_u32 s37, s29, s31
	global_load_ubyte v16, v29, s[36:37]
	s_add_u32 s30, s30, 1
	s_addc_u32 s31, s31, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v28, 0xffff, v16
	v_lshlrev_b64 v[16:17], s0, v[28:29]
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
	v_or_b32_e32 v14, v16, v14
	s_cmp_lg_u32 s34, s30
	v_or_b32_e32 v15, v17, v15
	s_cbranch_scc1 .LBB0_63
	s_branch .LBB0_66
.LBB0_64:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_mov_b32 s33, 0
	s_branch .LBB0_67
.LBB0_65:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[14:15], 0
.LBB0_66:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[0:1], s[28:29]
	s_mov_b32 s33, 0
	s_cbranch_execnz .LBB0_68
.LBB0_67:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	global_load_dwordx2 v[14:15], v29, s[28:29]
	s_add_i32 s33, s34, -8
.LBB0_68:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_add_u32 s28, s0, 8
	s_addc_u32 s29, s1, 0
	s_cmp_gt_u32 s33, 7
	s_cbranch_scc1 .LBB0_72
; %bb.69:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc1 .LBB0_73
; %bb.70:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_mov_b64 s[28:29], 0
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_mov_b64_e32 v[16:17], 0
	s_mov_b64 s[30:31], 0
.LBB0_71:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s34, s0, s30
	s_addc_u32 s35, s1, s31
	global_load_ubyte v18, v29, s[34:35]
	s_add_u32 s30, s30, 1
	s_addc_u32 s31, s31, 0
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v28, 0xffff, v18
	v_lshlrev_b64 v[18:19], s28, v[28:29]
	s_add_u32 s28, s28, 8
	s_addc_u32 s29, s29, 0
	v_or_b32_e32 v16, v18, v16
	s_cmp_lg_u32 s33, s30
	v_or_b32_e32 v17, v19, v17
	s_cbranch_scc1 .LBB0_71
	s_branch .LBB0_74
.LBB0_72:                               ;   in Loop: Header=BB0_28 Depth=1
                                        ; implicit-def: $vgpr16_vgpr17
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_mov_b32 s34, 0
	s_branch .LBB0_75
.LBB0_73:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[16:17], 0
.LBB0_74:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[28:29], s[0:1]
	s_mov_b32 s34, 0
	s_cbranch_execnz .LBB0_76
.LBB0_75:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	global_load_dwordx2 v[16:17], v29, s[0:1]
	s_add_i32 s34, s33, -8
.LBB0_76:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_cmp_gt_u32 s34, 7
	s_cbranch_scc1 .LBB0_80
; %bb.77:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s34, 0
	s_cbranch_scc1 .LBB0_81
; %bb.78:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_mov_b64 s[0:1], 0
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_mov_b64_e32 v[18:19], 0
	s_mov_b64 s[30:31], s[28:29]
.LBB0_79:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_ubyte v20, v29, s[30:31]
	s_add_i32 s34, s34, -1
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v28, 0xffff, v20
	v_lshlrev_b64 v[20:21], s0, v[28:29]
	s_add_u32 s0, s0, 8
	s_addc_u32 s1, s1, 0
	s_add_u32 s30, s30, 1
	s_addc_u32 s31, s31, 0
	v_or_b32_e32 v18, v20, v18
	s_cmp_lg_u32 s34, 0
	v_or_b32_e32 v19, v21, v19
	s_cbranch_scc1 .LBB0_79
	s_branch .LBB0_82
.LBB0_80:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_branch .LBB0_83
.LBB0_81:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[18:19], 0
.LBB0_82:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cbranch_execnz .LBB0_84
.LBB0_83:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	global_load_dwordx2 v[18:19], v29, s[28:29]
.LBB0_84:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_readfirstlane_b32 s0, v1
	v_mov_b64_e32 v[30:31], 0
	s_nop 0
	v_cmp_eq_u32_e64 s[0:1], s0, v1
	s_and_saveexec_b64 s[28:29], s[0:1]
	s_cbranch_execz .LBB0_90
; %bb.85:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[22:23], v29, s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[20:21], v29, s[18:19] offset:40
	global_load_dwordx2 v[30:31], v29, s[18:19]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v20, v20, v22
	v_and_b32_e32 v21, v21, v23
	v_mul_lo_u32 v21, v21, 24
	v_mul_hi_u32 v27, v20, 24
	v_add_u32_e32 v21, v27, v21
	v_mul_lo_u32 v20, v20, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[20:21], v[30:31], 0, v[20:21]
	global_load_dwordx2 v[20:21], v[20:21], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[30:31], v29, v[20:23], s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[30:31], v[22:23]
	s_and_saveexec_b64 s[30:31], vcc
	s_cbranch_execz .LBB0_89
; %bb.86:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_mov_b64 s[34:35], 0
.LBB0_87:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	s_sleep 1
	global_load_dwordx2 v[20:21], v29, s[18:19] offset:40
	global_load_dwordx2 v[32:33], v29, s[18:19]
	v_mov_b64_e32 v[22:23], v[30:31]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v20, v20, v22
	v_and_b32_e32 v27, v21, v23
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[20:21], s[36:37], v20, 24, v[32:33]
	v_mov_b32_e32 v28, v21
	v_mad_u64_u32 v[30:31], s[36:37], v27, 24, v[28:29]
	v_mov_b32_e32 v21, v30
	global_load_dwordx2 v[20:21], v[20:21], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[30:31], v29, v[20:23], s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[30:31], v[22:23]
	s_or_b64 s[34:35], vcc, s[34:35]
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execnz .LBB0_87
; %bb.88:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[34:35]
.LBB0_89:                               ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[30:31]
.LBB0_90:                               ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[28:29]
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	global_load_dwordx2 v[32:33], v29, s[18:19] offset:40
	global_load_dwordx4 v[20:23], v29, s[18:19]
	v_readfirstlane_b32 s29, v31
	v_readfirstlane_b32 s28, v30
	s_mov_b64 s[30:31], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s34, v32
	v_readfirstlane_b32 s35, v33
	s_and_b64 s[34:35], s[34:35], s[28:29]
	s_mul_i32 s33, s35, 24
	s_mul_hi_u32 s36, s34, 24
	s_add_i32 s37, s36, s33
	s_mul_i32 s36, s34, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[30:31], v[20:21], 0, s[36:37]
	s_and_saveexec_b64 s[36:37], s[0:1]
	s_cbranch_execz .LBB0_92
; %bb.91:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[32:33], s[30:31]
	global_store_dwordx4 v[30:31], v[32:35], off offset:8
.LBB0_92:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_lshl_b64 s[30:31], s[34:35], 12
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[30:31]
	v_cmp_lt_u64_e64 vcc, s[24:25], 57
	s_lshl_b32 s30, s26, 2
	s_add_i32 s30, s30, 28
	v_cndmask_b32_e32 v27, 0, v26, vcc
	v_and_b32_e32 v4, 0xffffff1f, v4
	s_and_b32 s30, s30, 0x1e0
	v_or_b32_e32 v4, v4, v27
	v_or_b32_e32 v4, s30, v4
	v_readfirstlane_b32 s30, v22
	v_readfirstlane_b32 s31, v23
	s_nop 4
	global_store_dwordx4 v24, v[4:7], s[30:31]
	global_store_dwordx4 v24, v[8:11], s[30:31] offset:16
	global_store_dwordx4 v24, v[12:15], s[30:31] offset:32
	global_store_dwordx4 v24, v[16:19], s[30:31] offset:48
	s_and_saveexec_b64 s[30:31], s[0:1]
	s_cbranch_execz .LBB0_100
; %bb.93:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[12:13], v29, s[18:19] offset:32 sc0 sc1
	global_load_dwordx2 v[4:5], v29, s[18:19] offset:40
	v_mov_b32_e32 v10, s28
	v_mov_b32_e32 v11, s29
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s34, v4
	v_readfirstlane_b32 s35, v5
	s_and_b64 s[34:35], s[34:35], s[28:29]
	s_mul_i32 s33, s35, 24
	s_mul_hi_u32 s35, s34, 24
	s_mul_i32 s34, s34, 24
	s_add_i32 s35, s35, s33
	v_lshl_add_u64 v[8:9], v[20:21], 0, s[34:35]
	global_store_dwordx2 v[8:9], v[12:13], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v29, v[10:13], s[18:19] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[6:7], v[12:13]
	s_and_saveexec_b64 s[34:35], vcc
	s_cbranch_execz .LBB0_96
; %bb.94:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_mov_b64 s[36:37], 0
.LBB0_95:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	s_sleep 1
	global_store_dwordx2 v[8:9], v[6:7], off
	v_mov_b32_e32 v4, s28
	v_mov_b32_e32 v5, s29
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v29, v[4:7], s[18:19] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]
	s_or_b64 s[36:37], vcc, s[36:37]
	v_mov_b64_e32 v[6:7], v[4:5]
	s_andn2_b64 exec, exec, s[36:37]
	s_cbranch_execnz .LBB0_95
.LBB0_96:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	global_load_dwordx2 v[4:5], v29, s[18:19] offset:16
	s_mov_b64 s[36:37], exec
	v_mbcnt_lo_u32_b32 v6, s36, 0
	v_mbcnt_hi_u32_b32 v6, s37, v6
	v_cmp_eq_u32_e32 vcc, 0, v6
	s_and_saveexec_b64 s[34:35], vcc
	s_cbranch_execz .LBB0_98
; %bb.97:                               ;   in Loop: Header=BB0_28 Depth=1
	s_bcnt1_i32_b64 s33, s[36:37]
	v_mov_b32_e32 v28, s33
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[4:5], v[28:29], off offset:8 sc1
.LBB0_98:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[6:7], v[4:5], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[6:7]
	s_cbranch_vccnz .LBB0_100
; %bb.99:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dword v28, v[4:5], off offset:24
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s33, v28
	s_and_b32 m0, s33, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[6:7], v[28:29], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_100:                              ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[30:31]
	v_lshl_add_u64 v[4:5], v[22:23], 0, v[24:25]
	s_branch .LBB0_104
.LBB0_101:                              ;   in Loop: Header=BB0_104 Depth=2
	s_or_b64 exec, exec, s[30:31]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_readfirstlane_b32 s30, v6
	s_cmp_eq_u32 s30, 0
	s_cbranch_scc1 .LBB0_103
; %bb.102:                              ;   in Loop: Header=BB0_104 Depth=2
	s_sleep 1
	s_cbranch_execnz .LBB0_104
	s_branch .LBB0_106
.LBB0_103:                              ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_branch .LBB0_106
.LBB0_104:                              ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v6, 1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	s_and_saveexec_b64 s[30:31], s[0:1]
	s_cbranch_execz .LBB0_101
; %bb.105:                              ;   in Loop: Header=BB0_104 Depth=2
	global_load_dword v6, v[30:31], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v6, 1, v6
	s_branch .LBB0_101
.LBB0_106:                              ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	global_load_dwordx2 v[4:5], v[4:5], off
	s_and_saveexec_b64 s[30:31], s[0:1]
	s_cbranch_execz .LBB0_27
; %bb.107:                              ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[6:7], v29, s[18:19] offset:40
	global_load_dwordx2 v[14:15], v29, s[18:19] offset:24 sc0 sc1
	global_load_dwordx2 v[8:9], v29, s[18:19]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s34, v6
	v_readfirstlane_b32 s35, v7
	s_add_u32 s33, s34, 1
	s_addc_u32 s36, s35, 0
	s_add_u32 s0, s33, s28
	s_addc_u32 s1, s36, s29
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s36, s1
	s_cselect_b32 s0, s33, s0
	s_and_b64 s[28:29], s[0:1], s[34:35]
	s_mul_i32 s29, s29, 24
	s_mul_hi_u32 s33, s28, 24
	s_mul_i32 s28, s28, 24
	s_add_i32 s29, s33, s29
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[8:9], 0, s[28:29]
	v_mov_b32_e32 v12, s0
	global_store_dwordx2 v[10:11], v[14:15], off
	v_mov_b32_e32 v13, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v29, v[12:15], s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[8:9], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_27
; %bb.108:                              ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_mov_b64 s[28:29], 0
.LBB0_109:                              ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	s_sleep 1
	global_store_dwordx2 v[10:11], v[8:9], off
	v_mov_b32_e32 v6, s0
	v_mov_b32_e32 v7, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v29, v[6:9], s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]
	s_or_b64 s[28:29], vcc, s[28:29]
	v_mov_b64_e32 v[8:9], v[6:7]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB0_109
	s_branch .LBB0_27
.LBB0_110:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_branch .LBB0_138
.LBB0_111:
                                        ; implicit-def: $vgpr4_vgpr5
	s_cbranch_execz .LBB0_138
; %bb.112:
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	v_readfirstlane_b32 s0, v1
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[4:5], 0
	v_cmp_eq_u32_e64 s[0:1], s0, v1
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_118
; %bb.113:
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[10:11], v6, s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[4:5], v6, s[18:19] offset:40
	global_load_dwordx2 v[8:9], v6, s[18:19]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v4, v10
	v_and_b32_e32 v5, v5, v11
	v_mul_lo_u32 v5, v5, 24
	v_mul_hi_u32 v7, v4, 24
	v_add_u32_e32 v5, v7, v5
	v_mul_lo_u32 v4, v4, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[4:5]
	global_load_dwordx2 v[8:9], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v6, v[8:11], s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[4:5], v[10:11]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_117
; %bb.114:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_mov_b64 s[26:27], 0
.LBB0_115:                              ; =>This Inner Loop Header: Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	s_sleep 1
	global_load_dwordx2 v[8:9], v6, s[18:19] offset:40
	global_load_dwordx2 v[12:13], v6, s[18:19]
	v_mov_b64_e32 v[10:11], v[4:5]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v4, v8, v10
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[4:5], s[28:29], v4, 24, v[12:13]
	v_and_b32_e32 v7, v9, v11
	v_mov_b32_e32 v8, v5
	v_mad_u64_u32 v[8:9], s[28:29], v7, 24, v[8:9]
	v_mov_b32_e32 v5, v8
	global_load_dwordx2 v[8:9], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v6, v[8:11], s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[4:5], v[10:11]
	s_or_b64 s[26:27], vcc, s[26:27]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_115
; %bb.116:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[26:27]
.LBB0_117:
	s_or_b64 exec, exec, s[24:25]
.LBB0_118:
	s_or_b64 exec, exec, s[4:5]
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	v_mov_b32_e32 v25, 0
	global_load_dwordx2 v[10:11], v25, s[18:19] offset:40
	global_load_dwordx4 v[6:9], v25, s[18:19]
	v_readfirstlane_b32 s5, v5
	v_readfirstlane_b32 s4, v4
	s_mov_b64 s[24:25], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s26, v10
	v_readfirstlane_b32 s27, v11
	s_and_b64 s[26:27], s[26:27], s[4:5]
	s_mul_i32 s28, s27, 24
	s_mul_hi_u32 s29, s26, 24
	s_add_i32 s29, s29, s28
	s_mul_i32 s28, s26, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[10:11], v[6:7], 0, s[28:29]
	s_and_saveexec_b64 s[28:29], s[0:1]
	s_cbranch_execz .LBB0_120
; %bb.119:
	v_mov_b64_e32 v[12:13], s[24:25]
	v_mov_b32_e32 v14, 2
	v_mov_b32_e32 v15, 1
	global_store_dwordx4 v[10:11], v[12:15], off offset:8
.LBB0_120:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[28:29]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_lshl_b64 s[24:25], s[26:27], 12
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[24:25]
	s_movk_i32 s24, 0xff1f
	v_and_or_b32 v2, v2, s24, 32
	s_mov_b32 s24, 0
	v_mov_b32_e32 v4, v25
	v_mov_b32_e32 v5, v25
	v_readfirstlane_b32 s28, v8
	v_readfirstlane_b32 s29, v9
	s_mov_b32 s25, s24
	s_mov_b32 s26, s24
	s_mov_b32 s27, s24
	s_nop 1
	global_store_dwordx4 v24, v[2:5], s[28:29]
	s_nop 1
	v_mov_b64_e32 v[2:3], s[24:25]
	v_mov_b64_e32 v[4:5], s[26:27]
	global_store_dwordx4 v24, v[2:5], s[28:29] offset:16
	global_store_dwordx4 v24, v[2:5], s[28:29] offset:32
	global_store_dwordx4 v24, v[2:5], s[28:29] offset:48
	s_and_saveexec_b64 s[24:25], s[0:1]
	s_cbranch_execz .LBB0_128
; %bb.121:
	v_mov_b32_e32 v12, 0
	global_load_dwordx2 v[16:17], v12, s[18:19] offset:32 sc0 sc1
	global_load_dwordx2 v[2:3], v12, s[18:19] offset:40
	v_mov_b32_e32 v14, s4
	v_mov_b32_e32 v15, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s26, v2
	v_readfirstlane_b32 s27, v3
	s_and_b64 s[26:27], s[26:27], s[4:5]
	s_mul_i32 s27, s27, 24
	s_mul_hi_u32 s28, s26, 24
	s_mul_i32 s26, s26, 24
	s_add_i32 s27, s28, s27
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[26:27]
	global_store_dwordx2 v[6:7], v[16:17], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v12, v[14:17], s[18:19] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[16:17]
	s_and_saveexec_b64 s[26:27], vcc
	s_cbranch_execz .LBB0_124
; %bb.122:
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_mov_b64 s[28:29], 0
.LBB0_123:                              ; =>This Inner Loop Header: Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	s_sleep 1
	global_store_dwordx2 v[6:7], v[4:5], off
	v_mov_b32_e32 v2, s4
	v_mov_b32_e32 v3, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v12, v[2:5], s[18:19] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]
	s_or_b64 s[28:29], vcc, s[28:29]
	v_mov_b64_e32 v[4:5], v[2:3]
	s_andn2_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB0_123
.LBB0_124:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[26:27]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_mov_b32_e32 v5, 0
	global_load_dwordx2 v[2:3], v5, s[18:19] offset:16
	s_mov_b64 s[26:27], exec
	v_mbcnt_lo_u32_b32 v4, s26, 0
	v_mbcnt_hi_u32_b32 v4, s27, v4
	v_cmp_eq_u32_e32 vcc, 0, v4
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB0_126
; %bb.125:
	s_bcnt1_i32_b64 s26, s[26:27]
	v_mov_b32_e32 v4, s26
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[2:3], v[4:5], off offset:8 sc1
.LBB0_126:
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[28:29]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[4:5], v[2:3], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[4:5]
	s_cbranch_vccnz .LBB0_128
; %bb.127:
	global_load_dword v2, v[2:3], off offset:24
	v_mov_b32_e32 v3, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s26, v2
	s_and_b32 m0, s26, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[4:5], v[2:3], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_128:
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[24:25]
	v_lshl_add_u64 v[2:3], v[8:9], 0, v[24:25]
	s_branch .LBB0_132
.LBB0_129:                              ;   in Loop: Header=BB0_132 Depth=1
	s_or_b64 exec, exec, s[24:25]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_readfirstlane_b32 s24, v4
	s_cmp_eq_u32 s24, 0
	s_cbranch_scc1 .LBB0_131
; %bb.130:                              ;   in Loop: Header=BB0_132 Depth=1
	s_sleep 1
	s_cbranch_execnz .LBB0_132
	s_branch .LBB0_134
.LBB0_131:
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_branch .LBB0_134
.LBB0_132:                              ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v4, 1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	s_and_saveexec_b64 s[24:25], s[0:1]
	s_cbranch_execz .LBB0_129
; %bb.133:                              ;   in Loop: Header=BB0_132 Depth=1
	global_load_dword v4, v[10:11], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v4, 1, v4
	s_branch .LBB0_129
.LBB0_134:
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	global_load_dwordx2 v[4:5], v[2:3], off
	s_and_saveexec_b64 s[24:25], s[0:1]
	s_cbranch_execz .LBB0_137
; %bb.135:
	v_mov_b32_e32 v10, 0
	global_load_dwordx2 v[2:3], v10, s[18:19] offset:40
	global_load_dwordx2 v[14:15], v10, s[18:19] offset:24 sc0 sc1
	global_load_dwordx2 v[6:7], v10, s[18:19]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s26, v2
	v_readfirstlane_b32 s27, v3
	s_add_u32 s28, s26, 1
	s_addc_u32 s29, s27, 0
	s_add_u32 s0, s28, s4
	s_addc_u32 s1, s29, s5
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s29, s1
	s_cselect_b32 s0, s28, s0
	s_and_b64 s[4:5], s[0:1], s[26:27]
	s_mul_i32 s5, s5, 24
	s_mul_hi_u32 s26, s4, 24
	s_mul_i32 s4, s4, 24
	s_add_i32 s5, s26, s5
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, s[4:5]
	v_mov_b32_e32 v12, s0
	global_store_dwordx2 v[2:3], v[14:15], off
	v_mov_b32_e32 v13, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[8:9], v10, v[12:15], s[18:19] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[8:9], v[14:15]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_137
.LBB0_136:                              ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[2:3], v[8:9], off
	v_mov_b32_e32 v6, s0
	v_mov_b32_e32 v7, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[6:7], v10, v[6:9], s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[8:9], v[6:7]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_136
.LBB0_137:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[24:25]
.LBB0_138:
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	v_readfirstlane_b32 s0, v1
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[2:3], 0
	v_cmp_eq_u32_e64 s[0:1], s0, v1
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_144
; %bb.139:
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[8:9], v1, s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	global_load_dwordx2 v[2:3], v1, s[18:19] offset:40
	global_load_dwordx2 v[6:7], v1, s[18:19]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v2, v8
	v_and_b32_e32 v3, v3, v9
	v_mul_lo_u32 v3, v3, 24
	v_mul_hi_u32 v10, v2, 24
	v_add_u32_e32 v3, v10, v3
	v_mul_lo_u32 v2, v2, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]
	global_load_dwordx2 v[6:7], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v1, v[6:9], s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e32 vcc, v[2:3], v[8:9]
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_143
; %bb.140:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_mov_b64 s[26:27], 0
.LBB0_141:                              ; =>This Inner Loop Header: Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	s_sleep 1
	global_load_dwordx2 v[6:7], v1, s[18:19] offset:40
	global_load_dwordx2 v[10:11], v1, s[18:19]
	v_mov_b64_e32 v[8:9], v[2:3]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v2, v6, v8
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[2:3], s[28:29], v2, 24, v[10:11]
	v_and_b32_e32 v7, v7, v9
	v_mov_b32_e32 v6, v3
	v_mad_u64_u32 v[6:7], s[28:29], v7, 24, v[6:7]
	v_mov_b32_e32 v3, v6
	global_load_dwordx2 v[6:7], v[2:3], off sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v1, v[6:9], s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e32 vcc, v[2:3], v[8:9]
	s_or_b64 s[26:27], vcc, s[26:27]
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB0_141
; %bb.142:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[26:27]
.LBB0_143:
	s_or_b64 exec, exec, s[24:25]
.LBB0_144:
	s_or_b64 exec, exec, s[4:5]
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[6:7], v1, s[18:19] offset:40
	global_load_dwordx4 v[8:11], v1, s[18:19]
	v_readfirstlane_b32 s5, v3
	v_readfirstlane_b32 s4, v2
	s_mov_b64 s[24:25], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s26, v6
	v_readfirstlane_b32 s27, v7
	s_and_b64 s[26:27], s[26:27], s[4:5]
	s_mul_i32 s28, s27, 24
	s_mul_hi_u32 s29, s26, 24
	s_add_i32 s29, s29, s28
	s_mul_i32 s28, s26, 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[12:13], v[8:9], 0, s[28:29]
	s_and_saveexec_b64 s[28:29], s[0:1]
	s_cbranch_execz .LBB0_146
; %bb.145:
	v_mov_b64_e32 v[14:15], s[24:25]
	v_mov_b32_e32 v16, 2
	v_mov_b32_e32 v17, 1
	global_store_dwordx4 v[12:13], v[14:17], off offset:8
.LBB0_146:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[28:29]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_lshl_b64 s[24:25], s[26:27], 12
	.loc	2 27 37                         ; ./v0_naive.h:27:37
	s_sub_u32 s20, s22, s20
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_lshl_add_u64 v[2:3], v[10:11], 0, s[24:25]
	s_movk_i32 s24, 0xff1d
	.loc	2 27 37                         ; ./v0_naive.h:27:37
	s_subb_u32 s21, s23, s21
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_mov_b32_e32 v6, s20
	s_mov_b32 s20, 0
	v_and_or_b32 v4, v4, s24, 34
	v_mov_b32_e32 v7, s21
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	s_mov_b32 s21, s20
	s_mov_b32 s22, s20
	s_mov_b32 s23, s20
	s_nop 1
	global_store_dwordx4 v24, v[4:7], s[24:25]
	v_mov_b64_e32 v[2:3], s[20:21]
	s_nop 0
	v_mov_b64_e32 v[4:5], s[22:23]
	global_store_dwordx4 v24, v[2:5], s[24:25] offset:16
	global_store_dwordx4 v24, v[2:5], s[24:25] offset:32
	global_store_dwordx4 v24, v[2:5], s[24:25] offset:48
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_154
; %bb.147:
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[16:17], v1, s[18:19] offset:32 sc0 sc1
	global_load_dwordx2 v[2:3], v1, s[18:19] offset:40
	v_mov_b32_e32 v14, s4
	v_mov_b32_e32 v15, s5
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v2
	v_readfirstlane_b32 s23, v3
	s_and_b64 s[22:23], s[22:23], s[4:5]
	s_mul_i32 s23, s23, 24
	s_mul_hi_u32 s24, s22, 24
	s_mul_i32 s22, s22, 24
	s_add_i32 s23, s24, s23
	v_lshl_add_u64 v[6:7], v[8:9], 0, s[22:23]
	global_store_dwordx2 v[6:7], v[16:17], off
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v1, v[14:17], s[18:19] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[16:17]
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB0_150
; %bb.148:
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_mov_b64 s[24:25], 0
.LBB0_149:                              ; =>This Inner Loop Header: Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	s_sleep 1
	global_store_dwordx2 v[6:7], v[4:5], off
	v_mov_b32_e32 v2, s4
	v_mov_b32_e32 v3, s5
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v1, v[2:5], s[18:19] offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]
	s_or_b64 s[24:25], vcc, s[24:25]
	v_mov_b64_e32 v[4:5], v[2:3]
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execnz .LBB0_149
.LBB0_150:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[22:23]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_mov_b32_e32 v5, 0
	global_load_dwordx2 v[2:3], v5, s[18:19] offset:16
	s_mov_b64 s[22:23], exec
	v_mbcnt_lo_u32_b32 v1, s22, 0
	v_mbcnt_hi_u32_b32 v1, s23, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB0_152
; %bb.151:
	s_bcnt1_i32_b64 s22, s[22:23]
	v_mov_b32_e32 v4, s22
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[2:3], v[4:5], off offset:8 sc1
.LBB0_152:
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[24:25]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[4:5], v[2:3], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[4:5]
	s_cbranch_vccnz .LBB0_154
; %bb.153:
	global_load_dword v2, v[2:3], off offset:24
	v_mov_b32_e32 v3, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s22, v2
	s_and_b32 m0, s22, 0xffffff
	buffer_wbl2 sc0 sc1
	global_store_dwordx2 v[4:5], v[2:3], off sc0 sc1
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB0_154:
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[20:21]
	s_branch .LBB0_158
.LBB0_155:                              ;   in Loop: Header=BB0_158 Depth=1
	s_or_b64 exec, exec, s[20:21]
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	v_readfirstlane_b32 s20, v1
	s_cmp_eq_u32 s20, 0
	s_cbranch_scc1 .LBB0_157
; %bb.156:                              ;   in Loop: Header=BB0_158 Depth=1
	s_sleep 1
	s_cbranch_execnz .LBB0_158
	s_branch .LBB0_160
.LBB0_157:
	.loc	2 0 5                           ; ./v0_naive.h:0:5
	s_branch .LBB0_160
.LBB0_158:                              ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v1, 1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_155
; %bb.159:                              ;   in Loop: Header=BB0_158 Depth=1
	global_load_dword v1, v[12:13], off offset:20 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_and_b32_e32 v1, 1, v1
	s_branch .LBB0_155
.LBB0_160:
	.loc	2 27 5                          ; ./v0_naive.h:27:5
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB0_163
; %bb.161:
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[2:3], v1, s[18:19] offset:40
	global_load_dwordx2 v[10:11], v1, s[18:19] offset:24 sc0 sc1
	global_load_dwordx2 v[4:5], v1, s[18:19]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s22, v2
	v_readfirstlane_b32 s23, v3
	s_add_u32 s24, s22, 1
	s_addc_u32 s25, s23, 0
	s_add_u32 s0, s24, s4
	s_addc_u32 s1, s25, s5
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s1, s25, s1
	s_cselect_b32 s0, s24, s0
	s_and_b64 s[4:5], s[0:1], s[22:23]
	s_mul_i32 s5, s5, 24
	s_mul_hi_u32 s22, s4, 24
	s_mul_i32 s4, s4, 24
	s_add_i32 s5, s22, s5
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[4:5], 0, s[4:5]
	v_mov_b32_e32 v8, s0
	global_store_dwordx2 v[6:7], v[10:11], off
	v_mov_b32_e32 v9, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v1, v[8:11], s[18:19] offset:24 sc0 sc1
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[10:11]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_163
.LBB0_162:                              ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[6:7], v[4:5], off
	v_mov_b32_e32 v2, s0
	v_mov_b32_e32 v3, s1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v1, v[2:5], s[18:19] offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]
	s_or_b64 s[4:5], vcc, s[4:5]
	v_mov_b64_e32 v[4:5], v[2:3]
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_162
.LBB0_163:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	2 22 25 is_stmt 1               ; ./v0_naive.h:22:25
	v_lshrrev_b32_e32 v3, 7, v0
	.loc	2 29 25                         ; ./v0_naive.h:29:25
	s_waitcnt lgkmcnt(0)
	s_cmp_gt_i32 s7, 0
	.loc	2 22 40                         ; ./v0_naive.h:22:40
	v_bfe_u32 v6, v0, 6, 1
	.loc	2 29 5                          ; ./v0_naive.h:29:5
	s_cbranch_scc1 .LBB0_165
; %bb.164:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	v_lshlrev_b32_e32 v29, 6, v3
	v_lshlrev_b32_e32 v30, 6, v6
	s_mov_b64 s[0:1], 0
	s_branch .LBB0_166
.LBB0_165:
	s_mov_b64 s[0:1], -1
                                        ; implicit-def: $vgpr29
                                        ; implicit-def: $vgpr30
.LBB0_166:
	v_and_b32_e32 v28, 15, v0
	v_bfe_u32 v2, v0, 4, 2
	s_lshl_b32 s5, s3, 7
	s_lshl_b32 s4, s2, 7
	v_mov_b32_e32 v1, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a63, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	s_andn2_b64 vcc, exec, s[0:1]
	v_accvgpr_write_b32 a3, 0
	s_cbranch_vccnz .LBB0_171
; %bb.167:
	.loc	2 25 24 is_stmt 1               ; ./v0_naive.h:25:24
	s_lshr_b32 s0, s7, 31
	s_add_i32 s0, s7, s0
	s_ashr_i32 s18, s0, 1
	s_ashr_i32 s0, s7, 31
	.loc	2 25 38 is_stmt 0               ; ./v0_naive.h:25:38
	s_lshr_b32 s0, s0, 27
	v_lshlrev_b32_e32 v29, 6, v3
	s_add_i32 s0, s7, s0
	v_or_b32_e32 v24, v29, v28
	v_mov_b32_e32 v3, v1
	s_ashr_i32 s0, s0, 5
	.loc	2 29 5 is_stmt 1                ; ./v0_naive.h:29:5
	v_lshl_add_u64 v[4:5], s[12:13], 0, v[2:3]
	v_lshlrev_b32_e32 v30, 6, v6
	v_lshl_add_u64 v[6:7], s[14:15], 0, v[2:3]
	v_add_u32_e32 v3, s5, v24
	v_mad_u64_u32 v[8:9], s[2:3], v3, s0, 0
	v_ashrrev_i32_e32 v11, 31, v3
	v_mov_b32_e32 v10, v9
	v_or_b32_e32 v16, v30, v28
	v_mad_u64_u32 v[10:11], s[2:3], v11, s0, v[10:11]
	v_or_b32_e32 v3, s4, v16
	s_ashr_i32 s2, s4, 31
	v_mov_b32_e32 v9, v10
	s_mul_i32 s12, s2, s0
	v_mad_u64_u32 v[10:11], s[2:3], v3, s0, 0
	v_or_b32_e32 v3, 16, v16
	v_lshlrev_b32_e32 v36, 6, v3
	v_or_b32_e32 v3, s4, v3
	v_mad_u64_u32 v[12:13], s[2:3], v3, s0, 0
	v_or_b32_e32 v3, 32, v16
	v_lshlrev_b32_e32 v37, 6, v3
	v_or_b32_e32 v3, s4, v3
	v_mad_u64_u32 v[14:15], s[2:3], v3, s0, 0
	v_or_b32_e32 v3, 48, v16
	v_lshlrev_b32_e32 v39, 6, v3
	v_or_b32_e32 v3, s4, v3
	v_lshlrev_b32_e32 v35, 6, v16
	v_mad_u64_u32 v[16:17], s[2:3], v3, s0, 0
	v_or_b32_e32 v3, 16, v24
	v_lshlrev_b32_e32 v40, 6, v3
	v_add_u32_e32 v3, s5, v3
	v_mad_u64_u32 v[18:19], s[2:3], v3, s0, 0
	v_ashrrev_i32_e32 v21, 31, v3
	v_mov_b32_e32 v20, v19
	v_or_b32_e32 v3, 32, v24
	v_mad_u64_u32 v[20:21], s[2:3], v21, s0, v[20:21]
	v_lshlrev_b32_e32 v41, 6, v3
	v_add_u32_e32 v3, s5, v3
	v_mov_b32_e32 v19, v20
	v_mad_u64_u32 v[20:21], s[2:3], v3, s0, 0
	v_ashrrev_i32_e32 v23, 31, v3
	v_mov_b32_e32 v22, v21
	v_or_b32_e32 v3, 48, v24
	v_mad_u64_u32 v[22:23], s[2:3], v23, s0, v[22:23]
	v_lshlrev_b32_e32 v42, 6, v3
	v_add_u32_e32 v3, s5, v3
	v_mov_b32_e32 v21, v22
	v_mad_u64_u32 v[22:23], s[2:3], v3, s0, 0
	v_lshlrev_b32_e32 v27, 4, v2
	v_lshlrev_b32_e32 v34, 6, v24
	v_ashrrev_i32_e32 v25, 31, v3
	v_mov_b32_e32 v24, v23
	v_lshrrev_b32_e32 v26, 4, v0
	v_or_b32_e32 v38, 0x2000, v27
	v_mad_u64_u32 v[24:25], s[2:3], v25, s0, v[24:25]
	s_mov_b32 s1, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a63, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a60, 0
	v_add_u32_e32 v11, s12, v11
	v_add_u32_e32 v13, s12, v13
	v_add_u32_e32 v15, s12, v15
	v_add_u32_e32 v17, s12, v17
	v_mov_b32_e32 v23, v24
	v_or_b32_e32 v3, 0xffffff00, v0
	v_add_u32_e32 v31, s4, v26
	v_add_u32_e32 v32, s5, v26
	v_lshlrev_b32_e32 v33, 2, v0
	s_movk_i32 s12, 0x6ff
	v_add_u32_e32 v34, v27, v34
	v_add_u32_e32 v35, v38, v35
	v_add_u32_e32 v36, v38, v36
	v_add_u32_e32 v37, v38, v37
	v_add_u32_e32 v38, v38, v39
	v_add_u32_e32 v39, v27, v40
	v_add_u32_e32 v40, v27, v41
	v_add_u32_e32 v41, v27, v42
	s_mov_b32 s13, 0
.LBB0_168:                              ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_169 Depth 2
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	s_lshr_b32 s0, s13, 1
	.loc	2 33 9 is_stmt 1                ; ./v0_naive.h:33:9
	s_add_u32 s14, s8, s0
	s_addc_u32 s15, s9, 0
	s_add_u32 s20, s10, s0
	s_addc_u32 s21, s11, 0
	s_mov_b64 s[2:3], 0
	v_mov_b64_e32 v[24:25], s[14:15]
	v_mov_b64_e32 v[26:27], s[20:21]
	v_mov_b32_e32 v42, v33
	v_mov_b32_e32 v43, v32
	v_mov_b32_e32 v44, v31
	v_mov_b32_e32 v45, v3
.LBB0_169:                              ;   Parent Loop BB0_168 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	2 37 17                         ; ./v0_naive.h:37:17
	v_mad_u64_u32 v[46:47], s[14:15], v43, s18, v[24:25]
	.loc	2 37 66 is_stmt 0               ; ./v0_naive.h:37:66
	v_ashrrev_i32_e32 v51, 31, v43
	.loc	2 39 17 is_stmt 1               ; ./v0_naive.h:39:17
	v_mad_u64_u32 v[48:49], s[14:15], v44, s18, v[26:27]
	.loc	2 37 17                         ; ./v0_naive.h:37:17
	v_mov_b32_e32 v50, v47
	.loc	2 39 66                         ; ./v0_naive.h:39:66
	v_ashrrev_i32_e32 v53, 31, v44
	.loc	2 39 17 is_stmt 0               ; ./v0_naive.h:39:17
	v_mov_b32_e32 v52, v49
	.loc	2 37 17 is_stmt 1               ; ./v0_naive.h:37:17
	v_mad_u64_u32 v[50:51], s[14:15], v51, s18, v[50:51]
	.loc	2 37 93 is_stmt 0               ; ./v0_naive.h:37:93
	v_and_b32_e32 v0, 60, v42
	.loc	2 39 17 is_stmt 1               ; ./v0_naive.h:39:17
	v_mad_u64_u32 v[52:53], s[14:15], v53, s18, v[52:53]
	.loc	2 37 17                         ; ./v0_naive.h:37:17
	v_mov_b32_e32 v47, v50
	.loc	2 39 17                         ; ./v0_naive.h:39:17
	v_mov_b32_e32 v49, v52
	.loc	2 37 17                         ; ./v0_naive.h:37:17
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[0:1]
	.loc	2 39 17                         ; ./v0_naive.h:39:17
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[0:1]
	.loc	2 37 17                         ; ./v0_naive.h:37:17
	global_load_dword v0, v[46:47], off
	.loc	2 39 17                         ; ./v0_naive.h:39:17
	global_load_dword v50, v[48:49], off
	.loc	2 33 29                         ; ./v0_naive.h:33:29
	v_add_u32_e32 v45, 0x100, v45
	v_cmp_lt_u32_e32 vcc, s12, v45
	v_add_u32_e32 v44, 16, v44
	v_add_u32_e32 v43, 16, v43
	.loc	2 33 9 is_stmt 0                ; ./v0_naive.h:33:9
	s_or_b64 s[2:3], vcc, s[2:3]
	.loc	2 36 48 is_stmt 1               ; ./v0_naive.h:36:48
	s_waitcnt vmcnt(0)
	ds_write2st64_b32 v42, v0, v50 offset1:32
	.loc	2 33 29                         ; ./v0_naive.h:33:29
	v_add_u32_e32 v42, 0x400, v42
	.loc	2 33 9 is_stmt 0                ; ./v0_naive.h:33:9
	s_andn2_b64 exec, exec, s[2:3]
	s_cbranch_execnz .LBB0_169
; %bb.170:                              ;   in Loop: Header=BB0_168 Depth=1
	.loc	2 0 9                           ; ./v0_naive.h:0:9
	s_or_b64 exec, exec, s[2:3]
	s_lshr_b32 s0, s13, 5
	.loc	2 46 14 is_stmt 1               ; ./v0_naive.h:46:14
	v_lshl_add_u64 v[58:59], v[4:5], 0, s[0:1]
	v_lshl_add_u64 v[24:25], v[6:7], 0, s[0:1]
	.loc	2 50 28                         ; ./v0_naive.h:50:28
	v_lshl_add_u64 v[26:27], v[58:59], 0, v[8:9]
.Ltmp2:
	.loc	1 691 5                         ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:691:5 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:50 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./v0_naive.h:41:9 ] ] ]
	s_waitcnt lgkmcnt(0)
	.loc	1 692 5                         ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:692:5 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:50 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./v0_naive.h:41:9 ] ] ]
	s_barrier
.Ltmp3:
	.loc	2 50 28                         ; ./v0_naive.h:50:28
	global_load_ubyte v0, v[26:27], off
	.loc	2 55 32                         ; ./v0_naive.h:55:32
	v_lshl_add_u64 v[26:27], v[24:25], 0, v[10:11]
	global_load_ubyte v62, v[26:27], off
	v_lshl_add_u64 v[26:27], v[24:25], 0, v[12:13]
	global_load_ubyte v63, v[26:27], off
	v_lshl_add_u64 v[26:27], v[24:25], 0, v[14:15]
	v_lshl_add_u64 v[24:25], v[24:25], 0, v[16:17]
	global_load_ubyte v64, v[26:27], off
	global_load_ubyte v65, v[24:25], off
	.loc	2 50 28                         ; ./v0_naive.h:50:28
	v_lshl_add_u64 v[24:25], v[58:59], 0, v[18:19]
	global_load_ubyte v66, v[24:25], off
	.loc	2 49 29                         ; ./v0_naive.h:49:29
	ds_read_b128 v[24:27], v34
	.loc	2 54 33                         ; ./v0_naive.h:54:33
	ds_read_b128 v[42:45], v35
	ds_read_b128 v[46:49], v36
	ds_read_b128 v[50:53], v37
	ds_read_b128 v[54:57], v38
	.loc	2 50 28                         ; ./v0_naive.h:50:28
	v_lshl_add_u64 v[60:61], v[58:59], 0, v[20:21]
	v_lshl_add_u64 v[58:59], v[58:59], 0, v[22:23]
	.loc	2 29 35                         ; ./v0_naive.h:29:35
	s_addk_i32 s13, 0x80
	.loc	2 29 25 is_stmt 0               ; ./v0_naive.h:29:25
	s_cmp_ge_i32 s13, s7
.Ltmp4:
	.file	3 "." "common.h" md5 0x1b7369279800554c32ed591e841c82de
	.loc	3 34 12 is_stmt 1               ; ./common.h:34:12 @[ ./v0_naive.h:56:31 ]
	s_waitcnt vmcnt(4) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[42:45], a[0:3], v0, v62 op_sel_hi:[0,0,0] cbsz:4 blgp:4
.Ltmp5:
	.loc	2 50 28                         ; ./v0_naive.h:50:28
	global_load_ubyte v58, v[58:59], off
.Ltmp6:
	.loc	3 34 12                         ; ./common.h:34:12 @[ ./v0_naive.h:56:31 ]
	s_waitcnt vmcnt(4) lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:27], v[46:49], a[4:7], v0, v63 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(3) lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[24:27], v[50:53], a[8:11], v0, v64 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(2) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:27], v[54:57], a[12:15], v0, v65 op_sel_hi:[0,0,0] cbsz:4 blgp:4
.Ltmp7:
	.loc	2 50 28                         ; ./v0_naive.h:50:28
	global_load_ubyte v0, v[60:61], off
	.loc	2 49 29                         ; ./v0_naive.h:49:29
	ds_read_b128 v[24:27], v39
.Ltmp8:
	.loc	3 34 12                         ; ./common.h:34:12 @[ ./v0_naive.h:56:31 ]
	s_waitcnt vmcnt(2) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[24:27], v[42:45], a[16:19], v66, v62 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[24:27], v[46:49], a[20:23], v66, v63 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[24:27], v[50:53], a[24:27], v66, v64 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[24:27], v[54:57], a[28:31], v66, v65 op_sel_hi:[0,0,0] cbsz:4 blgp:4
.Ltmp9:
	.loc	2 49 29                         ; ./v0_naive.h:49:29
	ds_read_b128 v[24:27], v40
.Ltmp10:
	.loc	3 34 12                         ; ./common.h:34:12 @[ ./v0_naive.h:56:31 ]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[24:27], v[42:45], a[32:35], v0, v62 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[24:27], v[46:49], a[36:39], v0, v63 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[24:27], v[50:53], a[40:43], v0, v64 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[24:27], v[54:57], a[44:47], v0, v65 op_sel_hi:[0,0,0] cbsz:4 blgp:4
.Ltmp11:
	.loc	2 49 29                         ; ./v0_naive.h:49:29
	ds_read_b128 v[24:27], v41
.Ltmp12:
	.loc	1 691 5                         ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:691:5 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:50 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./v0_naive.h:59:9 ] ] ]
	s_waitcnt lgkmcnt(0)
	.loc	1 692 5                         ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:692:5 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:50 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./v0_naive.h:59:9 ] ] ]
	s_barrier
.Ltmp13:
	.loc	3 34 12                         ; ./common.h:34:12 @[ ./v0_naive.h:56:31 ]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:27], v[42:45], a[48:51], v58, v62 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[24:27], v[46:49], a[52:55], v58, v63 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[24:27], v[50:53], a[56:59], v58, v64 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[24:27], v[54:57], a[60:63], v58, v65 op_sel_hi:[0,0,0] cbsz:4 blgp:4
.Ltmp14:
	.loc	2 29 5                          ; ./v0_naive.h:29:5
	s_cbranch_scc0 .LBB0_168
.LBB0_171:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5
	v_add_u32_e32 v29, s5, v29
	v_or3_b32 v28, v30, s4, v28
	v_accvgpr_read_b32 v67, a0
	v_mov_b32_e32 v68, s16
	v_mov_b32_e32 v69, s17
	.loc	2 63 5 is_stmt 1                ; ./v0_naive.h:63:5
	v_lshl_or_b32 v2, v2, 2, v29
	v_ashrrev_i32_e32 v29, 31, v28
	.loc	2 67 13                         ; ./v0_naive.h:67:13
	v_lshl_add_u64 v[28:29], v[28:29], 1, v[68:69]
.Ltmp15:
	.file	4 "/opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail" "amd_hip_bf16.h" md5 0x110c5195dca90dcd6b1e383f5f714869
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v67, s0
.Ltmp16:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[68:69], s[0:1], v2, s6, 0
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[68:69], v[68:69], 1, v[28:29]
	v_accvgpr_read_b32 v66, a1
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[68:69], v30, off
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62
	v_or_b32_e32 v30, 1, v2
.Ltmp17:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v70, v66, s0
.Ltmp18:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[66:67], s[0:1], v30, s6, 0
	v_accvgpr_read_b32 v65, a2
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[28:29]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62
	v_or_b32_e32 v30, 2, v2
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[66:67], v70, off
.Ltmp19:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v65, v65, s0
.Ltmp20:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[70:71], s[0:1], v30, s6, 0
	v_accvgpr_read_b32 v64, a3
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[70:71], v[70:71], 1, v[28:29]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62
	v_or_b32_e32 v30, 3, v2
	v_accvgpr_read_b32 v63, a4
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[70:71], v65, off
.Ltmp21:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v72, v64, s0
.Ltmp22:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[64:65], s[0:1], v30, s6, 0
	v_accvgpr_read_b32 v62, a5
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[64:65], v[64:65], 1, v[28:29]
.Ltmp23:
	.loc	4 214 87 is_stmt 1              ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v63, s0
	v_accvgpr_read_b32 v61, a6
.Ltmp24:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[64:65], v72, off
	global_store_short v[68:69], v30, off offset:32
.Ltmp25:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v62, s0
	v_accvgpr_read_b32 v60, a7
.Ltmp26:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[66:67], v30, off offset:32
.Ltmp27:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v61, s0
	v_accvgpr_read_b32 v59, a8
.Ltmp28:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[70:71], v30, off offset:32
.Ltmp29:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v60, s0
	v_accvgpr_read_b32 v58, a9
.Ltmp30:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[64:65], v30, off offset:32
.Ltmp31:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v59, s0
	v_accvgpr_read_b32 v57, a10
.Ltmp32:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[68:69], v30, off offset:64
.Ltmp33:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v58, s0
	v_accvgpr_read_b32 v56, a11
.Ltmp34:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[66:67], v30, off offset:64
.Ltmp35:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v57, s0
	v_accvgpr_read_b32 v55, a12
.Ltmp36:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[70:71], v30, off offset:64
.Ltmp37:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v56, s0
	v_accvgpr_read_b32 v54, a13
.Ltmp38:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[64:65], v30, off offset:64
.Ltmp39:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v55, s0
	v_accvgpr_read_b32 v53, a14
.Ltmp40:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[68:69], v30, off offset:96
.Ltmp41:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v54, s0
	v_accvgpr_read_b32 v52, a15
.Ltmp42:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[66:67], v30, off offset:96
.Ltmp43:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v53, s0
.Ltmp44:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[70:71], v30, off offset:96
.Ltmp45:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v52, s0
	v_accvgpr_read_b32 v51, a16
.Ltmp46:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[64:65], v30, off offset:96
	v_or_b32_e32 v30, 16, v2
.Ltmp47:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v51, v51, s0
.Ltmp48:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[52:53], s[0:1], v30, s6, 0
	v_accvgpr_read_b32 v50, a17
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[52:53], v[52:53], 1, v[28:29]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62
	v_or_b32_e32 v30, 17, v2
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[52:53], v51, off
.Ltmp49:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v54, v50, s0
.Ltmp50:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[50:51], s[0:1], v30, s6, 0
	v_accvgpr_read_b32 v49, a18
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[28:29]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62
	v_or_b32_e32 v30, 18, v2
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[50:51], v54, off
.Ltmp51:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v49, v49, s0
.Ltmp52:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[54:55], s[0:1], v30, s6, 0
	v_accvgpr_read_b32 v48, a19
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[54:55], v[54:55], 1, v[28:29]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62
	v_or_b32_e32 v30, 19, v2
	v_accvgpr_read_b32 v47, a20
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[54:55], v49, off
.Ltmp53:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v56, v48, s0
.Ltmp54:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[48:49], s[0:1], v30, s6, 0
	v_accvgpr_read_b32 v46, a21
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[48:49], v[48:49], 1, v[28:29]
.Ltmp55:
	.loc	4 214 87 is_stmt 1              ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v47, s0
	v_accvgpr_read_b32 v45, a22
.Ltmp56:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[48:49], v56, off
	global_store_short v[52:53], v30, off offset:32
.Ltmp57:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v46, s0
	v_accvgpr_read_b32 v44, a23
.Ltmp58:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[50:51], v30, off offset:32
.Ltmp59:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v45, s0
	v_accvgpr_read_b32 v43, a24
.Ltmp60:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[54:55], v30, off offset:32
.Ltmp61:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v44, s0
	v_accvgpr_read_b32 v42, a25
.Ltmp62:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[48:49], v30, off offset:32
.Ltmp63:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v43, s0
	v_accvgpr_read_b32 v41, a26
.Ltmp64:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[52:53], v30, off offset:64
.Ltmp65:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v42, s0
	v_accvgpr_read_b32 v40, a27
.Ltmp66:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[50:51], v30, off offset:64
.Ltmp67:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v41, s0
	v_accvgpr_read_b32 v39, a28
.Ltmp68:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[54:55], v30, off offset:64
.Ltmp69:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v40, s0
	v_accvgpr_read_b32 v38, a29
.Ltmp70:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[48:49], v30, off offset:64
.Ltmp71:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v39, s0
	v_accvgpr_read_b32 v37, a30
.Ltmp72:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[52:53], v30, off offset:96
.Ltmp73:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v38, s0
	v_accvgpr_read_b32 v36, a31
.Ltmp74:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[50:51], v30, off offset:96
.Ltmp75:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v37, s0
.Ltmp76:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[54:55], v30, off offset:96
.Ltmp77:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v30, v36, s0
	v_accvgpr_read_b32 v35, a32
.Ltmp78:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[48:49], v30, off offset:96
	v_or_b32_e32 v30, 32, v2
.Ltmp79:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v35, v35, s0
.Ltmp80:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[36:37], s[0:1], v30, s6, 0
	v_accvgpr_read_b32 v34, a33
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[36:37], v[36:37], 1, v[28:29]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62
	v_or_b32_e32 v30, 33, v2
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[36:37], v35, off
.Ltmp81:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v38, v34, s0
.Ltmp82:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[34:35], s[0:1], v30, s6, 0
	v_accvgpr_read_b32 v33, a34
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[28:29]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62
	v_or_b32_e32 v30, 34, v2
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[34:35], v38, off
.Ltmp83:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v33, v33, s0
.Ltmp84:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[38:39], s[0:1], v30, s6, 0
	v_accvgpr_read_b32 v32, a35
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[38:39], v[38:39], 1, v[28:29]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62
	v_or_b32_e32 v30, 35, v2
	v_accvgpr_read_b32 v20, a44
	v_accvgpr_read_b32 v19, a45
	v_accvgpr_read_b32 v18, a46
	v_accvgpr_read_b32 v17, a47
	v_accvgpr_read_b32 v24, a40
	v_accvgpr_read_b32 v23, a41
	v_accvgpr_read_b32 v22, a42
	v_accvgpr_read_b32 v21, a43
	v_accvgpr_read_b32 v31, a36
	v_accvgpr_read_b32 v27, a37
	v_accvgpr_read_b32 v26, a38
	v_accvgpr_read_b32 v25, a39
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[38:39], v33, off
.Ltmp85:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v40, v32, s0
.Ltmp86:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[32:33], s[0:1], v30, s6, 0
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[32:33], v[32:33], 1, v[28:29]
.Ltmp87:
	.loc	4 214 87 is_stmt 1              ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	s_nop 0
	v_cvt_pk_bf16_f32 v30, v31, s0
	v_cvt_pk_bf16_f32 v27, v27, s0
	v_cvt_pk_bf16_f32 v26, v26, s0
	v_cvt_pk_bf16_f32 v25, v25, s0
	v_cvt_pk_bf16_f32 v24, v24, s0
	v_cvt_pk_bf16_f32 v23, v23, s0
	v_cvt_pk_bf16_f32 v22, v22, s0
	v_cvt_pk_bf16_f32 v21, v21, s0
	v_cvt_pk_bf16_f32 v20, v20, s0
	v_cvt_pk_bf16_f32 v19, v19, s0
	v_cvt_pk_bf16_f32 v18, v18, s0
	v_cvt_pk_bf16_f32 v17, v17, s0
	v_accvgpr_read_b32 v16, a48
.Ltmp88:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[32:33], v40, off
	global_store_short v[36:37], v30, off offset:32
	global_store_short v[34:35], v27, off offset:32
	global_store_short v[38:39], v26, off offset:32
	global_store_short v[32:33], v25, off offset:32
	global_store_short v[36:37], v24, off offset:64
	global_store_short v[34:35], v23, off offset:64
	global_store_short v[38:39], v22, off offset:64
	global_store_short v[32:33], v21, off offset:64
	global_store_short v[36:37], v20, off offset:96
	global_store_short v[34:35], v19, off offset:96
	global_store_short v[38:39], v18, off offset:96
	global_store_short v[32:33], v17, off offset:96
	v_or_b32_e32 v17, 48, v2
.Ltmp89:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v18, v16, s0
.Ltmp90:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[16:17], s[0:1], v17, s6, 0
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[16:17], v[16:17], 1, v[28:29]
	v_accvgpr_read_b32 v15, a49
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[16:17], v18, off
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62
	v_or_b32_e32 v18, 49, v2
.Ltmp91:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v15, v15, s0
.Ltmp92:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[18:19], s[0:1], v18, s6, 0
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[28:29]
	v_accvgpr_read_b32 v14, a50
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[18:19], v15, off
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62
	v_or_b32_e32 v15, 50, v2
.Ltmp93:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v20, v14, s0
.Ltmp94:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[14:15], s[0:1], v15, s6, 0
	v_accvgpr_read_b32 v13, a51
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[14:15], v[14:15], 1, v[28:29]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62
	v_or_b32_e32 v2, 51, v2
	v_accvgpr_read_b32 v12, a52
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[14:15], v20, off
.Ltmp95:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v13, v13, s0
.Ltmp96:
	.loc	2 70 31                         ; ./v0_naive.h:70:31
	v_mad_i64_i32 v[20:21], s[0:1], v2, s6, 0
	v_accvgpr_read_b32 v11, a53
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17
	v_lshl_add_u64 v[20:21], v[20:21], 1, v[28:29]
.Ltmp97:
	.loc	4 214 87 is_stmt 1              ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v2, v12, s0
	v_accvgpr_read_b32 v10, a54
.Ltmp98:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[20:21], v13, off
	global_store_short v[16:17], v2, off offset:32
.Ltmp99:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v2, v11, s0
	v_accvgpr_read_b32 v9, a55
.Ltmp100:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[18:19], v2, off offset:32
.Ltmp101:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v2, v10, s0
	v_accvgpr_read_b32 v8, a56
.Ltmp102:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[14:15], v2, off offset:32
.Ltmp103:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v2, v9, s0
	v_accvgpr_read_b32 v7, a57
.Ltmp104:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[20:21], v2, off offset:32
.Ltmp105:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v2, v8, s0
	v_accvgpr_read_b32 v6, a58
.Ltmp106:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[16:17], v2, off offset:64
.Ltmp107:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v2, v7, s0
	v_accvgpr_read_b32 v5, a59
.Ltmp108:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[18:19], v2, off offset:64
.Ltmp109:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v2, v6, s0
	v_accvgpr_read_b32 v4, a60
.Ltmp110:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[14:15], v2, off offset:64
.Ltmp111:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v2, v5, s0
	v_accvgpr_read_b32 v3, a61
	v_accvgpr_read_b32 v1, a62
	v_accvgpr_read_b32 v0, a63
.Ltmp112:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[20:21], v2, off offset:64
.Ltmp113:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v2, v4, s0
.Ltmp114:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[16:17], v2, off offset:96
.Ltmp115:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]
	v_cvt_pk_bf16_f32 v2, v3, s0
	v_cvt_pk_bf16_f32 v1, v1, s0
	v_cvt_pk_bf16_f32 v0, v0, s0
.Ltmp116:
	.loc	2 70 42                         ; ./v0_naive.h:70:42
	global_store_short v[18:19], v2, off offset:96
	global_store_short v[14:15], v1, off offset:96
	global_store_short v[20:21], v0, off offset:96
	.loc	2 72 1                          ; ./v0_naive.h:72:1
	s_endpgm
.Ltmp117:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN2v013gemm_v0_naiveE7Problem
		.amdhsa_group_segment_fixed_size 16384
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 312
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 140
		.amdhsa_next_free_sgpr 38
		.amdhsa_accum_offset 76
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_ZN2v013gemm_v0_naiveE7Problem, .Lfunc_end0-_ZN2v013gemm_v0_naiveE7Problem
	.cfi_endproc
                                        ; -- End function
	.set _ZN2v013gemm_v0_naiveE7Problem.num_vgpr, 73
	.set _ZN2v013gemm_v0_naiveE7Problem.num_agpr, 64
	.set _ZN2v013gemm_v0_naiveE7Problem.numbered_sgpr, 38
	.set _ZN2v013gemm_v0_naiveE7Problem.num_named_barrier, 0
	.set _ZN2v013gemm_v0_naiveE7Problem.private_seg_size, 0
	.set _ZN2v013gemm_v0_naiveE7Problem.uses_vcc, 1
	.set _ZN2v013gemm_v0_naiveE7Problem.uses_flat_scratch, 0
	.set _ZN2v013gemm_v0_naiveE7Problem.has_dyn_sized_stack, 0
	.set _ZN2v013gemm_v0_naiveE7Problem.has_recursion, 0
	.set _ZN2v013gemm_v0_naiveE7Problem.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9948
; TotalNumSgprs: 44
; NumVgprs: 73
; NumAgprs: 64
; TotalNumVgprs: 140
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 16384 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 17
; NumSGPRsForWavesPerEU: 44
; NumVGPRsForWavesPerEU: 140
; AccumOffset: 76
; Occupancy: 3
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 18
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	.str,@object                    ; @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.str:
	.asciz	"load time: %llu\n"
	.size	.str, 17

	.type	__hip_cuid_1fd510d24433d8d9,@object ; @__hip_cuid_1fd510d24433d8d9
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_1fd510d24433d8d9
__hip_cuid_1fd510d24433d8d9:
	.byte	0                               ; 0x0
	.size	__hip_cuid_1fd510d24433d8d9, 1

	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	37                              ; DW_FORM_strx1
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	114                             ; DW_AT_str_offsets_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	37                              ; DW_FORM_strx1
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	115                             ; DW_AT_addr_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	116                             ; DW_AT_rnglists_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	32                              ; DW_AT_inline
	.byte	33                              ; DW_FORM_implicit_const
	.byte	1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	122                             ; DW_AT_call_all_calls
	.byte	25                              ; DW_FORM_flag_present
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	7                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	8                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	9                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	5                               ; DWARF version number
	.byte	1                               ; DWARF Unit Type
	.byte	8                               ; Address Size (in bytes)
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	1                               ; Abbrev [1] 0xc:0xdf DW_TAG_compile_unit
	.byte	0                               ; DW_AT_producer
	.short	33                              ; DW_AT_language
	.byte	1                               ; DW_AT_name
	.long	.Lstr_offsets_base0             ; DW_AT_str_offsets_base
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.byte	2                               ; DW_AT_comp_dir
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	.Laddr_table_base0              ; DW_AT_addr_base
	.long	.Lrnglists_table_base0          ; DW_AT_rnglists_base
	.byte	2                               ; Abbrev [2] 0x27:0x2 DW_TAG_subprogram
	.byte	3                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x29:0x2 DW_TAG_subprogram
	.byte	4                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x2b:0x2 DW_TAG_subprogram
	.byte	5                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x2d:0x2 DW_TAG_subprogram
	.byte	6                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x2f:0x2 DW_TAG_subprogram
	.byte	7                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x31:0x2 DW_TAG_subprogram
	.byte	8                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x33:0x2 DW_TAG_subprogram
	.byte	9                               ; DW_AT_name
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x35:0x2 DW_TAG_subprogram
	.byte	10                              ; DW_AT_name
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x37:0xb3 DW_TAG_subprogram
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	11                              ; DW_AT_name
	.byte	4                               ; Abbrev [4] 0x3e:0x1c DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	0                               ; DW_AT_low_pc
	.long	.Ltmp0-.Lfunc_begin0            ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.byte	17                              ; DW_AT_call_line
	.byte	23                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4b:0xe DW_TAG_inlined_subroutine
	.long	39                              ; DW_AT_abstract_origin
	.byte	0                               ; DW_AT_low_pc
	.long	.Ltmp0-.Lfunc_begin0            ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	634                             ; DW_AT_call_line
	.byte	83                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x5a:0x1c DW_TAG_inlined_subroutine
	.long	41                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_low_pc
	.long	.Ltmp1-.Ltmp0                   ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.byte	26                              ; DW_AT_call_line
	.byte	21                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x67:0xe DW_TAG_inlined_subroutine
	.long	39                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_low_pc
	.long	.Ltmp1-.Ltmp0                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	634                             ; DW_AT_call_line
	.byte	83                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x76:0x2b DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	2                               ; DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.byte	41                              ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x83:0x1d DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	2                               ; DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	714                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x91:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	2                               ; DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	707                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0xa1:0x9 DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	0                               ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	56                              ; DW_AT_call_line
	.byte	31                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0xaa:0x2b DW_TAG_inlined_subroutine
	.long	47                              ; DW_AT_abstract_origin
	.byte	3                               ; DW_AT_low_pc
	.long	.Ltmp13-.Ltmp12                 ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.byte	59                              ; DW_AT_call_line
	.byte	9                               ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0xb7:0x1d DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	3                               ; DW_AT_low_pc
	.long	.Ltmp13-.Ltmp12                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	714                             ; DW_AT_call_line
	.byte	3                               ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0xc5:0xe DW_TAG_inlined_subroutine
	.long	43                              ; DW_AT_abstract_origin
	.byte	3                               ; DW_AT_low_pc
	.long	.Ltmp13-.Ltmp12                 ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.short	707                             ; DW_AT_call_line
	.byte	50                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	8                               ; Abbrev [8] 0xd5:0x14 DW_TAG_inlined_subroutine
	.long	53                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	70                              ; DW_AT_call_line
	.byte	44                              ; DW_AT_call_column
	.byte	9                               ; Abbrev [9] 0xde:0xa DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_ranges
	.byte	4                               ; DW_AT_call_file
	.short	442                             ; DW_AT_call_line
	.byte	18                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 ; Length
.Ldebug_list_header_start0:
	.short	5                               ; Version
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
	.long	2                               ; Offset entry count
.Lrnglists_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_table_base0
	.long	.Ldebug_ranges1-.Lrnglists_table_base0
.Ldebug_ranges0:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp4-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp5-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp6-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp7-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp8-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp9-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp10-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp11-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp13-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp14-.Lfunc_begin0          ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges1:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp15-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp16-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp17-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp18-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp19-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp20-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp21-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp22-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp23-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp24-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp25-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp26-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp27-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp28-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp29-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp30-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp31-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp32-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp33-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp34-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp35-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp36-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp37-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp38-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp39-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp40-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp41-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp42-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp43-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp44-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp45-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp46-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp47-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp48-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp49-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp50-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp51-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp52-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp53-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp54-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp55-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp56-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp57-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp58-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp59-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp60-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp61-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp62-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp63-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp64-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp65-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp66-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp67-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp68-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp69-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp70-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp71-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp72-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp73-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp74-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp75-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp76-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp77-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp78-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp79-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp80-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp81-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp82-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp83-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp84-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp85-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp86-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp87-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp88-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp89-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp90-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp91-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp92-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp93-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp94-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp95-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp96-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp97-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp98-.Lfunc_begin0          ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp99-.Lfunc_begin0          ;   starting offset
	.uleb128 .Ltmp100-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp101-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp102-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp103-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp104-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp105-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp106-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp107-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp108-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp109-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp110-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp111-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp112-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp113-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp114-.Lfunc_begin0         ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp115-.Lfunc_begin0         ;   starting offset
	.uleb128 .Ltmp116-.Lfunc_begin0         ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	52                              ; Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)" ; string offset=0
.Linfo_string1:
	.asciz	"gemm_lab.hip"                  ; string offset=137
.Linfo_string2:
	.asciz	"/home/haoyanli/a121_quark/mbs/lab" ; string offset=150
.Linfo_string3:
	.asciz	"__clock64"                     ; string offset=184
.Linfo_string4:
	.asciz	"clock64"                       ; string offset=194
.Linfo_string5:
	.asciz	"__work_group_barrier"          ; string offset=202
.Linfo_string6:
	.asciz	"__barrier"                     ; string offset=223
.Linfo_string7:
	.asciz	"__syncthreads"                 ; string offset=233
.Linfo_string8:
	.asciz	"mfma_fp4"                      ; string offset=247
.Linfo_string9:
	.asciz	"__hip_bfloat16"                ; string offset=256
.Linfo_string10:
	.asciz	"__float2bfloat16"              ; string offset=271
.Linfo_string11:
	.asciz	"gemm_v0_naive"                 ; string offset=288
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string10
	.long	.Linfo_string11
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 ; Length of contribution
.Ldebug_addr_start0:
	.short	5                               ; DWARF version number
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Ltmp0
	.quad	.Ltmp2
	.quad	.Ltmp12
.Ldebug_addr_end0:
	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_1fd510d24433d8d9
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     64
    .args:
      - .offset:         0
        .size:           56
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         60
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         68
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         70
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         72
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         74
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         76
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         78
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         120
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         136
        .size:           8
        .value_kind:     hidden_hostcall_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 312
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN2v013gemm_v0_naiveE7Problem
    .private_segment_fixed_size: 0
    .sgpr_count:     44
    .sgpr_spill_count: 0
    .symbol:         _ZN2v013gemm_v0_naiveE7Problem.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     140
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
