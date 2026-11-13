; ============================================================================================
; gemm_lab_注释.s：lab/v0_naive.h 编译出来的 GPU 汇编（gfx950），加了中文注释。
; 原文件是 make asm 生成的 build/gemm_lab.s。注释都写在 ';'（元数据里是 '#'）后面，没有改动任何指令。
;
; 怎么读
;   * 一行一条指令：「指令名 目的操作数, 源操作数, ...」。';' 后面是注释。
;   * 寄存器
;       s0、s[8:9]   标量寄存器（SGPR）：一个值整个 wave 共用。s[8:9] 是 s8、s9 拼成的 64 位。
;       v0、v[46:47] 向量寄存器（VGPR）：64 个 lane 各有一份，各算各的。
;       a0、a[0:3]   累加寄存器（AGPR）：MFMA 的累加结果，就是源码里的 acc[4][4]。
;       exec         64 位掩码，每位对应一个 lane，为 1 的 lane 才执行向量指令。
;       vcc          向量比较的结果，每个 lane 一位。
;       scc          标量比较的结果，一位；s_cbranch_scc0/1 按它跳转。
;   * 指令名：s_ 开头是标量指令，v_ 是向量指令，global_ 读写显存，ds_ 读写 LDS，
;     buffer_wbl2/buffer_inv 是缓存控制。后缀 _e32/_e64 只是编码长度不同（4 或 8 字节）。
;   * .loc 文件号 行号 列号：从这一行往下的指令来自哪一行源码。文件 2 是 v0_naive.h，
;     注释里附上了那一行源码；行号 0 表示编译器生成的代码，不对应某一行。
;   * s_waitcnt vmcnt(N)：等到还没完成的显存读写不超过 N 条；lgkmcnt(N) 同理，管 LDS 和标量读。
;     load 发出去以后 wave 不等它，接着往下走，要用数据之前才用 s_waitcnt 等。
;   * global_load ... off：地址在 VGPR 里；offset:N 表示地址再加 N 字节。
;   * .LBB0_xxx: 是跳转目标（标签）；「; %bb.xxx」是编译器给代码块起的名字，只是注释。
;   * 注释里说的「第 N 行」都是 v0_naive.h 的行号。加了注释以后，这个 .s 文件自己的行号会往后移。
;
; 文件结构（搜【1】到【7】可以跳过去）
;   【1】开头：kernel 声明、两次 clock64()
;   【2】printf 的库代码，约 1,250 条指令，和 GEMM 无关（分【2a】【2b】【2c】三轮）
;   【3】GEMM 主体：算下标、清零累加器、K 循环（拷贝到 LDS、读 scale、16 条 MFMA）
;   【4】收尾：64 个累加结果转成 bf16，写回 C
;   【5】kernel 描述符、资源统计、printf 的格式串
;   【6】DWARF 调试信息：给调试器和 ATT 用的数据表，不是指令
;   【7】kernel 元数据：参数列表、寄存器数
;
; GEMM 部分常用的寄存器（下面的注释会反复用到）
;   s[0:1] = kernel 参数区的地址     s2 = blockIdx.x     s3 = blockIdx.y
;   s6 = p.N    s7 = p.K             s[8:9] = p.A    s[10:11] = p.B    s[16:17] = p.C
;   s[12:13] = p.As、s[14:15] = p.Bs（算完 scale 地址后这两对被挪作他用）
;   s4 = n0 = blockIdx.x*128         s5 = m0 = blockIdx.y*128
;   s18 = KB = K/2（每行字节数）     s13 = k0（K 循环变量）
;   v0 = threadIdx.x（tid，之后被复用）    v1 = 0（拼 64 位地址时当高 32 位）
;   v2 = g = lane/16    v28 = r16 = lane%16    v29 = wm*64    v30 = wn*64
;   a0–a63 = acc[4][4]，acc[mi][ni] 占 a[16*mi+4*ni] 起的 4 个
;   寄存器用完就会被挪作他用，具体某一行里存的是什么，以那一行的注释为准。
; ============================================================================================
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"  ; 目标 GPU：gfx950（MI350 系列）
	.amdhsa_code_object_version 6  ; code object 的格式版本
	.text  ; 下面是代码段
	.protected	_ZN2v013gemm_v0_naiveE7Problem ; -- Begin function _ZN2v013gemm_v0_naiveE7Problem  → 符号可见性，给链接器用
	.globl	_ZN2v013gemm_v0_naiveE7Problem  ; 导出 kernel 名：主机端按这个名字找到它
	.p2align	8  ; kernel 入口按 2^8 = 256 字节对齐
	.type	_ZN2v013gemm_v0_naiveE7Problem,@function  ; 这个符号是一个函数
; ============================================================================================
; 【1】开头：kernel 入口、两次 clock64()（v0_naive.h 第 17、26 行）
; ============================================================================================
_ZN2v013gemm_v0_naiveE7Problem:         ; @_ZN2v013gemm_v0_naiveE7Problem  ← kernel 入口。名字是 v0::gemm_v0_naive(Problem) 编码后的样子
.Lfunc_begin0:  ; 函数开始的位置（调试信息用）
	.file	0 "/home/haoyanli/a121_quark/mbs/lab" "gemm_lab.hip" md5 0x394887fa59d9a0b9e418e5cbe756be2c  ; 文件 0：gemm_lab.hip。.loc 用编号来引用文件
	.cfi_sections .debug_frame  ; 调用帧信息，给调试器用
	.cfi_startproc  ; 调用帧信息：函数开始
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ;   → 调用帧信息，给调试器用
	.cfi_undefined 16  ; 调用帧信息，给调试器用
	.file	1 "/opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail" "amd_device_functions.h" md5 0x1a8b7178a036d604cccb3d2554abd5c3  ; 文件 1：HIP 运行库的头文件
	.loc	1 623 25 prologue_end           ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:623:25 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:634:83 @[ ./v0_naive.h:17:23 ] ]  → clock64() 的实现
	s_memtime s[20:21]  ; start = clock64()（第 17 行）：读 GPU 的时间计数器 → s[20:21]
.Ltmp0:
	.loc	1 623 25 is_stmt 0              ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:623:25 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:634:83 @[ ./v0_naive.h:26:21 ] ]  → clock64() 的实现
	s_memtime s[22:23]  ; end = clock64()（第 26 行）。编译器把它提到了这里，中间那几行下标计算被挪到 printf 之后，所以两次读时间之间几乎没有指令，打印出来的 load time 不是那段代码的耗时
.Ltmp1:
	.file	2 "." "v0_naive.h" md5 0xdb790ced971ad47b552b828380ce9d6c  ; 文件 2：v0_naive.h
; ============================================================================================
; 【2】printf("load time: %llu\n", end - start)（v0_naive.h 第 27 行）
; 从这里到【3】之前约 1,470 行、1,250 条指令，都是 printf 的库代码，和 GEMM 无关。
; GPU 不能直接往终端打字。printf 的做法是：把格式串和参数写进一块和主机共享的内存
; （hostcall 缓冲区），再发中断让 CPU 来读、来打印。大致分三轮，每轮的流程差不多：
;   1. 选一个 lane 当代表（v_mbcnt、v_readfirstlane、s_and_saveexec）；
;   2. 用原子比较交换从缓冲区的空闲链表里抢一个槽位，抢不到就 s_sleep 一下再试；
;   3. 往槽位里写数据（global_store_dwordx4）；
;   4. 把槽位挂到待处理链表上，原子加更新计数，s_sendmsg 发中断通知主机；
;   5. 循环 s_sleep 等主机处理完，再把槽位还回空闲链表。
; 【2a】开始一条 printf；【2b】送格式串；【2c】送参数 end - start 并结束。
; GEMM 要用的几个 kernel 参数（p.A、p.B、p.As、p.Bs、p.C、p.N、p.K）也被编译器
; 提前到这一段里读，那几行单独标了注释。学 GEMM 时可以先删掉 printf 和 clock64 再看汇编。
; 【2a】第 1 轮：开始一条 printf
; ============================================================================================
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行：printf("load time: %llu\n", end - start);
	s_load_dwordx2 s[18:19], s[0:1], 0x88  ; 读参数区偏移 0x88 = 136：printf 用的 hostcall 缓冲区地址（见【7】hidden_hostcall_buffer）
	v_mbcnt_lo_u32_b32 v1, -1, 0  ; 数 exec 里排在本 lane 前面的有效 lane（低 32 位）
	v_mbcnt_hi_u32_b32 v1, -1, v1  ; 同上（高 32 位）：结果是本 lane 在有效 lane 里排第几
	v_mov_b64_e32 v[6:7], 0  ; 64 位向量赋值
	v_readfirstlane_b32 s4, v1  ; 取第一个有效 lane 的值，放进标量寄存器
	s_nop 1  ; 空几个周期，避开硬件的时序冲突
	v_cmp_eq_u32_e64 s[4:5], s4, v1  ; 比较「相等」，结果写到一对 SGPR
	s_and_saveexec_b64 s[6:7], s[4:5]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_6  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.1:
	v_mov_b32_e32 v2, 0  ; 向量赋值
	s_waitcnt lgkmcnt(0)  ; 等 LDS 或标量读完成
	global_load_dwordx2 v[8:9], v2, s[18:19] offset:24 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	global_load_dwordx2 v[4:5], v2, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[6:7], v2, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(1)  ; 等显存读写完成
	v_and_b32_e32 v3, v4, v8  ; 按位与
	v_and_b32_e32 v4, v5, v9  ; 按位与
	v_mul_lo_u32 v4, v4, 24  ; 乘法，取低 32 位
	v_mul_hi_u32 v5, v3, 24  ; 乘法，取高 32 位
	v_add_u32_e32 v5, v5, v4  ; 加法
	v_mul_lo_u32 v4, v3, 24  ; 乘法，取低 32 位
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_lshl_add_u64 v[4:5], v[6:7], 0, v[4:5]  ; 64 位：左移后相加（这里多用来算地址）
	global_load_dwordx2 v[6:7], v[4:5], off sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[6:7], v2, v[6:9], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	v_cmp_ne_u64_e32 vcc, v[6:7], v[8:9]  ; 64 位比较「不等」，结果写到 vcc
	s_and_saveexec_b64 s[8:9], vcc  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_5  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.2:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b64 s[10:11], 0  ; 64 位标量赋值
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	s_sleep 1  ; 睡一会儿再查
	global_load_dwordx2 v[4:5], v2, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[10:11], v2, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	v_mov_b64_e32 v[8:9], v[6:7]  ; 64 位向量赋值
	s_waitcnt vmcnt(1)  ; 等显存读写完成
	v_and_b32_e32 v4, v4, v8  ; 按位与
	v_and_b32_e32 v3, v5, v9  ; 按位与
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_mad_u64_u32 v[4:5], s[12:13], v4, 24, v[10:11]  ; 乘加：32 位 × 32 位得到 64 位，再加一个 64 位数
	v_mov_b32_e32 v6, v5  ; 向量赋值
	v_mad_u64_u32 v[6:7], s[12:13], v3, 24, v[6:7]  ; 乘加：32 位 × 32 位得到 64 位，再加一个 64 位数
	v_mov_b32_e32 v5, v6  ; 向量赋值
	global_load_dwordx2 v[6:7], v[4:5], off sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[6:7], v2, v[6:9], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]  ; 64 位比较「相等」，结果写到 vcc
	s_or_b64 s[10:11], vcc, s[10:11]  ; 64 位按位或
	s_andn2_b64 exec, exec, s[10:11]  ; 把已经完成的 lane 从 exec 里去掉（循环用）
	s_cbranch_execnz .LBB0_3  ; 还有 lane 没完成（exec ≠ 0）就跳回去
; %bb.4:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[10:11]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
.LBB0_5:
	s_or_b64 exec, exec, s[8:9]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
.LBB0_6:
	s_or_b64 exec, exec, s[6:7]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	v_mov_b32_e32 v25, 0  ; 向量赋值
	s_waitcnt lgkmcnt(0)  ; 等 LDS 或标量读完成
	global_load_dwordx2 v[8:9], v25, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx4 v[2:5], v25, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	s_load_dwordx8 s[8:15], s[0:1], 0x10  ; GEMM 要用：读 p.A、p.B、p.As、p.Bs 四个指针 → s[8:9]、s[10:11]、s[12:13]、s[14:15]
	s_load_dwordx2 s[16:17], s[0:1], 0x30  ; GEMM 要用：读 p.C 的地址 → s[16:17]
	v_readfirstlane_b32 s25, v7  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s24, v6  ; 取第一个有效 lane 的值，放进标量寄存器
	s_mov_b64 s[6:7], exec  ; 64 位标量赋值
	s_waitcnt vmcnt(1)  ; 等显存读写完成
	v_readfirstlane_b32 s26, v8  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s27, v9  ; 取第一个有效 lane 的值，放进标量寄存器
	s_and_b64 s[26:27], s[26:27], s[24:25]  ; 64 位按位与
	s_mul_i32 s28, s27, 24  ; 乘法，取低 32 位
	s_mul_hi_u32 s29, s26, 24  ; 乘法，取高 32 位
	s_add_i32 s29, s29, s28  ; 加法
	s_mul_i32 s28, s26, 24  ; 乘法，取低 32 位
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_lshl_add_u64 v[6:7], v[2:3], 0, s[28:29]  ; 64 位：左移后相加（这里多用来算地址）
	s_and_saveexec_b64 s[28:29], s[4:5]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_8  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.7:
	v_mov_b64_e32 v[8:9], s[6:7]  ; 64 位向量赋值
	v_mov_b32_e32 v10, 2  ; 向量赋值
	v_mov_b32_e32 v11, 1  ; 向量赋值
	global_store_dwordx4 v[6:7], v[8:11], off offset:8  ; 写 printf 缓冲区
.LBB0_8:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[28:29]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	s_load_dwordx2 s[6:7], s[0:1], 0x4  ; GEMM 要用：读 p.N、p.K → s6、s7（p.M 在偏移 0，没用到）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_lshl_b64 s[0:1], s[26:27], 12  ; 64 位左移
	v_lshl_add_u64 v[8:9], v[4:5], 0, s[0:1]  ; 64 位：左移后相加（这里多用来算地址）
	s_mov_b32 s28, 0  ; 标量赋值
	v_lshlrev_b32_e32 v24, 6, v1  ; 左移（乘以 2 的幂）
	v_mov_b32_e32 v10, 33  ; 向量赋值
	v_mov_b32_e32 v11, v25  ; 向量赋值
	v_mov_b32_e32 v12, v25  ; 向量赋值
	v_mov_b32_e32 v13, v25  ; 向量赋值
	v_readfirstlane_b32 s0, v8  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s1, v9  ; 取第一个有效 lane 的值，放进标量寄存器
	s_mov_b32 s29, s28  ; 标量赋值
	s_mov_b32 s30, s28  ; 标量赋值
	s_mov_b32 s31, s28  ; 标量赋值
	s_nop 1  ; 空几个周期，避开硬件的时序冲突
	global_store_dwordx4 v24, v[10:13], s[0:1]  ; 写 printf 缓冲区
	s_nop 1  ; 空几个周期，避开硬件的时序冲突
	v_mov_b64_e32 v[10:11], s[28:29]  ; 64 位向量赋值
	v_mov_b64_e32 v[12:13], s[30:31]  ; 64 位向量赋值
	global_store_dwordx4 v24, v[10:13], s[0:1] offset:16  ; 写 printf 缓冲区
	global_store_dwordx4 v24, v[10:13], s[0:1] offset:32  ; 写 printf 缓冲区
	global_store_dwordx4 v24, v[10:13], s[0:1] offset:48  ; 写 printf 缓冲区
	s_and_saveexec_b64 s[0:1], s[4:5]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_16  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.9:
	v_mov_b32_e32 v12, 0  ; 向量赋值
	global_load_dwordx2 v[16:17], v12, s[18:19] offset:32 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[4:5], v12, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	v_mov_b32_e32 v14, s24  ; 向量赋值
	v_mov_b32_e32 v15, s25  ; 向量赋值
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_and_b32_e32 v4, s24, v4  ; 按位与
	v_and_b32_e32 v5, s25, v5  ; 按位与
	v_mul_lo_u32 v5, v5, 24  ; 乘法，取低 32 位
	v_mul_hi_u32 v10, v4, 24  ; 乘法，取高 32 位
	v_mul_lo_u32 v4, v4, 24  ; 乘法，取低 32 位
	v_add_u32_e32 v5, v10, v5  ; 加法
	v_lshl_add_u64 v[10:11], v[2:3], 0, v[4:5]  ; 64 位：左移后相加（这里多用来算地址）
	global_store_dwordx2 v[10:11], v[16:17], off  ; 写 printf 缓冲区
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[4:5], v12, v[14:17], s[18:19] offset:32 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_ne_u64_e32 vcc, v[4:5], v[16:17]  ; 64 位比较「不等」，结果写到 vcc
	s_and_saveexec_b64 s[26:27], vcc  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_12  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.10:
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b64 s[28:29], 0  ; 64 位标量赋值
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	s_sleep 1  ; 睡一会儿再查
	global_store_dwordx2 v[10:11], v[4:5], off  ; 写 printf 缓冲区
	v_mov_b32_e32 v2, s24  ; 向量赋值
	v_mov_b32_e32 v3, s25  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[2:3], v12, v[2:5], s[18:19] offset:32 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]  ; 64 位比较「相等」，结果写到 vcc
	s_or_b64 s[28:29], vcc, s[28:29]  ; 64 位按位或
	v_mov_b64_e32 v[4:5], v[2:3]  ; 64 位向量赋值
	s_andn2_b64 exec, exec, s[28:29]  ; 把已经完成的 lane 从 exec 里去掉（循环用）
	s_cbranch_execnz .LBB0_11  ; 还有 lane 没完成（exec ≠ 0）就跳回去
.LBB0_12:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[26:27]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_mov_b32_e32 v5, 0  ; 向量赋值
	global_load_dwordx2 v[2:3], v5, s[18:19] offset:16  ; 读 printf 缓冲区（和主机共享的内存）
	s_mov_b64 s[26:27], exec  ; 64 位标量赋值
	v_mbcnt_lo_u32_b32 v4, s26, 0  ; 数 exec 里排在本 lane 前面的有效 lane（低 32 位）
	v_mbcnt_hi_u32_b32 v4, s27, v4  ; 同上（高 32 位）：结果是本 lane 在有效 lane 里排第几
	v_cmp_eq_u32_e32 vcc, 0, v4  ; 比较「相等」，结果写到 vcc
	s_and_saveexec_b64 s[28:29], vcc  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_14  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.13:
	s_bcnt1_i32_b64 s26, s[26:27]  ; 数 64 位里有几个 1（这里是有几个有效 lane）
	v_mov_b32_e32 v4, s26  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_add_x2 v[2:3], v[4:5], off offset:8 sc1  ; 原子加：更新通知主机用的计数
.LBB0_14:
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[28:29]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_load_dwordx2 v[4:5], v[2:3], off offset:16  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_eq_u64_e32 vcc, 0, v[4:5]  ; 64 位比较「相等」，结果写到 vcc
	s_cbranch_vccnz .LBB0_16  ; vcc 不为 0 就跳转
; %bb.15:
	global_load_dword v2, v[2:3], off offset:24  ; 读 printf 缓冲区（和主机共享的内存）
	v_mov_b32_e32 v3, 0  ; 向量赋值
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_readfirstlane_b32 s26, v2  ; 取第一个有效 lane 的值，放进标量寄存器
	s_and_b32 m0, s26, 0xffffff  ; 按位与
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	global_store_dwordx2 v[4:5], v[2:3], off sc0 sc1  ; 写 printf 缓冲区
	s_sendmsg sendmsg(MSG_INTERRUPT)  ; 发中断给主机：有 printf 数据要处理
.LBB0_16:
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[0:1]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	v_lshl_add_u64 v[2:3], v[8:9], 0, v[24:25]  ; 64 位：左移后相加（这里多用来算地址）
	s_branch .LBB0_20  ; 无条件跳转
.LBB0_17:                               ;   in Loop: Header=BB0_20 Depth=1
	s_or_b64 exec, exec, s[0:1]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_readfirstlane_b32 s0, v4  ; 取第一个有效 lane 的值，放进标量寄存器
	s_cmp_eq_u32 s0, 0  ; 比较「相等」，结果写到 scc
	s_cbranch_scc1 .LBB0_19  ; scc = 1 就跳转
; %bb.18:                               ;   in Loop: Header=BB0_20 Depth=1
	s_sleep 1  ; 睡一会儿再查
	s_cbranch_execnz .LBB0_20  ; 还有 lane 没完成（exec ≠ 0）就跳回去
	s_branch .LBB0_22  ; 无条件跳转
.LBB0_19:
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_branch .LBB0_22  ; 无条件跳转
.LBB0_20:                               ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v4, 1  ; 向量赋值
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	s_and_saveexec_b64 s[0:1], s[4:5]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_17  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.21:                               ;   in Loop: Header=BB0_20 Depth=1
	global_load_dword v4, v[6:7], off offset:20 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	v_and_b32_e32 v4, 1, v4  ; 按位与
	s_branch .LBB0_17  ; 无条件跳转
.LBB0_22:
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	global_load_dwordx2 v[2:3], v[2:3], off  ; 读 printf 缓冲区（和主机共享的内存）
	s_and_saveexec_b64 s[0:1], s[4:5]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_25  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.23:
	v_mov_b32_e32 v10, 0  ; 向量赋值
	global_load_dwordx2 v[4:5], v10, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[14:15], v10, s[18:19] offset:24 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[6:7], v10, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(2)  ; 等显存读写完成
	v_readfirstlane_b32 s26, v4  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s27, v5  ; 取第一个有效 lane 的值，放进标量寄存器
	s_add_u32 s28, s26, 1  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s29, s27, 0  ; 带进位加，64 位加法的高 32 位
	s_add_u32 s4, s28, s24  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s5, s29, s25  ; 带进位加，64 位加法的高 32 位
	s_cmp_eq_u64 s[4:5], 0  ; 64 位比较「相等」，结果写到 scc
	s_cselect_b32 s5, s29, s5  ; 按 scc 二选一
	s_cselect_b32 s4, s28, s4  ; 按 scc 二选一
	s_and_b64 s[24:25], s[4:5], s[26:27]  ; 64 位按位与
	s_mul_i32 s25, s25, 24  ; 乘法，取低 32 位
	s_mul_hi_u32 s26, s24, 24  ; 乘法，取高 32 位
	s_mul_i32 s24, s24, 24  ; 乘法，取低 32 位
	s_add_i32 s25, s26, s25  ; 加法
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_lshl_add_u64 v[8:9], v[6:7], 0, s[24:25]  ; 64 位：左移后相加（这里多用来算地址）
	v_mov_b32_e32 v12, s4  ; 向量赋值
	global_store_dwordx2 v[8:9], v[14:15], off  ; 写 printf 缓冲区
	v_mov_b32_e32 v13, s5  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[6:7], v10, v[12:15], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_mov_b64 s[24:25], 0  ; 64 位标量赋值
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_ne_u64_e32 vcc, v[6:7], v[14:15]  ; 64 位比较「不等」，结果写到 vcc
	s_and_b64 exec, exec, vcc  ; 只让条件成立的 lane 继续
	s_cbranch_execz .LBB0_25  ; 没有 lane 要执行（exec = 0）就跳过这段
.LBB0_24:                               ; =>This Inner Loop Header: Depth=1
	s_sleep 1  ; 睡一会儿再查
	global_store_dwordx2 v[8:9], v[6:7], off  ; 写 printf 缓冲区
	v_mov_b32_e32 v4, s4  ; 向量赋值
	v_mov_b32_e32 v5, s5  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[4:5], v10, v[4:7], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]  ; 64 位比较「相等」，结果写到 vcc
	s_or_b64 s[24:25], vcc, s[24:25]  ; 64 位按位或
	v_mov_b64_e32 v[6:7], v[4:5]  ; 64 位向量赋值
	s_andn2_b64 exec, exec, s[24:25]  ; 把已经完成的 lane 从 exec 里去掉（循环用）
	s_cbranch_execnz .LBB0_24  ; 还有 lane 没完成（exec ≠ 0）就跳回去
.LBB0_25:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[0:1]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
; ============================================================================================
; 【2b】第 2 轮：送格式串
; s_getpc 算出格式串 .str 的地址，长度 17 字节。每次最多送 56 字节：
; 逐字节读出来（global_load_ubyte），拼成 8 字节一组，再按【2】里的流程交给主机。
; ============================================================================================
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行：printf("load time: %llu\n", end - start);
	s_getpc_b64 s[4:5]  ; 取当前指令的地址
	s_add_u32 s4, s4, .str@rel32@lo+4  ; 加上偏移，算出格式串 .str 的地址（低 32 位）
	s_addc_u32 s5, s5, .str@rel32@hi+12  ; （高 32 位）
	s_cmp_lg_u64 s[4:5], 0  ; 64 位比较「不等」，结果写到 scc
	s_cbranch_scc0 .LBB0_111  ; scc = 0 就跳转
; %bb.26:
	v_mov_b32_e32 v29, 0  ; 向量赋值
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_and_b32_e32 v26, 2, v2  ; 按位与
	v_and_b32_e32 v4, -3, v2  ; 按位与
	v_mov_b32_e32 v5, v3  ; 向量赋值
	s_mov_b64 s[24:25], 17  ; 格式串长度 17 字节：16 个字符加结尾的 0
	v_mov_b32_e32 v25, v29  ; 向量赋值
	v_mov_b32_e32 v34, 2  ; 向量赋值
	v_mov_b32_e32 v35, 1  ; 向量赋值
	s_branch .LBB0_28  ; 无条件跳转
.LBB0_27:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[30:31]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_sub_u32 s24, s24, s26  ; 减法，64 位减法的低 32 位
	s_subb_u32 s25, s25, s27  ; 带借位减，64 位减法的高 32 位
	s_add_u32 s4, s4, s26  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s5, s5, s27  ; 带进位加，64 位加法的高 32 位
	s_cmp_lg_u64 s[24:25], 0  ; 64 位比较「不等」，结果写到 scc
	s_cbranch_scc0 .LBB0_110  ; scc = 0 就跳转
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
	v_cmp_lt_u64_e64 s[0:1], s[24:25], 56  ; 64 位比较「小于」，结果写到一对 SGPR
	s_and_b64 s[0:1], s[0:1], exec  ; 64 位按位与
	s_cselect_b32 s27, s25, 0  ; 按 scc 二选一
	s_cselect_b32 s26, s24, 56  ; 按 scc 二选一
	v_cmp_gt_u64_e64 s[28:29], s[24:25], 7  ; 64 位比较「大于」，结果写到一对 SGPR
	s_add_u32 s0, s4, 8  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s1, s5, 0  ; 带进位加，64 位加法的高 32 位
	s_and_b64 vcc, exec, s[28:29]  ; 64 位按位与
	s_cbranch_vccnz .LBB0_32  ; vcc 不为 0 就跳转
; %bb.29:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u64 s[24:25], 0  ; 64 位比较「相等」，结果写到 scc
	s_cbranch_scc1 .LBB0_33  ; scc = 1 就跳转
; %bb.30:                               ;   in Loop: Header=BB0_28 Depth=1
	s_lshl_b64 s[0:1], s[26:27], 3  ; 64 位左移
	s_mov_b64 s[28:29], 0  ; 64 位标量赋值
	v_mov_b64_e32 v[6:7], 0  ; 64 位向量赋值
	s_mov_b64 s[30:31], s[4:5]  ; 64 位标量赋值
.LBB0_31:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_ubyte v8, v29, s[30:31]  ; 逐字节读格式串
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_and_b32_e32 v28, 0xffff, v8  ; 按位与
	v_lshlrev_b64 v[8:9], s28, v[28:29]  ; 64 位左移
	s_add_u32 s28, s28, 8  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s29, s29, 0  ; 带进位加，64 位加法的高 32 位
	s_add_u32 s30, s30, 1  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s31, s31, 0  ; 带进位加，64 位加法的高 32 位
	v_or_b32_e32 v6, v8, v6  ; 按位或
	s_cmp_lg_u32 s0, s28  ; 比较「不等」，结果写到 scc
	v_or_b32_e32 v7, v9, v7  ; 按位或
	s_cbranch_scc1 .LBB0_31  ; scc = 1 就跳转
	s_branch .LBB0_34  ; 无条件跳转
.LBB0_32:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b32 s33, 0  ; 标量赋值
	s_branch .LBB0_35  ; 无条件跳转
.LBB0_33:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[6:7], 0  ; 64 位向量赋值
.LBB0_34:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[0:1], s[4:5]  ; 64 位标量赋值
	s_mov_b32 s33, 0  ; 标量赋值
	s_cbranch_execnz .LBB0_36  ; 还有 lane 没完成（exec ≠ 0）就跳回去
.LBB0_35:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	global_load_dwordx2 v[6:7], v29, s[4:5]  ; 读 printf 缓冲区（和主机共享的内存）
	s_add_i32 s33, s26, -8  ; 加法
.LBB0_36:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_add_u32 s28, s0, 8  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s29, s1, 0  ; 带进位加，64 位加法的高 32 位
	s_cmp_gt_u32 s33, 7  ; 比较「大于」，结果写到 scc
	s_cbranch_scc1 .LBB0_40  ; scc = 1 就跳转
; %bb.37:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s33, 0  ; 比较「相等」，结果写到 scc
	s_cbranch_scc1 .LBB0_41  ; scc = 1 就跳转
; %bb.38:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b64 s[28:29], 0  ; 64 位标量赋值
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_mov_b64_e32 v[8:9], 0  ; 64 位向量赋值
	s_mov_b64 s[30:31], 0  ; 64 位标量赋值
.LBB0_39:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s34, s0, s30  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s35, s1, s31  ; 带进位加，64 位加法的高 32 位
	global_load_ubyte v10, v29, s[34:35]  ; 逐字节读格式串
	s_add_u32 s30, s30, 1  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s31, s31, 0  ; 带进位加，64 位加法的高 32 位
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_and_b32_e32 v28, 0xffff, v10  ; 按位与
	v_lshlrev_b64 v[10:11], s28, v[28:29]  ; 64 位左移
	s_add_u32 s28, s28, 8  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s29, s29, 0  ; 带进位加，64 位加法的高 32 位
	v_or_b32_e32 v8, v10, v8  ; 按位或
	s_cmp_lg_u32 s33, s30  ; 比较「不等」，结果写到 scc
	v_or_b32_e32 v9, v11, v9  ; 按位或
	s_cbranch_scc1 .LBB0_39  ; scc = 1 就跳转
	s_branch .LBB0_42  ; 无条件跳转
.LBB0_40:                               ;   in Loop: Header=BB0_28 Depth=1
                                        ; implicit-def: $vgpr8_vgpr9
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b32 s34, 0  ; 标量赋值
	s_branch .LBB0_43  ; 无条件跳转
.LBB0_41:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[8:9], 0  ; 64 位向量赋值
.LBB0_42:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[28:29], s[0:1]  ; 64 位标量赋值
	s_mov_b32 s34, 0  ; 标量赋值
	s_cbranch_execnz .LBB0_44  ; 还有 lane 没完成（exec ≠ 0）就跳回去
.LBB0_43:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	global_load_dwordx2 v[8:9], v29, s[0:1]  ; 读 printf 缓冲区（和主机共享的内存）
	s_add_i32 s34, s33, -8  ; 加法
.LBB0_44:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_add_u32 s0, s28, 8  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s1, s29, 0  ; 带进位加，64 位加法的高 32 位
	s_cmp_gt_u32 s34, 7  ; 比较「大于」，结果写到 scc
	s_cbranch_scc1 .LBB0_48  ; scc = 1 就跳转
; %bb.45:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s34, 0  ; 比较「相等」，结果写到 scc
	s_cbranch_scc1 .LBB0_49  ; scc = 1 就跳转
; %bb.46:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b64 s[0:1], 0  ; 64 位标量赋值
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_mov_b64_e32 v[10:11], 0  ; 64 位向量赋值
	s_mov_b64 s[30:31], 0  ; 64 位标量赋值
.LBB0_47:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s36, s28, s30  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s37, s29, s31  ; 带进位加，64 位加法的高 32 位
	global_load_ubyte v12, v29, s[36:37]  ; 逐字节读格式串
	s_add_u32 s30, s30, 1  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s31, s31, 0  ; 带进位加，64 位加法的高 32 位
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_and_b32_e32 v28, 0xffff, v12  ; 按位与
	v_lshlrev_b64 v[12:13], s0, v[28:29]  ; 64 位左移
	s_add_u32 s0, s0, 8  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s1, s1, 0  ; 带进位加，64 位加法的高 32 位
	v_or_b32_e32 v10, v12, v10  ; 按位或
	s_cmp_lg_u32 s34, s30  ; 比较「不等」，结果写到 scc
	v_or_b32_e32 v11, v13, v11  ; 按位或
	s_cbranch_scc1 .LBB0_47  ; scc = 1 就跳转
	s_branch .LBB0_50  ; 无条件跳转
.LBB0_48:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b32 s33, 0  ; 标量赋值
	s_branch .LBB0_51  ; 无条件跳转
.LBB0_49:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[10:11], 0  ; 64 位向量赋值
.LBB0_50:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[0:1], s[28:29]  ; 64 位标量赋值
	s_mov_b32 s33, 0  ; 标量赋值
	s_cbranch_execnz .LBB0_52  ; 还有 lane 没完成（exec ≠ 0）就跳回去
.LBB0_51:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	global_load_dwordx2 v[10:11], v29, s[28:29]  ; 读 printf 缓冲区（和主机共享的内存）
	s_add_i32 s33, s34, -8  ; 加法
.LBB0_52:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_add_u32 s28, s0, 8  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s29, s1, 0  ; 带进位加，64 位加法的高 32 位
	s_cmp_gt_u32 s33, 7  ; 比较「大于」，结果写到 scc
	s_cbranch_scc1 .LBB0_56  ; scc = 1 就跳转
; %bb.53:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s33, 0  ; 比较「相等」，结果写到 scc
	s_cbranch_scc1 .LBB0_57  ; scc = 1 就跳转
; %bb.54:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b64 s[28:29], 0  ; 64 位标量赋值
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_mov_b64_e32 v[12:13], 0  ; 64 位向量赋值
	s_mov_b64 s[30:31], 0  ; 64 位标量赋值
.LBB0_55:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s34, s0, s30  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s35, s1, s31  ; 带进位加，64 位加法的高 32 位
	global_load_ubyte v14, v29, s[34:35]  ; 逐字节读格式串
	s_add_u32 s30, s30, 1  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s31, s31, 0  ; 带进位加，64 位加法的高 32 位
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_and_b32_e32 v28, 0xffff, v14  ; 按位与
	v_lshlrev_b64 v[14:15], s28, v[28:29]  ; 64 位左移
	s_add_u32 s28, s28, 8  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s29, s29, 0  ; 带进位加，64 位加法的高 32 位
	v_or_b32_e32 v12, v14, v12  ; 按位或
	s_cmp_lg_u32 s33, s30  ; 比较「不等」，结果写到 scc
	v_or_b32_e32 v13, v15, v13  ; 按位或
	s_cbranch_scc1 .LBB0_55  ; scc = 1 就跳转
	s_branch .LBB0_58  ; 无条件跳转
.LBB0_56:                               ;   in Loop: Header=BB0_28 Depth=1
                                        ; implicit-def: $vgpr12_vgpr13
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b32 s34, 0  ; 标量赋值
	s_branch .LBB0_59  ; 无条件跳转
.LBB0_57:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[12:13], 0  ; 64 位向量赋值
.LBB0_58:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[28:29], s[0:1]  ; 64 位标量赋值
	s_mov_b32 s34, 0  ; 标量赋值
	s_cbranch_execnz .LBB0_60  ; 还有 lane 没完成（exec ≠ 0）就跳回去
.LBB0_59:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	global_load_dwordx2 v[12:13], v29, s[0:1]  ; 读 printf 缓冲区（和主机共享的内存）
	s_add_i32 s34, s33, -8  ; 加法
.LBB0_60:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_add_u32 s0, s28, 8  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s1, s29, 0  ; 带进位加，64 位加法的高 32 位
	s_cmp_gt_u32 s34, 7  ; 比较「大于」，结果写到 scc
	s_cbranch_scc1 .LBB0_64  ; scc = 1 就跳转
; %bb.61:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s34, 0  ; 比较「相等」，结果写到 scc
	s_cbranch_scc1 .LBB0_65  ; scc = 1 就跳转
; %bb.62:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b64 s[0:1], 0  ; 64 位标量赋值
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_mov_b64_e32 v[14:15], 0  ; 64 位向量赋值
	s_mov_b64 s[30:31], 0  ; 64 位标量赋值
.LBB0_63:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s36, s28, s30  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s37, s29, s31  ; 带进位加，64 位加法的高 32 位
	global_load_ubyte v16, v29, s[36:37]  ; 逐字节读格式串
	s_add_u32 s30, s30, 1  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s31, s31, 0  ; 带进位加，64 位加法的高 32 位
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_and_b32_e32 v28, 0xffff, v16  ; 按位与
	v_lshlrev_b64 v[16:17], s0, v[28:29]  ; 64 位左移
	s_add_u32 s0, s0, 8  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s1, s1, 0  ; 带进位加，64 位加法的高 32 位
	v_or_b32_e32 v14, v16, v14  ; 按位或
	s_cmp_lg_u32 s34, s30  ; 比较「不等」，结果写到 scc
	v_or_b32_e32 v15, v17, v15  ; 按位或
	s_cbranch_scc1 .LBB0_63  ; scc = 1 就跳转
	s_branch .LBB0_66  ; 无条件跳转
.LBB0_64:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b32 s33, 0  ; 标量赋值
	s_branch .LBB0_67  ; 无条件跳转
.LBB0_65:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[14:15], 0  ; 64 位向量赋值
.LBB0_66:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[0:1], s[28:29]  ; 64 位标量赋值
	s_mov_b32 s33, 0  ; 标量赋值
	s_cbranch_execnz .LBB0_68  ; 还有 lane 没完成（exec ≠ 0）就跳回去
.LBB0_67:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	global_load_dwordx2 v[14:15], v29, s[28:29]  ; 读 printf 缓冲区（和主机共享的内存）
	s_add_i32 s33, s34, -8  ; 加法
.LBB0_68:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_add_u32 s28, s0, 8  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s29, s1, 0  ; 带进位加，64 位加法的高 32 位
	s_cmp_gt_u32 s33, 7  ; 比较「大于」，结果写到 scc
	s_cbranch_scc1 .LBB0_72  ; scc = 1 就跳转
; %bb.69:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s33, 0  ; 比较「相等」，结果写到 scc
	s_cbranch_scc1 .LBB0_73  ; scc = 1 就跳转
; %bb.70:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b64 s[28:29], 0  ; 64 位标量赋值
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_mov_b64_e32 v[16:17], 0  ; 64 位向量赋值
	s_mov_b64 s[30:31], 0  ; 64 位标量赋值
.LBB0_71:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_u32 s34, s0, s30  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s35, s1, s31  ; 带进位加，64 位加法的高 32 位
	global_load_ubyte v18, v29, s[34:35]  ; 逐字节读格式串
	s_add_u32 s30, s30, 1  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s31, s31, 0  ; 带进位加，64 位加法的高 32 位
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_and_b32_e32 v28, 0xffff, v18  ; 按位与
	v_lshlrev_b64 v[18:19], s28, v[28:29]  ; 64 位左移
	s_add_u32 s28, s28, 8  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s29, s29, 0  ; 带进位加，64 位加法的高 32 位
	v_or_b32_e32 v16, v18, v16  ; 按位或
	s_cmp_lg_u32 s33, s30  ; 比较「不等」，结果写到 scc
	v_or_b32_e32 v17, v19, v17  ; 按位或
	s_cbranch_scc1 .LBB0_71  ; scc = 1 就跳转
	s_branch .LBB0_74  ; 无条件跳转
.LBB0_72:                               ;   in Loop: Header=BB0_28 Depth=1
                                        ; implicit-def: $vgpr16_vgpr17
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b32 s34, 0  ; 标量赋值
	s_branch .LBB0_75  ; 无条件跳转
.LBB0_73:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[16:17], 0  ; 64 位向量赋值
.LBB0_74:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b64 s[28:29], s[0:1]  ; 64 位标量赋值
	s_mov_b32 s34, 0  ; 标量赋值
	s_cbranch_execnz .LBB0_76  ; 还有 lane 没完成（exec ≠ 0）就跳回去
.LBB0_75:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	global_load_dwordx2 v[16:17], v29, s[0:1]  ; 读 printf 缓冲区（和主机共享的内存）
	s_add_i32 s34, s33, -8  ; 加法
.LBB0_76:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_cmp_gt_u32 s34, 7  ; 比较「大于」，结果写到 scc
	s_cbranch_scc1 .LBB0_80  ; scc = 1 就跳转
; %bb.77:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_eq_u32 s34, 0  ; 比较「相等」，结果写到 scc
	s_cbranch_scc1 .LBB0_81  ; scc = 1 就跳转
; %bb.78:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b64 s[0:1], 0  ; 64 位标量赋值
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_mov_b64_e32 v[18:19], 0  ; 64 位向量赋值
	s_mov_b64 s[30:31], s[28:29]  ; 64 位标量赋值
.LBB0_79:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_ubyte v20, v29, s[30:31]  ; 逐字节读格式串
	s_add_i32 s34, s34, -1  ; 加法
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_and_b32_e32 v28, 0xffff, v20  ; 按位与
	v_lshlrev_b64 v[20:21], s0, v[28:29]  ; 64 位左移
	s_add_u32 s0, s0, 8  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s1, s1, 0  ; 带进位加，64 位加法的高 32 位
	s_add_u32 s30, s30, 1  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s31, s31, 0  ; 带进位加，64 位加法的高 32 位
	v_or_b32_e32 v18, v20, v18  ; 按位或
	s_cmp_lg_u32 s34, 0  ; 比较「不等」，结果写到 scc
	v_or_b32_e32 v19, v21, v19  ; 按位或
	s_cbranch_scc1 .LBB0_79  ; scc = 1 就跳转
	s_branch .LBB0_82  ; 无条件跳转
.LBB0_80:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_branch .LBB0_83  ; 无条件跳转
.LBB0_81:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[18:19], 0  ; 64 位向量赋值
.LBB0_82:                               ;   in Loop: Header=BB0_28 Depth=1
	s_cbranch_execnz .LBB0_84  ; 还有 lane 没完成（exec ≠ 0）就跳回去
.LBB0_83:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	global_load_dwordx2 v[18:19], v29, s[28:29]  ; 读 printf 缓冲区（和主机共享的内存）
.LBB0_84:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_readfirstlane_b32 s0, v1  ; 取第一个有效 lane 的值，放进标量寄存器
	v_mov_b64_e32 v[30:31], 0  ; 64 位向量赋值
	s_nop 0  ; 空几个周期，避开硬件的时序冲突
	v_cmp_eq_u32_e64 s[0:1], s0, v1  ; 比较「相等」，结果写到一对 SGPR
	s_and_saveexec_b64 s[28:29], s[0:1]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_90  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.85:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[22:23], v29, s[18:19] offset:24 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	global_load_dwordx2 v[20:21], v29, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[30:31], v29, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(1)  ; 等显存读写完成
	v_and_b32_e32 v20, v20, v22  ; 按位与
	v_and_b32_e32 v21, v21, v23  ; 按位与
	v_mul_lo_u32 v21, v21, 24  ; 乘法，取低 32 位
	v_mul_hi_u32 v27, v20, 24  ; 乘法，取高 32 位
	v_add_u32_e32 v21, v27, v21  ; 加法
	v_mul_lo_u32 v20, v20, 24  ; 乘法，取低 32 位
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_lshl_add_u64 v[20:21], v[30:31], 0, v[20:21]  ; 64 位：左移后相加（这里多用来算地址）
	global_load_dwordx2 v[20:21], v[20:21], off sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[30:31], v29, v[20:23], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	v_cmp_ne_u64_e32 vcc, v[30:31], v[22:23]  ; 64 位比较「不等」，结果写到 vcc
	s_and_saveexec_b64 s[30:31], vcc  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_89  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.86:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b64 s[34:35], 0  ; 64 位标量赋值
.LBB0_87:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	s_sleep 1  ; 睡一会儿再查
	global_load_dwordx2 v[20:21], v29, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[32:33], v29, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	v_mov_b64_e32 v[22:23], v[30:31]  ; 64 位向量赋值
	s_waitcnt vmcnt(1)  ; 等显存读写完成
	v_and_b32_e32 v20, v20, v22  ; 按位与
	v_and_b32_e32 v27, v21, v23  ; 按位与
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_mad_u64_u32 v[20:21], s[36:37], v20, 24, v[32:33]  ; 乘加：32 位 × 32 位得到 64 位，再加一个 64 位数
	v_mov_b32_e32 v28, v21  ; 向量赋值
	v_mad_u64_u32 v[30:31], s[36:37], v27, 24, v[28:29]  ; 乘加：32 位 × 32 位得到 64 位，再加一个 64 位数
	v_mov_b32_e32 v21, v30  ; 向量赋值
	global_load_dwordx2 v[20:21], v[20:21], off sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[30:31], v29, v[20:23], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	v_cmp_eq_u64_e32 vcc, v[30:31], v[22:23]  ; 64 位比较「相等」，结果写到 vcc
	s_or_b64 s[34:35], vcc, s[34:35]  ; 64 位按位或
	s_andn2_b64 exec, exec, s[34:35]  ; 把已经完成的 lane 从 exec 里去掉（循环用）
	s_cbranch_execnz .LBB0_87  ; 还有 lane 没完成（exec ≠ 0）就跳回去
; %bb.88:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[34:35]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
.LBB0_89:                               ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[30:31]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
.LBB0_90:                               ;   in Loop: Header=BB0_28 Depth=1
	s_or_b64 exec, exec, s[28:29]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	global_load_dwordx2 v[32:33], v29, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx4 v[20:23], v29, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	v_readfirstlane_b32 s29, v31  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s28, v30  ; 取第一个有效 lane 的值，放进标量寄存器
	s_mov_b64 s[30:31], exec  ; 64 位标量赋值
	s_waitcnt vmcnt(1)  ; 等显存读写完成
	v_readfirstlane_b32 s34, v32  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s35, v33  ; 取第一个有效 lane 的值，放进标量寄存器
	s_and_b64 s[34:35], s[34:35], s[28:29]  ; 64 位按位与
	s_mul_i32 s33, s35, 24  ; 乘法，取低 32 位
	s_mul_hi_u32 s36, s34, 24  ; 乘法，取高 32 位
	s_add_i32 s37, s36, s33  ; 加法
	s_mul_i32 s36, s34, 24  ; 乘法，取低 32 位
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_lshl_add_u64 v[30:31], v[20:21], 0, s[36:37]  ; 64 位：左移后相加（这里多用来算地址）
	s_and_saveexec_b64 s[36:37], s[0:1]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_92  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.91:                               ;   in Loop: Header=BB0_28 Depth=1
	v_mov_b64_e32 v[32:33], s[30:31]  ; 64 位向量赋值
	global_store_dwordx4 v[30:31], v[32:35], off offset:8  ; 写 printf 缓冲区
.LBB0_92:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[36:37]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_lshl_b64 s[30:31], s[34:35], 12  ; 64 位左移
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[30:31]  ; 64 位：左移后相加（这里多用来算地址）
	v_cmp_lt_u64_e64 vcc, s[24:25], 57  ; 64 位比较「小于」，结果写到一对 SGPR
	s_lshl_b32 s30, s26, 2  ; 左移
	s_add_i32 s30, s30, 28  ; 加法
	v_cndmask_b32_e32 v27, 0, v26, vcc  ; 按 vcc 每个 lane 二选一
	v_and_b32_e32 v4, 0xffffff1f, v4  ; 按位与
	s_and_b32 s30, s30, 0x1e0  ; 按位与
	v_or_b32_e32 v4, v4, v27  ; 按位或
	v_or_b32_e32 v4, s30, v4  ; 按位或
	v_readfirstlane_b32 s30, v22  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s31, v23  ; 取第一个有效 lane 的值，放进标量寄存器
	s_nop 4  ; 空几个周期，避开硬件的时序冲突
	global_store_dwordx4 v24, v[4:7], s[30:31]  ; 写 printf 缓冲区
	global_store_dwordx4 v24, v[8:11], s[30:31] offset:16  ; 写 printf 缓冲区
	global_store_dwordx4 v24, v[12:15], s[30:31] offset:32  ; 写 printf 缓冲区
	global_store_dwordx4 v24, v[16:19], s[30:31] offset:48  ; 写 printf 缓冲区
	s_and_saveexec_b64 s[30:31], s[0:1]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_100  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.93:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[12:13], v29, s[18:19] offset:32 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[4:5], v29, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	v_mov_b32_e32 v10, s28  ; 向量赋值
	v_mov_b32_e32 v11, s29  ; 向量赋值
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_readfirstlane_b32 s34, v4  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s35, v5  ; 取第一个有效 lane 的值，放进标量寄存器
	s_and_b64 s[34:35], s[34:35], s[28:29]  ; 64 位按位与
	s_mul_i32 s33, s35, 24  ; 乘法，取低 32 位
	s_mul_hi_u32 s35, s34, 24  ; 乘法，取高 32 位
	s_mul_i32 s34, s34, 24  ; 乘法，取低 32 位
	s_add_i32 s35, s35, s33  ; 加法
	v_lshl_add_u64 v[8:9], v[20:21], 0, s[34:35]  ; 64 位：左移后相加（这里多用来算地址）
	global_store_dwordx2 v[8:9], v[12:13], off  ; 写 printf 缓冲区
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[6:7], v29, v[10:13], s[18:19] offset:32 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_ne_u64_e32 vcc, v[6:7], v[12:13]  ; 64 位比较「不等」，结果写到 vcc
	s_and_saveexec_b64 s[34:35], vcc  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_96  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.94:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b64 s[36:37], 0  ; 64 位标量赋值
.LBB0_95:                               ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	s_sleep 1  ; 睡一会儿再查
	global_store_dwordx2 v[8:9], v[6:7], off  ; 写 printf 缓冲区
	v_mov_b32_e32 v4, s28  ; 向量赋值
	v_mov_b32_e32 v5, s29  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[4:5], v29, v[4:7], s[18:19] offset:32 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_eq_u64_e32 vcc, v[4:5], v[6:7]  ; 64 位比较「相等」，结果写到 vcc
	s_or_b64 s[36:37], vcc, s[36:37]  ; 64 位按位或
	v_mov_b64_e32 v[6:7], v[4:5]  ; 64 位向量赋值
	s_andn2_b64 exec, exec, s[36:37]  ; 把已经完成的 lane 从 exec 里去掉（循环用）
	s_cbranch_execnz .LBB0_95  ; 还有 lane 没完成（exec ≠ 0）就跳回去
.LBB0_96:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[34:35]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	global_load_dwordx2 v[4:5], v29, s[18:19] offset:16  ; 读 printf 缓冲区（和主机共享的内存）
	s_mov_b64 s[36:37], exec  ; 64 位标量赋值
	v_mbcnt_lo_u32_b32 v6, s36, 0  ; 数 exec 里排在本 lane 前面的有效 lane（低 32 位）
	v_mbcnt_hi_u32_b32 v6, s37, v6  ; 同上（高 32 位）：结果是本 lane 在有效 lane 里排第几
	v_cmp_eq_u32_e32 vcc, 0, v6  ; 比较「相等」，结果写到 vcc
	s_and_saveexec_b64 s[34:35], vcc  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_98  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.97:                               ;   in Loop: Header=BB0_28 Depth=1
	s_bcnt1_i32_b64 s33, s[36:37]  ; 数 64 位里有几个 1（这里是有几个有效 lane）
	v_mov_b32_e32 v28, s33  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_add_x2 v[4:5], v[28:29], off offset:8 sc1  ; 原子加：更新通知主机用的计数
.LBB0_98:                               ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[34:35]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_load_dwordx2 v[6:7], v[4:5], off offset:16  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_eq_u64_e32 vcc, 0, v[6:7]  ; 64 位比较「相等」，结果写到 vcc
	s_cbranch_vccnz .LBB0_100  ; vcc 不为 0 就跳转
; %bb.99:                               ;   in Loop: Header=BB0_28 Depth=1
	global_load_dword v28, v[4:5], off offset:24  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_readfirstlane_b32 s33, v28  ; 取第一个有效 lane 的值，放进标量寄存器
	s_and_b32 m0, s33, 0xffffff  ; 按位与
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	global_store_dwordx2 v[6:7], v[28:29], off sc0 sc1  ; 写 printf 缓冲区
	s_sendmsg sendmsg(MSG_INTERRUPT)  ; 发中断给主机：有 printf 数据要处理
.LBB0_100:                              ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[30:31]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	v_lshl_add_u64 v[4:5], v[22:23], 0, v[24:25]  ; 64 位：左移后相加（这里多用来算地址）
	s_branch .LBB0_104  ; 无条件跳转
.LBB0_101:                              ;   in Loop: Header=BB0_104 Depth=2
	s_or_b64 exec, exec, s[30:31]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_readfirstlane_b32 s30, v6  ; 取第一个有效 lane 的值，放进标量寄存器
	s_cmp_eq_u32 s30, 0  ; 比较「相等」，结果写到 scc
	s_cbranch_scc1 .LBB0_103  ; scc = 1 就跳转
; %bb.102:                              ;   in Loop: Header=BB0_104 Depth=2
	s_sleep 1  ; 睡一会儿再查
	s_cbranch_execnz .LBB0_104  ; 还有 lane 没完成（exec ≠ 0）就跳回去
	s_branch .LBB0_106  ; 无条件跳转
.LBB0_103:                              ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_branch .LBB0_106  ; 无条件跳转
.LBB0_104:                              ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v6, 1  ; 向量赋值
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	s_and_saveexec_b64 s[30:31], s[0:1]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_101  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.105:                              ;   in Loop: Header=BB0_104 Depth=2
	global_load_dword v6, v[30:31], off offset:20 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	v_and_b32_e32 v6, 1, v6  ; 按位与
	s_branch .LBB0_101  ; 无条件跳转
.LBB0_106:                              ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	global_load_dwordx2 v[4:5], v[4:5], off  ; 读 printf 缓冲区（和主机共享的内存）
	s_and_saveexec_b64 s[30:31], s[0:1]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_27  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.107:                              ;   in Loop: Header=BB0_28 Depth=1
	global_load_dwordx2 v[6:7], v29, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[14:15], v29, s[18:19] offset:24 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[8:9], v29, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(2)  ; 等显存读写完成
	v_readfirstlane_b32 s34, v6  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s35, v7  ; 取第一个有效 lane 的值，放进标量寄存器
	s_add_u32 s33, s34, 1  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s36, s35, 0  ; 带进位加，64 位加法的高 32 位
	s_add_u32 s0, s33, s28  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s1, s36, s29  ; 带进位加，64 位加法的高 32 位
	s_cmp_eq_u64 s[0:1], 0  ; 64 位比较「相等」，结果写到 scc
	s_cselect_b32 s1, s36, s1  ; 按 scc 二选一
	s_cselect_b32 s0, s33, s0  ; 按 scc 二选一
	s_and_b64 s[28:29], s[0:1], s[34:35]  ; 64 位按位与
	s_mul_i32 s29, s29, 24  ; 乘法，取低 32 位
	s_mul_hi_u32 s33, s28, 24  ; 乘法，取高 32 位
	s_mul_i32 s28, s28, 24  ; 乘法，取低 32 位
	s_add_i32 s29, s33, s29  ; 加法
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_lshl_add_u64 v[10:11], v[8:9], 0, s[28:29]  ; 64 位：左移后相加（这里多用来算地址）
	v_mov_b32_e32 v12, s0  ; 向量赋值
	global_store_dwordx2 v[10:11], v[14:15], off  ; 写 printf 缓冲区
	v_mov_b32_e32 v13, s1  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[8:9], v29, v[12:15], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_ne_u64_e32 vcc, v[8:9], v[14:15]  ; 64 位比较「不等」，结果写到 vcc
	s_and_b64 exec, exec, vcc  ; 只让条件成立的 lane 继续
	s_cbranch_execz .LBB0_27  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.108:                              ;   in Loop: Header=BB0_28 Depth=1
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b64 s[28:29], 0  ; 64 位标量赋值
.LBB0_109:                              ;   Parent Loop BB0_28 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	s_sleep 1  ; 睡一会儿再查
	global_store_dwordx2 v[10:11], v[8:9], off  ; 写 printf 缓冲区
	v_mov_b32_e32 v6, s0  ; 向量赋值
	v_mov_b32_e32 v7, s1  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[6:7], v29, v[6:9], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]  ; 64 位比较「相等」，结果写到 vcc
	s_or_b64 s[28:29], vcc, s[28:29]  ; 64 位按位或
	v_mov_b64_e32 v[8:9], v[6:7]  ; 64 位向量赋值
	s_andn2_b64 exec, exec, s[28:29]  ; 把已经完成的 lane 从 exec 里去掉（循环用）
	s_cbranch_execnz .LBB0_109  ; 还有 lane 没完成（exec ≠ 0）就跳回去
	s_branch .LBB0_27  ; 无条件跳转
.LBB0_110:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_branch .LBB0_138  ; 无条件跳转
; ============================================================================================
; （格式串地址为空时走的另一条路，实际不会走到，跳到【2c】）
; ============================================================================================
.LBB0_111:
                                        ; implicit-def: $vgpr4_vgpr5
	s_cbranch_execz .LBB0_138  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.112:
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行：printf("load time: %llu\n", end - start);
	v_readfirstlane_b32 s0, v1  ; 取第一个有效 lane 的值，放进标量寄存器
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_mov_b64_e32 v[4:5], 0  ; 64 位向量赋值
	v_cmp_eq_u32_e64 s[0:1], s0, v1  ; 比较「相等」，结果写到一对 SGPR
	s_and_saveexec_b64 s[4:5], s[0:1]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_118  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.113:
	v_mov_b32_e32 v6, 0  ; 向量赋值
	global_load_dwordx2 v[10:11], v6, s[18:19] offset:24 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	global_load_dwordx2 v[4:5], v6, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[8:9], v6, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(1)  ; 等显存读写完成
	v_and_b32_e32 v4, v4, v10  ; 按位与
	v_and_b32_e32 v5, v5, v11  ; 按位与
	v_mul_lo_u32 v5, v5, 24  ; 乘法，取低 32 位
	v_mul_hi_u32 v7, v4, 24  ; 乘法，取高 32 位
	v_add_u32_e32 v5, v7, v5  ; 加法
	v_mul_lo_u32 v4, v4, 24  ; 乘法，取低 32 位
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[4:5]  ; 64 位：左移后相加（这里多用来算地址）
	global_load_dwordx2 v[8:9], v[4:5], off sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[4:5], v6, v[8:11], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	v_cmp_ne_u64_e32 vcc, v[4:5], v[10:11]  ; 64 位比较「不等」，结果写到 vcc
	s_and_saveexec_b64 s[24:25], vcc  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_117  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.114:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b64 s[26:27], 0  ; 64 位标量赋值
.LBB0_115:                              ; =>This Inner Loop Header: Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	s_sleep 1  ; 睡一会儿再查
	global_load_dwordx2 v[8:9], v6, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[12:13], v6, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	v_mov_b64_e32 v[10:11], v[4:5]  ; 64 位向量赋值
	s_waitcnt vmcnt(1)  ; 等显存读写完成
	v_and_b32_e32 v4, v8, v10  ; 按位与
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_mad_u64_u32 v[4:5], s[28:29], v4, 24, v[12:13]  ; 乘加：32 位 × 32 位得到 64 位，再加一个 64 位数
	v_and_b32_e32 v7, v9, v11  ; 按位与
	v_mov_b32_e32 v8, v5  ; 向量赋值
	v_mad_u64_u32 v[8:9], s[28:29], v7, 24, v[8:9]  ; 乘加：32 位 × 32 位得到 64 位，再加一个 64 位数
	v_mov_b32_e32 v5, v8  ; 向量赋值
	global_load_dwordx2 v[8:9], v[4:5], off sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[4:5], v6, v[8:11], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	v_cmp_eq_u64_e32 vcc, v[4:5], v[10:11]  ; 64 位比较「相等」，结果写到 vcc
	s_or_b64 s[26:27], vcc, s[26:27]  ; 64 位按位或
	s_andn2_b64 exec, exec, s[26:27]  ; 把已经完成的 lane 从 exec 里去掉（循环用）
	s_cbranch_execnz .LBB0_115  ; 还有 lane 没完成（exec ≠ 0）就跳回去
; %bb.116:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[26:27]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
.LBB0_117:
	s_or_b64 exec, exec, s[24:25]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
.LBB0_118:
	s_or_b64 exec, exec, s[4:5]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	v_mov_b32_e32 v25, 0  ; 向量赋值
	global_load_dwordx2 v[10:11], v25, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx4 v[6:9], v25, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	v_readfirstlane_b32 s5, v5  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s4, v4  ; 取第一个有效 lane 的值，放进标量寄存器
	s_mov_b64 s[24:25], exec  ; 64 位标量赋值
	s_waitcnt vmcnt(1)  ; 等显存读写完成
	v_readfirstlane_b32 s26, v10  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s27, v11  ; 取第一个有效 lane 的值，放进标量寄存器
	s_and_b64 s[26:27], s[26:27], s[4:5]  ; 64 位按位与
	s_mul_i32 s28, s27, 24  ; 乘法，取低 32 位
	s_mul_hi_u32 s29, s26, 24  ; 乘法，取高 32 位
	s_add_i32 s29, s29, s28  ; 加法
	s_mul_i32 s28, s26, 24  ; 乘法，取低 32 位
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_lshl_add_u64 v[10:11], v[6:7], 0, s[28:29]  ; 64 位：左移后相加（这里多用来算地址）
	s_and_saveexec_b64 s[28:29], s[0:1]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_120  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.119:
	v_mov_b64_e32 v[12:13], s[24:25]  ; 64 位向量赋值
	v_mov_b32_e32 v14, 2  ; 向量赋值
	v_mov_b32_e32 v15, 1  ; 向量赋值
	global_store_dwordx4 v[10:11], v[12:15], off offset:8  ; 写 printf 缓冲区
.LBB0_120:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[28:29]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_lshl_b64 s[24:25], s[26:27], 12  ; 64 位左移
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[24:25]  ; 64 位：左移后相加（这里多用来算地址）
	s_movk_i32 s24, 0xff1f  ; 把 16 位常数赋给标量寄存器
	v_and_or_b32 v2, v2, s24, 32  ; (a & b) | c
	s_mov_b32 s24, 0  ; 标量赋值
	v_mov_b32_e32 v4, v25  ; 向量赋值
	v_mov_b32_e32 v5, v25  ; 向量赋值
	v_readfirstlane_b32 s28, v8  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s29, v9  ; 取第一个有效 lane 的值，放进标量寄存器
	s_mov_b32 s25, s24  ; 标量赋值
	s_mov_b32 s26, s24  ; 标量赋值
	s_mov_b32 s27, s24  ; 标量赋值
	s_nop 1  ; 空几个周期，避开硬件的时序冲突
	global_store_dwordx4 v24, v[2:5], s[28:29]  ; 写 printf 缓冲区
	s_nop 1  ; 空几个周期，避开硬件的时序冲突
	v_mov_b64_e32 v[2:3], s[24:25]  ; 64 位向量赋值
	v_mov_b64_e32 v[4:5], s[26:27]  ; 64 位向量赋值
	global_store_dwordx4 v24, v[2:5], s[28:29] offset:16  ; 写 printf 缓冲区
	global_store_dwordx4 v24, v[2:5], s[28:29] offset:32  ; 写 printf 缓冲区
	global_store_dwordx4 v24, v[2:5], s[28:29] offset:48  ; 写 printf 缓冲区
	s_and_saveexec_b64 s[24:25], s[0:1]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_128  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.121:
	v_mov_b32_e32 v12, 0  ; 向量赋值
	global_load_dwordx2 v[16:17], v12, s[18:19] offset:32 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[2:3], v12, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	v_mov_b32_e32 v14, s4  ; 向量赋值
	v_mov_b32_e32 v15, s5  ; 向量赋值
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_readfirstlane_b32 s26, v2  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s27, v3  ; 取第一个有效 lane 的值，放进标量寄存器
	s_and_b64 s[26:27], s[26:27], s[4:5]  ; 64 位按位与
	s_mul_i32 s27, s27, 24  ; 乘法，取低 32 位
	s_mul_hi_u32 s28, s26, 24  ; 乘法，取高 32 位
	s_mul_i32 s26, s26, 24  ; 乘法，取低 32 位
	s_add_i32 s27, s28, s27  ; 加法
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[26:27]  ; 64 位：左移后相加（这里多用来算地址）
	global_store_dwordx2 v[6:7], v[16:17], off  ; 写 printf 缓冲区
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[4:5], v12, v[14:17], s[18:19] offset:32 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_ne_u64_e32 vcc, v[4:5], v[16:17]  ; 64 位比较「不等」，结果写到 vcc
	s_and_saveexec_b64 s[26:27], vcc  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_124  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.122:
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b64 s[28:29], 0  ; 64 位标量赋值
.LBB0_123:                              ; =>This Inner Loop Header: Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	s_sleep 1  ; 睡一会儿再查
	global_store_dwordx2 v[6:7], v[4:5], off  ; 写 printf 缓冲区
	v_mov_b32_e32 v2, s4  ; 向量赋值
	v_mov_b32_e32 v3, s5  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[2:3], v12, v[2:5], s[18:19] offset:32 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]  ; 64 位比较「相等」，结果写到 vcc
	s_or_b64 s[28:29], vcc, s[28:29]  ; 64 位按位或
	v_mov_b64_e32 v[4:5], v[2:3]  ; 64 位向量赋值
	s_andn2_b64 exec, exec, s[28:29]  ; 把已经完成的 lane 从 exec 里去掉（循环用）
	s_cbranch_execnz .LBB0_123  ; 还有 lane 没完成（exec ≠ 0）就跳回去
.LBB0_124:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[26:27]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_mov_b32_e32 v5, 0  ; 向量赋值
	global_load_dwordx2 v[2:3], v5, s[18:19] offset:16  ; 读 printf 缓冲区（和主机共享的内存）
	s_mov_b64 s[26:27], exec  ; 64 位标量赋值
	v_mbcnt_lo_u32_b32 v4, s26, 0  ; 数 exec 里排在本 lane 前面的有效 lane（低 32 位）
	v_mbcnt_hi_u32_b32 v4, s27, v4  ; 同上（高 32 位）：结果是本 lane 在有效 lane 里排第几
	v_cmp_eq_u32_e32 vcc, 0, v4  ; 比较「相等」，结果写到 vcc
	s_and_saveexec_b64 s[28:29], vcc  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_126  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.125:
	s_bcnt1_i32_b64 s26, s[26:27]  ; 数 64 位里有几个 1（这里是有几个有效 lane）
	v_mov_b32_e32 v4, s26  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_add_x2 v[2:3], v[4:5], off offset:8 sc1  ; 原子加：更新通知主机用的计数
.LBB0_126:
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[28:29]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_load_dwordx2 v[4:5], v[2:3], off offset:16  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_eq_u64_e32 vcc, 0, v[4:5]  ; 64 位比较「相等」，结果写到 vcc
	s_cbranch_vccnz .LBB0_128  ; vcc 不为 0 就跳转
; %bb.127:
	global_load_dword v2, v[2:3], off offset:24  ; 读 printf 缓冲区（和主机共享的内存）
	v_mov_b32_e32 v3, 0  ; 向量赋值
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_readfirstlane_b32 s26, v2  ; 取第一个有效 lane 的值，放进标量寄存器
	s_and_b32 m0, s26, 0xffffff  ; 按位与
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	global_store_dwordx2 v[4:5], v[2:3], off sc0 sc1  ; 写 printf 缓冲区
	s_sendmsg sendmsg(MSG_INTERRUPT)  ; 发中断给主机：有 printf 数据要处理
.LBB0_128:
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[24:25]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	v_lshl_add_u64 v[2:3], v[8:9], 0, v[24:25]  ; 64 位：左移后相加（这里多用来算地址）
	s_branch .LBB0_132  ; 无条件跳转
.LBB0_129:                              ;   in Loop: Header=BB0_132 Depth=1
	s_or_b64 exec, exec, s[24:25]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_readfirstlane_b32 s24, v4  ; 取第一个有效 lane 的值，放进标量寄存器
	s_cmp_eq_u32 s24, 0  ; 比较「相等」，结果写到 scc
	s_cbranch_scc1 .LBB0_131  ; scc = 1 就跳转
; %bb.130:                              ;   in Loop: Header=BB0_132 Depth=1
	s_sleep 1  ; 睡一会儿再查
	s_cbranch_execnz .LBB0_132  ; 还有 lane 没完成（exec ≠ 0）就跳回去
	s_branch .LBB0_134  ; 无条件跳转
.LBB0_131:
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_branch .LBB0_134  ; 无条件跳转
.LBB0_132:                              ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v4, 1  ; 向量赋值
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	s_and_saveexec_b64 s[24:25], s[0:1]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_129  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.133:                              ;   in Loop: Header=BB0_132 Depth=1
	global_load_dword v4, v[10:11], off offset:20 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	v_and_b32_e32 v4, 1, v4  ; 按位与
	s_branch .LBB0_129  ; 无条件跳转
.LBB0_134:
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	global_load_dwordx2 v[4:5], v[2:3], off  ; 读 printf 缓冲区（和主机共享的内存）
	s_and_saveexec_b64 s[24:25], s[0:1]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_137  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.135:
	v_mov_b32_e32 v10, 0  ; 向量赋值
	global_load_dwordx2 v[2:3], v10, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[14:15], v10, s[18:19] offset:24 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[6:7], v10, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(2)  ; 等显存读写完成
	v_readfirstlane_b32 s26, v2  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s27, v3  ; 取第一个有效 lane 的值，放进标量寄存器
	s_add_u32 s28, s26, 1  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s29, s27, 0  ; 带进位加，64 位加法的高 32 位
	s_add_u32 s0, s28, s4  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s1, s29, s5  ; 带进位加，64 位加法的高 32 位
	s_cmp_eq_u64 s[0:1], 0  ; 64 位比较「相等」，结果写到 scc
	s_cselect_b32 s1, s29, s1  ; 按 scc 二选一
	s_cselect_b32 s0, s28, s0  ; 按 scc 二选一
	s_and_b64 s[4:5], s[0:1], s[26:27]  ; 64 位按位与
	s_mul_i32 s5, s5, 24  ; 乘法，取低 32 位
	s_mul_hi_u32 s26, s4, 24  ; 乘法，取高 32 位
	s_mul_i32 s4, s4, 24  ; 乘法，取低 32 位
	s_add_i32 s5, s26, s5  ; 加法
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_lshl_add_u64 v[2:3], v[6:7], 0, s[4:5]  ; 64 位：左移后相加（这里多用来算地址）
	v_mov_b32_e32 v12, s0  ; 向量赋值
	global_store_dwordx2 v[2:3], v[14:15], off  ; 写 printf 缓冲区
	v_mov_b32_e32 v13, s1  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[8:9], v10, v[12:15], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_mov_b64 s[4:5], 0  ; 64 位标量赋值
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_ne_u64_e32 vcc, v[8:9], v[14:15]  ; 64 位比较「不等」，结果写到 vcc
	s_and_b64 exec, exec, vcc  ; 只让条件成立的 lane 继续
	s_cbranch_execz .LBB0_137  ; 没有 lane 要执行（exec = 0）就跳过这段
.LBB0_136:                              ; =>This Inner Loop Header: Depth=1
	s_sleep 1  ; 睡一会儿再查
	global_store_dwordx2 v[2:3], v[8:9], off  ; 写 printf 缓冲区
	v_mov_b32_e32 v6, s0  ; 向量赋值
	v_mov_b32_e32 v7, s1  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[6:7], v10, v[6:9], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_eq_u64_e32 vcc, v[6:7], v[8:9]  ; 64 位比较「相等」，结果写到 vcc
	s_or_b64 s[4:5], vcc, s[4:5]  ; 64 位按位或
	v_mov_b64_e32 v[8:9], v[6:7]  ; 64 位向量赋值
	s_andn2_b64 exec, exec, s[4:5]  ; 把已经完成的 lane 从 exec 里去掉（循环用）
	s_cbranch_execnz .LBB0_136  ; 还有 lane 没完成（exec ≠ 0）就跳回去
.LBB0_137:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[24:25]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
; ============================================================================================
; 【2c】第 3 轮：送参数 end - start，并标记这是最后一段，主机收到后打印
; ============================================================================================
.LBB0_138:
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行：printf("load time: %llu\n", end - start);
	v_readfirstlane_b32 s0, v1  ; 取第一个有效 lane 的值，放进标量寄存器
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_mov_b64_e32 v[2:3], 0  ; 64 位向量赋值
	v_cmp_eq_u32_e64 s[0:1], s0, v1  ; 比较「相等」，结果写到一对 SGPR
	s_and_saveexec_b64 s[4:5], s[0:1]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_144  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.139:
	v_mov_b32_e32 v1, 0  ; 向量赋值
	global_load_dwordx2 v[8:9], v1, s[18:19] offset:24 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	global_load_dwordx2 v[2:3], v1, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[6:7], v1, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(1)  ; 等显存读写完成
	v_and_b32_e32 v2, v2, v8  ; 按位与
	v_and_b32_e32 v3, v3, v9  ; 按位与
	v_mul_lo_u32 v3, v3, 24  ; 乘法，取低 32 位
	v_mul_hi_u32 v10, v2, 24  ; 乘法，取高 32 位
	v_add_u32_e32 v3, v10, v3  ; 加法
	v_mul_lo_u32 v2, v2, 24  ; 乘法，取低 32 位
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_lshl_add_u64 v[2:3], v[6:7], 0, v[2:3]  ; 64 位：左移后相加（这里多用来算地址）
	global_load_dwordx2 v[6:7], v[2:3], off sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[2:3], v1, v[6:9], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	v_cmp_ne_u64_e32 vcc, v[2:3], v[8:9]  ; 64 位比较「不等」，结果写到 vcc
	s_and_saveexec_b64 s[24:25], vcc  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_143  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.140:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b64 s[26:27], 0  ; 64 位标量赋值
.LBB0_141:                              ; =>This Inner Loop Header: Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	s_sleep 1  ; 睡一会儿再查
	global_load_dwordx2 v[6:7], v1, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[10:11], v1, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	v_mov_b64_e32 v[8:9], v[2:3]  ; 64 位向量赋值
	s_waitcnt vmcnt(1)  ; 等显存读写完成
	v_and_b32_e32 v2, v6, v8  ; 按位与
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_mad_u64_u32 v[2:3], s[28:29], v2, 24, v[10:11]  ; 乘加：32 位 × 32 位得到 64 位，再加一个 64 位数
	v_and_b32_e32 v7, v7, v9  ; 按位与
	v_mov_b32_e32 v6, v3  ; 向量赋值
	v_mad_u64_u32 v[6:7], s[28:29], v7, 24, v[6:7]  ; 乘加：32 位 × 32 位得到 64 位，再加一个 64 位数
	v_mov_b32_e32 v3, v6  ; 向量赋值
	global_load_dwordx2 v[6:7], v[2:3], off sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[2:3], v1, v[6:9], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	v_cmp_eq_u64_e32 vcc, v[2:3], v[8:9]  ; 64 位比较「相等」，结果写到 vcc
	s_or_b64 s[26:27], vcc, s[26:27]  ; 64 位按位或
	s_andn2_b64 exec, exec, s[26:27]  ; 把已经完成的 lane 从 exec 里去掉（循环用）
	s_cbranch_execnz .LBB0_141  ; 还有 lane 没完成（exec ≠ 0）就跳回去
; %bb.142:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[26:27]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
.LBB0_143:
	s_or_b64 exec, exec, s[24:25]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
.LBB0_144:
	s_or_b64 exec, exec, s[4:5]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	v_mov_b32_e32 v1, 0  ; 向量赋值
	global_load_dwordx2 v[6:7], v1, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx4 v[8:11], v1, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	v_readfirstlane_b32 s5, v3  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s4, v2  ; 取第一个有效 lane 的值，放进标量寄存器
	s_mov_b64 s[24:25], exec  ; 64 位标量赋值
	s_waitcnt vmcnt(1)  ; 等显存读写完成
	v_readfirstlane_b32 s26, v6  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s27, v7  ; 取第一个有效 lane 的值，放进标量寄存器
	s_and_b64 s[26:27], s[26:27], s[4:5]  ; 64 位按位与
	s_mul_i32 s28, s27, 24  ; 乘法，取低 32 位
	s_mul_hi_u32 s29, s26, 24  ; 乘法，取高 32 位
	s_add_i32 s29, s29, s28  ; 加法
	s_mul_i32 s28, s26, 24  ; 乘法，取低 32 位
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_lshl_add_u64 v[12:13], v[8:9], 0, s[28:29]  ; 64 位：左移后相加（这里多用来算地址）
	s_and_saveexec_b64 s[28:29], s[0:1]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_146  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.145:
	v_mov_b64_e32 v[14:15], s[24:25]  ; 64 位向量赋值
	v_mov_b32_e32 v16, 2  ; 向量赋值
	v_mov_b32_e32 v17, 1  ; 向量赋值
	global_store_dwordx4 v[12:13], v[14:17], off offset:8  ; 写 printf 缓冲区
.LBB0_146:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[28:29]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_lshl_b64 s[24:25], s[26:27], 12  ; 64 位左移
	.loc	2 27 37                         ; ./v0_naive.h:27:37  → 第 27 行
	s_sub_u32 s20, s22, s20  ; end - start 的低 32 位：printf 要打印的参数（第 27 行）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_lshl_add_u64 v[2:3], v[10:11], 0, s[24:25]  ; 64 位：左移后相加（这里多用来算地址）
	s_movk_i32 s24, 0xff1d  ; 把 16 位常数赋给标量寄存器
	.loc	2 27 37                         ; ./v0_naive.h:27:37  → 第 27 行
	s_subb_u32 s21, s23, s21  ; end - start 的高 32 位（带借位）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_mov_b32_e32 v6, s20  ; 向量赋值
	s_mov_b32 s20, 0  ; 标量赋值
	v_and_or_b32 v4, v4, s24, 34  ; (a & b) | c
	v_mov_b32_e32 v7, s21  ; 向量赋值
	v_readfirstlane_b32 s24, v2  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s25, v3  ; 取第一个有效 lane 的值，放进标量寄存器
	s_mov_b32 s21, s20  ; 标量赋值
	s_mov_b32 s22, s20  ; 标量赋值
	s_mov_b32 s23, s20  ; 标量赋值
	s_nop 1  ; 空几个周期，避开硬件的时序冲突
	global_store_dwordx4 v24, v[4:7], s[24:25]  ; 写 printf 缓冲区
	v_mov_b64_e32 v[2:3], s[20:21]  ; 64 位向量赋值
	s_nop 0  ; 空几个周期，避开硬件的时序冲突
	v_mov_b64_e32 v[4:5], s[22:23]  ; 64 位向量赋值
	global_store_dwordx4 v24, v[2:5], s[24:25] offset:16  ; 写 printf 缓冲区
	global_store_dwordx4 v24, v[2:5], s[24:25] offset:32  ; 写 printf 缓冲区
	global_store_dwordx4 v24, v[2:5], s[24:25] offset:48  ; 写 printf 缓冲区
	s_and_saveexec_b64 s[20:21], s[0:1]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_154  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.147:
	v_mov_b32_e32 v1, 0  ; 向量赋值
	global_load_dwordx2 v[16:17], v1, s[18:19] offset:32 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[2:3], v1, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	v_mov_b32_e32 v14, s4  ; 向量赋值
	v_mov_b32_e32 v15, s5  ; 向量赋值
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_readfirstlane_b32 s22, v2  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s23, v3  ; 取第一个有效 lane 的值，放进标量寄存器
	s_and_b64 s[22:23], s[22:23], s[4:5]  ; 64 位按位与
	s_mul_i32 s23, s23, 24  ; 乘法，取低 32 位
	s_mul_hi_u32 s24, s22, 24  ; 乘法，取高 32 位
	s_mul_i32 s22, s22, 24  ; 乘法，取低 32 位
	s_add_i32 s23, s24, s23  ; 加法
	v_lshl_add_u64 v[6:7], v[8:9], 0, s[22:23]  ; 64 位：左移后相加（这里多用来算地址）
	global_store_dwordx2 v[6:7], v[16:17], off  ; 写 printf 缓冲区
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[4:5], v1, v[14:17], s[18:19] offset:32 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_ne_u64_e32 vcc, v[4:5], v[16:17]  ; 64 位比较「不等」，结果写到 vcc
	s_and_saveexec_b64 s[22:23], vcc  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_150  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.148:
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_mov_b64 s[24:25], 0  ; 64 位标量赋值
.LBB0_149:                              ; =>This Inner Loop Header: Depth=1
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	s_sleep 1  ; 睡一会儿再查
	global_store_dwordx2 v[6:7], v[4:5], off  ; 写 printf 缓冲区
	v_mov_b32_e32 v2, s4  ; 向量赋值
	v_mov_b32_e32 v3, s5  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[2:3], v1, v[2:5], s[18:19] offset:32 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]  ; 64 位比较「相等」，结果写到 vcc
	s_or_b64 s[24:25], vcc, s[24:25]  ; 64 位按位或
	v_mov_b64_e32 v[4:5], v[2:3]  ; 64 位向量赋值
	s_andn2_b64 exec, exec, s[24:25]  ; 把已经完成的 lane 从 exec 里去掉（循环用）
	s_cbranch_execnz .LBB0_149  ; 还有 lane 没完成（exec ≠ 0）就跳回去
.LBB0_150:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[22:23]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_mov_b32_e32 v5, 0  ; 向量赋值
	global_load_dwordx2 v[2:3], v5, s[18:19] offset:16  ; 读 printf 缓冲区（和主机共享的内存）
	s_mov_b64 s[22:23], exec  ; 64 位标量赋值
	v_mbcnt_lo_u32_b32 v1, s22, 0  ; 数 exec 里排在本 lane 前面的有效 lane（低 32 位）
	v_mbcnt_hi_u32_b32 v1, s23, v1  ; 同上（高 32 位）：结果是本 lane 在有效 lane 里排第几
	v_cmp_eq_u32_e32 vcc, 0, v1  ; 比较「相等」，结果写到 vcc
	s_and_saveexec_b64 s[24:25], vcc  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_152  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.151:
	s_bcnt1_i32_b64 s22, s[22:23]  ; 数 64 位里有几个 1（这里是有几个有效 lane）
	v_mov_b32_e32 v4, s22  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_add_x2 v[2:3], v[4:5], off offset:8 sc1  ; 原子加：更新通知主机用的计数
.LBB0_152:
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[24:25]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_load_dwordx2 v[4:5], v[2:3], off offset:16  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_eq_u64_e32 vcc, 0, v[4:5]  ; 64 位比较「相等」，结果写到 vcc
	s_cbranch_vccnz .LBB0_154  ; vcc 不为 0 就跳转
; %bb.153:
	global_load_dword v2, v[2:3], off offset:24  ; 读 printf 缓冲区（和主机共享的内存）
	v_mov_b32_e32 v3, 0  ; 向量赋值
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_readfirstlane_b32 s22, v2  ; 取第一个有效 lane 的值，放进标量寄存器
	s_and_b32 m0, s22, 0xffffff  ; 按位与
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	global_store_dwordx2 v[4:5], v[2:3], off sc0 sc1  ; 写 printf 缓冲区
	s_sendmsg sendmsg(MSG_INTERRUPT)  ; 发中断给主机：有 printf 数据要处理
.LBB0_154:
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[20:21]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	s_branch .LBB0_158  ; 无条件跳转
.LBB0_155:                              ;   in Loop: Header=BB0_158 Depth=1
	s_or_b64 exec, exec, s[20:21]  ; 恢复 exec，被关掉的 lane 重新参与（相当于 if 结束）
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	v_readfirstlane_b32 s20, v1  ; 取第一个有效 lane 的值，放进标量寄存器
	s_cmp_eq_u32 s20, 0  ; 比较「相等」，结果写到 scc
	s_cbranch_scc1 .LBB0_157  ; scc = 1 就跳转
; %bb.156:                              ;   in Loop: Header=BB0_158 Depth=1
	s_sleep 1  ; 睡一会儿再查
	s_cbranch_execnz .LBB0_158  ; 还有 lane 没完成（exec ≠ 0）就跳回去
	s_branch .LBB0_160  ; 无条件跳转
.LBB0_157:
	.loc	2 0 5                           ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_branch .LBB0_160  ; 无条件跳转
.LBB0_158:                              ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v1, 1  ; 向量赋值
	.loc	2 27 5 is_stmt 1                ; ./v0_naive.h:27:5  → 第 27 行
	s_and_saveexec_b64 s[20:21], s[0:1]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_155  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.159:                              ;   in Loop: Header=BB0_158 Depth=1
	global_load_dword v1, v[12:13], off offset:20 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	buffer_inv sc0 sc1  ; 作废缓存，确保读到主机刚写的数据
	v_and_b32_e32 v1, 1, v1  ; 按位与
	s_branch .LBB0_155  ; 无条件跳转
.LBB0_160:
	.loc	2 27 5                          ; ./v0_naive.h:27:5  → 第 27 行
	s_and_saveexec_b64 s[20:21], s[0:1]  ; 把 exec 存起来，只让条件成立的 lane 往下走（相当于 if 开始）
	s_cbranch_execz .LBB0_163  ; 没有 lane 要执行（exec = 0）就跳过这段
; %bb.161:
	v_mov_b32_e32 v1, 0  ; 向量赋值
	global_load_dwordx2 v[2:3], v1, s[18:19] offset:40  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[10:11], v1, s[18:19] offset:24 sc0 sc1  ; 读 printf 缓冲区（和主机共享的内存）
	global_load_dwordx2 v[4:5], v1, s[18:19]  ; 读 printf 缓冲区（和主机共享的内存）
	s_waitcnt vmcnt(2)  ; 等显存读写完成
	v_readfirstlane_b32 s22, v2  ; 取第一个有效 lane 的值，放进标量寄存器
	v_readfirstlane_b32 s23, v3  ; 取第一个有效 lane 的值，放进标量寄存器
	s_add_u32 s24, s22, 1  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s25, s23, 0  ; 带进位加，64 位加法的高 32 位
	s_add_u32 s0, s24, s4  ; 加法，64 位加法的低 32 位（产生进位）
	s_addc_u32 s1, s25, s5  ; 带进位加，64 位加法的高 32 位
	s_cmp_eq_u64 s[0:1], 0  ; 64 位比较「相等」，结果写到 scc
	s_cselect_b32 s1, s25, s1  ; 按 scc 二选一
	s_cselect_b32 s0, s24, s0  ; 按 scc 二选一
	s_and_b64 s[4:5], s[0:1], s[22:23]  ; 64 位按位与
	s_mul_i32 s5, s5, 24  ; 乘法，取低 32 位
	s_mul_hi_u32 s22, s4, 24  ; 乘法，取高 32 位
	s_mul_i32 s4, s4, 24  ; 乘法，取低 32 位
	s_add_i32 s5, s22, s5  ; 加法
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_lshl_add_u64 v[6:7], v[4:5], 0, s[4:5]  ; 64 位：左移后相加（这里多用来算地址）
	v_mov_b32_e32 v8, s0  ; 向量赋值
	global_store_dwordx2 v[6:7], v[10:11], off  ; 写 printf 缓冲区
	v_mov_b32_e32 v9, s1  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[4:5], v1, v[8:11], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_mov_b64 s[4:5], 0  ; 64 位标量赋值
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_ne_u64_e32 vcc, v[4:5], v[10:11]  ; 64 位比较「不等」，结果写到 vcc
	s_and_b64 exec, exec, vcc  ; 只让条件成立的 lane 继续
	s_cbranch_execz .LBB0_163  ; 没有 lane 要执行（exec = 0）就跳过这段
.LBB0_162:                              ; =>This Inner Loop Header: Depth=1
	s_sleep 1  ; 睡一会儿再查
	global_store_dwordx2 v[6:7], v[4:5], off  ; 写 printf 缓冲区
	v_mov_b32_e32 v2, s0  ; 向量赋值
	v_mov_b32_e32 v3, s1  ; 向量赋值
	buffer_wbl2 sc0 sc1  ; 把写过的数据从 L2 写回显存，让主机能看到
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	global_atomic_cmpswap_x2 v[2:3], v1, v[2:5], s[18:19] offset:24 sc0 sc1  ; 原子比较交换：抢或归还缓冲区里的一个槽位，抢失败就重试
	s_waitcnt vmcnt(0)  ; 等显存读写完成
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]  ; 64 位比较「相等」，结果写到 vcc
	s_or_b64 s[4:5], vcc, s[4:5]  ; 64 位按位或
	v_mov_b64_e32 v[4:5], v[2:3]  ; 64 位向量赋值
	s_andn2_b64 exec, exec, s[4:5]  ; 把已经完成的 lane 从 exec 里去掉（循环用）
	s_cbranch_execnz .LBB0_162  ; 还有 lane 没完成（exec ≠ 0）就跳回去
.LBB0_163:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[20:21]  ; printf 结束：恢复 exec，所有 lane 回到 GEMM
; ============================================================================================
; 【3】GEMM 主体（v0_naive.h 第 21–59 行）
; 【3.1】算各个下标（第 21–25 行），判断 K > 0，把 64 个累加器清零（第 28 行）
; ============================================================================================
	.loc	2 22 25 is_stmt 1               ; ./v0_naive.h:22:25  → 第 22 行：const int wm = wave / 2, wn = wave % 2;
	v_lshrrev_b32_e32 v3, 7, v0  ; v3 = tid >> 7 = tid / 128 = wave / 2 = wm（第 22 行）
	.loc	2 29 25                         ; ./v0_naive.h:29:25  → 第 29 行：for (int k0 = 0; k0 < p.K; k0 += BK)
	s_waitcnt lgkmcnt(0)  ; 等前面的 s_load 把 p.N、p.K 等参数读回来
	s_cmp_gt_i32 s7, 0  ; K > 0 ?（第 29 行 for 的第一次判断），结果放 scc
	.loc	2 22 40                         ; ./v0_naive.h:22:40  → 第 22 行
	v_bfe_u32 v6, v0, 6, 1  ; v6 = tid 的第 6 位 = wave % 2 = wn（第 22 行）
	.loc	2 29 5                          ; ./v0_naive.h:29:5  → 第 29 行
	s_cbranch_scc1 .LBB0_165  ; K > 0 就跳到 .LBB0_165
; %bb.164:
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	v_lshlrev_b32_e32 v29, 6, v3  ; v29 = wm × 64（K ≤ 0 的路）
	v_lshlrev_b32_e32 v30, 6, v6  ; v30 = wn × 64
	s_mov_b64 s[0:1], 0  ; s[0:1] = 0：标记「不进 K 循环」
	s_branch .LBB0_166  ; 跳到 .LBB0_166
.LBB0_165:  ; ← K > 0 的路
	s_mov_b64 s[0:1], -1  ; s[0:1] = 全 1：标记「要进 K 循环」
                                        ; implicit-def: $vgpr29
                                        ; implicit-def: $vgpr30
.LBB0_166:  ; ← 两条路在这里汇合
	v_and_b32_e32 v28, 15, v0  ; v28 = tid & 15 = lane % 16 = r16（第 23 行）
	v_bfe_u32 v2, v0, 4, 2  ; v2 = tid 的第 4–5 位 = lane / 16 = g（第 23 行）
	s_lshl_b32 s5, s3, 7  ; s5 = blockIdx.y × 128 = m0（第 24 行）
	s_lshl_b32 s4, s2, 7  ; s4 = blockIdx.x × 128 = n0（第 24 行）
	v_mov_b32_e32 v1, 0  ; v1 = 0：后面拼 64 位地址时当高 32 位
	v_accvgpr_write_b32 a60, 0  ; acc[3][3][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a61, 0  ; acc[3][3][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a62, 0  ; acc[3][3][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a63, 0  ; acc[3][3][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a56, 0  ; acc[3][2][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a57, 0  ; acc[3][2][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a58, 0  ; acc[3][2][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a59, 0  ; acc[3][2][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a52, 0  ; acc[3][1][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a53, 0  ; acc[3][1][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a54, 0  ; acc[3][1][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a55, 0  ; acc[3][1][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a48, 0  ; acc[3][0][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a49, 0  ; acc[3][0][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a50, 0  ; acc[3][0][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a51, 0  ; acc[3][0][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a44, 0  ; acc[2][3][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a45, 0  ; acc[2][3][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a46, 0  ; acc[2][3][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a47, 0  ; acc[2][3][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a40, 0  ; acc[2][2][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a41, 0  ; acc[2][2][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a42, 0  ; acc[2][2][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a43, 0  ; acc[2][2][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a36, 0  ; acc[2][1][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a37, 0  ; acc[2][1][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a38, 0  ; acc[2][1][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a39, 0  ; acc[2][1][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a32, 0  ; acc[2][0][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a33, 0  ; acc[2][0][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a34, 0  ; acc[2][0][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a35, 0  ; acc[2][0][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a28, 0  ; acc[1][3][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a29, 0  ; acc[1][3][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a30, 0  ; acc[1][3][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a31, 0  ; acc[1][3][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a24, 0  ; acc[1][2][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a25, 0  ; acc[1][2][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a26, 0  ; acc[1][2][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a27, 0  ; acc[1][2][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a20, 0  ; acc[1][1][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a21, 0  ; acc[1][1][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a22, 0  ; acc[1][1][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a23, 0  ; acc[1][1][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a16, 0  ; acc[1][0][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a17, 0  ; acc[1][0][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a18, 0  ; acc[1][0][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a19, 0  ; acc[1][0][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a12, 0  ; acc[0][3][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a13, 0  ; acc[0][3][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a14, 0  ; acc[0][3][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a15, 0  ; acc[0][3][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a8, 0  ; acc[0][2][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a9, 0  ; acc[0][2][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a10, 0  ; acc[0][2][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a11, 0  ; acc[0][2][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a4, 0  ; acc[0][1][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a5, 0  ; acc[0][1][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a6, 0  ; acc[0][1][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a7, 0  ; acc[0][1][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a0, 0  ; acc[0][0][0] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a1, 0  ; acc[0][0][1] = 0（第 28 行 f32x4 acc[4][4] = {}）
	v_accvgpr_write_b32 a2, 0  ; acc[0][0][2] = 0（第 28 行 f32x4 acc[4][4] = {}）
	s_andn2_b64 vcc, exec, s[0:1]  ; vcc = exec & ~s[0:1]：K ≤ 0 时不为 0
	v_accvgpr_write_b32 a3, 0  ; acc[0][0][3] = 0（第 28 行 f32x4 acc[4][4] = {}）
	s_cbranch_vccnz .LBB0_171  ; K ≤ 0：跳过整个 K 循环，直接去【4】收尾
; %bb.167:
	.loc	2 25 24 is_stmt 1               ; ./v0_naive.h:25:24  → 第 25 行：const int KB = p.K / 2, KS = p.K / 32;
	s_lshr_b32 s0, s7, 31  ; 取 K 的符号位：K 是 int，K/2 要按有符号除法向 0 取整
	s_add_i32 s0, s7, s0  ; K + 符号位
	s_ashr_i32 s18, s0, 1  ; s18 = K / 2 = KB（第 25 行）：A、B 每行多少字节
	s_ashr_i32 s0, s7, 31  ; K 的符号：0 或 −1
	.loc	2 25 38 is_stmt 0               ; ./v0_naive.h:25:38  → 第 25 行
	s_lshr_b32 s0, s0, 27  ; K 为负时是 31，否则是 0（除以 32 时向 0 取整用）
	v_lshlrev_b32_e32 v29, 6, v3  ; v29 = wm × 64
	s_add_i32 s0, s7, s0  ; K + 修正量
	v_or_b32_e32 v24, v29, v28  ; v24 = wm×64 + r16：mi=0 时 tile 内 A 的行号 ra（第 48 行）
	v_mov_b32_e32 v3, v1  ; v3 = 0：v[2:3] 成了 64 位的 g
	s_ashr_i32 s0, s0, 5  ; s0 = K / 32 = KS（第 25 行）：每行多少个 scale
	.loc	2 29 5 is_stmt 1                ; ./v0_naive.h:29:5  → 第 29 行
	v_lshl_add_u64 v[4:5], s[12:13], 0, v[2:3]  ; v[4:5] = p.As + g（第 50 行 A scale 地址的公共部分）
	v_lshlrev_b32_e32 v30, 6, v6  ; v30 = wn × 64
	v_lshl_add_u64 v[6:7], s[14:15], 0, v[2:3]  ; v[6:7] = p.Bs + g（第 55 行 B scale 地址的公共部分）
	v_add_u32_e32 v3, s5, v24  ; v3 = m0 + ra(mi=0)：A 的全局行号
	v_mad_u64_u32 v[8:9], s[2:3], v3, s0, 0  ; v[8:9] = (m0+ra0) × KS（第 50 行；先按无符号乘）
	v_ashrrev_i32_e32 v11, 31, v3  ; 行号的符号：(size_t) 转换按有符号扩展，要补一个修正
	v_mov_b32_e32 v10, v9  ; 把高 32 位挪出来做符号修正
	v_or_b32_e32 v16, v30, v28  ; v16 = wn×64 + r16：ni=0 时 tile 内 B 的行号 rb（第 53 行）
	v_mad_u64_u32 v[10:11], s[2:3], v11, s0, v[10:11]  ; 符号修正：高 32 位 += 符号 × KS
	v_or_b32_e32 v3, s4, v16  ; v3 = n0 + rb(ni=0)：B 的全局行号（n0 是 128 的倍数，或就等于加）
	s_ashr_i32 s2, s4, 31  ; n0 的符号
	v_mov_b32_e32 v9, v10  ; v[8:9] 完成：mi=0 的 A scale 行偏移 (m0+ra0)×KS
	s_mul_i32 s12, s2, s0  ; s12 = n0 的符号 × KS：下面 4 个 B 偏移共用的符号修正
	v_mad_u64_u32 v[10:11], s[2:3], v3, s0, 0  ; v[10:11] = (n0+rb0) × KS：ni=0 的 B scale 行偏移（修正在后面补）
	v_or_b32_e32 v3, 16, v16  ; rb(ni=1) = rb0 + 16
	v_lshlrev_b32_e32 v36, 6, v3  ; v36 = rb(ni=1) × 64：这一行在 sB 里的字节偏移（第 54 行）
	v_or_b32_e32 v3, s4, v3  ; n0 + rb(ni=1)
	v_mad_u64_u32 v[12:13], s[2:3], v3, s0, 0  ; v[12:13] = (n0+rb1) × KS：ni=1 的 B scale 行偏移
	v_or_b32_e32 v3, 32, v16  ; rb(ni=2) = rb0 + 32
	v_lshlrev_b32_e32 v37, 6, v3  ; v37 = rb(ni=2) × 64
	v_or_b32_e32 v3, s4, v3  ; n0 + rb(ni=2)
	v_mad_u64_u32 v[14:15], s[2:3], v3, s0, 0  ; v[14:15] = (n0+rb2) × KS：ni=2 的 B scale 行偏移
	v_or_b32_e32 v3, 48, v16  ; rb(ni=3) = rb0 + 48
	v_lshlrev_b32_e32 v39, 6, v3  ; v39 = rb(ni=3) × 64
	v_or_b32_e32 v3, s4, v3  ; n0 + rb(ni=3)
	v_lshlrev_b32_e32 v35, 6, v16  ; v35 = rb(ni=0) × 64
	v_mad_u64_u32 v[16:17], s[2:3], v3, s0, 0  ; v[16:17] = (n0+rb3) × KS：ni=3 的 B scale 行偏移
	v_or_b32_e32 v3, 16, v24  ; ra(mi=1) = ra0 + 16
	v_lshlrev_b32_e32 v40, 6, v3  ; v40 = ra(mi=1) × 64：这一行在 sA 里的字节偏移（第 49 行）
	v_add_u32_e32 v3, s5, v3  ; m0 + ra(mi=1)
	v_mad_u64_u32 v[18:19], s[2:3], v3, s0, 0  ; v[18:19] = (m0+ra1) × KS（先按无符号乘）
	v_ashrrev_i32_e32 v21, 31, v3  ; 符号
	v_mov_b32_e32 v20, v19  ; 高 32 位挪出来
	v_or_b32_e32 v3, 32, v24  ; ra(mi=2) = ra0 + 32
	v_mad_u64_u32 v[20:21], s[2:3], v21, s0, v[20:21]  ; 符号修正
	v_lshlrev_b32_e32 v41, 6, v3  ; v41 = ra(mi=2) × 64
	v_add_u32_e32 v3, s5, v3  ; m0 + ra(mi=2)
	v_mov_b32_e32 v19, v20  ; v[18:19] 完成：mi=1 的 A scale 行偏移
	v_mad_u64_u32 v[20:21], s[2:3], v3, s0, 0  ; v[20:21] = (m0+ra2) × KS（先按无符号乘）
	v_ashrrev_i32_e32 v23, 31, v3  ; 符号
	v_mov_b32_e32 v22, v21  ; 高 32 位挪出来
	v_or_b32_e32 v3, 48, v24  ; ra(mi=3) = ra0 + 48
	v_mad_u64_u32 v[22:23], s[2:3], v23, s0, v[22:23]  ; 符号修正
	v_lshlrev_b32_e32 v42, 6, v3  ; v42 = ra(mi=3) × 64
	v_add_u32_e32 v3, s5, v3  ; m0 + ra(mi=3)
	v_mov_b32_e32 v21, v22  ; v[20:21] 完成：mi=2 的 A scale 行偏移
	v_mad_u64_u32 v[22:23], s[2:3], v3, s0, 0  ; v[22:23] = (m0+ra3) × KS（先按无符号乘）
	v_lshlrev_b32_e32 v27, 4, v2  ; v27 = g × 16：片段在一行里的字节偏移（第 49、54 行的 g*16）
	v_lshlrev_b32_e32 v34, 6, v24  ; v34 = ra(mi=0) × 64
	v_ashrrev_i32_e32 v25, 31, v3  ; 符号
	v_mov_b32_e32 v24, v23  ; 高 32 位挪出来
	v_lshrrev_b32_e32 v26, 4, v0  ; v26 = tid / 16：拷贝循环第 0 圈搬的行号 r（第 35 行）
	v_or_b32_e32 v38, 0x2000, v27  ; v38 = 8192 + g×16：sB 从 LDS 第 8192 字节开始
	v_mad_u64_u32 v[24:25], s[2:3], v25, s0, v[24:25]  ; 符号修正
	s_mov_b32 s1, 0  ; s1 = 0：后面把 s[0:1] 当 64 位用
	v_accvgpr_write_b32 a3, 0  ; acc[0][0][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a2, 0  ; acc[0][0][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a1, 0  ; acc[0][0][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a0, 0  ; acc[0][0][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a7, 0  ; acc[0][1][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a6, 0  ; acc[0][1][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a5, 0  ; acc[0][1][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a4, 0  ; acc[0][1][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a11, 0  ; acc[0][2][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a10, 0  ; acc[0][2][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a9, 0  ; acc[0][2][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a8, 0  ; acc[0][2][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a15, 0  ; acc[0][3][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a14, 0  ; acc[0][3][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a13, 0  ; acc[0][3][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a12, 0  ; acc[0][3][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a19, 0  ; acc[1][0][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a18, 0  ; acc[1][0][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a17, 0  ; acc[1][0][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a16, 0  ; acc[1][0][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a23, 0  ; acc[1][1][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a22, 0  ; acc[1][1][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a21, 0  ; acc[1][1][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a20, 0  ; acc[1][1][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a27, 0  ; acc[1][2][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a26, 0  ; acc[1][2][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a25, 0  ; acc[1][2][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a24, 0  ; acc[1][2][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a31, 0  ; acc[1][3][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a30, 0  ; acc[1][3][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a29, 0  ; acc[1][3][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a28, 0  ; acc[1][3][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a35, 0  ; acc[2][0][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a34, 0  ; acc[2][0][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a33, 0  ; acc[2][0][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a32, 0  ; acc[2][0][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a39, 0  ; acc[2][1][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a38, 0  ; acc[2][1][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a37, 0  ; acc[2][1][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a36, 0  ; acc[2][1][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a43, 0  ; acc[2][2][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a42, 0  ; acc[2][2][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a41, 0  ; acc[2][2][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a40, 0  ; acc[2][2][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a47, 0  ; acc[2][3][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a46, 0  ; acc[2][3][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a45, 0  ; acc[2][3][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a44, 0  ; acc[2][3][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a51, 0  ; acc[3][0][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a50, 0  ; acc[3][0][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a49, 0  ; acc[3][0][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a48, 0  ; acc[3][0][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a55, 0  ; acc[3][1][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a54, 0  ; acc[3][1][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a53, 0  ; acc[3][1][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a52, 0  ; acc[3][1][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a59, 0  ; acc[3][2][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a58, 0  ; acc[3][2][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a57, 0  ; acc[3][2][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a56, 0  ; acc[3][2][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a63, 0  ; acc[3][3][3] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a62, 0  ; acc[3][3][2] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a61, 0  ; acc[3][3][1] = 0（编译器在这条路上又清零一遍，结果一样）
	v_accvgpr_write_b32 a60, 0  ; acc[3][3][0] = 0（编译器在这条路上又清零一遍，结果一样）
	v_add_u32_e32 v11, s12, v11  ; ni=0 的 B scale 行偏移补上符号修正
	v_add_u32_e32 v13, s12, v13  ; ni=1 同上
	v_add_u32_e32 v15, s12, v15  ; ni=2 同上
	v_add_u32_e32 v17, s12, v17  ; ni=3 同上
	v_mov_b32_e32 v23, v24  ; v[22:23] 完成：mi=3 的 A scale 行偏移
	v_or_b32_e32 v3, 0xffffff00, v0  ; v3 = tid − 256：i 的初值减 256（循环里每圈先 +256，正好是本圈的 i）
	v_add_u32_e32 v31, s4, v26  ; v31 = n0 + tid/16：第 0 圈要搬的 B 的全局行号（第 39 行 n0 + r）
	v_add_u32_e32 v32, s5, v26  ; v32 = m0 + tid/16：第 0 圈要搬的 A 的全局行号（第 37 行 m0 + r）
	v_lshlrev_b32_e32 v33, 2, v0  ; v33 = tid × 4：第 0 圈 sA[i] 的 LDS 字节地址（i = tid）
	s_movk_i32 s12, 0x6ff  ; s12 = 1791：i > 1791 说明这是最后一圈（i < 2048）
	v_add_u32_e32 v34, v27, v34  ; v34 = ra0×64 + g×16：mi=0 的 A 片段在 LDS 里的地址（第 49 行）
	v_add_u32_e32 v35, v38, v35  ; v35 = 8192 + rb0×64 + g×16：ni=0 的 B 片段地址（第 54 行）
	v_add_u32_e32 v36, v38, v36  ; v36 = ni=1 的 B 片段地址
	v_add_u32_e32 v37, v38, v37  ; v37 = ni=2 的 B 片段地址
	v_add_u32_e32 v38, v38, v39  ; v38 = ni=3 的 B 片段地址
	v_add_u32_e32 v39, v27, v40  ; v39 = mi=1 的 A 片段地址
	v_add_u32_e32 v40, v27, v41  ; v40 = mi=2 的 A 片段地址
	v_add_u32_e32 v41, v27, v42  ; v41 = mi=3 的 A 片段地址
	s_mov_b32 s13, 0  ; s13 = k0 = 0（第 29 行）
; ============================================================================================
; 【3.2】K 循环（第 29 行 for k0）：每步处理 128 个 K
; ============================================================================================
.LBB0_168:                              ; =>This Loop Header: Depth=1  ← K 循环开头
                                        ;     Child Loop BB0_169 Depth 2
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	s_lshr_b32 s0, s13, 1  ; s0 = k0 / 2：这一步在每行里从第几个字节开始
	.loc	2 33 9 is_stmt 1                ; ./v0_naive.h:33:9  → 第 33 行：for (int i = tid; i < BM * ROWB / 4; i += 256)
	s_add_u32 s14, s8, s0  ; s[14:15] = p.A + k0/2（低 32 位）
	s_addc_u32 s15, s9, 0  ; （高 32 位，加上进位）
	s_add_u32 s20, s10, s0  ; s[20:21] = p.B + k0/2（低 32 位）
	s_addc_u32 s21, s11, 0  ; （高 32 位）
	s_mov_b64 s[2:3], 0  ; s[2:3] = 0：记录哪些 lane 已经搬完
	v_mov_b64_e32 v[24:25], s[14:15]  ; v[24:25] = A 这一步的基址
	v_mov_b64_e32 v[26:27], s[20:21]  ; v[26:27] = B 这一步的基址
	v_mov_b32_e32 v42, v33  ; v42 = tid×4：本 lane 写 LDS 的地址（每圈 +1024）
	v_mov_b32_e32 v43, v32  ; v43 = m0 + r：A 的全局行号（每圈 +16）
	v_mov_b32_e32 v44, v31  ; v44 = n0 + r：B 的全局行号（每圈 +16）
	v_mov_b32_e32 v45, v3  ; v45 = i − 256
; ============================================================================================
; 【3.3】拷贝循环（第 33–40 行）：256 个线程把 A、B 这一步的 128×64 字节搬进 LDS，
; 每个 lane 每圈搬 4 字节，一共 8 圈
; ============================================================================================
.LBB0_169:                              ;   Parent Loop BB0_168 Depth=1  ← 拷贝循环开头
                                        ; =>  This Inner Loop Header: Depth=2
	.loc	2 37 17                         ; ./v0_naive.h:37:17  → 第 37 行：*reinterpret_cast<const uint32_t*>(p.A + (size_t)(m0 + r) * KB + k0 / 2 + c *...
	v_mad_u64_u32 v[46:47], s[14:15], v43, s18, v[24:25]  ; v[46:47] = (m0+r) × KB + A 基址：A 这一行的起始地址（第 37 行）
	.loc	2 37 66 is_stmt 0               ; ./v0_naive.h:37:66  → 第 37 行
	v_ashrrev_i32_e32 v51, 31, v43  ; A 行号的符号
	.loc	2 39 17 is_stmt 1               ; ./v0_naive.h:39:17  → 第 39 行：*reinterpret_cast<const uint32_t*>(p.B + (size_t)(n0 + r) * KB + k0 / 2 + c *...
	v_mad_u64_u32 v[48:49], s[14:15], v44, s18, v[26:27]  ; v[48:49] = (n0+r) × KB + B 基址：B 这一行的起始地址（第 39 行）
	.loc	2 37 17                         ; ./v0_naive.h:37:17  → 第 37 行
	v_mov_b32_e32 v50, v47  ; A 地址的高 32 位挪出来
	.loc	2 39 66                         ; ./v0_naive.h:39:66  → 第 39 行
	v_ashrrev_i32_e32 v53, 31, v44  ; B 行号的符号
	.loc	2 39 17 is_stmt 0               ; ./v0_naive.h:39:17  → 第 39 行
	v_mov_b32_e32 v52, v49  ; B 地址的高 32 位挪出来
	.loc	2 37 17 is_stmt 1               ; ./v0_naive.h:37:17  → 第 37 行
	v_mad_u64_u32 v[50:51], s[14:15], v51, s18, v[50:51]  ; A 地址的符号修正
	.loc	2 37 93 is_stmt 0               ; ./v0_naive.h:37:93  → 第 37 行
	v_and_b32_e32 v0, 60, v42  ; v0 = c × 4：这个 dword 在一行里的字节偏移（第 35、37 行）
	.loc	2 39 17 is_stmt 1               ; ./v0_naive.h:39:17  → 第 39 行
	v_mad_u64_u32 v[52:53], s[14:15], v53, s18, v[52:53]  ; B 地址的符号修正
	.loc	2 37 17                         ; ./v0_naive.h:37:17  → 第 37 行
	v_mov_b32_e32 v47, v50  ; A 行起始地址完成
	.loc	2 39 17                         ; ./v0_naive.h:39:17  → 第 39 行
	v_mov_b32_e32 v49, v52  ; B 行起始地址完成
	.loc	2 37 17                         ; ./v0_naive.h:37:17  → 第 37 行
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[0:1]  ; A 地址 + c×4
	.loc	2 39 17                         ; ./v0_naive.h:39:17  → 第 39 行
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[0:1]  ; B 地址 + c×4
	.loc	2 37 17                         ; ./v0_naive.h:37:17  → 第 37 行
	global_load_dword v0, v[46:47], off  ; 从显存读 A 的 4 字节 → v0（第 37 行）
	.loc	2 39 17                         ; ./v0_naive.h:39:17  → 第 39 行
	global_load_dword v50, v[48:49], off  ; 从显存读 B 的 4 字节 → v50（第 39 行）
	.loc	2 33 29                         ; ./v0_naive.h:33:29  → 第 33 行：for (int i = tid; i < BM * ROWB / 4; i += 256)
	v_add_u32_e32 v45, 0x100, v45  ; i += 256（第 33 行）
	v_cmp_lt_u32_e32 vcc, s12, v45  ; vcc = (i > 1791)：这是不是本 lane 的最后一圈
	v_add_u32_e32 v44, 16, v44  ; B 行号 += 16（下一圈 r 大 16）
	v_add_u32_e32 v43, 16, v43  ; A 行号 += 16
	.loc	2 33 9 is_stmt 0                ; ./v0_naive.h:33:9  → 第 33 行
	s_or_b64 s[2:3], vcc, s[2:3]  ; 记下已经搬完的 lane
	.loc	2 36 48 is_stmt 1               ; ./v0_naive.h:36:48  → 第 36 行：reinterpret_cast<uint32_t*>(sA)[i] =
	s_waitcnt vmcnt(0)  ; 等两条 load 都回来。v0 每圈都在这里等一次显存，是它最慢的地方
	ds_write2st64_b32 v42, v0, v50 offset1:32  ; 写 LDS：v0 写到 sA[i]，v50 写到往后 32×256 = 8192 字节处的 sB[i]（第 36、38 行）
	.loc	2 33 29                         ; ./v0_naive.h:33:29  → 第 33 行
	v_add_u32_e32 v42, 0x400, v42  ; LDS 地址 += 1024（i 加 256 个 dword）
	.loc	2 33 9 is_stmt 0                ; ./v0_naive.h:33:9  → 第 33 行
	s_andn2_b64 exec, exec, s[2:3]  ; 关掉已经搬完的 lane
	s_cbranch_execnz .LBB0_169  ; 还有 lane 没搬完，就跳回拷贝循环开头
; ============================================================================================
; 【3.4】拷贝完成后：__syncthreads、读 scale、从 LDS 读片段、16 条 MFMA（第 41–59 行）
; 执行顺序：先发出 6 个 scale 的读，再读 A 片段（mi=0）和 4 个 B 片段，
; 然后每凑齐一组数据就做 4 条 MFMA（mi = 0、1、2、3 各一组）
; ============================================================================================
; %bb.170:                              ;   in Loop: Header=BB0_168 Depth=1
	.loc	2 0 9                           ; ./v0_naive.h:0:9  → 编译器生成的代码，不对应某一行源码
	s_or_b64 exec, exec, s[2:3]  ; 拷贝循环结束：所有 lane 重新打开
	s_lshr_b32 s0, s13, 5  ; s0 = k0 / 32：这一步的 scale 从每行第几个开始
	.loc	2 46 14 is_stmt 1               ; ./v0_naive.h:46:14  → 第 46 行：for (int mi = 0; mi < 4; ++mi)
	v_lshl_add_u64 v[58:59], v[4:5], 0, s[0:1]  ; v[58:59] = p.As + g + k0/32
	v_lshl_add_u64 v[24:25], v[6:7], 0, s[0:1]  ; v[24:25] = p.Bs + g + k0/32
	.loc	2 50 28                         ; ./v0_naive.h:50:28  → 第 50 行：const int sa = p.As[(size_t)(m0 + ra) * KS + k0 / 32 + g];
	v_lshl_add_u64 v[26:27], v[58:59], 0, v[8:9]  ; &As[(m0+ra0)*KS + k0/32 + g]：mi=0 的 A scale 地址（第 50 行）
.Ltmp2:
	.loc	1 691 5                         ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:691:5 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:50 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./v0_naive.h:41:9 ] ] ]  → __syncthreads() 的实现
	s_waitcnt lgkmcnt(0)  ; __syncthreads 第 1 步（第 41 行）：等自己写 LDS 完成
	.loc	1 692 5                         ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:692:5 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:50 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./v0_naive.h:41:9 ] ] ]  → __syncthreads() 的实现
	s_barrier  ; __syncthreads 第 2 步：等 workgroup 的 4 个 wave 都到这里
.Ltmp3:
	.loc	2 50 28                         ; ./v0_naive.h:50:28  → 第 50 行
	global_load_ubyte v0, v[26:27], off  ; 读 1 字节 A scale → v0：sa(mi=0)（第 50 行）
	.loc	2 55 32                         ; ./v0_naive.h:55:32  → 第 55 行：const int sb = p.Bs[(size_t)(n0 + rb) * KS + k0 / 32 + g];
	v_lshl_add_u64 v[26:27], v[24:25], 0, v[10:11]  ; ni=0 的 B scale 地址（第 55 行）
	global_load_ubyte v62, v[26:27], off  ; sb(ni=0) → v62
	v_lshl_add_u64 v[26:27], v[24:25], 0, v[12:13]  ; ni=1 的 B scale 地址
	global_load_ubyte v63, v[26:27], off  ; sb(ni=1) → v63
	v_lshl_add_u64 v[26:27], v[24:25], 0, v[14:15]  ; ni=2 的 B scale 地址
	v_lshl_add_u64 v[24:25], v[24:25], 0, v[16:17]  ; ni=3 的 B scale 地址
	global_load_ubyte v64, v[26:27], off  ; sb(ni=2) → v64
	global_load_ubyte v65, v[24:25], off  ; sb(ni=3) → v65
	.loc	2 50 28                         ; ./v0_naive.h:50:28  → 第 50 行
	v_lshl_add_u64 v[24:25], v[58:59], 0, v[18:19]  ; mi=1 的 A scale 地址
	global_load_ubyte v66, v[24:25], off  ; sa(mi=1) → v66
	.loc	2 49 29                         ; ./v0_naive.h:49:29  → 第 49 行：const i32x4 a = *reinterpret_cast<const i32x4*>(sA + ra * ROWB + g * 16);
	ds_read_b128 v[24:27], v34  ; 从 LDS 读 mi=0 的 A 片段，16 字节 = 32 个 fp4 → v[24:27]（第 49 行）
	.loc	2 54 33                         ; ./v0_naive.h:54:33  → 第 54 行：const i32x4 b = *reinterpret_cast<const i32x4*>(sB + rb * ROWB + g * 16);
	ds_read_b128 v[42:45], v35  ; 读 ni=0 的 B 片段 → v[42:45]（第 54 行）
	ds_read_b128 v[46:49], v36  ; 读 ni=1 的 B 片段 → v[46:49]
	ds_read_b128 v[50:53], v37  ; 读 ni=2 的 B 片段 → v[50:53]
	ds_read_b128 v[54:57], v38  ; 读 ni=3 的 B 片段 → v[54:57]
	.loc	2 50 28                         ; ./v0_naive.h:50:28  → 第 50 行
	v_lshl_add_u64 v[60:61], v[58:59], 0, v[20:21]  ; mi=2 的 A scale 地址
	v_lshl_add_u64 v[58:59], v[58:59], 0, v[22:23]  ; mi=3 的 A scale 地址
	.loc	2 29 35                         ; ./v0_naive.h:29:35  → 第 29 行：for (int k0 = 0; k0 < p.K; k0 += BK)
	s_addk_i32 s13, 0x80  ; k0 += 128（第 29 行的 k0 += BK），提前算好
	.loc	2 29 25 is_stmt 0               ; ./v0_naive.h:29:25  → 第 29 行
	s_cmp_ge_i32 s13, s7  ; k0 ≥ K ?结果放 scc，留到循环末尾再判断
.Ltmp4:
	.file	3 "." "common.h" md5 0x1b7369279800554c32ed591e841c82de  ; 文件 3：common.h
	.loc	3 34 12 is_stmt 1               ; ./common.h:34:12 @[ ./v0_naive.h:56:31 ]  → common.h 第 34 行：mfma_fp4 里的 MFMA 内建函数（被第 56 行调用）
	s_waitcnt vmcnt(4) lgkmcnt(3)  ; 等 sa(mi=0)、sb(ni=0) 和 A 片段、ni=0 的 B 片段：最早发出的 2 个 load 和 2 个 LDS 读
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[42:45], a[0:3], v0, v62 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[0][0] += A(mi=0) × B(ni=0)（第 56 行）。v[24:27]、v[42:45] 是 A、B 片段；v0、v62 是两个 1 字节的 scale；cbsz:4、blgp:4 表示 A、B 都按 fp4 解读
.Ltmp5:
	.loc	2 50 28                         ; ./v0_naive.h:50:28  → 第 50 行
	global_load_ubyte v58, v[58:59], off  ; sa(mi=3) → v58
.Ltmp6:
	.loc	3 34 12                         ; ./common.h:34:12 @[ ./v0_naive.h:56:31 ]  → common.h 第 34 行：mfma_fp4 里的 MFMA 内建函数（被第 56 行调用）
	s_waitcnt vmcnt(4) lgkmcnt(2)  ; 等 sb(ni=1) 和 ni=1 的 B 片段
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:27], v[46:49], a[4:7], v0, v63 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[0][1] += A(mi=0) × B(ni=1)
	s_waitcnt vmcnt(3) lgkmcnt(1)  ; 等 sb(ni=2) 和 ni=2 的 B 片段
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[24:27], v[50:53], a[8:11], v0, v64 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[0][2] += A(mi=0) × B(ni=2)
	s_waitcnt vmcnt(2) lgkmcnt(0)  ; 等 sb(ni=3) 和 ni=3 的 B 片段
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:27], v[54:57], a[12:15], v0, v65 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[0][3] += A(mi=0) × B(ni=3)
.Ltmp7:
	.loc	2 50 28                         ; ./v0_naive.h:50:28  → 第 50 行
	global_load_ubyte v0, v[60:61], off  ; sa(mi=2) → v0（v0 里原来的 sa(mi=0) 已经用完）
	.loc	2 49 29                         ; ./v0_naive.h:49:29  → 第 49 行
	ds_read_b128 v[24:27], v39  ; 读 mi=1 的 A 片段，覆盖 mi=0 的（第 49 行）
.Ltmp8:
	.loc	3 34 12                         ; ./common.h:34:12 @[ ./v0_naive.h:56:31 ]  → common.h 第 34 行：mfma_fp4 里的 MFMA 内建函数（被第 56 行调用）
	s_waitcnt vmcnt(2) lgkmcnt(0)  ; 等 sa(mi=1) 和 mi=1 的 A 片段
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[24:27], v[42:45], a[16:19], v66, v62 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[1][0] += A(mi=1) × B(ni=0)，scale 是 v66、v62
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[24:27], v[46:49], a[20:23], v66, v63 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[1][1]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[24:27], v[50:53], a[24:27], v66, v64 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[1][2]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[24:27], v[54:57], a[28:31], v66, v65 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[1][3]
.Ltmp9:
	.loc	2 49 29                         ; ./v0_naive.h:49:29  → 第 49 行
	ds_read_b128 v[24:27], v40  ; 读 mi=2 的 A 片段
.Ltmp10:
	.loc	3 34 12                         ; ./common.h:34:12 @[ ./v0_naive.h:56:31 ]  → common.h 第 34 行：mfma_fp4 里的 MFMA 内建函数（被第 56 行调用）
	s_waitcnt vmcnt(0) lgkmcnt(0)  ; 等所有 scale 和 mi=2 的 A 片段
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[24:27], v[42:45], a[32:35], v0, v62 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[2][0] += A(mi=2) × B(ni=0)，scale 是 v0（sa(mi=2)）、v62
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[24:27], v[46:49], a[36:39], v0, v63 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[2][1]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[24:27], v[50:53], a[40:43], v0, v64 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[2][2]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[24:27], v[54:57], a[44:47], v0, v65 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[2][3]
.Ltmp11:
	.loc	2 49 29                         ; ./v0_naive.h:49:29  → 第 49 行
	ds_read_b128 v[24:27], v41  ; 读 mi=3 的 A 片段
.Ltmp12:
	.loc	1 691 5                         ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:691:5 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:50 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./v0_naive.h:59:9 ] ] ]  → __syncthreads() 的实现
	s_waitcnt lgkmcnt(0)  ; __syncthreads 第 1 步（第 59 行）：等 LDS 读完
	.loc	1 692 5                         ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:692:5 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:707:50 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_device_functions.h:714:3 @[ ./v0_naive.h:59:9 ] ] ]  → __syncthreads() 的实现
	s_barrier  ; __syncthreads 第 2 步：4 个 wave 都读完 LDS，下一步的拷贝才能覆盖它
.Ltmp13:
	.loc	3 34 12                         ; ./common.h:34:12 @[ ./v0_naive.h:56:31 ]  → common.h 第 34 行：mfma_fp4 里的 MFMA 内建函数（被第 56 行调用）
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[24:27], v[42:45], a[48:51], v58, v62 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[3][0] += A(mi=3) × B(ni=0)，scale 是 v58、v62。数据已在寄存器里，编译器把这 4 条挪到了 barrier 之后
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[24:27], v[46:49], a[52:55], v58, v63 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[3][1]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[24:27], v[50:53], a[56:59], v58, v64 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[3][2]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[24:27], v[54:57], a[60:63], v58, v65 op_sel_hi:[0,0,0] cbsz:4 blgp:4  ; acc[3][3]
.Ltmp14:
	.loc	2 29 5                          ; ./v0_naive.h:29:5  → 第 29 行
	s_cbranch_scc0 .LBB0_168  ; k0 < K（scc = 0）就回到 K 循环开头
; ============================================================================================
; 【4】收尾（第 63–71 行）：64 个累加结果转成 bf16，写回 C
; 每个结果都是同样 4 步：从 AGPR 读出 → 转 bf16 → 算 &C[m][n] → 写 2 字节。
; 同一行的 4 个 ni 共用一个行地址，用 offset:32/64/96 区分列（16 列 × 2 字节 = 32）
; ============================================================================================
.LBB0_171:  ; ← 收尾
	.loc	2 0 5 is_stmt 0                 ; ./v0_naive.h:0:5  → 编译器生成的代码，不对应某一行源码
	v_add_u32_e32 v29, s5, v29  ; v29 = m0 + wm×64
	v_or3_b32 v28, v30, s4, v28  ; v28 = n0 + wn×64 + r16 = n：ni=0 时这个 lane 写的列（第 69 行）
	v_accvgpr_read_b32 v67, a0  ; 取出 acc[0][0][0]：从 AGPR 搬到 VGPR
	v_mov_b32_e32 v68, s16  ; v[68:69] = p.C
	v_mov_b32_e32 v69, s17  ; （高 32 位）
	.loc	2 63 5 is_stmt 1                ; ./v0_naive.h:63:5  → 第 63 行：for (int mi = 0; mi < 4; ++mi)
	v_lshl_or_b32 v2, v2, 2, v29  ; v2 = m0 + wm×64 + 4g：mi=0、i=0 时的行号 m（第 68 行）
	v_ashrrev_i32_e32 v29, 31, v28  ; n 的符号，把 n 扩成 64 位
	.loc	2 67 13                         ; ./v0_naive.h:67:13  → 第 67 行：for (int i = 0; i < 4; ++i) {
	v_lshl_add_u64 v[28:29], v[28:29], 1, v[68:69]  ; v[28:29] = p.C + n×2：C 第 0 行第 n 列的地址
.Ltmp15:
	.file	4 "/opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail" "amd_hip_bf16.h" md5 0x110c5195dca90dcd6b1e383f5f714869  ; 文件 4：HIP 的 bf16 头文件
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v67, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）。pk 表示一次转两个、拼进一个 32 位寄存器；这里只用低 16 位，第二个操作数 s0 是凑数的
.Ltmp16:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行：p.C[(size_t)m * p.N + n] = __float2bfloat16(acc[mi][ni][i]);
	v_mad_i64_i32 v[68:69], s[0:1], v2, s6, 0  ; 行号 m × N（64 位）
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[68:69], v[68:69], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
	v_accvgpr_read_b32 v66, a1  ; 取出 acc[0][0][1]
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[68:69], v30, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62  → 第 68 行：const int m = m0 + wm * 64 + mi * 16 + 4 * g + i;
	v_or_b32_e32 v30, 1, v2  ; 行号 m = v2 + 1（mi = 0，i = 1）
.Ltmp17:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v70, v66, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp18:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行
	v_mad_i64_i32 v[66:67], s[0:1], v30, s6, 0  ; 行号 m × N（64 位）
	v_accvgpr_read_b32 v65, a2  ; 取出 acc[0][0][2]
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62  → 第 68 行
	v_or_b32_e32 v30, 2, v2  ; 行号 m = v2 + 2（mi = 0，i = 2）
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[66:67], v70, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
.Ltmp19:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v65, v65, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp20:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行
	v_mad_i64_i32 v[70:71], s[0:1], v30, s6, 0  ; 行号 m × N（64 位）
	v_accvgpr_read_b32 v64, a3  ; 取出 acc[0][0][3]
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[70:71], v[70:71], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62  → 第 68 行
	v_or_b32_e32 v30, 3, v2  ; 行号 m = v2 + 3（mi = 0，i = 3）
	v_accvgpr_read_b32 v63, a4  ; 取出 acc[0][1][0]
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[70:71], v65, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
.Ltmp21:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v72, v64, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp22:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行
	v_mad_i64_i32 v[64:65], s[0:1], v30, s6, 0  ; 行号 m × N（64 位）
	v_accvgpr_read_b32 v62, a5  ; 取出 acc[0][1][1]
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[64:65], v[64:65], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
.Ltmp23:
	.loc	4 214 87 is_stmt 1              ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v63, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v61, a6  ; 取出 acc[0][1][2]
.Ltmp24:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[64:65], v72, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
	global_store_short v[68:69], v30, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
.Ltmp25:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v62, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v60, a7  ; 取出 acc[0][1][3]
.Ltmp26:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[66:67], v30, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
.Ltmp27:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v61, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v59, a8  ; 取出 acc[0][2][0]
.Ltmp28:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[70:71], v30, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
.Ltmp29:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v60, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v58, a9  ; 取出 acc[0][2][1]
.Ltmp30:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[64:65], v30, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
.Ltmp31:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v59, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v57, a10  ; 取出 acc[0][2][2]
.Ltmp32:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[68:69], v30, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
.Ltmp33:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v58, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v56, a11  ; 取出 acc[0][2][3]
.Ltmp34:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[66:67], v30, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
.Ltmp35:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v57, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v55, a12  ; 取出 acc[0][3][0]
.Ltmp36:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[70:71], v30, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
.Ltmp37:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v56, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v54, a13  ; 取出 acc[0][3][1]
.Ltmp38:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[64:65], v30, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
.Ltmp39:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v55, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v53, a14  ; 取出 acc[0][3][2]
.Ltmp40:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[68:69], v30, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
.Ltmp41:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v54, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v52, a15  ; 取出 acc[0][3][3]
.Ltmp42:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[66:67], v30, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
.Ltmp43:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v53, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp44:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[70:71], v30, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
.Ltmp45:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v52, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v51, a16  ; 取出 acc[1][0][0]
.Ltmp46:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[64:65], v30, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
	v_or_b32_e32 v30, 16, v2  ; 行号 m = v2 + 16（mi = 1，i = 0）
.Ltmp47:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v51, v51, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp48:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行
	v_mad_i64_i32 v[52:53], s[0:1], v30, s6, 0  ; 行号 m × N（64 位）
	v_accvgpr_read_b32 v50, a17  ; 取出 acc[1][0][1]
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[52:53], v[52:53], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62  → 第 68 行
	v_or_b32_e32 v30, 17, v2  ; 行号 m = v2 + 17（mi = 1，i = 1）
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[52:53], v51, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
.Ltmp49:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v54, v50, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp50:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行
	v_mad_i64_i32 v[50:51], s[0:1], v30, s6, 0  ; 行号 m × N（64 位）
	v_accvgpr_read_b32 v49, a18  ; 取出 acc[1][0][2]
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62  → 第 68 行
	v_or_b32_e32 v30, 18, v2  ; 行号 m = v2 + 18（mi = 1，i = 2）
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[50:51], v54, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
.Ltmp51:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v49, v49, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp52:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行
	v_mad_i64_i32 v[54:55], s[0:1], v30, s6, 0  ; 行号 m × N（64 位）
	v_accvgpr_read_b32 v48, a19  ; 取出 acc[1][0][3]
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[54:55], v[54:55], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62  → 第 68 行
	v_or_b32_e32 v30, 19, v2  ; 行号 m = v2 + 19（mi = 1，i = 3）
	v_accvgpr_read_b32 v47, a20  ; 取出 acc[1][1][0]
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[54:55], v49, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
.Ltmp53:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v56, v48, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp54:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行
	v_mad_i64_i32 v[48:49], s[0:1], v30, s6, 0  ; 行号 m × N（64 位）
	v_accvgpr_read_b32 v46, a21  ; 取出 acc[1][1][1]
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[48:49], v[48:49], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
.Ltmp55:
	.loc	4 214 87 is_stmt 1              ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v47, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v45, a22  ; 取出 acc[1][1][2]
.Ltmp56:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[48:49], v56, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
	global_store_short v[52:53], v30, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
.Ltmp57:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v46, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v44, a23  ; 取出 acc[1][1][3]
.Ltmp58:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[50:51], v30, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
.Ltmp59:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v45, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v43, a24  ; 取出 acc[1][2][0]
.Ltmp60:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[54:55], v30, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
.Ltmp61:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v44, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v42, a25  ; 取出 acc[1][2][1]
.Ltmp62:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[48:49], v30, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
.Ltmp63:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v43, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v41, a26  ; 取出 acc[1][2][2]
.Ltmp64:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[52:53], v30, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
.Ltmp65:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v42, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v40, a27  ; 取出 acc[1][2][3]
.Ltmp66:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[50:51], v30, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
.Ltmp67:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v41, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v39, a28  ; 取出 acc[1][3][0]
.Ltmp68:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[54:55], v30, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
.Ltmp69:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v40, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v38, a29  ; 取出 acc[1][3][1]
.Ltmp70:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[48:49], v30, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
.Ltmp71:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v39, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v37, a30  ; 取出 acc[1][3][2]
.Ltmp72:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[52:53], v30, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
.Ltmp73:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v38, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v36, a31  ; 取出 acc[1][3][3]
.Ltmp74:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[50:51], v30, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
.Ltmp75:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v37, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp76:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[54:55], v30, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
.Ltmp77:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v30, v36, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v35, a32  ; 取出 acc[2][0][0]
.Ltmp78:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[48:49], v30, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
	v_or_b32_e32 v30, 32, v2  ; 行号 m = v2 + 32（mi = 2，i = 0）
.Ltmp79:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v35, v35, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp80:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行
	v_mad_i64_i32 v[36:37], s[0:1], v30, s6, 0  ; 行号 m × N（64 位）
	v_accvgpr_read_b32 v34, a33  ; 取出 acc[2][0][1]
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[36:37], v[36:37], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62  → 第 68 行
	v_or_b32_e32 v30, 33, v2  ; 行号 m = v2 + 33（mi = 2，i = 1）
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[36:37], v35, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
.Ltmp81:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v38, v34, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp82:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行
	v_mad_i64_i32 v[34:35], s[0:1], v30, s6, 0  ; 行号 m × N（64 位）
	v_accvgpr_read_b32 v33, a34  ; 取出 acc[2][0][2]
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62  → 第 68 行
	v_or_b32_e32 v30, 34, v2  ; 行号 m = v2 + 34（mi = 2，i = 2）
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[34:35], v38, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
.Ltmp83:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v33, v33, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp84:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行
	v_mad_i64_i32 v[38:39], s[0:1], v30, s6, 0  ; 行号 m × N（64 位）
	v_accvgpr_read_b32 v32, a35  ; 取出 acc[2][0][3]
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[38:39], v[38:39], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62  → 第 68 行
	v_or_b32_e32 v30, 35, v2  ; 行号 m = v2 + 35（mi = 2，i = 3）
	v_accvgpr_read_b32 v20, a44  ; 取出 acc[2][3][0]
	v_accvgpr_read_b32 v19, a45  ; 取出 acc[2][3][1]
	v_accvgpr_read_b32 v18, a46  ; 取出 acc[2][3][2]
	v_accvgpr_read_b32 v17, a47  ; 取出 acc[2][3][3]
	v_accvgpr_read_b32 v24, a40  ; 取出 acc[2][2][0]
	v_accvgpr_read_b32 v23, a41  ; 取出 acc[2][2][1]
	v_accvgpr_read_b32 v22, a42  ; 取出 acc[2][2][2]
	v_accvgpr_read_b32 v21, a43  ; 取出 acc[2][2][3]
	v_accvgpr_read_b32 v31, a36  ; 取出 acc[2][1][0]
	v_accvgpr_read_b32 v27, a37  ; 取出 acc[2][1][1]
	v_accvgpr_read_b32 v26, a38  ; 取出 acc[2][1][2]
	v_accvgpr_read_b32 v25, a39  ; 取出 acc[2][1][3]
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[38:39], v33, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
.Ltmp85:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v40, v32, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp86:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行
	v_mad_i64_i32 v[32:33], s[0:1], v30, s6, 0  ; 行号 m × N（64 位）
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[32:33], v[32:33], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
.Ltmp87:
	.loc	4 214 87 is_stmt 1              ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	s_nop 0  ; 空几个周期，避开硬件的时序冲突
	v_cvt_pk_bf16_f32 v30, v31, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_cvt_pk_bf16_f32 v27, v27, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_cvt_pk_bf16_f32 v26, v26, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_cvt_pk_bf16_f32 v25, v25, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_cvt_pk_bf16_f32 v24, v24, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_cvt_pk_bf16_f32 v23, v23, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_cvt_pk_bf16_f32 v22, v22, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_cvt_pk_bf16_f32 v21, v21, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_cvt_pk_bf16_f32 v20, v20, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_cvt_pk_bf16_f32 v19, v19, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_cvt_pk_bf16_f32 v18, v18, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_cvt_pk_bf16_f32 v17, v17, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v16, a48  ; 取出 acc[3][0][0]
.Ltmp88:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[32:33], v40, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
	global_store_short v[36:37], v30, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
	global_store_short v[34:35], v27, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
	global_store_short v[38:39], v26, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
	global_store_short v[32:33], v25, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
	global_store_short v[36:37], v24, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
	global_store_short v[34:35], v23, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
	global_store_short v[38:39], v22, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
	global_store_short v[32:33], v21, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
	global_store_short v[36:37], v20, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
	global_store_short v[34:35], v19, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
	global_store_short v[38:39], v18, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
	global_store_short v[32:33], v17, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
	v_or_b32_e32 v17, 48, v2  ; 行号 m = v2 + 48（mi = 3，i = 0）
.Ltmp89:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v18, v16, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp90:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行
	v_mad_i64_i32 v[16:17], s[0:1], v17, s6, 0  ; 行号 m × N（64 位）
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[16:17], v[16:17], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
	v_accvgpr_read_b32 v15, a49  ; 取出 acc[3][0][1]
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[16:17], v18, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62  → 第 68 行
	v_or_b32_e32 v18, 49, v2  ; 行号 m = v2 + 49（mi = 3，i = 1）
.Ltmp91:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v15, v15, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp92:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行
	v_mad_i64_i32 v[18:19], s[0:1], v18, s6, 0  ; 行号 m × N（64 位）
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
	v_accvgpr_read_b32 v14, a50  ; 取出 acc[3][0][2]
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[18:19], v15, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62  → 第 68 行
	v_or_b32_e32 v15, 50, v2  ; 行号 m = v2 + 50（mi = 3，i = 2）
.Ltmp93:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v20, v14, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp94:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行
	v_mad_i64_i32 v[14:15], s[0:1], v15, s6, 0  ; 行号 m × N（64 位）
	v_accvgpr_read_b32 v13, a51  ; 取出 acc[3][0][3]
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[14:15], v[14:15], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
	.loc	2 68 62 is_stmt 1               ; ./v0_naive.h:68:62  → 第 68 行
	v_or_b32_e32 v2, 51, v2  ; 行号 m = v2 + 51（mi = 3，i = 3）
	v_accvgpr_read_b32 v12, a52  ; 取出 acc[3][1][0]
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[14:15], v20, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
.Ltmp95:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v13, v13, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp96:
	.loc	2 70 31                         ; ./v0_naive.h:70:31  → 第 70 行
	v_mad_i64_i32 v[20:21], s[0:1], v2, s6, 0  ; 行号 m × N（64 位）
	v_accvgpr_read_b32 v11, a53  ; 取出 acc[3][1][1]
	.loc	2 70 17 is_stmt 0               ; ./v0_naive.h:70:17  → 第 70 行
	v_lshl_add_u64 v[20:21], v[20:21], 1, v[28:29]  ; × 2 字节，加上 C 第 0 行第 n 列的地址，得到 &C[m][n]
.Ltmp97:
	.loc	4 214 87 is_stmt 1              ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v2, v12, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v10, a54  ; 取出 acc[3][1][2]
.Ltmp98:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[20:21], v13, off  ; 写 2 字节 bf16 到 C[m][n]（ni = 0）
	global_store_short v[16:17], v2, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
.Ltmp99:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v2, v11, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v9, a55  ; 取出 acc[3][1][3]
.Ltmp100:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[18:19], v2, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
.Ltmp101:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v2, v10, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v8, a56  ; 取出 acc[3][2][0]
.Ltmp102:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[14:15], v2, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
.Ltmp103:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v2, v9, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v7, a57  ; 取出 acc[3][2][1]
.Ltmp104:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[20:21], v2, off offset:32  ; 写 2 字节 bf16 到 C[m][n + 16]（ni = 1）
.Ltmp105:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v2, v8, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v6, a58  ; 取出 acc[3][2][2]
.Ltmp106:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[16:17], v2, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
.Ltmp107:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v2, v7, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v5, a59  ; 取出 acc[3][2][3]
.Ltmp108:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[18:19], v2, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
.Ltmp109:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v2, v6, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v4, a60  ; 取出 acc[3][3][0]
.Ltmp110:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[14:15], v2, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
.Ltmp111:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v2, v5, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_accvgpr_read_b32 v3, a61  ; 取出 acc[3][3][1]
	v_accvgpr_read_b32 v1, a62  ; 取出 acc[3][3][2]
	v_accvgpr_read_b32 v0, a63  ; 取出 acc[3][3][3]
.Ltmp112:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[20:21], v2, off offset:64  ; 写 2 字节 bf16 到 C[m][n + 32]（ni = 2）
.Ltmp113:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v2, v4, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp114:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[16:17], v2, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
.Ltmp115:
	.loc	4 214 87                        ; /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:214:87 @[ /opt/rocm-7.2.4/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:442:18 @[ ./v0_naive.h:70:44 ] ]  → __float2bfloat16 的实现（被第 70 行调用）
	v_cvt_pk_bf16_f32 v2, v3, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_cvt_pk_bf16_f32 v1, v1, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
	v_cvt_pk_bf16_f32 v0, v0, s0  ; fp32 → bf16（第 70 行的 __float2bfloat16）
.Ltmp116:
	.loc	2 70 42                         ; ./v0_naive.h:70:42  → 第 70 行
	global_store_short v[18:19], v2, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
	global_store_short v[14:15], v1, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
	global_store_short v[20:21], v0, off offset:96  ; 写 2 字节 bf16 到 C[m][n + 48]（ni = 3）
	.loc	2 72 1                          ; ./v0_naive.h:72:1  → 第 72 行：}
	s_endpgm  ; 这个 wave 结束（第 72 行的 }）
.Ltmp117:
; ============================================================================================
; 【5】kernel 描述符：运行时按这些设置启动 kernel（寄存器、LDS、参数区……）
; ============================================================================================
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN2v013gemm_v0_naiveE7Problem  ; kernel 描述符开始
		.amdhsa_group_segment_fixed_size 16384  ; LDS 16384 字节：sA、sB 各 8 KB
		.amdhsa_private_segment_fixed_size 0  ; 每个 lane 的私有内存（scratch）0 字节：没有寄存器溢出
		.amdhsa_kernarg_size 312  ; 参数区 312 字节：Problem 的 56 字节 + 运行时附加的隐藏参数
		.amdhsa_user_sgpr_count 2  ; 启动时预先填好 2 个 SGPR：s[0:1] = 参数区地址
		.amdhsa_user_sgpr_dispatch_ptr 0  ; 不需要
		.amdhsa_user_sgpr_queue_ptr 0  ; 不需要
		.amdhsa_user_sgpr_kernarg_segment_ptr 1  ; s[0:1] 放参数区地址
		.amdhsa_user_sgpr_dispatch_id 0  ; 不需要
		.amdhsa_user_sgpr_kernarg_preload_length 0  ; 不把参数预先装进寄存器
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0  ; 不需要
		.amdhsa_uses_dynamic_stack 0  ; 没有动态栈
		.amdhsa_enable_private_segment 0  ; 没有私有内存
		.amdhsa_system_sgpr_workgroup_id_x 1  ; s2 = blockIdx.x
		.amdhsa_system_sgpr_workgroup_id_y 1  ; s3 = blockIdx.y
		.amdhsa_system_sgpr_workgroup_id_z 0  ; 不需要 blockIdx.z
		.amdhsa_system_sgpr_workgroup_info 0  ; 不需要
		.amdhsa_system_vgpr_workitem_id 0  ; 只要 threadIdx.x，放在 v0
		.amdhsa_next_free_vgpr 140  ; VGPR + AGPR 合计 140 行（每个 lane）
		.amdhsa_next_free_sgpr 38  ; 用到 38 个 SGPR
		.amdhsa_accum_offset 76  ; AGPR 从第 76 行开始（73 个 VGPR 之后对齐到 4）
		.amdhsa_reserve_vcc 1  ; 预留 vcc
		.amdhsa_float_round_mode_32 0  ; fp32 舍入：就近舍入
		.amdhsa_float_round_mode_16_64 0  ; fp16/fp64 舍入：就近舍入
		.amdhsa_float_denorm_mode_32 3  ; fp32 保留非常小的数（denormal），不清零
		.amdhsa_float_denorm_mode_16_64 3  ; fp16/fp64 同上
		.amdhsa_dx10_clamp 1  ; 浮点行为设置
		.amdhsa_ieee_mode 1  ; 浮点按 IEEE 规则处理 NaN
		.amdhsa_fp16_overflow 0  ; 默认
		.amdhsa_tg_split 0  ; 默认
		.amdhsa_exception_fp_ieee_invalid_op 0  ; 不开这类浮点、整数异常
		.amdhsa_exception_fp_denorm_src 0  ; 不开这类浮点、整数异常
		.amdhsa_exception_fp_ieee_div_zero 0  ; 不开这类浮点、整数异常
		.amdhsa_exception_fp_ieee_overflow 0  ; 不开这类浮点、整数异常
		.amdhsa_exception_fp_ieee_underflow 0  ; 不开这类浮点、整数异常
		.amdhsa_exception_fp_ieee_inexact 0  ; 不开这类浮点、整数异常
		.amdhsa_exception_int_div_zero 0  ; 不开这类浮点、整数异常
	.end_amdhsa_kernel  ; kernel 描述符结束
	.text  ; 回到代码段
.Lfunc_end0:  ; 函数结束的位置
	.size	_ZN2v013gemm_v0_naiveE7Problem, .Lfunc_end0-_ZN2v013gemm_v0_naiveE7Problem  ; kernel 的大小 = 结束地址 − 开始地址
	.cfi_endproc  ; 调用帧信息：函数结束
                                        ; -- End function
	.set _ZN2v013gemm_v0_naiveE7Problem.num_vgpr, 73  ; VGPR 73 个
	.set _ZN2v013gemm_v0_naiveE7Problem.num_agpr, 64  ; AGPR 64 个
	.set _ZN2v013gemm_v0_naiveE7Problem.numbered_sgpr, 38  ; SGPR 38 个
	.set _ZN2v013gemm_v0_naiveE7Problem.num_named_barrier, 0  ; 没有命名 barrier
	.set _ZN2v013gemm_v0_naiveE7Problem.private_seg_size, 0  ; 没有私有内存
	.set _ZN2v013gemm_v0_naiveE7Problem.uses_vcc, 1  ; 用到了 vcc
	.set _ZN2v013gemm_v0_naiveE7Problem.uses_flat_scratch, 0  ; 没用 scratch
	.set _ZN2v013gemm_v0_naiveE7Problem.has_dyn_sized_stack, 0  ; 没有动态栈
	.set _ZN2v013gemm_v0_naiveE7Problem.has_recursion, 0  ; 没有递归
	.set _ZN2v013gemm_v0_naiveE7Problem.has_indirect_call, 0  ; 没有间接调用
	.section	.AMDGPU.csdata,"",@progbits  ; 编译器的资源统计（全是注释）
; Kernel info:
; codeLenInByte = 9948  → 代码 9,948 字节，printf 占了一大半
; TotalNumSgprs: 44  → 加上 vcc 等保留的，共 44 个 SGPR
; NumVgprs: 73  → VGPR 73 个
; NumAgprs: 64  → AGPR 64 个：acc[4][4] 共 64 个 float
; TotalNumVgprs: 140  → 76 + 64 = 140
; ScratchSize: 0  → 没有溢出到私有内存
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 16384 bytes/workgroup (compile time only)  → LDS 16 KB
; SGPRBlocks: 5  → SGPR 按块分配的块数（编码值）
; VGPRBlocks: 17  → VGPR 按 8 行一组分配，编码值 17 表示 18 组 = 144 行
; NumSGPRsForWavesPerEU: 44  → 算 occupancy 时按 44 个 SGPR 算
; NumVGPRsForWavesPerEU: 140  → 算 occupancy 时按 140 行 VGPR 算（再按 8 行一组取整到 144）
; AccumOffset: 76  → AGPR 从第 76 行开始
; Occupancy: 3  → 每个 SIMD 最多同时驻留 3 个 wave：512 ÷ 144 ≈ 3.6
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0  → 不用 scratch
; COMPUTE_PGM_RSRC2:USER_SGPR: 2  → 预先填 2 个 SGPR（参数区地址）
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1  → 要 blockIdx.x
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1  → 要 blockIdx.y
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0  → 不要 blockIdx.z
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0  → 只要 threadIdx.x
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 18  → 编码值 18 表示 (18+1)×4 = 76
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864  ; 代码末尾对齐
	.fill 256, 4, 3212836864  ; 填 256 条 s_nop 0（编码 0xBF800000）：编译器加的填充，防止取指令时预取到代码后面的无关数据
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	.str,@object                    ; @.str  → 下面是 printf 的格式串
	.section	.rodata.str1.1,"aMS",@progbits,1  ; 只读字符串段
.str:  ; 格式串的标签，【2b】里 s_getpc 算的就是它的地址
	.asciz	"load time: %llu\n"  ; 格式串内容，结尾自动补一个 0
	.size	.str, 17  ; 17 字节

	.type	__hip_cuid_1fd510d24433d8d9,@object ; @__hip_cuid_1fd510d24433d8d9  → HIP 编译单元的 ID：运行时用它把主机和设备代码对上
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_1fd510d24433d8d9
__hip_cuid_1fd510d24433d8d9:
	.byte	0                               ; 0x0
	.size	__hip_cuid_1fd510d24433d8d9, 1

; ============================================================================================
; 【6】DWARF 调试信息（到 .note.GNU-stack 之前）
; 记录源码文件、函数、内联关系、地址范围，供 rocgdb、反汇编器、ATT 把指令对回源码。
; 这些是编译器生成的数据表，不会执行，不用逐行读。
; ============================================================================================
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
	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"  ; 编译器版本
	.section	".note.GNU-stack","",@progbits  ; 给链接器的标记
	.addrsig  ; 地址重要性表，给链接器用
	.addrsig_sym __hip_cuid_1fd510d24433d8d9  ; 同上
; ============================================================================================
; 【7】kernel 元数据（YAML 格式，注释用 #）：运行时读它来准备参数、分配资源
; ============================================================================================
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     64    # AGPR 64 个
    .args:    # 参数列表：每项是参数在参数区里的偏移、大小、种类
      - .offset:         0
        .size:           56
        .value_kind:     by_value    # Problem p：M、N、K（12 字节）+ 补齐 4 字节 + 5 个指针（40 字节）= 56 字节，按值传
      - .offset:         56
        .size:           4
        .value_kind:     hidden_block_count_x    # 以下是运行时附加的隐藏参数。grid 在 x 方向的 workgroup 个数
      - .offset:         60
        .size:           4
        .value_kind:     hidden_block_count_y    # grid 在 y 方向的 workgroup 个数
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_z    # grid 在 z 方向的 workgroup 个数
      - .offset:         68
        .size:           2
        .value_kind:     hidden_group_size_x    # workgroup 在 x 方向的大小
      - .offset:         70
        .size:           2
        .value_kind:     hidden_group_size_y    # workgroup 在 y 方向的大小
      - .offset:         72
        .size:           2
        .value_kind:     hidden_group_size_z    # workgroup 在 z 方向的大小
      - .offset:         74
        .size:           2
        .value_kind:     hidden_remainder_x    # 最后一个不满的 workgroup 的大小
      - .offset:         76
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         78
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_x    # 全局偏移（这里没用）
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         120
        .size:           2
        .value_kind:     hidden_grid_dims    # grid 的维数
      - .offset:         136
        .size:           8
        .value_kind:     hidden_hostcall_buffer    # printf 用的 hostcall 缓冲区地址，在偏移 136 = 0x88；【2】开头的 s_load ... 0x88 读的就是它
    .group_segment_fixed_size: 16384    # LDS 16384 字节
    .kernarg_segment_align: 8    # 参数区按 8 字节对齐
    .kernarg_segment_size: 312    # 参数区 312 字节
    .language:       OpenCL C    # HIP kernel 在这里也记成 OpenCL C，不用管
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256    # 每个 workgroup 最多 256 个线程：__launch_bounds__(256)
    .name:           _ZN2v013gemm_v0_naiveE7Problem    # kernel 名
    .private_segment_fixed_size: 0    # 没有私有内存
    .sgpr_count:     44    # SGPR 44 个
    .sgpr_spill_count: 0    # 没有 SGPR 溢出
    .symbol:         _ZN2v013gemm_v0_naiveE7Problem.kd    # kernel 描述符的符号（.kd）
    .uniform_work_group_size: 1    # 所有 workgroup 一样大
    .uses_dynamic_stack: false    # 没有动态栈
    .vgpr_count:     140    # VGPR + AGPR 合计 140
    .vgpr_spill_count: 0    # 没有 VGPR 溢出
    .wavefront_size: 64    # 一个 wave 64 个 lane
amdhsa.target:   amdgcn-amd-amdhsa--gfx950    # 目标 GPU
amdhsa.version:    # 元数据格式版本
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits  ; 行号表：.loc 最后就变成这张表
.Lline_table_start0:
