#ifndef OS_ELM_CORE_H
#define OS_ELM_CORE_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// Network dimensions (compile-time)
#define STATE_DIM   6
#define HIDDEN_DIM  64
#define NUM_ACTIONS 3

// Fixed-point type: 32-bit, 12 integer bits (Q20)
typedef ap_fixed<32, 12> fixed_t;

// AXI-Stream word: 32-bit data, no sideband signals
typedef ap_axis<32, 0, 0, 0> axis_word_t;

// DMA opcodes (raw integers in first stream word)
#define OP_PREDICT_Q    0
#define OP_PREDICT_TGT  1
#define OP_TRAIN_SEQ    2
#define OP_LOAD_WEIGHTS 3
#define OP_READ_WEIGHTS 4
#define OP_SYNC_TARGET  5

// Top-level function
void os_elm_core(
    hls::stream<axis_word_t> &in_stream,
    hls::stream<axis_word_t> &out_stream
);

#endif // OS_ELM_CORE_H
