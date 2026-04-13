// os_elm_core.cpp — Vitis HLS kernel for OS-ELM Q-Network on AUP-ZU3
//
// Implements predict (θ₁ and θ₂) and sequential RLS training (θ₁)
// for the multi-output OS-ELM-L2-Lipschitz DQN.
//
// All on-chip state is in static arrays (synthesised to BRAM):
//   W_in     [STATE_DIM][HIDDEN_DIM]   — input weights α  (fixed after load)
//   b        [HIDDEN_DIM]              — bias              (fixed after load)
//   beta     [HIDDEN_DIM][NUM_ACTIONS] — output weights θ₁ (updated by train)
//   beta_tgt [HIDDEN_DIM][NUM_ACTIONS] — target net θ₂     (copied from θ₁)
//   P        [HIDDEN_DIM][HIDDEN_DIM]  — RLS precision      (updated by train)
//
// Communication: single AXI-Stream in + AXI-Stream out via AXI DMA.
// Protocol is documented in training_loop_jupyter.py.
//
// ── Resource estimates (XCZU3EG) ─────────────────────────────
//   BRAM: P (32 BRAM36K) + W_in/b/beta/beta_tgt (~6 BRAM36K) ≈ 18%
//   DSP:  64 peak (time-multiplexed across stages)            ≈ 18%
//   FF/LUT: moderate (partition registers + control logic)
//
// ── Tuning ───────────────────────────────────────────────────
//   If timing fails at target frequency, reduce P partition:
//     Change: ARRAY_PARTITION variable=P complete dim=2
//     To:     ARRAY_PARTITION variable=P cyclic factor=16 dim=2
//     And:    Change corresponding UNROLL to UNROLL factor=16
//   This trades 4× latency for much less routing pressure.
//
// ── Suggested Vitis HLS Tcl directives ───────────────────────
//   set_clock_uncertainty 12.5%
//   config_compile -pipeline_loops 64
//   config_bind_op -op mul -impl dsp       ;# force muls to DSP48E2

#include "os_elm_core.h"


//  Helpers (force-inlined — zero overhead)
static fixed_t read_fixed(hls::stream<axis_word_t> &s) {
    #pragma HLS INLINE
    axis_word_t word = s.read();
    fixed_t val;
    val.range(31, 0) = word.data;
    return val;
}

static int read_int(hls::stream<axis_word_t> &s) {
    #pragma HLS INLINE
    axis_word_t word = s.read();
    return (int)word.data;
}

static void write_fixed(hls::stream<axis_word_t> &s, fixed_t val, bool last = false) {
    #pragma HLS INLINE
    axis_word_t word;
    word.data = val.range(31, 0);
    word.keep = 0xF;
    word.strb = 0xF;
    word.last = last ? 1 : 0;
    word.user = 0;
    word.id   = 0;
    word.dest = 0;
    s.write(word);
}

static void write_int(hls::stream<axis_word_t> &s, int val, bool last = false) {
    #pragma HLS INLINE
    axis_word_t word;
    word.data = val;
    word.keep = 0xF;
    word.strb = 0xF;
    word.last = last ? 1 : 0;
    word.user = 0;
    word.id   = 0;
    word.dest = 0;
    s.write(word);
}

static fixed_t relu(fixed_t x) {
    #pragma HLS INLINE
    return (x > fixed_t(0)) ? x : fixed_t(0);
}

//  Top-level kernel
void os_elm_core(
    hls::stream<axis_word_t> &in_stream,
    hls::stream<axis_word_t> &out_stream
) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

    // Persistent BRAM storage
    static fixed_t W_in[STATE_DIM][HIDDEN_DIM];
    static fixed_t b[HIDDEN_DIM];
    static fixed_t beta[HIDDEN_DIM][NUM_ACTIONS];
    static fixed_t beta_tgt[HIDDEN_DIM][NUM_ACTIONS];
    static fixed_t P[HIDDEN_DIM][HIDDEN_DIM];

    // BRAM partitioning 
    // W_in: partition dim 1 (STATE_DIM=6) so all 6 inputs
    //       are read in parallel when computing one hidden node.
    #pragma HLS ARRAY_PARTITION variable=W_in complete dim=1

    // beta / beta_tgt: partition dim 2 (NUM_ACTIONS=3) so all
    //       3 action columns are read/written in parallel.
    #pragma HLS ARRAY_PARTITION variable=beta     complete dim=2
    #pragma HLS ARRAY_PARTITION variable=beta_tgt complete dim=2

    // P: partition dim 2 (HIDDEN_DIM=64) so an entire row
    //    P[r][0..63] is accessed in one cycle.  This is the
    //    single most impactful pragma — enables II=1 on every
    //    matrix-vector loop involving P.
    //    Cost: 64 BRAM18K banks × 64-deep = 32 BRAM36K (15%).
    #pragma HLS ARRAY_PARTITION variable=P complete dim=2

    // Compile-time constants
    // Replace P /= λ with P *= λ_inv to avoid 64 HW dividers.
    static const fixed_t LAM_INV  = fixed_t(1.0 / 0.9999);  // ≈ 1.0001
    static const fixed_t P_MAX    = fixed_t(10.0);           // 1/REG
    static const fixed_t REG_INV  = fixed_t(10.0);           // 1/REG

    // Read opcode
    int opcode = read_int(in_stream);

    
    //  OP 0 / 1: PREDICT (θ₁ or θ₂)
    //  Latency: ~6 (stream reads) + 64 (hidden) + 64 (dot) = ~134 cycles
    
    if (opcode == OP_PREDICT_Q || opcode == OP_PREDICT_TGT) {

        bool use_tgt = (opcode == OP_PREDICT_TGT);

        // Read state[6]
        fixed_t state[STATE_DIM];
        #pragma HLS ARRAY_PARTITION variable=state complete
        PRED_READ: for (int i = 0; i < STATE_DIM; i++) {
            state[i] = read_fixed(in_stream);
        }

        // Hidden layer: h = ReLU(state @ W_in + b) 
        //    W_in partitioned on dim 1 → 6 reads in parallel.
        //    Inner loop fully unrolled → 6 MACs per cycle.
        //    Outer loop pipelined II=1 → 64 cycles total.
        fixed_t h[HIDDEN_DIM];
        #pragma HLS ARRAY_PARTITION variable=h complete
        PRED_HIDDEN: for (int j = 0; j < HIDDEN_DIM; j++) {
            #pragma HLS PIPELINE II=1
            fixed_t acc = b[j];
            PRED_H_MAC: for (int i = 0; i < STATE_DIM; i++) {
                #pragma HLS UNROLL
                acc += state[i] * W_in[i][j];
            }
            h[j] = relu(acc);
        }

        // Q-values: q = h @ beta (or beta_tgt
        //    All 3 actions computed simultaneously (beta dim 2
        //    partitioned).  j loop pipelined → 64 cycles total.
        fixed_t q[NUM_ACTIONS];
        #pragma HLS ARRAY_PARTITION variable=q complete
        PRED_Q_INIT: for (int a = 0; a < NUM_ACTIONS; a++) {
            #pragma HLS UNROLL
            q[a] = fixed_t(0);
        }
        PRED_Q_DOT: for (int j = 0; j < HIDDEN_DIM; j++) {
            #pragma HLS PIPELINE II=1
            PRED_Q_ACT: for (int a = 0; a < NUM_ACTIONS; a++) {
                #pragma HLS UNROLL
                fixed_t bval = use_tgt ? beta_tgt[j][a] : beta[j][a];
                q[a] += h[j] * bval;
            }
        }

        // Write 3 Q-values
        PRED_WRITE: for (int a = 0; a < NUM_ACTIONS; a++) {
            write_fixed(out_stream, q[a], (a == NUM_ACTIONS - 1));
        }
    }

    
    //  OP 2: TRAIN_SEQ — RLS rank-1 update on θ₁
    //  Matches os_elm_dqn.py update_single().
    //
    //  Estimated latency (all loops II=1):
    //    stream reads:  ~9
    //    hidden:        64
    //    P *= λ_inv:    64
    //    P clamp:       64 + 64 (worst case)
    //    Ph = P @ h:    64
    //    denom:         64
    //    k = Ph/denom:  64 + 1 (one real division)
    //    e + β update:  64 + 64
    //    hP = h^T P:    64
    //    P -= outer:    64
    //  Total: ~640 cycles (+ safety path if denom reset)
    
    else if (opcode == OP_TRAIN_SEQ) {

        // Read inputs 
        fixed_t state[STATE_DIM];
        #pragma HLS ARRAY_PARTITION variable=state complete
        TRAIN_READ_S: for (int i = 0; i < STATE_DIM; i++) {
            state[i] = read_fixed(in_stream);
        }
        int action = read_int(in_stream);
        fixed_t target = read_fixed(in_stream);

        // Hidden layer: h = ReLU(state @ W_in + b)
        fixed_t h[HIDDEN_DIM];
        #pragma HLS ARRAY_PARTITION variable=h complete
        TRAIN_HIDDEN: for (int j = 0; j < HIDDEN_DIM; j++) {
            #pragma HLS PIPELINE II=1
            fixed_t acc = b[j];
            TRAIN_H_MAC: for (int i = 0; i < STATE_DIM; i++) {
                #pragma HLS UNROLL
                acc += state[i] * W_in[i][j];
            }
            h[j] = relu(acc);
        }

        // Forgetting: P *= (1/λ) 
        //    Multiply instead of divide: saves 64 HW dividers.
        //    P partitioned on dim 2 → full row written per cycle.
        TRAIN_FORGET: for (int r = 0; r < HIDDEN_DIM; r++) {
            #pragma HLS PIPELINE II=1
            TRAIN_FORGET_C: for (int c = 0; c < HIDDEN_DIM; c++) {
                #pragma HLS UNROLL
                P[r][c] = P[r][c] * LAM_INV;
            }
        }

        // Clamp: if max(diag(P)) > P_MAX, scale P
        fixed_t max_diag = P[0][0];
        TRAIN_MAXDIAG: for (int i = 1; i < HIDDEN_DIM; i++) {
            #pragma HLS PIPELINE II=1
            if (P[i][i] > max_diag) max_diag = P[i][i];
        }

        if (max_diag > P_MAX) {
            fixed_t scale = P_MAX / max_diag;    // single division
            TRAIN_CLAMP: for (int r = 0; r < HIDDEN_DIM; r++) {
                #pragma HLS PIPELINE II=1
                TRAIN_CLAMP_C: for (int c = 0; c < HIDDEN_DIM; c++) {
                    #pragma HLS UNROLL
                    P[r][c] = P[r][c] * scale;
                }
            }
        }

        // Ph = P @ h 
        //    P dim 2 partitioned + h partitioned → full row
        //    dot product per cycle.
        fixed_t Ph[HIDDEN_DIM];
        #pragma HLS ARRAY_PARTITION variable=Ph complete
        TRAIN_PH: for (int r = 0; r < HIDDEN_DIM; r++) {
            #pragma HLS PIPELINE II=1
            fixed_t acc = fixed_t(0);
            TRAIN_PH_C: for (int c = 0; c < HIDDEN_DIM; c++) {
                #pragma HLS UNROLL
                acc += P[r][c] * h[c];
            }
            Ph[r] = acc;
        }

        // denom = 1 + h^T @ Ph
        //    h and Ph are fully partitioned → all products
        //    computed in parallel, reduced with adder tree.
        fixed_t denom = fixed_t(1);
        TRAIN_DENOM: for (int j = 0; j < HIDDEN_DIM; j++) {
            #pragma HLS PIPELINE II=1
            denom += h[j] * Ph[j];
        }

        // Safety: if denom collapsed, reset P = I / REG
        fixed_t DENOM_MIN = fixed_t(0.00001);
        if (denom < DENOM_MIN) {
            TRAIN_PRST: for (int r = 0; r < HIDDEN_DIM; r++) {
                #pragma HLS PIPELINE II=1
                TRAIN_PRST_C: for (int c = 0; c < HIDDEN_DIM; c++) {
                    #pragma HLS UNROLL
                    P[r][c] = (r == c) ? REG_INV : fixed_t(0);
                }
            }
            // Recompute Ph and denom after reset
            TRAIN_PH2: for (int r = 0; r < HIDDEN_DIM; r++) {
                #pragma HLS PIPELINE II=1
                fixed_t acc2 = fixed_t(0);
                TRAIN_PH2_C: for (int c = 0; c < HIDDEN_DIM; c++) {
                    #pragma HLS UNROLL
                    acc2 += P[r][c] * h[c];
                }
                Ph[r] = acc2;
            }
            denom = fixed_t(1);
            TRAIN_DENOM2: for (int j = 0; j < HIDDEN_DIM; j++) {
                #pragma HLS PIPELINE II=1
                denom += h[j] * Ph[j];
            }
        }

        // k = Ph / denom
        //    One real division (1/denom), then 64 multiplies.
        fixed_t denom_inv = fixed_t(1) / denom;
        fixed_t k[HIDDEN_DIM];
        #pragma HLS ARRAY_PARTITION variable=k complete
        TRAIN_K: for (int j = 0; j < HIDDEN_DIM; j++) {
            #pragma HLS PIPELINE II=1
            k[j] = Ph[j] * denom_inv;
        }

        // e = target - h^T @ beta[:,action]
        fixed_t e = target;
        TRAIN_ERR: for (int j = 0; j < HIDDEN_DIM; j++) {
            #pragma HLS PIPELINE II=1
            e -= h[j] * beta[j][action];
        }

        // beta[:,action] += k * e
        TRAIN_BETA: for (int j = 0; j < HIDDEN_DIM; j++) {
            #pragma HLS PIPELINE II=1
            beta[j][action] += k[j] * e;
        }

        // hP = h^T @ P  (row vector)
        //    Restructured: iterate j (outer), accumulate into
        //    hP[c] for all c.  P[j][:] is a full row read
        //    (dim 2 partitioned) → 64 MACs in parallel.
        fixed_t hP[HIDDEN_DIM];
        #pragma HLS ARRAY_PARTITION variable=hP complete
        TRAIN_HP_INIT: for (int c = 0; c < HIDDEN_DIM; c++) {
            #pragma HLS UNROLL
            hP[c] = fixed_t(0);
        }
        TRAIN_HP: for (int j = 0; j < HIDDEN_DIM; j++) {
            #pragma HLS PIPELINE II=1
            TRAIN_HP_C: for (int c = 0; c < HIDDEN_DIM; c++) {
                #pragma HLS UNROLL
                hP[c] += h[j] * P[j][c];
            }
        }

        // P -= outer(k, hP)
        //    For each row r: P[r][c] -= k[r] * hP[c] for all c.
        //    k[r] broadcast, hP fully partitioned, P row write.
        TRAIN_PUPD: for (int r = 0; r < HIDDEN_DIM; r++) {
            #pragma HLS PIPELINE II=1
            fixed_t kr = k[r];
            TRAIN_PUPD_C: for (int c = 0; c < HIDDEN_DIM; c++) {
                #pragma HLS UNROLL
                P[r][c] -= kr * hP[c];
            }
        }

        // complete (send ack)
        write_int(out_stream, 1, true);
    }

    
    //  OP 3: LOAD_WEIGHTS — fill BRAMs from DMA stream
    //  Order: W_in[S*H] + b[H] + beta[H*A] + P[H*H]  (C-order)
    //  Not performance-critical (happens once per init).
    
    else if (opcode == OP_LOAD_WEIGHTS) {

        LOAD_W: for (int i = 0; i < STATE_DIM; i++) {
            for (int j = 0; j < HIDDEN_DIM; j++) {
                #pragma HLS PIPELINE II=1
                W_in[i][j] = read_fixed(in_stream);
            }
        }
        LOAD_B: for (int j = 0; j < HIDDEN_DIM; j++) {
            #pragma HLS PIPELINE II=1
            b[j] = read_fixed(in_stream);
        }
        LOAD_BETA: for (int j = 0; j < HIDDEN_DIM; j++) {
            for (int a = 0; a < NUM_ACTIONS; a++) {
                #pragma HLS PIPELINE II=1
                beta[j][a] = read_fixed(in_stream);
            }
        }
        LOAD_P: for (int r = 0; r < HIDDEN_DIM; r++) {
            for (int c = 0; c < HIDDEN_DIM; c++) {
                #pragma HLS PIPELINE II=1
                P[r][c] = read_fixed(in_stream);
            }
        }

        // θ₂ ← θ₁
        LOAD_SYNC: for (int j = 0; j < HIDDEN_DIM; j++) {
            #pragma HLS PIPELINE II=1
            LOAD_SYNC_A: for (int a = 0; a < NUM_ACTIONS; a++) {
                #pragma HLS UNROLL
                beta_tgt[j][a] = beta[j][a];
            }
        }

        write_int(out_stream, 1, true);
    }

    
    //  OP 4: READ_WEIGHTS — stream beta[H*A] + P[H*H] back
    //  Not performance-critical (happens on save).
    
    else if (opcode == OP_READ_WEIGHTS) {

        int total_words = HIDDEN_DIM * NUM_ACTIONS + HIDDEN_DIM * HIDDEN_DIM;
        int count = 0;

        READ_BETA: for (int j = 0; j < HIDDEN_DIM; j++) {
            for (int a = 0; a < NUM_ACTIONS; a++) {
                #pragma HLS PIPELINE II=1
                count++;
                write_fixed(out_stream, beta[j][a], (count == total_words));
            }
        }
        READ_P: for (int r = 0; r < HIDDEN_DIM; r++) {
            for (int c = 0; c < HIDDEN_DIM; c++) {
                #pragma HLS PIPELINE II=1
                count++;
                write_fixed(out_stream, P[r][c], (count == total_words));
            }
        }
    }

    
    //  OP 5: SYNC_TARGET — copy θ₁ β → θ₂ β
    
    else if (opcode == OP_SYNC_TARGET) {

        SYNC_LOOP: for (int j = 0; j < HIDDEN_DIM; j++) {
            #pragma HLS PIPELINE II=1
            SYNC_ACT: for (int a = 0; a < NUM_ACTIONS; a++) {
                #pragma HLS UNROLL
                beta_tgt[j][a] = beta[j][a];
            }
        }

        write_int(out_stream, 1, true);
    }

    
    //  Unknown opcode
    
    else {
        write_int(out_stream, -1, true);
    }
}
