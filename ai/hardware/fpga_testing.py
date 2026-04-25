# fpga_testing.py — FPGA kernel smoke test on AUP-ZU3 (Jupyter)
#
# Runs each opcode of os_elm_core over AXI DMA and verifies the
# result against a numpy reference. Mirrors os_elm_tb.cpp.
#
# Each test is a standalone function. Failures are caught and
# reported; subsequent tests still run. Comment out any test
# call at the bottom to skip it.
#
# Usage in a notebook cell:
#   %run fpga_testing.py

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "ai" / "software"))

import numpy as np
import time
from pynq import Overlay, allocate

from training_config import HIDDEN_DIM, NUM_ACTIONS, REG
from encoder import STATE_DIM

# ══════════════════════════════════════════════════════════════
# Q20 fixed-point helpers
# ══════════════════════════════════════════════════════════════
FRAC_BITS = 20
SCALE = 1 << FRAC_BITS

def to_q20(val):
    return np.round(np.asarray(val, dtype=np.float64) * SCALE).astype(np.int32)

def from_q20(val):
    return np.asarray(val, dtype=np.float64) / SCALE

# ══════════════════════════════════════════════════════════════
# DMA opcodes
# ══════════════════════════════════════════════════════════════
OP_PREDICT_Q    = 0
OP_PREDICT_TGT  = 1
OP_TRAIN_SEQ    = 2
OP_LOAD_WEIGHTS = 3
OP_READ_WEIGHTS = 4
OP_SYNC_TARGET  = 5

# ══════════════════════════════════════════════════════════════
# Overlay + DMA setup
# ══════════════════════════════════════════════════════════════
OVERLAY_PATH = str(Path(__file__).resolve().parent / "os_elm.bit")
print(f"Loading overlay: {OVERLAY_PATH}")
ol  = Overlay(OVERLAY_PATH)
dma = ol.axi_dma_0
print("Overlay loaded. DMA ready.\n")

_W_SIZE    = STATE_DIM * HIDDEN_DIM      # 384
_B_SIZE    = HIDDEN_DIM                  # 64
_BETA_SIZE = HIDDEN_DIM * NUM_ACTIONS    # 192
_P_SIZE    = HIDDEN_DIM * HIDDEN_DIM     # 4096

def dma_xfer(send_buf, recv_buf):
    """Start recv first, then send, then wait both."""
    dma.recvchannel.transfer(recv_buf)
    dma.sendchannel.transfer(send_buf)
    dma.sendchannel.wait()
    dma.recvchannel.wait()

# ══════════════════════════════════════════════════════════════
# Reference weights (must match os_elm_tb.cpp init_ref_weights)
# ══════════════════════════════════════════════════════════════
ref_W = np.zeros((STATE_DIM, HIDDEN_DIM), dtype=np.float64)
for i in range(STATE_DIM):
    for j in range(HIDDEN_DIM):
        ref_W[i, j] = (i * HIDDEN_DIM + j + 1) * 0.01

ref_b = np.array([(j + 1) * 0.005 for j in range(HIDDEN_DIM)],
                 dtype=np.float64)

ref_beta = np.zeros((HIDDEN_DIM, NUM_ACTIONS), dtype=np.float64)
for j in range(HIDDEN_DIM):
    for a in range(NUM_ACTIONS):
        ref_beta[j, a] = (j * NUM_ACTIONS + a + 1) * 0.02

ref_P = 5.0 * np.eye(HIDDEN_DIM, dtype=np.float64)

# Test inputs (match os_elm_tb.cpp)
TEST_STATE  = np.array([0.5, 0.3, 1.0, 0.1, 0.6, 0.45], dtype=np.float64)
TRAIN_ACTION = 1
TRAIN_TARGET = 3.5

def ref_predict(state, beta):
    """Float64 forward pass (software reference)."""
    h = np.maximum(0.0, state @ ref_W + ref_b)
    return h @ beta

# ══════════════════════════════════════════════════════════════
# Timing + result tracking
# ══════════════════════════════════════════════════════════════
_results = {"pass": 0, "fail": 0, "skip": 0}

# Shared state between tests
_state = {
    "q_theta1":       None,   # set by test_predict_q
    "weights_loaded": False,  # set by test_load_weights
    "trained":        False,  # set by test_train_seq
}

def _run(name, fn):
    """Run a test function, time it, catch failures."""
    print(f"─── {name} " + "─" * max(0, 50 - len(name)))
    t0 = time.perf_counter()
    try:
        fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  PASS  [{elapsed_ms:7.2f} ms]\n")
        _results["pass"] += 1
    except AssertionError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  FAIL  [{elapsed_ms:7.2f} ms]  {e}\n")
        _results["fail"] += 1
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  ERROR [{elapsed_ms:7.2f} ms]  {type(e).__name__}: {e}\n")
        _results["fail"] += 1

# ══════════════════════════════════════════════════════════════
# TEST FUNCTIONS
# ══════════════════════════════════════════════════════════════

def test_load_weights():
    """OP 3: LOAD_WEIGHTS — fills FPGA BRAM with ref_W, ref_b, ref_beta, ref_P."""
    load_size = 1 + _W_SIZE + _B_SIZE + _BETA_SIZE + _P_SIZE
    send_buf = allocate(shape=(load_size,), dtype=np.int32)
    recv_buf = allocate(shape=(1,), dtype=np.int32)

    idx = 0
    send_buf[idx] = OP_LOAD_WEIGHTS; idx += 1
    send_buf[idx:idx+_W_SIZE]    = to_q20(ref_W.flatten(order='C'));    idx += _W_SIZE
    send_buf[idx:idx+_B_SIZE]    = to_q20(ref_b);                       idx += _B_SIZE
    send_buf[idx:idx+_BETA_SIZE] = to_q20(ref_beta.flatten(order='C')); idx += _BETA_SIZE
    send_buf[idx:idx+_P_SIZE]    = to_q20(ref_P.flatten(order='C'));    idx += _P_SIZE

    dma_xfer(send_buf, recv_buf)
    ack = int(recv_buf[0])
    print(f"    ack = {ack}")
    assert ack == 1, f"expected ack=1, got {ack}"
    _state["weights_loaded"] = True


def test_predict_q():
    """OP 0: PREDICT_Q — Q-values via θ₁ should match numpy reference."""
    assert _state["weights_loaded"], "run test_load_weights first"

    send_buf = allocate(shape=(7,), dtype=np.int32)
    recv_buf = allocate(shape=(3,), dtype=np.int32)

    send_buf[0]   = OP_PREDICT_Q
    send_buf[1:7] = to_q20(TEST_STATE)

    dma_xfer(send_buf, recv_buf)
    q_hw  = from_q20(np.array(recv_buf))
    q_ref = ref_predict(TEST_STATE, ref_beta)

    for a in range(NUM_ACTIONS):
        err = abs(q_hw[a] - q_ref[a])
        print(f"    Q[{a}]  hw={q_hw[a]:8.4f}  ref={q_ref[a]:8.4f}  err={err:.4f}")
        assert err < 0.5, f"Q[{a}] error {err:.4f} exceeds tolerance 0.5"

    _state["q_theta1"] = q_hw.copy()


def test_sync_target():
    """OP 5: SYNC_TARGET — copy θ₁ β → θ₂ β inside FPGA."""
    assert _state["weights_loaded"], "run test_load_weights first"

    send_buf = allocate(shape=(1,), dtype=np.int32)
    recv_buf = allocate(shape=(1,), dtype=np.int32)

    send_buf[0] = OP_SYNC_TARGET
    dma_xfer(send_buf, recv_buf)
    ack = int(recv_buf[0])
    print(f"    ack = {ack}")
    assert ack == 1, f"expected ack=1, got {ack}"


def test_predict_tgt_after_sync():
    """OP 1: PREDICT_TGT after sync — must match PREDICT_Q exactly."""
    assert _state["q_theta1"] is not None, "run test_predict_q first"

    send_buf = allocate(shape=(7,), dtype=np.int32)
    recv_buf = allocate(shape=(3,), dtype=np.int32)

    send_buf[0]   = OP_PREDICT_TGT
    send_buf[1:7] = to_q20(TEST_STATE)

    dma_xfer(send_buf, recv_buf)
    q_tgt = from_q20(np.array(recv_buf))
    q_q1  = _state["q_theta1"]

    for a in range(NUM_ACTIONS):
        err = abs(q_tgt[a] - q_q1[a])
        print(f"    Q_tgt[{a}] hw={q_tgt[a]:8.4f}  q1={q_q1[a]:8.4f}  err={err:.6f}")
        # After sync, θ₂ should equal θ₁ exactly (Q20-identical)
        assert err < 0.001, f"Q_tgt[{a}] differs from Q[{a}]: {err:.6f}"


def test_train_seq():
    """OP 2: TRAIN_SEQ — one RLS rank-1 update on θ₁."""
    assert _state["weights_loaded"], "run test_load_weights first"

    send_buf = allocate(shape=(9,), dtype=np.int32)
    recv_buf = allocate(shape=(1,), dtype=np.int32)

    send_buf[0]   = OP_TRAIN_SEQ
    send_buf[1:7] = to_q20(TEST_STATE)
    send_buf[7]   = int(TRAIN_ACTION)         # raw int
    send_buf[8]   = int(to_q20(TRAIN_TARGET))  # Q20 scalar

    dma_xfer(send_buf, recv_buf)
    ack = int(recv_buf[0])
    print(f"    action={TRAIN_ACTION}  target={TRAIN_TARGET}  ack={ack}")
    assert ack == 1, f"expected ack=1, got {ack}"
    _state["trained"] = True


def test_predict_post_train():
    """OP 0: PREDICT_Q after training — Q-values must have changed."""
    assert _state["trained"], "run test_train_seq first"
    assert _state["q_theta1"] is not None, "run test_predict_q first"

    send_buf = allocate(shape=(7,), dtype=np.int32)
    recv_buf = allocate(shape=(3,), dtype=np.int32)

    send_buf[0]   = OP_PREDICT_Q
    send_buf[1:7] = to_q20(TEST_STATE)

    dma_xfer(send_buf, recv_buf)
    q_post = from_q20(np.array(recv_buf))
    q_pre  = _state["q_theta1"]

    changed = False
    for a in range(NUM_ACTIONS):
        diff = abs(q_post[a] - q_pre[a])
        print(f"    Q_post[{a}] = {q_post[a]:8.4f}  (pre={q_pre[a]:8.4f}  Δ={diff:.6f})")
        if diff > 0.001:
            changed = True
    assert changed, "Q-values unchanged after TRAIN_SEQ (training had no effect)"


def test_read_weights():
    """OP 4: READ_WEIGHTS — stream β and P back, verify training effect + P sanity."""
    assert _state["weights_loaded"], "run test_load_weights first"

    send_buf = allocate(shape=(1,), dtype=np.int32)
    recv_buf = allocate(shape=(_BETA_SIZE + _P_SIZE,), dtype=np.int32)

    send_buf[0] = OP_READ_WEIGHTS
    dma_xfer(send_buf, recv_buf)

    data = np.array(recv_buf)
    read_beta = from_q20(data[:_BETA_SIZE]).reshape(HIDDEN_DIM, NUM_ACTIONS)
    read_P    = from_q20(data[_BETA_SIZE:]).reshape(HIDDEN_DIM, HIDDEN_DIM)

    print(f"    beta[0][0]: orig={ref_beta[0,0]:.4f}  read={read_beta[0,0]:.4f}")
    print(f"    P[0][0]:    orig={ref_P[0,0]:.4f}    read={read_P[0,0]:.4f}")

    # If training ran, β should differ from the original
    if _state["trained"]:
        beta_diff = np.max(np.abs(read_beta - ref_beta))
        print(f"    max |Δβ| = {beta_diff:.6f}")
        assert beta_diff > 0.001, "β unchanged after TRAIN_SEQ"

    # P diagonal should always be positive (never corrupted)
    P_diag = np.diag(read_P)
    neg_count = int(np.sum(P_diag <= 0.0))
    print(f"    P diagonal: min={P_diag.min():.4f}  max={P_diag.max():.4f}  "
          f"non-positive count={neg_count}")
    assert neg_count == 0, f"{neg_count} P diagonal elements are non-positive"


def test_unknown_opcode():
    """Unknown opcode should return ack=-1."""
    send_buf = allocate(shape=(1,), dtype=np.int32)
    recv_buf = allocate(shape=(1,), dtype=np.int32)

    send_buf[0] = 99
    dma_xfer(send_buf, recv_buf)
    ack = int(recv_buf[0])
    print(f"    opcode=99  ack={ack}")
    assert ack == -1, f"expected ack=-1, got {ack}"


def test_predict_throughput():
    """Bonus: time 1000 back-to-back PREDICT_Q calls."""
    assert _state["weights_loaded"], "run test_load_weights first"

    N = 1000
    send_buf = allocate(shape=(7,), dtype=np.int32)
    recv_buf = allocate(shape=(3,), dtype=np.int32)
    send_buf[0]   = OP_PREDICT_Q
    send_buf[1:7] = to_q20(TEST_STATE)

    t0 = time.perf_counter()
    for _ in range(N):
        dma_xfer(send_buf, recv_buf)
    elapsed = time.perf_counter() - t0

    per_call_us = (elapsed / N) * 1e6
    print(f"    {N} PREDICT_Q calls in {elapsed*1000:.1f} ms "
          f"→ {per_call_us:.1f} µs/call  ({N/elapsed:.0f} Hz)")


# ══════════════════════════════════════════════════════════════
# Individual test invocations — comment out any failing test
# to skip it while running the rest.
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print(" FPGA Kernel Tests")
print("=" * 60 + "\n")

_run("TEST 1: LOAD_WEIGHTS",             test_load_weights)
_run("TEST 2: PREDICT_Q",                test_predict_q)
_run("TEST 3: SYNC_TARGET",              test_sync_target)
_run("TEST 4: PREDICT_TGT (post-sync)",  test_predict_tgt_after_sync)
_run("TEST 5: TRAIN_SEQ",                test_train_seq)
_run("TEST 6: PREDICT_Q (post-train)",   test_predict_post_train)
_run("TEST 7: READ_WEIGHTS",             test_read_weights)
_run("TEST 8: Unknown opcode",           test_unknown_opcode)
_run("TEST 9: PREDICT_Q throughput",     test_predict_throughput)

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print("=" * 60)
total = _results["pass"] + _results["fail"]
print(f" Results: {_results['pass']}/{total} passed, "
      f"{_results['fail']} failed")
print("=" * 60)
