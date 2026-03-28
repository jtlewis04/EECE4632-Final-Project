# os_elm_dqn.py — OS-ELM Q-Network for Breakout
#
# Paper techniques retained (FPGA-compatible):
#   1. Spectral normalization of α (W_in) at initialization
#   2. L2 regularization (ReOS-ELM) for β during initial training
#   3. Q-value clipping to [-1, 1]
#   4. Random update gating (ε₂)
#   5. Fixed target Q-network updated every UPDATE_STEP episodes
#   6. No experience replay after initial training
#   7. ReLU activation
#   8. Batch size 1 for sequential training (scalar reciprocal, no SVD/QRD)
#
# Changed from paper: multi-output model instead of simplified output model.
# Input = state (14 features), output = Q-value per action (3 outputs).
# Each action column of β is updated independently via the same RLS formula.
# FPGA impact: β is (hidden × 3) instead of (hidden × 1), same P matrix,
# same predict/train operations. Predict is FASTER (one forward pass vs three).

import numpy as np
from collections import deque
from encoder import STATE_DIM

NUM_ACTIONS = 3  # LEFT, STAY, RIGHT


class OSELM_QNetwork:

    def __init__(self, input_dim=STATE_DIM, hidden_dim=128, output_dim=NUM_ACTIONS,
                 reg=0.1, lam=0.999, seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.reg = reg
        self.lam = lam  # forgetting factor: 1.0 = paper (no forgetting), <1.0 = keep learning

        rng = np.random.RandomState(seed)

        # Algorithm 1 line 1: Initialize α with random values R ∈ [0, 1]
        self.W_in = rng.rand(input_dim, hidden_dim).astype(np.float64)
        self.b = rng.rand(hidden_dim).astype(np.float64)

        # Algorithm 1 lines 2-3: Spectral normalization of α
        _, S, _ = np.linalg.svd(self.W_in, full_matrices=False)
        self.W_in = self.W_in / S[0]

        # β: output weights (hidden_dim × num_actions)
        self.beta = np.zeros((hidden_dim, output_dim), dtype=np.float64)
        self.P = None
        self.P_max = 1.0 / reg  # cap for P diagonal to prevent forgetting overflow

        self.is_initialized = False
        self._act = lambda x: np.maximum(0.0, x)

    def _hidden(self, X):
        return self._act(X @ self.W_in + self.b)

    def predict_single(self, state):
        """Q-values for all actions from one state. Returns (num_actions,)."""
        h = self._hidden(state.reshape(1, -1))
        return (h @ self.beta).flatten()

    def predict_batch(self, states):
        """Q-values for a batch. Returns (batch, num_actions)."""
        H = self._hidden(states)
        return H @ self.beta

    # ── initial training (Equation 7: ReOS-ELM) ────────────
    def init_batch(self, states, actions, targets):
        """
        Batch solve with L2 regularization.
        states:  (N, input_dim)
        actions: (N,) int action indices
        targets: (N,) clipped TD target for the taken action
        """
        H = self._hidden(states)
        N = len(states)

        # P0 = (H^T H + δI)^{-1}   (Equation 7)
        HtH = H.T @ H + self.reg * np.eye(self.hidden_dim, dtype=np.float64)
        self.P = np.linalg.inv(HtH)

        # Build target matrix: current predictions for non-taken actions,
        # TD target for taken action
        Y = H @ self.beta  # (N, output_dim) — all zeros before init
        for i in range(N):
            Y[i, actions[i]] = targets[i]

        self.beta = self.P @ (H.T @ Y)
        self.is_initialized = True

    # ── sequential training (Equation 5, batch size 1) ─────
    def update_single(self, state, action, target):
        """
        RLS rank-1 update for ONE action column of β.
        Same P matrix, same scalar reciprocal — FPGA-identical to paper.
        Forgetting factor prevents P from collapsing to zero.
        FPGA cost: one extra scalar multiply on P per update.
        """
        h = self._hidden(state.reshape(1, -1)).flatten()

        # Forgetting: inflate P slightly to keep learning alive
        # FPGA cost: scalar multiply on P diagonal (or full matrix)
        self.P /= self.lam

        # Clamp P to prevent unbounded growth
        max_diag = np.max(np.diag(self.P))
        if max_diag > self.P_max:
            self.P *= (self.P_max / max_diag)

        Ph = self.P @ h
        denom = 1.0 + h @ Ph  # scalar reciprocal — FPGA key

        if not np.isfinite(denom) or denom < 1e-12:
            self.P = np.eye(self.hidden_dim, dtype=np.float64) / self.reg
            Ph = self.P @ h
            denom = 1.0 + h @ Ph

        k = Ph / denom

        # Update ONLY the taken action's column of β
        e = target - h @ self.beta[:, action]
        self.beta[:, action] += k * e

        # P update is shared across all action columns
        self.P -= np.outer(k, h @ self.P)

    def copy_to(self, other):
        other.beta = self.beta.copy()

    def save(self, path):
        np.savez(path, W_in=self.W_in, b=self.b, beta=self.beta, P=self.P)

    def load(self, path):
        d = np.load(path)
        self.W_in = d['W_in']; self.b = d['b']
        self.beta = d['beta']; self.P = d['P']
        self.is_initialized = True


class DQNAgent:
    """OS-ELM-L2-Lipschitz agent. Actions: 0=LEFT, 1=STAY, 2=RIGHT"""

    def __init__(self, hidden_dim=128, gamma=0.99,
                 eps1=0.9, eps2=0.7, update_step=10,
                 reset_after=500, reg=0.1, lam=0.9995, seed=42):
        self.gamma = gamma
        self.eps1 = eps1
        self.eps2 = eps2
        self.update_step = update_step
        self.reset_after = reset_after
        self.reg = reg
        self.lam = lam
        self.seed = seed
        self.hidden_dim = hidden_dim
        self._best_beta = None
        self._best_avg_reward = 0.0
        self._recent_rewards = deque(maxlen=25)

        self._build_nets()
        self.global_step = 0
        self.ep_since_reset = 0
        self._init_buf = []
        self._recent_scores = deque(maxlen=reset_after)

    def _build_nets(self):
        self.q_net = OSELM_QNetwork(STATE_DIM, self.hidden_dim,
                                     NUM_ACTIONS, self.reg, self.lam, self.seed)
        self.target_net = OSELM_QNetwork(STATE_DIM, self.hidden_dim,
                                          NUM_ACTIONS, self.reg, self.lam, self.seed)

    def reset_weights(self):
        self.seed += 1
        self._build_nets()
        self.global_step = 0
        self._init_buf = []
        self.ep_since_reset = 0
        self._recent_scores.clear()
        self._recent_rewards.clear()
        self._best_avg_reward = 0.0
        self._best_beta = None

    def select_action(self, state):
        if self.q_net.is_initialized and np.random.rand() < self.eps1:
            q = self.q_net.predict_single(state)
            if np.all(np.isfinite(q)):
                return int(np.argmax(q))
        return np.random.randint(NUM_ACTIONS)

    def store(self, s, a, r, s_next, done):
        self.global_step += 1
        if not self.q_net.is_initialized:
            self._init_buf.append((s, a, r, s_next, done))

    def update(self, s, a, r, s_next, done):
        # --- initial training (t == N̄) ---
        if not self.q_net.is_initialized:
            if len(self._init_buf) < self.hidden_dim:
                return

            buf = self._init_buf[:self.hidden_dim]
            ss, aa, rr, sn, dd = zip(*buf)
            S = np.array(ss, dtype=np.float64)
            A = np.array(aa, dtype=np.int32)
            R = np.array(rr, dtype=np.float64)

            targets = np.clip(R, -10.0, 20.0)
            self.q_net.init_batch(S, A, targets)
            self.q_net.copy_to(self.target_net)
            self._init_buf.clear()
            return

        # --- sequential training (t > N̄) ---
        if np.random.rand() >= self.eps2:
            return

        q_next = self.target_net.predict_single(s_next)
        if not np.all(np.isfinite(q_next)):
            return

        target = r + (1.0 - float(done)) * self.gamma * np.max(q_next)
        target = np.clip(target, -10.0, 20.0)

        self.q_net.update_single(s, a, target)

    def on_episode_end(self, ep_num, ep_score, ep_reward):
        self.ep_since_reset += 1
        self._recent_scores.append(ep_score)
        self._recent_rewards.append(ep_reward)

        if self.q_net.is_initialized and ep_num % self.update_step == 0:
            self.q_net.copy_to(self.target_net)

        # Restore to best network if average declines
        """
        recent = list(self._recent_rewards)
        if len(recent) >= 25:
            avg25 = np.mean(recent[-25:])
            if avg25 > self._best_avg_reward:
                self._best_avg_reward = avg25
                self._best_beta = self.q_net.beta.copy()
            elif self._best_beta is not None and avg25 < self._best_avg_reward * 0.5:
                # Performance dropped, restore best
                print(f'Restored to {self._best_avg_reward:.1f}')
                self.q_net.beta = self._best_beta.copy()
                self.q_net.copy_to(self.target_net)
        """
        # Reset if no improvement over reset_after episodes
        if self.ep_since_reset % self.reset_after == 0:
            min_avg = 1.5 + (self.ep_since_reset // self.reset_after) * 0.25 #threshold for reset

            recent = self._recent_scores
            # Compare last half of window to first half
            half = len(recent) // 2
            first_half_avg = np.mean(list(recent)[:half]) if half > 0 else 0
            second_half_avg = np.mean(list(recent)[half:]) if half > 0 else 0

            if second_half_avg <= first_half_avg + 0.1 and second_half_avg < min_avg:
                print(f"second half avg:{second_half_avg:.1f}")
                self.reset_weights()
                return True
        return False