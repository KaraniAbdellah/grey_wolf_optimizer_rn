import tkinter as tk
from tkinter import ttk, font
import numpy as np
import math
import time
import threading

# ─────────────────────────────────────────────
#  COLORS & THEME
# ─────────────────────────────────────────────
BG        = "#f0f2f5"
PANEL     = "#ffffff"
CARD      = "#f8f9fb"
BORDER    = "#dde1ea"
ACCENT    = "#4f46e5"
ALPHA_C   = "#d97706"
BETA_C    = "#7c3aed"
DELTA_C   = "#0891b2"
OMEGA_C   = "#64748b"
GREEN     = "#16a34a"
RED       = "#dc2626"
TEXT1     = "#0f172a"
TEXT2     = "#475569"
TEXT3     = "#94a3b8"

WOLF_COLORS = [ALPHA_C, BETA_C, DELTA_C, OMEGA_C, "#ec4899", "#fb923c", "#34d399", "#60a5fa", "#f87171", "#c084fc"]

# ─────────────────────────────────────────────
#  NEURAL NETWORK
# ─────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forward_pass(weights, X):
    w1, w2, w3, w4, w5, w6, b1, b2, b3 = weights
    h1 = sigmoid(w1 * X[0] + w2 * X[1] + b1)
    h2 = sigmoid(w3 * X[0] + w4 * X[1] + b2)
    y  = sigmoid(w5 * h1   + w6 * h2   + b3)
    return y, h1, h2

def compute_mse(weights, data):
    total = 0
    for X, y in data:
        pred, _, _ = forward_pass(weights, X)
        total += (y - pred) ** 2
    return total / len(data)

# ─────────────────────────────────────────────
#  GWO
# ─────────────────────────────────────────────
class GWO:
    def __init__(self, n_wolves, n_iter, data):
        self.n_wolves = n_wolves
        self.n_iter   = n_iter
        self.data     = data
        self.dim      = 9  # 6 weights + 3 biases
        self.wolves   = np.random.uniform(-1, 1, (n_wolves, self.dim))
        self.fitness  = np.array([compute_mse(w, data) for w in self.wolves])
        self._rank()
        self.history  = []
        self.iter     = 0

    def _rank(self):
        idx = np.argsort(self.fitness)
        self.alpha_idx = idx[0]
        self.beta_idx  = idx[1]
        self.delta_idx = idx[2]
        self.alpha = self.wolves[self.alpha_idx].copy()
        self.beta  = self.wolves[self.beta_idx].copy()
        self.delta = self.wolves[self.delta_idx].copy()
        self.alpha_fit = self.fitness[self.alpha_idx]
        self.beta_fit  = self.fitness[self.beta_idx]
        self.delta_fit = self.fitness[self.delta_idx]

    def step(self):
        if self.iter >= self.n_iter:
            return False
        a = 2 - 2 * self.iter / self.n_iter
        for i in range(self.n_wolves):
            new_pos = np.zeros(self.dim)
            for d in range(self.dim):
                for leader in [self.alpha, self.beta, self.delta]:
                    r1, r2 = np.random.rand(), np.random.rand()
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = abs(C * leader[d] - self.wolves[i][d])
                    new_pos[d] += leader[d] - A * D
            self.wolves[i] = np.clip(new_pos / 3, -5, 5)
            self.fitness[i] = compute_mse(self.wolves[i], self.data)
        self._rank()
        self.iter += 1
        self.history.append(self.alpha_fit)
        return True

# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gray Wolf Optimizer — Neural Network Visualizer")
        self.configure(bg=BG)
        self.geometry("1400x860")
        self.resizable(True, True)
        self.minsize(1100, 700)

        # State
        self.gwo        = None
        self.running    = False
        self.speed      = 200   # ms between steps
        self.n_wolves   = tk.IntVar(value=8)
        self.n_iter     = tk.IntVar(value=60)
        self.step_delay = tk.IntVar(value=200)
        self._after_id  = None

        self._build_fonts()
        self._build_layout()
        self._build_data()

    # ── fonts ────────────────────────────────
    def _build_fonts(self):
        self.f_title  = font.Font(family="Helvetica", size=16, weight="bold")
        self.f_head   = font.Font(family="Helvetica", size=12, weight="bold")
        self.f_body   = font.Font(family="Helvetica", size=10)
        self.f_small  = font.Font(family="Helvetica", size=9)
        self.f_mono   = font.Font(family="Courier",   size=9)
        self.f_big    = font.Font(family="Helvetica", size=22, weight="bold")

    # ── training data ────────────────────────
    def _build_data(self):
        # XOR-like: study hours, sleep hours → pass/fail
        self.data = [
            ([6, 7], 1), ([8, 6], 1), ([7, 8], 1), ([9, 7], 1),
            ([2, 4], 0), ([1, 3], 0), ([3, 2], 0), ([2, 5], 0),
        ]

    # ── layout ───────────────────────────────
    def _build_layout(self):
        # Top bar
        top = tk.Frame(self, bg=PANEL, height=56)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="🐺  Gray Wolf Optimizer  ×  Neural Network", font=self.f_title,
                 bg=PANEL, fg=TEXT1).pack(side="left", padx=20, pady=12)
        tk.Label(top, text="Real-time learning visualizer", font=self.f_body,
                 bg=PANEL, fg=TEXT2).pack(side="left")

        # Main area
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=10)

        # Left controls
        left = tk.Frame(main, bg=PANEL, width=240)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        self._build_controls(left)

        # Center: network + wolves
        center = tk.Frame(main, bg=BG)
        center.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.canvas_nn = tk.Canvas(center, bg=PANEL, highlightthickness=0, height=340)
        self.canvas_nn.pack(fill="x", pady=(0, 8))

        self.canvas_wolves = tk.Canvas(center, bg=PANEL, highlightthickness=0)
        self.canvas_wolves.pack(fill="both", expand=True)

        # Right: log + stats
        right = tk.Frame(main, bg=PANEL, width=270)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self._build_right(right)

        self.canvas_nn.bind("<Configure>", lambda e: self._draw_network())
        self.canvas_wolves.bind("<Configure>", lambda e: self._draw_wolves())

    # ── left controls ────────────────────────
    def _build_controls(self, parent):
        pad = dict(padx=14, pady=4)

        tk.Label(parent, text="⚙  Settings", font=self.f_head,
                 bg=PANEL, fg=TEXT1).pack(anchor="w", padx=14, pady=(16, 8))

        self._ctrl_label(parent, "Number of Wolves")
        s1 = tk.Scale(parent, from_=3, to=15, orient="horizontal",
                      variable=self.n_wolves, bg=PANEL, fg=TEXT1,
                      troughcolor=BORDER, activebackground=ACCENT,
                      highlightthickness=0, relief="flat")
        s1.pack(fill="x", **pad)

        self._ctrl_label(parent, "Iterations")
        s2 = tk.Scale(parent, from_=20, to=200, orient="horizontal",
                      variable=self.n_iter, bg=PANEL, fg=TEXT1,
                      troughcolor=BORDER, activebackground=ACCENT,
                      highlightthickness=0, relief="flat")
        s2.pack(fill="x", **pad)

        self._ctrl_label(parent, "Speed (ms/step)")
        s3 = tk.Scale(parent, from_=50, to=800, orient="horizontal",
                      variable=self.step_delay, bg=PANEL, fg=TEXT1,
                      troughcolor=BORDER, activebackground=ACCENT,
                      highlightthickness=0, relief="flat")
        s3.pack(fill="x", **pad)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=14, pady=12)

        # Buttons
        self.btn_start = self._btn(parent, "▶  Start Training", ACCENT, self._start)
        self.btn_start.pack(fill="x", padx=14, pady=4)

        self.btn_pause = self._btn(parent, "⏸  Pause", BORDER, self._pause)
        self.btn_pause.pack(fill="x", padx=14, pady=4)

        self.btn_reset = self._btn(parent, "↺  Reset", BORDER, self._reset)
        self.btn_reset.pack(fill="x", padx=14, pady=4)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=14, pady=12)

        # Legend
        tk.Label(parent, text="🏆  Wolf Ranks", font=self.f_head,
                 bg=PANEL, fg=TEXT1).pack(anchor="w", padx=14)
        for name, color, desc in [
            ("α Alpha",  ALPHA_C, "Best fitness"),
            ("β Beta",   BETA_C,  "2nd best"),
            ("δ Delta",  DELTA_C, "3rd best"),
            ("ω Omega",  OMEGA_C, "Learning wolves"),
        ]:
            row = tk.Frame(parent, bg=PANEL)
            row.pack(fill="x", padx=14, pady=2)
            tk.Canvas(row, width=12, height=12, bg=PANEL, highlightthickness=0).pack(side="left")
            dot = tk.Canvas(row, width=14, height=14, bg=PANEL, highlightthickness=0)
            dot.pack(side="left")
            dot.create_oval(2, 2, 12, 12, fill=color, outline="")
            tk.Label(row, text=name, font=self.f_small, bg=PANEL, fg=color, width=8, anchor="w").pack(side="left")
            tk.Label(row, text=desc, font=self.f_small, bg=PANEL, fg=TEXT3).pack(side="left")

        # Stats cards
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=14, pady=12)
        tk.Label(parent, text="📊  Live Stats", font=self.f_head,
                 bg=PANEL, fg=TEXT1).pack(anchor="w", padx=14)

        stats_frame = tk.Frame(parent, bg=PANEL)
        stats_frame.pack(fill="x", padx=14, pady=6)

        self.lbl_iter  = self._stat_card(stats_frame, "Iteration", "0 / 0", 0)
        self.lbl_error = self._stat_card(stats_frame, "Best Error", "—", 1)
        self.lbl_a     = self._stat_card(stats_frame, "a (explore)", "2.00", 2)
        self.lbl_pred  = self._stat_card(stats_frame, "α Accuracy", "—", 3)

    def _ctrl_label(self, parent, text):
        tk.Label(parent, text=text, font=self.f_small, bg=PANEL, fg=TEXT2).pack(anchor="w", padx=14, pady=(6,0))

    def _btn(self, parent, text, color, cmd):
        b = tk.Button(parent, text=text, font=self.f_body, bg=color, fg=TEXT1,
                      activebackground=ACCENT, activeforeground=TEXT1,
                      relief="flat", cursor="hand2", command=cmd, pady=8)
        return b

    def _stat_card(self, parent, label, value, row):
        f = tk.Frame(parent, bg=CARD, padx=10, pady=6)
        f.grid(row=row, column=0, sticky="ew", pady=3)
        parent.columnconfigure(0, weight=1)
        tk.Label(f, text=label, font=self.f_small, bg=CARD, fg=TEXT2).pack(anchor="w")
        lbl = tk.Label(f, text=value, font=self.f_head, bg=CARD, fg=TEXT1)
        lbl.pack(anchor="w")
        return lbl

    # ── right panel ──────────────────────────
    def _build_right(self, parent):
        tk.Label(parent, text="📋  Step-by-Step Log", font=self.f_head,
                 bg=PANEL, fg=TEXT1).pack(anchor="w", padx=14, pady=(16, 6))

        frame = tk.Frame(parent, bg=PANEL)
        frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.log_text = tk.Text(frame, bg=CARD, fg=TEXT2, font=self.f_mono,
                                relief="flat", wrap="word", state="disabled",
                                insertbackground=TEXT1, selectbackground=ACCENT)
        scroll = tk.Scrollbar(frame, command=self.log_text.yview, bg=PANEL)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

        # Error chart canvas
        tk.Label(parent, text="📉  Error Over Time", font=self.f_head,
                 bg=PANEL, fg=TEXT1).pack(anchor="w", padx=14, pady=(4, 4))
        self.canvas_chart = tk.Canvas(parent, bg=CARD, highlightthickness=0, height=180)
        self.canvas_chart.pack(fill="x", padx=8, pady=(0, 10))

    # ── control callbacks ─────────────────────
    def _start(self):
        if self.gwo is None:
            self.gwo = GWO(self.n_wolves.get(), self.n_iter.get(), self.data)
            self._log_clear()
            self._log(f"🐺 Pack initialized: {self.n_wolves.get()} wolves, {self.n_iter.get()} iterations\n")
            self._log(f"📦 Training data: {len(self.data)} samples\n")
            self._log("─" * 34 + "\n")
        self.running = True
        self._loop()

    def _pause(self):
        self.running = False
        if self._after_id:
            self.after_cancel(self._after_id)

    def _reset(self):
        self.running = False
        if self._after_id:
            self.after_cancel(self._after_id)
        self.gwo = None
        self._log_clear()
        self._draw_network()
        self._draw_wolves()
        self._draw_chart()
        self.lbl_iter.config(text="0 / 0")
        self.lbl_error.config(text="—")
        self.lbl_a.config(text="2.00")
        self.lbl_pred.config(text="—")

    def _loop(self):
        if not self.running or self.gwo is None:
            return
        ok = self.gwo.step()
        self._update_ui()
        if ok and self.gwo.iter < self.gwo.n_iter:
            self._after_id = self.after(self.step_delay.get(), self._loop)
        else:
            self.running = False
            self._log("\n✅ Training complete!\n")
            self._log(f"   Best error: {self.gwo.alpha_fit:.5f}\n")
            acc = self._compute_accuracy()
            self._log(f"   Accuracy:   {acc:.0f}%\n")

    def _compute_accuracy(self):
        if self.gwo is None:
            return 0
        correct = 0
        for X, y in self.data:
            pred, _, _ = forward_pass(self.gwo.alpha, X)
            if round(pred) == y:
                correct += 1
        return 100 * correct / len(self.data)

    # ── UI update ────────────────────────────
    def _update_ui(self):
        g = self.gwo
        a_val = 2 - 2 * g.iter / g.n_iter
        self.lbl_iter.config(text=f"{g.iter} / {g.n_iter}")
        self.lbl_error.config(text=f"{g.alpha_fit:.5f}")
        self.lbl_a.config(text=f"{a_val:.3f}")
        acc = self._compute_accuracy()
        self.lbl_pred.config(text=f"{acc:.0f}%")

        # Log every 5 steps or first 3
        if g.iter <= 3 or g.iter % 5 == 0:
            self._log(f"Iter {g.iter:>3}  |  a={a_val:.2f}  |  err={g.alpha_fit:.5f}  |  acc={acc:.0f}%\n")

        self._draw_network()
        self._draw_wolves()
        self._draw_chart()

    # ── network canvas ────────────────────────
    def _draw_network(self):
        c = self.canvas_nn
        c.delete("all")
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 10 or H < 10:
            return

        cx = W // 2
        cy = H // 2

        # Title
        c.create_text(cx, 18, text="Neural Network  (2 inputs → 2 hidden → 1 output)",
                      fill=TEXT2, font=self.f_small)

        # Node positions
        layer_x = [cx - 200, cx, cx + 200]
        nodes = {
            "x1": (layer_x[0], cy - 60),
            "x2": (layer_x[0], cy + 60),
            "h1": (layer_x[1], cy - 60),
            "h2": (layer_x[1], cy + 60),
            "y":  (layer_x[2], cy),
        }

        weights = self.gwo.alpha if self.gwo else [0]*9
        w1,w2,w3,w4,w5,w6,b1,b2,b3 = weights

        # Connections with weight-based color
        edges = [
            ("x1","h1", w1), ("x1","h2", w3),
            ("x2","h1", w2), ("x2","h2", w4),
            ("h1","y",  w5), ("h2","y",  w6),
        ]
        for src, dst, w in edges:
            sx, sy = nodes[src]
            dx, dy = nodes[dst]
            intensity = min(1.0, abs(w) / 3)
            col = self._lerp_color("#1e293b", GREEN if w > 0 else RED, intensity)
            thick = max(1, int(abs(w) * 2))
            c.create_line(sx, sy, dx, dy, fill=col, width=thick, smooth=True)
            # Weight label at midpoint
            mx, my = (sx+dx)//2, (sy+dy)//2
            c.create_text(mx, my-8, text=f"{w:.2f}", fill=TEXT3, font=self.f_small)

        # Compute activations for sample input
        sample_X = [6, 7]
        if self.gwo:
            _, h1_act, h2_act = forward_pass(weights, sample_X)
            y_act, _, _       = forward_pass(weights, sample_X)
        else:
            h1_act = h2_act = y_act = 0.5

        act = {"x1": 0.8, "x2": 0.9, "h1": h1_act, "h2": h2_act, "y": y_act}
        labels = {"x1": "x₁=6", "x2": "x₂=7",
                  "h1": f"h₁\n{h1_act:.2f}", "h2": f"h₂\n{h2_act:.2f}",
                  "y":  f"ŷ={y_act:.2f}"}
        colors = {"x1": DELTA_C, "x2": DELTA_C,
                  "h1": BETA_C,  "h2": BETA_C,
                  "y":  ALPHA_C}
        r = 30
        for name, (nx, ny) in nodes.items():
            fill = self._lerp_color(CARD, colors[name], act.get(name, 0.5))
            c.create_oval(nx-r, ny-r, nx+r, ny+r, fill=fill, outline=colors[name], width=2)
            c.create_text(nx, ny, text=labels[name], fill=TEXT1, font=self.f_small, justify="center")

        # Layer labels
        for lx, ltxt in zip(layer_x, ["Input\nLayer", "Hidden\nLayer", "Output\nLayer"]):
            c.create_text(lx, H - 20, text=ltxt, fill=TEXT3, font=self.f_small, justify="center")

        # Bias labels
        c.create_text(layer_x[1]-50, cy-85, text=f"b₁={b1:.2f}", fill=TEXT3, font=self.f_small)
        c.create_text(layer_x[1]-50, cy+38,  text=f"b₂={b2:.2f}", fill=TEXT3, font=self.f_small)
        c.create_text(layer_x[2]+50, cy-16,  text=f"b₃={b3:.2f}", fill=TEXT3, font=self.f_small)

    # ── wolves canvas ─────────────────────────
    def _draw_wolves(self):
        c = self.canvas_wolves
        c.delete("all")
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 10 or H < 10:
            return

        c.create_text(12, 12, text="Wolf Pack  —  each circle = one wolf (candidate set of weights)",
                      fill=TEXT2, font=self.f_small, anchor="w")

        if self.gwo is None:
            c.create_text(W//2, H//2, text="Press  ▶ Start Training  to begin",
                          fill=TEXT3, font=self.f_head)
            return

        g = self.gwo
        n = g.n_wolves

        # Layout wolves in a grid
        cols = min(n, 8)
        rows = math.ceil(n / cols)
        cell_w = min((W - 40) // cols, 140)
        cell_h = min((H - 50) // rows, 90)
        start_x = (W - cols * cell_w) // 2

        sorted_idx = np.argsort(g.fitness)
        rank_map = {idx: rank for rank, idx in enumerate(sorted_idx)}
        rank_labels = {0: ("α", ALPHA_C), 1: ("β", BETA_C), 2: ("δ", DELTA_C)}

        for i in range(n):
            col = i % cols
            row = i // cols
            bx = start_x + col * cell_w + 6
            by = 36 + row * cell_h + 4
            bw = cell_w - 12
            bh = cell_h - 10

            rank = rank_map[i]
            if rank in rank_labels:
                rlabel, rcolor = rank_labels[rank]
            else:
                rlabel, rcolor = "ω", OMEGA_C

            # Card background
            c.create_rectangle(bx, by, bx+bw, by+bh, fill=CARD, outline=rcolor,
                                width=2 if rank < 3 else 1)

            # Wolf icon + rank
            c.create_text(bx+18, by+14, text="🐺", font=self.f_body)
            c.create_text(bx+36, by+14, text=f"Wolf {i+1}", fill=TEXT2, font=self.f_small, anchor="w")
            c.create_text(bx+bw-6, by+14, text=rlabel, fill=rcolor,
                          font=font.Font(family="Helvetica", size=11, weight="bold"), anchor="e")

            # Error bar
            err = g.fitness[i]
            max_err = max(g.fitness) if max(g.fitness) > 0 else 1
            bar_w = int((bw - 16) * min(1, (1 - err / max_err)))
            c.create_rectangle(bx+8, by+bh-18, bx+8+bw-16, by+bh-8,
                                fill=BORDER, outline="")
            c.create_rectangle(bx+8, by+bh-18, bx+8+bar_w, by+bh-8,
                                fill=rcolor, outline="")

            # Error value
            c.create_text(bx+bw//2, by+bh//2+4, text=f"err={err:.4f}",
                          fill=TEXT2, font=self.f_small)

        # a-value progress bar
        a_val = 2 - 2 * g.iter / g.n_iter
        bar_total = W - 40
        bar_filled = int(bar_total * (1 - a_val / 2))
        by_prog = H - 22
        c.create_text(12, by_prog, text=f"Convergence  (a={a_val:.2f})", fill=TEXT3, font=self.f_small, anchor="w")
        c.create_rectangle(12, by_prog+10, 12+bar_total, by_prog+20, fill=BORDER, outline="")
        c.create_rectangle(12, by_prog+10, 12+bar_filled, by_prog+20, fill=ACCENT, outline="")
        c.create_text(W-12, by_prog+14, text="converged ✓" if a_val < 0.05 else f"{int(100*bar_filled/bar_total)}%",
                      fill=ACCENT if a_val < 0.05 else TEXT2, font=self.f_small, anchor="e")

    # ── error chart ───────────────────────────
    def _draw_chart(self):
        c = self.canvas_chart
        c.delete("all")
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 10 or H < 10:
            return

        if not self.gwo or len(self.gwo.history) < 2:
            c.create_text(W//2, H//2, text="Waiting for data…", fill=TEXT3, font=self.f_small)
            return

        hist = self.gwo.history
        pad_l, pad_r, pad_t, pad_b = 36, 14, 14, 30
        pw = W - pad_l - pad_r
        ph = H - pad_t - pad_b

        mn, mx = min(hist), max(hist)
        rng = mx - mn if mx != mn else 1

        # Axes
        c.create_line(pad_l, pad_t, pad_l, pad_t+ph, fill=BORDER, width=1)
        c.create_line(pad_l, pad_t+ph, pad_l+pw, pad_t+ph, fill=BORDER, width=1)

        # Y labels
        for i in range(3):
            v = mx - i * rng / 2
            y = pad_t + i * ph // 2
            c.create_text(pad_l-4, y, text=f"{v:.3f}", fill=TEXT3, font=self.f_small, anchor="e")
            c.create_line(pad_l, y, pad_l+pw, y, fill=BORDER, width=1, dash=(2,4))

        # X label
        c.create_text(pad_l + pw//2, H-8, text="Iteration", fill=TEXT3, font=self.f_small)

        # Line
        pts = []
        for i, v in enumerate(hist):
            x = pad_l + int(pw * i / max(len(hist)-1, 1))
            y = pad_t + ph - int(ph * (v - mn) / rng)
            pts.extend([x, y])

        if len(pts) >= 4:
            c.create_line(*pts, fill=ACCENT, width=2, smooth=True)
            # Last dot
            lx, ly = pts[-2], pts[-1]
            c.create_oval(lx-4, ly-4, lx+4, ly+4, fill=ALPHA_C, outline="")

    # ── helpers ───────────────────────────────
    def _lerp_color(self, c1, c2, t):
        t = max(0, min(1, t))
        def parse(col):
            col = col.lstrip("#")
            return tuple(int(col[i:i+2], 16) for i in (0, 2, 4))
        r1,g1,b1 = parse(c1)
        r2,g2,b2 = parse(c2)
        r = int(r1 + (r2-r1)*t)
        g = int(g1 + (g2-g1)*t)
        b = int(b1 + (b2-b1)*t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _log_clear(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
