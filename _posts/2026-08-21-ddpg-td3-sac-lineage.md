---
layout: post
title: "From DDPG to SAC: a lineage of fixes"
date: 2026-08-21
description: Deep Deterministic Policy Gradient, Twin Delayed DDPG, and Soft Actor-Critic are usually presented as three algorithms. They're better understood as one algorithm and two rounds of debugging.
tags: reinforcement-learning sac td3 ddpg actor-critic
categories: reinforcement-learning-for-control
marimo: true
mermaid:
  enabled: true
  zoomable: true
chart:
  plotly: true
toc:
  sidebar: left
giscus_comments: false
related_posts: false
---

Most tutorials present **DDPG**, **TD3**, and **SAC** as three separate continuous-control algorithms you choose between. That framing hides the useful part.

They are better read as **one algorithm and two rounds of debugging**. Each successor exists because someone diagnosed a specific, nameable failure in its predecessor and fixed it.

<div class="alert alert-info" role="alert" markdown="1">
**The abbreviations, once:**

- **RL** — Reinforcement Learning
- **DQN** — Deep Q-Network
- **DPG** — Deterministic Policy Gradient
- **DDPG** — Deep Deterministic Policy Gradient
- **TD3** — Twin Delayed Deep Deterministic Policy Gradient
- **SAC** — Soft Actor-Critic
- **TD** — Temporal Difference (as in "TD target")
- **MPC** — Model Predictive Control
</div>

<div class="alert alert-secondary" role="alert" markdown="1">
**Reading the visuals.** Diagrams are click-to-zoom — click once to enlarge, click the background or press <kbd>Esc</kbd> to reset. The interactive chart has a toolbar in its top-right corner on hover; the **house icon** resets the axes after you pan or zoom.
</div>

---

## The lineage at a glance

```mermaid
graph LR
    DQN["<b>DQN</b><br/>Deep Q-Network<br/>2015 · discrete actions"]
    DDPG["<b>DDPG</b><br/>Deep Deterministic<br/>Policy Gradient · 2015"]
    TD3["<b>TD3</b><br/>Twin Delayed DDPG<br/>2018"]
    SAC["<b>SAC</b><br/>Soft Actor-Critic<br/>2018"]

    DQN -->|"continuous actions via<br/>a deterministic actor"| DDPG
    DDPG -->|"fixes value<br/>overestimation"| TD3
    DDPG -->|"fixes exploration<br/>and brittleness"| SAC
    TD3 -.->|"twin critics<br/>carried over"| SAC

    classDef ancestor fill:#dfe4e8,stroke:#5d6d7e,stroke-width:2px,color:#14202b
    classDef problem  fill:#f7cfa4,stroke:#b9701a,stroke-width:2px,color:#3b2205
    classDef fixblue  fill:#a9cce3,stroke:#21618c,stroke-width:2px,color:#0b2233
    classDef fixgreen fill:#a9dfbf,stroke:#1d8348,stroke-width:2px,color:#0a2b18

    class DQN ancestor
    class DDPG problem
    class TD3 fixblue
    class SAC fixgreen
```

TD3 and SAC are **not competitors**. They are two different diagnoses of the same patient — and SAC happens to have taken the other one's medicine too.

---

## 1. DDPG — the original bet

**Deep Deterministic Policy Gradient** (Lillicrap et al., 2015) took DQN's machinery — replay buffer, target networks — and made it work for *continuous* actions.

<div class="alert alert-secondary" role="alert" markdown="1">
**The problem it solves.** DQN picks actions with `argmax_a Q(s,a)`. That is intractable when `a` is a vector of real numbers — you can't enumerate a continuum. DDPG's answer: train a **deterministic actor** `μ(s)` to output the action the critic likes best, and update it along `∇_a Q(s,a)`.
</div>

```mermaid
graph LR
    ACTOR["<b>Actor</b> μ(s)<br/>deterministic"]
    NOISE["+ exploration noise<br/>externally injected"]
    ENV["Environment"]
    BUF[("Replay buffer")]
    CRITIC["<b>Critic</b> Q(s,a)"]
    TARGETS["Target nets μ′, Q′<br/>Polyak τ ≈ 0.005"]

    ACTOR --> NOISE --> ENV --> BUF
    BUF --> CRITIC
    BUF --> ACTOR
    TARGETS --> CRITIC
    CRITIC -->|"∇a Q(s,a)"| ACTOR

    classDef core    fill:#f7cfa4,stroke:#b9701a,stroke-width:2px,color:#3b2205
    classDef support fill:#e8eaed,stroke:#7f8c8d,stroke-width:1.5px,color:#1c2833
    classDef store   fill:#d4e6f1,stroke:#2471a3,stroke-width:1.5px,color:#0b2233

    class ACTOR,CRITIC core
    class NOISE,ENV,TARGETS support
    class BUF store
```

The TD target is

$$ y = r + \gamma\, Q'\big(s',\, \mu'(s')\big) $$

It works. It also has a reputation for being miserable to tune, and that reputation is deserved.

---

## 2. What actually goes wrong in DDPG

Two failures, and they compound.

### Failure A — Overestimation bias

```mermaid
graph LR
    A["Critic has<br/>approximation error"]
    B["Actor searches<br/>for the argmax"]
    C["Finds the error,<br/>not the truth"]
    D["Inflated Q enters<br/>the next TD target"]
    E["Critic learns<br/>the inflated value"]

    A --> B --> C --> D --> E
    E -->|"and round again"| A

    classDef neutral fill:#e8eaed,stroke:#7f8c8d,stroke-width:1.5px,color:#1c2833
    classDef bad     fill:#f5b7b1,stroke:#a93226,stroke-width:2px,color:#3d0e0a

    class A,B,D neutral
    class C,E bad
```

The TD target contains an implicit maximization. Approximation error is noise on top of the true value, and taking a near-maximum over a noisy estimate is **biased upward**. That inflated value feeds the next target. The bias doesn't wash out — it accumulates.

<div class="alert alert-warning" role="alert" markdown="1">
**You can watch this happen.** Log the critic's predicted `Q(s,a)` against the discounted return actually achieved from that state. In a failing run the prediction drifts steadily above the truth and never comes back.
</div>

**Click a legend entry to hide that trace.** Hide "critic — overestimating" and the healthy run looks unremarkable. That is exactly why this failure goes unnoticed if you only plot returns. Pan or zoom, then use the **house icon** in the toolbar to recenter.

```plotly
{
  "data": [
    {
      "x": [0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300],
      "y": [0,11.3,19.5,25.3,29.5,32.4,34.6,36.1,37.2,38.0,38.6,39.0,39.3,39.5,39.6,39.7],
      "type": "scatter", "mode": "lines",
      "name": "realized return (truth)",
      "line": {"color": "#9aa7b1", "width": 4},
      "hovertemplate": "truth: %{y:.1f}<extra></extra>"
    },
    {
      "x": [0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300],
      "y": [1.2,11.9,18.7,26.1,28.9,33.1,34.0,35.6,38.0,37.4,39.2,38.3,39.9,39.1,40.2,39.5],
      "type": "scatter", "mode": "lines+markers",
      "name": "critic — healthy",
      "line": {"color": "#2ecc71", "width": 2.5},
      "marker": {"size": 6},
      "hovertemplate": "healthy: %{y:.1f}<extra></extra>"
    },
    {
      "x": [0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300],
      "y": [0.8,13.9,24.7,33.1,39.9,45.4,50.2,54.3,58.0,61.4,64.6,67.6,70.5,73.3,76.0,78.7],
      "type": "scatter", "mode": "lines+markers",
      "name": "critic — overestimating",
      "line": {"color": "#ff6b5a", "width": 2.5, "dash": "dash"},
      "marker": {"size": 6, "symbol": "diamond"},
      "hovertemplate": "overestimating: %{y:.1f}<extra></extra>"
    }
  ],
  "layout": {
    "title": {"text": "Critic prediction vs. realized return", "font": {"size": 16}},
    "xaxis": {"title": {"text": "training step (thousands)"}, "gridcolor": "rgba(128,128,128,0.20)", "zeroline": false},
    "yaxis": {"title": {"text": "value"}, "gridcolor": "rgba(128,128,128,0.20)", "zeroline": false},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#9aa7b1"},
    "hovermode": "x unified",
    "margin": {"t": 55, "r": 25, "b": 55, "l": 65},
    "legend": {"orientation": "h", "y": -0.25}
  },
  "config": {"displayModeBar": true, "displaylogo": false, "responsive": true}
}
```

### Failure B — The actor exploits the critic's errors

The actor's entire job is to find the argmax of the critic. If the critic has a spurious bump — a region of action space it wrongly rates highly, usually because it has little data there — the actor will find it and go there.

This is adversarial in a way that's easy to miss: **the actor is optimizing against the critic's mistakes.**

Add a deterministic policy whose exploration comes entirely from externally injected noise, and you have something very sensitive to the noise schedule, the learning rates, and the random seed.

---

## 3. TD3 — three fixes for the bias

**Twin Delayed Deep Deterministic Policy Gradient** (Fujimoto, van Hoof & Meger, 2018). The paper's title is refreshingly literal: *Addressing Function Approximation Error in Actor-Critic Methods*.

Rather than three disconnected tricks, here is where each one sits in the flow that builds the TD target:

```mermaid
graph LR
    S["s′ from<br/>replay buffer"]
    MU["target actor<br/>μ′(s′)"]
    SM["<b>FIX 3</b><br/>+ clipped noise ε<br/>target smoothing"]
    AT["ã"]
    Q1["Q′₁(s′, ã)"]
    Q2["Q′₂(s′, ã)"]
    MIN["<b>FIX 1</b><br/>min( · , · )<br/>clipped double Q"]
    Y["TD target<br/>y = r + γ·min"]
    CU["critic update<br/>every step"]
    AU["<b>FIX 2</b><br/>actor update<br/>every d steps"]

    S --> MU --> SM --> AT
    AT --> Q1 --> MIN
    AT --> Q2 --> MIN
    MIN --> Y --> CU --> AU

    classDef fix     fill:#a9cce3,stroke:#21618c,stroke-width:2.5px,color:#0b2233
    classDef neutral fill:#e8eaed,stroke:#7f8c8d,stroke-width:1.5px,color:#1c2833
    classDef target  fill:#a9dfbf,stroke:#1d8348,stroke-width:2px,color:#0a2b18

    class SM,MIN,AU fix
    class S,MU,AT,Q1,Q2,CU neutral
    class Y target
```

| Fix | What it does | Why it works |
| --- | --- | --- |
| **1 — Clipped double Q-learning** | Two critics; target uses the smaller | Underestimation doesn't compound — an underrated action just isn't selected, so it never poisons future targets |
| **2 — Delayed policy updates** | Actor updated once per `d` critic updates (`d = 2`) | An actor chasing a moving critic chases noise. Let the value settle first |
| **3 — Target policy smoothing** | Noise added to the target action | Enforces "nearby actions have similar values," flattening the narrow spurious peaks the actor would exploit |

$$ y = r + \gamma \min_{i=1,2} Q_i'(s', \tilde a), \qquad \tilde a = \mu'(s') + \varepsilon, \quad \varepsilon \sim \mathrm{clip}\big(\mathcal N(0,\sigma), -c, c\big) $$

<div class="alert alert-secondary" role="alert" markdown="1">
**A control-theory reading of Fix 2.** Delayed policy updates are timescale separation — the inner loop (value estimation) should converge faster than the outer loop (policy improvement). The same instinct appears throughout adaptive control.
</div>

TD3 is substantially more stable than DDPG. But exploration is still **extrinsic** — you're still hand-tuning an external noise process, and the policy is still deterministic.

---

## 4. SAC — change the objective instead

**Soft Actor-Critic** (Haarnoja et al., 2018) attacks from a different direction. Rather than patching a deterministic policy, it changes what "optimal" means.

$$ J(\pi) = \sum_t \mathbb{E}\Big[\, r(s_t, a_t) \;+\; \alpha\, \mathcal{H}\big(\pi(\cdot \mid s_t)\big) \Big] $$

The agent is now rewarded for **acting as randomly as it can while still doing well**.

```mermaid
graph LR
    OBJ["<b>Maximum-entropy<br/>objective</b>"]
    E1["exploration<br/>becomes intrinsic"]
    E2["policy stays<br/>stochastic"]
    E3["α becomes the<br/>knob that matters"]
    R1["no noise schedule<br/>to tune"]
    R2["can't sit on a<br/>razor-thin optimum"]
    R3["auto-tuned<br/>in SAC v2"]

    OBJ --> E1 --> R1
    OBJ --> E2 --> R2
    OBJ --> E3 --> R3

    classDef root   fill:#a9dfbf,stroke:#1d8348,stroke-width:2.5px,color:#0a2b18
    classDef mid    fill:#d4efdf,stroke:#27ae60,stroke-width:2px,color:#0a2b18
    classDef leaf   fill:#e8eaed,stroke:#7f8c8d,stroke-width:1.5px,color:#1c2833

    class OBJ root
    class E1,E2,E3 mid
    class R1,R2,R3 leaf
```

### Move the slider

This is the part no static figure explains well. Below, a critic has two optima: a **broad, genuine** one near `a = +0.3`, and a **narrow, spurious spike** near `a = −0.6` — the kind of artifact a critic invents where it has little data.

Drag `α` and watch what the policy does.

- **`α → 0`** — the policy collapses onto the tallest point, spike included. This is DDPG's failure mode.
- **`α` moderate** — the policy prefers the broad optimum, because broad regions carry more entropy for the same value.
- **`α` large** — the task reward stops mattering and the policy spreads out aimlessly.

<div class="al-marimo-inline" data-show-code="false" markdown="1">

```python
import marimo as mo

alpha = mo.ui.slider(
    start=0.02, stop=1.5, step=0.02, value=0.20,
    label="temperature α", show_value=True,
)
alpha
```

```python
import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(-1.0, 1.0, 600)
da = float(a[1] - a[0])

Q = 1.00 * np.exp(-((a - 0.30) ** 2) / 0.090) + 1.12 * np.exp(-((a + 0.60) ** 2) / 0.0022)

z = Q / alpha.value
z = z - z.max()
pi = np.exp(z)
pi = pi / (pi.sum() * da)

entropy = float(-(pi * np.log(pi + 1e-12)).sum() * da)

# how much probability mass sits on each optimum
mass_spike = float(pi[a < -0.25].sum() * da)
mass_broad = float(pi[a >= -0.25].sum() * da)

fig, ax = plt.subplots(figsize=(7.6, 3.6))

ax.plot(a, Q, color="#5d6d7e", lw=2.4, label="critic  Q(s,a)")
ax.set_xlabel("action  a")
ax.set_ylabel("critic value  Q", color="#5d6d7e")
ax.set_ylim(0, float(Q.max()) * 1.20)
ax.grid(alpha=0.22)

ax2 = ax.twinx()
ax2.fill_between(a, 0, pi, color="#27ae60", alpha=0.35, label="policy  pi(a|s)")
ax2.plot(a, pi, color="#1e8449", lw=1.4)
ax2.set_ylabel("policy density", color="#1e8449")
ax2.set_ylim(0, float(pi.max()) * 1.20 + 1e-9)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

ax.set_title(
    "spurious spike: " + format(100 * mass_spike, ".0f") + "%"
    + "     broad optimum: " + format(100 * mass_broad, ".0f") + "%"
    + "     H = " + format(entropy, ".2f")
)
fig.tight_layout()
fig
```

</div>

Three consequences, all downstream of that single change to the objective:

**Exploration becomes intrinsic.** No injected noise, no schedule. The policy stays stochastic because staying stochastic *is worth reward*. Where several actions are near-equally good, SAC keeps its options open instead of committing prematurely.

**Brittleness drops.** A stochastic policy can't balance on a razor-thin spike — as the slider shows, the entropy term actively prefers the broad optimum. Broad optima generalize better and survive small model mismatch, which matters enormously when you eventually deploy on hardware that isn't your simulator.

**The temperature `α` becomes the knob that matters.** Too high and the policy is aimlessly random; too low and you are back to a deterministic policy with all of DDPG's problems.

<div class="alert alert-success" role="alert" markdown="1">
**Use the auto-tuning version.** The 2019 follow-up makes `α` adjust itself: you specify a target entropy (commonly `−dim(A)`, i.e. minus the action-space dimension) and `α` is tuned by dual gradient descent to hit it. There is no good reason to tune it by hand.
</div>

SAC also keeps TD3's twin critics with the `min`. The bias problem didn't go away — it is orthogonal to the entropy idea.

---

## 5. Side by side

| | **DDPG** | **TD3** | **SAC** |
| --- | --- | --- | --- |
| Full name | Deep Deterministic Policy Gradient | Twin Delayed DDPG | Soft Actor-Critic |
| Policy | deterministic | deterministic | **stochastic** |
| Exploration | injected noise | injected noise | **intrinsic (entropy)** |
| Critics | 1 | **2, take min** | **2, take min** |
| Actor update | every step | **every `d` steps** | every step |
| Target smoothing | — | **yes** | implicit (stochastic policy) |
| Main tuning burden | noise schedule, learning rates | noise schedule | temperature `α` (auto-tunable) |
| Robustness to seed | low | medium | **high** |

**The one-sentence version:** TD3 fixes DDPG's *value estimation*; SAC fixes DDPG's *exploration and brittleness*, and inherits TD3's value fix along the way.

---

## 6. A practitioner's note

I used SAC as one component of a [hybrid controller for vehicle lateral control](https://arxiv.org/abs/2608.17258), blending a learned policy with a constrained **MPC** (Model Predictive Control). Four things I would tell my earlier self:

<div class="alert alert-danger" role="alert" markdown="1">
**Report multiple seeds or don't report.** Single-seed continuous-control results are close to meaningless. My published numbers are mean ± standard deviation over five training seeds, and the spread was wide enough that any single run would have told a misleading story. This is the most common flaw in the RL results I read.
</div>

**SAC's stability doesn't transfer to your reward function.** SAC is robust to hyperparameters in a way DDPG isn't. It is *not* robust to a reward that accidentally makes a degenerate behavior optimal. Most of my debugging time went into the reward, not the algorithm.

**Watch the entropy, not just the return.** Two failure modes look identical on a return curve:

| Symptom | Likely cause |
| --- | --- |
| Entropy collapses to ~0 early | `α` too low — you've quietly reverted to a deterministic policy |
| Entropy stays pinned high | Task reward isn't outweighing the entropy bonus |

Drag the slider above to either extreme and you can see both failures directly.

**Understand what "it works" means before deploying.** SAC gave the best tracking error in my comparison. It also gives **no guarantee whatsoever** about what it will command in a state it hasn't seen. That gap — between *best average performance* and *acceptable worst case* — is the entire reason for pairing it with a model-based controller.

That's the next post.

---

## References

- Lillicrap et al., *Continuous Control with Deep Reinforcement Learning*, ICLR 2016 — DDPG. [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)
- Fujimoto, van Hoof & Meger, *Addressing Function Approximation Error in Actor-Critic Methods*, ICML 2018 — TD3. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)
- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*, ICML 2018 — SAC. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)
- Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*, 2019 — automatic temperature tuning. [arXiv:1812.05905](https://arxiv.org/abs/1812.05905)
