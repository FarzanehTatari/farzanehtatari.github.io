---
layout: post
title: "From DDPG to SAC: a lineage of fixes"
date: 2026-08-21
description: Deep Deterministic Policy Gradient, Twin Delayed DDPG, and Soft Actor-Critic are usually presented as three algorithms. They're better understood as one algorithm and two rounds of debugging.
tags: reinforcement-learning sac td3 ddpg actor-critic
categories: reinforcement-learning-for-control
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

```plotly

{"data":[{"x":[-1.0,-0.98,-0.96,-0.94,-0.92,-0.9,-0.88,-0.86,-0.84,-0.82,-0.8,-0.78,-0.76,-0.74,-0.72,-0.7,-0.68,-0.66,-0.64,-0.62,-0.6,-0.58,-0.56,-0.54,-0.52,-0.5,-0.48,-0.46,-0.44,-0.42,-0.4,-0.38,-0.36,-0.34,-0.32,-0.3,-0.28,-0.26,-0.24,-0.22,-0.2,-0.18,-0.16,-0.14,-0.12,-0.1,-0.08,-0.06,-0.04,-0.02,0.0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.6,0.62,0.64,0.66,0.68,0.7,0.72,0.74,0.76,0.78,0.8,0.82,0.84,0.86,0.88,0.9,0.92,0.94,0.96,0.98,1.0],"y":[0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.0121,0.0129,0.0169,0.0404,0.2434,2.1555,6.0659,2.1567,0.2437,0.0405,0.0169,0.0129,0.0122,0.0122,0.0122,0.0122,0.0123,0.0124,0.0126,0.0128,0.013,0.0133,0.0137,0.0143,0.015,0.0158,0.017,0.0185,0.0204,0.023,0.0263,0.0308,0.0368,0.0449,0.056,0.0714,0.0929,0.123,0.1655,0.2252,0.3087,0.424,0.5803,0.7866,1.0493,1.3688,1.7352,2.1255,2.503,2.822,3.0365,3.1122,3.0365,2.822,2.503,2.1255,1.7352,1.3688,1.0493,0.7866,0.5803,0.424,0.3087,0.2252,0.1655,0.123,0.0929,0.0714,0.056,0.0449,0.0368,0.0308,0.0263,0.023,0.0204,0.0185,0.017,0.0158,0.015,0.0143,0.0137,0.0133,0.013,0.0128,0.0126,0.0124,0.0123],"type":"scatter","mode":"lines","name":"policy  \u03c0(a|s)","fill":"tozeroy","line":{"color":"#27ae60","width":2},"fillcolor":"rgba(39,174,96,0.35)","yaxis":"y2","hovertemplate":"\u03c0 = %{y:.2f}<extra></extra>"},{"x":[-1.0,-0.98,-0.96,-0.94,-0.92,-0.9,-0.88,-0.86,-0.84,-0.82,-0.8,-0.78,-0.76,-0.74,-0.72,-0.7,-0.68,-0.66,-0.64,-0.62,-0.6,-0.58,-0.56,-0.54,-0.52,-0.5,-0.48,-0.46,-0.44,-0.42,-0.4,-0.38,-0.36,-0.34,-0.32,-0.3,-0.28,-0.26,-0.24,-0.22,-0.2,-0.18,-0.16,-0.14,-0.12,-0.1,-0.08,-0.06,-0.04,-0.02,0.0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.6,0.62,0.64,0.66,0.68,0.7,0.72,0.74,0.76,0.78,0.8,0.82,0.84,0.86,0.88,0.9,0.92,0.94,0.96,0.98,1.0],"y":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0002,0.0016,0.0119,0.0611,0.2181,0.5413,0.9339,1.1201,0.934,0.5415,0.2184,0.0616,0.0127,0.0028,0.0018,0.0023,0.0032,0.0043,0.0059,0.0079,0.0106,0.014,0.0183,0.0238,0.0307,0.0392,0.0496,0.0622,0.0773,0.0953,0.1164,0.1409,0.169,0.201,0.2369,0.2768,0.3205,0.3679,0.4185,0.4718,0.5273,0.584,0.6412,0.6977,0.7524,0.8043,0.8521,0.8948,0.9314,0.9608,0.9824,0.9956,1.0,0.9956,0.9824,0.9608,0.9314,0.8948,0.8521,0.8043,0.7524,0.6977,0.6412,0.584,0.5273,0.4718,0.4185,0.3679,0.3205,0.2768,0.2369,0.201,0.169,0.1409,0.1164,0.0953,0.0773,0.0622,0.0496,0.0392,0.0307,0.0238,0.0183,0.014,0.0106,0.0079,0.0059,0.0043],"type":"scatter","mode":"lines","name":"critic  Q(s,a)","line":{"color":"#8fa0b0","width":3},"hovertemplate":"Q = %{y:.2f}<extra></extra>"}],"layout":{"title":{"text":"spike 23%   |   broad 77%   |   H = -0.61","font":{"size":15}},"xaxis":{"title":{"text":"action  a"},"gridcolor":"rgba(128,128,128,0.18)","zeroline":false,"domain":[0,1]},"yaxis":{"title":{"text":"critic value  Q"},"gridcolor":"rgba(128,128,128,0.18)","zeroline":false},"yaxis2":{"title":{"text":"policy density"},"overlaying":"y","side":"right","showgrid":false,"rangemode":"tozero"},"paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)","font":{"color":"#9aa7b1"},"margin":{"t":80,"r":70,"b":110,"l":65},"legend":{"orientation":"h","y":1.14,"x":0,"font":{"size":11}},"sliders":[{"active":3,"y":-0.34,"x":0.05,"len":0.9,"currentvalue":{"prefix":"temperature \u03b1 = ","font":{"size":14},"xanchor":"left","offset":12},"pad":{"t":10,"b":10},"steps":[{"label":"0.05","method":"update","args":[{"y":[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0003,0.7551,31.3055,0.7566,0.0003,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0001,0.0002,0.0007,0.0022,0.0067,0.02,0.0566,0.1472,0.3458,0.7178,1.2932,1.9915,2.5925,2.833,2.5925,1.9915,1.2932,0.7178,0.3458,0.1472,0.0566,0.02,0.0067,0.0022,0.0007,0.0002,0.0001,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]},{"title":{"text":"spike 66%   |   broad 34%   |   H = -2.33"}},[0]]},{"label":"0.08","method":"update","args":[{"y":[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0002,0.0123,1.6642,17.0699,1.6663,0.0123,0.0002,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0001,0.0001,0.0001,0.0002,0.0003,0.0005,0.0008,0.0014,0.0027,0.0052,0.0103,0.021,0.0429,0.0869,0.1722,0.3294,0.599,1.0215,1.6124,2.3295,3.0511,3.5979,3.8029,3.5979,3.0511,2.3295,1.6124,1.0215,0.599,0.3294,0.1722,0.0869,0.0429,0.021,0.0103,0.0052,0.0027,0.0014,0.0008,0.0005,0.0003,0.0002,0.0001,0.0001,0.0001,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]},{"title":{"text":"spike 41%   |   broad 59%   |   H = -1.47"}},[0]]},{"label":"0.12","method":"update","args":[{"y":[[0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.001,0.0015,0.0054,0.08,2.11,9.9608,2.1118,0.0802,0.0054,0.0015,0.001,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.001,0.001,0.001,0.0011,0.0011,0.0012,0.0013,0.0015,0.0017,0.0019,0.0023,0.0028,0.0036,0.0047,0.0063,0.0088,0.0127,0.0189,0.0288,0.0449,0.0712,0.1143,0.1841,0.2947,0.4651,0.7167,1.0677,1.5239,2.066,2.6403,3.1607,3.5278,3.6606,3.5278,3.1607,2.6403,2.066,1.5239,1.0677,0.7167,0.4651,0.2947,0.1841,0.1143,0.0712,0.0449,0.0288,0.0189,0.0127,0.0088,0.0063,0.0047,0.0036,0.0028,0.0023,0.0019,0.0017,0.0015,0.0013,0.0012,0.0011,0.0011,0.001,0.001,0.001,0.0009,0.0009,0.0009]]},{"title":{"text":"spike 29%   |   broad 71%   |   H = -1.02"}},[0]]},{"label":"0.18","method":"update","args":[{"y":[[0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.0121,0.0129,0.0169,0.0404,0.2434,2.1555,6.0659,2.1567,0.2437,0.0405,0.0169,0.0129,0.0122,0.0122,0.0122,0.0122,0.0123,0.0124,0.0126,0.0128,0.013,0.0133,0.0137,0.0143,0.015,0.0158,0.017,0.0185,0.0204,0.023,0.0263,0.0308,0.0368,0.0449,0.056,0.0714,0.0929,0.123,0.1655,0.2252,0.3087,0.424,0.5803,0.7866,1.0493,1.3688,1.7352,2.1255,2.503,2.822,3.0365,3.1122,3.0365,2.822,2.503,2.1255,1.7352,1.3688,1.0493,0.7866,0.5803,0.424,0.3087,0.2252,0.1655,0.123,0.0929,0.0714,0.056,0.0449,0.0368,0.0308,0.0263,0.023,0.0204,0.0185,0.017,0.0158,0.015,0.0143,0.0137,0.0133,0.013,0.0128,0.0126,0.0124,0.0123]]},{"title":{"text":"spike 23%   |   broad 77%   |   H = -0.61"}},[0]]},{"label":"0.25","method":"update","args":[{"y":[[0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0467,0.0469,0.0489,0.0595,0.1115,0.4063,1.954,4.1158,1.9548,0.4067,0.1117,0.0597,0.0491,0.0471,0.047,0.0471,0.0472,0.0474,0.0477,0.0481,0.0486,0.0493,0.0502,0.0513,0.0527,0.0545,0.0568,0.0598,0.0635,0.0682,0.0743,0.0819,0.0917,0.1042,0.1203,0.1411,0.168,0.2031,0.2486,0.3078,0.3842,0.4822,0.606,0.7596,0.9456,1.1636,1.409,1.6714,1.9343,2.176,2.3723,2.5008,2.5455,2.5008,2.3723,2.176,1.9343,1.6714,1.409,1.1636,0.9456,0.7596,0.606,0.4822,0.3842,0.3078,0.2486,0.2031,0.168,0.1411,0.1203,0.1042,0.0917,0.0819,0.0743,0.0682,0.0635,0.0598,0.0568,0.0545,0.0527,0.0513,0.0502,0.0493,0.0486,0.0481,0.0477,0.0474]]},{"title":{"text":"spike 21%   |   broad 79%   |   H = -0.24"}},[0]]},{"label":"0.35","method":"update","args":[{"y":[[0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1137,0.1171,0.1347,0.211,0.5313,1.6313,2.7773,1.6317,0.5316,0.2112,0.135,0.1174,0.1141,0.1137,0.1139,0.1142,0.1146,0.1151,0.1158,0.1166,0.1178,0.1192,0.1211,0.1235,0.1266,0.1304,0.1352,0.1411,0.1486,0.1578,0.1692,0.1834,0.201,0.2227,0.2496,0.2828,0.3237,0.3741,0.4357,0.5105,0.6004,0.7068,0.8307,0.9713,1.1265,1.2915,1.4591,1.6195,1.7616,1.8737,1.9456,1.9704,1.9456,1.8737,1.7616,1.6195,1.4591,1.2915,1.1265,0.9713,0.8307,0.7068,0.6004,0.5105,0.4357,0.3741,0.3237,0.2828,0.2496,0.2227,0.201,0.1834,0.1692,0.1578,0.1486,0.1411,0.1352,0.1304,0.1266,0.1235,0.1211,0.1192,0.1178,0.1166,0.1158,0.1151,0.1146]]},{"title":{"text":"spike 22%   |   broad 78%   |   H = 0.13"}},[0]]},{"label":"0.50","method":"update","args":[{"y":[[0.1998,0.1998,0.1998,0.1998,0.1998,0.1998,0.1998,0.1998,0.1998,0.1998,0.1998,0.1998,0.1998,0.1999,0.2005,0.2047,0.2258,0.3091,0.5899,1.2937,1.8776,1.294,0.5902,0.3093,0.2261,0.205,0.2009,0.2006,0.2008,0.2011,0.2016,0.2022,0.203,0.2041,0.2055,0.2073,0.2096,0.2125,0.2161,0.2207,0.2263,0.2333,0.2418,0.2522,0.2649,0.2802,0.2987,0.321,0.3476,0.3794,0.4171,0.4615,0.5135,0.5737,0.6426,0.7204,0.8066,0.9,0.9984,1.0986,1.1965,1.2872,1.3652,1.4255,1.4636,1.4766,1.4636,1.4255,1.3652,1.2872,1.1965,1.0986,0.9984,0.9,0.8066,0.7204,0.6426,0.5737,0.5135,0.4615,0.4171,0.3794,0.3476,0.321,0.2987,0.2802,0.2649,0.2522,0.2418,0.2333,0.2263,0.2207,0.2161,0.2125,0.2096,0.2073,0.2055,0.2041,0.203,0.2022,0.2016]]},{"title":{"text":"spike 25%   |   broad 75%   |   H = 0.40"}},[0]]},{"label":"0.70","method":"update","args":[{"y":[[0.2755,0.2755,0.2755,0.2755,0.2755,0.2755,0.2755,0.2755,0.2755,0.2755,0.2755,0.2755,0.2755,0.2756,0.2761,0.2802,0.3006,0.3762,0.5969,1.0459,1.3647,1.0461,0.5971,0.3764,0.3008,0.2805,0.2766,0.2762,0.2764,0.2767,0.2772,0.2778,0.2786,0.2797,0.281,0.2828,0.285,0.2878,0.2913,0.2957,0.3011,0.3077,0.3157,0.3253,0.3369,0.3507,0.3671,0.3865,0.4091,0.4355,0.466,0.5009,0.5406,0.5851,0.6345,0.6885,0.7464,0.8071,0.8692,0.9307,0.9892,1.0422,1.0869,1.121,1.1423,1.1495,1.1423,1.121,1.0869,1.0422,0.9892,0.9307,0.8692,0.8071,0.7464,0.6885,0.6345,0.5851,0.5406,0.5009,0.466,0.4355,0.4091,0.3865,0.3671,0.3507,0.3369,0.3253,0.3157,0.3077,0.3011,0.2957,0.2913,0.2878,0.285,0.2828,0.281,0.2797,0.2786,0.2778,0.2772]]},{"title":{"text":"spike 28%   |   broad 72%   |   H = 0.55"}},[0]]},{"label":"1.00","method":"update","args":[{"y":[[0.3391,0.3391,0.3391,0.3391,0.3391,0.3391,0.3391,0.3391,0.3391,0.3391,0.3391,0.3391,0.3391,0.3392,0.3397,0.3432,0.3605,0.4218,0.5827,0.8629,1.0395,0.863,0.5828,0.4219,0.3607,0.3435,0.3401,0.3397,0.3399,0.3402,0.3406,0.3411,0.3418,0.3427,0.3439,0.3454,0.3473,0.3497,0.3527,0.3564,0.3609,0.3664,0.373,0.381,0.3904,0.4016,0.4146,0.4298,0.4473,0.4673,0.4899,0.5154,0.5436,0.5746,0.6082,0.6439,0.6813,0.7197,0.758,0.7951,0.8298,0.8607,0.8864,0.9058,0.9178,0.9219,0.9178,0.9058,0.8864,0.8607,0.8298,0.7951,0.758,0.7197,0.6813,0.6439,0.6082,0.5746,0.5436,0.5154,0.4899,0.4673,0.4473,0.4298,0.4146,0.4016,0.3904,0.381,0.373,0.3664,0.3609,0.3564,0.3527,0.3497,0.3473,0.3454,0.3439,0.3427,0.3418,0.3411,0.3406]]},{"title":{"text":"spike 31%   |   broad 69%   |   H = 0.63"}},[0]]},{"label":"1.50","method":"update","args":[{"y":[[0.3909,0.3909,0.3909,0.3909,0.3909,0.3909,0.3909,0.3909,0.3909,0.3909,0.3909,0.3909,0.3909,0.391,0.3914,0.3941,0.4072,0.4521,0.5608,0.7286,0.8249,0.7287,0.5609,0.4522,0.4073,0.3943,0.3917,0.3914,0.3915,0.3918,0.3921,0.3925,0.393,0.3937,0.3946,0.3957,0.3972,0.399,0.4013,0.4041,0.4075,0.4116,0.4166,0.4225,0.4294,0.4376,0.447,0.4578,0.4702,0.4841,0.4996,0.5167,0.5354,0.5556,0.577,0.5994,0.6225,0.6456,0.6683,0.69,0.7099,0.7274,0.7418,0.7525,0.7592,0.7614,0.7592,0.7525,0.7418,0.7274,0.7099,0.69,0.6683,0.6456,0.6225,0.5994,0.577,0.5556,0.5354,0.5167,0.4996,0.4841,0.4702,0.4578,0.447,0.4376,0.4294,0.4225,0.4166,0.4116,0.4075,0.4041,0.4013,0.399,0.3972,0.3957,0.3946,0.3937,0.393,0.3925,0.3921]]},{"title":{"text":"spike 33%   |   broad 67%   |   H = 0.67"}},[0]]}]}]}}

```

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
