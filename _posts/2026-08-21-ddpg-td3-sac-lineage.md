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
{"data":[{"x":[-1.0,-0.9833,-0.9667,-0.95,-0.9333,-0.9167,-0.9,-0.8833,-0.8667,-0.85,-0.8333,-0.8167,-0.8,-0.7833,-0.7667,-0.75,-0.7333,-0.7167,-0.7,-0.6833,-0.6667,-0.65,-0.6333,-0.6167,-0.6,-0.5833,-0.5667,-0.55,-0.5333,-0.5167,-0.5,-0.4833,-0.4667,-0.45,-0.4333,-0.4167,-0.4,-0.3833,-0.3667,-0.35,-0.3333,-0.3167,-0.3,-0.2833,-0.2667,-0.25,-0.2333,-0.2167,-0.2,-0.1833,-0.1667,-0.15,-0.1333,-0.1167,-0.1,-0.0833,-0.0667,-0.05,-0.0333,-0.0167,0.0,0.0167,0.0333,0.05,0.0667,0.0833,0.1,0.1167,0.1333,0.15,0.1667,0.1833,0.2,0.2167,0.2333,0.25,0.2667,0.2833,0.3,0.3167,0.3333,0.35,0.3667,0.3833,0.4,0.4167,0.4333,0.45,0.4667,0.4833,0.5,0.5167,0.5333,0.55,0.5667,0.5833,0.6,0.6167,0.6333,0.65,0.6667,0.6833,0.7,0.7167,0.7333,0.75,0.7667,0.7833,0.8,0.8167,0.8333,0.85,0.8667,0.8833,0.9,0.9167,0.9333,0.95,0.9667,0.9833,1.0],"y":[0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.0121,0.0122,0.0129,0.0157,0.0275,0.0887,0.5145,2.9003,6.0682,2.9016,0.515,0.0888,0.0275,0.0157,0.0129,0.0123,0.0122,0.0122,0.0122,0.0123,0.0123,0.0124,0.0125,0.0127,0.0128,0.0131,0.0133,0.0137,0.0141,0.0146,0.0152,0.016,0.017,0.0182,0.0197,0.0216,0.024,0.027,0.0308,0.0356,0.0419,0.05,0.0606,0.0745,0.0929,0.1173,0.1497,0.1929,0.2501,0.3255,0.4241,0.5512,0.712,0.911,1.1502,1.4275,1.7358,2.0609,2.3822,2.6738,2.9082,3.0605,3.1134,3.0605,2.9082,2.6738,2.3822,2.0609,1.7358,1.4275,1.1502,0.911,0.712,0.5512,0.4241,0.3255,0.2501,0.1929,0.1497,0.1173,0.0929,0.0745,0.0606,0.05,0.0419,0.0356,0.0308,0.027,0.024,0.0216,0.0197,0.0182,0.017,0.016,0.0152,0.0146,0.0141,0.0137,0.0133,0.0131,0.0128,0.0127,0.0125,0.0124,0.0123],"type":"scatter","mode":"lines","name":"policy  \u03c0(a|s)","fill":"tozeroy","line":{"color":"#27ae60","width":2},"fillcolor":"rgba(39,174,96,0.35)","yaxis":"y2","hovertemplate":"\u03c0 = %{y:.2f}<extra></extra>"},{"x":[-1.0,-0.9833,-0.9667,-0.95,-0.9333,-0.9167,-0.9,-0.8833,-0.8667,-0.85,-0.8333,-0.8167,-0.8,-0.7833,-0.7667,-0.75,-0.7333,-0.7167,-0.7,-0.6833,-0.6667,-0.65,-0.6333,-0.6167,-0.6,-0.5833,-0.5667,-0.55,-0.5333,-0.5167,-0.5,-0.4833,-0.4667,-0.45,-0.4333,-0.4167,-0.4,-0.3833,-0.3667,-0.35,-0.3333,-0.3167,-0.3,-0.2833,-0.2667,-0.25,-0.2333,-0.2167,-0.2,-0.1833,-0.1667,-0.15,-0.1333,-0.1167,-0.1,-0.0833,-0.0667,-0.05,-0.0333,-0.0167,0.0,0.0167,0.0333,0.05,0.0667,0.0833,0.1,0.1167,0.1333,0.15,0.1667,0.1833,0.2,0.2167,0.2333,0.25,0.2667,0.2833,0.3,0.3167,0.3333,0.35,0.3667,0.3833,0.4,0.4167,0.4333,0.45,0.4667,0.4833,0.5,0.5167,0.5333,0.55,0.5667,0.5833,0.6,0.6167,0.6333,0.65,0.6667,0.6833,0.7,0.7167,0.7333,0.75,0.7667,0.7833,0.8,0.8167,0.8333,0.85,0.8667,0.8833,0.9,0.9167,0.9333,0.95,0.9667,0.9833,1.0],"y":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0004,0.0023,0.0119,0.0477,0.1486,0.3595,0.676,0.9872,1.1201,0.9873,0.6761,0.3598,0.149,0.0483,0.0127,0.0034,0.0018,0.002,0.0025,0.0033,0.0043,0.0056,0.0072,0.0091,0.0116,0.0146,0.0183,0.0228,0.0282,0.0347,0.0424,0.0515,0.0622,0.0746,0.0889,0.1054,0.1241,0.1453,0.169,0.1954,0.2245,0.2564,0.291,0.3282,0.3679,0.4098,0.4538,0.4994,0.5461,0.5936,0.6412,0.6884,0.7344,0.7788,0.8208,0.8596,0.8948,0.9257,0.9518,0.9726,0.9877,0.9969,1.0,0.9969,0.9877,0.9726,0.9518,0.9257,0.8948,0.8596,0.8208,0.7788,0.7344,0.6884,0.6412,0.5936,0.5461,0.4994,0.4538,0.4098,0.3679,0.3282,0.291,0.2564,0.2245,0.1954,0.169,0.1453,0.1241,0.1054,0.0889,0.0746,0.0622,0.0515,0.0424,0.0347,0.0282,0.0228,0.0183,0.0146,0.0116,0.0091,0.0072,0.0056,0.0043],"type":"scatter","mode":"lines","name":"critic  Q(s,a)","line":{"color":"#8fa0b0","width":3},"hovertemplate":"Q = %{y:.2f}<extra></extra>"}],"layout":{"title":{"text":"spike 23%   |   broad 77%   |   H = -0.61","font":{"size":15}},"xaxis":{"title":{"text":"action  a"},"gridcolor":"rgba(128,128,128,0.20)","zeroline":false},"yaxis":{"title":{"text":"critic value  Q"},"gridcolor":"rgba(128,128,128,0.20)","zeroline":false},"yaxis2":{"title":{"text":"policy density"},"overlaying":"y","side":"right","showgrid":false,"rangemode":"tozero"},"paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(0,0,0,0)","font":{"color":"#9aa7b1"},"margin":{"t":55,"r":70,"b":95,"l":65},"legend":{"orientation":"h","y":-0.32},"sliders":[{"active":3,"y":-0.06,"x":0.06,"len":0.9,"currentvalue":{"prefix":"temperature \u03b1 = ","font":{"size":14}},"pad":{"t":40},"steps":[{"label":"0.05","method":"animate","args":[["0.05"],{"mode":"immediate","frame":{"duration":0,"redraw":true},"transition":{"duration":0}}]},{"label":"0.08","method":"animate","args":[["0.08"],{"mode":"immediate","frame":{"duration":0,"redraw":true},"transition":{"duration":0}}]},{"label":"0.12","method":"animate","args":[["0.12"],{"mode":"immediate","frame":{"duration":0,"redraw":true},"transition":{"duration":0}}]},{"label":"0.18","method":"animate","args":[["0.18"],{"mode":"immediate","frame":{"duration":0,"redraw":true},"transition":{"duration":0}}]},{"label":"0.25","method":"animate","args":[["0.25"],{"mode":"immediate","frame":{"duration":0,"redraw":true},"transition":{"duration":0}}]},{"label":"0.35","method":"animate","args":[["0.35"],{"mode":"immediate","frame":{"duration":0,"redraw":true},"transition":{"duration":0}}]},{"label":"0.50","method":"animate","args":[["0.50"],{"mode":"immediate","frame":{"duration":0,"redraw":true},"transition":{"duration":0}}]},{"label":"0.70","method":"animate","args":[["0.70"],{"mode":"immediate","frame":{"duration":0,"redraw":true},"transition":{"duration":0}}]},{"label":"1.00","method":"animate","args":[["1.00"],{"mode":"immediate","frame":{"duration":0,"redraw":true},"transition":{"duration":0}}]},{"label":"1.50","method":"animate","args":[["1.50"],{"mode":"immediate","frame":{"duration":0,"redraw":true},"transition":{"duration":0}}]}]}]},"frames":[{"name":"0.05","data":[{"y":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0046,2.3379,33.3474,2.3418,0.0046,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0001,0.0001,0.0003,0.0009,0.0023,0.0059,0.0149,0.0362,0.0837,0.1822,0.3684,0.6834,1.1512,1.7447,2.3611,2.8374,3.0178,2.8374,2.3611,1.7447,1.1512,0.6834,0.3684,0.1822,0.0837,0.0362,0.0149,0.0059,0.0023,0.0009,0.0003,0.0001,0.0001,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}],"layout":{"title":{"text":"spike 63%   |   broad 37%   |   H = -2.23"}}},{"name":"0.08","data":[{"y":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0001,0.0013,0.0671,3.287,17.3057,3.2904,0.0673,0.0013,0.0001,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0001,0.0001,0.0001,0.0001,0.0002,0.0002,0.0004,0.0005,0.0009,0.0014,0.0024,0.0042,0.0074,0.0132,0.024,0.0435,0.0784,0.1395,0.2428,0.4102,0.667,1.0356,1.5239,2.1111,2.7375,3.3073,3.7098,3.8555,3.7098,3.3073,2.7375,2.1111,1.5239,1.0356,0.667,0.4102,0.2428,0.1395,0.0784,0.0435,0.024,0.0132,0.0074,0.0042,0.0024,0.0014,0.0009,0.0005,0.0004,0.0002,0.0002,0.0001,0.0001,0.0001,0.0001,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}],"layout":{"title":{"text":"spike 40%   |   broad 60%   |   H = -1.43"}}},{"name":"0.12","data":[{"y":[0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.001,0.0013,0.003,0.0176,0.2465,3.2993,9.9852,3.3016,0.2469,0.0177,0.0031,0.0013,0.001,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.0009,0.001,0.001,0.001,0.001,0.0011,0.0011,0.0012,0.0013,0.0014,0.0015,0.0016,0.0019,0.0021,0.0025,0.003,0.0036,0.0045,0.0057,0.0075,0.01,0.0136,0.0189,0.0268,0.0387,0.0566,0.0835,0.1241,0.1845,0.2734,0.4014,0.5809,0.824,1.1394,1.5277,1.9764,2.456,2.9206,3.3129,3.5765,3.6696,3.5765,3.3129,2.9206,2.456,1.9764,1.5277,1.1394,0.824,0.5809,0.4014,0.2734,0.1845,0.1241,0.0835,0.0566,0.0387,0.0268,0.0189,0.0136,0.01,0.0075,0.0057,0.0045,0.0036,0.003,0.0025,0.0021,0.0019,0.0016,0.0015,0.0014,0.0013,0.0012,0.0011,0.0011,0.001,0.001,0.001,0.001,0.0009,0.0009,0.0009]}],"layout":{"title":{"text":"spike 29%   |   broad 71%   |   H = -1.02"}}},{"name":"0.18","data":[{"y":[0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.0121,0.0122,0.0129,0.0157,0.0275,0.0887,0.5145,2.9003,6.0682,2.9016,0.515,0.0888,0.0275,0.0157,0.0129,0.0123,0.0122,0.0122,0.0122,0.0123,0.0123,0.0124,0.0125,0.0127,0.0128,0.0131,0.0133,0.0137,0.0141,0.0146,0.0152,0.016,0.017,0.0182,0.0197,0.0216,0.024,0.027,0.0308,0.0356,0.0419,0.05,0.0606,0.0745,0.0929,0.1173,0.1497,0.1929,0.2501,0.3255,0.4241,0.5512,0.712,0.911,1.1502,1.4275,1.7358,2.0609,2.3822,2.6738,2.9082,3.0605,3.1134,3.0605,2.9082,2.6738,2.3822,2.0609,1.7358,1.4275,1.1502,0.911,0.712,0.5512,0.4241,0.3255,0.2501,0.1929,0.1497,0.1173,0.0929,0.0745,0.0606,0.05,0.0419,0.0356,0.0308,0.027,0.024,0.0216,0.0197,0.0182,0.017,0.016,0.0152,0.0146,0.0141,0.0137,0.0133,0.0131,0.0128,0.0127,0.0125,0.0124,0.0123]}],"layout":{"title":{"text":"spike 23%   |   broad 77%   |   H = -0.61"}}},{"name":"0.25","data":[{"y":[0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0466,0.0467,0.0471,0.0489,0.0564,0.0845,0.1965,0.6965,2.4193,4.1166,2.4201,0.697,0.1967,0.0846,0.0566,0.0491,0.0473,0.047,0.047,0.0471,0.0473,0.0474,0.0477,0.048,0.0484,0.0488,0.0494,0.0502,0.0511,0.0522,0.0536,0.0553,0.0573,0.0598,0.0628,0.0666,0.0711,0.0766,0.0834,0.0917,0.1019,0.1145,0.13,0.1493,0.1733,0.2031,0.2403,0.2864,0.3437,0.4144,0.501,0.6061,0.7319,0.8801,1.051,1.243,1.4523,1.6718,1.8918,2.0997,2.2818,2.4241,2.5149,2.5461,2.5149,2.4241,2.2818,2.0997,1.8918,1.6718,1.4523,1.243,1.051,0.8801,0.7319,0.6061,0.501,0.4144,0.3437,0.2864,0.2403,0.2031,0.1733,0.1493,0.13,0.1145,0.1019,0.0917,0.0834,0.0766,0.0711,0.0666,0.0628,0.0598,0.0573,0.0553,0.0536,0.0522,0.0511,0.0502,0.0494,0.0488,0.0484,0.048,0.0477,0.0474]}],"layout":{"title":{"text":"spike 21%   |   broad 79%   |   H = -0.24"}}},{"name":"0.35","data":[{"y":[0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1132,0.1133,0.114,0.1171,0.1297,0.1731,0.3163,0.781,1.9006,2.7783,1.9011,0.7814,0.3165,0.1733,0.13,0.1174,0.1143,0.1138,0.1139,0.114,0.1143,0.1146,0.115,0.1156,0.1162,0.117,0.118,0.1193,0.1208,0.1227,0.125,0.1278,0.1312,0.1352,0.1401,0.146,0.153,0.1614,0.1715,0.1835,0.1979,0.215,0.2355,0.26,0.2891,0.3239,0.3651,0.414,0.4715,0.5389,0.6172,0.7071,0.8091,0.923,1.0478,1.1812,1.32,1.4596,1.5944,1.7177,1.8228,1.9033,1.9539,1.9712,1.9539,1.9033,1.8228,1.7177,1.5944,1.4596,1.32,1.1812,1.0478,0.923,0.8091,0.7071,0.6172,0.5389,0.4715,0.414,0.3651,0.3239,0.2891,0.26,0.2355,0.215,0.1979,0.1835,0.1715,0.1614,0.153,0.146,0.1401,0.1352,0.1312,0.1278,0.125,0.1227,0.1208,0.1193,0.118,0.117,0.1162,0.1156,0.115,0.1146]}],"layout":{"title":{"text":"spike 22%   |   broad 78%   |   H = 0.13"}}},{"name":"0.50","data":[{"y":[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2001,0.2009,0.2048,0.22,0.2692,0.4105,0.7728,1.4404,1.8789,1.4406,0.7731,0.4107,0.2694,0.2202,0.2051,0.2013,0.2007,0.2008,0.201,0.2013,0.2017,0.2022,0.2029,0.2037,0.2047,0.2059,0.2074,0.2093,0.2116,0.2143,0.2177,0.2217,0.2265,0.2321,0.2389,0.2469,0.2563,0.2674,0.2804,0.2956,0.3133,0.3339,0.3578,0.3855,0.4174,0.4539,0.4956,0.5429,0.5961,0.6554,0.7209,0.7922,0.8688,0.9493,1.0324,1.116,1.1973,1.2737,1.3419,1.3988,1.4418,1.4685,1.4776,1.4685,1.4418,1.3988,1.3419,1.2737,1.1973,1.116,1.0324,0.9493,0.8688,0.7922,0.7209,0.6554,0.5961,0.5429,0.4956,0.4539,0.4174,0.3855,0.3578,0.3339,0.3133,0.2956,0.2804,0.2674,0.2563,0.2469,0.2389,0.2321,0.2265,0.2217,0.2177,0.2143,0.2116,0.2093,0.2074,0.2059,0.2047,0.2037,0.2029,0.2022,0.2017]}],"layout":{"title":{"text":"spike 25%   |   broad 75%   |   H = 0.40"}}},{"name":"0.70","data":[{"y":[0.2757,0.2757,0.2757,0.2757,0.2757,0.2757,0.2757,0.2757,0.2757,0.2757,0.2757,0.2757,0.2757,0.2757,0.2757,0.2758,0.2759,0.2767,0.2805,0.2952,0.3409,0.4609,0.7242,1.1298,1.366,1.13,0.7244,0.4611,0.3411,0.2954,0.2808,0.2771,0.2765,0.2765,0.2767,0.2771,0.2775,0.278,0.2786,0.2794,0.2804,0.2816,0.2831,0.2849,0.2871,0.2898,0.293,0.2968,0.3014,0.3068,0.3131,0.3206,0.3292,0.3393,0.351,0.3645,0.38,0.3977,0.4179,0.4407,0.4664,0.4952,0.5273,0.5627,0.6016,0.6438,0.6891,0.7372,0.7874,0.8389,0.8907,0.9416,0.9901,1.0348,1.0741,1.1064,1.1306,1.1455,1.1506,1.1455,1.1306,1.1064,1.0741,1.0348,0.9901,0.9416,0.8907,0.8389,0.7874,0.7372,0.6891,0.6438,0.6016,0.5627,0.5273,0.4952,0.4664,0.4407,0.4179,0.3977,0.38,0.3645,0.351,0.3393,0.3292,0.3206,0.3131,0.3068,0.3014,0.2968,0.293,0.2898,0.2871,0.2849,0.2831,0.2816,0.2804,0.2794,0.2786,0.278,0.2775]}],"layout":{"title":{"text":"spike 28%   |   broad 72%   |   H = 0.55"}}},{"name":"1.00","data":[{"y":[0.3395,0.3395,0.3395,0.3395,0.3395,0.3395,0.3395,0.3395,0.3395,0.3395,0.3395,0.3395,0.3395,0.3395,0.3395,0.3395,0.3396,0.3403,0.3436,0.3561,0.3939,0.4864,0.6675,0.9112,1.0407,0.9113,0.6676,0.4866,0.3941,0.3563,0.3439,0.3407,0.3401,0.3402,0.3404,0.3406,0.341,0.3414,0.342,0.3426,0.3435,0.3445,0.3458,0.3473,0.3492,0.3515,0.3542,0.3575,0.3613,0.3658,0.3711,0.3773,0.3844,0.3926,0.402,0.4128,0.425,0.4387,0.4542,0.4714,0.4905,0.5115,0.5345,0.5594,0.5862,0.6147,0.6446,0.6758,0.7077,0.7398,0.7715,0.802,0.8308,0.8568,0.8795,0.898,0.9116,0.9201,0.9229,0.9201,0.9116,0.898,0.8795,0.8568,0.8308,0.802,0.7715,0.7398,0.7077,0.6758,0.6446,0.6147,0.5862,0.5594,0.5345,0.5115,0.4905,0.4714,0.4542,0.4387,0.425,0.4128,0.402,0.3926,0.3844,0.3773,0.3711,0.3658,0.3613,0.3575,0.3542,0.3515,0.3492,0.3473,0.3458,0.3445,0.3435,0.3426,0.342,0.3414,0.341]}],"layout":{"title":{"text":"spike 30%   |   broad 70%   |   H = 0.63"}}},{"name":"1.50","data":[{"y":[0.3914,0.3914,0.3914,0.3914,0.3914,0.3914,0.3914,0.3914,0.3914,0.3914,0.3914,0.3914,0.3914,0.3914,0.3914,0.3915,0.3915,0.3921,0.3946,0.4041,0.4322,0.4975,0.6143,0.756,0.826,0.756,0.6144,0.4976,0.4323,0.4043,0.3948,0.3923,0.3919,0.392,0.3921,0.3923,0.3926,0.3929,0.3933,0.3938,0.3945,0.3953,0.3963,0.3974,0.3989,0.4006,0.4027,0.4051,0.408,0.4114,0.4154,0.4199,0.4252,0.4313,0.4381,0.4459,0.4546,0.4644,0.4752,0.4872,0.5002,0.5144,0.5297,0.5461,0.5634,0.5815,0.6002,0.6194,0.6387,0.6579,0.6766,0.6943,0.7108,0.7256,0.7383,0.7486,0.7562,0.7609,0.7624,0.7609,0.7562,0.7486,0.7383,0.7256,0.7108,0.6943,0.6766,0.6579,0.6387,0.6194,0.6002,0.5815,0.5634,0.5461,0.5297,0.5144,0.5002,0.4872,0.4752,0.4644,0.4546,0.4459,0.4381,0.4313,0.4252,0.4199,0.4154,0.4114,0.408,0.4051,0.4027,0.4006,0.3989,0.3974,0.3963,0.3953,0.3945,0.3938,0.3933,0.3929,0.3926]}],"layout":{"title":{"text":"spike 33%   |   broad 67%   |   H = 0.67"}}}]}
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
