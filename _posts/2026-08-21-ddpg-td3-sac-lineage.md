---
layout: post
title: "From DDPG to SAC: a lineage of fixes"
date: 2026-08-21
description: DDPG, TD3, and SAC are usually presented as three algorithms. They're better understood as one algorithm and two rounds of debugging.
tags: reinforcement-learning sac td3 ddpg actor-critic
categories: reinforcement-learning-for-control
giscus_comments: false
related_posts: false
---

Most tutorials present DDPG, TD3, and SAC as three separate continuous-control algorithms you choose between. That framing hides the useful part. They're better read as **one algorithm and two rounds of debugging** — each successor exists because someone diagnosed a specific, nameable failure in its predecessor and fixed it.

If you understand what broke, you understand why the fixes look the way they do. And you get a much better sense of when you can get away with the simpler thing.

---

## DDPG: the original bet

DDPG (Lillicrap et al., 2015) took DQN's machinery — replay buffer, target networks — and made it work for continuous actions.

The problem with continuous actions is that DQN's `argmax_a Q(s,a)` is intractable when `a` is a vector of real numbers. DDPG's answer: train a **deterministic actor** `μ(s)` to output the action the critic likes best, and update it by pushing its parameters along `∇_a Q(s,a)` — the deterministic policy gradient.

The pieces:

- **Critic** `Q(s,a)` trained on TD targets `y = r + γ Q'(s', μ'(s'))`
- **Actor** `μ(s)` trained to maximize `Q(s, μ(s))`
- **Target networks** `Q'`, `μ'` updated by slow Polyak averaging (`τ ≈ 0.005`)
- **Replay buffer** for off-policy sample reuse
- **Exploration noise** added to actions at collection time, since `μ` is deterministic and would otherwise never explore

It works. It also has a reputation for being miserable to tune, and that reputation is deserved.

---

## What actually goes wrong in DDPG

Two failures, and they compound.

**Overestimation bias.** The TD target contains an implicit maximization: the actor is trained to find high-`Q` actions, so `Q(s, μ(s))` sits near the top of the critic's estimate. Function approximation error is noise on top of the true value — and taking a near-max over a noisy estimate is biased upward. That inflated value goes into the next TD target, the critic learns it, the actor chases it further. The bias doesn't wash out; it accumulates.

You can watch this happen. Log your critic's predicted `Q` against the discounted return actually achieved. In a failing DDPG run, the prediction drifts steadily above the truth and never comes back.

**The actor exploits the critic's errors.** The actor's whole job is to find the argmax of the critic. If the critic has a spurious bump — a region of action space it wrongly thinks is great, usually because it has little data there — the actor will find it and go there. This is adversarial in a way that's easy to miss: the actor is *optimizing against the critic's mistakes*.

Add a deterministic policy whose exploration comes entirely from externally injected noise, and you have something that's very sensitive to the noise schedule, the learning rates, and the random seed.

---

## TD3: three fixes for the bias

TD3 (Fujimoto, van Hoof & Meger, 2018) is DDPG plus three modifications. The paper's title is refreshingly literal — *Addressing Function Approximation Error in Actor-Critic Methods*.

**1. Clipped double Q-learning.** Train two independent critics. Build the TD target from the *smaller* of the two:

```
y = r + γ · min( Q'₁(s', ã), Q'₂(s', ã) )
```

Taking the minimum introduces a downward bias, which is the point — underestimation doesn't compound the way overestimation does. An underestimated action just doesn't get selected; it doesn't poison future targets.

**2. Delayed policy updates.** Update the actor once every `d` critic updates (`d = 2` in the paper). The reasoning: if the actor chases a critic that's still moving, it chases noise. Let the value estimate settle first. This is the same instinct as timescale separation in adaptive control — the inner loop should converge faster than the outer one.

**3. Target policy smoothing.** Add clipped noise to the action used in the TD target:

```
ã = μ'(s') + ε,    ε ~ clip( N(0, σ), −c, c )
```

This enforces a smoothness prior: nearby actions should have similar values. It flattens exactly the narrow spurious peaks the actor would otherwise exploit.

TD3 is substantially more stable than DDPG. But exploration is still extrinsic — you're still hand-tuning an external noise process, and the policy is still deterministic.

---

## SAC: change the objective instead

SAC (Haarnoja et al., 2018) attacks the problem from a different direction. Rather than patching a deterministic policy, it changes what "optimal" means.

The maximum-entropy objective adds a policy-entropy term to the reward:

```
J(π) = Σ_t E[ r(s_t, a_t) + α · H( π(·|s_t) ) ]
```

The agent is now rewarded for **acting as randomly as it can while still doing well**. Three things follow, and they're all consequences of that one change:

**Exploration becomes intrinsic.** No injected noise, no schedule to tune. The policy stays stochastic because staying stochastic is *worth reward*. When a region of state space has multiple near-equally-good actions, SAC keeps its options open there instead of prematurely committing.

**Brittleness drops.** A stochastic policy can't sit on a razor-thin spike in the critic — the entropy term pushes it toward broad, flat optima. Those generalize better and survive small model mismatch, which matters enormously when you eventually deploy on hardware that isn't your simulator.

**The temperature `α` becomes the knob that matters.** Too high and the policy is aimlessly random; too low and you're back to a deterministic policy with all of DDPG's problems. This was SAC's main tuning burden until the 2019 follow-up made `α` automatically adjusted — you specify a target entropy (commonly `−dim(A)`) and `α` is tuned by dual gradient descent to hit it. Use the auto-tuning version. There is no good reason not to.

SAC also keeps TD3's twin critics with the `min`. The bias problem didn't go away; it's orthogonal to the entropy idea.

---

## Reading it as a sequence

| | Exploration | Bias control | Policy |
| --- | --- | --- | --- |
| **DDPG** | injected noise | none | deterministic |
| **TD3** | injected noise | twin critics, delay, smoothing | deterministic |
| **SAC** | intrinsic (entropy) | twin critics | stochastic |

TD3 fixes DDPG's *value estimation*. SAC fixes DDPG's *exploration and brittleness*, and inherits TD3's value fix along the way. They're not competitors so much as two different diagnoses of the same patient, and SAC happens to have taken the other one's medicine too.

---

## A practitioner's note

I used SAC as one of the components in a [hybrid controller for vehicle lateral control](https://arxiv.org/abs/2608.17258) — blending a learned policy with a constrained MPC. A few things I'd tell my earlier self:

**Report multiple seeds or don't report.** Single-seed continuous-control results are close to meaningless. My own numbers are mean ± std over five training seeds, and the spread was large enough that any single run would have told a misleading story. This is the single most common flaw in RL results I read.

**SAC's stability doesn't transfer to your reward function.** SAC is robust to hyperparameters in a way DDPG isn't. It is not robust to a reward that accidentally makes a degenerate behavior optimal. Most of my debugging time went into the reward, not the algorithm.

**Watch the entropy, not just the return.** Entropy collapsing to near-zero early usually means `α` is too low and you've quietly reverted to a deterministic policy. Entropy staying pinned high means the task reward isn't outweighing the entropy bonus. The return curve alone won't tell you which failure you have.

**Understand what "it works" means before deploying.** SAC gave the best tracking error in my comparison. It also gives no guarantee whatsoever about what it will command in a state it hasn't seen. That gap between "best average performance" and "acceptable worst case" is the whole reason for pairing it with a model-based controller — but that's the next post.

---

## References

- Lillicrap et al., *Continuous Control with Deep Reinforcement Learning*, ICLR 2016. [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)
- Fujimoto, van Hoof & Meger, *Addressing Function Approximation Error in Actor-Critic Methods*, ICML 2018. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)
- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor*, ICML 2018. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)
- Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*, 2019 — the automatic temperature tuning. [arXiv:1812.05905](https://arxiv.org/abs/1812.05905)
