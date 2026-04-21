# Algorithm Overview

All examples on this page use the same small problem from the [Quick Start](index.md):
- 2 wind turbines, demand = 2 MW
- `gas_costs = [0.4]`, `wind_costs = [0.05, 0.10]`, `recourse_cost = 1.0`
- 4 equally-likely wind scenarios: `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)` each with probability 0.25

---

## Two-Stage Stochastic Optimization

The problem has two decisions made at different times:

**First stage — before wind is known ($x$)**

Commit the gas generator for $x$ MW at cost $c_x \cdot x$.
With `gas_costs = [0.4]` and `demand = 2`:

| $x$ | Gas committed | Gas cost |
|-----|--------------|----------|
| 0   | 0 MW         | \$0.00   |
| 1   | 1 MW         | \$0.40   |
| 2   | 2 MW         | \$0.80   |

**Second stage — after wind is revealed ($y$, $r$)**

Given $x$ and the actual wind scenario $\xi$, decide which turbines to dispatch ($y_j \in \{0,1\}$) and how much recourse $r$ (expensive backup power) to buy to cover the shortfall.

With $x = 0$ (no gas), the wind must cover all 2 MW. For each scenario:

| Scenario $\xi$ | Best dispatch $y^*$ | Wind output | Shortfall $r$ | Second-stage cost $q$ |
|---------------|---------------------|-------------|---------------|-----------------------|
| `(0,0)` — no wind | `[0,0]` (both off) | 0 MW | 2 MW | $1.0 \times 2 = \$2.00$ |
| `(0,1)` — turbine 1 only | `[0,1]` | 1 MW | 1 MW | $0.10 + 1.0 \times 1 = \$1.10$ |
| `(1,0)` — turbine 0 only | `[1,0]` | 1 MW | 1 MW | $0.05 + 1.0 \times 1 = \$1.05$ |
| `(1,1)` — both turbines | `[1,1]` | 2 MW | 0 MW | $0.05 + 0.10 = \$0.15$ |

**Objective**: minimize total cost = gas cost + expected second-stage cost

$$\min_x \left[ c_x \cdot x + \underbrace{\mathbb{E}_\xi\!\left[\min_y\, q(x,y,\xi)\right]}_{\phi(x)} \right]$$

For $x = 0$: $\phi(0) = 0.25 \times (2.00 + 1.10 + 1.05 + 0.15) = 0.25 \times 4.30 = \$1.075$

Total cost for $x=0$: $0 + 1.075 = \$1.075$

**Now compare $x=1$** (gas covers 1 MW, wind must cover the remaining 1 MW).
With `wind_demand=1`, only one turbine is dispatched at a time (`sum(y)=1`). The optimizer picks whichever turbine has wind and is cheaper:

| Scenario $\xi$ | Best dispatch $y^*$ | Wind output | Shortfall $r$ | Second-stage cost $q$ |
|---------------|---------------------|-------------|---------------|-----------------------|
| `(0,0)` — no wind | `[1,0]` (either works, same cost) | 0 MW | 1 MW | $0.05 \times 0 + 1.0 \times 1 = \$1.00$ |
| `(0,1)` — turbine 1 only | `[0,1]` (use turbine 1) | 1 MW | 0 MW | $0.10 \times 1 = \$0.10$ |
| `(1,0)` — turbine 0 only | `[1,0]` (use turbine 0, cheaper!) | 1 MW | 0 MW | $0.05 \times 1 = \$0.05$ |
| `(1,1)` — both turbines | `[1,0]` (turbine 0 cheaper) | 1 MW | 0 MW | $0.05 \times 1 = \$0.05$ |

$\phi(1) = 0.25 \times (1.00 + 0.10 + 0.05 + 0.05) = 0.25 \times 1.20 = \$0.30$

Total cost for $x=1$: $0.40 \times 1 + 0.30 = \$0.70$ ← **cheaper than $x=0$!**

**And $x=2$** (gas covers everything, no wind needed): $\phi(2) = \$0.00$ but gas cost is $0.40 \times 2 = \$0.80$.

**Summary — the optimization result:**

| $x$ (gas MW) | Gas cost | $\phi(x)$ (wind+recourse) | Total cost |
|---|---|---|---|
| 0 | \$0.00 | \$1.075 | \$1.075 |
| **1** | **\$0.40** | **\$0.300** | **\$0.700 ← optimal** |
| 2 | \$0.80 | \$0.000 | \$0.800 |

The surprise: committing *some* gas ($x=1$) and relying on wind for the rest is cheaper than either full wind ($x=0$, too much recourse risk) or full gas ($x=2$, too expensive). This is the core insight the quantum algorithm is designed to find efficiently by computing $\phi(x)$ for each $x$.

The algorithm must compute $\phi(x)$ for each candidate $x$ — this is the hard part, because the $\min$ over $y$ is *inside* the expectation over $\xi$. Every scenario needs its own optimal dispatch, and classical Monte Carlo needs $O(1/\epsilon^2)$ samples to estimate this accurately.

---

## Step 1: DQA — Discrete Quantum Annealing

DQA's job is to prepare a quantum state whose **average energy equals $\phi(x)$**.

**What is a quantum state here?**
The quantum computer holds a superposition over all possible turbine dispatch decisions $y$. Each basis state $|y\rangle|\xi\rangle$ represents one specific dispatch $y$ under one specific wind scenario $\xi$, and has a complex amplitude (probability weight).

**Initialization — Dicke state $|D_n^k\rangle$**

Start with a *uniform* superposition over all feasible dispatches (those satisfying $\sum y_j = k$, where $k = \text{wind\_demand}$). With $k=2$ (both turbines needed), there is only one feasible dispatch: $y = [1,1]$. The wind scenarios are loaded with amplitudes proportional to $\sqrt{\Pr[\xi]}$.

With our uniform distribution, the initial state is:

$$|D_2^2\rangle \otimes |\text{pdf}\rangle = |11\rangle \otimes \frac{1}{2}\!\left(|00\rangle + |01\rangle + |10\rangle + |11\rangle\right)$$

Each of the 4 terms has equal amplitude $\frac{1}{2}$, so each scenario is equally likely.

**Annealing — alternating cost and mixer layers**

The circuit applies $T$ Trotter steps. At step $t$ out of $T$, the schedule parameter is $s_t = t/T$ (goes from 0 to 1):

$$U_{\text{DQA}}(T) = \underbrace{U_q(s_{T-1}) \cdot U_d(1-s_{T-1})}_{\text{last step, mostly cost}} \cdots \underbrace{U_q(s_0) \cdot U_d(1-s_0)}_{\text{first step, mostly mixing}}$$

- **$U_q(\gamma)$ — cost operator**: adds a phase $e^{-i\gamma \cdot q(y,\xi)}$ to each basis state proportional to its cost. States with *lower* cost accumulate *less* phase → they are relatively amplified over many steps.

  *Concretely*: state $|11\rangle|11\rangle$ (both turbines dispatch, both have wind, cost \$0.15) gets a small phase; state $|11\rangle|00\rangle$ (both dispatch, no wind, cost \$2.00) gets a large phase.

- **$U_d(\beta)$ — XY mixer**: swaps amplitude between different dispatch decisions while keeping $\sum y_j = k$ fixed (the demand constraint is never violated). Early in annealing ($s_t \approx 0$), mixing is strong so the state explores broadly. Late in annealing ($s_t \approx 1$), the cost operator dominates and the state concentrates on low-cost solutions.

**Output**

After $T$ steps, the state $|\tilde{\psi}_x^*\rangle$ has most amplitude on low-cost $(y, \xi)$ pairs. Its expected energy is:

$$\langle H_Q \rangle = \phi(x) + \delta$$

where $\delta \geq 0$ is the *residual temperature* — the error from finite annealing depth. With $T=4$ steps, $\delta$ is small but nonzero. With $T \to \infty$, $\delta \to 0$.

---

## Step 2: Oracle — Encoding cost as amplitude

To use QAE, we need to turn the *cost* of each scenario into a *probability* on an extra qubit (the ancilla).

The oracle $\mathcal{F}$ does this by rotating the ancilla qubit by an angle that depends on the cost:

$$\mathcal{F}|y\rangle|\xi\rangle\underbrace{|0\rangle}_\text{ancilla} \;\longrightarrow\; |y\rangle|\xi\rangle\!\left(\sqrt{1-\bar{q}}\,|0\rangle + \sqrt{\bar{q}}\,|1\rangle\right)$$

where $\bar{q} = q(y,\xi) / q_{\max} \in [0,1]$ is the normalized cost. After the oracle:

$$\Pr[\text{ancilla} = |1\rangle] = \bar{q}$$

**Concrete example** with $q_{\max} = 3.0$:

| State $|y\rangle|\xi\rangle$ | Cost $q$ | $\bar{q} = q/3$ | Ancilla rotation angle $2\arcsin(\sqrt{\bar{q}})$ |
|------------------------------|----------|-----------------|---------------------------------------------------|
| $|11\rangle|11\rangle$        | \$0.15   | 0.050           | 0.45 rad |
| $|11\rangle|10\rangle$        | \$1.05   | 0.350           | 1.28 rad |
| $|11\rangle|01\rangle$        | \$1.10   | 0.367           | 1.31 rad |
| $|11\rangle|00\rangle$        | \$2.00   | 0.667           | 1.91 rad |

After $\mathcal{F}$ acts on the full superposition, the *average* probability of ancilla = $|1\rangle$ equals $\mathbb{E}[\bar{q}] = \phi(x)/q_{\max}$ — the normalised expected cost. This is the number QAE will estimate.

**Two oracle variants:**

| Oracle | Method | Gates | When to use |
|--------|--------|-------|-------------|
| $\mathcal{F}_\text{exact}$ | Multi-controlled RY per basis state | $O(2^{2n_y})$ — grows exponentially | Small problems, exact benchmarking |
| $\mathcal{F}_\text{sin}$ | One CCRY pair per turbine | $O(n_y)$ — constant per turbine | Large problems, practical use |

$\mathcal{F}_\text{sin}$ works because the cost is a *sum* over independent turbines — each turbine's contribution can be encoded separately, and the rotations add up.

---

## Step 3: QAE — Quantum Amplitude Estimation

QAE estimates $a = \mathbb{E}[\bar{q}]$ — the normalised expected cost — by exploiting a geometric structure in the quantum state.

**The key insight: the Grover operator $\mathcal{Q}$ is a rotation**

After applying $\mathcal{A} = \mathcal{F} \cdot U_\text{DQA}$, the full state lives in a 2D plane spanned by:
- $|\text{good}\rangle$ = states where ancilla = $|1\rangle$ (high cost, probability $a$)
- $|\text{bad}\rangle$ = states where ancilla = $|0\rangle$ (low cost, probability $1-a$)

The Grover operator $\mathcal{Q} = \mathcal{A} S_0 \mathcal{A}^\dagger S_{\psi_0}$ **rotates this 2D state by exactly $2\theta_a$** per application, where $a = \sin^2(\theta_a)$.

*Analogy*: imagine a clock hand that rotates by a fixed (but unknown) angle each tick. If you watch it for $M$ ticks, you can figure out the angle much more precisely than if you just looked at it once.

**QPE reads off the rotation angle** (Fig. 2 of paper):

```
m ancilla qubits → [ H ] ─ control Q¹ ────────────────── [ IQFT ] → measure b
                   [ H ] ─ ─────────── control Q² ─────── [ IQFT ] → measure b
                   [ H ] ─ ─────────── ─────────── control Q⁴ ─ [ IQFT ] → measure b
                                                                        ↑
system qubits  → [ U_DQA ] → [ Oracle F ] ─────────────────────── (untouched)
```

1. Put $m$ estimate qubits in equal superposition with Hadamard gates
2. Apply $\mathcal{A}$ once to prepare the DQA + oracle state
3. Estimate qubit $j$ controls $2^j$ applications of $\mathcal{Q}$ — this coherently accumulates the rotation angle into the estimate register
4. Apply inverse QFT to decode the accumulated phase into a readable integer $b$
5. Measure b → compute the answer:

$$\tilde{a} = \sin^2\!\left(\frac{b \cdot \pi}{2^m}\right) \qquad \tilde{\phi}(x) = \tilde{a} \cdot q_{\max}$$

**Concrete example** with $m = 3$ estimate qubits ($2^3 = 8$ possible values of $b$):

Suppose the true $a = 0.358$ (so $\phi(0) = 0.358 \times 3.0 = \$1.075$). Then $\theta_a = \arcsin(\sqrt{0.358}) = 0.641$ rad.

QPE finds $b = 2$ (since $b \cdot \pi / 8 \approx 0.785$ rad is the closest grid point to $\theta_a$):

$$\tilde{a} = \sin^2(2\pi/8) = \sin^2(0.785) = 0.500 \quad \Rightarrow \quad \tilde{\phi}(0) = 0.5 \times 3.0 = \$1.50$$

With only $m=3$ qubits the estimate is rough. With $m=5$: $b=7$, $\tilde{a} = \sin^2(7\pi/32) = 0.364$, $\tilde{\phi}(0) = \$1.09$ — much closer.

**Why QAE beats Monte Carlo:**

| Method | Evaluations $M$ | Error |
|--------|-----------------|-------|
| Monte Carlo (classical) | $M$ samples | $\sim 1/\sqrt{M}$ |
| QAE (quantum) | $M = 2^m$ | $\sim 1/M$ |

To get error $\epsilon = 0.01$: Monte Carlo needs $M = 10{,}000$ samples; QAE needs only $M \approx 100$ circuit evaluations.

---

## Convergence

Two independent sources of error, each controlled by one parameter:

| Source | Parameter | Effect on error | How to reduce |
|--------|-----------|-----------------|---------------|
| DQA residual $\delta$ | Trotter steps $T$ | $\delta \to 0$ as $T \to \infty$ | Increase `time_steps` |
| QAE resolution | QPE qubits $m$ | Error $\sim 1/2^m$ | Increase `m` in `canonical_qae()` |

In practice, $T=4$–$10$ steps and $m=3$–$5$ qubits give good results for small problems.
See [Fig. 4 in arXiv:2402.15029](https://arxiv.org/abs/2402.15029) for empirical convergence curves.
