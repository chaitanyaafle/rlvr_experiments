# Research Plan: Atomic Skill Decomposition for RLVR

**Working Title:** *Does Atomic Skill Pre-Training Enable Compositional Transfer? A 20-Task Study of Small LMs under RLVR*

**Target venues:** ICLR 2027, NeurIPS 2026 (RL track or LLM track)

**Timeline:** 4 weeks (2026-02-22 to 2026-03-22) → submission-ready draft

---

## 1. The Problem

When RLVR (GRPO with verifiable rewards) fails on a complex task for a small model, *why* does it fail? Two competing explanations:

- **Reasoning failure**: The model understands the task but can't plan well enough.
- **Grounding failure**: The model can't reliably parse the task representation — it misreads coordinates, misidentifies symbols, confuses relational structure — so reasoning never gets a clean substrate.

For small models (≤2B), we suspect grounding failure is underappreciated. If a model can't reliably answer "what is at position (row=3, col=5)?" it will fail at maze pathfinding for the wrong reason — no amount of reasoning-focused RLVR will fix it.

**This paper's idea:** Decompose any RLVR task into its *atomic sub-skills* — the minimal verifiable operations a model must perform correctly to have any chance at the full task. Train RLVR on these atoms first. Then measure how much this transfers to the full task, including zero-shot, across **20 tasks from 5 distinct categories**. We also ask: do shared atoms across tasks produce shared competence? This gives us an empirically derived **cross-task atomic transfer matrix** — a principled map of how reasoning skills compose and generalize.

---

## 2. What Are Atomic Sub-Skills?

Atomic sub-skills are the smallest independently verifiable questions a model must answer correctly to solve the full task. They are *not* simpler versions of the same task — they are the building blocks *underneath* the task.

### General Taxonomy (3 Layers, universally applicable)

- **Layer 1 — Representational (Grounding)**: Parse the input representation. What is where? What do symbols mean?
- **Layer 2 — Local Reasoning**: Single-step inference over grounded elements. No multi-step planning.
- **Layer 3 — Global Planning**: Multi-step strategy composition. This is what the full task requires.

**The hypothesis**: Full-task RLVR on a small model requires simultaneously discovering L1, L2, and L3. This is too much for a 0.5B model under sparse reward. Explicit L1+L2 training removes that barrier, allowing RLVR to focus on L3 alone.

### Per-Domain Examples

**Games / Grid (Maze, Mini-Sudoku, Tower of Hanoi)**

| Atomic Skill | Example Question | Layer |
|---|---|---|
| Grid reading | "What character is at row 3, col 5?" | L1 |
| Coordinate parsing | "What row is position (3,5) on?" | L1 |
| Constraint reading | "What values are already placed in row 2?" | L1 |
| Neighbor enumeration | "What are valid non-wall neighbors of (3,5)?" | L2 |
| Move legality | "Is moving East from (3,5) legal?" | L2 |
| Progress judgment | "Is (3,5) closer to goal (7,7) than (2,5)?" | L2 |

**Logic (Knights & Knaves, Syllogisms, Propositional Logic)**

| Atomic Skill | Example Question | Layer |
|---|---|---|
| Statement parsing | "Who is the subject of premise 1?" | L1 |
| Type identification | "Is statement 2 a universal or existential claim?" | L1 |
| Single-hop inference | "If A→B and A is true, what follows?" | L2 |
| Negation | "What is the negation of 'all X are Y'?" | L2 |
| Consistency check | "Are premises 1 and 3 contradictory?" | L2 |

**Algorithmic (Graph Coloring, Word Ladder, Caesar Cipher)**

| Atomic Skill | Example Question | Layer |
|---|---|---|
| Graph reading | "What are the neighbors of node C?" | L1 |
| Symbol mapping | "What does letter A decode to with shift 3?" | L1 |
| Edge validity | "Is there an edge between A and B?" | L2 |
| Local constraint | "Can node D be colored red given neighbors?" | L2 |

**Arithmetic / Graph (Shortest Path, GCD, Prime Factorization)**

| Atomic Skill | Example Question | Layer |
|---|---|---|
| Number extraction | "What is the weight of edge (A,B)?" | L1 |
| Divisibility check | "Is 12 divisible by 4?" | L2 |
| Comparison | "Which of these two paths is shorter?" | L2 |

---

## 3. The 20-Task Benchmark Suite

Tasks span 5 categories from reasoning_gym. Final task list to be confirmed, but drawn from:

| # | Task | Category | Atomic Complexity | Full-Task Difficulty |
|---|---|---|---|---|
| 1 | `maze` | Games | Medium (6 atoms) | High |
| 2 | `mini_sudoku` | Games | High (8 atoms) | High |
| 3 | `tower_of_hanoi` | Games | Medium (5 atoms) | Medium |
| 4 | `n_queens` | Games | Medium (5 atoms) | High |
| 5 | `countdown` | Games | Low (3 atoms) | Medium |
| 6 | `knights_knaves` | Logic | Medium (5 atoms) | High |
| 7 | `syllogisms` | Logic | Low (4 atoms) | Medium |
| 8 | `zebra_puzzles` | Logic | High (7 atoms) | Very High |
| 9 | `propositional_logic` | Logic | Medium (5 atoms) | High |
| 10 | `circuit_logic` | Logic | Medium (4 atoms) | Medium |
| 11 | `graph_color` | Algorithmic | Medium (4 atoms) | High |
| 12 | `word_ladder` | Algorithmic | Low (3 atoms) | Medium |
| 13 | `caesar_cipher` | Algorithmic | Low (3 atoms) | Low |
| 14 | `number_sorting` | Algorithmic | Low (2 atoms) | Low |
| 15 | `cryptarithm` | Algorithmic | Medium (5 atoms) | High |
| 16 | `shortest_path` | Graphs | Medium (4 atoms) | Medium |
| 17 | `family_relationships` | Graphs | Medium (5 atoms) | High |
| 18 | `basic_arithmetic` | Arithmetic | Low (3 atoms) | Low |
| 19 | `prime_factorization` | Arithmetic | Medium (4 atoms) | Medium |
| 20 | `gcd` | Arithmetic | Low (3 atoms) | Low |

> **User note:** Supply the exact 20 tasks when confirmed. Tasks above are candidates. Prefer a spread of difficulty and atomic complexity for maximal experimental variation.

**Key property of this suite:** Tasks within a category share atoms (grid reading across maze/sudoku/n_queens; single-hop inference across syllogisms/knights_knaves/propositional_logic). Tasks across categories share some atoms (comparison, extraction) but differ on domain-specific ones. This structure enables the transfer matrix experiment.

---

## 4. Core Research Questions

**Q1 (Transfer):** Does RLVR training on atomic sub-skills alone improve performance on the full task, even without any direct training on that task? Does this hold across all 20 tasks?

**Q2 (Cross-Task Atomic Sharing):** When two tasks share atomic skills, does atomic training on Task A transfer to Task B's full-task performance? This yields the **cross-task transfer matrix**.

**Q3 (Necessity):** Which atomic skills are *necessary* for each full task? Can we identify a minimum sufficient set by ablation? Is this set consistent across tasks in the same category?

**Q4 (Diagnostic):** Can pre-training atomic skill accuracy predict RLVR failure modes? Is atomic probing a useful and cheap diagnostic before expensive RLVR runs?

**Q5 (Grounding vs. Reasoning):** Is the bottleneck for small models primarily in L1 (grounding) atoms or L2 (local reasoning) atoms, or does this vary by task category?

**Q6 (Weight Locality):** Are atomic skills encoded in identifiable, localized subsets of the model's weights/attention heads? Does atomic training modify the same weight regions as full-task RLVR?

---

## 5. Metrics and Theoretical Contributions

This section defines formal metrics that go beyond accuracy curves. These are the theoretical contributions that make the paper more than a benchmark paper.

### 5.1 Atomic Transfer Coefficient (ATC)

For a given task T and atom set A:

```
ATC(A, T) = [Acc(model trained on A, eval on T) - Acc(base model, eval on T)]
           / [Acc(oracle model with perfect A, eval on T) - Acc(base model, eval on T)]
```

- Numerator: how much atomic training actually helped
- Denominator: how much it *could* have helped if atoms were perfectly mastered
- Range: [0, 1] if no negative transfer; negative values indicate interference

**Why this matters:** ATC normalizes for task difficulty, making results comparable across all 20 tasks. A table of ATCs across the 20-task suite is a core result.

### 5.2 Atomic Necessity Index (ANI)

For each atomic skill a_i within task T's atom set:

```
ANI(a_i, T) = Acc(full atom set A, eval on T) - Acc(A \ {a_i}, eval on T)
```

High ANI → atom is necessary. Near-zero ANI → atom is redundant (pre-training already covers it or it's incidentally learned).

**Expected finding:** For grid-based tasks, L1 atoms (grid reading, coordinate parsing) will have the highest ANI. For logic tasks, single-hop inference will dominate.

**This gives us the Minimum Sufficient Atomic Set (MSAS)**: the smallest A' ⊆ A such that ATC(A', T) ≈ ATC(A, T). A goal: show MSAS is typically 2-3 atoms, not all atoms.

### 5.3 Cross-Task Atomic Compatibility Score (CTACS)

For any two tasks T_i, T_j in the benchmark:

```
CTACS(T_i → T_j) = ATC(atoms(T_i) ∩ atoms(T_j), T_j)
                  / ATC(atoms(T_j), T_j)
```

This measures: how much of task T_j's atomic benefit can be achieved by training on task T_i's atoms alone?

- CTACS close to 1.0: tasks are atomically compatible (shared atoms are sufficient)
- CTACS near 0: tasks are atomically disjoint (no shared substrate)

The 20×20 CTACS matrix is a **core deliverable** of the paper. It reveals the latent atomic structure of the task space.

### 5.4 Atomic Readiness Score (ARS)

Before any training:

```
ARS(model, T) = mean over a_i in atoms(T) of [Acc(model, eval on a_i)]
```

This is a cheap pre-training diagnostic: probe the base model on atoms for <1 hour before any RLVR. **Hypothesis:** ARS is predictive of final full-task RLVR accuracy (measured as Spearman correlation across the 20-task suite).

If the correlation is high (ρ > 0.7), the ARS is a practical tool: measure it before training to decide whether atomic pre-training is needed.

### 5.5 Layer Locality Score (LLS) — Weight Distribution Analysis

For each atomic skill a_i, identify which layers are most responsible for encoding it using linear probing:

```
For each layer l = 1,...,L:
    Train a linear probe on layer-l hidden states → predicts a_i answer
    Record probe accuracy: P(l, a_i)

LLS(a_i) = argmax_l P(l, a_i)   [dominant layer]
LLS_spread(a_i) = entropy over P(1:L, a_i)  [how distributed the encoding is]
```

**Key question:** Are L1 atoms encoded in early layers and L2 atoms in later layers, consistent with the hierarchical representation hypothesis? This would be a mechanistic finding supporting the 3-layer taxonomy.

**Implementation:** Use `logit lens` or simple linear probes on each layer's final hidden state, evaluated on atomic QA pairs for each task.

### 5.6 Gradient Flow Analysis (GFA) — Weight Modification Patterns

During RLVR training, track which weights are most modified:

```
For each parameter θ_k:
    GFA_atomic(θ_k) = mean ||∇θ_k L|| during atomic RLVR training
    GFA_full(θ_k) = mean ||∇θ_k L|| during full-task RLVR training

Overlap(atomic, full) = cosine similarity between GFA_atomic and GFA_full vectors
```

**Hypothesis:** High overlap (>0.7) between atomic and full-task gradient flows would suggest that atomic training is preparing the *same* weights that full-task RLVR would update anyway — just more reliably. Low overlap would suggest atomic training creates a different internal representation entirely.

**This is the theoretical "why does it work" section** — connecting behavioral transfer to weight-space geometry.

### 5.7 Atomic Forgetting Rate (AFR)

After full-task RLVR following atomic pre-training:

```
AFR(a_i, T) = Acc(atomic-trained model, eval on a_i)
            - Acc(atomic-warm model after full-task RLVR, eval on a_i)
```

High AFR → catastrophic forgetting of atoms during full-task RL.
Near-zero AFR → atoms are preserved as stable representations.

Across 20 tasks, this gives us an AFR distribution. If AFR correlates with ATC drop (tasks where forgetting is high are also tasks where Atomic-Warm underperforms Atomic-Only), this is mechanistic evidence that forgetting is the bottleneck.

---

## 6. Experimental Conditions

The core experiment runs 5 training conditions on every task in the 20-task suite:

| Condition | Training | Eval |
|---|---|---|
| **Base** | No training | Full task |
| **Atomic-Only** | GRPO on L1+L2 atoms only | Full task (zero-shot transfer) |
| **Atomic-Warm** | GRPO on atoms → GRPO on full task | Full task |
| **Direct-RLVR** | GRPO on full task only | Full task |
| **L1-Only** | GRPO on L1 (grounding) atoms only | Full task |

All conditions use matched compute (same total GRPO steps). This produces:
- Full-task accuracy for each condition and task → primary results table
- ATC, ANI, ARS, AFR for each task
- Cross-task CTACS matrix (requires training atomic sets from one task, evaluating on another)

---

## 7. 4-Week Implementation Plan

### Week 1: Infrastructure + Scoping (Feb 22 – Mar 1)

**Goal:** Training pipeline ready, evaluation pipeline ready, all 20 tasks catalogued with atomic decompositions.

- [ ] Confirm the final 20 tasks; for each, define L1 atoms (2-4) and L2 atoms (2-4) as Q&A templates with programmatically verifiable ground truth
- [ ] Build the Atomic Dependency DAG for each task (edges = prerequisite ordering between atoms); define Compositional Depth as the longest DAG path
- [ ] Implement a generic `AtomicEnv` wrapper that takes any reasoning_gym dataset + an atomic Q&A template and produces a `BaseEnvironment`-compatible class; implement atomic envs for all 20 tasks using the wrapper
- [ ] Extend `train_grpo.py` with `--atomic_mode`: supports multi-env joint training via cycled or weighted sampling; saves ARS probe results at each checkpoint
- [ ] Implement evaluation scripts: `evaluate_all.py` (full-task eval across all 20 tasks), `probe_atomics.py` (L1/L2 zero-shot probing), `layer_probe.py` (per-layer linear probes for LLS), `gradient_tracker.py` (per-parameter gradient norm logging for GFA)
- [ ] If the base model fails to produce structured `<think>...<answer>` output, run a short SFT warm-up (1 epoch, ~500 examples) using `train_sft.py` before any RLVR

**Deliverable:** All 20 atomic envs producing clean binary rewards. Evaluation pipeline runs end-to-end. Atomic DAGs documented.

---

### Week 2: Baselines + Transfer Matrix (Mar 1 – Mar 8)

**Goal:** Establish all baselines; compute the cross-task atomic transfer matrix.

- [ ] Run `probe_atomics.py` on Qwen2-0.5B (and Qwen2-1.5B) with no training → compute ARS for all 20 tasks and baseline LLS profiles; identify atoms already solved (>80%) that can be excluded from atomic training
- [ ] Run Direct-RLVR (GRPO on full task, 300–500 steps) for all 20 tasks; log accuracy curves and gradient norms; probe atomics on post-RLVR checkpoints to measure which atoms Direct-RLVR incidentally teaches
- [ ] For task pairs sharing atoms (e.g., grid tasks, logic tasks): train atoms on Task A, zero-shot eval on Task B; compute CTACS for all relevant pairs and produce the preliminary 20×20 CTACS heatmap

**Deliverable:** Full baselines table. ARS table. Direct-RLVR accuracy across all 20 tasks. Preliminary transfer matrix.

---

### Week 3: Atomic Training + Transfer Experiments (Mar 8 – Mar 15)

**Goal:** Train all atomic conditions; measure transfer; collect weight analysis data.

- [ ] **Atomic-Only:** GRPO on L1+L2 atoms jointly (matched compute to Direct-RLVR); confirm each atom reaches >80% accuracy; run zero-shot full-task eval; compute ATC per task
- [ ] **L1-Only:** GRPO on grounding atoms only; zero-shot full-task eval; compare to Atomic-Only to isolate the grounding vs. local-reasoning contribution (Q5)
- [ ] **Atomic-Warm:** Initialize from Atomic-Only checkpoint, continue GRPO on full task; measure how many steps to match Direct-RLVR final accuracy; compute AFR by re-probing atoms after full-task fine-tuning
- [ ] **Weight analysis** on a 5–6 task subset: compute GFA Overlap (cosine similarity of gradient norm vectors between atomic and full-task training); run `layer_probe.py` on Atomic-Only and Atomic-Warm checkpoints; identify attention heads most activated during correct atomic responses

**Deliverable:** Full results for all 5 conditions across 20 tasks. ATC table. AFR analysis. GFA overlap and LLS profiles.

---

### Week 4: Analysis, Ablations, and Writing (Mar 15 – Mar 22)

**Goal:** Ablations, cross-task analysis, paper draft.

- [ ] **MSAS ablation:** For the 5–6 hardest tasks, ablate one atom at a time; compute ANI per atom; identify the Minimum Sufficient Atomic Set (target: 2–3 atoms account for >90% of ATC); check whether MSAS is consistent within a task category
- [ ] **Cross-category analysis:** Cluster the 20 tasks by their ATC vectors (t-SNE/PCA); check whether clustering recovers the original 5 categories; compute Spearman ρ between ARS and Direct-RLVR accuracy, and between task difficulty and ATC
- [ ] **Forgetting mitigation:** For the 3 tasks with highest AFR, test interleaved atomic + full-task training (1:1 batch alternation); compare AFR and final accuracy to sequential Atomic-Warm
- [ ] **Paper draft:** Introduction, method (3-layer taxonomy + formal metrics), experiments (baselines → atomic transfer → MSAS ablation → weight analysis), figures (CTACS heatmap, ATC bar chart, LLS layer profiles, GFA scatter), discussion

**Deliverable:** Complete paper draft. All figures and results tables finalized.

---

## 8. Theoretical Contributions Summary

This work makes the following formal/theoretical contributions (beyond empirical results):

### T1: Atomic Dependency DAG (ADD)
A principled, programmatically-derived directed acyclic graph for any reasoning_gym task, where nodes are atomic skills and edges are prerequisite dependencies. We define the **Compositional Depth (CD)** of a task as the longest path in its ADD. Tasks with higher CD are hypothesized to benefit more from atomic pre-training.

### T2: Atomic Transfer Bounds
**Theorem sketch (empirical):** For tasks with shared atoms, CTACS(T_i → T_j) is lower-bounded by a function of (|atoms(T_i) ∩ atoms(T_j)| / |atoms(T_j)|). The bound is tight when atoms are truly independent (which we test empirically). This is an *empirical bound*, validated across the 20-task suite.

### T3: Layer Locality of Atomic Skills
**Claim:** L1 (grounding) atoms are encoded in earlier transformer layers; L2 (local reasoning) atoms in middle layers; L3 (global planning) in later layers. We test this via linear probing and gradient attribution across all layers. If confirmed, this is a mechanistic result connecting the 3-layer taxonomy to the model's computational hierarchy.

### T4: Weight Overlap Predicts Transfer
**Claim:** GFA Overlap between atomic and full-task training is predictive of ATC. When the same weights are modified by both (high overlap), atomic training is effectively warm-starting the full-task gradient direction. We verify this as a Spearman correlation across tasks.

### T5: Atomic Readiness as a Cheap Predictor
**Claim:** ARS (computed without any training, in <1 hour) is predictive of final Direct-RLVR accuracy (ρ > 0.7 across the 20-task suite). This would mean: before running expensive RLVR, probe your atoms — the result tells you whether atomic pre-training will help and which atoms to prioritize.

---

## 9. Novel Ideas (Extended)

These go beyond the 4-week plan but are in-scope for the paper's appendix or a follow-on.

### Idea A: Inverse Atom Discovery (Failure-Driven)
Run full-task RLVR. Collect failed rollouts. Cluster failure modes by their earliest decision error. The clusters *are* the atoms — discovered from failure distributions rather than human analysis. Auto-generate atomic training tasks from cluster templates. This makes the method self-improving and generalizes to new tasks without manual decomposition.

```
1. Run GRPO(full_task) for N steps
2. Collect failed rollouts; for each, find step k where solution first diverges
3. Cluster step-k contexts → each cluster defines an atomic skill
4. Generate atomic training data from cluster templates
5. Run GRPO(atomic_tasks) for M steps
6. Repeat
```

### Idea B: Atom-Conditioned Scratchpad (Joint Training)
Force the model to explicitly answer atomic questions inside its chain of thought before reasoning:

```
<atoms>
  cell(3,5)=wall | neighbors(3,4)=[(3,3),(4,4)] | blocked_east(3,4)=yes
</atoms>
<think>Since east is blocked, go south...</think>
<answer>South, South, East, East</answer>
```

The `<atoms>` block is verified programmatically (binary reward per fact). The `<answer>` is verified as before. Both flow into GRPO. This makes grounding explicit and auditable — a practical interpretability benefit.

### Idea C: DPO for Atoms, RL for Full Task
For atomic training: use DPO on (correct, wrong) pairs instead of GRPO. Since atomic answers are fully determined, DPO is more stable and sample-efficient. Then use DPO-atoms model as RLVR warm-start. If DPO-atoms + RLVR-full > GRPO-atoms + RLVR-full, this suggests: *use supervised objectives for fully-determined task components, RL only for exploration-requiring components*.

### Idea D: Representation-Agnostic Atomics
Take the same task, render it in 3 formats (ASCII, natural language, JSON). Train atoms on ASCII format, test zero-shot on NL and JSON. If atoms transfer: they are genuine abstractions, not surface pattern matching. If not: atoms are format-locked, motivating format-specific atomic training.

### Idea E: Atom Ordering Within Layers
Train L2 atoms in dependency order (wall detection → neighbor enumeration → movement) vs. reverse order vs. jointly. Does dependency-consistent ordering improve final full-task accuracy? This tests whether the model builds genuinely compositional representations or treats atoms as independent.

---

## 10. What's Novel vs. Prior Work

| Related Work | How We Differ |
|---|---|
| **Curriculum RL** (Bengio 2009, PLR) | They make the *same task* easier progressively. We train on *different tasks* (the atoms) that are categorically distinct from the full task. |
| **Hierarchical RL** (Options, HAL) | HRL learns sub-policies in action space for robotic/game tasks. We work in token space with natural language sub-questions and verifiable answers. |
| **Chain-of-thought / Least-to-most prompting** | Decompose at inference time with no training signal. We train explicitly on atoms with RLVR. |
| **Multi-task learning** | MTL trains on related tasks simultaneously. We train on *prerequisite* tasks sequentially, motivated by cognitive dependency structure. |
| **Probing studies** (BERTology) | Prior probing asks "does the model know X?" passively. We ask "if we *teach* the model X via RL, does it transfer to task Y?" — an active intervention. |
| **DeepSeek-R1 / GRPO** | They scale RLVR to large models. We study *why small models fail* and how atomic decomposition systematically fixes it. |
| **Compositional generalization** (SCAN, COGS) | Those test generalization via held-out compositions in i.i.d. distribution. We test transfer via *held-out tasks* after training on constituent atoms. |

**The core novelty:** The 20-task cross-task atomic transfer matrix, the formal metrics (ATC, ANI, CTACS, ARS, LLS, GFA), and the mechanistic claim connecting atomic skill locality in weight space to behavioral transfer. No prior work has done this.

---

## 11. Hypotheses

**H1 (Zero-shot atomic transfer):** Across the 20-task suite, Atomic-Only achieves mean ATC > 0.3 (i.e., atomic training produces 30% of the possible improvement without any full-task training).

**H2 (Atomic warm-start efficiency):** Atomic-Warm reaches Direct-RLVR's final accuracy in ≤50% of the GRPO steps, across the majority of tasks.

**H3 (Grounding dominates):** L1-Only outperforms L2-Only on full-task zero-shot transfer for grid-based tasks (maze, sudoku, n_queens), but not for logic tasks (where L2 single-hop inference matters more).

**H4 (Small MSAS):** For each task, the MSAS contains at most 2-3 atoms and accounts for >90% of ATC. The remaining atoms are either pre-trained or incidentally learned.

**H5 (ARS predicts performance):** ARS (pre-training probe, no training) correlates with Direct-RLVR final accuracy across the 20 tasks (Spearman ρ > 0.65), making it a practical pre-training diagnostic.

**H6 (Layer locality):** Linear probe accuracy for L1 atoms peaks in layers 1-8, L2 atoms in layers 8-16, for a 28-layer Qwen2-0.5B model — consistent with hierarchical computation.

**H7 (Weight overlap predicts ATC):** GFA Overlap between atomic and full-task training correlates with ATC across tasks (Spearman ρ > 0.6), mechanistically explaining why atomic warm-starting works.

**H8 (Atomic forgetting):** AFR is higher for tasks where Atomic-Warm underperforms the max of Atomic-Only + Direct-RLVR — i.e., forgetting is the active bottleneck in those cases.

---

## 12. Failure Modes + Mitigations

| Risk | Mitigation |
|---|---|
| Base model already solves all atoms (floor effect) | Use harder atomic variants (larger grids, more premises); or use smaller model (Qwen2-0.5B vs 1.5B) |
| Atomic-Only has near-zero transfer (H1 fails) | Still publishable: negative result on atomic transfer is informative; focus shifts to Atomic-Warm + MSAS; Q4 becomes the main contribution |
| Direct-RLVR also accidentally teaches atoms | Measure atomic probes before/after Direct-RLVR — if it already teaches atoms, that itself is a finding about RLVR implicit curriculum |
| Atomic envs trivially solved (<50 GRPO steps) | OK — the point is scaffolding, not difficulty; increase atomic difficulty (harder instances, noisy formatting) if needed |
| 20 tasks too many to train in 4 weeks | Prioritize the 8-10 hardest tasks for all 5 conditions; run only Direct-RLVR + Atomic-Warm for the rest |
| GFA analysis too expensive | Sample 5-6 representative tasks instead of all 20; report as "case study" rather than full analysis |
| CTACS matrix is too sparse (tasks don't share atoms) | This is itself a finding: the task space is atomically disjoint across categories; focus on within-category transfer |

---

## 13. Quick De-risk Checks (Run First in Week 1)

Before committing to the full plan, run these in order within the first 2 days:

1. **Base model atomic probe** (1-2 hours): Zero-shot probe Qwen2-0.5B on 50 examples of each atomic skill for 3 representative tasks (one from each major category). If accuracy >90%, atomics are too easy. Target: 30-70% for interesting range.

2. **Atomic env reward sanity check** (30 min): Run 10 GRPO steps on a single atomic env. Confirm reward is non-degenerate (not all 0 or all 1, distribution is informative).

3. **Direct-RLVR ceiling check** (2-3 hours): Run Direct-RLVR on 3 tasks for 300 steps. If accuracy >70%, tasks are too easy — increase difficulty. Target: Direct-RLVR saturates at 30-60% for most tasks.

4. **Atomic learnability** (1-2 hours): Run GRPO on one L1 atomic env for 200 steps. Confirm accuracy improves above baseline. If not, fix reward function or env format before proceeding.

These 4 checks take <1 day and validate the entire experimental framework before any multi-day training runs.

---

## 14. Compute Budget

| Experiment | Model | Est. GPU-Hours |
|---|---|---|
| Base model atomic probing (all 20 tasks) | 0.5B | ~5h |
| Direct-RLVR baseline (all 20 tasks, 300 steps) | 0.5B | ~25h |
| Atomic-Only training (all 20 tasks) | 0.5B | ~25h |
| L1-Only training (20 tasks) | 0.5B | ~15h |
| Atomic-Warm training (all 20 tasks) | 0.5B | ~25h |
| MSAS ablation (6 tasks × 4 ablations) | 0.5B | ~20h |
| CTACS matrix (selected pairs, ~30 runs) | 0.5B | ~20h |
| Layer probing + GFA analysis | 0.5B | ~10h |
| Atomic forgetting / AFR analysis | 0.5B | ~5h |
| **Total** | | **~150h GPU** |

Target hardware: 2×RTX 3090 (or equivalent). Expected wall-clock: 10-14 days of training, fits within 4-week timeline with analysis and writing in parallel.

Optional: run Qwen2-1.5B on a subset of tasks (5-6) to test whether atomic transfer effects scale with model size. Budget an additional ~40h if this is included.

---

## 15. Paper Outline

**Title:** *Atomic Skill Decomposition for RLVR: A 20-Task Study of Compositional Transfer in Small Language Models*

1. **Introduction** — the grounding failure hypothesis; atomic decomposition as the solution; contributions
2. **Background** — RLVR/GRPO; reasoning_gym; prior work table
3. **Method**
   - 3-layer atomic taxonomy (L1/L2/L3)
   - Atomic Dependency DAG construction
   - Formal metrics: ATC, ANI, CTACS, ARS, LLS, GFA, AFR
   - The 20-task benchmark suite
4. **Experiments**
   - 4.1 Baseline: ARS probing (pre-training diagnostic)
   - 4.2 Core: 5 training conditions across all 20 tasks → ATC table
   - 4.3 Cross-task transfer: CTACS heatmap
   - 4.4 Ablation: MSAS per task → ANI analysis
   - 4.5 Weight analysis: LLS layer profiles + GFA overlap
   - 4.6 Forgetting: AFR analysis + interleaved training
5. **Analysis** — what the transfer matrix reveals about task space structure; which hypotheses were confirmed
6. **Discussion** — practical implications; limitations; future work
7. **Conclusion**

---

*Last updated: 2026-02-22*
