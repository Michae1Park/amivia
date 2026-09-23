# Preparation notes

Background concepts for `README.md`'s Preparation checklist — not the lab's
findings, so there's nothing to spoil by reading this first. View this in a Markdown renderer that supports LaTeX (GitHub, VS Code preview) — the equations below are raw `$...$` / `$$...$$` math and will look like noise in a plain-text viewer.

| # | Concept |
|---|---|
| 1 | [Implicit vs. explicit feedback, and confidence](#1-implicit-vs-explicit-feedback-and-confidence) |
| 2 | [Graded relevance, and why impressions don't count](#2-graded-relevance-and-why-impressions-dont-count) |
| 3 | [Item-kNN](#3-item-knn) |
| 4 | [SVD](#4-svd) |
| 5 | [ALS](#5-als) |
| 6 | [BPR](#6-bpr) |
| 7 | [NDCG@k](#7-ndcgk) |
| 8 | [Spearman correlation, the popularity diagnostic](#8-spearman-correlation-the-popularity-diagnostic) |
| 9 | [Matrix density / sparsity](#9-matrix-density--sparsity) |

---

## 1. Implicit vs. explicit feedback, and confidence

**Plain English:** a 5-star rating is explicit — the user told you exactly
how they feel. A click is implicit — it tells you *something* was interesting
enough to open, but not how much, and says nothing at all about the items
never shown. ALS handles this by splitting "did we observe anything?" from
"how sure are we it means something?"

**Equations** (Hu, Koren & Volinsky 2008):

$$p_{ui} = \begin{cases} 1 & \text{if user } u \text{ interacted with item } i \\ 0 & \text{otherwise} \end{cases} \qquad C_{ui} = 1 + \alpha \, r_{ui}$$

| Symbol | Meaning |
|---|---|
| $p_{ui}$ | binary preference — did an interaction happen at all |
| $r_{ui}$ | the raw graded signal (this project: save=2, click=1) |
| $\alpha$ | confidence scaling — how much a stronger signal should be trusted over baseline |
| $C_{ui}$ | confidence — always $\geq 1$, even for unobserved pairs |

**Why it matters:** an unobserved item isn't treated as "confidently
disliked" — it's confidence 1 (the floor), preference 0. An observed item
gets confidence $>1$, scaled by how strong the signal was. That's the whole
reason `models.py`'s `ALS` class takes both `alpha` and the graded matrix
rather than just a 0/1 matrix.

---

## 2. Graded relevance, and why impressions don't count

**Plain English:** being *shown* an item isn't a taste signal — it's
exposure. Only acting on it (click, save) tells you anything, and a save
should count for more than a click.

**Convention used throughout this project** (`eval/data.py`'s `GRADE`):

$$\text{grade}(u,i) = \max\big(\text{save}=2,\ \text{click}=1,\ \text{impression-only}=0\big)$$

`max`, not sum — two clicks across separate sessions score identically to
one click, and don't get promoted past a single save. Summing would invent
signal strength that isn't in the log.

**Evaluation split:** a per-user *temporal* holdout — each user's most
recent sessions become the test set, everything earlier is train. This
mimics a real deployment: you can't train on the future.

---

## 3. Item-kNN

**Plain English:** two items are similar if the same users tend to interact
with both. No training loop, no latent space — just a similarity matrix.

**Equation:**

$$\text{sim}(i,j) = \frac{\mathbf{r}_i \cdot \mathbf{r}_j}{\|\mathbf{r}_i\|\,\|\mathbf{r}_j\|} \qquad \text{score}(u,j) = \sum_{i \,\in\, \text{items } u \text{ interacted with}} r_{ui}\cdot \text{sim}(i,j)$$

$\mathbf{r}_i$ is item $i$'s column of interaction values across all users —
cosine similarity between two such columns. A user's score for a
not-yet-seen item $j$ is their interacted items' values, weighted by how
similar each one is to $j$.

**In the code:** `n_neighbors` prunes each item's similarity row to only its
strongest matches before scoring — the long tail of weak similarities is
mostly co-popularity noise, not signal.

---

## 4. SVD

**Plain English:** compress the whole user×item matrix into two much
smaller matrices whose product approximately reconstructs it. The
compression forces it to find the axes ("factors") that explain the most
variance in who-likes-what.

**Equation:**

$$R \approx U_k \, \Sigma_k \, V_k^\top$$

| Symbol | Meaning |
|---|---|
| $R$ | the full (sparse) user×item matrix |
| $U_k$, $V_k$ | the top-$k$ left/right singular vectors — user and item factors |
| $\Sigma_k$ | the top-$k$ singular values, scaling each factor's importance |

User factors = $U_k \Sigma_k$; item factors = $V_k$; a score is their dot
product, same shape as every other model here.

**The honest weakness, stated in `models.py`:** plain SVD treats every
*unobserved* cell as a literal $0$ — mathematically identical to "actively
disliked." It cannot tell "never shown" from "shown and ignored." Contrast
with ALS below, which keeps those separate on purpose.

---

## 5. ALS

**Plain English:** alternate between fixing the item factors and solving
exactly for the best user factors (a per-user least-squares problem), then
flipping — fixing the freshly-solved user factors and solving for the best
item factors. Each half-step has a closed-form solution; only the
alternation itself is iterative.

**Loss being minimized:**

$$\min_{x_*,\,y_*} \; \sum_{u,i} C_{ui}\,(p_{ui} - x_u^\top y_i)^2 \;+\; \lambda\Big(\sum_u \|x_u\|^2 + \sum_i \|y_i\|^2\Big)$$

**Closed-form update for one user** $u$ (item factors $Y$ held fixed):

$$x_u = \big(Y^\top C^u Y + \lambda I\big)^{-1} Y^\top C^u p_u$$

| Symbol | Meaning |
|---|---|
| $x_u$, $y_i$ | the user/item factor vectors being solved for |
| $C^u$ | the diagonal matrix of user $u$'s confidences $C_{ui}$ (see §1) |
| $\lambda$ | regularization strength — this project's `reg` knob |

**Why this is the model that "collapsed onto popularity" at high `reg`:**
push $\lambda$ high enough, and the regularization term dominates the loss —
every user's solved factors get pulled toward the same low-norm point,
which recovers a near-rank-1 solution close to the item's raw popularity.

---

## 6. BPR

**Plain English:** don't reconstruct the matrix at all — directly optimize
*ranking*. For every (user, item-they-liked, item-they-didn't) triple, push
the liked item's score above the other's. Nothing about the reconstruction
error is optimized; only relative order.

**Equation** (Rendle et al. 2009):

$$\max_{\Theta} \sum_{(u,i,j)\,\in\, D_S} \ln\,\sigma(\hat{x}_{uij}) \;-\; \lambda_\Theta \|\Theta\|^2, \qquad \hat{x}_{uij} = \hat{x}_{ui} - \hat{x}_{uj}$$

| Symbol | Meaning |
|---|---|
| $D_S$ | sampled triples: $u$, a positive item $i$ (observed), a negative $j$ (sampled, unobserved) |
| $\hat{x}_{ui}$ | model's predicted score, $= x_u^\top y_i$ — same dot-product shape as ALS/SVD |
| $\sigma$ | the sigmoid function |

**Why the confidence-weighting knob does nothing to BPR:** it samples
triples from the *positions* of observed interactions — which cell has a
value — and never reads the value itself. Binary and graded matrices produce
the identical set of triples, hence a bit-identical trained model.

---

## 7. `NDCG@k`

**Plain English:** like `recall@k`, but a relevant item ranked 1st counts
more than one ranked 10th, and a save (grade 2) counts more than a click
(grade 1) if it's found at all.

**Equation:**

$$\text{DCG@}k = \sum_{r=1}^{k} \frac{\text{grade}_r}{\log_2(r+1)} \qquad \text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

$\text{IDCG@}k$ is the same formula computed on the *ideal* ordering (best
grades first) — dividing by it rescales the score to $[0, 1]$ regardless of
how many relevant items exist for that user.

`recall@k` and `hit_rate@k` — also printed alongside `NDCG@k` in this
project's output — are defined in `content_filter/PREPARATION_NOTES.md` §2,
not repeated here.

---

## 8. Spearman correlation, the popularity diagnostic

**Plain English:** does a model's per-user ranking just look like "most
popular items first," rebranded? Spearman answers by comparing rank
*orders*, not raw scores.

**Equation** (no tied ranks):

$$\rho = 1 - \frac{6\sum_i d_i^2}{n(n^2-1)}$$

$d_i$ is the difference between item $i$'s rank under the model's scores and
its rank under raw popularity counts, over all $n$ items. $\rho \to 1$ means
the two orderings are nearly identical.

**Why it's necessary, not a nice-to-have:** `synthetic_interactions` samples
*impressions* by popularity, so a model can score well on offline metrics by
simply relearning "recommend what's popular." High $\rho$ **and** only a
small edge over the `popularity` baseline together mean a model learned
exposure, not taste — neither number alone tells you that.

---

## 9. Matrix density / sparsity

**Plain English:** what fraction of all possible user-item pairs actually
have an observed interaction.

**Equation:**

$$\text{density} = \frac{\big|\{(u,i) : \text{interaction observed}\}\big|}{|U| \times |I|}$$

MovieLens-25M sits around $0.25\%$. This project's default 560-item catalog
is $2.8\%$ — an order of magnitude denser, with **zero** items below one
interaction. See `README.md`'s Sparsity procedure for a variant closer to
the real number.
