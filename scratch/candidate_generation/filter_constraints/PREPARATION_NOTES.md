# Preparation notes

Background concepts for `README.md`'s Preparation checklist — not the lab's
findings, so there's nothing to spoil by reading this first. View this in a Markdown renderer that supports LaTeX (GitHub, VS Code preview) — the equations below are raw `$...$` / `$$...$$` math and will look like noise in a plain-text viewer.

| # | Concept |
|---|---|
| 1 | [Predicate satisfaction and violation count](#1-predicate-satisfaction-and-violation-count) |
| 2 | [Hard-fail vs. soft-penalty rescoring](#2-hard-fail-vs-soft-penalty-rescoring) |
| 3 | [The structural floor on soft-penalty](#3-the-structural-floor-on-soft-penalty) |
| 4 | [Retention](#4-retention) |
| 5 | [Why retrieve-then-filter is capacity-limited](#5-why-retrieve-then-filter-is-capacity-limited) |
| 6 | [The predicate schema](#6-the-predicate-schema) |

---

## 1. Predicate satisfaction and violation count

**Plain English:** each hard constraint a query states (e.g. "no nightlife")
becomes one predicate. A candidate either satisfies it or doesn't — no
partial credit. Violations just count how many of a query's predicates a
candidate fails.

**Equation:**

$$\text{violations}(c, P) = \sum_{p \,\in\, P} \big[\, \lnot\, p.\text{satisfied}(c) \,\big]$$

$[\cdot]$ is the Iverson bracket — 1 if the condition inside is true, 0
otherwise. A candidate is fully compliant exactly when this sum is 0:

$$\text{satisfies\_all}(c, P) \iff \text{violations}(c, P) = 0$$

**In the code:** `constraints.violations()` and `constraints.satisfies_all()`
implement these two directly, over whatever tuple of predicates a query maps to.

---

## 2. Hard-fail vs. soft-penalty rescoring

**Plain English:** hard-fail *removes* violators from the candidate pool.
Soft-penalty *demotes* them — it lowers their similarity score in proportion
to how many constraints they break, then re-sorts. Both start from the same
top-$N$ retrieved candidates; they differ only in what happens next.

**Equations:**

$$\text{hard\_fail}(C, P) = \big[\, c \in C : \text{violations}(c,P) = 0 \,\big]_{:k}$$

$$\text{soft\_penalty}(C, P, \rho) = \text{sort}_{\downarrow}\Big(\, c \mapsto \text{score}(c) - \rho \cdot \text{violations}(c, P) \;\; \forall\, c \in C \,\Big)_{:k}$$

$\rho$ (`penalty` in the code) is the per-violation demotion strength; $C$ is
the top-$N$ retrieved candidate set (not the whole catalog) — see §5 for why
that restriction matters.

---

## 3. The structural floor on soft-penalty

**Plain English:** demoting violators can push them to the bottom of the
*pool you already have* — it can never make the pool bigger. If fewer than
$k$ candidates in that pool actually satisfy every constraint, soft-penalty
runs out of compliant items to promote, no matter how harsh the penalty.

**Equation:** let $n_{\text{valid}}(P) = |\{c \in C : \text{satisfies\_all}(c, P)\}|$
— the count of fully-compliant candidates inside the retrieved pool $C$. As
$\rho \to \infty$, every violator sorts below every compliant candidate, but
the top-$k$ slots still must be filled from a pool of size $|C|$:

$$\lim_{\rho \to \infty} \text{violation\_rate} = \frac{\max(0,\; k - n_{\text{valid}}(P))}{k}$$

If $n_{\text{valid}}(P) < k$, this floor is strictly positive — **no penalty
strength reaches zero violations.** Only removing items (hard-fail) can,
because hard-fail is allowed to return fewer than $k$ results; soft-penalty
by construction always returns exactly $k$.

---

## 4. Retention

**Plain English:** how much of the *unfiltered* top-$k$ survived filtering —
a measure of how much filtering reshuffled the ranking, not how much
relevance it destroyed. Those are easy to conflate, and the difference
matters when reading a low retention number.

**Equation:**

$$\text{retention} = \frac{\big|\,\text{filtered top-}k \,\cap\, \text{unfiltered top-}k\,\big|}{\min(k, |\text{unfiltered top-}k|)}$$

**Why low retention isn't automatically bad:** if the unfiltered baseline is
mostly violators (see §1), a filter is *supposed* to displace most of it —
that's the entire job. Low retention there means the filter is working, not
that it's destroying value. Read it next to the violation rate it started
from, never in isolation.

---

## 5. Why retrieve-then-filter is capacity-limited

**Plain English:** a filter applied after retrieval can only rearrange or
remove what retrieval already handed it. If a genuinely compliant city
exists in the catalog but ranked, say, 340th on similarity, and only the
top-100 candidates ever reach the filter, that city is invisible to it —
no filter setting, hard or soft, can recover it.

**Equation:** $n_{\text{valid}}(P)$ from §3 is bounded by the retrieval cutoff $N$,
not by the catalog size $|\mathcal{D}|$:

$$n_{\text{valid}}(P) \;=\; \big|\{c \in \text{top-}N(\mathcal{D}) : \text{satisfies\_all}(c, P)\}\big| \;\leq\; N \;\ll\; |\mathcal{D}|$$

Whenever the true number of catalog-wide compliant items exceeds
$n_{\text{valid}}(P)$, retrieval already threw some of them away before the
filter ever ran. **This is the argument for pushing hard predicates into
retrieval itself** (a pre-filtered ANN search, or a metadata index queried
first) rather than applying them as a post-pass — see `README.md`'s
Deliverables.

---

## 6. The predicate schema

Same schema `content_filter`'s `PREPARATION_NOTES.md` §6 documents —
`TagAtLeast`, `TagAtMost` (negation), `BudgetAtMost`, `MonthTempAtLeast`,
`MonthTempAtMost` — this project is where those predicates and
`LABELLED_QUERIES` are actually defined (`constraints.py`); `content_filter`
borrows them as retrieval ground truth.
