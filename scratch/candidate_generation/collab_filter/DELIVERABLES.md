# Project 2: Collaborative-filtering candidate retrieval — Deliverables

Fill this in as you go through `README.md`'s Procedure, not after finishing
it. Write a prediction before you run the relevant script, and your answer
right after — that gap is the point. Don't check this file's git history
for a given item until you've written your own answer to it.

## Pre-registered predictions (before touching any code)

- Which of the four models do you expect to win at the default settings, and why:
- Do you expect that ranking to survive under realistic sparsity (Part f)? Why or why not:

## Questions

### 1. Do all four models actually beat `popularity`? Is beating it enough to conclude a model "learned taste"?
**Prediction:**

**Answer (Part a/b):**

### 2. Which model's ranking correlates most with raw popularity, and is that a red flag or an expected byproduct of exposure-biased training data?
**Prediction:**

**Answer (Part c):**

### 3. Does confidence-weighting (graded save=2/click=1 vs. binary) change anything — for each model individually, and why or why not?
**Prediction:**

**Answer (Part d):**

### 4. A model's score can climb while it's actually collapsing onto a popularity ranking. What in the results would expose that, and what would it look like if you only checked the headline metric?
**Prediction:**

**Answer (Part c):**

### 5. Which model needs the least compute, and does the cheapest model also win?
**Prediction:**

**Answer (Part b):**

### 6. Does the winner at this project's default (small, dense) scale stay the winner once the log is realistically sparse?
**Prediction:**

**Answer (Part f):**

### 7. Cross-checking ALS/BPR against a reference library (`implicit`) either confirms your implementation or catches a bug. What would each outcome look like, concretely, in the numbers?
**Prediction:**

**Answer (Part e):**

## Deliverables

### 1. Your own answer to each of the seven questions above
See Questions section — filled in above.

### 2. Full-grid table from Part b: best config per model, against `random`/`popularity`

| model | best config | NDCG@5 | NDCG@10 | hit@10 | recall@10 | pop_rho |
|---|---|---|---|---|---|---|
| random | — | | | | | — |
| popularity | — | | | | | — |
| item_knn | | | | | | |
| svd | | | | | | |
| als | | | | | | |
| bpr | | | | | | |

### 3. The Part c config that "wins" on NDCG while `pop_rho` gives away it's collapsed toward popularity

**Config found:**

**Why that combination is misleading if read from NDCG alone:**

### 4. Part d comparison (binary vs. confidence-weighted), and an explanation for BPR's result specifically

| model | NDCG@10 (binary) | NDCG@10 (confidence) | moved? |
|---|---|---|---|
| item_knn | | | |
| svd | | | |
| als | | | |
| bpr | | | |

**BPR explanation:**

### 5. What Part e's cross-check told you — agreement, or a bug worth fixing

### 6. Dense-vs-sparse comparison table (Part f), and whether your Part b winner survived

| variant | density% | cold% | model | recall@10 | ndcg@10 | hit_rate@10 | pop_rho |
|---|---|---|---|---|---|---|---|
| dense (560) | | | item_knn | | | | |
| dense (560) | | | svd | | | | |
| dense (560) | | | als | | | | |
| dense (560) | | | bpr | | | | |
| sparse (...) | | | item_knn | | | | |
| sparse (...) | | | svd | | | | |
| sparse (...) | | | als | | | | |
| sparse (...) | | | bpr | | | | |

**Did your Part b winner survive?**

### 7. One paragraph: which model would you reach for first on a real, sparse, large-catalog dataset, and why that might differ from your Part b answer

## Status

- [ ] Pre-registered predictions written, before running anything
- [ ] All 7 questions answered
- [ ] All 7 deliverables filled in
- [ ] Checked against git history (only after finishing each item above)
