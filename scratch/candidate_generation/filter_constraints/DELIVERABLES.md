# Project 3: Hard-constraint filtering — Deliverables

Fill this in as you go through `README.md`'s Procedure, not after finishing
it. Write a prediction before you run the relevant script, and your answer
right after — that gap is the point. Don't check this file's git history
for a given item until you've written your own answer to it.

## Pre-registered predictions (before touching any code)

- What fraction of the unfiltered top-10 do you expect to violate its own query's constraint:
- Do you think soft-penalty can reach zero violations if you push the penalty high enough? Why or why not:

## Questions

### 1. How often does the unfiltered retriever actually violate the constraint its own query stated?
**Prediction:**

**Answer (Part a/b):**

### 2. Does hard-fail reach zero violations? Does soft-penalty, at any strength?
**Prediction:**

**Answer (Part b/d):**

### 3. If soft-penalty can't reach zero, is that a tuning failure or a structural limit — what's the argument either way?
**Prediction:**

**Answer (Part d):**

### 4. What does "retention" actually measure — lost relevance, or something else? What would you wrongly conclude if you read it the other way?
**Prediction:**

**Answer (Part c):**

### 5. For which queries does the pool of constraint-satisfying candidates inside the retrieved top-N run out? What does that imply about filtering after retrieval vs. during it?
**Prediction:**

**Answer (Part e):**

### 6. Which queries have no honest predicate in this schema at all, and what should happen to them instead of forcing a fit?
**Prediction:**

**Answer (Part f):**

## Deliverables

### 1. Your own answer to each of the six questions above
See Questions section — filled in above.

### 2. Full condition table from Part b

| condition | violation rate | retention | mean score | n returned |
|---|---|---|---|---|
| no filter | | | | |
| hard-fail | | | | |
| soft, p=0.01 | | | | |
| soft, p=0.02 | | | | |
| soft, p=0.05 | | | | |
| soft, p=0.1 | | | | |
| soft, p=0.2 | | | | |
| soft, p=0.5 | | | | |
| soft, p=1.0 | | | | |

### 3. One paragraph on Part c: why a low retention number here doesn't mean what it would mean in a different context

### 4. Your Part d/e finding as a general claim: under what condition (feasibility count vs. $k$) can soft-penalty *never* reach zero violations, regardless of tuning?

### 5. One-line recommendation: where should hard constraints actually be enforced in a production version of this funnel, and why

## Status

- [ ] Pre-registered predictions written, before running anything
- [ ] All 6 questions answered
- [ ] All 5 deliverables filled in
- [ ] Checked against git history (only after finishing each item above)
