# Amivia Architecture

Amivia is a travel recommendation agent. This document covers:
- The target end-to-end architecture
- The technology considered for each layer
- The toy-project roadmap for learning each piece before assembling the full
  system

The backbone:
- A classic recommender-system funnel — the same pattern used by YouTube,
  TikTok, Instagram, and Amazon
- Works without any LLM at all (a plain search/filter UI over it is a
  complete, real product)
- A conversational LLM layer sits on top as an **optional** wrapper at the
  two ends of the funnel, turning it from a search box into a chat agent

Learning goal ordering: classic backbone first, LLM wrapper second.

## Pipeline overview

```
CORE BACKBONE — classic recsys funnel, no LLM required
─────────────────────────────────────────────────────────────────────────
catalog
  -> (3) candidate retrieval   (multi-source, high recall: embeddings + collaborative
                                 filtering + popularity/rules, merged)
  -> (4) filtering              (hard constraints: budget, dates, negation)
  -> (5a) coarse ranking        (cheap model, narrows hundreds -> tens)
  -> (5b) precise ranking       (expensive model, narrows tens -> final list)
  -> (6) reranking               (diversity, business rules, exploration)
  -> results
  -> (9) feedback loop           (logged interactions feed back into retrieval/ranking)

OPTIONAL LLM WRAPPER — conversational agent layer around the backbone
─────────────────────────────────────────────────────────────────────────
user
  -> (1) conversation layer (LLM, optional)   parses free text into the structured
                                                intent the backbone expects
  -> (2) user understanding                    structured intent + persistent taste
                                                profile — can be filled by an LLM or by
                                                plain forms/filters (classic UI approach)
  -> [ backbone ]
  -> (7) response generation (LLM, optional)  turns the backbone's ranked results into
                                                a conversational answer
  -> (8) agent tooling (LLM, optional)        lets the agent call external tools/APIs
                                                mid-conversation (live prices, weather)
```

Why "user understanding" sits between the two optional LLM steps but isn't
itself marked optional:
- Every recommender needs *some* structured representation of what the user
  wants
- What's optional is *how* that gets filled in — an LLM parsing free text is
  one way, explicit search filters (the classic approach most production
  systems use) is another

**A note on RAG:**
- Retrieval-Augmented Generation isn't a separate technology to add to this
  pipeline — it's the name for candidate retrieval (layer 3) and response
  generation (layer 7) composed together
- Classic RAG for Q&A: embed a query, retrieve top-k chunks, feed them to an
  LLM as grounding, generate an answer — the same retrieve-then-generate
  shape as this pipeline, just applied to catalog items and a ranked list
  instead of document chunks and a text answer
- Once response generation is turned on, this system *is* doing RAG

**A note on shared infrastructure:** a few things that sound agent/LLM-specific
are actually recsys concepts under a different name.
- Vector search/ANN indexes predate the RAG/LLM boom — built for search and
  recsys retrieval first (e.g. FAISS, released 2017), later adopted for LLM
  grounding. "Vector database" isn't agent-specific infrastructure.
- "Agent long-term memory" serves the same purpose as the persistent taste
  profile in layer 2 — both persist per-user signal across sessions — though
  the concrete storage shape can differ: memory tools like Mem0 often store
  retrievable text/fact snippets, while a taste profile here is a fixed
  feature vector

See the per-layer notes below for where these overlaps show up.

## Layers

### 1. Conversation layer *(optional — LLM wrapper)*
**Purpose:**
- Turn free-text user input into something structured the rest of the
  pipeline can use
- Hold multi-turn dialogue state (clarifying questions, follow-ups)
- Not part of the classic recsys backbone — most production recommenders
  (Netflix, Amazon, Booking.com) capture intent through explicit search/
  filter UI instead of a chat interface. This layer exists because the goal
  here is a conversational agent, not because it's a required recsys
  component.

**Technology:**
- LLM via API (Claude/GPT) with a structured-output or tool-calling schema
- Short-term memory: a message-history buffer holding multi-turn conversation
  state. Distinct from the long-term memory in layer 2 — this is scoped to
  the current session only and disappears when it does
- Prompt-based slot filling for the fields below

### 2. User understanding
**Purpose:** produce two distinct representations that later stages consume
separately:
- **Session intent** — what this specific trip/query wants (e.g.
  "honeymoon, quiet, beach, 5 nights, budget ~$150/day")
- **Persistent taste profile** — what this user tends to like across
  visits/trips (e.g. "usually books hostels," "prefers food-forward
  destinations"), built up over time from feedback

Kept separate because:
- They can conflict (a usually-budget traveler planning a honeymoon)
- A ranking model benefits from seeing both as separate signals rather than
  one blended query

**Technology:**
- Structured intent: filled either by an LLM (JSON schema populated from
  free text — optional path) or by explicit form/filter input (classic
  path: budget slider, date picker, tag checkboxes)
- Taste profile: a persisted embedding or feature vector per user, updated
  from historical interactions (start as a simple average of liked-item
  embeddings; later could be a learned user-tower in a two-tower model).
  Serves the same purpose as what agent frameworks call "long-term memory"
  (tools like Mem0 or Zep exist specifically to manage it) — same underlying
  idea, though those tools typically store retrievable text/facts rather
  than a fixed vector

### 3. Candidate retrieval
**Purpose:** cheaply narrow a catalog of thousands/millions of
destinations/items down to a few hundred plausible candidates, optimizing
for recall over precision.

**Technology:**
- Dense embeddings (sentence-transformers) + vector similarity search — this
  is what `scratch/candidate_generation/embed_retrieve` already does
- At production scale: approximate nearest neighbor index (FAISS, HNSW), or
  a managed vector database (Pinecone, Weaviate, Qdrant, `pgvector` on
  Postgres) instead of brute-force dot product. This is the same
  infrastructure RAG pipelines and agent memory systems use — vector search
  is recsys/search infrastructure that got reused for LLM grounding, not
  the other way around
- In real systems this is standard practice, not an enhancement: multiple
  retrieval sources are generated in parallel and merged before ranking
  ever sees them — embedding search, popularity/trending lists,
  collaborative filtering ("users with similar taste liked X"), geo/
  rule-based candidates. A single-source retriever (just embeddings) is a
  simplification made for learning purposes here.

### 4. Filtering
**Purpose:** enforce hard constraints that similarity search is bad at —
negation, numeric thresholds, availability/eligibility.
`scratch/candidate_generation/embed_retrieve/batch_test.py` already
demonstrates the failure mode this layer exists to fix: embeddings conflate
"no nightlife" with "vibrant nightlife," and don't reason about "under
$50/day" at all.

**Technology:**
- Plain structured queries/predicates over item metadata (budget range,
  dates, tags) applied to the retrieved candidate set
- No ML needed here — this is deliberately the "boring," deterministic layer

### 5. Ranking
Production systems split ranking into two stages rather than one:
- Running an expensive model over every retrieved candidate is too costly
  at scale
- A cheap model first narrows hundreds of candidates down to tens
- Only then does an expensive model score that much smaller set

#### 5a. Coarse ranking
**Purpose:** cheaply score hundreds of filtered candidates down to a few
tens, trading precision for speed.

**Technology:**
- A simple, fast model: logistic regression, a shallow GBDT, or even just
  the retrieval similarity score plus a couple of cheap features
  (popularity, recency)

#### 5b. Precise ranking
**Purpose:** score the smaller surviving candidate set precisely, using
richer features than embedding similarity alone (structured attributes,
taste-profile match, popularity, seasonality).

**Technology:**
- Start simple: gradient-boosted trees (LightGBM/XGBoost) with a
  learning-to-rank objective (pairwise/listwise loss) over structured
  features — this is what `scratch/ranking/rank_destinations` is scoped for
- Progress to: a small feedforward DNN or two-tower model (user-tower +
  item-tower, dot product score) once there's enough interaction data to
  learn embeddings jointly
- Transformer-based rankers (e.g. cross-encoders re-scoring query+item
  pairs) are the heavyweight end of this layer — worth trying as a toy
  project, likely overkill for the final system's data scale

### 6. Reranking
**Purpose:** adjust the top-ranked list for concerns the ranking model
doesn't optimize for directly:
- Diversity (don't return 5 beach towns in a row)
- Business rules (deprioritize closed-for-season destinations)
- Exploration (occasionally surface a lower-scored but novel item to gather
  feedback)

**Technology:**
- Maximal Marginal Relevance (MMR) or simple diversity bucketing by
  tag/region
- Deterministic business-rule filters/boosts
- Epsilon-greedy or similar exploration injection

### 7. Response generation *(optional — LLM wrapper)*
**Purpose:**
- Turn the final ranked list into a natural-language, conversational answer
  — an itinerary suggestion, not a bare list
- A classic recsys backbone stops at "results" (a ranked results page); this
  layer exists only because the goal here is a conversational agent rather
  than a search UI

**Technology:**
- LLM, prompted with the ranked candidates + original intent, generating
  grounded natural-language output (avoid letting it invent destinations
  not in the candidate list)
- This layer combined with candidate retrieval (layer 3) is what a RAG
  pipeline is — see the note above. There's no separate "RAG technology" to
  add on top of these two layers already existing

### 8. Agent tooling *(optional — LLM wrapper, beyond RAG)*
**Purpose:**
- Covers what happens once the agent needs to take actions or fetch live
  data mid-conversation, rather than just generating text grounded in a
  static catalog (e.g. checking real-time flight prices or weather before
  answering)
- The point where the system stops being "RAG over a fixed catalog" and
  becomes an agent that acts
- No classic-recsys analog — a ranking model never decides mid-run to call
  an external service

**Technology:**
- Tool use / function calling — the general mechanism for an LLM to invoke
  an external function during a conversation. MCP (Model Context Protocol)
  is a standardized protocol for exposing tools to an agent, so a tool can
  be written once and reused across agents instead of needing a bespoke
  schema per provider
- Agent orchestration frameworks (LangChain/LangGraph, Claude Agent SDK) —
  provide the control loop (call LLM -> maybe call a tool -> feed result
  back -> repeat) instead of hand-rolling it
- Multi-agent patterns — splitting into cooperating sub-agents (e.g. a
  flights sub-agent, a hotels sub-agent, an itinerary-planner sub-agent)
  with an orchestrator routing between them. Google's A2A (Agent2Agent)
  protocol is an emerging standard for agent-to-agent communication,
  analogous to what MCP is for agent-to-tool
- Guardrails / grounding checks — validating that the LLM only references
  real items/data and doesn't hallucinate. Only a concern because this
  layer generates free text and can act on the world, unlike a ranking
  model, which can only ever score items that already exist
- Agent tracing/observability (LangSmith, Langfuse) — the agent-world
  analogue of the feedback loop's event log below, but for debugging
  multi-step reasoning and tool-call traces rather than collecting training
  data

### 9. Feedback loop
**Purpose:** capture implicit/explicit signals (clicks, saves, itinerary
edits, thumbs up/down) to improve the taste profile and, eventually,
retrain the ranker.

**Technology:**
- Event logging (even a simple append-only log/SQLite table is enough at
  portfolio scale)
- Periodic batch job to refresh taste-profile vectors and, later, retrain
  the ranking model on logged interactions

## Toy project roadmap

Layout:
- Each project lives under `scratch/`, grouped by classic-backbone funnel
  stage into `candidate_generation/` and `ranking/`, plus `llm_wrapper/`
  for the optional conversational layers
- `feedback_taste_profile` and the shared synthetic-data prerequisite live
  at the `scratch/` root (see `scratch/README.md` for the exact layout)
- Each project targets one layer in isolation, using small public (or,
  where noted, synthetic) datasets

Goal of each project: not just one working implementation, but implementing
several of the production algorithms used for that layer side by side, on
the same data, so the differences in results, tuning knobs, and data
sensitivity are felt directly rather than read about.

Numbering follows the backbone first (classic recsys, no LLM needed), then
the optional LLM wrapper layers, though nothing stops jumping around.

| # | Project | Layer | Algorithms to compare | Knobs to tune | Status |
|---|---------|-------|------------------------|----------------|--------|
| 1 | `candidate_generation/embed_retrieve` | Candidate retrieval (embeddings) | Brute-force dot product (done) vs. FAISS IVF vs. HNSW vs. LSH | `nlist`/`nprobe` (IVF), `M`/`efConstruction`/`efSearch` (HNSW), embedding model choice | In progress — brute-force baseline done, ANN variants not yet built; `batch_test.py` surfaces negation/numeric failure modes |
| 2 | `candidate_generation/collab_filter` | Candidate retrieval (collaborative filtering) | Matrix factorization (ALS/SVD) vs. item-based neighborhood CF vs. implicit-feedback BPR, on synthetic user-item interactions | latent dimension, regularization strength, implicit vs. explicit feedback handling | Blocked on `synthetic_interactions` (below) |
| 3 | `candidate_generation/filter_constraints` | Filtering | N/A — deterministic predicate logic, not an algorithm-comparison layer | predicate strictness (hard-fail vs. soft-penalize a near-miss) | Not started — parse hard constraints out of a query and apply as predicates over Project 1's candidate set |
| 4 | `ranking/rank_coarse` | Coarse ranking | Raw retrieval score baseline vs. logistic regression vs. shallow GBDT | feature set size, regularization/tree depth | Not started — measure how much of the precise ranker's top results survive coarse pruning, at what latency savings |
| 5 | `ranking/rank_destinations` | Precise ranking (GBDT / LTR) | Pointwise regression vs. pairwise (LambdaMART) vs. listwise (LambdaRank/ListNet) objectives in LightGBM/XGBoost | `num_leaves`, learning rate, boosting rounds, objective function | Scaffolded — dataset chosen, no code yet |
| 6 | `ranking/rank_two_tower` | Precise ranking (DNN / two-tower) | Two-tower DNN vs. Project 5's GBDT baseline | embedding dimension, negative sampling strategy (random / in-batch / hard negatives), tower depth | Blocked on `synthetic_interactions` (below) |
| 7 | `ranking/rank_cross_encoder` | Precise ranking (transformer) | Cross-encoder re-scoring vs. two-tower (Project 6) vs. GBDT (Project 5) | model size, pair batch size | Not started — quality vs. latency tradeoff at the expensive end of ranking |
| 8 | `ranking/rerank_diversity` | Reranking | MMR vs. determinantal point processes (DPP) vs. simple tag-bucketing | MMR's relevance/diversity tradeoff weight, DPP kernel choice | Not started — plot the diversity-vs-relevance tradeoff curve over Project 5/6/7's output |
| 9 | `feedback_taste_profile` | Feedback loop / user understanding | Simple average of liked-item embeddings vs. recency-decay-weighted average vs. learned user-tower | decay rate, aggregation window | Blocked on `synthetic_interactions` (below) |
| 10 | `llm_wrapper/intent_parsing` *(optional, LLM)* | Conversation layer / user understanding | N/A — LLM structured-output extraction | prompt/schema design, temperature | Not started — LLM structured-output extraction of intent JSON from free text |
| 11 | `llm_wrapper/response_generation` *(optional, LLM)* | Response generation | N/A — grounded generation | prompt design, temperature | Not started — grounded natural-language generation over a fixed candidate list (this + Project 1 is a RAG pipeline) |
| 12 | `llm_wrapper/tool_calling_agent` *(optional, LLM)* | Agent tooling | N/A — tool-calling integration | — | Not started — wire a mock MCP tool (e.g. live weather or price lookup) into the conversational flow to see the retrieve-then-act loop firsthand |

Projects 3, 10, 11, and 12 are marked N/A for algorithm comparison:
- Those layers are deterministic logic or LLM-prompting concerns rather
  than a choice between competing statistical/ML algorithms
- Nothing analogous to "FAISS vs. HNSW" to compare there

**A practical note on data:**
- `collab_filter`, `rank_two_tower`, and `feedback_taste_profile` all need
  user-item interaction history that the current catalog-only datasets
  (city descriptions, structured tourist-destination features) don't have
- `scratch/synthetic_interactions/` is a shared prerequisite that generates
  this once for all three:
  - Synthetic users built from a handful of hidden "traveler persona"
    preference vectors over `embed_retrieve`'s existing tag columns
    (culture, adventure, nature, beaches, nightlife, cuisine, wellness,
    urban, seclusion)
  - An impression → click → save funnel (not just raw positive pairs), so
    popularity bias and exposure noise are present
  - Tunable knobs (persona mix purity, popularity skew, signal sparsity) —
    see that project's README for the full design
- Build this before starting any of the three blocked projects above

Once each toy project has produced a working comparison:
- Not just one algorithm working, but a felt sense of how each option
  trades off quality, latency, and sensitivity to data/tuning against the
  others
- Then: assemble the full pipeline as a single service, choosing one
  algorithm per layer deliberately based on what was learned, rather than
  defaulting to whichever was easiest to stand up first

## Note: production infrastructure not covered here

Deliberately out of scope for this learning project:
- Feature store (online/offline feature parity)
- A/B testing framework
- Multi-objective ranking (engagement + revenue + diversity combined as
  weighted losses, rather than one relevance score)
- Tool sandboxing/permissioning and auth for external API access (agent
  side)
