# Recommender scope, data, and roadmap

`docs/architecture.md` plans one thing well: **which city should this user go
to.** This document extends that to the whole system — what else to recommend,
whether the data exists, and in what order to assemble it.

**Scope decision:** Amivia stays a *recommender system*. Itinerary
scheduling and live booking are out — see [Out of scope](#out-of-scope).

## How to read this document

**Goal: a portfolio demo.** Not a commercial product — a functioning
end-to-end system that demonstrates each layer of a real recsys pipeline, and
that could be improved with more data and users later. That target changes
what "done" means throughout this document:

- **Breadth beats depth.** One working, explained instance of each layer is
  worth more than a heavily tuned ranker and four empty directories.
- **Synthetic data is fine, stated plainly.** It can't validate a modeling
  hypothesis (see [below](#synthetic-models-ship-cold)), but a demo
  that says so explicitly demonstrates better judgment than one that quietly
  overclaims.
- **The measurement work is itself a deliverable.** `scratch/eval/` — and
  especially its exposure-bias finding — shows more than any single model
  would. Infrastructure that can't reach significance at demo scale is still
  worth building and explaining.
- **Commercial constraints mostly relax.** Non-commercial data licences,
  live inventory, per-request cost control, and catalog scale stop being
  blockers. Attribution still matters; competitive/commercial terms don't.

**This is a destination map, not a work queue.** The build order to follow is
[`architecture.md`](architecture.md)'s toy-project roadmap — learn the
components first. The phase table at the end is what assembling them would
look like, not the order to learn in.

Three findings here are worth carrying back into the toy projects now, because
they change what those projects can *prove*, not when to do them:

- **Synthetic data validates an implementation, not a hypothesis**
  ([below](#synthetic-models-ship-cold)) — read `collab_filter` and
  `rank_two_tower` results accordingly.
- **`popularity` is the bar to beat, not `random`** — the exposure bias in
  `scratch/eval/README.md` penalises anything ranking by inferred taste.
- **Open POI data has no quality signal**
  ([below](#no-quality-signal-in-open-poi-data)) — why ranking, not retrieval,
  is the hard part once the catalog outgrows cities.

## At a glance

| | |
|---|---|
| **Recommend** | Cities · When-to-go · Neighborhoods · Attractions · Restaurants |
| **Have now** | City catalog + features (560 cities, MIT-licensed) |
| **Need to acquire** | POI catalog (free, Foursquare/OSM) |
| **Known limits** | No quality signal in open POI data; interaction data is synthetic — both documented, neither blocking a demo |
| **Build order** | Toy projects first (`architecture.md`), then assemble: serving v1 → POIs → neighborhoods → learned ranking on real logs |

---

## 1. What to recommend

Each entity type is its own recommender — own catalog, own features, own
funnel — sharing one taste profile and intent.

| # | Entity | Status | Why it earns a place |
|---|---|---|---|
| 1 | **Cities** | Built (`embed_retrieve`) | Rich features, MIT-licensed, real preference variance |
| 2 | **When to go** (city × month) | Proposed | Nearly free — uses climate data already in `cities.csv` |
| 3 | **Neighborhoods / areas** | Proposed | The honest answer to "more specific destinations" |
| 4 | **Attractions / things to do** | Proposed | Good catalog coverage, notability signal recoverable |
| 5 | **Restaurants / food** | Proposed | Highest volume and highest personalization value |

**When to go.** `cities.csv` carries full monthly min/avg/max temperature per
city, currently unused by the retrieval pipeline. Ranking (city, month) pairs
instead of cities is a distinct, cheap surface built on an asset already
owned. Caveat: climate is the only seasonality signal available — crowds,
price, and festivals are not in the data.

**Neighborhoods.** Not acquired — *derived*, by aggregating POIs into OSM
boundary polygons: bar density → nightlife character, cuisine spread → food
scene, POI density → walkability. Feature engineering over data collected for
(4) and (5) anyway, producing genuinely useful output ("stay in Trastevere,
not by the station").

**Hotels — deliberately excluded.** The features that decide a hotel choice
(price, availability, review score, star rating, amenities) are exactly the
ones open POI data lacks. A hotel recommender on Foursquare/Overture would be
ranking on name and coordinates. Recommending the *area* instead covers most
of the real need; revisit properties only if live inventory is integrated.

---

## 2. Does the data exist?

Every recommender needs four layers. Catalogs are the easy part — the bottom
two rows are where the real gaps are.

| Layer | Cities | When to go | Neighborhoods | Attractions | Restaurants |
|---|---|---|---|---|---|
| **Item catalog** | ✅ `cities.csv` | ✅ derived from `cities.csv` | ⚠️ OSM boundaries | ✅ Foursquare OS | ✅ Foursquare OS |
| **Content features** | ✅ `cities.csv` | ✅ `cities.csv` climate | ⚠️ aggregate from POIs | ⚠️ FSQ category + Wikivoyage, LLM-tagged | ⚠️ FSQ category + OSM tags, LLM-tagged |
| **Quality / popularity** | ❌ none | — n/a | ⚠️ POI density | ⚠️ Wikipedia pageviews | ❌ commercial API only |
| **Interaction data** | 🧪 synthetic | 🧪 synthetic | 🧪 synthetic † | 🧪 synthetic † | 🧪 synthetic † |

✅ have it, or straightforward · ⚠️ obtainable with work · ❌ no free source ·
🧪 synthetic — fine for a demo, stated plainly

† The generator exists for cities only; pointing it at POIs needs the catalog
first, so it lands with Phase 2. Not a harder problem — just a later one. What
synthesis can and can't fill is [below](#what-synthesis-can-and-cant-fill).

### Where each source comes from

| Source type | What | Notes |
|---|---|---|
| **Existing local data** | `cities.csv` — 560 cities | Kaggle, MIT-licensed. Already in the repo. |
| **Free bulk download** | Foursquare OS Places, Overture, OSM, Wikivoyage | One-time download, load locally. No API, no rate limits. |
| **Free API** | Wikipedia pageviews | Real popularity signal, but only for notable places. |
| **Commercial API** | Google Places, Yelp Fusion, Foursquare Pro | Request-time enrichment only — see [No quality signal](#no-quality-signal-in-open-poi-data). |
| **Derived** | Neighborhoods, when-to-go, density priors | Computed from the above. No acquisition needed. |
| **LLM-inferred** | POI tag vectors, from name + category + editorial text | Labeling over real evidence, not fabrication. One-time batch pass, nothing in the request path. |
| **Synthetic** | `synthetic_interactions/` | Trains the v1 personalization path; retrained on real logs as they arrive — see [Synthetic models ship cold](#synthetic-models-ship-cold). |

**The two systemic gaps** — no quality prior, no *real* interaction data —
apply to every entity type, and are what most determine whether ranking works
at all. Both are covered below.

---

## 3. Gaps to close

### The catalog stops at cities

**The problem.** The catalog is 560 cities. "What to do, where to eat" needs
POI data — 4-6 orders of magnitude more rows, with different features
(coordinates, category, hours) and a different freshness requirement
(restaurants close; cities don't).

**Why it's the highest priority.** `architecture.md` §7 has the LLM generate
its answer from the ranked *city* list. The moment a user asks anything more
specific than which city, the LLM has no real POIs to ground in — so it
invents plausible-sounding restaurants. That is the most user-visible failure
mode available to this system.

Same lesson `embed_retrieve/batch_test.py` surfaced one level down:
embeddings can't do negation or numeric thresholds, so a deterministic layer
was added. The LLM can't be trusted with facts it wasn't handed, so the facts
have to exist first.

**Source options:**

| Source | Scale | License | Notes |
|---|---|---|---|
| [Foursquare OS Places](https://opensource.foursquare.com/os-places/) | 100M+ POIs | Apache 2.0 | Commercial OK, monthly updates, Parquet on S3. **Best default.** |
| [Overture Maps Places](https://overturemaps.org/download/) | ~60M+ POIs | CDLA Permissive v2.0 | Meta/Microsoft/AWS/TomTom-backed. Also commercial-friendly. |
| OpenStreetMap / Overpass | Global | ODbL (share-alike) | Neighborhood boundaries and per-POI tags (`cuisine`, `opening_hours`). |
| Wikivoyage | ~30k destinations | CC BY-SA | Human-written "See / Do / Eat / Sleep". Best source of *editorial* voice. |
| Google Places / Yelp Fusion | Best quality | Restrictive | Caching limits rule them out as primary storage. Enrichment only. |

**Recommendation:** Foursquare OS Places as the base catalog, Wikivoyage for
editorial text, OSM for tags and boundaries — but read the next section
before assuming that is sufficient.

### No quality signal in open POI data

**Verified from the schemas**, and this is the more serious half of the
catalog problem:

- **Foursquare OS Places** ships 26 attributes — id, name, coordinates,
  address parts, category, phone, website, socials, dates. **No ratings, no
  review counts, no popularity, no price tier, no opening hours.** The docs
  point to "Places Pro & Premium" for those.
- **Overture Places** likewise has no ratings, price, or hours. Its only
  quality field is `confidence` (0-1, "likelihood that the place exists"),
  and its own docs warn the theme "is known to contain duplicates, a high
  junk rate, and low property completeness."

Open POI data tells you a place **exists and what category it is** — not
whether it is any good. Those are precisely the fields Foursquare, Google,
and Yelp monetize.

**Why it matters.** Retrieval and filtering work fine on this data. *Ranking*
does not: with no popularity or quality prior, a restaurant ranker has
nothing beyond content match, and every mediocre place looks identical to the
best one on the street.

**The fix — a hybrid split:**

- **Open data as the persistent catalog.** Retrieval and filtering run over
  the full thousands-of-POIs-per-city set, stored locally.
- **Live API as display-time enrichment.** After ranking narrows to ~10, call
  Google Places / Yelp / FSQ Pro to attach rating, price, and hours to just
  those. Cost scales with sessions rather than catalog size, and it stays
  within terms — the caching restrictions exist to prevent building a
  competing database, which this pattern does not do.

**Free partial substitutes** (no API, usable as ranking features):

- Wikipedia/Wikivoyage presence and pageviews → attraction notability. Good
  for attractions, useless for restaurants.
- OSM tags (`cuisine`, `opening_hours`, `diet:vegetarian`) → uneven coverage,
  but real and unrestricted.
- Chain-vs-independent, POI density, category rarity → weak but genuine
  priors derivable from the catalog itself.

### Synthetic models ship cold

`synthetic_interactions/` generates users from six invented personas. That is
genuinely good for *learning and validating algorithms* — real latent
structure to recover, tunable noise — and good enough to launch on.

Nothing about invented personas stops Projects 2 (`collab_filter`), 6
(`rank_two_tower`), and 9 (`feedback_taste_profile`) from training, serving,
and being demonstrated end to end. The training code, the serving path, the
event log, and the retraining loop are all real regardless of what the data
was. Swapping synthetic interactions for real logs is then a data-source
change, not a rebuild — which is the entire reason to build the loop before
there are users.

What synthetic training does *not* buy is a model that is any good on day one:

- **The latent space is the generator's, not the world's.** Six clean personas
  are lower-noise and more separable than real taste, so the model doesn't
  start uniformly noisy — it starts *confidently wrong in a structured way*, a
  strong prior pointed slightly askew.
- **Cold start is arithmetic, and retraining can't outrun it.** CF and
  two-tower need many users × many interactions. On launch day there is one
  real user, then five. Retraining conjures no neighborhood that doesn't exist
  yet.
- **`popularity` is still the bar** (`scratch/eval/README.md`). A
  synthetically trained ranker isn't guaranteed to clear it on real traffic.

**So v1 ships both paths and blends them:**

| Path | Runs on | Covers |
|---|---|---|
| **Cold-start floor** | Content retrieval (1) + hard filters (3) + onboarding quiz over the 9 tag dimensions + heuristic ranking | Every user from their first request, zero interactions required |
| **Learned personalization** | Projects 2 / 6 / 9, trained on `synthetic_interactions/`, retrained on real logs as they arrive | Users with enough history for the models to say anything |

LLM intent parsing (10) and grounded response generation (11) sit across both.

- **Blend by interaction count, not by launch date.** Weight the learned score
  against the content/quiz score as a function of how much real history a user
  has, shrinking toward `popularity` where it's thin. The ramp from synthetic
  to real is then continuous and per-user, with no cutover to plan.
- **The quiz is what makes the floor work.** Eliciting the taste vector
  directly is the only way to personalise at zero interactions — and it seeds
  the learned models with a real prior instead of a cold one.

Retraining on real logs is the deliverable, not a caveat. The honest demo is
one that shows a model trained on invented users, the loop that corrects it,
and the fallback carrying the product while that correction happens.

### What synthesis can and can't fill

Most of the ⚠️ and ❌ cells in §2 can be filled by generating data. One can't,
and the line between them is worth stating once rather than re-deciding per
entity:

- **Simulate behavior — freely.** Interactions are already invented for
  cities; pointing the same personas × tag-vector generator at POIs is the
  same machinery, not a new claim about the world. Gated on the POI catalog,
  so it arrives with Phase 2.
- **Infer features — from real evidence.** POI content features are thin, not
  absent. Name, category, and Wikivoyage text are real, and an LLM can batch
  them into the same 9-dimension tag vector `cities.csv` already uses. That's
  a labeling function, not synthesis, and it's what lets one funnel serve
  every entity — same feature schema throughout. One-time pass over the
  catalog, nothing in the request path. Coverage is uneven and skews toward
  well-known places, so spot-check against a small hand-labeled holdout.
- **Never fabricate attributes of real entities.** A synthetic user is
  disclosed fiction and harms nobody. A synthetic 4.6★ on a real, named
  restaurant is a false claim about a real business, handed to the user as
  fact by the response layer. It's the hallucinated-restaurant failure mode
  moved upstream into the datastore, where it looks authoritative instead of
  improvised. It also voids the eval: a ranker trained on invented quality
  scores measures only how well it recovered the generator.

This is why restaurant quality stays ❌ while everything around it fills in —
that gap is closed by the [live-enrichment
split](#no-quality-signal-in-open-poi-data), not by generation.

**The escape hatch, should that gap ever block the demo:** make the catalog
fictional too. Invented POIs in invented cities assert nothing about the real
world, so every cell in §2 fills, quality included. That trades real places
for complete data — a defensible trade, provided it's stated up front rather
than discovered by a user who tries to visit one.

### Nothing in the repo is a service yet

Everything under `scratch/` is scripts. Becoming a service needs:

- **API layer** — FastAPI
- **Datastore** — Postgres + `pgvector` handles catalog, embeddings, and
  event log in one system. A dedicated vector DB is unjustified complexity at
  this scale.
- **User accounts** — required for persistent taste profiles to mean anything
- **Frontend** — real users imply a real interface
- **LLM cost control** — the LLM sits in the request path twice (intent,
  response), so per-session cost becomes an operational number to watch

### Eval needs to cover text and multiple entity types

`scratch/eval/` measures ranked lists (recall, hit-rate, NDCG) for one
catalog. Two extensions:

- **Per-entity offline eval.** Same metrics, rerun per entity type. Returning
  10 of 560 cities and 10 of 8,000 restaurants are very different difficulties
  — one blended number hides that.
- **Grounding check on generated output.** Does every place named in the
  response exist in the catalog that was actually retrieved? One string-match
  assertion catches the hallucinated-restaurant failure mode.
- **Soft quality** (expensive, sampled) — human rating or LLM-as-judge on a
  rubric over fixed scenarios.

### Online experimentation: build the mechanism, not the verdict

`architecture.md` lists A/B testing as out of scope. For a demo it's back in
— but as a *mechanism to demonstrate*, not a source of conclusions.

A split test needs ~5,000+ sessions per arm to detect a realistic ranking lift
(CTR 15% → 17%, 80% power). At demo scale any result is noise, so the
deliverable is the harness plus the reasoning behind it:

- **Interleaving (team-draft)** — blend two rankers into one list, attribute
  clicks back to the source. ~10-100x more sensitive than A/B, because every
  user contributes to every comparison. Runs fine on synthetic users.
- **Offline eval stays the arbiter.** `scratch/eval/` is the quality bar;
  online testing is a sanity check.

Choosing interleaving over A/B, and being able to say why, is the artifact
here. Event logging and variant assignment can't be backfilled — build them
with the service, whatever their statistical power.

### Licensing

For a portfolio demo, commercial terms mostly don't bind — attribution does.

- **`cities.csv`** — MIT (checked). Keep the notice.
- **Foursquare OS** (Apache 2.0), **Overture** (CDLA), **Wikivoyage**
  (CC BY-SA), **OSM** (ODbL) — all fine with attribution; the last two are
  share-alike.
- **Google Places / Yelp** — caching limits still apply to stored data.
- **Non-commercial datasets** (e.g. Booking.com MDT) are usable in a demo,
  provided it isn't commercialised later without revisiting this.

---

## 4. Architecture: a pipeline of pipelines

`architecture.md` assumes one catalog and one funnel. The product needs the
funnel pattern repeated per entity:

```
intent + taste profile
  ├─> CITY FUNNEL           (560 items — retrieve → filter → rank → rerank)
  │     └─> chosen city
  │           ├─> WHEN-TO-GO        (12 month-slots, reranks the city itself)
  │           ├─> AREA FUNNEL       (tens of neighborhoods)
  │           ├─> ATTRACTION FUNNEL (hundreds–thousands)
  │           └─> FOOD FUNNEL       (thousands–tens of thousands)
  │                 └─> live enrichment of final ~10 (rating/price/hours)
  └─> RESPONSE GENERATION (grounded strictly in the above)
```

Two consequences:

- **The per-city funnels hold the real data volume and latency.** The
  560-city funnel is the small one. This is where the unbuilt ANN half of
  Project 1 stops being academic — brute force over 560 cities is instant,
  brute force over every restaurant in Tokyo per request is not.
- **Write the funnel once, parameterized by entity type**, not copy-pasted
  five times. Each entity supplies its own catalog, embeddings, filter
  predicates, and features; `retrieve → filter → rank → rerank` is shared.
  Cheap to do at Phase 2, expensive to retrofit after three copies exist.

---

## 5. Build phases

Assembly order once the toy projects have covered the components. Each phase
ends with something demonstrable.

| Phase | Goal | Entities | Contents |
|---|---|---|---|
| **0. Foundations** | — | — | Choose POI source; Postgres + pgvector; event logging from the first request |
| **1. Demo v1** | The cold-start floor, serving | Cities, When-to-go | Projects 1 + 3 + heuristic ranking + onboarding quiz + Projects 10/11, behind FastAPI and a minimal UI |
| **2. Specificity** | The "what to do there" half | + Attractions, Food | POI ingest; entity-parameterized funnel; Wikipedia/OSM quality priors; grounding eval |
| **3. Areas** | Sub-city granularity | + Neighborhoods | Aggregate Phase 2's POIs into OSM polygons; derive area features. No new acquisition. |
| **4. Learned ranking** | Add the personalization path above the floor | all | Projects 2 / 5 / 6 / 9 trained on whatever logs exist, synthetic or real; blend weight keyed on per-user interaction count; retraining job |
| **5. Experimentation** | Show the online-eval loop | all | Interleaving harness + variant assignment, runnable on synthetic users |

**Why this order:**

- Phase 1 is model-free by choice, not by necessity — it's achievable now, and
  it forces the serving/UI work everything else depends on. It also has to
  exist regardless: it's the cold-start floor the learned path falls back to
  ([above](#synthetic-models-ship-cold)), so it is never throwaway
  scaffolding.
- Phase 3 follows Phase 2 because neighborhoods are *derived from* the POI
  catalog.
- Phases 4 and 5 are **not gated on real users**. A product would wait for
  real logs; a demo builds the mechanism on synthetic data and says so. Real
  logs improve the numbers, not the demonstration.

---

## Out of scope

Recorded so the reasoning isn't relitigated later.

**Itinerary scheduling.** "Pick k places, fit them into d days respecting
opening hours and travel time" is the **Tourist Trip Design Problem** — a
Team Orienteering Problem with Time Windows variant, NP-hard, with its own
literature. The funnel produces *scores*; a planner *selects and sequences*
under hard constraints. No amount of ranking work produces it. If ever
wanted, it sits between reranking and response generation and needs a
travel-time matrix plus OR-Tools — not a bigger model.

**Live inventory (hotel/flight booking).** Priced, availability-constrained,
changes by the minute, gated behind partner agreements (Amadeus for a
prototype; Booking/Expedia affiliates need approval). An integration problem
with no recsys content. The one in-scope exception is live API calls as
display-time enrichment — a much smaller commitment.

---

## Appendix: relationship to the toy-project roadmap

The 12 toy projects remain the right way to *learn* each layer, and several
feed directly into the phases above. But they are a learning curriculum, not
a build order — Phase 1 needs only 4 of the 12, while needing several things
(POI ingest, entity-parameterized funnels, serving, logging, interleaving)
that no toy project covers.
