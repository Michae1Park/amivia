"""Four collaborative-filtering retrievers over the synthetic interaction log.

All hand-rolled in NumPy/SciPy — the point of this project is to feel how the four
differ, which reading library call signatures does not teach. `verify_vs_implicit.py`
cross-checks ALS and BPR against the `implicit` library so a silent gradient bug
can't masquerade as a finding.

Every model implements the eval harness contract, `.recommend(user_id, k)`, plus
`.scores(user_id)` over the full catalog — the popularity-correlation diagnostic
needs raw scores, not just a truncated list.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eval"))
from data import GRADE  # noqa: E402

# Impressions carry no positive signal (same convention as eval/data.py), so the
# matrix is built from clicks and saves only.
WEIGHTINGS = ("binary", "confidence")


def build_matrix(train_rows, item_ids, weighting="confidence"):
    """train rows -> (CSR users x items, user_id -> row index).

    binary:     click or save == 1, the plain implicit-feedback view
    confidence: save=2, click=1, the graded view the eval harness scores against
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    item_index = {item_id: i for i, item_id in enumerate(item_ids)}
    user_ids = sorted({r["user_id"] for r in train_rows})
    user_index = {u: i for i, u in enumerate(user_ids)}

    # Deduplicate to the STRONGEST grade per (user, item), matching
    # eval/data.py's `max(cur, grade)`. Summing and clipping instead would score two
    # clicks in separate sessions the same as a save, inventing signal that isn't there.
    strongest: dict[tuple[int, int], float] = {}
    for r in train_rows:
        grade = GRADE.get(r["event_type"])
        if grade is None or r["item_id"] not in item_index:
            continue
        key = (user_index[r["user_id"]], item_index[r["item_id"]])
        value = 1.0 if weighting == "binary" else float(grade)
        if value > strongest.get(key, 0.0):
            strongest[key] = value

    keys = list(strongest)
    matrix = sp.csr_matrix(
        (list(strongest.values()), ([k[0] for k in keys], [k[1] for k in keys])),
        shape=(len(user_ids), len(item_ids)), dtype=np.float64,
    )
    return matrix, user_index


class CFRecommender:
    """Shared plumbing: seen-filtering, top-k, and a popularity fallback for users
    with no training history (rare here, but a model shouldn't be scored on a
    cold-start artefact that has nothing to do with its algorithm)."""

    name = "cf"

    def __init__(self, matrix, user_index, item_ids, seen):
        self.matrix = matrix.tocsr()
        self.user_index = user_index
        self.item_ids = item_ids
        self.seen = seen
        counts = np.asarray(self.matrix.sum(axis=0)).ravel()
        self.popularity_order = np.argsort(-counts)

    def scores(self, user_id) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_id: str, k: int) -> list[str]:
        if user_id not in self.user_index:
            order = self.popularity_order
        else:
            order = np.argsort(-self.scores(user_id))
        seen = self.seen.get(user_id, set())
        out = []
        for idx in order:
            item_id = self.item_ids[idx]
            if item_id not in seen:
                out.append(item_id)
                if len(out) == k:
                    break
        return out

    def _row(self, user_id) -> np.ndarray:
        return np.asarray(self.matrix[self.user_index[user_id]].todense()).ravel()


class ItemKNN(CFRecommender):
    """Neighbourhood CF: cosine similarity between items over their co-interaction
    columns. No latent space, no training loop — just a 560x560 similarity matrix."""

    name = "item_knn"

    def __init__(self, matrix, user_index, item_ids, seen, n_neighbors=50, **_):
        super().__init__(matrix, user_index, item_ids, seen)
        self.n_neighbors = n_neighbors

        norms = np.sqrt(np.asarray(self.matrix.power(2).sum(axis=0))).ravel()
        norms[norms == 0] = 1.0
        normalized = self.matrix.multiply(sp.csr_matrix(1.0 / norms)).tocsr()
        sim = np.asarray((normalized.T @ normalized).todense())
        np.fill_diagonal(sim, 0.0)  # an item is not its own neighbour

        if n_neighbors and n_neighbors < sim.shape[0]:
            # Keep only each item's strongest neighbours: the long tail of weak
            # similarities is mostly co-popularity noise.
            cutoff = np.partition(sim, -n_neighbors, axis=1)[:, -n_neighbors][:, None]
            sim = np.where(sim >= cutoff, sim, 0.0)
        self.similarity = sim

    def scores(self, user_id) -> np.ndarray:
        return self._row(user_id) @ self.similarity


class PureSVD(CFRecommender):
    """Truncated SVD over the raw matrix, missing entries treated as 0.

    That is the honest weakness to contrast against ALS: an unobserved item and a
    disliked one are indistinguishable here, whereas ALS separates preference from
    confidence.
    """

    name = "svd"

    def __init__(self, matrix, user_index, item_ids, seen, n_factors=32, random_state=42, **_):
        super().__init__(matrix, user_index, item_ids, seen)
        n_factors = min(n_factors, min(self.matrix.shape) - 1)
        rng = np.random.default_rng(random_state)
        v0 = rng.standard_normal(min(self.matrix.shape))
        u, s, vt = svds(self.matrix.astype(np.float64), k=n_factors, v0=v0)
        self.user_factors = u * s
        self.item_factors = vt.T

    def scores(self, user_id) -> np.ndarray:
        return self.user_factors[self.user_index[user_id]] @ self.item_factors.T


class ALS(CFRecommender):
    """Implicit-feedback ALS (Hu, Koren & Volinsky 2008).

    Confidence C = 1 + alpha*r, preference p = 1 for anything observed. Each
    alternating step is a ridge solve per row; the standard trick is to precompute
    Y^T Y once and add only the observed items' contribution per user, which is
    what makes this O(nnz * f^2) rather than O(users * items * f^2).
    """

    name = "als"

    def __init__(self, matrix, user_index, item_ids, seen, n_factors=32, reg=0.01,
                 alpha=40.0, n_iterations=15, random_state=42, **_):
        super().__init__(matrix, user_index, item_ids, seen)
        rng = np.random.default_rng(random_state)
        n_users, n_items = self.matrix.shape
        self.user_factors = 0.01 * rng.standard_normal((n_users, n_factors))
        self.item_factors = 0.01 * rng.standard_normal((n_items, n_factors))

        user_major = self.matrix.tocsr()
        item_major = self.matrix.T.tocsr()

        for _ in range(n_iterations):
            self.user_factors = self._solve(user_major, self.item_factors, reg, alpha)
            self.item_factors = self._solve(item_major, self.user_factors, reg, alpha)

    @staticmethod
    def _solve(observations, fixed, reg, alpha):
        n_factors = fixed.shape[1]
        FtF = fixed.T @ fixed
        eye = reg * np.eye(n_factors)
        out = np.zeros((observations.shape[0], n_factors))
        indptr, indices, data = observations.indptr, observations.indices, observations.data
        for row in range(observations.shape[0]):
            start, end = indptr[row], indptr[row + 1]
            if start == end:
                continue
            idx = indices[start:end]
            confidence_minus_one = alpha * data[start:end]
            F = fixed[idx]
            A = FtF + (F.T * confidence_minus_one) @ F + eye
            b = F.T @ (1.0 + confidence_minus_one)  # Y^T C_u p_u, with p_u == 1
            out[row] = np.linalg.solve(A, b)
        return out

    def scores(self, user_id) -> np.ndarray:
        return self.user_factors[self.user_index[user_id]] @ self.item_factors.T


class BPR(CFRecommender):
    """Bayesian Personalised Ranking (Rendle 2009): SGD on (user, positive, negative)
    triples maximising log sigmoid(score_i - score_j).

    Optimises ranking directly rather than reconstructing the matrix, which is the
    interesting contrast against ALS/SVD. Scatter-adds use np.add.at because a
    minibatch can contain the same user twice and plain fancy-index assignment
    would silently drop one of the updates.
    """

    name = "bpr"

    def __init__(self, matrix, user_index, item_ids, seen, n_factors=32, reg=0.01,
                 learning_rate=0.05, n_epochs=30, batch_size=4096, random_state=42, **_):
        super().__init__(matrix, user_index, item_ids, seen)
        rng = np.random.default_rng(random_state)
        n_users, n_items = self.matrix.shape
        self.user_factors = 0.1 * rng.standard_normal((n_users, n_factors))
        self.item_factors = 0.1 * rng.standard_normal((n_items, n_factors))

        coo = self.matrix.tocoo()
        pos_users, pos_items = coo.row, coo.col
        n_positives = len(pos_users)
        positive_sets = [set(self.matrix.indices[self.matrix.indptr[u]:self.matrix.indptr[u + 1]].tolist())
                         for u in range(n_users)]

        for _ in range(n_epochs):
            order = rng.permutation(n_positives)
            for start in range(0, n_positives, batch_size):
                batch = order[start:start + batch_size]
                u, i = pos_users[batch], pos_items[batch]
                j = self._sample_negatives(u, positive_sets, n_items, rng)

                U, I, J = self.user_factors[u], self.item_factors[i], self.item_factors[j]
                x_uij = np.einsum("ij,ij->i", U, I - J)
                # sigmoid(-x): the gradient weight, large where the model has the pair wrong
                g = (1.0 / (1.0 + np.exp(np.clip(x_uij, -30, 30))))[:, None]

                grad_u = g * (I - J) - reg * U
                grad_i = g * U - reg * I
                grad_j = -g * U - reg * J
                np.add.at(self.user_factors, u, learning_rate * grad_u)
                np.add.at(self.item_factors, i, learning_rate * grad_i)
                np.add.at(self.item_factors, j, learning_rate * grad_j)

    @staticmethod
    def _sample_negatives(users, positive_sets, n_items, rng, max_retries=5):
        """Uniform negatives, resampled where they collide with a real positive.
        Positives are ~20 of 560 items, so collisions are rare and a few retries
        clear nearly all of them."""
        j = rng.integers(0, n_items, size=len(users))
        for _ in range(max_retries):
            clashes = np.array([item in positive_sets[u] for u, item in zip(users, j)])
            if not clashes.any():
                break
            j[clashes] = rng.integers(0, n_items, size=int(clashes.sum()))
        return j

    def scores(self, user_id) -> np.ndarray:
        return self.user_factors[self.user_index[user_id]] @ self.item_factors.T


MODELS = {m.name: m for m in (ItemKNN, PureSVD, ALS, BPR)}
