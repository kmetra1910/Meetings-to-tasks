"""Estimate the number of speakers with NME-SC (Park et al., 2019).

For each pruning level p only the p most similar windows are kept per row of the
cosine affinity matrix. The Laplacian eigengap of the pruned graph gives a speaker
count; the level whose largest gap is most pronounced relative to p is selected.
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples


def pruned_affinity(x, p):
    sim = x @ x.T
    keep = np.argpartition(-sim, p, axis=1)[:, :p]
    a = np.zeros_like(sim)
    a[np.arange(len(x))[:, None], keep] = 1.0
    return (a + a.T) / 2


def estimate(x, min_speakers=2, max_speakers=12, levels=30):
    n = len(x)
    max_speakers = min(max_speakers, n - 1)
    min_speakers = min(min_speakers, max_speakers)
    best = None
    for p in np.unique(np.geomspace(max(5, n*0.005), max(6, n*0.25), levels).astype(int)):
        a = pruned_affinity(x, int(p))
        laplacian = np.diag(a.sum(1)) - a
        eig = eigh(laplacian, eigvals_only=True)
        gaps = np.diff(eig[:max_speakers+1]) / max(eig[-1], 1e-10)
        k = int(np.argmax(gaps[min_speakers-1:max_speakers])) + min_speakers
        score = (p/n) / (gaps[k-1] + 1e-10)
        if best is None or score < best['score']:
            ranked = np.argsort(-gaps[min_speakers-1:max_speakers]) + min_speakers
            best = {'speakers': k, 'pruning': int(p), 'score': float(score), 'affinity': a,
                    'candidates': [{'speakers': int(c), 'eigengap': float(gaps[c-1])} for c in ranked[:4]]}
    return best


def cluster(affinity, k):
    return SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='cluster_qr', random_state=42).fit_predict(affinity)


def cluster_quality(x, labels):
    s = silhouette_samples(x, labels, metric='cosine')
    return {int(k): {'windows': int((labels == k).sum()), 'silhouette': float(s[labels == k].mean())} for k in np.unique(labels)}


def diarize_auto(x, min_speakers=2, max_speakers=12, min_silhouette=0.15):
    """Estimate the count, then drop groups that are mixtures rather than voices.

    Overlapping speech and noise tend to form a diffuse extra group spread over the
    whole meeting; its windows have a near-zero silhouette. Such a group lowers the
    count by one until every group is coherent or the minimum is reached.
    """
    est = estimate(x, min_speakers, max_speakers)
    k = est['speakers']
    while True:
        labels = cluster(est['affinity'], k)
        quality = cluster_quality(x, labels)
        worst = min(q['silhouette'] for q in quality.values())
        if worst >= min_silhouette or k <= min_speakers:
            break
        print(f'  {k} groups: one group is incoherent (silhouette {worst:.2f}), trying {k-1}', flush=True)
        k -= 1
    info = {'estimated': est['speakers'], 'chosen': k, 'pruning': est['pruning'], 'eigengap_candidates': est['candidates'],
            'min_silhouette': min_silhouette, 'groups': list(quality.values())}
    return labels, info
