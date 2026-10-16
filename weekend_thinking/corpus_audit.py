"""Deterministic corpus screening with exact candidate verification.

MinHash/LSH discovers likely near duplicates; full word-shingle Jaccard verifies
every proposed edge. This is a probabilistic candidate search, not an exhaustive
all-pairs proof. Connected components are formed before assigning corpus splits.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit
import numpy as np


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def hash_words(items: list[str]) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(items).encode(), digest_size=8).digest(), "little")


def shingle_hashes(items: list[str], n: int = 5) -> np.ndarray:
    if n < 1:
        raise ValueError("Positive shingle width required.")
    if not items:
        return np.empty(0, dtype=np.uint64)
    values = (hash_words(items[i:i+n]) for i in range(max(1, len(items)-n+1)))
    return np.unique(np.fromiter(values, dtype=np.uint64))


def page_identity(url: str, content_hash: str) -> str:
    parsed = urlsplit(url)
    # Query parameters often ARE the document ID (display.cgi?id=..., forum
    # showthread.php?t=..., wiki/index.php?title=...). Removing every query
    # conflates unrelated pages and can turn one overlap into thousands of
    # exclusions. Remove only named tracking fields and preserve other values.
    tracking = {"gclid", "fbclid", "msclkid", "dclid", "mc_cid", "mc_eid"}
    parameters = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                  if not key.lower().startswith("utm_") and key.lower() not in tracking]
    query = urlencode(sorted(parameters))
    group = parsed.netloc.lower() + parsed.path.rstrip("/")
    if query:
        group += "?" + query
    if not parsed.netloc:
        group = content_hash
    return hashlib.sha256(group.encode()).hexdigest()


class QuestionScreen:
    """Exact normalized 13-word screening and full shorter-question screening."""
    def __init__(self, questions: list[dict], n: int = 13, minimum_short_words: int = 5):
        if n < 3 or not 1 <= minimum_short_words < n:
            raise ValueError("Invalid question-screen widths.")
        self.n = n
        self.needles = defaultdict(set)
        self.short = defaultdict(list)
        self.counts = Counter()
        prepared = [(words(row["question"]), row.get("task", "unknown")) for row in questions]
        frequency = Counter(word for items, _ in prepared for word in set(items))
        for items, task in prepared:
            self.counts["questions"] += 1
            if len(items) >= n:
                for i in range(len(items)-n+1):
                    self.needles[hash_words(items[i:i+n])].add(task)
            elif len(items) >= minimum_short_words:
                anchor = min(set(items), key=lambda word: (frequency[word], word))
                self.short[anchor].append((" " + " ".join(items) + " ", task))
                self.counts["short_questions_screened_in_full"] += 1
            else:
                self.counts["questions_below_minimum_short_words"] += 1
        if not self.needles:
            raise ValueError("No benchmark n-grams: empty screening is not a pass.")

    def matches(self, items: list[str]) -> set[str]:
        tasks = set()
        for i in range(len(items)-self.n+1):
            tasks.update(self.needles.get(hash_words(items[i:i+self.n]), ()))
        anchors = set(items).intersection(self.short)
        if anchors:
            text = " " + " ".join(items) + " "
            for anchor in anchors:
                for pattern, task in self.short[anchor]:
                    if pattern in text:
                        tasks.add(task)
        return tasks


class NearDuplicateIndex:
    """64-permutation MinHash, 16x4 LSH, then exact set Jaccard >= threshold."""
    def __init__(self, threshold: float = 0.8):
        if not 0 < threshold <= 1:
            raise ValueError("Jaccard threshold must be in (0,1].")
        self.threshold = threshold
        rng = np.random.default_rng(20261016)
        self.prime = np.uint64(4294967291)
        self.a = rng.integers(1, int(self.prime), size=64, dtype=np.uint64)
        self.b = rng.integers(0, int(self.prime), size=64, dtype=np.uint64)
        self.bands = [defaultdict(list) for _ in range(16)]
        self.features = {}
        self.comparisons = self.candidate_pairs = self.max_candidates = 0

    def signature(self, features: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            raise ValueError("Cannot index empty shingles.")
        signature = np.full(64, self.prime, dtype=np.uint64)
        # Bound temporary memory for unusually long documents.
        for offset in range(0, len(features), 2048):
            values = (features[offset:offset+2048] % self.prime)[:, None]
            hashes = (values * self.a + self.b) % self.prime
            signature = np.minimum(signature, hashes.min(axis=0))
        return signature.astype(np.uint32)

    def add(self, index: int, features: np.ndarray) -> list[tuple[int, float]]:
        if index in self.features:
            raise ValueError("Index already present.")
        signature = self.signature(features)
        keys = [signature[i*4:i*4+4].tobytes() for i in range(16)]
        candidates = set()
        for band, key in zip(self.bands, keys):
            candidates.update(band.get(key, ()))
        self.candidate_pairs += len(candidates)
        self.max_candidates = max(self.max_candidates, len(candidates))
        matches = []
        for other in sorted(candidates):
            previous = self.features[other]
            if min(len(previous), len(features)) / max(len(previous), len(features)) < self.threshold:
                continue
            self.comparisons += 1
            intersection = len(np.intersect1d(previous, features, assume_unique=True))
            similarity = intersection / (len(previous) + len(features) - intersection)
            if similarity >= self.threshold:
                matches.append((other, similarity))
        self.features[index] = features
        for band, key in zip(self.bands, keys):
            band[key].append(index)
        return matches


class Components:
    """Union-find preserving the first source index as deterministic root."""
    def __init__(self):
        self.parents = []

    def add(self) -> int:
        index = len(self.parents)
        self.parents.append(index)
        return index

    def find(self, index: int) -> int:
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def union(self, first: int, second: int) -> None:
        left, right = self.find(first), self.find(second)
        self.parents[max(left, right)] = min(left, right)
