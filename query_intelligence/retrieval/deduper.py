from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Deduper:
    def __init__(self, similarity_threshold: float = 0.85, max_per_source: int = 3) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_per_source = max_per_source

    def dedupe(self, candidates: list[dict]) -> tuple[list[dict], list[dict]]:
        if not candidates:
            return [], []
        if len(candidates) == 1:
            return candidates, [
                {
                    "group_id": f"group_{candidates[0]['evidence_id']}",
                    "group_type": "single",
                    "members": [candidates[0]["evidence_id"]],
                }
            ]

        texts = [
            f"{c.get('title', '')} {c.get('summary', '') or ''}"
            for c in candidates
        ]
        non_empty = [i for i, t in enumerate(texts) if t.strip()]
        if len(non_empty) < 2:
            groups = [
                {
                    "group_id": f"group_{c['evidence_id']}",
                    "group_type": "single",
                    "members": [c["evidence_id"]],
                }
                for c in candidates
            ]
            return candidates, groups

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)

        deduped: list[dict] = []
        groups: list[dict] = []
        removed: set[int] = set()
        source_counts: dict[str, int] = {}

        for i, candidate in enumerate(candidates):
            if i in removed:
                continue
            source_type = candidate.get("source_type", "")
            if source_counts.get(source_type, 0) >= self.max_per_source:
                removed.add(i)
                continue
            source_counts[source_type] = source_counts.get(source_type, 0) + 1

            group_members = [candidate["evidence_id"]]
            for j in range(i + 1, len(candidates)):
                if j in removed:
                    continue
                if sim_matrix[i, j] >= self.similarity_threshold:
                    removed.add(j)
                    group_members.append(candidates[j]["evidence_id"])

            deduped.append(candidate)
            groups.append(
                {
                    "group_id": f"group_{candidate['evidence_id']}",
                    "group_type": "news_cluster" if len(group_members) > 1 else "single",
                    "members": group_members,
                }
            )

        return deduped, groups
