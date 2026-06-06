from __future__ import annotations

from dataclasses import dataclass

from tqdm import tqdm

from wildlife_reid.config import AppConfig
from wildlife_reid.data.catalog import filter_records, load_records
from wildlife_reid.retrieval import RetrievalService


@dataclass
class EvaluationResult:
    total: int
    top1_correct: int
    top5_correct: int

    @property
    def top1_accuracy(self) -> float:
        return self.top1_correct / self.total if self.total else 0.0

    @property
    def top5_accuracy(self) -> float:
        return self.top5_correct / self.total if self.total else 0.0


def evaluate_retrieval(config: AppConfig, service: RetrievalService | None = None) -> EvaluationResult:
    records = load_records(config)
    queries = [record for record in filter_records(records, config.dataset.query_split) if record.exists()]
    if not queries:
        raise FileNotFoundError("No query images found on disk for evaluation.")

    service = service or RetrievalService(config)
    if not service.index.items:
        service.build_gallery_index(records)

    top1_correct = 0
    top5_correct = 0

    for query in tqdm(queries, desc="Evaluating retrieval"):
        result = service.search(query.path, top_k=max(config.index.top_k, 5))
        predicted = [match["identity"] for match in result.matches]
        if predicted and predicted[0] == query.identity:
            top1_correct += 1
        if query.identity in predicted[:5]:
            top5_correct += 1

    return EvaluationResult(total=len(queries), top1_correct=top1_correct, top5_correct=top5_correct)
