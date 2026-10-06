# -*- coding: utf-8 -*-
"""
RAG 评估模块 - 检索质量评估

评估指标：
1. Hit Rate@K：前K个结果中是否至少命中一个相关文档
2. MRR (Mean Reciprocal Rank)：第一个相关文档的倒数排名
3. Precision@K：前K个结果中相关文档的占比
4. Recall@K：前K个结果召回的相关文档占总相关文档的比例

使用示例：
    python eval_retrieval.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import config


@dataclass
class EvalQuestion:
    """单个评估问题"""
    question: str
    ground_truth_docs: list[str]  # 相关文档的 source（文件名）


@dataclass
class RetrievalMetrics:
    """检索质量指标"""
    hit_rate: float  # 命中率
    mrr: float  # Mean Reciprocal Rank
    precision: float  # 精确度
    recall: float  # 召回率

    def __repr__(self):
        return (
            f"RetrievalMetrics(hit_rate={self.hit_rate:.2%}, "
            f"mrr={self.mrr:.4f}, "
            f"precision={self.precision:.2%}, "
            f"recall={self.recall:.2%})"
        )


def load_eval_dataset(path: Path | str) -> list[EvalQuestion]:
    """加载评估数据集

    数据集格式（JSON）：
    [
        {
            "question": "HNSW 是什么",
            "ground_truth_docs": ["12-补充-检索与RAG.md"]
        },
        ...
    ]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"评估数据集不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        EvalQuestion(
            question=item["question"],
            ground_truth_docs=item["ground_truth_docs"],
        )
        for item in data
    ]


def compute_retrieval_metrics(
    retrieved_docs: list[str],
    ground_truth_docs: list[str],
) -> RetrievalMetrics:
    """计算检索质量指标

    参数:
        retrieved_docs: 检索返回的文档 source 列表（按相关性排序）
        ground_truth_docs: 标准答案文档的 source 列表

    返回:
        RetrievalMetrics 对象
    """
    if not ground_truth_docs:
        raise ValueError("ground_truth_docs 不能为空")

    gt_set = set(ground_truth_docs)

    # 1. Hit Rate：是否至少命中一个相关文档
    hit = any(doc in gt_set for doc in retrieved_docs)
    hit_rate = 1.0 if hit else 0.0

    # 2. MRR (Mean Reciprocal Rank)：第一个相关文档的倒数排名
    mrr = 0.0
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in gt_set:
            mrr = 1.0 / rank
            break

    # 3. Precision@K：召回结果中相关文档的占比
    if retrieved_docs:
        hits = sum(1 for doc in retrieved_docs if doc in gt_set)
        precision = hits / len(retrieved_docs)
    else:
        precision = 0.0

    # 4. Recall@K：相关文档中被召回的占比
    hits = sum(1 for doc in retrieved_docs if doc in gt_set)
    recall = hits / len(gt_set)

    return RetrievalMetrics(
        hit_rate=hit_rate,
        mrr=mrr,
        precision=precision,
        recall=recall,
    )


def evaluate_single_query(
    question: str,
    retrieved_docs: list[str],
    ground_truth_docs: list[str],
) -> tuple[str, list[str], RetrievalMetrics]:
    """评估单个查询

    返回: (问题, 检索到的文档, 指标)
    """
    metrics = compute_retrieval_metrics(retrieved_docs, ground_truth_docs)
    return question, retrieved_docs, metrics


def print_metrics_summary(all_metrics: list[RetrievalMetrics], method_name: str):
    """打印平均指标摘要"""
    if not all_metrics:
        return

    n = len(all_metrics)
    avg_hit_rate = sum(m.hit_rate for m in all_metrics) / n
    avg_mrr = sum(m.mrr for m in all_metrics) / n
    avg_precision = sum(m.precision for m in all_metrics) / n
    avg_recall = sum(m.recall for m in all_metrics) / n

    print(f"\n{'='*60}")
    print(f"方法: {method_name}")
    print(f"问题数: {n}")
    print(f"{'='*60}")
    print(f"Hit Rate@{config.TOP_K}:  {avg_hit_rate:.2%}  (至少命中一个相关文档)")
    print(f"MRR:          {avg_mrr:.4f}  (相关文档的平均倒数排名)")
    print(f"Precision@{config.TOP_K}: {avg_precision:.2%}  (检索结果中相关文档占比)")
    print(f"Recall@{config.TOP_K}:    {avg_recall:.2%}  (相关文档被召回的比例)")
    print(f"{'='*60}\n")

    return {
        "method": method_name,
        "total_questions": n,
        "avg_hit_rate": round(avg_hit_rate, 4),
        "avg_mrr": round(avg_mrr, 4),
        "avg_precision": round(avg_precision, 4),
        "avg_recall": round(avg_recall, 4),
    }


def save_eval_report(results: list[dict], summary: dict, output_path: Path | str):
    """保存评估报告到 JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "summary": summary,
        "details": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[评估报告] 已保存到 {output_path}")


if __name__ == "__main__":
    # 简单测试
    print("测试评估指标计算...")

    # 测试用例 1：完美召回
    retrieved = ["12-补充-检索与RAG.md", "03-AgentLoop.md"]
    ground_truth = ["12-补充-检索与RAG.md"]
    metrics = compute_retrieval_metrics(retrieved, ground_truth)
    print(f"测试1 - 完美召回: {metrics}")

    # 测试用例 2：第二个位置命中
    retrieved = ["03-AgentLoop.md", "12-补充-检索与RAG.md"]
    ground_truth = ["12-补充-检索与RAG.md"]
    metrics = compute_retrieval_metrics(retrieved, ground_truth)
    print(f"测试2 - 第二位命中: {metrics}")

    # 测试用例 3：没有命中
    retrieved = ["03-AgentLoop.md", "02-工具系统.md"]
    ground_truth = ["12-补充-检索与RAG.md"]
    metrics = compute_retrieval_metrics(retrieved, ground_truth)
    print(f"测试3 - 未命中: {metrics}")
