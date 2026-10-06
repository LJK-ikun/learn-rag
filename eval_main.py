# -*- coding: utf-8 -*-
"""
RAG 检索质量评估 - 主程序

对比三种检索方法：
1. 纯向量检索（Vector Only）
2. 纯 BM25 检索（BM25 Only）
3. 混合检索（Hybrid: Vector + BM25 + RRF）

运行方式：
    python eval_main.py
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

import config
from chunker import chunk_documents
from evaluator import (
    compute_retrieval_metrics,
    load_eval_dataset,
    print_metrics_summary,
    save_eval_report,
)
from loader import load_documents
from store import load_or_build_store, search


def vector_only_search(vector_store, query: str, k: int) -> list[Document]:
    """纯向量检索"""
    results = vector_store.similarity_search(query, k=k)
    return results


def bm25_only_search(bm25, chunks: list[Document], query: str, k: int) -> list[Document]:
    """纯 BM25 检索"""
    from store import _tokenize

    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # 按分数排序，取前 k 个
    ranked_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:k]

    return [chunks[i] for i in ranked_indices]


def hybrid_search(vector_store, bm25, chunks: list[Document], query: str, k: int) -> list[Document]:
    """混合检索（当前 store.py 中的实现）"""
    results_with_scores = search(vector_store, bm25, chunks, query, k=k)
    return [doc for doc, score in results_with_scores]


def evaluate_method(
    method_name: str,
    search_fn,
    eval_questions: list,
    k: int = None,
) -> tuple[list[dict], dict]:
    """评估单个检索方法

    参数:
        method_name: 方法名称
        search_fn: 检索函数，接受 query 参数，返回 list[Document]
        eval_questions: 评估问题列表
        k: 返回前 k 个结果

    返回:
        (详细结果列表, 摘要统计)
    """
    k = k or config.TOP_K

    print(f"\n开始评估: {method_name}")
    print(f"{'='*60}")

    results = []
    all_metrics = []

    for i, eq in enumerate(eval_questions, 1):
        # 执行检索
        retrieved_docs_objs = search_fn(eq.question)

        # 提取 source（文件名）
        retrieved_docs = [doc.metadata["source"] for doc in retrieved_docs_objs]

        # 计算指标
        metrics = compute_retrieval_metrics(retrieved_docs, eq.ground_truth_docs)
        all_metrics.append(metrics)

        # 记录详细结果
        results.append({
            "question": eq.question,
            "ground_truth_docs": eq.ground_truth_docs,
            "retrieved_docs": retrieved_docs,
            "hit": metrics.hit_rate > 0,
            "mrr": metrics.mrr,
            "precision": metrics.precision,
            "recall": metrics.recall,
        })

        # 打印进度
        hit_mark = "[OK]" if metrics.hit_rate > 0 else "[X]"
        print(f"  [{i:2d}/{len(eval_questions)}] {hit_mark} {eq.question[:40]:<40} MRR={metrics.mrr:.3f}")

    # 打印摘要
    summary = print_metrics_summary(all_metrics, method_name)

    return results, summary


def main():
    """评估主流程"""
    print("\n" + "="*60)
    print("RAG 检索质量评估")
    print("="*60)

    # 1. 加载语料和索引
    print("\n[1/4] 加载语料和构建索引...")
    docs = load_documents(config.INTERVIEW_DIR)
    chunks = chunk_documents(docs)
    vector_store, bm25 = load_or_build_store(chunks)
    print(f"  → 加载完成：共 {len(chunks)} 个 chunk")

    # 2. 加载评估数据集
    print("\n[2/4] 加载评估数据集...")
    dataset_path = Path("eval/dataset.json")
    eval_questions = load_eval_dataset(dataset_path)
    print(f"  → 加载完成：共 {len(eval_questions)} 个评估问题")

    # 3. 评估三种方法
    print("\n[3/4] 开始评估...")

    k = config.TOP_K
    all_results = {}
    all_summaries = {}

    # 方法 1：纯向量检索
    results_vector, summary_vector = evaluate_method(
        "Vector Only (纯向量)",
        lambda q: vector_only_search(vector_store, q, k),
        eval_questions,
        k,
    )
    all_results["vector_only"] = results_vector
    all_summaries["vector_only"] = summary_vector

    # 方法 2：纯 BM25 检索
    results_bm25, summary_bm25 = evaluate_method(
        "BM25 Only (纯关键词)",
        lambda q: bm25_only_search(bm25, chunks, q, k),
        eval_questions,
        k,
    )
    all_results["bm25_only"] = results_bm25
    all_summaries["bm25_only"] = summary_bm25

    # 方法 3：混合检索
    results_hybrid, summary_hybrid = evaluate_method(
        "Hybrid (混合检索: Vector + BM25 + RRF)",
        lambda q: hybrid_search(vector_store, bm25, chunks, q, k),
        eval_questions,
        k,
    )
    all_results["hybrid"] = results_hybrid
    all_summaries["hybrid"] = summary_hybrid

    # 4. 生成对比报告
    print("\n[4/4] 生成对比报告...")

    # 打印对比表格
    print(f"\n{'='*80}")
    print("检索方法对比")
    print(f"{'='*80}")
    print(f"{'方法':<30} {'Hit Rate':<12} {'MRR':<12} {'Precision':<12} {'Recall':<12}")
    print(f"{'-'*80}")

    for method_key in ["vector_only", "bm25_only", "hybrid"]:
        s = all_summaries[method_key]
        print(
            f"{s['method']:<30} "
            f"{s['avg_hit_rate']:<12.2%} "
            f"{s['avg_mrr']:<12.4f} "
            f"{s['avg_precision']:<12.2%} "
            f"{s['avg_recall']:<12.2%}"
        )

    print(f"{'='*80}\n")

    # 保存详细报告
    report_dir = Path("eval/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    for method_key, results in all_results.items():
        output_path = report_dir / f"{method_key}_report.json"
        save_eval_report(results, all_summaries[method_key], output_path)

    # 保存对比报告
    comparison_report = {
        "comparison": all_summaries,
        "config": {
            "top_k": k,
            "fetch_k": config.FETCH_K,
            "embedding_model": config.EMBEDDING_MODEL,
            "total_chunks": len(chunks),
            "total_questions": len(eval_questions),
        }
    }

    import json
    comparison_path = report_dir / "comparison_report.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, ensure_ascii=False, indent=2)
    print(f"[对比报告] 已保存到 {comparison_path}")

    print("\n[完成] 评估完成！")


if __name__ == "__main__":
    main()
