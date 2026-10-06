# -*- coding: utf-8 -*-
"""
测试具体的检索案例 - 可视化检索结果

用途：
1. 查看具体问题的检索结果（前5条）
2. 对比三种方法的差异
3. 分析失败案例的原因
"""
from pathlib import Path

import config
from chunker import chunk_documents
from evaluator import load_eval_dataset
from loader import load_documents
from store import load_or_build_store, _tokenize


def clean_text_for_print(text: str) -> str:
    """清理文本中无法在 Windows 终端显示的字符"""
    # 移除或替换常见的 Unicode 特殊字符
    replacements = {
        '⭐': '*',
        '✓': '[OK]',
        '✗': '[X]',
        '→': '->',
        '←': '<-',
        '⚠': '[!]',
        '📝': '[Note]',
        '🔍': '[Search]',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 过滤掉其他无法编码的字符
    try:
        text.encode('gbk')
        return text
    except UnicodeEncodeError:
        # 如果还有问题，只保留 ASCII 和常见中文字符
        return ''.join(c if ord(c) < 0x10000 else '?' for c in text)


def test_single_query(query: str, vector_store, bm25, chunks):
    """测试单个查询，展示三种方法的检索结果"""
    print(f"\n{'='*80}")
    print(f"问题: {query}")
    print(f"{'='*80}")

    k = config.TOP_K

    # 1. 纯向量检索
    print(f"\n【方法1: 纯向量检索】")
    vector_results = vector_store.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(vector_results, 1):
        source = clean_text_for_print(doc.metadata["source"])
        title = clean_text_for_print(doc.metadata.get("title_path", ""))
        content_preview = clean_text_for_print(doc.page_content[:80].replace("\n", " "))
        print(f"  {i}. [{source}]")
        print(f"     标题: {title}")
        print(f"     分数: {score:.4f}")
        print(f"     内容: {content_preview}...")

    # 2. 纯 BM25 检索
    print(f"\n【方法2: 纯BM25检索】")
    query_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:k]

    for i, idx in enumerate(ranked_indices, 1):
        doc = chunks[idx]
        score = bm25_scores[idx]
        source = clean_text_for_print(doc.metadata["source"])
        title = clean_text_for_print(doc.metadata.get("title_path", ""))
        content_preview = clean_text_for_print(doc.page_content[:80].replace("\n", " "))
        print(f"  {i}. [{source}]")
        print(f"     标题: {title}")
        print(f"     分数: {score:.4f}")
        print(f"     内容: {content_preview}...")

    # 3. 混合检索
    print(f"\n【方法3: 混合检索 (Vector + BM25 + RRF)】")
    from store import search
    hybrid_results = search(vector_store, bm25, chunks, query, k=k)

    for i, (doc, score) in enumerate(hybrid_results, 1):
        source = clean_text_for_print(doc.metadata["source"])
        title = clean_text_for_print(doc.metadata.get("title_path", ""))
        content_preview = clean_text_for_print(doc.page_content[:80].replace("\n", " "))
        print(f"  {i}. [{source}]")
        print(f"     标题: {title}")
        print(f"     RRF分数: {score:.4f}")
        print(f"     内容: {content_preview}...")

    print(f"\n{'-'*80}\n")


def main():
    """测试主流程"""
    print("\n" + "="*80)
    print("RAG 检索效果测试 - 可视化检索结果")
    print("="*80)

    # 1. 加载索引
    print("\n[1/3] 加载索引...")
    docs = load_documents(config.INTERVIEW_DIR)
    chunks = chunk_documents(docs)
    vector_store, bm25 = load_or_build_store(chunks)
    print(f"  → 加载完成：共 {len(chunks)} 个 chunk")

    # 2. 加载评估数据集
    print("\n[2/3] 加载评估数据集...")
    dataset_path = Path("eval/dataset.json")
    eval_questions = load_eval_dataset(dataset_path)
    print(f"  → 加载完成：共 {len(eval_questions)} 个问题")

    # 3. 测试几个典型案例
    print("\n[3/3] 测试检索效果...\n")

    # 测试案例1: 成功案例 - 精确术语
    test_single_query(
        "HNSW 是什么",
        vector_store, bm25, chunks
    )

    # 测试案例2: 失败案例 - 泛化概念
    test_single_query(
        "Agent 中的工具系统怎么设计",
        vector_store, bm25, chunks
    )

    # 测试案例3: 语义改写
    test_single_query(
        "向量检索和关键词检索各有什么优缺点",
        vector_store, bm25, chunks
    )

    # 测试案例4: 用户自定义问题
    print(f"\n{'='*80}")
    print("你可以修改 test_retrieval.py 添加更多测试问题")
    print("或者运行: python test_retrieval.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
