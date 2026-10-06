# -*- coding: utf-8 -*-
"""
分析 chunk 切分统计信息

输出：
- chunk 数量
- 平均大小（字符数）
- 大小分布（最小、最大、中位数）
- 按文件统计
"""
import statistics

import config
from chunker import chunk_documents
from loader import load_documents


def analyze_chunks():
    """分析 chunk 统计信息"""
    print("加载语料并切分...")
    docs = load_documents(config.INTERVIEW_DIR)
    chunks = chunk_documents(docs)

    print(f"\n{'='*60}")
    print("Chunk 切分统计")
    print(f"{'='*60}")

    # 1. 整体统计
    chunk_sizes = [len(doc.page_content) for doc in chunks]

    print(f"\n【整体统计】")
    print(f"  总 chunk 数: {len(chunks)}")
    print(f"  平均大小: {statistics.mean(chunk_sizes):.0f} 字符")
    print(f"  中位数: {statistics.median(chunk_sizes):.0f} 字符")
    print(f"  最小: {min(chunk_sizes)} 字符")
    print(f"  最大: {max(chunk_sizes)} 字符")
    print(f"  标准差: {statistics.stdev(chunk_sizes):.0f} 字符")

    # 2. 大小分布
    print(f"\n【大小分布】")
    buckets = {
        "0-200字": 0,
        "200-400字": 0,
        "400-600字": 0,
        "600-800字": 0,
        "800字以上": 0,
    }

    for size in chunk_sizes:
        if size < 200:
            buckets["0-200字"] += 1
        elif size < 400:
            buckets["200-400字"] += 1
        elif size < 600:
            buckets["400-600字"] += 1
        elif size < 800:
            buckets["600-800字"] += 1
        else:
            buckets["800字以上"] += 1

    for bucket, count in buckets.items():
        percentage = count / len(chunks) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {bucket:<12} {count:>4} ({percentage:>5.1f}%) {bar}")

    # 3. 按文件统计（取前10个）
    print(f"\n【按文件统计】（前10个文件）")
    file_stats = {}
    for doc in chunks:
        source = doc.metadata["source"]
        if source not in file_stats:
            file_stats[source] = []
        file_stats[source].append(len(doc.page_content))

    sorted_files = sorted(file_stats.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"  {'文件名':<40} {'chunk数':<10} {'平均大小'}")
    print(f"  {'-'*60}")
    for source, sizes in sorted_files[:10]:
        avg_size = statistics.mean(sizes)
        print(f"  {source:<40} {len(sizes):<10} {avg_size:.0f} 字符")

    # 4. 切分策略回顾
    print(f"\n【切分策略】")
    print(f"  MAX_CHARS: {config.MAX_CHARS} 字符（单个chunk上限）")
    print(f"  MIN_CHARS: {config.MIN_CHARS} 字符（太短的块合并）")
    print(f"  策略: 按 Markdown 标题层级切分 + 代码块保护 + 段落二次切分")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    analyze_chunks()
