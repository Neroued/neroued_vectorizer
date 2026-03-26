# V2 矢量化管线调研报告

本目录包含 neroued_vectorizer V2 管线架构的系统化调研报告，作为团队内部技术决策参考。

## 目录

| 文件 | 内容 |
|------|------|
| [00-executive-summary.md](00-executive-summary.md) | 总报告（执行摘要）：问题定义、核心结论、推荐方案 |
| [01-current-architecture-analysis.md](01-current-architecture-analysis.md) | 现有 V1 架构深度诊断 |
| [02-literature/](02-literature/) | 参考文献深度分析 |
| [03-algorithm-deep-dive/](03-algorithm-deep-dive/) | 核心算法实现调研 |
| [04-implementation-plan.md](04-implementation-plan.md) | 初步实施计划 |
| [05-risk-assessment.md](05-risk-assessment.md) | 风险评估与缓解 |

## 文献索引

详见 [02-literature/README.md](02-literature/README.md)。

## 算法专题索引

| 专题 | 文件 |
|------|------|
| 层叠 vs 剪切模型 | [stacking-vs-cutout.md](03-algorithm-deep-dive/stacking-vs-cutout.md) |
| 深度排序算法 | [depth-ordering.md](03-algorithm-deep-dive/depth-ordering.md) |
| 形状延伸 / 凸化 | [shape-extension.md](03-algorithm-deep-dive/shape-extension.md) |
| 颜色量化对比 | [color-quantization.md](03-algorithm-deep-dive/color-quantization.md) |
| 路径优化与合并 | [path-optimization.md](03-algorithm-deep-dive/path-optimization.md) |

## 调研方法论

本调研采用迭代式检索驱动：已知材料只是起点，每个专题撰写前做专项检索，发现新论文/专利/开源实现后纳入分析。每篇文献分析末尾标注引用链线索，用于持续扩展材料范围。
