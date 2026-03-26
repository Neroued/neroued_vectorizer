# 参考文献索引

本目录收录调研过程中发现的重要文献，每篇一个 Markdown 文件。索引随检索进展动态更新。

## 文献列表

### 核心文献（深度分析）

| 编号 | 文献 | 文件 | 来源 | 年份 | 关键技术 |
|------|------|------|------|------|----------|
| L01 | Image Vectorization with Depth | [vectorization-with-depth.md](vectorization-with-depth.md) | arXiv | 2024 | 深度排序能量、形状凸化、层叠 SVG |
| L02 | Perception-Driven Semi-Structured Boundary Vectorization | [perception-driven-vectorization.md](perception-driven-vectorization.md) | SIGGRAPH (ACM TOG) | 2018 | 感知角点检测、学习度量、曲线拟合 |
| L03 | VTracer | [vtracer-analysis.md](vtracer-analysis.md) | GitHub (开源) | 持续更新 | 层叠策略、分层聚类、O(n) 路径简化 |
| L04 | LIVE: Layer-wise Image Vectorization | [live-vectorization.md](live-vectorization.md) | CVPR 2022 (Oral) | 2022 | DiffVG 可微分栅格化、分层路径优化 |
| L05 | LIVSS: Layered Image Vectorization via Semantic Simplification | [livss-vectorization.md](livss-vectorization.md) | CVPR 2025 | 2024 | 渐进式语义简化、分层矢量化 |
| L06 | Depixelizing Pixel Art | [depixelizing-pixel-art.md](depixelizing-pixel-art.md) | SIGGRAPH (ACM TOG) | 2011 | 像素连通性、样条拟合、T-junction 处理 |

### 工具与实现分析

| 编号 | 文献 | 文件 | 来源 | 关键技术 |
|------|------|------|------|----------|
| T01 | Potrace & color_trace | [potrace-and-color-trace.md](potrace-and-color-trace.md) | 开源 | 位图追踪、逐层着色、--stack 选项 |
| T02 | Adobe 相关专利 | [adobe-patents.md](adobe-patents.md) | USPTO | 内容感知矢量路径、位图多边形转换 |
| T03 | kurbo Bézier Path Simplification | [kurbo-bezier-simplify.md](kurbo-bezier-simplify.md) | Raph Levien blog / Rust crate | Green 定理面积度量、近最优贝塞尔拟合 |

### 综述与参考

| 编号 | 文献 | 文件 | 来源 | 年份 | 关键技术 |
|------|------|------|------|------|----------|
| S01 | Forty Years of Color Quantization: A Modern Survey | [color-quantization-survey.md](color-quantization-survey.md) | Springer (AI Review) | 2023 | 量化算法综述、色彩空间、质量评估 |
| S02 | Printing Trapping Techniques | [printing-trapping.md](printing-trapping.md) | 行业标准 | — | 色彩重叠、密度方向规则 |

> 后续检索发现的文献将持续添加到此列表。
