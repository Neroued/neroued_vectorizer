# 路径优化与形状合并

## 1. 问题定义与约束

**问题 A（路径简化 / 优化）**：在曲线拟合阶段已得到分段三次贝塞尔（或折线）之后，希望在**可控几何误差**下减少段数、降低 SVG 路径数据量，并消除冗余控制点（例如近直线仍用完整三次表示）。

**问题 B（同色形状合并）**：在输出 SVG 前，将多个**填充色相同**且**在视觉与层叠语义上可合并**的 `VectorizedShape` 合并为更少的路径元素，减少 DOM 节点与重复属性，同时**不引入错误的遮挡关系**。

**约束**：

- 路径优化必须在用户可配置的**误差预算**内工作；默认应**保守**，避免肉眼可见形变。
- 合并策略须与**绘制顺序（z-order / 深度）**一致：合并后单一路径内的子路径顺序须与原视觉一致。
- 实现复杂度需与管线其余阶段匹配；**两遍式**（轻量后处理 + 可选进阶合并）优于单次巨型全局优化。
- 与现有 **Schneider 拟合**、**Potrace 追踪** 输出兼容：优化是**后处理**，不宜破坏闭合性、绕向与端点连续性（除非显式重参数化）。

---

## 2. 本专题检索记录

### 检索关键词（keywords searched）

- `bezier simplification`
- `SVG path optimization`
- `SVGO mergePaths`
- `kurbo simplify`
- `shape merging z-order`

### 检索到的材料与结论要点

| 方向 | 材料 / 工具 | 要点 |
|------|----------------|------|
| 工业 SVG 压缩 | SVGO 插件（`mergePaths`、`convertPathData` 等） | 舍入坐标、扁平三次改直线、同向线段合并；**工程实用**，误差模型不统一。 |
| 高质量贝塞尔简化 | Raph Levien，kurbo（2023 博客 + Rust 库） | **Green 定理** 面积/矩度量；`ParamCurveFit`；近最优三次拟合；可 **重采样 + 重拟合** 简化已有路径。 |
| 经典拟合 | Schneider 算法（本库 V1 已用） | 逐点误差、一次通过；**无**全局段间合并。 |
| 多边形简化 | Douglas–Peucker | 对**控制点序列**直接套用会**过度**简化曲线几何。 |
| 形状合并语义 | SVG 规范、常见编辑器导出 | `mergePaths` 通常要求 **DOM 相邻** 且属性一致；**z-order** 决定实际遮挡。 |

---

## 3. 候选方案对比表（总览）

| 层级 | 方案 | 作用 | 主要风险 |
|------|------|------|----------|
| **路径** | 近线性段合并 | 三次 → 直线或等价退化表示 | 容差过大时拐角变钝 |
| **路径** | 相邻段最小二乘重拟合 | 两段并一段，减段数 | 实现与误差度量需仔细设计 |
| **路径** | kurbo 式重采样 + Green 矩拟合 | 高质量、可解析误差 | 需移植或 FFI，与现有栈集成成本 |
| **路径** | SVGO 式启发式 | 体量下降明显 | 与「数学误差」不对齐 |
| **形状** | V1：仅相邻同色 + bbox 不重叠 | 安全 | **过于保守**，合并率低 |
| **形状** | z-order 安全合并（同色、同深度层） | 平衡体量与正确性 | 需稳定 **深度层** 定义 |
| **形状** | 激进 bbox 不重叠合并 | 合并率更高 | **跨深度** 验证复杂，易出错 |

以下分专题展开路径侧与形状侧，并给出推荐与伪代码。

---

## 专题 A：路径简化 / 优化（Path Simplification / Optimization）

### A.1 各方法说明

| 方法 | 摘要 |
|------|------|
| **V1 当前** | **Schneider 拟合** + `merge_segment_tolerance` 驱动的**近线性段合并**（相邻段在容差内视为可合并为更简洁表示）。**一次性**管线，**无**专门的全路径后优化Pass。 |
| **近线性段合并（后处理 Pass）** | 若控制点 **p1、p2** 到弦 **p0–p3** 的最大距离 < **ε_linear**，将三次贝塞尔**替换为直线**（或标准参数化下等价的三次退化）。实现简单、副作用面小。 |
| **相邻段合并** | 若**连续两段**三次曲线可用**单条三次**在误差界内逼近，则合并；内部控制点可用 **Schneider 类最小二乘** 在合并后的采样点集上重拟合。 |
| **kurbo 思路（Raph Levien，2023）** | 用 **Green 定理** 计算面积/矩偏差；**ParamCurveFit** 抽象源曲线；**近最优** 贝塞尔拟合。对**已有**贝塞尔路径可先 **重采样** 再 **重拟合**，达到简化效果。 |
| **SVGO 思路** | 坐标舍入、扁平三次改直线、**同方向**线段合并等；偏**工程压缩**，**非**严格一致的几何误差模型。 |
| **Douglas–Peucker 作用于控制点** | 把控制点当折线顶点删除；对真实曲线形状**过于激进**，一般不单独用于贝塞尔路径。 |

### A.2 路径方法对比表

| 方法 | 压缩比 | 质量 | 复杂度 | 实现难度 |
|------|--------|------|--------|----------|
| V1（Schneider + 近线性合并） | 中 | 高（拟合阶段已控误差） | 中（管线内） | 已具备 |
| 近线性后处理 Pass | 中–高（图依赖） | 高（阈值可控） | **低** | **低** |
| 相邻段最小二乘合并 | 高 | 高（依赖误差与采样） | 中–高 | 中 |
| kurbo / Green 矩 | 高 | **很高** | 高（解析矩、优化） | 高（Rust 移植或新实现） |
| SVGO 式启发式 | **很高** | 中–高（舍入可见） | 低 | 低 |
| Douglas–Peucker（控制点） | 很高 | **低–中**（易损曲率） | 低 | 低 |

### A.3 推荐：两遍路径优化

1. **Pass 1：近线性合并** — 简单、安全、与现有容差语义一致。  
2. **Pass 2：相邻段重拟合** — 中等复杂度，**压缩比**通常明显优于仅 Pass 1。

### A.4 伪代码：两遍路径优化

```
Pass 1: Near-linear merge
  for each contour:
    for each segment s (cubic Bezier p0,p1,p2,p3):
      if max_distance(p1, p2, to chord line(p0, p3)) < epsilon_linear:
        mark as linear
        // 可选：写回为标准退化三次，便于下游统一
        p1 = p0 + (1/3) * (p3 - p0)
        p2 = p0 + (2/3) * (p3 - p0)

Pass 2: Adjacent segment merge
  for each contour:
    i = 0
    while i < segments.size() - 1:
      merged = try_merge_cubic_pair(segments[i], segments[i+1], error_threshold)
      if merged is not empty:
        replace segments[i..i+1] with merged  // 新的一段三次
        // 不增加 i：继续尝试与下一段合并
      else:
        i++
```

**`try_merge_cubic_pair` 要点**：在 **p0→p3（第一段）** 与 **p3→p6（第二段）** 共享端点处，对合并区间上的曲线采样点集执行 **Schneider 式或最小二乘单三次拟合**；若最大误差或积分型误差（可选：面积偏差）小于阈值则接受。

---

## 专题 B：同色形状合并（Same-Color Shape Merging）

### B.1 各方法说明

| 方法 | 摘要 |
|------|------|
| **V1 当前：`MergeAdjacentSameColorShapes`** | 在**已排序**的形状序列上，仅合并**连续同色**填充形状，且要求各形状 **bbox 互不重叠**；遇重叠即终止当前 run。**保守**，合并机会少。 |
| **SVGO `mergePaths`** | 合并 **DOM 相邻**、**填充等属性相同** 的 `<path>`；**不**解决任意 z-order 下的全局合并。 |
| **Z-order 安全合并** | 在层叠模型中，**同色**且处于**同一深度层**的所有形状，若在该层内无其他颜色插入其间，可合并为**单一路径**，内含多个子路径（`M…Z M…Z`）。 |
| **激进合并** | 即使深度不同，若**不存在**异色形状与待合并集合的 **bbox 发生遮挡意义上的相交**，则尝试合并；验证与实现复杂度更高。 |

### B.2 形状合并策略对比表

| 方法 | 合并率 | 正确性风险 | 实现难度 |
|------|--------|------------|----------|
| V1 相邻 + bbox 不重叠 | 低 | **很低** | 低 |
| SVGO 式（仅 DOM 相邻） | 中（视导出顺序） | 低（若顺序与绘制一致） | 低 |
| **Z-order 安全、同深度** | **中–高** | **低**（定义清晰时） | 中 |
| 激进 bbox / 遮挡检验 | 高 | **高** | 高 |

### B.3 推荐：同深度层上的 z-order 安全合并

在已有**线性绘制序**与**深度层**（或等价分组）的前提下，按 **(颜色, 深度层)** 分组；组内形状在 z-order 上**连续且无其他颜色穿插**时，合并为单个 `VectorizedShape`，轮廓表拼接。

### B.4 伪代码：按颜色与深度分组合并

```
group shapes by (color_key, depth_level)
for each group G:
  if G.shapes.size() <= 1:
    continue
  // 可选：检查 G 内形状在全局序中是否「连续且无异色穿插」
  if not z_order_contiguous_same_color(G):
    continue
  merged = new VectorizedShape
  merged.color = G.representative_color
  merged.contours = concatenate(all contours from shapes in G in z-order)
  merged.area = sum(shape.area for shape in G)  // 或按需仅保留最大块面积等策略
  replace G with single merged shape in output list
```

---

## 4. 推荐方案详细描述（汇总）

**路径侧**：采用 **Pass 1 近线性** + **Pass 2 相邻三次合并**（失败则保持两段）。与 V1 兼容：`merge_segment_tolerance` 可映射为 **ε_linear** 或与 Pass 2 的 **error_threshold** 建立比例关系（需调参）。

**形状侧**：采用 **(颜色, 深度层)** 下的 **z-order 安全合并**，替代或扩展当前仅「相邻 + bbox 不重叠」的策略；**不**默认启用激进跨层合并，除非后续有完整遮挡证明与测试。

---

## 5. 关键实现细节

### 5.1 合并判据与误差度量

- **近线性**：**p1、p2** 到线段 **p0p3** 的**有符号距离**的最大绝对值（或 Hausdorff 距离上界）与 **ε_linear** 比较。  
- **相邻段合并**：对采样点 **{q_k}** 到候选单三次 **B(t)** 的 **最大欧氏距离**，或 **均方根误差**；进阶可采用 **kurbo 式面积偏差** 作为一致目标（实现成本更高）。  
- **阈值**：`ε_linear` 可与像素半宽、视图缩放挂钩；Pass 2 通常取 **≤ Pass 1** 或同量级，避免「先合并再劣化」。

### 5.2 闭合与开放曲线

- **闭合轮廓**：合并与简化后须保持 **闭合标志**；最后一段与第一段衔接处若合并，需保证 **C⁰**（位置连续），**C¹** 为可选目标。  
- **开放曲线**：端点 **不得** 被相邻段合并错误吸收；仅在**内部**断点尝试 `try_merge`。  
- **薄线 / 描边**（`is_stroke`）：V1 对描边形状**不参与**同色块合并；路径优化也应区分 **fill 与 stroke**，避免把描边几何当填充简化。

---

## 6. 与现有代码的复用关系

| 组件 | 路径 |
|------|------|
| **可复用** | `src/curve/bezier.cpp` 等贝塞尔工具（求值、细分、距离上界）；Schneider 拟合相关实现可作为 **try_merge** 中重拟合内核。 |
| **需替换或扩展** | `src/output/shape_merge.cpp` 中 **`MergeAdjacentSameColorShapes`** 的核心策略（当前过于保守）；若引入深度层，需与 **`pipeline.cpp`** 中排序/合并调用顺序一致。 |
| **配置** | `VectorizerConfig::merge_segment_tolerance`（`include/neroued/vectorizer/config.h`）可扩展为分 Pass 容差，或保持单参数比例缩放。 |

---

## 7. 开放问题

- **最优误差阈值**：屏幕分辨率、导出 DPI、抗锯齿宽度与 **ε** 的解析关系仍依赖**数据集调参**。  
- **路径优化 vs 视觉质量**：Pass 2 合并可能在**高曲率拐点**附近产生控制点漂移；需 **A/B 评估**（`eval/` 指标）与典型艺术图、图标集回归。  
- **与 SVG 数值精度**：舍入策略（SVGO 类）与 **几何误差** 联合优化时，**谁先谁后**影响结果稳定性。  
- **形状合并与 Cutout**：若存在镂空/裁剪语义，合并前须确认 **winding / 子路径方向** 与渲染器一致。  
- **kurbo 级面积矩**：若长期目标是「近最优」简化，是否引入 **Rust FFI** 或 **独立 C++ 移植** 需权衡维护成本。

---

## 参考文献与延伸阅读（仓库内）

- `docs/research/02-literature/kurbo-bezier-simplify.md` — kurbo / Green 定理与 ParamCurveFit 摘要。  
- `docs/research/01-current-architecture-analysis.md` — 管线中与 `MergeAdjacentSameColorShapes`、`merge_segment_tolerance` 的衔接说明。
