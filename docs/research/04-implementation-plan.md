# 初步实施计划

## 1. 总体策略

V2 管线与 V1 **并行共存**，通过 `VectorizerConfig::pipeline_mode` 枚举切换。V1 代码**零修改**（除最小化的 dispatch 逻辑），所有新功能在独立文件中实现。

## 2. 里程碑

### M1：基础设施 + 层叠管线骨架（预计 2-3 周）

**目标**：端到端跑通 V2 管线，可用 eval 框架对比 V1。

**任务**：
1. `config.h` 添加 `PipelineMode` 枚举（`V1`/`V2`），默认 `V1`
2. `vectorizer.cpp` 添加 dispatch 逻辑
3. `pipeline.h` 添加 `RunPipelineV2` 声明
4. 实现 `pipeline_v2.cpp` 管线编排骨架
5. 实现形状层提取（`cv::connectedComponents`）
6. 实现深度排序（覆盖面积能量 + 有向图 + 拓扑排序）
7. 实现形状延伸（形态学膨胀到遮挡区域）
8. 逐层 Potrace 追踪（复用 `TraceMaskWithPotraceBezier`）
9. 按深度排序输出 SVG（复用 `WriteSvg`）
10. CMakeLists.txt 添加新源文件

**量化颜色 placeholder**：M1 暂用 V1 的 K-Means 分割作为颜色量化（直接调用 `BgrToLab` + OpenCV `kmeans`），在 M2 替换。

**验收标准**：
- V2 管线端到端可运行
- eval 框架可对比 V1/V2 结果
- 无缝隙（形状延伸有效）
- 形状数 ≤ V1（层叠模型无孔洞）

### M2：感知色彩量化（预计 2-3 周）

**目标**：用 OKLab + MMCQ 替换 placeholder 量化，提升颜色还原质量。

**任务**：
1. 实现 `src/quantize/oklab.h`：sRGB ↔ OKLab 转换
2. 实现 `src/quantize/color_quantize.cpp`：Modified Median Cut Quantization
3. 支持 `num_colors = 0` 自动选择
4. 集成到 `pipeline_v2.cpp` 替换 placeholder
5. 用 MergeSmallComponents 做量化后清理

**验收标准**：
- MMCQ 在标准测试图上 ΔE 优于 V1 的 SLIC+KMeans
- 自动颜色数选择可用
- 量化速度 ≤ V1 分割时间

### M3：路径优化 + 形状合并（预计 2-3 周）

**目标**：降低文件尺寸，提升路径紧凑度。

**任务**：
1. 实现 `src/curve/path_optimize.cpp`：近线性段合并
2. 实现相邻段重拟合合并
3. 实现碎片过滤（小面积形状合并到同色最近形状）
4. 实现 z-order 安全的同色形状合并（同深度层同色合并为单 `<path>`）
5. 集成到 `pipeline_v2.cpp`

**验收标准**：
- SVG 文件尺寸 ≤ V1 的 80%
- 路径段数显著减少
- 视觉质量无退化（eval 指标不下降）

## 3. 代码变更范围

### 修改的现有文件（4 个）

| 文件 | 变更内容 | 变更量 |
|------|----------|--------|
| `include/neroued/vectorizer/config.h` | 添加 `PipelineMode` 枚举 + `pipeline_mode` 字段 | ~10 行 |
| `src/vectorizer.cpp` | 添加 dispatch `if (config.pipeline_mode == V2)` | ~5 行 |
| `src/pipeline.h` | 添加 `RunPipelineV2` 声明 | ~3 行 |
| `CMakeLists.txt` | 添加新 .cpp 到 target | ~8 行 |

### 新增文件

| 阶段 | 文件 | 职责 |
|------|------|------|
| M1 | `src/pipeline_v2.cpp` | V2 管线编排 |
| M1 | `src/stacking/depth_order.h` / `.cpp` | 深度排序 |
| M1 | `src/stacking/shape_extend.h` / `.cpp` | 形状延伸 |
| M2 | `src/quantize/oklab.h` | OKLab 转换 |
| M2 | `src/quantize/color_quantize.h` / `.cpp` | MMCQ 量化 |
| M3 | `src/curve/path_optimize.h` / `.cpp` | 路径优化 |

### 复用的现有模块

| 模块 | 复用方式 |
|------|----------|
| `src/preprocess/preprocess.cpp` | 直接调用 `PreprocessForVectorize` |
| `src/trace/potrace.cpp` | 直接调用 `TraceMaskWithPotraceBezier` |
| `src/curve/bezier.cpp` | 工具函数直接复用 |
| `src/output/svg_writer.cpp` | 调用 `WriteSvg`（可能需小幅调整无孔洞模式） |
| `src/segment/morphology.cpp` | `MergeSmallComponents` 用于量化后清理 |
| `src/detail/` | 工具函数直接复用 |
| `eval/` | 评估框架直接复用 |

## 4. V1/V2 并行共存策略

```cpp
// config.h
enum class PipelineMode { V1, V2 };

struct VectorizerConfig {
    PipelineMode pipeline_mode = PipelineMode::V1;
    // ... 其他字段不变
};
```

```cpp
// vectorizer.cpp (dispatch)
auto result = (config.pipeline_mode == PipelineMode::V2)
    ? detail::RunPipelineV2(prepared.bgr, config, prepared.opaque_mask)
    : detail::RunPipeline(prepared.bgr, config, prepared.opaque_mask);
```

- V1 用户完全不受影响（默认 `V1`）
- V2 通过 `config.pipeline_mode = PipelineMode::V2` 启用
- 两个管线共享预处理和 SVG 输出，中间阶段完全独立
- Python 绑定和 CLI 工具需同步更新以暴露 `pipeline_mode` 参数

## 5. 测试与评估方案

- 使用现有 eval 框架（`evaluate_svg` CLI 工具）做 V1/V2 对比
- 关键指标：PSNR、SSIM、ΔE mean/p95、coverage、overlap、edge_f1、文件尺寸
- 建立包含多类图像的测试集（扁平色插画、像素艺术、简单照片、复杂图案）
- 每个里程碑结束时做全量对比测试

## 6. 依赖分析

V2 不引入任何新的外部依赖：
- OpenCV：已有，用于连通域分析、凸包、形态学操作
- Potrace：已有，逐层追踪
- Clipper2：已有（但 V2 可能不需要，形状延伸用 OpenCV 形态学替代）
- spdlog：已有，日志
