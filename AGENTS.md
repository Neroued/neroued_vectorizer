# neroued_vectorizer AGENTS 协作指南

本文件是仓库级导航入口，帮助开发者与 AI agent 快速定位模块、理解约束和执行常见任务。

## 1 分钟上手

1. 读总览：[README.md](README.md)
2. 按任务类型跳转到对应模块
3. 改代码前确认规则：`.cursor/rules/`
4. 改代码后执行格式化与文档同步检查

## 项目全景

本库实现栅格到 SVG 矢量化的完整管线，支持两种管线模式：

**V1（默认）**：边界图 + 剪切模型
```
输入图像 → [预处理] → [颜色分割] → [边界提取] → [轮廓装配] → [曲线拟合] → [轮廓追踪] → [SVG输出]
```

**V2（层叠模型）**：深度排序 + 画家算法
```
输入图像 → [预处理] → [OKLab MMCQ量化] → [小区域合并] → [连通域提取] → [深度排序] → [形状延伸] → [逐层Potrace] → [路径优化] → [同色合并] → [覆盖率修补] → [SVG输出]
```

通过 `VectorizerConfig::pipeline_mode` 选择 `PipelineMode::V1` 或 `PipelineMode::V2`。

### 目录映射

- `include/neroued/vectorizer/`：公共 API 头文件（VectorizerConfig / VectorizerResult / Vectorize）
- `src/preprocess/`：图像缩放、Mean Shift 平滑
- `src/segment/`：SLIC 超像素、K-Means 聚类、形态学清理、小区域合并
- `src/boundary/`：像素级边界图构建、亚像素细化、抗锯齿边缘检测
- `src/contour/`：链式轮廓装配、薄线矢量化
- `src/curve/`：贝塞尔工具函数、Schneider 曲线拟合、路径优化（V2）
- `src/trace/`：Potrace 位图追踪、覆盖率修补、Clipper2 拓扑修复
- `src/stacking/`：V2 层叠模型（深度排序、形状延伸）
- `src/quantize/`：V2 OKLab MMCQ 颜色量化
- `src/output/`：SVG 文档生成、同色形状合并
- `src/detail/`：内部工具（OpenCV 辅助、ICC 色彩管理、VectorizedShape 核心类型）
- `python/`：Python 绑定（pybind11 绑定代码、Python 包、测试）
- `eval/`：质量评估库（像素/边缘/路径指标、基线对比）
- `apps/`：CLI 工具（raster_to_svg、evaluate_svg）
- `ci/`：CI 依赖安装脚本（Linux/macOS/Windows）
- `.github/workflows/`：GitHub Actions CI/CD
- `tests/`：单元测试

## 按任务快速定位

| 任务 | 首先查看 |
|------|----------|
| 调整颜色分割行为 | `src/segment/color_segment.cpp`、`src/segment/slic.cpp` |
| 修改曲线拟合策略 | `src/curve/fitting.cpp`、`src/curve/bezier.cpp` |
| 优化边界提取精度 | `src/boundary/boundary_graph.cpp`、`src/boundary/subpixel_refine.cpp` |
| 修改 SVG 输出格式 | `src/output/svg_writer.cpp` |
| 调整预处理流程 | `src/preprocess/preprocess.cpp` |
| 修改 Potrace 追踪行为 | `src/trace/potrace.cpp` |
| 修改覆盖率修补逻辑 | `src/trace/coverage.cpp` |
| 修改拓扑修复策略 | `src/trace/topology.cpp` |
| 调整 V1 管线编排流程 | `src/pipeline.cpp` |
| 调整 V2 管线编排流程 | `src/pipeline_v2.cpp` |
| 修改 V2 深度排序逻辑 | `src/stacking/depth_order.cpp` |
| 修改 V2 形状延伸策略 | `src/stacking/shape_extend.cpp` |
| 修改 V2 OKLab 颜色量化 | `src/quantize/color_quantize.cpp`、`src/quantize/oklab.h` |
| 修改 V2 路径优化 | `src/curve/path_optimize.cpp` |
| 新增/修改公共 API | `include/neroued/vectorizer/vectorizer.h`、`src/vectorizer.cpp` |
| 新增/修改配置参数 | `include/neroued/vectorizer/config.h` |
| 修改质量评估指标 | `eval/src/pixel_metrics.cpp`、`eval/src/edge_metrics.cpp`、`eval/src/path_metrics.cpp` |
| 修改 CLI 工具参数 | `apps/raster_to_svg.cpp`、`apps/evaluate_svg.cpp` |
| 修改 Python 绑定 | `python/bindings.cpp`、`python/neroued_vectorizer/__init__.py` |
| 修改 Python 类型桩 | `python/neroued_vectorizer/_core.pyi` |
| 修改 CI/CD | `.github/workflows/ci.yml`、`.github/workflows/wheels.yml` |
| 修改 wheel 构建配置 | `pyproject.toml`、`ci/install-deps-*.sh` |

## 核心类型与 API

### 公共头文件

| 文件 | 内容 |
|------|------|
| `vectorizer.h` | 3 个 `Vectorize` 重载（文件路径 / 内存缓冲区 / cv::Mat） |
| `config.h` | `PipelineMode` 枚举、`VectorizerConfig` — 管线完整配置 |
| `result.h` | `VectorizerResult` — SVG 内容、尺寸、形状数、调色板 |
| `color.h` | `Rgb`、`Lab` 颜色类型及空间转换 |
| `vec2.h` / `vec3.h` | 2D/3D 向量类型 |
| `error.h` | `InputError`、`IOError` 异常类型 |
| `logging.h` | `InitLogging`、`ParseLogLevel` |
| `eval.h`（eval 库） | 评估配置、指标、基线对比 |

### 命名空间

- 公共 API：`neroued::vectorizer`
- 内部实现：`neroued::vectorizer::detail`
- 评估模块：`neroued::vectorizer`（及 `neroued::vectorizer::eval` 辅助工具）

## 全局协作规则

### 头文件边界

- **公共头文件**（`include/neroued/vectorizer/`）：对外 API，保持稳定且最小化
- **内部头文件**（`src/` 各子目录的 `.h`）：仅供库内部使用，不对外暴露
- 公共头文件使用 `<neroued/vectorizer/xxx.h>` 引用
- 内部头文件使用相对路径 `"subdir/xxx.h"` 引用

### 代码风格

- C++20 标准
- 使用 `.clang-format` 格式化（LLVM 基础、4 空格、100 列宽）
- 修改 C++ 文件后执行 `clang-format -i <modified-files>`

### 依赖管理

- OpenCV 和 Potrace 为必需系统依赖
- spdlog 和 Clipper2 通过 FetchContent 自动获取（也支持 3rdparty/ 子目录）
- lcms2 + libjpeg 为可选依赖（ICC 色彩管理）
- pybind11 通过 scikit-build-core 构建时自动获取（Python 绑定）

## 提交前检查清单

- [ ] C++ 文件执行 `clang-format -i <modified-files>`
- [ ] 构建通过：`cmake --build build -j$(nproc)`
- [ ] 公共 API 变更 → 更新 README.md 参数表
- [ ] CLI 参数变更 → 更新 README.md CLI 用法
- [ ] 配置参数增删 → 更新 `config.h` 注释和 README.md
- [ ] 新增文件 → 更新 CMakeLists.txt 和 AGENTS.md 模块索引
- [ ] 不引入重复工具函数
- [ ] 不在公共头文件中暴露内部实现细节
- [ ] Python 绑定变更 → 同步更新 `_core.pyi` 类型桩
- [ ] Python API 变更 → 更新 README.md Python 用法
