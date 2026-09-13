---
type: agent_context
project: deepwiki-rs
title: Agent Architecture Context
source: .
---

## 架构设计

```
┌─────────────────────── CLI 壳层 ───────────────────────┐
│  clap(CLI 解析/子命令) · config.rs(TOML 配置)   i18n.rs │
└──────────────────────┬────────────────────────────────┘
┌──────────────────────▼────────────────────────────────┐
│           生成引擎 Generator (workflow::launch)         │
│  预处理 preprocess → 研究 research → 组合 compose → 输出│
│                     outlet                             │
│      共享纽带: GeneratorContext + DocTree + Memory      │
└───────┬───────────────┬────────────────┬───────────────┘
   ┌────▼───┐      ┌─────▼─────┐    ┌────▼─────────┐
   │LLM 客户端│      │ 分层缓存 Cache │   │知识集成/工具域 │
```

**分层与容器**：

| 层 | 容器 | 说明 |
|---|---|---|
| 壳层 | CLI 接口 / 配置 / i18n | clap 解析子命令、TOML 配置加载、8 语言消息与文件名映射 |
| 编排层 | `Generator`（workflow.rs） | `launch()` 串起 preprocess→research→compose→output，产出 DocTree |
| Agent 抽象 | `StepForwardAgent` trait | 声明式 Agent：声明数据源、提示模板、输出 schema、缓存与并发策略 |
| 生成层 | preprocess / research / compose / outlet | 四阶段管道，每阶段模块内含独立的通用+专用 Agent |
| 基础设施 | llm / memory / cache / integrations / utils / types | 被生成层复用的共享内核，所有阶段共享拓扑 |

**关键架构决策**：链式四阶段管道 + 阶段作用域内存（`MemoryScope::PREPROCESS/RESEARCH/...`）；LLM 结果分层缓存降低重复调用；ReAct（Reasoning+Acting）工具增强，允许研究 Agent 动态探索仓库；研究/组合逻辑统一收敛到 `StepForwardAgent` **模板方法**模式，`AgentExecutor` 按需走 `extract`（结构化）/`prompt`（自由文本）两类调用。

## 模块地图

| 模块 | 职责 | 主要路径 |
|---|---|---|
| CLI/入口 | 参数解析、子命令分发、知识同步入口 | `src/main.rs`、`src/cli.rs`、`src/config.rs` |
| 生成编排 | 四阶段流水线 launch、流程计时、上下文 | `src/generator/workflow.rs`、`context.rs`、`types.rs` |
| Agent 框架 | StepForwardAgent trait、执行器、文档树 | `src/generator/step_forward_agent.rs`、`agent_executor.rs` |
| 预处理 | 结构/依赖/复杂度抽取、目录评分与摘要、关系分析 | `src/generator/preprocess/` |
| 语言处理器 | 12 种语言源码解析（依赖/接口/复杂度） | `src/generator/preprocess/extractors/language_processors/` |
| 研究阶段 | 7 个研究 Agent 的 C1–C4 多视角分析编排 | `src/generator/research/orchestrator.rs`、`agents/` |
| 组合阶段 | 由研究结果撰写 6 类 C4 文档章节，写入 DocTree | `src/generator/compose/agents/`（`*_editor.rs`） |
| 输出阶段 | 总览文档汇总、生成器、修复器、落盘 | `src/generator/outlet/` |
| LLM 客户端 | Provider 适配、prompt/extract、模型选择与回退、ReAct 执行 | `src/llm/client/` |
| ReAct 工具 | 文件浏览/读取/时钟等 preset 工具 | `src/llm/tools/` |
| 内存层 | 阶段作用域共享存储（按 scope/key） | `src/memory/`、`src/generator/*/memory.rs` |
| 缓存层 | LLM 响应分层缓存 + 命中/性能监控报告 | `src/cache/` |
| 类型/工具域 | 领域模型、并发、token 估算、prompt 压缩、文件工具 | `src/types/`、`src/utils/` |
| 知识集成 | 本地文档库解析、外部知识同步 | `src/integrations/` |

## 核心流程

**流程 1 — 整体流水线（`launch`）**
1. 读取配置（TOML）+ 目标项目路径，初始化 `GeneratorContext` 与阶段 `MemoryScope`。
2. **预处理**：抽取项目结构、源码依赖与复杂度 → LLM 生成目录摘要/评分 → 分析文件间关系，产物写入 `PREPROCESS` 内存并缓存压缩。
3. **研究**：按序执行研究 Agent 组（见流程 3），理解产物理入 `RESEARCH` 内存。
4. **组合**：6 个 editor Agent 从内存取研究结论，逐章生成 C4 文档片段并写入 `DocTree`。
5. **输出**：汇总 DocTree → 生成物经 summary/fixer 后按目标语言命名落盘（如 `1.概述.md`、`2.架构.md`）。

**流程 2 — 预处理管道**
1. `structure_extractor` + 12 语言处理器并行（受控并发，默认 5 文件）抽取每个文件的源码依赖、接口、复杂度指标。
2. `original_document_extractor` 构建原始文档语料。
3. `directory_summary` / `directory_scoring` 用 LLM 对目录做业务摘要与价值评分，产出目录档案。
4. `relationships_analyze` 分析文件/模块间关系（结构化输出），聚合为 `CodeAndDirectoryInsights` 存入内存。

**流程 3 — 研究多 Agent 编排**
1. `SystemContextResearcher`：项目定位（做什么/给谁用）。
2. `DomainModulesDetector`：划分领域模块并打分。
3. `ArchitectureResearcher` + `WorkflowResearcher`：深挖 C2 架构与 C3 工作流。
4. `KeyModulesInsight` + `BoundaryAnalyzer`：逐模块见解、盘点对外接口边界。
5. 若检测到数据库文件，追加 `DatabaseOverviewAnalyzer` 生成 C4 数据库视图。

**流程 4 — StepForwardAgent 执行与 ReAct**
1. StepForwardAgent 校验内存数据源，构造提示（必要时 token 压缩）。
2. `AgentExecutor` 按声明选择 `extract`（带 JSON schema 的结构化抽取）或 `prompt`（含 ReAct 工具循环）。
3. 模型按提示规模选型（高效/强势双模型 + 失败回退），带重试退避。
4. 结果经上下文压缩/缓存策略后存入对应 `MemoryScope`。

## 技术选型

- **语言/运行时**：Rust（edition 2024）、tokio 异步运行时、async-trait。
- **CLI/配置**：clap 4（derive）、toml、schemars（JSON schema 生成）。
- **LLM 集成**：rig-core 0.35（`ProviderClient`、Agent、CompletionModel）；reqwest 0.12（OpenAI 兼容 HTTP）。双 Provider：云端 OpenAI/Anthropic 兼容 API + 本地 Ollama（`ollama_extractor`）。
- **Agent 执行**：自研 ReAct 循环（`react.rs`/`react_executor.rs`）+ preset 工具（file_explorer、file_reader、time）。
- **代码解析**：walkdir、regex、12 语言处理器（静态正则/关键字解析，非 AST）；pdf-extract、markdown（文档解析）。
- **序列化/工具**：serde、serde_json、md-5、base64、chrono、uuid、glob、pathdiff、rand、futures。
- **错误处理**：anyhow（调用方）+ thiserror（库内）。
- **性能**：信号量限流并行、`do_parallel_with_limit`、分层缓存 + `CachePerformanceMonitor`、token 估算与 prompt 压缩。

## 系统边界

| 边界 | 交互对象 | 说明 / 信任考量 |
|---|---|---|
| LLM API | 云端 OpenAI/Anthropic 兼容端点、本地 Ollama | HTTP/JSON，非可信外部服务；带重试退避、双模型回退、token 控制、按提示规模选模型 |
| ReAct 工具 | 目标仓库文件系统 | `file_explorer`/`file_reader` 受限访问，白名单校验（日志 `access denied`）；LLM 输出视为非可信指令输入 |
| 文件系统 | 目标源码目录、本地文档库、缓存目录（`.litho/cache/...`）、输出目录 | 统一读写文件系统，无网络数据库或常驻服务 |
| 知识集成 | 本地文档分类库、外部知识同步源 | `local_docs`（分块/分类）+ `knowledge_sync`（同步缓存） |
| 配置面 | `litho-example.toml`（模型、并发、缓存、知识库） | 通过 clap 覆盖（`--llm-api-base-url`、`--model-efficient` 等） |

## 代码映射索引

| 概念 | 位置 | 备注 |
|---|---|---|
| CLI 入口 / 子命令 | `src/cli.rs`、`src/main.rs` | `handle_subcommand`、`sync_knowledge` |
| 配置结构 | `src/config.rs` | + 根目录 `litho-example.toml` |
| 输出语言 / 文案 | `src/i18n.rs` | `TargetLanguage`（8 语）、文件名映射 |
| 流水线编排 | `src/generator/workflow.rs` | `launch()`、阶段计时 `TimingKeys` |
| 上下文 / 文档树 | `src/generator/context.rs` | `GeneratorContext`、DocTree |
| Agent 抽象 | `src/generator/step_forward_agent.rs` | `StepForwardAgent`、`LLMCallMode`、`FormatterConfig` |
| Agent 执行 | `src/generator/agent_executor.rs` | prompt/extract 分发 |
| 预处理主入口 | `src/generator/preprocess/mod.rs` | 结构/依赖/关系分析 + 目录评分摘要 |
| 语言处理器 | `src/generator/preprocess/extractors/language_processors/` | 12 种语言 |
| 研究编排与 Agent | `src/generator/research/orchestrator.rs`、`agents/` | 7 个研究 Agent |
| 组合 Agent | `src/generator/compose/agents/` | `overview/architecture/workflow/database/boundary/key_modules_insight_editor.rs` |
| 输出 | `src/generator/outlet/` | `summary_outlet.rs`、`summary_generator.rs`、`fixer.rs` |
| LLM 客户端 | `src/llm/client/` | `LLMClient`、`react.rs`、`summary_reasoner.rs`、`ollama_extractor.rs` |
| ReAct 工具 | `src/llm/tools/` | `file_explorer.rs`、`file_reader.rs`、`time.rs` |
| 内存 / 缓存 | `src/memory/`、`src/cache/` | `MemoryScope`、`CachePerformanceMonitor` |
| 知识集成 / 工具域 | `src/integrations/`、`src/utils/` | `knowledge_sync.rs`、`local_docs.rs`、`threads.rs`、`token_estimator.rs` |