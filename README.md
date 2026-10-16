<div align="center">

<img src="assets/osai.png" alt="Awesome Open Source AI" width="120" />



# Awesome Open Source AI

精选的开源人工智能模型、库、基础设施与开发者工具。

</div>

## 目录

- [1. 核心框架与库](#1-核心框架与库)
- [2. 模型代码库与模型家族](#2-模型代码库与模型家族)
- [3. 推理引擎与服务化](#3-推理引擎与服务化)
- [4. 智能体 AI 与多智能体系统](#4-智能体-ai-与多智能体系统)
- [5. 检索增强生成（RAG）与知识库](#5-检索增强生成rag与知识库)
- [6. 生成式媒体工具](#6-生成式媒体工具)
- [7. 训练与微调生态](#7-训练与微调生态)
- [8. MLOps / LLMOps 与生产化](#8-mlops--llmops-与生产化)
- [9. 评估、基准与数据集](#9-评估基准与数据集)
- [10. AI 安全、对齐与可解释性](#10-ai-安全对齐与可解释性)
- [11. 专业领域](#11-专业领域)
- [12. 用户界面与自托管平台](#12-用户界面与自托管平台)
- [13. 开发者工具与集成](#13-开发者工具与集成)
- [14. 资源与学习](#14-资源与学习)

---

## 关于本列表

Awesome Open Source AI 是一份为使用 AI 进行构建的人整理的开源项目清单。

目标是帮助读者找到有用的模型、库、工具、基础设施、数据集和学习资源，而不必在一堆目录式条目中大海捞针。

项目收录不要求最低的 GitHub star 数量。Star 数可以作为有用的参考，但它只是一个信号。如果一个较小的项目足够有用、维护良好、技术上有意思、文档清晰，或者对 AI 生态的某个特定部分很重要，那么它就可以收录进来。

好的条目应该有清晰的存在理由。它们应当帮助人们构建、研究、运行、评估或理解 AI 系统。

---

## 1. 核心框架与库

> 用于构建、训练和运行 AI 与机器学习系统的核心库与框架。

#### 深度学习框架

- [PyTorch](https://github.com/pytorch/pytorch) - 动态计算图、Python 风格的 API，在研究与生产领域占据主导地位。是当前大多数前沿 AI 工作的默认选择。 ![GitHub stars](https://img.shields.io/github/stars/pytorch/pytorch?style=social)
- [TensorFlow](https://github.com/tensorflow/tensorflow) - 端到端平台，生产部署能力出色，支持 TPU，并提供大规模服务化工具。 ![GitHub stars](https://img.shields.io/github/stars/tensorflow/tensorflow?style=social)
- [JAX](https://github.com/jax-ml/jax) - 高性能数值计算，具备可组合的变换（JIT、vmap、grad）。在研究和科学机器学习领域日益受到青睐。 ![GitHub stars](https://img.shields.io/github/stars/jax-ml/jax?style=social)
- [Flax](https://github.com/google/flax) - 面向 JAX 的神经网络库，设计上强调灵活性。Apache-2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/google/flax?style=social)
- [dm-haiku](https://github.com/google-deepmind/dm-haiku) - Google DeepMind 基于 JAX 的神经网络库。拥有带状态管理的优雅函数式 API，广泛用于 DeepMind 的研究。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/google-deepmind/dm-haiku?style=social)
- [Equinox](https://github.com/patrick-kidger/equinox) - 在 JAX 中提供优雅易用的神经网络与科学计算。可调用的 PyTrees 与过滤变换，与 JAX 生态无缝互通。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/patrick-kidger/equinox?style=social)
- [Diffrax](https://github.com/patrick-kidger/diffrax) - JAX 中的数值微分方程求解器。支持自动微分、可运行在 GPU 上的 ODE/SDE/CDE 求解器，面向科学机器学习与神经常微分方程。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/patrick-kidger/diffrax?style=social)
- [vit-pytorch](https://github.com/lucidrains/vit-pytorch) - PyTorch 中完整的 Vision Transformer (ViT) 实现。涵盖所有主流视觉 Transformer 变体的参考实现，包括 ViT、DeiT、Swin 等。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/lucidrains/vit-pytorch?style=social)
- [NumPyro](https://github.com/pyro-ppl/numpyro) - 基于 NumPy 的概率编程，借助 JAX 提供自动微分与 JIT 编译支持。大规模贝叶斯建模与推断。 ![GitHub stars](https://img.shields.io/github/stars/pyro-ppl/numpyro?style=social)
- [Keras](https://github.com/keras-team/keras) - 高级、对新手友好的 API，如今可运行在多个后端（TensorFlow、JAX、PyTorch）之上。非常适合快速实验。 ![GitHub stars](https://img.shields.io/github/stars/keras-team/keras?style=social)
- [tinygrad](https://github.com/tinygrad/tinygrad) - 极简的深度学习框架，代码体量极小。秉持"你喜欢 PyTorch？你喜欢 micrograd？你一定会爱上 tinygrad！"的理念——简单却强大。 ![GitHub stars](https://img.shields.io/github/stars/tinygrad/tinygrad?style=social)
- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) - 百度推出的工业级深度学习平台，服务 2300 万+ 开发者和 76 万+ 企业。中国首个自主研发框架，具备先进的分布式训练与部署能力。 ![GitHub stars](https://img.shields.io/github/stars/PaddlePaddle/Paddle?style=social)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) - 面向图、点云、流形等不规则输入数据做深度学习的库。属于 PyTorch 生态的一部分。 ![GitHub stars](https://img.shields.io/github/stars/pyg-team/pytorch_geometric?style=social)
- [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models) - 最大的 PyTorch 图像编码器与骨干网络集合。包含 900+ 个预训练模型，覆盖 ResNet、EfficientNet、Vision Transformer、ConvNeXt 等，并附训练与推理脚本。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/pytorch-image-models?style=social)
- [Triton](https://github.com/triton-lang/triton) - 用于编写高效自定义深度学习原语的语言与编译器。为 PyTorch、JAX 等框架的 kernel 优化提供支撑。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/triton-lang/triton?style=social)
- [GGML](https://github.com/ggml-org/ggml) - 面向机器学习的张量库。驱动 llama.cpp 以及众多端侧推理引擎的底层 C/C++ 库。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/ggml-org/ggml?style=social)
- [MLX](https://github.com/ml-explore/mlx) - 面向 Apple silicon 的机器学习数组框架。采用高效的统一内存设计，提供类 NumPy 的 API、自动微分与多设备支持。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/ml-explore/mlx?style=social)
- [notorch](https://github.com/ariannamethod/notorch) - 纯 C 实现的神经网络框架，具备反向模式自动微分、模型训练与 GGUF 推理能力，无需 Python 运行时。 ![GitHub stars](https://img.shields.io/github/stars/ariannamethod/notorch?style=social)

#### 高性能计算库

- [oneDNN](https://github.com/uxlfoundation/oneDNN) - oneAPI 深度神经网络库。为深度学习提供跨平台的基础组件性能库，针对 Intel CPU、GPU 与 Arm 架构优化。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/uxlfoundation/oneDNN?style=social)
- [ONNX](https://github.com/onnx/onnx) - 机器学习互操作的开放标准。开放神经网络交换格式提供了开放生态，让 AI 开发者能随项目演进选择合适的工具。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/onnx/onnx?style=social)
- [IREE](https://github.com/iree-org/iree) - 可重定目标的、基于 MLIR 的机器学习编译器与运行时工具包。将 ML 模型降级到统一 IR，涵盖从数据中心到移动端与边缘端的部署场景。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/iree-org/iree?style=social)
- [Modular Platform](https://github.com/modular/modular) - 围绕 MAX Engine 与 Mojo 编程语言构建的开源 AI 计算与编程平台。 ![GitHub stars](https://img.shields.io/github/stars/modular/modular?style=social)

#### Rust 机器学习框架

- [Burn](https://github.com/tracel-ai/burn) - 用 Rust 编写的下一代深度学习框架。后端无关，支持 CPU、GPU 与 WebAssembly。 ![GitHub stars](https://img.shields.io/github/stars/tracel-ai/burn?style=social)
- [Candle (Hugging Face)](https://github.com/huggingface/candle) - 极简的 Rust 机器学习框架。类 PyTorch 的 API，注重性能与简洁。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/candle?style=social)
- [linfa](https://github.com/rust-ml/linfa) - 完备的 Rust 机器学习工具包，内含多种经典算法。相当于 Rust 版的 scikit-learn，涵盖聚类、回归与预处理。 ![GitHub stars](https://img.shields.io/github/stars/rust-ml/linfa?style=social)

#### Julia 机器学习框架

- [Flux.jl](https://github.com/FluxML/Flux.jl) - 100% 纯 Julia 的机器学习技术栈，在原生 GPU 与自动微分支持之上提供轻量抽象。优雅、可魔改，并与 Julia 的科学计算生态完全集成。 ![GitHub stars](https://img.shields.io/github/stars/FluxML/Flux.jl?style=social)
- [MLJ.jl](https://github.com/JuliaAI/MLJ.jl) - 完备的 Julia 机器学习框架，通过统一接口接入 200+ 个模型，并提供用于模型选择、调优与评估的元算法。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/JuliaAI/MLJ.jl?style=social)
- [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) - 面向科学机器学习的高性能符号-数值建模框架。可为雅可比矩阵、海森矩阵等模型组件自动生成快速函数，并支持自动稀疏化与并行化。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/SciML/ModelingToolkit.jl?style=social)

#### NLP 与 Transformer

- [spaCy (Explosion AI)](https://github.com/explosion/spaCy) - 工业级自然语言处理，支持 75+ 种语言、Transformer 流水线，以及生产级 NER、句法分析与文本分类。 ![GitHub stars](https://img.shields.io/github/stars/explosion/spaCy?style=social)
- [Transformers (Hugging Face)](https://github.com/huggingface/transformers) - 预训练 NLP 模型的事实标准库。100 万+ 模型，日下载量 25 万+。涵盖 BERT、GPT、Llama、Qwen 以及数百个其他模型。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/transformers?style=social)
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - 用于句子与图像嵌入的经典库。 ![GitHub stars](https://img.shields.io/github/stars/UKPLab/sentence-transformers?style=social)
- [tokenizers (Hugging Face)](https://github.com/huggingface/tokenizers) - 面向训练与推理的快速前沿分词器。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/tokenizers?style=social)
- [fairseq2](https://github.com/facebookresearch/fairseq2) - FAIR 序列建模工具包 2。基于现代 PyTorch API 完全重写 fairseq，原生支持 LLM 训练（700 亿+ 参数模型）、vLLM 集成，以及指令微调与偏好优化的官方配方。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/facebookresearch/fairseq2?style=social)
- [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate) - 由 Argos Translate 引擎驱动的自托管机器翻译 API。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/LibreTranslate/LibreTranslate?style=social)

#### 数据处理与操作

- [Pandas](https://github.com/pandas-dev/pandas) - Python 数据分析与操作的黄金标准。 ![GitHub stars](https://img.shields.io/github/stars/pandas-dev/pandas?style=social)
- [Polars](https://github.com/pola-rs/polars) - 速度飞快的 DataFrame 库（Rust 后端）——面向大规模工作负载的 Pandas 现代替代方案。 ![GitHub stars](https://img.shields.io/github/stars/pola-rs/polars?style=social)
- [cuDF](https://github.com/rapidsai/cudf) - RAPIDS 出品的 GPU DataFrame 库。借助 cuDF.pandas 加速模式，无需改动一行代码即可在 NVIDIA GPU 上加速 Pandas 工作流。 ![GitHub stars](https://img.shields.io/github/stars/rapidsai/cudf?style=social)
- [Dask](https://github.com/dask/dask) - 面向大数据的并行计算——将 Pandas/NumPy/scikit-learn 扩展到集群上。 ![GitHub stars](https://img.shields.io/github/stars/dask/dask?style=social)
- [DataFlow](https://github.com/OpenDCAI/DataFlow) - 面向 LLM 的数据准备系统，可将原始 PDF、对话、代码、数据库等来源转换为适用于 SFT、QA 与 RAG 的数据集。 ![GitHub stars](https://img.shields.io/github/stars/OpenDCAI/DataFlow?style=social)
- [NumPy](https://github.com/numpy/numpy) - 支撑几乎所有 AI 技术栈的基础数组计算库。 ![GitHub stars](https://img.shields.io/github/stars/numpy/numpy?style=social)
- [SciPy](https://github.com/scipy/scipy) - 科学计算算法库（优化、线性代数、统计、信号处理）。 ![GitHub stars](https://img.shields.io/github/stars/scipy/scipy?style=social)
- [CuPy](https://github.com/cupy/cupy) - 与 NumPy、SciPy 兼容的数组库，为 Python 提供 GPU 加速计算。 ![GitHub stars](https://img.shields.io/github/stars/cupy/cupy?style=social)
- [NetworkX](https://github.com/networkx/networkx) - 复杂网络的创建、操作与研究。Python 数据科学领域的基础图分析库。 ![GitHub stars](https://img.shields.io/github/stars/networkx/networkx?style=social)
- [cuGraph](https://github.com/rapidsai/cugraph) - GPU 图分析库，提供与 NetworkX 兼容的 API。在大规模图算法上比 CPU 快 10-100 倍。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/rapidsai/cugraph?style=social)
- [Datashader](https://github.com/holoviz/datashader) - 高性能大数据可视化。可交互式渲染数十亿个数据点，且不产生聚合伪影。BSD-3-Clause 许可。 ![GitHub stars](https://img.shields.io/github/stars/holoviz/datashader?style=social)
- [Zarr](https://github.com/zarr-developers/zarr-python) - 分块、压缩的 N 维数组存储。面向云端与并行计算优化的可扩展张量数据格式。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/zarr-developers/zarr-python?style=social)
- [NVIDIA DALI](https://github.com/NVIDIA/DALI) - GPU 加速的数据加载与增强库，为深度学习应用提供高度优化的基础组件。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/DALI?style=social)
- [Narwhals](https://github.com/narwhals-dev/narwhals) - DataFrame 库之间的轻量兼容层。编写类 Polars 代码，即可在 Pandas、Polars、cuDF、Modin 等库之间无缝运行。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/narwhals-dev/narwhals?style=social)
- [Ibis](https://github.com/ibis-project/ibis) - 支持 20+ 后端的可移植 Python dataframe 库。只需改一行代码，就能把类 Pandas 代码从本地 DuckDB 扩展到生产级数据库（BigQuery、Snowflake、PostgreSQL）。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/ibis-project/ibis?style=social)
- [skrub](https://github.com/skrub-data/skrub) - 面向脏乱类别数据的 dataframe 机器学习。为异构数据提供预处理与特征工程，并与 Pandas/Polars 无缝集成。BSD-3-Clause 许可。 ![GitHub stars](https://img.shields.io/github/stars/skrub-data/skrub?style=social)
- [Oxen](https://github.com/Oxen-AI/Oxen) - 为机器学习打造的闪电般快速的数据版本控制。面向大型数据集优化，具备高效的差异比较、分支与协作能力。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Oxen-AI/Oxen?style=social)
- [Pandera](https://github.com/unionai-oss/pandera) - 面向 dataframe 的统计型数据测试与校验。为 Pandas、Polars 等 dataframe 库提供类 Pydantic 的 API，支持类型提示与惰性校验。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/unionai-oss/pandera?style=social)
- [Snorkel](https://github.com/snorkel-team/snorkel) - 用弱监督快速生成训练数据的系统。通过标注函数与概率一致性模型，以编程方式标注、构建和管理训练数据。是 Snorkel Flow 的底层支撑，已被 Google、Apple 与 Intel 使用。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/snorkel-team/snorkel?style=social)
- [DuckDB](https://github.com/duckdb/duckdb) - 高性能分析型进程内 SQL 数据库系统。快速、可靠、可移植、易用，SQL 方言支持丰富。非常适合数据处理与分析类工作负载。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/duckdb/duckdb?style=social)
- [FiftyOne](https://github.com/voxel51/fiftyone) - 可视化 AI 开发工具包，用于可视化、标注与评估视觉数据集和模型。以数据集探索与模型分析赋能计算机视觉工作流。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/voxel51/fiftyone?style=social)
- [Label Studio](https://github.com/HumanSignal/label-studio) - 多类型数据标注与注释工具，输出格式标准化。为图像、文本、音频、视频与时间序列提供可配置界面，并支持 ML 辅助标注。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/HumanSignal/label-studio?style=social)
- [Delta Lake](https://github.com/delta-io/delta) - 开源存储框架，支撑 Lakehouse 架构，提供 ACID 事务、可扩展的元数据处理以及统一的批流一体处理。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/delta-io/delta?style=social)
- [Apache Iceberg](https://github.com/apache/iceberg) - 面向超大型分析表的高性能开放表格式。为大数据带来 SQL 表级别的可靠性，支持时间旅行、隐藏分区与 schema 演进。可与 Spark、Trino、Flink、Presto、Hive 和 Impala 协同工作。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/apache/iceberg?style=social)
- [Apache Hudi](https://github.com/apache/hudi) - 开放的数据湖仓平台，用于跨云环境摄取、索引、存储、服务、转换与管理数据。支持大数据上的 upsert、删除与增量处理，并内置 Spark 与 Flink 的摄取工具。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/apache/hudi?style=social)
- [lakeFS](https://github.com/treeverse/lakeFS) - 为数据湖提供的数据版本控制，把对象存储变成类 Git 仓库。支持分支、提交与合并，让数据流水线获得原子化、带版本的数据湖操作能力。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/treeverse/lakeFS?style=social)
- [Apache Airflow](https://github.com/apache/airflow) - 以编程方式编写、调度与监控工作流的平台。数据流水线与 ML 工作流的行业标准编排工具，拥有 500+ 集成。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/apache/airflow?style=social)
- [Apache Spark](https://github.com/apache/spark) - 面向大规模数据处理的统一分析引擎。基于内存的集群计算，提供 Python、Scala、Java 与 R 的高级 API。其 MLlib 支撑分布式机器学习，Structured Streaming 支撑实时数据。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/apache/spark?style=social)
- [Apache Flink](https://github.com/apache/flink) - 兼具强大批处理与流处理能力的框架。高吞吐、低延迟的运行时，提供恰好一次的处理保证。非常适合实时 AI 推理流水线与事件驱动的 ML 应用。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/apache/flink?style=social)
- [Apache Beam](https://github.com/apache/beam) - 批流一体的统一编程模型。流水线只写一次，即可在 Flink、Spark 或 Google Cloud Dataflow 上运行。可移植、可扩展，为 AI 数据流水线而生，已达企业级成熟度。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/apache/beam?style=social)
- [Scrapy](https://github.com/scrapy/scrapy) - 快速的高层 Python 网络爬虫与数据抓取框架。可大规模从网站提取结构化数据，内置支持分页、Cookie 与并发请求等常见挑战的处理。BSD-3-Clause 许可。 ![GitHub stars](https://img.shields.io/github/stars/scrapy/scrapy?style=social)
- [Temporal](https://github.com/temporalio/temporal) - 面向可靠工作流编排的持久化执行平台。构建可从故障中恢复、从中断处精确续跑的有韧性数据流水线与 ML 工作流。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/temporalio/temporal?style=social)
- [Luigi](https://github.com/spotify/luigi) - 用于构建复杂批处理作业流水线的 Python 模块。处理依赖解析、工作流管理、可视化与 Hadoop 集成。由 Spotify 打造，并经生产环境实战检验。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/spotify/luigi?style=social)
- [Mage.ai](https://github.com/mage-ai/mage-ai) - 现代开源数据流水线工具，用于数据集成与转换。AI 原生的 ETL/ELT 平台，拥有 100+ 集成、实时监控与协作特性。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/mage-ai/mage-ai?style=social)
- [Hamilton](https://github.com/apache/hamilton) - 声明式数据流框架，用于构建可测试、模块化、自文档化的数据流水线。直接在 Python 函数中编码血缘与元数据。最初源自 Stitch Fix，现为 Apache 孵化项目。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/apache/hamilton?style=social)
- [D-Tale](https://github.com/man-group/dtale) - 面向 Pandas 数据结构的可视化工具，采用 Flask 后端与 React 前端。提供交互式数据探索，支持图表、筛选与代码导出。LGPL-2.1 许可。 ![GitHub stars](https://img.shields.io/github/stars/man-group/dtale?style=social)
- [TextAttack](https://github.com/QData/TextAttack) - 用于 NLP 对抗攻击、数据增强与模型训练的 Python 框架。可对数据集做增强以提升模型鲁棒性，并生成对抗样本。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/QData/TextAttack?style=social)
- [uv](https://github.com/astral-sh/uv) - 用 Rust 编写的极速 Python 包与项目管理器。比 pip 快 10-100 倍，内置虚拟环境管理、依赖解析与 lockfile。现代 AI/ML 开发工作流的必备工具。Apache 2.0 与 MIT 双许可。 ![GitHub stars](https://img.shields.io/github/stars/astral-sh/uv?style=social)
- [Vector](https://github.com/vectordotdev/vector) - 高性能可观测性数据流水线，用于采集、转换与路由日志和指标。支持 50+ 种数据源与接收端，包括 Kafka、S3 与 Elasticsearch 的实时数据处理。非常适合 AI/ML 的日志处理与数据摄取。MPL 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/vectordotdev/vector?style=social)

#### 经典机器学习与梯度提升

- [scikit-learn](https://github.com/scikit-learn/scikit-learn) - 传统机器学习（分类、回归、聚类、流水线）的行业标准库。 ![GitHub stars](https://img.shields.io/github/stars/scikit-learn/scikit-learn?style=social)
- [XGBoost](https://github.com/dmlc/xgboost) - 可扩展的高性能梯度提升库。在 Kaggle 与表格数据竞赛中依然占据主导。 ![GitHub stars](https://img.shields.io/github/stars/dmlc/xgboost?style=social)
- [LightGBM](https://github.com/microsoft/LightGBM) - 微软的超快速梯度提升框架，针对速度与内存优化。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/LightGBM?style=social)
- [CatBoost](https://github.com/catboost/catboost) - 原生处理类别特征的梯度提升，开箱即用的性能表现出色。 ![GitHub stars](https://img.shields.io/github/stars/catboost/catboost?style=social)
- [sktime](https://github.com/sktime/sktime) - 面向时间序列机器学习的统一框架。提供与 scikit-learn 兼容的 API，覆盖预测、分类、聚类与异常检测。 ![GitHub stars](https://img.shields.io/github/stars/sktime/sktime?style=social)
- [StatsForecast](https://github.com/Nixtla/statsforecast) - 闪电般的统计预测，内置 ARIMA、ETS、CES 与 Theta 模型。针对高性能时间序列工作负载优化。 ![GitHub stars](https://img.shields.io/github/stars/Nixtla/statsforecast?style=social)
- [MLForecast](https://github.com/Nixtla/mlforecast) - 可扩展的时间序列预测机器学习。在数百万条时间序列上训练任意兼容 sklearn 的模型，特征工程高效。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Nixtla/mlforecast?style=social)
- [cuML](https://github.com/rapidsai/cuml) - GPU 加速的机器学习算法，提供与 scikit-learn 兼容的 API。在大型数据集上比 CPU 实现快 10-50 倍。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/rapidsai/cuml?style=social)
- [SynapseML](https://github.com/microsoft/SynapseML) - 基于 Apache Spark 的分布式机器学习。为文本分析、视觉、异常检测提供可扩展、可组合的 API，并与 Python/Scala/R/.NET 无缝集成。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/SynapseML?style=social)
- [Darts](https://github.com/unit8co/darts) - 易用的时间序列预测与异常检测库。在同一套兼容 scikit-learn 的 API 中统一了经典统计模型（ARIMA、ETS）与现代神经网络（N-BEATS、TFT、DeepAR）。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/unit8co/darts?style=social)
- [PyTorch Forecasting](https://github.com/sktime/pytorch-forecasting) - 基于 PyTorch 的时间序列预测。支持多种神经架构（N-BEATS、TFT、DeepAR），内置可解释性能力，并基于 PyTorch Lightning 实现分布式训练。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/sktime/pytorch-forecasting?style=social)

#### 数据工程与特征存储

- [DataHub](https://github.com/datahub-project/datahub) - 数据与 AI 领域排名第一的开源元数据平台。提供数据发现、治理与可观测性，拥有 80+ 连接器、列级血缘，并集成 AI 助手。最初构建于 LinkedIn。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/datahub-project/datahub?style=social)
- [OpenMetadata](https://github.com/open-metadata/OpenMetadata) - 统一元数据平台，涵盖数据发现、可观测性与治理。提供列级血缘、语义搜索与团队协作，接入 70+ 数据服务连接器。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/open-metadata/OpenMetadata?style=social)
- [Amundsen](https://github.com/amundsen-io/amundsen) - 来自 Lyft 的数据发现与元数据引擎。以 PageRank 式搜索检索数据资源，并基于使用情况排序。LF AI & Data Foundation 项目。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/amundsen-io/amundsen?style=social)

#### 数据转换与分析工程

- [Apache Ossie](https://github.com/apache/ossie) - 厂商中立的规范，用于在分析、BI 与 AI 智能体平台之间统一语义模型。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/apache/ossie?style=social)
- [dbt-core](https://github.com/dbt-labs/dbt-core) - 用软件工程的最佳实践来转换数据。分析工程领域的行业标准框架，月下载量 1500 万+。为 SQL 转换提供版本控制、测试与文档能力。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/dbt-labs/dbt-core?style=social)
- [SQLMesh](https://github.com/TobikoData/sqlmesh) - 可扩展且高效的数据转换框架，兼容 dbt。具备自动数据血缘、时间旅行与用于测试的虚拟数据环境等特性。针对大规模数据仓库优化。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/TobikoData/sqlmesh?style=social)
- [SLayer](https://github.com/MotleyAI/slayer) - 面向 AI 驱动数据分析的语义层。让 AI 智能体用包含度量、维度与过滤器的表达式格式描述数据模型并查询数据，无需编写原始 SQL。提供 MCP、CLI、API 与 Python 客户端，可作为 Python 库嵌入。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/MotleyAI/slayer?style=social)
- [WrenAI](https://github.com/Canner/WrenAI) - 开源生成式 BI 引擎与上下文层，供 AI 智能体查询数据库并生成可信的 SQL 与仪表盘。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Canner/WrenAI?style=social)

#### 数据质量与校验

- [Deequ](https://github.com/awslabs/deequ) - 构建在 Apache Spark 之上的库，用于定义"数据单元测试"。通过约束校验、异常检测与增量校验来衡量大型数据集的数据质量。已在 Amazon 用于生产环境的数据质量保障。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/awslabs/deequ?style=social)
- [Great Expectations](https://github.com/great-expectations/great_expectations) - 随时了解应该对数据抱有什么预期。为数据流水线提供数据校验、剖析与文档。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/great-expectations/great_expectations?style=social)
- [ydata-profiling](https://github.com/ydataai/ydata-profiling) - 一行代码完成全面的数据质量剖析与探索性数据分析。为 Pandas 与 Spark DataFrame 生成详细报告，涵盖统计量、相关性、缺失值与数据质量告警。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/ydataai/ydata-profiling?style=social)
- [Soda Core](https://github.com/sodadata/soda-core) - 面向现代数据栈的数据契约引擎。用 YAML 定义数据质量检查，并在各条流水线上自动校验 schema 与数据。支持 Snowflake、BigQuery、PostgreSQL 等 20+ 数据源。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/sodadata/soda-core?style=social)
- [TFX (TensorFlow Extended)](https://github.com/tensorflow/tfx) - 部署生产级 ML 流水线的端到端平台。基于 TensorFlow 打通数据校验、转换、模型训练与服务化。支撑 Google 的生产 ML 基础设施。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/tensorflow/tfx?style=social)

#### 数据标注与注释

- [Doccano](https://github.com/doccano/doccano) - 面向机器学习从业者的开源文本标注工具。支持文本分类、序列标注与序列到序列任务，可用于情感分析、NER 与摘要生成。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/doccano/doccano?style=social)
- [OpenRefine](https://github.com/OpenRefine/OpenRefine) - 免费开源的脏数据处理利器。借助 Web 服务完成数据清洗、转换与扩展。前身为 Google Refine。BSD-3-Clause 许可。 ![GitHub stars](https://img.shields.io/github/stars/OpenRefine/OpenRefine?style=social)

#### AutoML 与超参数优化

- [Optuna](https://github.com/optuna/optuna) - 现代的 define-by-run 超参数优化框架，支持剪枝与可视化。在 2026 年极受欢迎。 ![GitHub stars](https://img.shields.io/github/stars/optuna/optuna?style=social)
- [AutoGluon](https://github.com/autogluon/autogluon) - AWS 出品的 AutoML 工具包，覆盖表格、图像、文本与多模态数据——几乎零代码即可达到最先进水平。 ![GitHub stars](https://img.shields.io/github/stars/autogluon/autogluon?style=social)
- [FLAML](https://github.com/microsoft/FLAML) - 微软快速轻量的 AutoML 方案，聚焦效率与低算力开销。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/FLAML?style=social)
- [Katib (Kubeflow)](https://github.com/kubeflow/katib) - Kubernetes 原生的 AutoML 框架，支持超参数调优、早停与神经架构搜索。框架无关，可对接 TensorFlow、PyTorch、XGBoost 及自定义训练算子。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/kubeflow/katib?style=social)

#### 交互式 ML 应用与笔记本

- [Streamlit](https://github.com/streamlit/streamlit) - 构建与分享数据应用最快的方式。用极少代码即可把 Python 脚本变成精美的 Web 应用。广泛用于 ML 模型演示、数据可视化与内部工具。 ![GitHub stars](https://img.shields.io/github/stars/streamlit/streamlit?style=social)
- [Gradio](https://github.com/gradio-app/gradio) - 纯 Python 就能构建并分享令人愉悦的机器学习应用。从函数签名自动生成 UI，是创建交互式 ML 演示的事实标准。支撑了成千上万个 Hugging Face Space。 ![GitHub stars](https://img.shields.io/github/stars/gradio-app/gradio?style=social)
- [Marimo](https://github.com/marimo-team/marimo) - Python 的响应式笔记本——运行可复现实验、用 SQL 查询、可作为脚本执行、可作为应用部署、用 git 做版本管理。完全以纯 Python 形式存储。全部集成在现代的 AI 原生编辑器中。 ![GitHub stars](https://img.shields.io/github/stars/marimo-team/marimo?style=social)

#### 模型训练与优化工具

- [Hugging Face Accelerate](https://github.com/huggingface/accelerate) - 简单的 API，让训练脚本只需极小改动即可在任何硬件上运行（多 GPU、TPU、混合精度）。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/accelerate?style=social)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - 微软的深度学习优化库，面向超大规模训练（ZeRO、卸载、MoE）。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/DeepSpeed?style=social)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - 快速且精确的注意力 kernel，降低显存占用并加速 Transformer 的训练与推理。 ![GitHub stars](https://img.shields.io/github/stars/Dao-AILab/flash-attention?style=social)
- [xFormers](https://github.com/facebookresearch/xformers) - 面向 PyTorch 优化的 Transformer 基础组件与注意力算子。 ![GitHub stars](https://img.shields.io/github/stars/facebookresearch/xformers?style=social)
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning) - PyTorch 的高级封装层，消除样板代码并内置最佳实践。 ![GitHub stars](https://img.shields.io/github/stars/Lightning-AI/lightning?style=social)
- [fastai](https://github.com/fastai/fastai) - 深度学习库，为从业者提供达成先进水平的高级组件。构建于 PyTorch 之上，注重易用性与迁移学习。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/fastai/fastai?style=social)
- [PyTorch Ignite](https://github.com/pytorch/ignite) - 在 PyTorch 中训练与评估神经网络的高级库，提供引擎、事件与处理器系统以获得最大的灵活性。BSD-3-Clause 许可。 ![GitHub stars](https://img.shields.io/github/stars/pytorch/ignite?style=social)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - 跨硬件为 ONNX 模型提供高性能推理与训练。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/onnxruntime?style=social)
- [einops](https://github.com/arogozhnikov/einops) - 灵活强大的张量操作库，让代码更易读、更可靠。支持 PyTorch、JAX、TensorFlow、NumPy、MLX。 ![GitHub stars](https://img.shields.io/github/stars/arogozhnikov/einops?style=social)
- [safetensors](https://github.com/huggingface/safetensors) - 存储与分发张量的简单安全方案。模型序列化上比 pickle 更快更安全的替代品。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/safetensors?style=social)
- [torchmetrics](https://github.com/Lightning-AI/torchmetrics) - 面向分布式、可扩展 PyTorch 应用的机器学习指标库。80+ 指标，内置分布式同步。 ![GitHub stars](https://img.shields.io/github/stars/Lightning-AI/torchmetrics?style=social)
- [torchao](https://github.com/pytorch/ao) - PyTorch 原生的量化与稀疏化方案，用于训练与推理。为生产部署提供即插即用的优化。 ![GitHub stars](https://img.shields.io/github/stars/pytorch/ao?style=social)
- [SHAP](https://github.com/shap/shap) - 用博弈论方法解释任意机器学习模型的输出。模型可解释性的行业标准。 ![GitHub stars](https://img.shields.io/github/stars/shap/shap?style=social)
- [skorch](https://github.com/skorch-dev/skorch) - 封装 PyTorch 的 scikit-learn 兼容神经网络库。将 PyTorch 模型无缝接入 scikit-learn 的流水线、网格搜索与交叉验证。 ![GitHub stars](https://img.shields.io/github/stars/skorch-dev/skorch?style=social)
- [Composer](https://github.com/mosaicml/composer) - 为你的模型训练提速。MosaicML 的 PyTorch 训练库，内置高效训练算法（FSDP、梯度压缩、渐进式缩放），并支持在大规模集群上无缝分布式训练。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/mosaicml/composer?style=social)
- [NVIDIA Apex](https://github.com/NVIDIA/apex) - 用于混合精度训练与分布式训练优化的 PyTorch 扩展。提供自动混合精度（AMP）、分布式数据并行与融合优化器等工具，支撑众多生产级深度学习工作负载。BSD-3-Clause 许可。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/apex?style=social)

---

## 2. 模型代码库与模型家族

> 各模型家族的权威代码库，附带实用代码、训练配方、评估工具或工程背景。这不是完整的模型排行榜；如需获取最新的权重，请使用 Hugging Face 与各类模型中心。

#### 语言模型家族

- [RWKV](https://github.com/BlinkDL/RWKV-LM) - 无注意力机制的语言模型架构，推理时间为线性，提供训练代码、推理示例与活跃的开源生态。 ![GitHub stars](https://img.shields.io/github/stars/BlinkDL/RWKV-LM?style=social)
- [MiniCPM](https://github.com/OpenBMB/MiniCPM) - 紧凑的开放模型家族，提供实用代码、部署说明，并持续关注边缘端与端侧场景。 ![GitHub stars](https://img.shields.io/github/stars/OpenBMB/MiniCPM?style=social)
- [GPT-OSS](https://github.com/openai/gpt-oss) - OpenAI 的开放权重模型仓库，附推理示例、训练配方与部署指南。 ![GitHub stars](https://img.shields.io/github/stars/openai/gpt-oss?style=social)
- [Mamba](https://github.com/state-spaces/mamba) - 状态空间模型实现，含预训练检查点、架构代码与面向长序列高效建模的研究工具。 ![GitHub stars](https://img.shields.io/github/stars/state-spaces/mamba?style=social)
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) - EleutherAI 的大规模语言模型训练代码库，支持分布式训练，在开放模型史上具有重要地位。 ![GitHub stars](https://img.shields.io/github/stars/EleutherAI/gpt-neox?style=social)
- [GLM-5](https://github.com/zai-org/GLM-5) - 开源混合专家（MoE）语言模型家族，针对长程规划、智能体任务与代码生成优化。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/zai-org/GLM-5?style=social)
- [Bonsai Demo](https://github.com/PrismML-Eng/Bonsai-demo) - Bonsai 三值推理模型家族的本地运行时与模型包，支持 GGUF 与 MLX，可满足端侧的文本、视觉与工具调用负载。Apache-2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/PrismML-Eng/Bonsai-demo?style=social)

#### 多模态与视觉-语言代码库

- [OpenCLIP](https://github.com/mlfoundations/open_clip) - CLIP 的开放实现，含训练代码、预训练模型与零样本评估工具。 ![GitHub stars](https://img.shields.io/github/stars/mlfoundations/open_clip?style=social)
- [OmniParser](https://github.com/microsoft/OmniParser) - 基于视觉的 GUI 解析模型与配套工具，面向计算机操作类智能体。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/OmniParser?style=social)
- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) - 紧凑的视觉-语言模型家族，提供面向边缘端的部署示例，在 OCR 相关场景表现突出。 ![GitHub stars](https://img.shields.io/github/stars/OpenBMB/MiniCPM-V?style=social)
- [Eagle](https://github.com/NVlabs/Eagle) - NVIDIA 的多模态模型代码库，含开放检查点与可复用的视觉-语言、视频-语言任务研究素材。 ![GitHub stars](https://img.shields.io/github/stars/NVlabs/Eagle?style=social)
- [Moondream](https://github.com/m87-labs/moondream) - 小型视觉-语言模型，面向边缘端与实时图像理解提供实用推理示例。 ![GitHub stars](https://img.shields.io/github/stars/m87-labs/moondream?style=social)
- [NVIDIA Cosmos](https://github.com/NVIDIA/cosmos) - 面向物理 AI、机器人与自主系统的开放平台，含世界模型、分词器与后训练工具。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/cosmos?style=social)

#### 语音与音频模型代码库

- [Whisper](https://github.com/openai/whisper) - 权威的开源语音转文字模型代码库，生态支持广泛，拥有众多下游实现。 ![GitHub stars](https://img.shields.io/github/stars/openai/whisper?style=social)
- [FunASR](https://github.com/modelscope/FunASR) - 语音识别工具包，含预训练模型、流式支持、说话人分离、VAD 与面向生产的示例。 ![GitHub stars](https://img.shields.io/github/stars/modelscope/FunASR?style=social)
- [NVIDIA NeMo](https://github.com/NVIDIA-NeMo/NeMo) - 面向语音、语言与多模态 AI 的可扩展框架与模型代码库，附训练配方与部署指南。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA-NeMo/NeMo?style=social)
- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) - 完整的语音工具包，通过 ONNX Runtime 在嵌入式与边缘环境中提供 ASR、TTS、说话人分离、源分离与 VAD。 ![GitHub stars](https://img.shields.io/github/stars/k2-fsa/sherpa-onnx?style=social)
- [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) - 开放的语音与声音生成家族，专注富有表现力的长文本语音合成，支持流式与多说话人。 ![GitHub stars](https://img.shields.io/github/stars/OpenMOSS/MOSS-TTS?style=social)
- [VoxCPM](https://github.com/OpenBMB/VoxCPM) - 开源的无分词器多语言语音合成模型，具备高质量 TTS 与风格迁移工作流。 ![GitHub stars](https://img.shields.io/github/stars/OpenBMB/VoxCPM?style=social)
- [VibeVoice](https://github.com/microsoft/VibeVoice) - 前沿的开放语音 AI 工具包，涵盖语音理解、生成与多语言 TTS 工作流，并配有活跃的研究与部署工具。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/VibeVoice?style=social)
- [SpeechBrain](https://github.com/speechbrain/speechbrain) - PyTorch 语音工具包，提供 ASR、TTS、说话人识别与语音增强的训练配方。 ![GitHub stars](https://img.shields.io/github/stars/speechbrain/speechbrain?style=social)
- [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) - 轻量级语音合成引擎，针对 CPU 推理优化，低延迟并支持流式。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/kyutai-labs/pocket-tts?style=social)
- [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) - C/C++ 语音转文字推理库，基于 ggml 运行时运行 16+ 个模型家族，并支持 GPU 加速。 ![GitHub stars](https://img.shields.io/github/stars/handy-computer/transcribe.cpp?style=social)
- [Moonshine](https://github.com/moonshine-ai/moonshine) - 开源端侧语音 AI 工具包，支持低延迟语音转文字、意图识别与语音合成。 ![GitHub stars](https://img.shields.io/github/stars/moonshine-ai/moonshine?style=social)

---
## 3. 推理引擎与服务化

> 用于在本地或生产环境运行模型的推理运行时、服务系统与优化工具。

#### 本地 / 端侧推理

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - 纯 C/C++ 推理引擎，支持 GGUF 格式。是 CPU/GPU/Apple Silicon 端侧运行的金标准。内置 llama-server，提供 OpenAI 兼容 API。现已获得 100K+ stars。 ![GitHub stars](https://img.shields.io/github/stars/ggml-org/llama.cpp?style=social)
- [Ollama](https://github.com/ollama/ollama) - 极简的本地 LLM 运行器，一行命令安装，附带模型注册表和 OpenAI 兼容 API。 ![GitHub stars](https://img.shields.io/github/stars/ollama/ollama?style=social)
- [Foundry Local](https://github.com/microsoft/Foundry-Local) - 开源端侧 AI 平台，涵盖模型发现、模型运行、沙箱执行与开放模型评估。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/Foundry-Local?style=social)
- [Potato OS](https://github.com/slomin/potato-os) - 面向树莓派 5 和 4 的 Linux 发行版，支持完全本地化的 AI 推理，针对在边缘设备运行开放模型做了优化。 ![GitHub stars](https://img.shields.io/github/stars/slomin/potato-os?style=social)
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm) - 部署引擎，可编译并在浏览器、移动设备和本地硬件上运行 LLM。 ![GitHub stars](https://img.shields.io/github/stars/mlc-ai/mlc-llm?style=social)
- [WebLLM](https://github.com/mlc-ai/web-llm) - 高性能浏览器内 LLM 推理引擎。通过 WebGPU 加速直接在浏览器中运行模型。 ![GitHub stars](https://img.shields.io/github/stars/mlc-ai/web-llm?style=social)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - llama.cpp 的官方 Python 绑定。 ![GitHub stars](https://img.shields.io/github/stars/abetlen/llama-cpp-python?style=social)
- [KoboldCpp](https://github.com/LostRuins/koboldcpp) - 易于使用的 llama.cpp 分支，专注于角色扮演与创意写作。 ![GitHub stars](https://img.shields.io/github/stars/LostRuins/koboldcpp?style=social)
- [RamaLama](https://github.com/containers/ramalama) - 以容器为中心的工具，简化本地 AI 模型服务化。自动检测 GPU、拉取优化过的容器镜像，并以企业级隔离的方式在无 root 容器中安全运行模型。 ![GitHub stars](https://img.shields.io/github/stars/containers/ramalama?style=social)
- [LiteRT](https://github.com/google-ai-edge/LiteRT) - Google 出品、可用于生产环境的端侧 ML 与 GenAI 部署框架。通过统一的边缘优先运行时支持 Android、iOS、Web、桌面和 IoT 目标，并具备 GPU/NPU 加速。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/google-ai-edge/LiteRT?style=social)
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) - 生产就绪的运行时，用于在边缘设备上部署 LLM，具备低延迟推理以及面向移动与嵌入式平台的优化硬件路径。 ![GitHub stars](https://img.shields.io/github/stars/google-ai-edge/LiteRT-LM?style=social)
- [exo](https://github.com/exo-explore/exo) - 将你的所有设备连接成 AI 集群，在本地运行前沿 AI。具备自动设备发现、通过 Thunderbolt 的 RDMA 降低 99% 延迟、感知拓扑的自动并行以及张量并行。使用 MLX 后端在 Apple Silicon 设备间进行分布式推理。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/exo-explore/exo?style=social)
- [ds4](https://github.com/antirez/ds4) - 原生推理引擎，针对 DeepSeek V4 与 GLM 模型优化，支持 Metal、CUDA 和 ROCm。 ![GitHub stars](https://img.shields.io/github/stars/antirez/ds4?style=social)
- [qwen3.8-flash-next-in-c](https://github.com/shyringo/qwen3.8-flash-next-in-c) - Qwen3.8-Flash-Next 的原生 C 实现，用于本地 CPU 推理，提供终端聊天、常驻的 OpenAI 兼容 API 以及可复现的基准测试。 ![GitHub stars](https://img.shields.io/github/stars/shyringo/qwen3.8-flash-next-in-c?style=social)
- [omlx](https://github.com/jundot/omlx) - 以 Apple 生态为中心的推理服务器，面向本地优先的 AI 工作流，提供模型管理、GPU 编排以及用于自托管部署的 OpenAI 兼容 API。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/jundot/omlx?style=social)
- [llmfit](https://github.com/AlexsJones/llmfit) - 终端工具与 TUI，可按硬件规格为 LLM 模型分配合适的规模，并从 GPU、CPU 和 RAM 三个维度评估本地兼容性评分。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/AlexsJones/llmfit?style=social)
- [Needle](https://github.com/cactus-compute/needle) - 紧凑的 45M 参数基础模型与 14MB 推理引擎，用于微型设备上的工具调用与结构化抽取。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/cactus-compute/needle?style=social)
- [Colibri](https://github.com/JustVugg/colibri) - 零依赖 C 推理引擎，通过在磁盘、内存和显存之间流式调度专家，实现在本地运行大型 MoE 模型。 ![GitHub stars](https://img.shields.io/github/stars/JustVugg/colibri?style=social)
- [Claude Code Local](https://github.com/nicedreamzapp/claude-code-local) - MLX 原生服务器，实现 Anthropic Messages API，使未经修改的 Claude Code CLI 可以在 Apple Silicon 上对接本地模型运行，并能解析本地模型的工具调用格式。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/nicedreamzapp/claude-code-local?style=social)
- [nemotron-omni-mlx](https://github.com/nicedreamzapp/nemotron-omni-mlx) - 纯 MLX 运行时，在 Apple Silicon 上运行 NVIDIA Nemotron 3 Nano Omni 的视觉与音频塔，并已针对 NVIDIA 的 PyTorch 参考实现完成一致性测试。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/nicedreamzapp/nemotron-omni-mlx?style=social)
- [Magnitude](https://github.com/magnitudedev/magnitude) - 硬件感知的本地推理引擎，可分析主机硬件、推荐合适的开放模型，并在 Apple Silicon、NVIDIA、AMD 和 CPU 上调优执行。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/magnitudedev/magnitude?style=social)
- [llm-vram-calculator](https://github.com/159753a52/llm-vram-calculator) - 无依赖的 TypeScript 核心，可根据任意 Hugging Face 模型的 config.json 和 safetensors 元数据估算推理所需内存（权重、KV 缓存、开销），覆盖 MoE、MLA、滑动窗口和线性注意力层；可在浏览器中通过 [modelvram.com](https://modelvram.com/llm-vram-calculator/) 运行。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/159753a52/llm-vram-calculator?style=social)
- [llmash](https://github.com/omgitsbase/llmash) - 基于 llama.cpp 分支的 Ollama 兼容 LLM 服务器，针对 NVIDIA GPU 上的投机解码做了调优。保留 Ollama 的命令、API 和模型仓库，使用模型自带的 MTP 头进行草稿生成，并可在加载时应用控制向量来解除模型审查。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/omgitsbase/llmash?style=social)

#### 高性能服务化与 API 服务器

- [llm-d](https://github.com/llm-d/llm-d) - Kubernetes 原生的分布式 LLM 推理框架。由 RedHat、Google 和 IBM 捐赠给 CNCF。具备智能调度、KV 缓存优化，并跨加速器实现业界领先的性能。 ![GitHub stars](https://img.shields.io/github/stars/llm-d/llm-d?style=social)
- [LMDeploy](https://github.com/InternLM/lmdeploy) - 来自 OpenMMLab 的 LLM 压缩、部署与服务化工具包。4-bit 推理性能比 FP16 高 2.4 倍，支持跨机器的分布式多模型服务化。 ![GitHub stars](https://img.shields.io/github/stars/InternLM/lmdeploy?style=social)
- [vLLM](https://github.com/vllm-project/vllm) - 采用 PagedAttention 和连续批处理的最先进服务引擎。目前最快的生产级 LLM 服务器。 ![GitHub stars](https://img.shields.io/github/stars/vllm-project/vllm?style=social)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) - 多模态推理栈，在 vLLM 基础上扩展图像、音频和视频生成负载，提供统一的服务接口。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/vllm-project/vllm-omni?style=social)
- [LMCache](https://github.com/LMCache/LMCache) - 用最快的 KV Cache 层为 LLM 推理提速。多轮问答和 RAG 可节省 3-10 倍延迟并减少 GPU 占用周期。与 vLLM 无缝集成，适用于分布式高吞吐部署。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/LMCache/LMCache?style=social)
- [vLLM Production Stack](https://github.com/vllm-project/production-stack) - 面向 vLLM 推理的 Kubernetes 原生生产栈。为企业级 LLM 服务化提供自动化部署、自动扩缩容和监控。由 vLLM 团队打造，集成顺畅。 ![GitHub stars](https://img.shields.io/github/stars/vllm-project/production-stack?style=social)
- [Open Model Engine (OME)](https://github.com/sgl-project/ome) - 面向 LLM 服务化的 Kubernetes Operator，支持 GPU 调度以及跨 vLLM、SGLang 和 TensorRT-LLM 的模型生命周期管理。 ![GitHub stars](https://img.shields.io/github/stars/sgl-project/ome?style=social)
- [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) - 约 1,200 行 Python 的极简 vLLM 实现。兼具教学价值与性能，支持前缀缓存、张量并行和 CUDA graph 加速。推理速度可与完整版 vLLM 媲美。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/GeeeekExplorer/nano-vllm?style=social)
- [SGLang](https://github.com/sgl-project/sglang) - 采用 RadixAttention 的新一代服务框架。在 100K+ GPU 规模上支撑 xAI 的生产负载。 ![GitHub stars](https://img.shields.io/github/stars/sgl-project/sglang?style=social)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA 官方的高性能推理后端。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social)
- [Aphrodite Engine](https://github.com/aphrodite-engine/aphrodite-engine) - 针对角色扮演与创意写作优化的 vLLM 分支。支持多种量化方法（AQLM、AWQ、GPTQ、GGUF、FP8）和现代采样器。开发活跃，支持多 LoRA 与投机解码。 ![GitHub stars](https://img.shields.io/github/stars/aphrodite-engine/aphrodite-engine?style=social)
- [AIBrix](https://github.com/vllm-project/aibrix) - 面向 GenAI 推理的高性价比、可插拔基础设施组件。为 vLLM 提供 Kubernetes 原生控制平面，具备分布式 KV 缓存、异构 GPU 服务化和智能路由。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/vllm-project/aibrix?style=social)
- [AISIX](https://github.com/api7/aisix) - 自托管、采用 Apache-2.0 许可的 AI 网关，使用 Rust 编写，提供 OpenAI 兼容端点、Anthropic 兼容 Messages 端点、模型路由、流量策略、Prometheus 指标以及 OTLP 追踪导出。 ![GitHub stars](https://img.shields.io/github/stars/api7/aisix?style=social)
- [Triton Inference Server](https://github.com/triton-inference-server/server) - NVIDIA 的生产级开源推理服务软件。支持多种框架（TensorRT、PyTorch、ONNX），针对云端与边缘部署做了优化。 ![GitHub stars](https://img.shields.io/github/stars/triton-inference-server/server?style=social)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) - 基于 Candle 构建的快速、灵活的 Rust 原生 LLM 推理引擎。支持文本、视觉、音频、图像生成与嵌入，并具备硬件感知的自动调优。 ![GitHub stars](https://img.shields.io/github/stars/EricLBuehler/mistral.rs?style=social)
- [KTransformers](https://github.com/kvcache-ai/ktransformers) - 灵活的异构 CPU-GPU LLM 推理与微调框架。通过将专家卸载到 CPU，并支持 BF16/FP8 精度，让大型 MoE 模型得以运行。 ![GitHub stars](https://img.shields.io/github/stars/kvcache-ai/ktransformers?style=social)
- [llamafile](https://github.com/mozilla-ai/llamafile) - Mozilla 的单文件可分发 LLM 方案。把模型权重、推理引擎和运行时打包进一个便携可执行文件，无需安装即可在六种操作系统上运行。 ![GitHub stars](https://img.shields.io/github/stars/mozilla-ai/llamafile?style=social)
- [Xinference](https://github.com/xorbitsai/inference) - 统一、生产就绪的推理 API，覆盖 LLM、语音和多模态模型。仅需一行代码改动即可作为 GPT 的替代方案。支持数千个模型，具备自动批处理和分布式推理。 ![GitHub stars](https://img.shields.io/github/stars/xorbitsai/inference?style=social)
- [RTP-LLM (Alibaba)](https://github.com/alibaba/rtp-llm) - 阿里高性能 LLM 推理加速引擎。支撑淘宝、天猫及阿里国际 AI 平台的生产级 LLM 服务。支持 PagedAttention、FlashAttention、FlashDecoding、INT8/INT4 量化以及异构硬件（GPU/ARM CPU/Intel）。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/alibaba/rtp-llm?style=social)
- [LitServe (Lightning AI)](https://github.com/Lightning-AI/LitServe) - 极简 Python 框架，用于构建自定义 AI 推理服务器，完全掌控逻辑、批处理与扩缩容。相比 FastAPI 快 2 倍，内置批处理、流式传输和多 GPU 自动扩缩容。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Lightning-AI/LitServe?style=social)
- [LightLLM](https://github.com/ModelTC/LightLLM) - 纯 Python 的 LLM 推理与服务化框架，设计轻量、易扩展、性能高速。整合了 FasterTransformer、TGI、vLLM 和 SGLang 的优化经验。 ![GitHub stars](https://img.shields.io/github/stars/ModelTC/LightLLM?style=social)
- [TabbyAPI](https://github.com/theroyallab/tabbyAPI) - 基于 FastAPI 的 API 服务器，后端为 ExLlamaV2/V3。提供 OpenAI 兼容 API，支持模型加载/卸载、嵌入、投机解码、多 LoRA 和流式输出。 ![GitHub stars](https://img.shields.io/github/stars/theroyallab/tabbyAPI?style=social)
- [GPUStack](https://github.com/gpustack/gpustack) - GPU 集群管理器，可编排 vLLM、SGLang 等推理引擎。支持自动引擎选择、参数优化和分布式多 GPU 部署，以承载高性能 AI 负载。 ![GitHub stars](https://img.shields.io/github/stars/gpustack/gpustack?style=social)
- [OpenLLM (BentoML)](https://github.com/bentoml/OpenLLM) - 生产级平台，可将任意开源 LLM 作为 OpenAI 兼容 API 端点运行。支持 50+ 模型，内置流式传输、批处理和自动加速。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/bentoml/OpenLLM?style=social)
- [Higress (Alibaba)](https://github.com/alibaba/higress) - 源自阿里内部基础设施、经 2 年以上生产验证的 AI 原生 API 网关。提供统一的 LLM API 与 MCP（Model Context Protocol）管理，具备企业级 99.99% 可用性。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/alibaba/higress?style=social)
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) - NVIDIA 的数据中心级分布式推理服务框架。位于 vLLM/SGLang/TensorRT-LLM 之上的编排层，支持分离式服务化、KV 感知路由和自动扩缩容。以 Rust 构建，并具备 Python 可扩展性。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/ai-dynamo/dynamo?style=social)
- [Microsoft BitNet](https://github.com/microsoft/BitNet) - 面向 1-bit LLM（BitNet b1.58）的官方推理框架。让大型模型能在 CPU 上以极小内存占用运行。提供三值权重量化的定制内核和高效矩阵乘运算。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/BitNet?style=social)
- [FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi) - OpenAI 兼容的代理网关，把多家 LLM 提供商的免费额度汇聚到单一端点，具备自动故障切换和速率追踪。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/tashfeenahmed/freellmapi?style=social)
- [OmniRoute](https://github.com/diegosouzapw/OmniRoute) - 统一 AI 网关与代理，支持 230 多家提供商，具备令牌压缩、自动故障切换和路由策略。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/diegosouzapw/OmniRoute?style=social)
- [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) - Rust 代理与库，用于跨 LLM 后端和编码智能体进行路由、协议转换与运行指标采集。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA-NeMo/Switchyard?style=social)
- [Bifrost](https://github.com/maximhq/bifrost) - LLM 网关，提供跨提供商的统一 OpenAI 兼容 API，并涵盖路由、负载均衡、降级、护栏与可观测性。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/maximhq/bifrost?style=social)
- [freelm](https://github.com/shihabshahrier/freelm) - Python 与 TypeScript 客户端库，附带可选的本地 OpenAI 兼容服务器，可汇聚 Gemini、Groq、OpenRouter、Cloudflare Workers AI 等提供商的免费额度，支持密钥轮换与故障切换。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/shihabshahrier/freelm?style=social)
- [fitcheck](https://github.com/SIDDARTHAREDDY8/fitcheck) - 零依赖 Python CLI，在服务化之前先规划 LLM 显存占用：真实的逐 token KV 缓存计算、FITS/TIGHT/OOM 判定、最大安全上下文长度，以及可直接粘贴的 vLLM/llama.cpp 启动参数。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/SIDDARTHAREDDY8/fitcheck?style=social)

#### 其他推理引擎

- [AirLLM](https://github.com/lyogavin/airllm) - 单 GPU 运行 70B 的推理栈，针对在普通硬件上本地部署做了深度内存/性能优化。 ![GitHub stars](https://img.shields.io/github/stars/lyogavin/airllm?style=social)
- [Mirage Persistent Kernel](https://github.com/mirage-project/mirage) - 编译器，将模型执行融合为单个巨型内核，以获得更紧凑的性能。 ![GitHub stars](https://img.shields.io/github/stars/mirage-project/mirage?style=social)
- [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend) - 在华为昇腾加速器上运行 vLLM 的硬件插件。 ![GitHub stars](https://img.shields.io/github/stars/vllm-project/vllm-ascend?style=social)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) - 面向 Transformer 模型的高速推理引擎，支持 OpenNMT 和 Hugging Face 模型。针对 CPU 和 GPU 做了优化，提供批处理、量化（INT8/FP16）和动态内存管理。为 faster-whisper 等生产部署提供动力。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/OpenNMT/CTranslate2?style=social)
- [llama-swap](https://github.com/mostlygeek/llama-swap) - 面向 llama.cpp 的智能模型切换代理。支持在不重启服务器的情况下在不同 GGUF 模型间无缝热切换，自动加载/卸载模型，并提供 OpenAI 兼容 API。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/mostlygeek/llama-swap?style=social)
- [optillm](https://github.com/algorithmicsuperintelligence/optillm) - 面向 LLM 的优化推理代理，支持跨多家提供商和模型的负载均衡、故障切换与请求路由。提升生产部署的可靠性与性能。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/algorithmicsuperintelligence/optillm?style=social)
- [Fugusashi](https://github.com/eulogik/fugusashi) - 开源智能模型路由器，具备 CMA-ES 演化路由权重、用于隐私保护型协同路由的联邦学习，以及对每个决策的人类可解释说明。提供 OpenAI 兼容 API 与 Web 仪表盘。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/eulogik/fugusashi?style=social)
- [OpenRoutiQ](https://github.com/antat-ai/openroutiq) - 可解释、受策略控制的 Python 路由器，可选择模型、提供商、部署方式和推理级别，并可选支持结果学习以及 OpenAI 兼容代理。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/antat-ai/openroutiq?style=social)
- [mllm](https://github.com/UbiquitousLearning/mllm) - 面向移动与边缘设备的高速轻量多模态 LLM 推理引擎。针对资源受限硬件运行视觉-语言模型做了优化，具备高效的内存管理。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/UbiquitousLearning/mllm?style=social)
- [shimmy](https://github.com/Michael-A-Kuykendall/shimmy) - 无需 Python 的 Rust 推理服务器，兼容 OpenAI API。支持 GGUF 和 SafeTensors 格式，具备模型热切换、自动发现和单二进制部署，实现零依赖推理。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Michael-A-Kuykendall/shimmy?style=social)
- [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) - 面向消费级 GPU 本地部署的高速 LLM 推理。利用幂律神经元激活模式，在 RTX 4090 上相较 llama.cpp 最高可获得 11 倍加速。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/SJTU-IPADS/PowerInfer?style=social)
- [distributed-llama](https://github.com/b4rtaz/distributed-llama) - 分布式 LLM 推理，把家用设备连接成强大集群。设备越多，通过以太网张量并行实现的推理就越快。支持 Linux、macOS、Windows、ARM 和 x86_64 AVX2 CPU。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/b4rtaz/distributed-llama?style=social)
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) - 高性能 llama.cpp 分支，具备更优的 CPU 和 GPU/CPU 混合性能、SOTA 量化类型、一等公民的 BitNet 支持，以及通过 MLA、FlashMLA 和融合 MoE 运算提升的 DeepSeek 性能。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/ikawrakow/ik_llama.cpp?style=social)
- [xLLM](https://github.com/jd-opensource/xllm) - 面向中国 AI 加速器（寒武纪 MLU、海光 DCU、华为昇腾）优化的高性能推理引擎。具备服务引擎解耦架构，支持弹性调度、PD 分离和全局 KV 缓存管理。支撑京东的核心零售业务。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/jd-opensource/xllm?style=social)
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - Kimi（月之暗面）专用的生产级服务化平台。具备带智能卸载的分布式 KV 缓存池、prefill/decode 分离以及跨实例 KV 复用。与 vLLM、SGLang 和 TensorRT-LLM 集成。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/kvcache-ai/Mooncake?style=social)
- [gemma.cpp](https://github.com/google/gemma.cpp) - 面向 Google Gemma 模型的轻量独立 C++ 推理引擎。依赖极少、内存占用高效，针对端侧部署做了优化。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/google/gemma.cpp?style=social)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - 面向 LLM 服务化的内核库。提供高性能 CUDA 内核，覆盖注意力、采样和矩阵乘法。以优化的 GPU 内核支撑 vLLM、SGLang 等推理引擎。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/flashinfer-ai/flashinfer?style=social)

#### 推理内核与运行时原语

- [DeepEP](https://github.com/deepseek-ai/DeepEP) - 面向专家并行训练/推理流水线的高效通信库，降低 MoE 跨设备通信开销。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/deepseek-ai/DeepEP?style=social)
- [FlashKDA](https://github.com/MoonshotAI/FlashKDA) - 基于 CUTLASS 的高性能 Kimi Delta Attention CUDA 内核，用于线性注意力机制。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/MoonshotAI/FlashKDA?style=social)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) - 简洁的高性能 FP8 GEMM 内核，带细粒度缩放，面向现代推理负载。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/deepseek-ai/DeepGEMM?style=social)
- [RAFT](https://github.com/rapidsai/raft) - CUDA 加速算法与 ANN 构建模块，用于高性能相似度搜索、聚类和矩阵学习负载。 ![GitHub stars](https://img.shields.io/github/stars/rapidsai/raft?style=social)
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) - 用于加速 transformer 注意力块的 CUDA tile 原语与内核模板。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/HazyResearch/ThunderKittens?style=social)
- [tt-metal](https://github.com/tenstorrent/tt-metal) - TT-Metalium + TT-NN 算子栈，用于在 Tenstorrent AI 加速器上构建和优化内核。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/tenstorrent/tt-metal?style=social)
- [mini-sglang](https://github.com/sgl-project/mini-sglang) - 紧凑的 SGLang 实现，旨在揭示现代 LLM 服务化系统的原理。兼具教学价值与生产级质量，包含 RadixAttention、连续批处理和投机解码。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/sgl-project/mini-sglang?style=social)

#### 量化、蒸馏与优化

- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) - 8-bit 和 4-bit 优化器与量化。 ![GitHub stars](https://img.shields.io/github/stars/bitsandbytes-foundation/bitsandbytes?style=social)
- [Optimum](https://github.com/huggingface/optimum) - 面向特定硬件的加速与量化。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/optimum?style=social)

---

## 4. 智能体 AI 与多智能体系统

> 用于构建智能体系统和多智能体工作流的框架与平台。

#### 单智能体框架

- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) - 引发智能体革命的开创性自主 AI 智能体框架。愿景是让人人都能用上 AI，提供模块化智能体架构、基准测试以及基于 forge 的智能体构建方式。183k+ stars。 ![GitHub stars](https://img.shields.io/github/stars/Significant-Gravitas/AutoGPT?style=social)
- [LangGraph](https://github.com/langchain-ai/langgraph) - 有状态、可控的智能体编排。 ![GitHub stars](https://img.shields.io/github/stars/langchain-ai/langgraph?style=social)
- [CrewAI](https://github.com/crewAIInc/crewAI) - 基于角色的智能体框架。 ![GitHub stars](https://img.shields.io/github/stars/crewAIInc/crewAI?style=social)
- [AutoGen (AG2)](https://github.com/microsoft/autogen) - 灵活的多智能体对话框架。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/autogen?style=social)
- [DSPy](https://github.com/stanfordnlp/dspy) - 用于以模块、优化器和评估回路来编程语言模型流水线的框架。 ![GitHub stars](https://img.shields.io/github/stars/stanfordnlp/dspy?style=social)
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel) - 面向多种编程语言构建和编排 AI 智能体与工作流的 SDK。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/semantic-kernel?style=social)
- [smolagents](https://github.com/huggingface/smolagents) - 围绕工具使用和代码执行工作流打造的轻量智能体框架。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/smolagents?style=social)
- [LangChain](https://github.com/langchain-ai/langchain) - 面向智能体、链和记忆的基础库。 ![GitHub stars](https://img.shields.io/github/stars/langchain-ai/langchain?style=social)
- [Neuron AI](https://github.com/neuron-core/neuron-ai) - PHP 智能体框架，用于构建生产就绪的 AI 驱动应用。把组件（LLM、向量数据库、记忆）连接到可与你的数据交互的智能体上。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/neuron-core/neuron-ai?style=social)
- [II-Agent (Intelligent Internet)](https://github.com/Intelligent-Internet/ii-agent) - 全新开源框架，用于构建和部署智能体，支持 Claude、Gemini 和 OpenAI 模型。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Intelligent-Internet/ii-agent?style=social)
- [Hermes Agent (NousResearch)](https://github.com/NousResearch/hermes-agent) - 与你一起成长的智能体。具备持久记忆的服务端自主智能体，能够随时间学习和改进。 ![GitHub stars](https://img.shields.io/github/stars/NousResearch/hermes-agent?style=social)
- [Strands Agents](https://github.com/strands-agents/sdk-python) - 模型驱动的 AI 智能体构建方式，只需几行代码。支持多智能体系统、自主智能体和流式处理，内置 MCP。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/strands-agents/sdk-python?style=social)
- [Agno](https://github.com/agno-agi/agno) - 大规模构建、运行和管理智能体软件。面向多智能体系统的高性能框架，涵盖记忆、知识和工具。 ![GitHub stars](https://img.shields.io/github/stars/agno-agi/agno?style=social)
- [Upsonic](https://github.com/Upsonic/Upsonic) - 面向金融科技和银行业的智能体框架，内置 MCP 支持、护栏和工具服务器架构。 ![GitHub stars](https://img.shields.io/github/stars/Upsonic/Upsonic?style=social)
- [VoltAgent](https://github.com/VoltAgent/voltagent) - TypeScript 优先的 AI 智能体工程平台，提供记忆、RAG、工作流、MCP 集成和语音支持。 ![GitHub stars](https://img.shields.io/github/stars/VoltAgent/voltagent?style=social)
- [PocketFlow](https://github.com/The-Pocket/PocketFlow) - 100 行的极简 LLM 框架，用于构建智能体工作流。轻量、可扩展的架构，支持工具使用与自主任务执行。 ![GitHub stars](https://img.shields.io/github/stars/The-Pocket/PocketFlow?style=social)
- [Agent Development Kit (Google)](https://github.com/google/adk-python) - 代码优先的 Python 工具包，用于构建复杂的 AI 智能体，支持多智能体编排、内置评估和灵活部署。模型无关，与 Google 生态深度集成。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/google/adk-python?style=social)
- [PydanticAI](https://github.com/pydantic/pydantic-ai) - 来自 Pydantic 创造者的类型安全 AI 智能体框架。模型无关，支持 20+ 提供商，通过 Logfire 提供内置可观测性，支持 MCP/A2A 协议，并可用 YAML/JSON 定义智能体。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/pydantic/pydantic-ai?style=social)
- [Griptape](https://github.com/griptape-ai/griptape) - 面向 AI 智能体与工作流的模块化 Python 框架，提供思维链推理、工具和记忆。强制约束顺序流水线、DAG 工作流等结构，保证 AI 系统行为可预期。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/griptape-ai/griptape?style=social)
- [Langroid](https://github.com/langroid/langroid) - 用多智能体编程来驾驭 LLM。基于 Pydantic 的成熟工具调用系统，支持包括 OpenAI 和本地服务器在内的数百家 LLM 提供商。为真实场景下稳健的智能体行为而构建。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/langroid/langroid?style=social)
- [Octomind](https://github.com/Muvon/octomind) - 用 Rust 编写的模型无关 AI 智能体运行时，具备专家智能体、MCP 支持、多提供商集成和零配置安装。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Muvon/octomind?style=social)
- [Marvin](https://github.com/PrefectHQ/marvin) - 面向结构化输出和智能体 AI 工作流的 Python 框架。通过类型安全接口、自动 Schema 生成和内置可观测性简化 LLM 交互。出自 Prefect 团队之手。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/PrefectHQ/marvin?style=social)

- [Burr](https://github.com/apache/burr) - Apache 孵化中的框架，用于构建有状态的 AI 应用（聊天机器人、智能体、仿真）。可在自有基础设施上监控、追踪、持久化和执行，并内置 UI 与可插拔记忆。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/apache/burr?style=social)
- [KaibanJS](https://github.com/kaiban-ai/KaibanJS) - JavaScript 原生框架，采用看板启发式方法构建和管理多智能体系统。为 AI 智能体提供可视化任务板和实时协作功能。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/kaiban-ai/KaibanJS?style=social)
- [Jido](https://github.com/agentjido/jido) - 面向 Elixir 的自主智能体框架。基于 actor 模型并发，为分布式自主行为和动态工作流而构建。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/agentjido/jido?style=social)
- [Flue](https://github.com/withastro/flue) - 可编程的 TypeScript harness 与沙箱智能体框架，用于构建自主工作流和智能体。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/withastro/flue?style=social)
- [Agent-Native](https://github.com/BuilderIO/agent-native) - TypeScript 优先的框架，用于构建智能体优先的应用，具备共享数据库状态、实时多人编辑和动作驱动的工具。ISC 许可。 ![GitHub stars](https://img.shields.io/github/stars/BuilderIO/agent-native?style=social)
- [rlm](https://github.com/alexzhang13/rlm) - 面向递归语言模型（RLM）的通用即插即用推理库，可在隔离的代码沙箱中以编程方式执行子 LM 调用。 ![GitHub stars](https://img.shields.io/github/stars/alexzhang13/rlm?style=social)
- [Page Agent](https://github.com/alibaba/page-agent) - JavaScript 原生的页面内 GUI 智能体框架，无需截图或外部浏览器自动化即可用自然语言控制网页界面。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/alibaba/page-agent?style=social)
- [Computer (Cloudflare)](https://github.com/cloudflare/computer) - Durable Object 内部的虚拟文件系统，为 AI 智能体提供沙箱执行环境。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/cloudflare/computer?style=social)
- [Embabel](https://github.com/embabel/embabel-agent) - 用 Kotlin 编写的 JVM 智能体框架，具备动态目标导向规划并集成 Spring Boot。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/embabel/embabel-agent?style=social)
- [Ouroboros](https://github.com/razzant/ouroboros) - 自托管的通用智能体，具备持久身份与记忆、专家子智能体协调，并允许对自身实现进行受审修改。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/razzant/ouroboros?style=social)
- [OpenAmer](https://github.com/openamer/openamer) - 开源 Windows 原生智能体，在进程内运行其认知（五个原生工具：思考、学习、记忆、触发、心跳），并在本地操作真实桌面 —— 文件系统、终端、GUI 和通过 Chrome DevTools Protocol 驱动的浏览器。单一 10 子系统心跳取代 84 条 cron 条目，A2A 网格在实例之间点对点路由工作。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/openamer/openamer?style=social)

#### 多智能体编排

- [ChatDev](https://github.com/OpenBMB/ChatDev) - 多智能体软件开发框架，AI 智能体以程序员、设计师和测试人员的角色协作构建软件。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/OpenBMB/ChatDev?style=social)
- [CAMEL](https://github.com/camel-ai/camel) - 首个也是最好的多智能体框架，用于构建可扩展的智能体系统。采用 Apache 2.0 许可，并提供面向智能体通信和任务自动化的丰富工具。 ![GitHub stars](https://img.shields.io/github/stars/camel-ai/camel?style=social)
- [DeepAgents](https://github.com/langchain-ai/deepagents) - 开箱即用的 LangChain 智能体 harness，用于借助可复用的运行时模式构建和运行结构化的多智能体工作流。 ![GitHub stars](https://img.shields.io/github/stars/langchain-ai/deepagents?style=social)

- [EvoFlux](https://github.com/evoelsewhere/evoflux) - 本地优先的桌面工作区，可跨对话、文件、终端、浏览器、Git、记忆、插件和验证来协调 AI 智能体团队。 ![GitHub stars](https://img.shields.io/github/stars/evoelsewhere/evoflux?style=social)
- [Swarms](https://github.com/kyegomez/swarms) - 前沿的企业级多智能体编排。 ![GitHub stars](https://img.shields.io/github/stars/kyegomez/swarms?style=social)
- [Mastra](https://github.com/mastra-ai/mastra) - TypeScript 优先的智能体框架，内置 RAG、工作流、工具集成、可观测性和观测型记忆。 ![GitHub stars](https://img.shields.io/github/stars/mastra-ai/mastra?style=social)
- [Nika](https://github.com/supernovae-st/nika) - AI 工作流引擎，将智能体工作记录为可审查的 .nika.yaml DAG 文件，执行前做静态检查（schema、权限、成本上限），执行后提供防篡改追踪。本地优先（Ollama、llama.cpp、vLLM），既是 MCP 客户端也是服务器。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/supernovae-st/nika?style=social)
- [Deer-Flow (ByteDance)](https://github.com/bytedance/deer-flow) - 开源的长周期 SuperAgent harness，可做研究、写代码和创作。借助沙箱、记忆、工具、技能、子智能体和消息网关，处理从几分钟到数小时的任务。 ![GitHub stars](https://img.shields.io/github/stars/bytedance/deer-flow?style=social)
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) - 生产就绪的轻量多智能体工作流框架。Swarm 的演进版本，增强了编排能力并提供企业级特性。 ![GitHub stars](https://img.shields.io/github/stars/openai/openai-agents-python?style=social)
- [Symphony](https://github.com/openai/symphony) - 将项目工作转化为隔离的自主实现运行。监控工作看板、派生智能体处理任务，并提供工作证明，包括 CI 状态、PR 评审和演示视频。用于管理编码智能体而非盯人监督的工程预览版。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/openai/symphony?style=social)
- [Paperclip](https://github.com/paperclipai/paperclip) - AI 智能体公司与编排框架，55K+ stars。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/paperclipai/paperclip?style=social)
- [AgentScope](https://github.com/agentscope-ai/agentscope) - 阿里出品、生产就绪的多智能体框架，23K+ stars。内置 MCP 和 A2A 支持，提供消息中枢用于灵活编排，并提供 AgentScope Runtime 用于生产部署。 ![GitHub stars](https://img.shields.io/github/stars/agentscope-ai/agentscope?style=social)
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) - 微软官方框架，融合 AutoGen 的智能体抽象与 Semantic Kernel 的企业特性。支持 Python 和 .NET，提供基于图的工作流。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/agent-framework?style=social)
- [NarraNexus](https://github.com/NetMindAI-Open/NarraNexus) - 开箱即用的 AI 智能体团队工作区，其智能体从第一天起就能记忆、协作并使用工具。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/NetMindAI-Open/NarraNexus?style=social)
- [Agency Swarm](https://github.com/VRSEN/agency-swarm) - 构建在 OpenAI Assistants API 之上的可靠多智能体编排框架，具备组织结构建模。 ![GitHub stars](https://img.shields.io/github/stars/VRSEN/agency-swarm?style=social)
- [elizaOS](https://github.com/elizaOS/eliza) - 用于构建和部署 AI 驱动应用的自主多智能体框架。具备 Discord/Telegram/Farcaster 连接器、RAG 支持和现代化 Web 仪表盘。 ![GitHub stars](https://img.shields.io/github/stars/elizaOS/eliza?style=social)
- [OpenAgents](https://github.com/openagents-org/openagents) - 面向开放协作的 AI 智能体网络。用于构建具备共享知识和分布式任务执行的协作型多智能体系统。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/openagents-org/openagents?style=social)
- [Hive (Aden)](https://github.com/aden-hive/hive) - 生产级多智能体编排框架，10K+ stars。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/aden-hive/hive?style=social)
- [Agent Squad (AWS Labs)](https://github.com/awslabs/agent-squad) - 灵活的多智能体编排框架，具备智能意图分类和上下文管理。支持 Python 和 TypeScript，并预置面向 Bedrock、Lex 及自定义集成的智能体。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/awslabs/agent-squad?style=social)
- [DeepResearchAgent](https://github.com/SkyworkAI/DeepResearchAgent) - 面向深度研究任务的层级化多智能体系统，可在复杂领域自动完成任务拆解与执行。 ![GitHub stars](https://img.shields.io/github/stars/SkyworkAI/DeepResearchAgent?style=social)
- [Composio Agent Orchestrator](https://github.com/ComposioHQ/agent-orchestrator) - 面向并行编码智能体的智能体编排器。规划任务、派生智能体，并自主处理 CI 修复、合并冲突和代码评审。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/ComposioHQ/agent-orchestrator?style=social)
- [Open Multi-Agent](https://github.com/open-multi-agent/open-multi-agent) - MIT 许可的 TypeScript 原生编排框架，可在运行时规划多智能体任务 DAG，并在自有环境中提供审批门禁、追踪、评估、检查点和可恢复执行。 ![GitHub stars](https://img.shields.io/github/stars/open-multi-agent/open-multi-agent?style=social)
- [BeeAI Framework (IBM)](https://github.com/i-am-bee/beeai-framework) - 生产就绪的 Python 和 TypeScript 多智能体框架。具备工作流编排、ACP/MCP 协议支持以及与 watsonx 的深度集成。属于 Linux Foundation AI & Data 计划。 ![GitHub stars](https://img.shields.io/github/stars/i-am-bee/beeai-framework?style=social)
- [AI Town](https://github.com/a16z-infra/ai-town) - 可部署的启动套件，用于构建 AI 角色生活、聊天和社交的虚拟小镇。灵感来自斯坦福的 Generative Agents 研究，具备持久化的智能体记忆和社交互动。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/a16z-infra/ai-town?style=social)
- [Conductor OSS](https://github.com/conductor-oss/conductor) - 事件驱动的智能体编排平台，为应用和 AI 智能体提供持久、弹性执行引擎。已在 Netflix、Tesla、LinkedIn 和 J.P. Morgan 经过实战检验，30K+ stars。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/conductor-oss/conductor?style=social)
- [A2A Protocol](https://github.com/a2aproject/A2A) - Agent2Agent（A2A）开放协议，让不透明的智能体应用之间能够通信和互操作。由 Google 联合 50+ 技术伙伴捐赠给 Linux Foundation。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/a2aproject/A2A?style=social)
- [777genius/agent-teams-ai](https://github.com/777genius/agent-teams-ai) - 开源桌面应用，用于协调自主编码智能体团队，提供智能体间消息传递、看板任务管理和代码评审，覆盖 Claude Code、Codex、OpenCode、Cursor、Grok、GitHub Copilot、Kiro、Z.AI、MiniMax、Kimi 以及 75+ LLM 提供商。 ![GitHub stars](https://img.shields.io/github/stars/777genius/agent-teams-ai?style=social)
- [Panniantong/Agent-Reach](https://github.com/Panniantong/Agent-Reach) - 为 AI 智能体提供可复用的搜索与网页摄取层，通过一个 CLI 覆盖 Reddit、X/Twitter、YouTube、GitHub、Bilibili 等平台。 ![GitHub stars](https://img.shields.io/github/stars/Panniantong/Agent-Reach?style=social)
- [xerrors/Yuxi](https://github.com/xerrors/Yuxi) - 自托管的多租户智能体 harness，融合检索、知识图谱接地和工作流编排，服务生产团队。 ![GitHub stars](https://img.shields.io/github/stars/xerrors/Yuxi?style=social)
- [Sim Studio](https://github.com/simstudioai/sim) - 开源 AI 工作区，用于构建、部署和编排 AI 智能体。可视化画布集成 1000+ 服务，支持多框架（Agno、OpenAI、LangChain、Google ADK），可自托管或云端部署。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/simstudioai/sim?style=social)
- [2FastLabs Agent Squad](https://github.com/2FastLabs/agent-squad) - 灵活、轻量的开源框架，用于编排多个 AI 智能体并具备并行执行能力，以处理复杂对话。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/2FastLabs/agent-squad?style=social)
- [SIA](https://github.com/hexo-ai/sia) - 自我改进框架，编排元智能体、目标智能体和反馈智能体，在基准任务上自主优化 AI 模型与智能体的表现。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/hexo-ai/sia?style=social)
- [Council of High Intelligence](https://github.com/0xNyk/council-of-high-intelligence) - 多智能体审议框架，将不同专家人格路由到不同 LLM 提供商进行辩论并达成共识。 ![GitHub stars](https://img.shields.io/github/stars/0xNyk/council-of-high-intelligence?style=social)
- [Gas Town](https://github.com/gastownhall/gastown) - 面向 Claude Code 等编码智能体的多智能体工作区管理与编排系统，具备持久工作追踪、邮箱和自动合并队列。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/gastownhall/gastown?style=social)
- [Agentlas OS Agent Operation Environment (AOE)](https://github.com/agentlas-ai/Agentlas-OS) - 本地优先、模型无关的环境，用于在受支持的 LLM 宿主上构建、借用和编排专家智能体与团队，具备归属方范围的包和宿主强制的权限控制。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/agentlas-ai/Agentlas-OS?style=social)
- [Open Deep Research](https://github.com/langchain-ai/open_deep_research) - 开源深度研究助手，编排语言模型与搜索工具执行多步网络研究并汇编报告。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/langchain-ai/open_deep_research?style=social)
- [Claudexor](https://github.com/razzant/claudexor) - 本地优先的控制平面，在 Claude Code、Codex、Cursor 和 OpenCode 之间路由编码工作，支持跨多个 Claude/Codex 订阅配额的感知配额轮换、共享线程上下文、best-of-N 运行和跨模型评审。 ![GitHub stars](https://img.shields.io/github/stars/razzant/claudexor?style=social)
- [MiroFish](https://github.com/666ghj/MiroFish) - 多智能体集群智能引擎，构建带有自主智能体的并行数字环境，用于模拟和预测现实世界结果。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/666ghj/MiroFish?style=social)
- [Maka](https://github.com/apache/maka) - Apache 孵化中的本地优先 AI 智能体工作区，具备持久化的仅追加智能体执行历史，采用 Apache-2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/apache/maka?style=social)
- [Ruflo](https://github.com/ruvnet/ruflo) - 用于协调具备共享记忆的智能体集群的多智能体编排与元 harness，采用 MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/ruvnet/ruflo?style=social)
- [munder-difflin](https://github.com/chaitanyagiri/munder-difflin) - MIT 许可的桌面多智能体工作区与 harness，用于跨多个 LLM 后端协调本地编码智能体。 ![GitHub stars](https://img.shields.io/github/stars/chaitanyagiri/munder-difflin?style=social)
- [fractal](https://github.com/plasma-ai/fractal) - 层级化的编码智能体编排器，具备递归委派、每节点 Git worktree、可配置限制、持久化 SQLite 状态以及实时终端监控与干预。 ![GitHub stars](https://img.shields.io/github/stars/plasma-ai/fractal?style=social)
- [AX (Google)](https://github.com/google/ax) - 声明式、高吞吐的智能体编排运行时，面向集群规模的沙箱智能体负载。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/google/ax?style=social)
- [Strands Agent Harness](https://github.com/strands-agents/harness-sdk) - 生产级 SDK 与 harness 运行时，用于在 Python 和 TypeScript 中构建、监控和控制端到端 AI 智能体生命周期。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/strands-agents/harness-sdk?style=social)
- [StarNet](https://github.com/androoAGI/starnet) - 本地优先的桌面多智能体 harness，具备持久工作区、按能力划分的工具、智能体记忆、预算、调度和实时运行时可视化。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/androoAGI/starnet?style=social)
- [OpenRig](https://github.com/mvschwarz/openrig) - 本地多智能体 harness，用于定义持久化的 Claude Code 与 Codex 团队，具备 YAML 拓扑、tmux 会话、持久状态和共享终端 UI。采用 Apache-2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/mvschwarz/openrig?style=social)
- [Raven](https://github.com/EverMind-AI/Raven) - 多智能体 harness，其宿主智能体把复杂任务规划为 DAG 并委派给内置的研究、编码、设计和值班智能体，或通过 ACP、CLI、OpenAI 兼容 API 调用第三方智能体，具备跨会话记忆和实验性的自我进化循环（只安装经过验证的改动）。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/EverMind-AI/Raven?style=social)
- [5dive](https://github.com/5dive-ai/5dive) - 自托管 CLI，在单台 Linux 服务器上运行一支 AI 智能体团队，每个智能体使用独立的 Linux 用户，并以 systemd 服务形式运行 Claude Code、Codex 或其他官方智能体 CLI，共享组织架构和待办列表。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/5dive-ai/5dive?style=social)
- [subagent-tuner](https://github.com/Kruhlikau/subagent-tuner) - Claude Code 修改模块，可通过询问用户、按任务类型推荐或使用已保存的默认值来设置每个子智能体的模型和推理强度。 ![GitHub stars](https://img.shields.io/github/stars/Kruhlikau/subagent-tuner?style=social)
- [apowerb](https://github.com/apowerb/apowerb) - 基于 ADK 构建、运行和治理多智能体系统的平台，提供 RAG、Text-to-SQL、MCP 工具、令牌配额和按会话审计。采用 Apache 2.0 许可，可用 Docker Compose 或 Helm 自托管。 ![GitHub stars](https://img.shields.io/github/stars/apowerb/apowerb?style=social)

#### 智能体协议与标准

- [Agent Gateway](https://github.com/agentgateway/agentgateway) - 面向 AI 智能体和 MCP 服务器的新一代代理与路由层，具备 Kubernetes 原生传输、协议互操作和服务网格式隔离，构成可靠的智能体基础设施。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/agentgateway/agentgateway?style=social)
- [Microsandbox](https://github.com/superradcompany/microsandbox) - 快速、本地优先的 microVM 运行时与库，用于隔离执行 AI 智能体和不可信负载。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/superradcompany/microsandbox?style=social)
- [Agent Substrate](https://github.com/agent-substrate/substrate) - Kubernetes 原生执行运行时，面向高密度、有状态的 AI 智能体沙箱，具备 actor 生命周期管理、挂起/恢复和工作负载路由。采用 Apache-2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/agent-substrate/substrate?style=social)
- [DESIGN.md (Google)](https://github.com/google-labs-code/design.md) - 用于向编码智能体描述视觉身份的格式规范，结合 YAML 标记和 Markdown 文字，让智能体结构化地理解设计系统。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/google-labs-code/design.md?style=social)
- [Agent Skills](https://github.com/agentskills/agentskills) - 用于打包和渐进式加载 AI 智能体能力、指令、脚本和资源的标准化规范与文档格式。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/agentskills/agentskills?style=social)
- [FastMCP](https://github.com/PrefectHQ/fastmcp) - 用于构建 Model Context Protocol（MCP）服务器与客户端的 Python 框架，带自动 Schema 生成与校验。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/PrefectHQ/fastmcp?style=social)

#### 智能体上下文、记忆与知识

- [OpenViking](https://github.com/volcengine/OpenViking) - 面向 AI 智能体的开源上下文数据库，将智能体记忆、知识 RAG 和技能统一为虚拟文件系统。AGPL 3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/volcengine/OpenViking?style=social)
- [LWC (Local Wiki CLI)](https://github.com/JanYork/llm-wiki-cli) - 本地优先、有来源依据的项目记忆，面向编码智能体，提供受限的 MCP 检索、引用与溯源、原子变更集，以及可选的文档与代码知识图谱。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/JanYork/llm-wiki-cli?style=social)
- [Obsidian Agent Skills](https://github.com/kepano/obsidian-skills) - 面向 Obsidian 库的智能体技能与开放格式工具，兼容 Claude Code、Codex 和 OpenCode，支持 Obsidian 库、Markdown、Bases 和 JSON Canvas。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/kepano/obsidian-skills?style=social)
- [claude-obsidian](https://github.com/AgriciDaniel/claude-obsidian) - MIT 许可的 Obsidian 与 Claude Code 集成，通过本地索引和链接来组织 Markdown 库。 ![GitHub stars](https://img.shields.io/github/stars/AgriciDaniel/claude-obsidian?style=social)
- [Cangjie Skill](https://github.com/kangarooking/cangjie-skill) - 通过验证与测试工作流，把书籍、视频和播客提炼为结构化、可执行的智能体技能。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/kangarooking/cangjie-skill?style=social)
- [Hexis](https://github.com/Bevel-Software/Hexis) - 基于 Git 的平台，通过远程 MCP 服务器在多个 AI 智能体之间管理和共享技能、工具与上下文。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Bevel-Software/Hexis?style=social)
- [Codegraph](https://github.com/colbymchenry/codegraph) - 本地预索引的代码知识图谱，供编码智能体在 Claude、Codex 等之间减少令牌消耗和重复工具调用。 ![GitHub stars](https://img.shields.io/github/stars/colbymchenry/codegraph?style=social)
- [Wenlan](https://github.com/7xuanlu/wenlan) - 本地有来源支撑的 AI 知识库和面向编码智能体的 LLM wiki，提供 Markdown 页面、混合检索、知识图谱、CLI 工作流和 MCP 集成。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/7xuanlu/wenlan?style=social)
- [Gortex](https://github.com/zzet/gortex) - 用 Golang 构建的本地优先代码知识图谱与代码智能引擎，支持多仓库和实时图更新。为 AI 编码智能体而设计，只暴露必要信息，令牌用量最多降低 50 倍。可与 Claude、Codex、Hermes 等智能体原生协作。 ![GitHub stars](https://img.shields.io/github/stars/zzet/gortex?style=social)
- [Graphify](https://github.com/safishamsi/graphify) - AI 编码助手技能，把代码库、数据库和文档索引为可供编码智能体查询的知识图谱。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/safishamsi/graphify?style=social)
- [Workspai](https://github.com/chistiq/workspai) - 开源的工作区智能 CLI，可为多语言工作区建模、构建有证据支撑的知识图谱、运行健康度和发布门禁，并为 AI 编码智能体生成受限上下文。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/chistiq/workspai?style=social)
- [Headroom](https://github.com/chopratejas/headroom) - 面向工具输出、日志和 RAG 片段的上下文压缩代理，在保留 AI 智能体意图的同时降低令牌压力。 ![GitHub stars](https://img.shields.io/github/stars/chopratejas/headroom?style=social)
- [llmtrim](https://github.com/fkiene/llmtrim) - 自托管的 Rust 代理、MCP 服务器、CLI 和库，可压缩 LLM 提示词、工具输出与回复以削减令牌消耗，并有质量门禁确保绝不会让你的账单变高。MPL-2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/fkiene/llmtrim?style=social)
- [MailFathom](https://github.com/Krzysztof318/MailFathom) - 安全优先、自托管的邮件知识系统，把 IMAP 邮箱同步到用户自有的 PostgreSQL，支持词法与语义检索、带引用的答案、本地模型运行和 MCP 访问。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Krzysztof318/MailFathom?style=social)
- [MemPalace](https://github.com/MemPalace/mempalace) - 高性能、经过基准测试的 AI 记忆系统，用于长周期自主工作流中的持久召回与检索。 ![GitHub stars](https://img.shields.io/github/stars/MemPalace/mempalace?style=social)
- [Tree Ring Memory](https://github.com/TerminallyLazy/Tree-Ring-Memory) - 与框架无关、本地优先的 AI 智能体记忆生命周期管理，配备 Rust CLI、SQLite/FTS 召回、脱敏、遗忘、审计检查、记忆整合以及智能体技能指引。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/TerminallyLazy/Tree-Ring-Memory?style=social)
- [deja](https://github.com/vshulcz/deja-vu) - 构建在编码智能体已写入磁盘的会话记录之上的记忆层 —— Claude Code、Codex、Cursor、opencode、Zed 等 —— 包括安装之前的历史会话；基于 BM25 在本地召回，无需 LLM 或嵌入，并提供 MCP 工具，凭证在索引时即被脱敏。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/vshulcz/deja-vu?style=social)
- [Claude Mem](https://github.com/thedotmack/claude-mem) - 持久化的智能体记忆系统，捕获会话观察、进行摘要，并在不同编码智能体会话之间检索相关上下文。采用 Apache-2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/thedotmack/claude-mem?style=social)
- [codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp) - 高性能 C 语言代码库智能引擎与 MCP 服务器，把仓库索引为本地类型解析知识图谱。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/DeusData/codebase-memory-mcp?style=social)
- [book-to-skill](https://github.com/virgiliojr94/book-to-skill) - CLI 工具，把技术书籍、文档和参考资料提炼为结构化、按需加载的智能体技能。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/virgiliojr94/book-to-skill?style=social)
- [gknowledge](https://github.com/openapi/gknowledge) - 原生 GNOME 桌面应用（GTK 4、Libadwaita），实现了 Karpathy 的 LLM Wiki 模式：将文档摄取到兼容 Obsidian 的 Markdown wiki 并保留来源溯源，可离线运行或对接 Ollama 与 OpenAI 兼容端点，并通过 MCP 服务器把 wiki 暴露给编码智能体。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/openapi/gknowledge?style=social)
- [AI Memory](https://github.com/akitaonrails/ai-memory) - 面向智能体编码 CLI 的 Rust 原生长期记忆服务器与 MCP 客户端，具备 Karpathy 风格的 LLM wiki 编译、FTS5 召回和跨智能体会话交接。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/akitaonrails/ai-memory?style=social)
- [Busabase](https://github.com/busabase/busabase) - 开源数据库与工作区，通过 MCP、OpenAPI、CLI 和编码智能体技能为 AI 智能体提供结构化数据、持久知识、文档、技能和应用；关键写入可以先保持为可审查的 ChangeRequest 再转为正式内容。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/busabase/busabase?style=social)
- [ThreadShelf](https://github.com/ChrystianSchutz/ThreadShelf) - 本地优先的归档与语义搜索，覆盖多家提供商的 AI 对话历史，具备本地嵌入、LanceDB 存储、完整线程检索、HTTP 和 CLI 访问以及 stdio MCP 服务器。 ![GitHub stars](https://img.shields.io/github/stars/ChrystianSchutz/ThreadShelf?style=social)
- [drevon](https://github.com/csakash/drevon) - `npx drevon init` 可把任意目录变成 AI 工作区，共享一份配置，提供跨会话持久记忆，以及面向 Claude Code、Codex、Copilot、Cursor、Windsurf、Cline、Aider 和 Continue 的提示词与技能。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/csakash/drevon?style=social)

#### 自主编码智能体

- [BitFun](https://github.com/GCWing/BitFun) - 开源编码智能体，具备 Rust 运行时、桌面与 CLI 界面、自托管远程访问，以及自定义工具和技能支持。 ![GitHub stars](https://img.shields.io/github/stars/GCWing/BitFun?style=social)
- [Free Claude Code](https://github.com/Alishahryar1/free-claude-code) - 面向 Claude Code、Codex 和 Pi 的多提供商代理与启动器，带本地 Admin UI，可把编码智能体路由到 31+ 云端和本地 LLM 后端。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Alishahryar1/free-claude-code?style=social)
- [Background Agents](https://github.com/ColeMurray/background-agents) - 受 Ramp 的 Inspect 启发而构建的开源后台编码智能体系统，支持文件与环境快照、基于 cron 的自动化和多提供商模型。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/ColeMurray/background-agents?style=social)
- [OpenHands (ex-OpenDevin)](https://github.com/All-Hands-AI/OpenHands) - 功能完备的开源 AI 软件工程师。 ![GitHub stars](https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=social)
- [HEXStrike AI](https://github.com/0x4m4/hexstrike-ai) - 由 MCP 驱动、聚焦编码的网络安全智能体框架，用于自动化渗透测试和漏洞挖掘工作流。 ![GitHub stars](https://img.shields.io/github/stars/0x4m4/hexstrike-ai?style=social)
- [Darkmoon](https://github.com/ASCIT31/Dark-Moon) - 开源自主 AI 渗透测试平台和 MCP 宿主，通过按技术划分的攻击型子智能体编排 80+ 攻击工具，支持 Active Directory 和 Kubernetes。GPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/ASCIT31/Dark-Moon?style=social)
- [VulnClaw](https://github.com/Unclecheng-li/VulnClaw) - 自主渗透测试智能体，利用 Model Context Protocol（MCP）工具链、黑板状态空间搜索和结构化推理。 ![GitHub stars](https://img.shields.io/github/stars/Unclecheng-li/VulnClaw?style=social)
- [Goose](https://github.com/aaif-goose/goose) - 面向开发任务、可扩展的本机 AI 智能体。 ![GitHub stars](https://img.shields.io/github/stars/aaif-goose/goose?style=social)
- [OpenShell (NVIDIA)](https://github.com/NVIDIA/OpenShell) - 安全私密的自主 AI 智能体运行时，具备策略驱动的执行边界和 CLI 集成。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/OpenShell?style=social)
- [OpenCode](https://github.com/anomalyco/opencode) - 终端原生的自主编码智能体。 ![GitHub stars](https://img.shields.io/github/stars/anomalyco/opencode?style=social)
- [ECC](https://github.com/affaan-m/ECC) - 面向性能的编码智能体 harness，覆盖 Claude Code、Codex 等，提供技能、记忆和安全感知编排。 ![GitHub stars](https://img.shields.io/github/stars/affaan-m/ECC?style=social)
- [oh-my-pi](https://github.com/can1357/oh-my-pi) - 终端编码智能体，具备基于哈希锚定的编辑、子智能体、LSP、浏览器集成和终端原生工作流。 ![GitHub stars](https://img.shields.io/github/stars/can1357/oh-my-pi?style=social)
- [Pi (earendil-works)](https://github.com/earendil-works/pi) - 模块化智能体工具包，具备终端优先的 CLI、统一的模型/提供商层，以及面向编码工作流和 TUI/Web UI 的运行时集成。 ![GitHub stars](https://img.shields.io/github/stars/earendil-works/pi?style=social)
- [Aider](https://github.com/paul-gauthier/aider) - 命令行结对编程智能体。 ![GitHub stars](https://img.shields.io/github/stars/paul-gauthier/aider?style=social)
- [Pi (badlogic)](https://github.com/badlogic/pi-mono) - 终端编码智能体，具备基于哈希锚定的编辑、LSP 集成、子智能体、MCP 支持和包生态。 ![GitHub stars](https://img.shields.io/github/stars/badlogic/pi-mono?style=social)
- [Mistral-Vibe (Mistral)](https://github.com/mistralai/mistral-vibe) - Mistral 出品的极简 CLI 编码智能体。轻量、快速，为本地开发工作流而设计。 ![GitHub stars](https://img.shields.io/github/stars/mistralai/mistral-vibe?style=social)
- [Nanocoder (Nano-Collective)](https://github.com/Nano-Collective/nanocoder) - 在终端中运行的、美观的本地优先编码智能体。出于隐私和可控性考虑而构建，支持通过 OpenRouter 接入多家 AI 提供商。 ![GitHub stars](https://img.shields.io/github/stars/Nano-Collective/nanocoder?style=social)
- [Gemini CLI (Google)](https://github.com/google-gemini/gemini-cli) - 开源 AI 智能体，把 Gemini 的能力直接带进终端。支持代码生成、Shell 执行和文件编辑，采用完整的 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/google-gemini/gemini-cli?style=social)
- [Archon](https://github.com/coleam00/Archon) - 面向确定性 AI 编码智能体的工作流引擎。把开发流程定义为 YAML 工作流（规划 → 实现 → 验证 → 评审 → PR），并用隔离的 Git worktree 实现并行执行。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/coleam00/Archon?style=social)
- [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) - 面向仓库和 issue 修复工作流的轻量编码智能体，为简单的智能体软件工程实验而设计。 ![GitHub stars](https://img.shields.io/github/stars/SWE-agent/mini-swe-agent?style=social)
- [Kilo Code](https://github.com/Kilo-Org/kilocode) - 开源智能体编码助手，具备 IDE 工作流、工具使用，并支持本地或 OpenAI 兼容模型。 ![GitHub stars](https://img.shields.io/github/stars/Kilo-Org/kilocode?style=social)
- [Open SWE](https://github.com/langchain-ai/open-swe) - 来自 LangChain 生态的异步编码智能体，用于后台软件工程任务。 ![GitHub stars](https://img.shields.io/github/stars/langchain-ai/open-swe?style=social)
- [Letta Code](https://github.com/letta-ai/letta-code) - 记忆优先的编码 harness，专为能从经验中学习的长期存续智能体设计。持久化智能体具备跨模型（Claude、GPT、Gemini、GLM、Kimi）的可移植记忆。提供 macOS、Windows 和 Linux 的 CLI 与桌面应用。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/letta-ai/letta-code?style=social)
- [LoopTroop](https://github.com/looptroop-ai/LoopTroop) - 本地优先的 AI 编码工作区，可编排多模型规划委员会、git worktree 和任务循环。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/looptroop-ai/LoopTroop?style=social)
- [gptme](https://github.com/gptme/gptme) - 运行在你终端里的智能体，配备本地工具：编写代码、使用终端、浏览网页。在此之上构建你自己的持久自主智能体。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/gptme/gptme?style=social)
- [Superpowers](https://github.com/obra/superpowers) - 面向编码智能体的可组合技能框架与软件开发方法论，将规划、测试驱动开发和代码评审等流程结构化。 ![GitHub stars](https://img.shields.io/github/stars/obra/superpowers?style=social)
- [Agent Skills](https://github.com/addyosmani/agent-skills) - 面向 AI 编码智能体的生产级工程技能与质量门禁，将规范细化、规划和测试等开发者工作流打包。 ![GitHub stars](https://img.shields.io/github/stars/addyosmani/agent-skills?style=social)
- [Harness](https://github.com/revfactory/harness) - 面向 C‍laude Code 的团队架构工厂，设计领域专属的智能体团队、定义专业智能体并生成其技能。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/revfactory/harness?style=social)
- [jcode](https://github.com/1jehuang/jcode) - 面向性能、内存高效的编码智能体 harness，为多会话工作流和无限可定制性而构建。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/1jehuang/jcode?style=social)
- [firstmate](https://github.com/kunchenguid/firstmate) - 智能体发行版，用于在可见的终端会话后端上，以隔离的 Git worktree 运行一支自主编码智能体小队。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/kunchenguid/firstmate?style=social)
- [DeepSeek-Reasonix](https://github.com/esengine/DeepSeek-Reasonix) - DeepSeek 原生的终端 AI 编码智能体，围绕前缀缓存稳定性而设计。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/esengine/DeepSeek-Reasonix?style=social)
- [gstack](https://github.com/garrytan/gstack) - 面向 Claude Code 和编码智能体的多专家智能体技能框架，把开发工作流结构化为规划、设计、QA 和发布阶段。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/garrytan/gstack?style=social)
- [taste-skill](https://github.com/Leonxlnx/taste-skill) - 面向编码智能体的反粗制滥造前端设计技能，在生成过程中改进布局、排版和设计系统一致性。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Leonxlnx/taste-skill?style=social)
- [Agents CLI (Google)](https://github.com/google/agents-cli) - CLI 与技能集合，让编码助手成为在 Google Cloud 上创建、评估和部署 AI 智能体的专家。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/google/agents-cli?style=social)
- [Claude Code Skills & Plugins](https://github.com/alirezarezvani/claude-skills) - 面向 Claude Code、Gemini CLI、Cursor 和其他 AI 编码智能体的模块化指令包、自定义命令和实用脚本。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/alirezarezvani/claude-skills?style=social)
- [AG Kit](https://github.com/vudovn/ag-kit) - 以 Antigravity 为先的智能体工程套件，包含规则、技能、持久记忆、MCP 指引和原生安全钩子。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/vudovn/ag-kit?style=social)
- [Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent) - 自我改进的递归语言模型（RLM）智能体，面向编码工作流和自主任务。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/PrimeIntellect-ai/prime-agent?style=social)
- [Agent Skills (Google)](https://github.com/google/skills) - 面向 Google Cloud 和 Google 开发者平台的官方 Agent Skills 合集，以产品和技术工作流扩展 AI 编码智能体。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/google/skills?style=social)
- [Agent Skills (Anthropic)](https://github.com/anthropics/skills) - 面向 Claude Code、Claude API 和 AI 智能体的官方 Agent Skills 合集与参考实现。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/anthropics/skills?style=social)
- [Superagent](https://github.com/pungme/superagent-desktop) - macOS 桌面应用，为 Claude Code 和 Codex 提供可真实驱动的浏览器、可安装并截图应用的 iOS 模拟器，以及用于远程监控的手机配套应用。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/pungme/superagent-desktop?style=social)
- [BrowserSkill (Tencent)](https://github.com/Tencent/BrowserSkill) - CLI 与扩展，使 AI 编码智能体能在后台标签页中控制已登录的浏览器会话，而不打断用户工作流。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Tencent/BrowserSkill?style=social)
- [Ordewell](https://github.com/ordewell/ordewell) - 终端 CLI 与 TUI，把一个目标转化为有序、可编辑的编码智能体任务计划，每个任务自带运行器（Claude Code、Codex、OpenCode）、模型和模式；只有当完成标记出现在该运行器的输出中，任务才算完成。采用 Apache-2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/ordewell/ordewell?style=social)
- [Orbi](https://github.com/orbi-build/orbi) - 自托管运行器，把一个打了标签的 GitHub Issue 变成 pull request，由独立模型在合并前评审冻结的 diff，并发布打了标签的 GitHub Release。可配合任意 OpenAI 兼容模型或 Codex 订阅使用。采用 AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/orbi-build/orbi?style=social)
- [Threadlines](https://github.com/Threadlines/threadlines) - 面向 Codex 和 Claude Code 的桌面工作区，在每个智能体线程旁保留实时浏览器、终端和源代码管理，并允许第二个智能体加入线程来回答问题或评审改动。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Threadlines/threadlines?style=social)

#### 提示工程与结构化输出

- [Outlines](https://github.com/dottxt-ai/outlines) - 面向 LLM 的结构化输出。在生成过程中保证合法 JSON、符合正则的文本和 Pydantic 模型输出。受到 NVIDIA、Cohere、Hugging Face 和 vLLM 的信赖。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/dottxt-ai/outlines?style=social)
- [LangGPT](https://github.com/langgptai/LangGPT) - 结构化提示与元提示设计的开创性框架。经全球数千用户实战验证，10,000+ stars。是构建可复用、可维护提示模板最流行的提示工程范式。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/langgptai/LangGPT?style=social)
- [Prompt Optimizer](https://github.com/linshenkx/prompt-optimizer) - AI 提示优化工具，支持多轮迭代改进、系统提示与用户提示的双模式优化，以及多模型支持。以网页应用、桌面应用、Chrome 扩展和 Docker 部署形式提供。采用 AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/linshenkx/prompt-optimizer?style=social)
- [Guidance](https://github.com/guidance-ai/guidance) - 用于引导语言模型的高效编程范式。可用循环、条件语句和正则约束内联控制输出结构。相比传统提示方式降低延迟和成本。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/guidance-ai/guidance?style=social)
- [XGrammar](https://github.com/mlc-ai/xgrammar) - 快速、灵活、可移植的结构化生成引擎。是 vLLM、SGLang、TensorRT-LLM 和 MLC-LLM 的默认后端，具备灵活的语法支持和零开销的掩码生成。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/mlc-ai/xgrammar?style=social)

- [AdalFlow](https://github.com/SylphAI-Inc/AdalFlow) - 用于构建并自动优化 LLM 应用的库，借助 LLM-AutoDiff 实现免微调优化。支持端到端工作流优化，具备追踪和人在回路能力。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/SylphAI-Inc/AdalFlow?style=social)
- [guidelint](https://github.com/SIDDARTHAREDDY8/guidelint) - 零依赖 Python CLI，依据 23 条版本感知规则检查 vLLM、llama.cpp 和 Ollama 的结构化输出配置，在上线前捕获被静默忽略的字段。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/SIDDARTHAREDDY8/guidelint?style=social)


#### 领域专属智能体

- [Composio](https://github.com/ComposioHQ/composio) - 面向 AI 智能体的工具集成层，提供 1000+ 工具包、认证管理和沙箱化工作台。支撑各大框架的工具使用。 ![GitHub stars](https://img.shields.io/github/stars/ComposioHQ/composio?style=social)
- [Langflow](https://github.com/langflow-ai/langflow) - 面向智能体工作流的可视化低代码平台。 ![GitHub stars](https://img.shields.io/github/stars/langflow-ai/langflow?style=social)
- [Dify](https://github.com/langgenius/dify) - 生产就绪的智能体工作流平台。 ![GitHub stars](https://img.shields.io/github/stars/langgenius/dify?style=social)
- [OWL (camel-ai/owl)](https://github.com/camel-ai/owl) - 先进的多智能体协作系统。 ![GitHub stars](https://img.shields.io/github/stars/camel-ai/owl?style=social)
- [gpt-researcher](https://github.com/assafelovic/gpt-researcher) - 可就任意主题进行深度在线研究的自主智能体。通过编排网络搜索、内容抓取和综合，生成带引用的完整报告。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/assafelovic/gpt-researcher?style=social)
- [last30days-skill](https://github.com/mvanhorn/last30days-skill) - AI 智能体，从 Reddit、X、YouTube、Hacker News 和预测市场研究并综合主题趋势。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/mvanhorn/last30days-skill?style=social)
- [Jev Social](https://github.com/socai-io/jev-social) - 本地社交媒体研究智能体，用 TypeSafe Jev 为 Instagram、TikTok 和 LinkedIn 选择受限的 socai CLI 浏览器操作，流式输出所采集的证据，并生成带引用的报告。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/socai-io/jev-social?style=social)
- [pm-skills](https://github.com/phuryn/pm-skills) - 面向 Claude Code、Codex 和 Claude Cowork 的产品管理插件技能与工作流市场。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/phuryn/pm-skills?style=social)
- [claude-code-best-practice](https://github.com/shanraisshan/claude-code-best-practice) - 面向 Claude Code 开发者智能体的自定义子智能体、技能、环境配置和最佳实践。 ![GitHub stars](https://img.shields.io/github/stars/shanraisshan/claude-code-best-practice?style=social)
- [PPT Master](https://github.com/hugohe3/ppt-master) - AI 驱动的多角色智能体技能，通过 SVG 和 DrawingML 工作流把文档转换为可编辑的 PowerPoint 演示文稿。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/hugohe3/ppt-master?style=social)
- [PraisonAI](https://github.com/MervinPraison/PraisonAI) - 7×24 小时工作的 AI 员工团队，用于自动化应对复杂挑战。低代码多智能体框架，具备交接、护栏、记忆、RAG 和 100+ LLM 提供商。 ![GitHub stars](https://img.shields.io/github/stars/MervinPraison/PraisonAI?style=social)
- [Agent-S (Simular AI)](https://github.com/simular-ai/Agent-S) - 像人一样使用计算机的开放智能体框架。在 OSWorld 基准测试上达到 SOTA（72.6%），用于 GUI 自动化和计算机控制。 ![GitHub stars](https://img.shields.io/github/stars/simular-ai/Agent-S?style=social)
- [MobileAgent (Alibaba/X-PLUG)](https://github.com/X-PLUG/MobileAgent) - 强大的 GUI 智能体家族，可自主控制移动设备。设计为通过视觉 UI 感知和推理来操作智能手机应用的多模态智能体框架。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/X-PLUG/MobileAgent?style=social)
- [UI-TARS Desktop (ByteDance)](https://github.com/bytedance/UI-TARS-desktop) - 开源多模态 AI 智能体技术栈，具备原生 GUI 智能体能力。把 GUI 智能体和视觉能力带到你电脑、浏览器和终端的桌面应用。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/bytedance/UI-TARS-desktop?style=social)
- [Browser Use](https://github.com/browser-use/browser-use) - 让网站对 AI 智能体变得可用。只需自然语言指令即可实现自主网页自动化、数据提取和任务完成。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/browser-use/browser-use?style=social)
- [Jet Browser](https://github.com/masakaai/jet-browser) - 自托管的 WPE WebKit 运行时，为 AI 智能体提供隔离的 Docker 浏览器会话、原生输入、截图以及有序的 JSONL 控制接口。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/masakaai/jet-browser?style=social)
- [Steel Browser](https://github.com/steel-dev/steel-browser) - 面向 AI 智能体和应用的浏览器 API。提供开箱即用的浏览器沙箱，免除基础设施烦恼的网页自动化。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/steel-dev/steel-browser?style=social)
- [Cua](https://github.com/trycua/cua) - 面向计算机使用型智能体的开源沙箱、SDK 和基准测试，用于控制桌面环境。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/trycua/cua?style=social)
- [Webwright](https://github.com/microsoft/Webwright) - 终端风格的 Web 智能体框架，让编码模型通过编写和执行 Playwright 脚本来作为浏览器智能体运行。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/Webwright?style=social)
- [invisible_playwright_mcp](https://github.com/feder-cr/invisible_playwright_mcp) - 自托管浏览器智能体，驱动打过补丁的 Firefox 并使用真实的指针和键盘输入，可作为本地 MCP 服务器供 Claude Code、Codex、Cursor 或 Gemini CLI 使用，也可通过其自带的 Web UI 使用。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/feder-cr/invisible_playwright_mcp?style=social)
- [TradingAgents](https://github.com/TauricResearch/TradingAgents) - 面向金融交易的多智能体框架。模拟专业交易公司的运作，包含 6+ 专业智能体角色、回测、风险管理和投资组合优化。基于 LangGraph 构建，支持多家 LLM 提供商。 ![GitHub stars](https://img.shields.io/github/stars/TauricResearch/TradingAgents?style=social)
- [Parlant](https://github.com/emcie-co/parlant) - 面向客户侧 AI 智能体的对话控制层。企业级上下文工程框架，针对一致性强、符合合规且符合品牌调性的 B2C 和敏感 B2B 交互做了优化。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/emcie-co/parlant?style=social)
- [n8n](https://github.com/n8n-io/n8n) - 可自托管的工作流自动化平台，具备 AI 智能体节点、工具集成和生产自动化工作流。 ![GitHub stars](https://img.shields.io/github/stars/n8n-io/n8n?style=social)
- [Activepieces](https://github.com/activepieces/activepieces) - 开源自动化平台，具备 AI 智能体、MCP 集成和自托管工作流编排。 ![GitHub stars](https://img.shields.io/github/stars/activepieces/activepieces?style=social)
- [Julep](https://github.com/julep-ai/julep) - 有状态的智能体工作流平台，具备记忆、工具、分支和长任务执行能力。 ![GitHub stars](https://img.shields.io/github/stars/julep-ai/julep?style=social)
- [uAgents (Fetch.ai)](https://github.com/fetchai/uAgents) - 用于轻松创建去中心化智能体的快速轻量框架。智能体通过在 Almanac 智能合约上注册自动加入网络，开箱即用地支持智能体间通信。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/fetchai/uAgents?style=social)
- [Tracecat](https://github.com/TracecatHQ/tracecat) - 可自托管的安全自动化平台，用于在告警、案件和运维之间构建智能体工作流。 ![GitHub stars](https://img.shields.io/github/stars/TracecatHQ/tracecat?style=social)
- [PentAGI](https://github.com/vxcontrol/pentagi) - 完全自主的 AI 智能体系统，用于执行道德黑客行动和复杂渗透测试任务。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/vxcontrol/pentagi?style=social)
- [ToolJet](https://github.com/ToolJet/ToolJet) - 可自托管的内部应用构建器，为运维团队提供 AI 应用和智能体工作流。 ![GitHub stars](https://img.shields.io/github/stars/ToolJet/ToolJet?style=social)
- [The Agency](https://github.com/msitarzewski/agency-agents) - 海量专业开发者与工作流智能体人格库，支持为 Claude Code、Cursor、Codex 和其他编码助手安装。 ![GitHub stars](https://img.shields.io/github/stars/msitarzewski/agency-agents?style=social)
- [CubeSandbox (Tencent Cloud)](https://github.com/TencentCloud/CubeSandbox) - 基于 RustVMM 和 KVM 的高性能、安全智能体沙箱，兼容 E2B SDK。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/TencentCloud/CubeSandbox?style=social)
- [DeepTutor](https://github.com/HKUDS/DeepTutor) - 面向终身个性化辅导的多智能体系统，具备交互式和自适应学习工作流。 ![GitHub stars](https://img.shields.io/github/stars/HKUDS/DeepTutor?style=social)
- [Corsair](https://github.com/corsairdev/corsair) - 面向特定领域的 AI 智能体框架，用于构建和运行专业化智能体。 ![GitHub stars](https://img.shields.io/github/stars/corsairdev/corsair?style=social)
- [DeskcommCRM](https://github.com/melgarafael/DeskcommCRM) - 自托管的销售 CRM，利用自主 AI 智能体、WhatsApp 集成、多租户 RAG 和 MCP 工具支撑基于对话的销售工作流。 ![GitHub stars](https://img.shields.io/github/stars/melgarafael/DeskcommCRM?style=social)
- [OpenResearch](https://github.com/alphaXiv/OpenResearch) - 本地优先的工作区与 harness，把编码智能体变成自主研究智能体，用于文献综述、假设生成和实验。 ![GitHub stars](https://img.shields.io/github/stars/alphaXiv/OpenResearch?style=social)
- [call4me](https://github.com/skeptrunedev/call4me) - 语音智能体与远程 MCP 服务器，可代编码智能体用户致电商家，自动应对电话语音菜单，并返回通话记录与结果。 ![GitHub stars](https://img.shields.io/github/stars/skeptrunedev/call4me?style=social)
- [AdsTurbo Creative MCP](https://github.com/AdsTurbo/adsturbo-creative-mcp) - 本地 stdio MCP 服务器与 CLI，把产品页面或参考广告转换为结构化的视频广告方案、钩子变体、UGC 脚本和分镜，无需 API 密钥和网络请求。 ![GitHub stars](https://img.shields.io/github/stars/AdsTurbo/adsturbo-creative-mcp?style=social)
- [Unsora MCP](https://github.com/Shipped-Studio/unsora/tree/main/apps/mcp) - 远程 MCP 服务器，让智能体生成图像、视频、配音和音乐，把长视频剪成短片，并定时或发布内容到已连接社交账号，同时提供交互式 MCP Apps 预览。它调用的 Express API、Next.js 应用和 Trigger.dev worker 都位于同一个 AGPL-3.0 仓库中。 ![GitHub stars](https://img.shields.io/github/stars/Shipped-Studio/unsora?style=social)
- [AI Agent by TerraLab](https://github.com/TerraLabAI/QGIS_AI-Agent) - QGIS 插件，AI 智能体从一句自然语言任务出发查找开放数据、执行分析并制作地图，每一步变更都可撤销。GPL-2.0 许可；它调用的是带免费额度的托管服务。 ![GitHub stars](https://img.shields.io/github/stars/TerraLabAI/QGIS_AI-Agent?style=social)

#### 智能体记忆与状态

- [Letta (ex-MemGPT)](https://github.com/letta-ai/letta) - 用于构建具备高级记忆、能够随时间学习和自我改进的有状态智能体的平台。 ![GitHub stars](https://img.shields.io/github/stars/letta-ai/letta?style=social)
- [Mem0](https://github.com/mem0ai/mem0) - 面向 AI 智能体的通用记忆层。跨模型和环境的持久化、多会话记忆。 ![GitHub stars](https://img.shields.io/github/stars/mem0ai/mem0?style=social)
- [Caura](https://github.com/caura-ai/caura) - 面向 AI 智能体的开源记忆基础设施，为构建有状态、长运行的智能体提供持久记忆和上下文管理。 ![GitHub stars](https://img.shields.io/github/stars/caura-ai/caura?style=social)
- [Rohitg (agentmemory)](https://github.com/rohitg00/agentmemory) - 面向智能体的开源记忆服务，具备经过基准测试的检索、结构化实体，以及在工具工作流中实现持久个性化记忆的 API/SDK 支持。 ![GitHub stars](https://img.shields.io/github/stars/rohitg00/agentmemory?style=social)
- [Forgetful](https://github.com/ScottRBK/forgetful) - 用于持久化 AI 智能体记忆的 MCP 服务器，具备原子笔记、语义链接以及 SQLite 或 PostgreSQL 存储。 ![GitHub stars](https://img.shields.io/github/stars/ScottRBK/forgetful?style=social)
- [Hindsight](https://github.com/vectorize-io/hindsight) - Vectorize 出品的业界领先 AI 智能体长期记忆方案。完全自托管、MIT 许可，可与 LangChain、CrewAI、LlamaIndex、Vercel AI SDK 等集成。 ![GitHub stars](https://img.shields.io/github/stars/vectorize-io/hindsight?style=social)
- [Supermemory](https://github.com/supermemoryai/supermemory) - 面向 AI 智能体的记忆 API 与应用，提供跨项目快速、可扩展且具备上下文感知的存储与检索。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/supermemoryai/supermemory?style=social)
- [TencentDB Agent Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory) - 完全本地的 AI 智能体长期记忆层，具备四级渐进式流水线和零外部依赖。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/TencentCloud/TencentDB-Agent-Memory?style=social)
- [Cognee](https://github.com/topoteretes/cognee) - 面向智能体的 AI 记忆平台，构建自托管知识图谱以实现跨会话持久长期记忆。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/topoteretes/cognee?style=social)
- [LoopX](https://github.com/huangruiteng/loopx) - 面向长运行 AI 智能体循环的轻量状态内核与本地控制平面，具备目标追踪、自动唤醒和可验证交接。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/huangruiteng/loopx?style=social)
- [Unlimited Context](https://github.com/AetherAI3/Unlimited-Context-LLM) - 本地优先的上下文与记忆引擎，把十亿 token 的编码池保存在磁盘上，并把相关切片调页到小型常驻窗口中，供 Ollama 等本地 LLM 使用。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/AetherAI3/Unlimited-Context-LLM?style=social)
- [inspeximus](https://github.com/DanceNitra/inspeximus) - 记忆层，可按 key 让已更正的事实退役，之后仅凭 key 就能撤销该更正。零依赖内核并附带 MCP 服务器。 ![GitHub stars](https://img.shields.io/github/stars/DanceNitra/inspeximus?style=social)
- [Mnemoverse](https://github.com/mnemoverse/mcp-memory-server) - MIT 许可的智能体记忆 MCP 服务器，由托管引擎支撑，它依据上报结果而非仅凭相似度对召回进行重排序，并在 Claude Code、Cursor、VS Code 和 ChatGPT 之间共用一套 key 或 OAuth。 ![GitHub stars](https://img.shields.io/github/stars/mnemoverse/mcp-memory-server?style=social)
- [Statewave](https://github.com/smaramwbc/statewave) - 面向 AI 智能体的开源记忆运行时，编译可复现、带来源标签的上下文包而非依赖查询时检索，基于 Postgres 和 pgvector 自托管，并提供 Python 和 TypeScript SDK。采用 Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/smaramwbc/statewave?style=social)
- [Vestige](https://github.com/samvallad33/vestige) - 基于 Strata（一个仅追加的签名日志）构建的 AI 智能体因果证明引擎与操作系统，不使用向量也不使用 RAG；每个答案都携带其证明（记忆 id、边路径或收据），失败可追溯到导致它的提交。采用 AGPL 3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/samvallad33/vestige?style=social)

---
- [Xenon](https://github.com/xianyu-sheng/Xenon) - 可验证的终端编码智能体，结合仅追加的证据哈希链、单一确定性验证门禁，以及针对无法检查项的人工验证清单（支持多提供商、就绪 MCP、MIT）。 ![GitHub stars](https://img.shields.io/github/stars/xianyu-sheng/Xenon?style=social)

## 5. 检索增强生成（RAG）与知识库

> 面向 RAG 流水线的检索系统、向量数据库、嵌入模型及相关工具。

#### 向量数据库与搜索引擎

- [Chroma](https://github.com/chroma-core/chroma) - 最受欢迎的开源嵌入数据库。 ![GitHub stars](https://img.shields.io/github/stars/chroma-core/chroma?style=social)
- [Qdrant](https://github.com/qdrant/qdrant) - 用 Rust 编写的高性能向量搜索引擎。 ![GitHub stars](https://img.shields.io/github/stars/qdrant/qdrant?style=social)
- [Weaviate](https://github.com/weaviate/weaviate) - 原生 GraphQL 的向量搜索引擎。 ![GitHub stars](https://img.shields.io/github/stars/weaviate/weaviate?style=social)
- [Milvus](https://github.com/milvus-io/milvus) - 可扩展的云原生向量数据库。 ![GitHub stars](https://img.shields.io/github/stars/milvus-io/milvus?style=social)
- [NornicDB](https://github.com/orneryd/NornicDB) - 用 Go 编写的低延迟图与向量混合检索数据库，提供 Neo4j 和兼容 Qdrant 的驱动。 ![GitHub stars](https://img.shields.io/github/stars/orneryd/NornicDB?style=social)
- [Faiss](https://github.com/facebookresearch/faiss) - 面向稠密向量的相似度搜索与聚类库，同时提供 CPU 和 GPU 实现。 ![GitHub stars](https://img.shields.io/github/stars/facebookresearch/faiss?style=social)
- [LanceDB](https://github.com/lancedb/lancedb) - 针对多模态数据优化的无服务器向量数据库。 ![GitHub stars](https://img.shields.io/github/stars/lancedb/lancedb?style=social)
- [Vespa](https://github.com/vespa-engine/vespa) - AI + 数据平台，提供混合搜索（向量 + 关键字）和大规模实时索引。经过实战检验，每天服务数十亿次查询。 ![GitHub stars](https://img.shields.io/github/stars/vespa-engine/vespa?style=social)
- [pgvector](https://github.com/pgvector/pgvector) - 用于向量相似度搜索的 PostgreSQL 扩展。 ![GitHub stars](https://img.shields.io/github/stars/pgvector/pgvector?style=social)
- [pgvectorscale](https://github.com/timescale/pgvectorscale) - 基于 DiskANN 算法的可扩展向量搜索 PostgreSQL 扩展。作为 pgvector 的补充，在大规模场景下搜索速度显著更快、召回率更高。基于 PostgreSQL 许可。 ![GitHub stars](https://img.shields.io/github/stars/timescale/pgvectorscale?style=social)
- [VectorChord](https://github.com/tensorchord/VectorChord) - 在 Postgres 中实现的可扩展、高速、对磁盘友好的向量搜索。是 pgvecto.rs 的继任者，具备生产级性能和高效存储。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/tensorchord/VectorChord?style=social)
- [Quickwit](https://github.com/quickwit-oss/quickwit) - 面向可观测性的云原生搜索引擎。Datadog、Elasticsearch、Loki 和 Tempo 的开源替代方案，原生支持向量搜索。 ![GitHub stars](https://img.shields.io/github/stars/quickwit-oss/quickwit?style=social)
- [Tantivy](https://github.com/quickwit-oss/tantivy) - 受 Apache Lucene 启发、用 Rust 编写的全文搜索引擎库。驱动 Quickwit 及其他生产级搜索系统。 ![GitHub stars](https://img.shields.io/github/stars/quickwit-oss/tantivy?style=social)
- [Manticore Search](https://github.com/manticoresoftware/manticoresearch) - 易于使用的开源高速搜索数据库。Elasticsearch 的良好替代品，提供类 SQL 接口和向量搜索能力。 ![GitHub stars](https://img.shields.io/github/stars/manticoresoftware/manticoresearch?style=social)
- [OpenSearch](https://github.com/opensearch-project/OpenSearch) - 开源分布式 RESTful 搜索与分析套件，原生支持向量搜索。Elasticsearch 的企业级分支，内置 k-NN 插件以支撑大规模语义搜索。 ![GitHub stars](https://img.shields.io/github/stars/opensearch-project/OpenSearch?style=social)
- [Marqo](https://github.com/marqo-ai/marqo) - 面向文本、图像和结构化数据的多模态向量搜索。端到端的索引与搜索，内置嵌入模型。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/marqo-ai/marqo?style=social)
- [Vald](https://github.com/vdaas/vald) - 高扩展性的分布式向量搜索引擎。云原生架构，支持自动索引、水平扩展和多种 ANN 算法。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/vdaas/vald?style=social)
- [hnswlib](https://github.com/nmslib/hnswlib) - 仅头文件的头文件级 C++ 库，用于快速近似最近邻搜索，附带 Python 绑定。支持 CRUD 操作和并发读写——这是 ANN 库中独有的特性。为众多生产级向量数据库提供底层支撑。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/nmslib/hnswlib?style=social)
- [turbovec](https://github.com/RyanCodrai/turbovec) - Rust 原生的高性能向量索引，附带 Python 绑定，针对快速 ANN 搜索和现代 CPU 上的 SIMD 加速做了优化。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/RyanCodrai/turbovec?style=social)
- [sqlite-vec](https://github.com/asg017/sqlite-vec) - 随处可运行的 SQLite 向量搜索扩展。体积极小，纯 C 编写、零依赖、速度"够快"的向量搜索。非常适合嵌入式和边缘部署。MIT/Apache-2.0 双许可。 ![GitHub stars](https://img.shields.io/github/stars/asg017/sqlite-vec?style=social)
- [zvec](https://github.com/alibaba/zvec) - 来自阿里巴巴的轻量级、极速进程内向量数据库。构建于 Proxima（阿里巴巴久经考验的向量搜索引擎）之上，提供生产级、低延迟的相似度搜索。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/alibaba/zvec?style=social)
- [Meilisearch](https://github.com/meilisearch/meilisearch) - 极速搜索引擎 API，内置 AI 驱动的混合搜索。容错全文搜索结合基于 HNSW 的向量搜索，实现语义检索。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/meilisearch/meilisearch?style=social)
- [Typesense](https://github.com/typesense/typesense) - Algolia + Pinecone 的开源替代品。快速、容错的内存模糊搜索引擎，原生支持向量搜索。GPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/typesense/typesense?style=social)
- [Elasticsearch](https://github.com/elastic/elasticsearch) - 分布式搜索与分析引擎，原生支持 k-NN 向量搜索、混合搜索和稠密向量索引。作为全文搜索的行业标准，如今还具备强大的语义搜索能力。AGPL-3.0/Elastic-2.0 双许可。 ![GitHub stars](https://img.shields.io/github/stars/elastic/elasticsearch?style=social)
- [Apache Solr](https://github.com/apache/solr) - 成熟的基于 Lucene 的搜索平台，支持稠密向量搜索、过滤、分面和混合检索模式，适合搜索密集型的生产级 RAG 系统。 ![GitHub stars](https://img.shields.io/github/stars/apache/solr?style=social)
- [RediSearch](https://github.com/RediSearch/RediSearch) - 为 Redis 部署提供全文、二级索引和向量相似度搜索。当检索需要低延迟的 Redis 原生搜索时非常有用。 ![GitHub stars](https://img.shields.io/github/stars/RediSearch/RediSearch?style=social)
- [ParadeDB](https://github.com/paradedb/paradedb) - 原生于 Postgres 的搜索与分析引擎，无需将数据移出 PostgreSQL 即可完成全文、分面和混合检索。 ![GitHub stars](https://img.shields.io/github/stars/paradedb/paradedb?style=social)
- [Orama](https://github.com/oramasearch/orama) - 轻量级搜索引擎，提供全文、向量和混合搜索，适用于浏览器、服务器和边缘应用。 ![GitHub stars](https://img.shields.io/github/stars/oramasearch/orama?style=social)
- [HelixDB](https://github.com/HelixDB/helix-db) - 图向量数据库，适合那些在语义搜索之外还需要关系遍历的检索系统。 ![GitHub stars](https://img.shields.io/github/stars/HelixDB/helix-db?style=social)
- [USearch](https://github.com/unum-cloud/usearch) - 快速的单文件向量相似度搜索与聚类引擎。比 FAISS 更小更快，提供 20 多种语言绑定（C++、Python、JavaScript、Rust、Java、Go 等）并支持自定义度量。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/unum-cloud/usearch?style=social)
- [Deep Lake](https://github.com/activeloopai/deeplake) - 面向 Agent 的 AI 数据运行时，基于无服务器 PostgreSQL 和多模态数据湖。存储并搜索向量、图像、文本、视频等，提供 LangChain/LlamaIndex 集成。已被 Intel、Bayer、耶鲁和牛津大学使用。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/activeloopai/deeplake?style=social)
- [DiskANN (Microsoft)](https://github.com/microsoft/DiskANN) - 基于图结构的索引，用于可扩展、快速、新鲜且带过滤条件的近似最近邻搜索。借助基于 SSD 的索引，可在单节点上处理十亿级向量数据集。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/DiskANN?style=social)
- [SPTAG (Microsoft)](https://github.com/microsoft/SPTAG) - 分布式近似最近邻搜索库，提供高质量的向量索引构建和在线服务工具包。驱动 Bing 在万亿级向量规模上的向量搜索。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/SPTAG?style=social)
- [nanoflann](https://github.com/jlblancoc/nanoflann) - C++11 仅头文件库，基于 KD 树实现快速最近邻搜索。零依赖、单文件集成，在现代 C++ 下比 FLANN 快 2-3 倍。BSD 许可。 ![GitHub stars](https://img.shields.io/github/stars/jlblancoc/nanoflann?style=social)
- [Vearch](https://github.com/vearch/vearch) - 面向 AI 原生应用的云原生分布式向量数据库。支持嵌入向量的高效相似度搜索，具备水平扩展和实时索引能力。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/vearch/vearch?style=social)
- [JVector (DataStax)](https://github.com/datastax/jvector) - 最先进的 Java 嵌入式向量搜索引擎。基于 DiskANN 算法的十亿级向量搜索，具备高效的内存映射。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/datastax/jvector?style=social)
- [VectorDBBench (Zilliz)](https://github.com/zilliztech/VectorDBBench) - 向量数据库的行业标准基准测试套件。可用自己的数据集测试并比较 Milvus、Zilliz Cloud 等向量数据库的性能。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/zilliztech/VectorDBBench?style=social)

#### 嵌入模型

- [BGE (FlagEmbedding)](https://github.com/FlagOpen/FlagEmbedding) - BAAI 一流的嵌入模型家族。 ![GitHub stars](https://img.shields.io/github/stars/FlagOpen/FlagEmbedding?style=social)
- [FastEmbed (Qdrant)](https://github.com/qdrant/fastembed) - 基于 ONNX Runtime 的轻量级高速 Python 嵌入生成库。无需 GPU 依赖即可支持文本、稀疏（SPLADE）和后期交互（ColBERT）嵌入。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/qdrant/fastembed?style=social)
- [EmbedAnything](https://github.com/StarlightSearch/EmbedAnything) - 用 Rust 构建的极简高性能多模态嵌入流水线。内存安全、模块化、生产就绪，支持文本、图像和音频嵌入，可无缝对接向量数据库。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/StarlightSearch/EmbedAnything?style=social)
- [Text Embeddings Inference (Hugging Face)](https://github.com/huggingface/text-embeddings-inference) - 极速的文本嵌入模型推理方案。基于 token 的动态批处理和 Flash Attention 实现高性能特征提取，支持 FlagEmbedding、E5、GTE 等多种模型。提供兼容 OpenAI 的 API 与 Docker 部署方式。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/text-embeddings-inference?style=social)

#### 嵌入基准测试

- [MTEB](https://github.com/embeddings-benchmark/mteb) - 大规模文本嵌入基准测试（MTEB），涵盖 1000 多种语言和多样任务。是评估和比较嵌入模型的行业标准。 ![GitHub stars](https://img.shields.io/github/stars/embeddings-benchmark/mteb?style=social)

#### RAG 框架与高级检索工具

- [EmbedChain](https://github.com/embedchain/embedchain) - 面向 AI Agent 的通用记忆层。简洁的 API，可在任意数据集上构建 RAG 应用，支持多种向量存储、嵌入模型和 LLM 提供商。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/embedchain/embedchain?style=social)
- [LlamaIndex](https://github.com/run-llama/llama_index) - 功能完备的 RAG 流水线，具备高级索引能力。 ![GitHub stars](https://img.shields.io/github/stars/run-llama/llama_index?style=social)
- [Haystack](https://github.com/deepset-ai/haystack) - 端到端的 NLP 与 RAG 框架。 ![GitHub stars](https://img.shields.io/github/stars/deepset-ai/haystack?style=social)
- [RAGFlow](https://github.com/infiniflow/ragflow) - 具备深度文档理解能力的 RAG 引擎。 ![GitHub stars](https://img.shields.io/github/stars/infiniflow/ragflow?style=social)
- [GraphRAG (Microsoft)](https://github.com/microsoft/graphrag) - 基于知识图谱的 RAG。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/graphrag?style=social)
- [Docling](https://github.com/docling-project/docling) - 文档处理工具包，将 PDF 等文件转换为 GenAI 工作流所需的结构化数据。 ![GitHub stars](https://img.shields.io/github/stars/docling-project/docling?style=social)
- [Unstructured](https://github.com/Unstructured-IO/unstructured) - 一流的文档预处理工具。 ![GitHub stars](https://img.shields.io/github/stars/Unstructured-IO/unstructured?style=social)
- [MinerU](https://github.com/opendatalab/MinerU) - 面向 LLM 和 RAG 工作流的高精度文档解析。通过 VLM+OCR 双引擎将 PDF、Word、PPT 和图像转换为结构化的 Markdown/JSON。 ![GitHub stars](https://img.shields.io/github/stars/opendatalab/MinerU?style=social)
- [Marker](https://github.com/datalab-to/marker) - 快速准确的 PDF 转 Markdown 工具，支持表格提取、公式处理，并可选 LLM 增强以服务 RAG 流水线。 ![GitHub stars](https://img.shields.io/github/stars/datalab-to/marker?style=social)
- [ColPali / ColQwen](https://github.com/illuin-tech/colpali) - 用于文档检索的视觉语言模型。 ![GitHub stars](https://img.shields.io/github/stars/illuin-tech/colpali?style=social)
- [LightRAG](https://github.com/HKUDS/LightRAG) - 基于图的 RAG，配有双层检索系统。简单快速，具备全面的知识发现能力（EMNLP 2025）。 ![GitHub stars](https://img.shields.io/github/stars/HKUDS/LightRAG?style=social)
- [RAG-Anything](https://github.com/HKUDS/RAG-Anything) - 一体化多模态 RAG 系统，无缝处理文本、图像、表格和公式。构建于 LightRAG 之上。 ![GitHub stars](https://img.shields.io/github/stars/HKUDS/RAG-Anything?style=social)
- [RAGLite (Superlinear)](https://github.com/superlinear-ai/raglite) - 基于 DuckDB 或 PostgreSQL 的 Python RAG 工具包。轻量高效，无重型依赖即可实现检索增强生成。MPL 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/superlinear-ai/raglite?style=social)
- [GPT-RAG (Azure)](https://github.com/Azure/GPT-RAG) - 面向 Azure OpenAI 的企业级 RAG 模式，可大规模部署。安全、生产就绪的架构，使用 Azure Cognitive Search 与 Azure OpenAI LLM 提供 ChatGPT 风格的问答体验。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Azure/GPT-RAG?style=social)
- [LangChain4j](https://github.com/langchain4j/langchain4j) - 将 LLM 集成到 Java 应用的 Java 库。实现了 RAG、工具调用（含 MCP 支持）和 Agent，可无缝集成到 Spring Boot 等企业级 Java 框架。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/langchain4j/langchain4j?style=social)
- [Kernel Memory (Microsoft)](https://github.com/microsoft/kernel-memory) - 面向用户、团队和应用的记忆方案。提供包含文档摄取、向量索引和带引用来源的自然语言查询的 RAG 流水线。支持多种 LLM 提供商和向量存储。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/kernel-memory?style=social)
- [txtai](https://github.com/neuml/txtai) - 一体化 AI 框架，涵盖语义搜索、LLM 编排和语言模型工作流。附带可定制流水线的嵌入数据库。 ![GitHub stars](https://img.shields.io/github/stars/neuml/txtai?style=social)
- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) - 高效的 RAG 研究工具包，内置 40 多个检索与重排模型、20 多个基准数据集和优化的评估流水线（WWW 2025 Resource）。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/RUC-NLPIR/FlashRAG?style=social)
- [DocsGPT](https://github.com/arc53/DocsGPT) - 私有 AI 平台，用于结合企业搜索构建智能 Agent 与助手。提供 Agent Builder、深度研究工具、多格式文档分析和多模型支持。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/arc53/DocsGPT?style=social)
- [llmware](https://github.com/llmware-ai/llmware) - 使用小型专用模型构建企业级 RAG 流水线的统一框架。针对 AI PC 和本地部署优化，目录中收录 300 多个模型。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/llmware-ai/llmware?style=social)
- [AutoFlow](https://github.com/pingcap/autoflow) - 构建于 TiDB Vector 和 LlamaIndex 之上的图 RAG 对话式知识库工具。提供 Perplexy 风格的搜索并内置网站爬虫。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/pingcap/autoflow?style=social)
- [Chonkie](https://github.com/chonkie-inc/chonkie) - 轻量级文档分块库，为快速高效的 RAG 流水线而生。内存安全，提供多种分块策略（语义、token、递归）并支持直接对接向量数据库。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/chonkie-inc/chonkie?style=social)
- [PageIndex (VectifyAI)](https://github.com/VectifyAI/PageIndex) - 无需向量数据库、基于推理的 RAG 框架，利用文档索引结构。通过智能上下文工程和基于推理的检索，在不依赖向量数据库的情况下实现高精度。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/VectifyAI/PageIndex?style=social)
- [Kotaemon (Cinnamon)](https://github.com/Cinnamon/kotaemon) - 开源的、基于 RAG 的文档对话工具。混合 RAG 流水线，兼具全文与向量检索器、重排序和多模态能力。界面基于 Gradio，简洁清爽，支持本地 LLM 和 API 方式调用。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Cinnamon/kotaemon?style=social)
- [Reader (Jina AI)](https://github.com/jina-ai/reader) - 只需加上简单前缀（r.jina.ai）即可将任意 URL 转换为适合 LLM 的输入。免费服务，可提取文章内容、去除杂乱元素，为 RAG 和 Agent 工作流返回干净的 Markdown。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/jina-ai/reader?style=social)
- [UltraRAG (OpenBMB)](https://github.com/OpenBMB/UltraRAG) - 首个基于 Model Context Protocol (MCP) 架构的轻量级 RAG 框架。低代码 RAG 流水线构建器，配有完整的评估体系和 DeepResearch 能力。来自清华 THUNLP、NEUIR、OpenBMB 和 AI9stars。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/OpenBMB/UltraRAG?style=social)
- [Semantic Router](https://github.com/aurelio-labs/semantic-router) - 面向 LLM 和 Agent 的超快速 AI 决策层。利用语义向量空间按语义含义路由请求，无需等待缓慢的 LLM 生成。将路由时间从数秒缩短至毫秒。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/aurelio-labs/semantic-router?style=social)
- [Pathway](https://github.com/pathwaycom/pathway) - 面向流式处理、实时分析、LLM 流水线和 RAG 的 Python ETL 框架。提供 350 多个连接器，可从 SharePoint、Google Drive、S3、Kafka、PostgreSQL 等保持数据实时同步。BSL 1.1 许可（4 年后转为 Apache 2.0）。 ![GitHub stars](https://img.shields.io/github/stars/pathwaycom/pathway?style=social)
- [Infinity (AI Database)](https://github.com/infiniflow/infinity) - 为 LLM 应用打造的 AI 原生数据库，对稠密向量、稀疏向量、张量（多向量）和全文提供极速混合搜索。驱动 RAGFlow 的文档引擎。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/infiniflow/infinity?style=social)
- [PrivateGPT](https://github.com/zylon-ai/private-gpt) - 私有文档问答项目，面向本地和离线 RAG 工作流，数据始终留在用户自己的环境中。 ![GitHub stars](https://img.shields.io/github/stars/zylon-ai/private-gPT?style=social)
- [FastGPT](https://github.com/labring/FastGPT) - 知识库平台，提供 RAG 检索、文档处理、可视化 AI 工作流以及自托管部署选项。 ![GitHub stars](https://img.shields.io/github/stars/labring/FastGPT?style=social)
- [MaxKB](https://github.com/1Panel-dev/MaxKB) - 可自托管的知识库与 Agent 平台，支持文档摄取、RAG 流水线和企业助手工作流。 ![GitHub stars](https://img.shields.io/github/stars/1Panel-dev/MaxKB?style=social)
- [DB-GPT](https://github.com/eosphoros-ai/DB-GPT) - 自托管 AI 数据助手，面向私有知识、数据库感知对话和数据密集型 RAG 工作流。 ![GitHub stars](https://img.shields.io/github/stars/eosphoros-ai/DB-GPT?style=social)
- [WeKnora](https://github.com/Tencent/WeKnora) - 企业知识平台，融合 RAG 检索、自主推理 Agent、Wiki 生成和多来源文档摄取。 ![GitHub stars](https://img.shields.io/github/stars/Tencent/WeKnora?style=social)
- [localGPT](https://github.com/PromtEngineer/localGPT) - 本地文档对话项目，可对文件进行私有的端侧问答而无需将数据发送到外部 API。 ![GitHub stars](https://img.shields.io/github/stars/PromtEngineer/localGPT?style=social)
- [SurfSense](https://github.com/MODSetter/SurfSense) - 注重隐私的 NotebookLM 风格工作区，团队可借助自托管 RAG 搜索、整理和查询知识。 ![GitHub stars](https://img.shields.io/github/stars/MODSetter/SurfSense?style=social)
- [PieKBS](https://github.com/pieteams/piekbs) - 本地优先的 Agent 知识搜索引擎，基于 FTS5 的全文搜索，并通过引用/支持/wiki 链接实现基于图的文档扩展。纯 Go 实现，无需嵌入模型。 ![GitHub stars](https://img.shields.io/github/stars/pieteams/piekbs?style=social)
- [Morphik](https://github.com/morphik-org/morphik-core) - 开源多模态 RAG 框架，用于在私有知识之上构建 AI 应用。可处理文本、图像和文档，内置嵌入生成与向量搜索。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/morphik-org/morphik-core?style=social)
- [Beever Atlas](https://github.com/Beever-AI/beever-atlas) - 开源 LLM 知识库，将 Neo4j 知识图谱与 Weaviate 向量数据库结合以实现基于图的 RAG。原生 MCP 服务器，支持团队聊天记录摄取（Slack、Discord、Teams、Telegram），并可通过 LiteLLM 接入自有 LLM。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Beever-AI/beever-atlas?style=social)
- [VidXP](https://github.com/grayhatdevelopers/vidxp) - 本地优先的多模态视频索引与语义搜索工具包，提供转录提取、嵌入和场景感知搜索。可作为 CLI、API 和 MCP 使用。 ![GitHub stars](https://img.shields.io/github/stars/grayhatdevelopers/vidxp?style=social)

#### 面向 RAG 的知识图谱

- [Graphiti](https://github.com/getzep/graphiti) - 为 AI Agent 构建实时时序知识图谱。追踪事实随时间的变化并保留来源出处。支持预设和可学习的本体，以适应不断演变的真实世界数据。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/getzep/graphiti?style=social)
- [Semantica](https://github.com/semantica-agi/semantica) - 原生图的基础设施，用于上下文管理、双时态知识图谱、GraphRAG 文档分块和可解释 AI 推理。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/semantica-agi/semantica?style=social)
- [Code-Graph-RAG](https://github.com/vitali87/code-graph-rag) - 多语言代码库 RAG 框架，借助 Tree-sitter 和 Memgraph 知识图谱，用 AI 查询、编辑和优化代码库。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/vitali87/code-graph-rag?style=social)

#### 网络数据摄取

- [Crawl4AI](https://github.com/unclecode/crawl4ai) - 对 LLM 友好的网页爬虫，将网站转换为干净的 Markdown，服务于 RAG 和 Agent 工作流。 ![GitHub stars](https://img.shields.io/github/stars/unclecode/crawl4ai?style=social)
- [Scrapling](https://github.com/D4Vinci/Scrapling) - 自适应网页抓取与爬取框架，可从页面到大规模流水线稳健地提取结构化数据。 ![GitHub stars](https://img.shields.io/github/stars/D4Vinci/Scrapling?style=social)
- [Lightpanda](https://github.com/lightpanda-io/browser) - 用 Zig 编写的机器优先无头浏览器；无需渲染、超轻量，专为 AI Agent 浏览场景设计。 ![GitHub stars](https://img.shields.io/github/stars/lightpanda-io/browser?style=social)
- [Paperless-AI](https://github.com/clusterzx/paperless-ai) - 面向 Paperless-ngx 的自动化文档分析器，为你的文档归档提供 RAG 驱动的语义搜索。 ![GitHub stars](https://img.shields.io/github/stars/clusterzx/paperless-ai?style=social)
- [Firecrawl](https://github.com/firecrawl/firecrawl) - 面向 AI 的 Web 数据 API——大规模搜索、抓取并与网络交互。输出干净的 markdown/JSON，自动处理代理轮换和 JS 拦截。 ![GitHub stars](https://img.shields.io/github/stars/firecrawl/firecrawl?style=social)
- [wigolo](https://github.com/KnockOutEZ/wigolo) - 本地优先的网络情报与 Model Context Protocol (MCP) 服务器，支持网络搜索、爬取和结构化数据提取。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/KnockOutEZ/wigolo?style=social)
- [Zoom Search](https://github.com/goofrey/zoom-search) - 面向 AI Agent 的 MCP 搜索与证据工具，提供查询重写、来源精确定位、带出处的答案和运行时指标。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/goofrey/zoom-search?style=social)
- [invisible-playwright](https://github.com/feder-cr/invisible_playwright) - 针对打过隐身补丁的 Firefox 150 二进制的 Playwright 封装。为需要从设有反爬防护（reCAPTCHA、FingerprintPro、Cloudflare）的站点摄取网络数据的 AI Agent 提供即插即用的 Browser 对象。伪装在 C++ 源码层面完成，而非通过 JS 覆写。MIT（封装层）+ MPL-2.0（补丁）。 ![GitHub stars](https://img.shields.io/github/stars/feder-cr/invisible_playwright?style=social)

#### 文档转换与预处理

- [pdf-inspector](https://github.com/firecrawl/pdf-inspector) - 快速的 Rust PDF 分类与文本提取库，可区分扫描件与文本型文档，从而在无需 OCR 的情况下为 RAG 流水线分流。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/firecrawl/pdf-inspector?style=social)
- [OpenDataLoader PDF](https://github.com/opendataloader-project/opendataloader-pdf) - 注重无障碍访问的 PDF 解析与转换流水线，面向 AI 就绪的 markdown 和结构化数据工作流。 ![GitHub stars](https://img.shields.io/github/stars/opendataloader-project/opendataloader-pdf?style=social)
- [MarkItDown (Microsoft)](https://github.com/microsoft/markitdown) - 将文件和办公文档转换为 Markdown 的 Python 工具。支持 PDF、PowerPoint、Word、Excel、图像、音频、HTML 等，并具备 OCR 和转录能力。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/markitdown?style=social)
- [LiteParse](https://github.com/run-llama/liteparse) - 面向 AI 和 RAG 流水线的轻量级文档解析工具包，提供 PDF/OCR 提取和开箱即用的干净预处理默认值。 ![GitHub stars](https://img.shields.io/github/stars/run-llama/liteparse?style=social)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 大规模 OCR 套件，涵盖检测、识别和版面分析，广泛用于文档数字化及下游 RAG 流水线。 ![GitHub stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?style=social)
- [DocETL (UC Berkeley)](https://github.com/ucbepic/docetl) - 由 Agent 驱动、基于 LLM 的数据处理与 ETL 系统，面向复杂文档处理。提供查询重写和评估能力，非结构化数据分析准确率比基线高出 80%。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/ucbepic/docetl?style=social)
- [kordoc](https://github.com/chrisryugj/kordoc) - TypeScript 库、CLI 和 MCP 服务器，可将韩文文档（HWP 3.x/5.x、HWPX、HWPML）以及 PDF、DOCX、XLSX、PPTX 和图像转换为 Markdown 和结构化数据。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/chrisryugj/kordoc?style=social)

#### LLM 应用框架

- [aisuite](https://github.com/andrewyng/aisuite) - 简洁统一的接口，连接多家生成式 AI 提供商。以类似 OpenAI 的标准化 API 使用 OpenAI、Anthropic、Google 等 10 多家提供商。一行代码即可切换模型或提供商。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/andrewyng/aisuite?style=social)
- [Spring AI](https://github.com/spring-projects/spring-ai) - 面向 Spring 生态的 AI 工程应用框架。为 LLM、向量存储和嵌入模型提供统一 API，可无缝集成到 Spring Boot 应用。支持 RAG、工具调用和结构化输出。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/spring-projects/spring-ai?style=social)
- [Rig](https://github.com/0xPlaygrounds/rig) - 用于构建可扩展、模块化 LLM 应用的 Rust 库。类型安全的 Agent 框架，提供统一的 LLM 接口、内置向量存储集成，以及面向生产级 AI 系统的 ergonomic 抽象。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/0xPlaygrounds/rig?style=social)
- [Ax](https://github.com/ax-llm/ax) - 用于构建可靠 AI 应用的 TypeScript 框架。TypeScript 生态中受 DSPy 启发的"官方"框架，提供类型安全的 LLM 交互、思维链推理和结构化输出校验。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/ax-llm/ax?style=social)
- [Genkit](https://github.com/genkit-ai/genkit) - 用于在 JavaScript、Go 和 Python 中构建全栈 AI 应用的开源框架。由 Google 的 Firebase 构建并用于生产环境。提供统一的接口来集成多家提供商的 AI 模型，并内置 RAG、工具调用、结构化输出和开发者工具。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/genkit-ai/genkit?style=social)
- [ContextGem](https://github.com/shcherbak-ai/contextgem) - 轻松易用的 LLM 文档抽取框架。提供强大的抽象来构建抽取工作流，包含自动化的动态提示、数据建模、校验和精确的引用映射。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/shcherbak-ai/contextgem?style=social)
- [Eino](https://github.com/cloudwego/eino) - Go 语言中终极的 LLM/AI 应用开发框架。汲取 LangChain 和 Google ADK 的经验，遵循 Go 惯例设计，为链、Agent 和工作流提供可组合组件。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/cloudwego/eino?style=social)
- [ruby_llm](https://github.com/crmne/ruby_llm) - 一个优雅的 Ruby API，覆盖 OpenAI、Anthropic、Gemini、Bedrock、Azure、OpenRouter、DeepSeek、Ollama 等 15 家以上提供商。支持 Agent、对话、视觉、音频、PDF、图像、嵌入、工具、流式输出和 Rails 集成。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/crmne/ruby_llm?style=social)
- [LangChain.rb](https://github.com/patterns-ai-core/langchainrb) - 用 Ruby 构建 LLM 驱动应用。符合 Ruby 惯例的库，用于构建 AI 应用，支持多种 LLM 提供商、向量存储和 RAG 流水线。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/patterns-ai-core/langchainrb?style=social)



---

## 6. 生成式媒体工具

> 面向图像、视频、音频和 3D 生成与编辑的开源模型和应用。

#### 图像生成与编辑

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) - 面向 Stable Diffusion、FLUX 等模型的节点式可视化工作流编辑器。 ![GitHub stars](https://img.shields.io/github/stars/Comfy-Org/ComfyUI?style=social)
- [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic) - 持续维护、基于 Forge 的 Stable Diffusion Web 界面，保留用户熟悉的扩展驱动工作方式。 ![GitHub stars](https://img.shields.io/github/stars/Haoming02/sd-webui-forge-classic?style=social)
- [Diffusers](https://github.com/huggingface/diffusers) - 面向图像、视频和音频生成的 PyTorch 扩散流水线库。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/diffusers?style=social)
- [InvokeAI](https://github.com/invoke-ai/InvokeAI) - 功能完备的创意工作室。 ![GitHub stars](https://img.shields.io/github/stars/invoke-ai/InvokeAI?style=social)
- [SD.Next](https://github.com/vladmandic/sdnext) - 一体化 Web 界面，支持 AI 生成式图像与视频创作，具备多平台支持、SDNQ 量化和均衡的 CPU/GPU 显存卸载。 ![GitHub stars](https://img.shields.io/github/stars/vladmandic/sdnext?style=social)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) - 生产级 C++ 推理运行时，支持 SD、Flux 等扩散模型，针对 CPU 和 GPU 部署做了优化。 ![GitHub stars](https://img.shields.io/github/stars/leejet/stable-diffusion.cpp?style=social)
- [ArtCraft](https://github.com/storytold/artcraft) - 交互式 AI 图像与视频创作工作室，融合 2D/3D 场景合成、受控生成、图像编辑和模型提供方选择。MIT/Apache-2.0 双许可。 ![GitHub stars](https://img.shields.io/github/stars/storytold/artcraft?style=social)
- [Upscayl](https://github.com/upscayl/upscayl) - 免费开源的 AI 图像放大工具，支持 Linux、macOS 和 Windows。采用 Real-ESRGAN 和 Vulkan 架构，通过重建高分辨率细节来增强图像。跨平台桌面应用，支持批处理。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/upscayl/upscayl?style=social)
- [upres-cli](https://github.com/auroracapital/upres-cli) - [UpRes](https://upres.ai) 的开源（MIT）CLI 和 MCP 服务器。UpRes 是一款图像与视频放大工具，提供 14 个公开模型别名（人脸、文本、印刷、绘画、动态）并支持 8K 输出。 ![GitHub stars](https://img.shields.io/github/stars/auroracapital/upres-cli?style=social)
- [Krita AI Diffusion](https://github.com/Acly/krita-ai-diffusion) - 面向 Krita 的精简 AI 图像生成插件。支持局部重绘与外扩，可选文本提示词，无需调参。集成 ComfyUI 后端，服务专业数字绘画工作流。GPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Acly/krita-ai-diffusion?style=social)

#### 换脸与深度伪造

- [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) - 仅需一张图片即可实现实时换脸和一键视频深度伪造。为直播和内容创作提供高质量换脸。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/hacksider/Deep-Live-Cam?style=social)
- [Faceswap](https://github.com/deepfakes/faceswap) - 用于识别和替换图片、视频中人脸的开源深度学习软件。GPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/deepfakes/faceswap?style=social)

#### 肖像动画


#### 视频生成

- [Hyperframes (HeyGen)](https://github.com/heygen-com/hyperframes) - 开源长视频生成平台，面向电影级和社交媒体视频创作，提供时间线控制、基于扩散的动态模块和多模态条件控制。 ![GitHub stars](https://img.shields.io/github/stars/heygen-com/hyperframes?style=social)
- [LTX-2 (Lightricks)](https://github.com/Lightricks/LTX-2) - LTX-2 音视频生成模型的官方 Python 推理与 LoRA 训练器包。 ![GitHub stars](https://img.shields.io/github/stars/Lightricks/LTX-2?style=social)
- [Helios (PKU-YuanGroup)](https://github.com/PKU-YuanGroup/Helios) - 高效的长视频生成框架，24GB 显存即可支持最长 10,000 帧（5 分钟以上）和 1280×768 分辨率。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/PKU-YuanGroup/Helios?style=social)
- [Pixelle-Video (AIDC-AI)](https://github.com/AIDC-AI/Pixelle-Video) - 优化的文生视频基础模型，专为长时长连贯场景和可控生成工作流而生。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/AIDC-AI/Pixelle-Video?style=social)
- [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) - 端到端的短视频生成流水线，可自动完成脚本、素材收集、配音和字幕合成。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/harry0703/MoneyPrinterTurbo?style=social)
- [OpenMontage](https://github.com/calesthio/OpenMontage) - Agent 驱动的视频制作平台，编排剧本创作、素材生成、语音合成、剪辑和渲染。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/calesthio/OpenMontage?style=social)
- [ViMax](https://github.com/HKUDS/ViMax) - 多 Agent 视频生成框架，编排剧本创作、分镜、角色设计和时序视觉一致性，实现端到端视频合成。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/HKUDS/ViMax?style=social)
- [WhisperLive](https://github.com/collabora/WhisperLive) - 近乎实时的 OpenAI Whisper 实现，用于实时语音转文字。支持 faster-whisper、tensorrt 和 openvino 后端，并提供 WebSocket 流式传输。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/collabora/WhisperLive?style=social)
- [OpenShorts](https://github.com/mutonby/openshorts) - 自托管应用，利用 faster-whisper 转录、LLM 片段选取、人脸跟踪 9:16 重构和动态字幕，把长视频变成竖版短片。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/mutonby/openshorts?style=social)
#### 音频 / 音乐 / 语音生成

- [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) - 本地优先的音乐生成模型，广泛支持 Mac、AMD、Intel 和 CUDA 等各类硬件。 ![GitHub stars](https://img.shields.io/github/stars/ace-step/ACE-Step-1.5?style=social)
- [Magenta RealTime 2](https://github.com/magenta/magenta-realtime) - 开放权重的实时音乐模型，支持流式生成和实时交互。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/magenta/magenta-realtime?style=social)
- [YuE](https://github.com/multimodal-art-projection/YuE) - 整曲音乐生成基础模型，具备符号化的旋律与和弦规划、声学合成、零样本翻唱和歌曲编辑能力。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/multimodal-art-projection/YuE?style=social)
- [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools) - Stability AI 的开源音频与音乐生成模型。潜空间扩散模型，可基于元数据和时序条件生成音频，提供更快的推理速度，并为音效和音乐创作带来更强的控制力。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Stability-AI/stable-audio-tools?style=social)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) - 仅需 1 分钟语音数据即可完成少样本声音克隆。结合 GPT 与 SoVITS 架构实现高质量 TTS，支持跨语言和情感表达。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/RVC-Boss/GPT-SoVITS?style=social)
- [Supertonic](https://github.com/supertone-inc/supertonic) - 极速端侧多语言语音合成系统，通过 ONNX 原生运行。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/supertone-inc/supertonic?style=social)
- [Voicebox](https://github.com/jamiepine/voicebox) - 本地优先的 AI 语音工作室，可克隆音色、生成多语言语音并在本地听写文本。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/jamiepine/voicebox?style=social)
- [Voice-Pro](https://github.com/abus-aikorea/voice-pro) - 多语言 AI 语音识别、文本转语音和零样本声音克隆 Web 界面。GPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/abus-aikorea/voice-pro?style=social)
- [VoiceStudio](https://github.com/debpalash/VoiceStudio) - 本地优先的语音平台，支持多引擎文本转语音、声音克隆与设计、视频配音和听写。 ![GitHub stars](https://img.shields.io/github/stars/debpalash/VoiceStudio?style=social)

#### 3D 与创意工具

- [Modly](https://github.com/lightningpixel/modly) - 使用本地 GPU 加速 AI 模型进行图像转 3D 网格生成的桌面应用。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/lightningpixel/modly?style=social)
- [gsplat (3D Gaussian Splatting tools)](https://github.com/nerfstudio-project/gsplat) - 高性能 3D Gaussian Splatting 库。 ![GitHub stars](https://img.shields.io/github/stars/nerfstudio-project/gsplat?style=social)
- [LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio) - 原生应用，用于训练、编辑和导出 3D Gaussian Splatting 场景，支持 MCMC 优化和延时摄影生成。GPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/MrNeRF/LichtFeld-Studio?style=social)
- [OpenSplat](https://github.com/pierotofy/OpenSplat) - 生产级、可移植的 3D Gaussian Splatting 实现，支持 Windows、Mac 和 Linux 的 CPU/GPU。可从相机位姿和稀疏点云创建 3D 场景。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/pierotofy/OpenSplat?style=social)
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) - 用于高保真图像转 3D 生成的 3D 生成模型，采用紧凑的结构化潜表示。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/TRELLIS.2?style=social)
- [MCP for Blender](https://github.com/ahujasid/mcp-for-blender) - Blender 插件和 MCP 服务器，让 LLM Agent 通过结构化工具创建和操作 3D 场景。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/ahujasid/mcp-for-blender?style=social)

---

<a id="section-7"></a>

## 7. 训练与微调生态

> 面向模型训练、微调、合成数据生成和分布式训练的工具。

#### 完整训练框架

- [Oumi](https://github.com/oumi-ai/oumi) - 完全开源的基础模型全生命周期平台——覆盖从数据准备、训练到评估和部署的全流程。支持 100 多个模型，提供 200 多套配方，可微调 gpt-oss、Qwen3、DeepSeek-R1 等。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/oumi-ai/oumi?style=social)
- [Marin](https://github.com/marin-community/marin) - 面向基础模型开发的开源框架，为现代 LLM 研究提供可组合的数据、训练和评估流水线。 ![GitHub stars](https://img.shields.io/github/stars/marin-community/marin?style=social)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - 一站式统一框架，支持 SFT、DPO、ORPO、KTO，并提供 Web 界面。 ![GitHub stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) - 由 YAML 驱动的完整流水线，覆盖 SFT、DPO、GRPO。 ![GitHub stars](https://img.shields.io/github/stars/axolotl-ai-cloud/axolotl?style=social)
- [ms-swift](https://github.com/modelscope/ms-swift) - 统一训练框架，支持 600+ LLM 和 300+ MLLM，涵盖 CPT/SFT/DPO/GRPO（AAAI 2025）。 ![GitHub stars](https://img.shields.io/github/stars/modelscope/ms-swift?style=social)
- [Unsloth](https://github.com/unslothai/unsloth) - 微调速度提升 2 倍，显存占用减少 70%。 ![GitHub stars](https://img.shields.io/github/stars/unslothai/unsloth?style=social)
- [LitGPT](https://github.com/Lightning-AI/litgpt) - 20 多个 LLM 的清晰从零实现。 ![GitHub stars](https://img.shields.io/github/stars/Lightning-AI/litgpt?style=social)
- [torchtune](https://github.com/pytorch/torchtune) - PyTorch 原生库，用于 LLM 的后训练、微调和实验。 ![GitHub stars](https://img.shields.io/github/stars/pytorch/torchtune?style=social)
- [kohya_ss](https://github.com/bmaltais/kohya_ss) - 基于 Gradio 的 GUI 和 CLI，用于训练 Stable Diffusion 模型（LoRA、Dreambooth、微调、SDXL）。为 Kohya 功能强大的训练脚本提供了易用的界面。 ![GitHub stars](https://img.shields.io/github/stars/bmaltais/kohya_ss?style=social)
- [TRL (Transformers Reinforcement Learning)](https://github.com/huggingface/trl) - 支持 RLHF、SFT、DPO、ORPO 的官方库。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/trl?style=social)
- [verl](https://github.com/volcengine/verl) - 火山引擎的 LLM 强化学习框架，支持 PPO、GRPO、REINFORCE++、DAPO（EuroSys 2025）。 ![GitHub stars](https://img.shields.io/github/stars/volcengine/verl?style=social)
- [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) - 可扩展的高效模型强化工具包，基于 DTensor 和 Megatron 后端。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA-NeMo/RL?style=social)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - 易于使用、可扩展的 RLHF 框架，基于 Ray。支持 PPO、GRPO、REINFORCE++、DAPO，内置 vLLM 集成和异步训练。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/OpenRLHF/OpenRLHF?style=social)
- [LMFlow](https://github.com/OptimalScale/LMFlow) - 用于大型基础模型微调和推理的可扩展工具包。提供 RAFT 对齐算法和全面的模型支持。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/OptimalScale/LMFlow?style=social)
- [XTuner](https://github.com/InternLM/xtuner) - 面向超大规模 MoE 模型打造的下一代训练引擎，支持高效 QLoRA 和全参数微调。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)
- [Ludwig](https://github.com/ludwig-ai/ludwig) - 用于构建自定义 LLM 和深度神经网络的低代码框架。通过声明式 YAML 配置训练前沿模型，支持 PEFT/LoRA、4-bit 量化、基于 Hugging Face Accelerate 的分布式训练以及原生 Kubernetes 支持。Linux Foundation AI 项目。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/ludwig-ai/ludwig?style=social)
- [TorchTitan (PyTorch)](https://github.com/pytorch/torchtitan) - PyTorch 原生平台，用于大规模训练生成式 AI 模型。展示了 4D 并行（FSDP、张量、流水线、上下文）在大语言模型预训练中的用法，相比优化后的基线提速 65% 以上。BSD-3-Clause 许可。 ![GitHub stars](https://img.shields.io/github/stars/pytorch/torchtitan?style=social)
- [VeOmni (ByteDance)](https://github.com/ByteDance-Seed/VeOmni) - 兼容单模态和多模态预训练与后训练的通用框架。以模型为中心的分布式配方库，支持文本、视觉、音频和视频模型，提供统一的训练接口。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/ByteDance-Seed/VeOmni?style=social)
- [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio) - 用于微调 LLM 的无代码 GUI 框架。为 SFT、奖励建模和模型部署提供精简界面。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/h2oai/h2o-llmstudio?style=social)
- [PRIME-RL](https://github.com/PrimeIntellect-ai/prime-rl) - 来自 Prime Intellect 的规模化 Agent 强化学习训练框架。面向大规模强化学习，可扩展至 1000 多块 GPU，具备全异步 RL、FSDP2 训练和 vLLM 推理。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/PrimeIntellect-ai/prime-rl?style=social)
- [slime](https://github.com/THUDM/slime) - 来自 THUDM 的 LLM 后训练框架，面向 RL Scaling。支持带多轮编译反馈的 SFT 和 RL 训练，支撑了 TritonForge 等自动 GPU kernel 生成项目。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/THUDM/slime?style=social)
- [rLLM](https://github.com/rllm-org/rllm) - 让 LLM 强化学习更大众化。用于通过 RL 训练 AI Agent 的框架，特点是代码改动接近于零、CLI 优先的工作流和 50 多个内置基准测试。支持 GRPO、REINFORCE、RLOO，可搭配 verl 和 tinker 后端。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/rllm-org/rllm?style=social)
- [EasyR1](https://github.com/hiyouga/EasyR1) - 基于 veRL 的高效、可扩展多模态 RL 训练框架。扩展 veRL 以支持视觉语言模型，使用 GRPO 算法进行高效 RL 训练。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/hiyouga/EasyR1?style=social)
- [LeRobot](https://github.com/huggingface/lerobot) - 通过端到端学习让机器人领域的 AI 更易获取。提供模仿学习和强化学习的前沿方案，配套预训练模型、数据集和仿真环境。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/lerobot?style=social)
- [AI-Toolkit](https://github.com/ostris/ai-toolkit) - 微调扩散模型的终极训练工具包。易用的一体化训练套件，支持 FLUX.1、FLUX.2、Stable Diffusion 和视频模型，提供 GUI 和 CLI 两种界面。适合消费级硬件，全面支持 LoRA 和全量微调。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/ostris/ai-toolkit?style=social)
- [OneTrainer](https://github.com/Nerogar/OneTrainer) - 一站式满足所有扩散模型训练需求。支持 FLUX、Stable Diffusion 1.5/2.x/3.x/SDXL、Würstchen、PixArt、Hunyuan Video 等。具备全量微调、LoRA、嵌入、掩码训练、自动备份和 TensorBoard 集成。GPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Nerogar/OneTrainer?style=social)
- [FluxGym](https://github.com/cocktailpeanut/fluxgym) - 极简的 FLUX LoRA 训练界面，支持低显存（12GB/16GB/20GB）。Web 界面派生自 AI-Toolkit，后端由 Kohya Scripts 提供支持。兼具 Gradio 界面的简洁与 Kohya 训练脚本的灵活性。GPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/cocktailpeanut/fluxgym?style=social)
- [MiniMind](https://github.com/jingyaogong/minimind) - 只需 2 小时、3 美元即可从零训练一个 6400 万参数的 LLM。完整的从零实现，涵盖 MoE、数据清洗、预训练、SFT、LoRA、RLHF（DPO/PPO/GRPO）、工具使用和模型蒸馏。所有核心算法均以纯 PyTorch 实现，不依赖高层抽象。是理解 LLM 内部原理的教学框架。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/jingyaogong/minimind?style=social)
- [FastChat](https://github.com/lm-sys/FastChat) - 用于训练、服务和评测大语言模型聊天机器人的开放平台。驱动 Chatbot Arena (lmarena.ai)，为 70 多个 LLM 服务每日 1000 多万次请求。包含 Vicuna 的训练代码、MT-Bench 评测，以及兼容 OpenAI API 的分布式多模型服务。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/lm-sys/FastChat?style=social)
- [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) - 构建于百度 PaddlePaddle 框架之上、易用且强大的 LLM 库。支持 100 多个模型，可在多种硬件上进行高效训练、压缩和高性能推理。具备 RsLoRA+ 算法、支持 DeepSeek V3/R1 与 FP8/INT8 量化，并提供统一的检查点机制。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleNLP?style=social)
- [Soup](https://github.com/MakazhanAlpamys/Soup) - LLM 后训练的一站式 CLI：SFT、DPO、GRPO、KTO、ORPO，外加评估门控与导出。分层流式训练可将 8B 模型放在 4 GB 显存的笔记本 GPU 上训练，方法是从主机内存中逐解码层流式传输冻结的基础模型。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/MakazhanAlpamys/Soup?style=social)

#### LoRA / PEFT 工具

- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft) - 提供 LoRA、QLoRA、DoRA 等方法的官方库。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/peft?style=social)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) - 用于加速训练的极速自定义 kernel。 ![GitHub stars](https://img.shields.io/github/stars/linkedin/Liger-Kernel?style=social)
- [MergeKit](https://github.com/arcee-ai/mergekit) - 高级模型合并工具。 ![GitHub stars](https://img.shields.io/github/stars/arcee-ai/mergekit?style=social)

#### 合成数据生成

- [distilabel](https://github.com/argilla-io/distilabel) - 端到端的合成指令数据流水线。 ![GitHub stars](https://img.shields.io/github/stars/argilla-io/distilabel?style=social)
- [Data-Juicer](https://github.com/alibaba/data-juicer) - 面向 LLM 训练的高性能数据处理工具。 ![GitHub stars](https://img.shields.io/github/stars/alibaba/data-juicer?style=social)
- [Argilla](https://github.com/argilla-io/argilla) - 开源数据标注 + 合成数据平台。 ![GitHub stars](https://img.shields.io/github/stars/argilla-io/argilla?style=social)
- [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV) - 高保真的表格和关系型合成数据。 ![GitHub stars](https://img.shields.io/github/stars/sdv-dev/SDV?style=social)
- [DataTrove (Hugging Face)](https://github.com/huggingface/datatrove) - 与平台无关的数据处理流水线，面向大规模 LLM 训练。可在本地机器或 SLURM 集群上完成过滤、去重和分词。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/datatrove?style=social)
- [Bespoke Curator](https://github.com/bespokelabsai/curator) - 面向后训练和结构化数据提取的合成数据整理工具。借助批处理和进度跟踪，轻松围绕 LLM 构建流水线。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/bespokelabsai/curator?style=social)
- [SDG (Harbin Institute)](https://github.com/hitsz-ids/synthetic-data-generator) - 专门用于生成高质量结构化表格合成数据的框架，基于 CTGAN 模型并支持十亿级数据处理。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/hitsz-ids/synthetic-data-generator?style=social)

#### 分布式训练

- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) - 面向超大规模训练的优化方案。 ![GitHub stars](https://img.shields.io/github/stars/deepspeedai/DeepSpeed?style=social)
- [Colossal-AI](https://github.com/hpcaitech/ColossalAI) - 面向 1000 亿参数以上模型的统一系统。 ![GitHub stars](https://img.shields.io/github/stars/hpcaitech/ColossalAI?style=social)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - 分布式训练框架和大规模 Transformer 模型的参考实现。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/Megatron-LM?style=social)
- [Ray Train](https://github.com/ray-project/ray) - 可扩展的分布式训练。 ![GitHub stars](https://img.shields.io/github/stars/ray-project/ray?style=social)
- [Nanotron (Hugging Face)](https://github.com/huggingface/nanotron) - 极简的张量并行、流水线并行和数据并行的 LLM 预训练方案。以简洁和快速为设计目标。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/nanotron?style=social)
- [RLinf](https://github.com/RLinf/RLinf) - 可扩展的开源 RL 基础设施，通过强化学习对基础模型进行后训练。提供 M2Flow 范式以支持具身智能和 Agent 工作流，并可与真实世界的机器人集成。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/RLinf/RLinf?style=social)
- [dstack](https://github.com/dstackai/dstack) - 与厂商无关的编排方案，在云、Kubernetes 和裸机上跨 NVIDIA、AMD、TPU 和 Tenstorrent 运行训练、推理和 Agent 工作负载。MPL-2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/dstackai/dstack?style=social)
- [Streaming (MosaicML)](https://github.com/mosaicml/streaming) - 高性能数据流式传输库，用于高效的神经网络训练。通过本地缓存和确定性打乱，将训练数据从云存储（S3、GCS、Azure）流式读取。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/mosaicml/streaming?style=social)
- [Higgsfield](https://github.com/higgsfield-ai/higgsfield) - 容错 GPU 编排与分布式机器学习框架，用于训练十亿级至万亿级参数规模的模型。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/higgsfield-ai/higgsfield?style=social)

#### 模型量化与优化

- [LLM Compressor (vLLM)](https://github.com/vllm-project/llm-compressor) - 兼容 Transformers 的库，为优化部署于 vLLM 的 LLM 应用多种压缩算法。支持 GPTQ、AWQ、SmoothQuant、AutoRound 和 FP8/INT8 量化，与 Hugging Face 无缝集成。 ![GitHub stars](https://img.shields.io/github/stars/vllm-project/llm-compressor?style=social)
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) - 汇集 SOTA 模型优化技术的统一库，涵盖量化、剪枝、蒸馏和投机解码。可压缩深度学习模型，配合 TensorRT-LLM、TensorRT 和 vLLM 部署，在 NVIDIA 硬件上优化推理速度。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/Model-Optimizer?style=social)

---
## 8. MLOps / LLMOps 与生产化

> 用于在生产环境中跟踪、部署、监控和运维 AI 系统的工具链。

#### 实验跟踪与版本管理

- [MLflow](https://github.com/mlflow/mlflow) - 面向 ML/LLM 全生命周期的端到端开放平台。 ![GitHub stars](https://img.shields.io/github/stars/mlflow/mlflow?style=social)
- [DVC (Data Version Control)](https://github.com/iterative/dvc) - 面向数据和模型的类 Git 版本控制。 ![GitHub stars](https://img.shields.io/github/stars/iterative/dvc?style=social)
- [ClearML](https://github.com/clearml/clearml) - 开源平台，覆盖实验跟踪、编排、数据管理和模型服务。 ![GitHub stars](https://img.shields.io/github/stars/clearml/clearml?style=social)
- [Weights & Biases Weave](https://github.com/wandb/weave) - 开源的追踪与实验跟踪工具。 ![GitHub stars](https://img.shields.io/github/stars/wandb/weave?style=social)
- [Aim](https://github.com/aimhubio/aim) - 可自托管的 ML 实验跟踪器，设计上可承载数万次训练运行，提供高性能 UI 以及用于程序化访问的 SDK。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/aimhubio/aim?style=social)
- [Feast](https://github.com/feast-dev/feast) - 面向 ML 的开源特征存储。管理离线/在线特征存储，并通过时间点一致性防止数据泄漏。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/feast-dev/feast?style=social)
- [OpenLineage](https://github.com/OpenLineage/OpenLineage) - 用于采集血缘元数据的开放标准，旨在对运行中的作业进行埋点。定义了 run、job、dataset 实体的通用模型，以实现一致的数据血缘追踪。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/OpenLineage/OpenLineage?style=social)
- [Marquez](https://github.com/MarquezProject/marquez) - LF AI & Data Foundation 毕业项目，用于元数据的采集、聚合与可视化。维护数据集如何被消费和产生的溯源信息，并全局呈现作业运行时与数据集生命周期管理。与 OpenLineage 集成。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/MarquezProject/marquez?style=social)

#### 模型中心与注册中心

- [Civitai](https://github.com/civitai/civitai) - 开源 AI 模型中心与社区平台，用于分享和发现生成式 AI 模型，聚焦图像生成模型。提供模型版本管理、评价与集成推理。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/civitai/civitai?style=social)
- [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) - Hugging Face Hub 的官方 Python 客户端。以程序化方式下载、上传和管理 100 万+ 开源 ML 模型与数据集。是模型共享与分发的事实标准。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/huggingface_hub?style=social)
- [ModelScope](https://github.com/modelscope/modelscope) - Model-as-a-Service 平台，汇聚 AI 社区 700+ 前沿 ML 模型。覆盖 NLP、CV、音频、多模态以及 AI for Science，提供精简的模型推理、微调与评估能力。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/modelscope/modelscope?style=social)
- [OpenVINO Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) - 针对 Intel 硬件优化的预训练深度学习模型与示例。提供 200+ 视觉、语音和 NLP 公开预训练模型，附带基准测试工具与精度指标。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/openvinotoolkit/open_model_zoo?style=social)
- [ONNX Model Zoo](https://github.com/onnx/models) - ONNX 格式的预训练前沿模型合集。包含 80+ 覆盖视觉、NLP 和音频的模型，并提供验证数据与参考实现。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/onnx/models?style=social)
- [Transformers.js](https://github.com/huggingface/transformers.js) - 面向 Web 的前沿机器学习方案。无需服务器即可在浏览器中直接运行 Hugging Face Transformers。通过 ONNX Runtime Web 支持 1000+ 模型，包括 BERT、GPT-2、T5 等。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/transformers.js?style=social)
- [DJL (Deep Java Library)](https://github.com/deepjavalibrary/djl) - 与引擎无关的 Java 深度学习框架，内置模型库。以统一 API 加载和运行 PyTorch、TensorFlow、MXNet 和 ONNX 模型。包含 80+ 面向 CV 和 NLP 的预训练模型。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/deepjavalibrary/djl?style=social)
- [TorchVision Models](https://github.com/pytorch/vision) - PyTorch 官方计算机视觉库，包含 50+ 预训练模型架构，涵盖 ResNet、EfficientNet、Vision Transformers (ViT)、ConvNeXt 等。是 PyTorch 计算机视觉的事实标准模型库。BSD-3-Clause 许可。 ![GitHub stars](https://img.shields.io/github/stars/pytorch/vision?style=social)
- [TensorFlow Model Garden](https://github.com/tensorflow/models) - TensorFlow 官方前沿 (SOTA) 模型与建模方案仓库。包含 BERT、ResNet、Transformer 等大量参考实现，附带预训练权重和训练脚本。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/tensorflow/models?style=social)
- [PINTO Model Zoo](https://github.com/PINTO0309/PINTO_model_zoo) - 存放各框架之间相互转换模型的仓库。支持 TensorFlow、PyTorch、ONNX、OpenVINO、TFJS、TFTRT、TensorFlowLite (Float32/16/INT8)、EdgeTPU 和 CoreML。4100+ stars，并提供面向边缘部署的丰富模型转换工具。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/PINTO0309/PINTO_model_zoo?style=social)
- [Cerebras Model Zoo](https://github.com/Cerebras/modelzoo) - 针对 Cerebras 硬件优化的深度学习模型与工具合集。包含 Llama、Mixtral、DINOv2 和 Llava 的参考实现，附配置文件、数据预处理工具和检查点转换器。1150+ stars。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Cerebras/modelzoo?style=social)
- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) -  Comprehensive 图像识别与分类工具包，模型库丰富。5800+ stars，涵盖 24 个分类网络系列、122 个预训练模型，以及包含 PP-ShiTuV2 在内的端到端图像识别系统。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleClas?style=social)


#### 模型打包与部署

- [Cog (Replicate)](https://github.com/replicate/cog) - 将 ML 模型容器化并以生产级推理服务器进行部署。把模型打包进标准化容器，自动生成 API、支持 GPU，并可用一条命令完成部署。支撑着 Replicate 上数千个生产级 AI 模型。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/replicate/cog?style=social)

#### 部署与编排

- [BentoML](https://github.com/bentoml/BentoML) - 用于构建、发布和扩展 AI 应用的统一框架。 ![GitHub stars](https://img.shields.io/github/stars/bentoml/BentoML?style=social)
- [ZenML](https://github.com/zenml-io/zenml) - 面向 ML 与 LLM 系统从开发走向生产的流水线和编排框架。 ![GitHub stars](https://img.shields.io/github/stars/zenml-io/zenml?style=social)
- [Kubeflow](https://github.com/kubeflow/kubeflow) - Kubernetes 原生的 ML/LLM 平台。 ![GitHub stars](https://img.shields.io/github/stars/kubeflow/kubeflow?style=social)
- [KServe](https://github.com/kserve/kserve) - 基于 Kubernetes 的模型服务。 ![GitHub stars](https://img.shields.io/github/stars/kserve/kserve?style=social)
- [Metaflow](https://github.com/Netflix/metaflow) - Netflix 用于构建和管理真实生产 AI 系统的 ML 平台。支撑着 Netflix、Amazon 和 DoorDash 的数千个项目。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Netflix/metaflow?style=social)
- [Flyte](https://github.com/flyteorg/flyte) - 面向 AI/ML 流水线的 Kubernetes 原生工作流编排平台。提供动态、有韧性的编排，具备很强的类型安全性和可复现性。被 Lyft、Spotify 和 Gojek 使用。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/flyteorg/flyte?style=social)
- [Prefect](https://github.com/prefecthq/prefect) - 用于构建高韧性数据与 ML 流水线的工作流编排框架。Python 原生，具备现代可观测性能力和 200+ 集成。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/prefecthq/prefect?style=social)
- [Dagster](https://github.com/dagster-io/dagster) - 云原生的编排平台，用于开发和维护包括 ML 模型在内的数据资产。声明式编程模型，内置血缘追踪与可观测性。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/dagster-io/dagster?style=social)
- [Kubeflow Pipelines](https://github.com/kubeflow/pipelines) - Kubeflow 的机器学习流水线。基于 Kubernetes 和 Argo 构建、部署可移植可扩展 ML 工作流的平台。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/kubeflow/pipelines?style=social)
- [Argo Workflows](https://github.com/argoproj/argo-workflows) - CNCF 毕业的容器原生工作流引擎，用于在 Kubernetes 上编排并行作业。是 Kubeflow Pipelines 的底层支撑，并广泛用于大规模 ML/数据处理。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/argoproj/argo-workflows?style=social)
- [MLRun](https://github.com/mlrun/mlrun) - 开源 AI 编排平台，用于快速构建和管理贯穿全生命周期的持续 ML 与生成式 AI 应用。自动化数据准备、模型调优与部署。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/mlrun/mlrun?style=social)
- [Kestra](https://github.com/kestra-io/kestra) - 面向关键任务工作流的事件驱动编排与调度平台。采用 Infrastructure-as-Code 理念，提供声明式 YAML、Git 版本控制集成，以及面向数据流水线和 ML 工作流的数百个插件。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/kestra-io/kestra?style=social)
- [KitOps](https://github.com/jozu-ai/kitops) - CNCF 开源 DevOps 工具，用于对 AI/ML 模型、数据集、代码和配置进行打包、版本管理与安全共享。将所有内容打包为 OCI 制品，存储在现有的容器镜像仓库中。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/jozu-ai/kitops?style=social)
- [Polyaxon](https://github.com/polyaxon/polyaxon) - 用于管理与编排机器学习全生命周期的 MLOps 工具。在 Kubernetes 上提供可复现、可扩展的机器学习工作流，涵盖实验跟踪、模型管理和流水线编排。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/polyaxon/polyaxon?style=social)
- [Netflix Maestro](https://github.com/Netflix/maestro) - Netflix 面向超大规模数据与 ML 流水线的下一代工作流编排器。高度可扩展、灵活的调度器，可处理跨数千节点、规模达百万级的工作流。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Netflix/maestro?style=social)
- [HAMi](https://github.com/Project-HAMi/HAMi) - Kubernetes 上的异构 GPU 共享。CNCF 沙盒项目，为异构加速器 (GPU、NPU、MLU) 提供 GPU 虚拟化、切片和调度，以高效管理 AI 工作负载。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Project-HAMi/HAMi?style=social)
- [NVIDIA KAI Scheduler](https://github.com/NVIDIA/KAI-Scheduler) - 面向大规模 AI 工作负载的 Kubernetes 原生 GPU 调度器。最初由 Run:ai 开发，现由 NVIDIA 开源。通过动态分配和高效队列管理优化 GPU 资源分配。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/KAI-Scheduler?style=social)
- [NVIDIA DeepOps](https://github.com/NVIDIA/deepops) - 基础设施自动化工具，用于基于 Kubernetes 和 Slurm 构建 GPU 集群。可部署多节点 GPU 集群，并提供监控、日志和存储，服务于 AI/HPC 工作负载。BSD-3-Clause 许可。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/deepops?style=social)
- [SkyPilot](https://github.com/skypilot-org/skypilot) - 在任意 AI 基础设施上运行、管理和扩展 AI 工作负载。统一接口接入和管理 Kubernetes、Slurm 以及 20+ 云厂商的计算资源。被 Shopify 和研究机构用于训练与推理。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/skypilot-org/skypilot?style=social)
- [Volcano](https://github.com/volcano-sh/volcano) - 面向计算密集型工作负载的云原生批调度系统。CNCF 孵化项目，提供 gang scheduling、作业依赖管理和拓扑感知调度，服务于 AI/ML 与深度学习。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/volcano-sh/volcano?style=social)
- [Apache YuniKorn](https://github.com/apache/yunikorn-core) - 面向批处理、数据和 ML 工作负载的 Kubernetes 资源调度器。提供分层资源队列、多租户公平性和 gang scheduling，服务于大数据与机器学习应用。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/apache/yunikorn-core?style=social)
- [Kueue](https://github.com/kubernetes-sigs/kueue) - 面向批处理、HPC、AI/ML 等应用的 Kubernetes 原生作业排队系统。云原生的作业排队，支持资源规格可替换性、公平共享、队列组 (cohorts) 和抢占策略。与 Kubeflow、Ray 和 JobSet 集成。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/kubernetes-sigs/kueue?style=social)

#### 特征工程与数据准备

- [Featuretools](https://github.com/alteryx/featuretools) - 自动化特征工程的开源 Python 库。通过 Deep Feature Synthesis 和可复用的原语，将事务型与关系型数据集转换为机器学习用的特征矩阵。BSD-3-Clause 许可。 ![GitHub stars](https://img.shields.io/github/stars/alteryx/featuretools?style=social)
- [Kedro](https://github.com/kedro-org/kedro) - 面向生产就绪数据科学的工具箱。运用软件工程最佳实践，帮助你构建可复现、可维护、模块化的数据工程与数据科学流水线。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/kedro-org/kedro?style=social)
- [Feature-engine](https://github.com/feature-engine/feature_engine) - 提供多种 transformer 的 Python 库，用于为机器学习模型构建和筛选特征。兼容 scikit-learn，提供 fit() 和 transform() 方法，支持编码、缺失值填补、变量变换和特征选择。BSD-3-Clause 许可。 ![GitHub stars](https://img.shields.io/github/stars/feature-engine/feature_engine?style=social)
- [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) - GPU 加速的表格数据特征工程与预处理库。可操作 TB 级数据集以训练深度学习推荐系统。是 NVIDIA Merlin 框架的组件。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA-Merlin/NVTabular?style=social)
- [OpenMLDB](https://github.com/4paradigm/OpenMLDB) - 开源机器学习数据库，提供特征平台以保证训练与推理之间特征的一致性。面向在线 ML 应用的实时关系型数据特征计算系统。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/4paradigm/OpenMLDB?style=social)

#### 监控、评估与可观测性

- [Langfuse](https://github.com/langfuse/langfuse) - 排名第一的开源 LLM 可观测性平台。 ![GitHub stars](https://img.shields.io/github/stars/langfuse/langfuse?style=social)
- [Phoenix (Arize)](https://github.com/Arize-ai/phoenix) - AI 可观测性与评估平台。 ![GitHub stars](https://img.shields.io/github/stars/Arize-ai/phoenix?style=social)
- [Evidently](https://github.com/evidentlyai/evidently) - ML 与 LLM 监控框架。 ![GitHub stars](https://img.shields.io/github/stars/evidentlyai/evidently?style=social)
- [Opik (Comet)](https://github.com/comet-ml/opik) - 生产就绪的 LLM 评估平台。 ![GitHub stars](https://img.shields.io/github/stars/comet-ml/opik?style=social)
- [LiteLLM](https://github.com/BerriAI/litellm) - AI 网关，以 OpenAI 格式调用 100+ LLM API，提供统一的成本追踪、护栏、负载均衡和日志记录。 ![GitHub stars](https://img.shields.io/github/stars/BerriAI/litellm?style=social)
- [OpenLIT](https://github.com/openlit/openlit) - 基于 OpenTelemetry 原生的 LLM 可观测性平台，集成 GPU 监控、评估、提示词管理和护栏。 ![GitHub stars](https://img.shields.io/github/stars/openlit/openlit?style=social)
- [flameox](https://github.com/morluto/flameox) - 面向 agent 的运行时证据工具包，可协调 PyTorch Profiler 与 Nsight Systems 的采集、保留原生 trace，并对比 GPU kernel 与推理运行结果。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/morluto/flameox?style=social)
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) - 基于 OpenTelemetry 的 GenAI/LLM 应用开源可观测方案，支持 25+ 集成后端。 ![GitHub stars](https://img.shields.io/github/stars/traceloop/openllmetry?style=social)
- [Agenta](https://github.com/Agenta-AI/agenta) - 开源 LLMOps 平台，集提示词试验台、提示词管理、LLM 评估与可观测性于一体。 ![GitHub stars](https://img.shields.io/github/stars/Agenta-AI/agenta?style=social)
- [Latitude](https://github.com/latitude-dev/latitude-llm) - 开源的 agent 工程平台，提供提示词管理、评估与优化。具备提示词试验台、LLM-as-judge 评估，以及面向生产级 LLM 功能的 GEPA 提示词优化器。LGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/latitude-dev/latitude-llm?style=social)
- [Helicone](https://github.com/helicone/helicone) - 开源 LLM 可观测性平台，提供请求日志、缓存、提示词管理、速率限制和成本分析。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/helicone/helicone?style=social)
- [Giskard](https://github.com/Giskard-AI/giskard-oss) - 面向 LLM agent 的开源评估与测试库。提供红队测试、漏洞扫描、RAG 评估和安全测试，架构模块化。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Giskard-AI/giskard-oss?style=social)
- [Awesome Agentic Engineering](https://github.com/lindixu6-hash/awesome-agentic-engineering) - 零依赖的 AI agent 生产就绪度工具包，提供 GitHub Actions 评分与风险画像门禁、提示词注入测试夹具、机器可读结果和可复现的 LangGraph trace。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/lindixu6-hash/awesome-agentic-engineering?style=social)
- [Portkey Gateway](https://github.com/Portkey-AI/gateway) - 高速 AI 网关，通过统一 API 路由 200+ LLM。集成护栏、负载均衡、降级回退和成本追踪。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Portkey-AI/gateway?style=social)
- [Envoy AI Gateway](https://github.com/envoyproxy/ai-gateway) - 管理对构建在 Envoy Gateway 之上的生成式 AI 服务的统一访问。Kubernetes 原生的 AI 网关，用于 LLM 流量的路由、负载均衡与管理，具备企业级可靠性。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/envoyproxy/ai-gateway?style=social)
- [Unified AI System](https://github.com/happy520ai/unified-ai-system) - 可自托管的 Apache-2.0 AI 网关与 MCP 服务器：按 key 发放虚拟令牌并设定令牌预算、精确与词法近似的响应缓存（语义级匹配需要挂载 embedding 端点）、仅追加的审计链、完全不发起供应商调用的确定性本地提示词增强，以及反向 MCP 治理——将上游 MCP 服务器和 OpenAPI 3 操作暴露为白名单工具。 ![GitHub stars](https://img.shields.io/github/stars/happy520ai/unified-ai-system?style=social)
- [Pezzo](https://github.com/pezzolabs/pezzo) - 云原生 LLMOps 平台，提供提示词管理、版本管理和可观测性。具备协作式提示词编辑、A/B 测试和成本分析。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/pezzolabs/pezzo?style=social)
- [Microsoft PromptFlow](https://github.com/microsoft/promptflow) - 覆盖从原型到生产的 LLM AI 应用开发全套工具。包含提示词工程、评估和部署工具，并集成 VS Code。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/promptflow?style=social)
- [ChainForge](https://github.com/ianarawjo/ChainForge) - 用于提示词压力测试和 LLM 输出评估的可视化编程环境。具备基于节点的提示词链、多模型对比和假设检验。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/ianarawjo/ChainForge?style=social)
- [Future AGI](https://github.com/future-agi/future-agi) - 开源、可自托管的端到端 agent 工程与优化平台，统一整合 tracing、eval、仿真、数据集、网关和护栏。旨在用一条从原型到生产的反馈闭环，打造自我改进的 AI agent。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/future-agi/future-agi?style=social)
- [KubeStellar Console](https://github.com/kubestellar/console) - AI 驱动的多集群 Kubernetes 仪表盘，提供 GPU 工作负载监控、AI 流水线可观测性和 CNCF 生态集成。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/kubestellar/console?style=social)
- [OrcaReplay](https://github.com/Continuum-AI-Corp/OrcaReplay) - 在 harness 之下记录 coding agent 的运行过程，可在断网状态下离线回放，或从某个检查点分叉到另一个模型上运行。 ![GitHub stars](https://img.shields.io/github/stars/Continuum-AI-Corp/OrcaReplay?style=social)

#### 护栏与安全工具

- [PurpleLlama (Meta)](https://github.com/meta-llama/PurpleLlama) - 用于评估和提升 LLM 安全性的综合工具集。包含 Llama Guard 安全分类器、CyberSec Eval 基准和用于检测提示词注入的 Prompt Guard。BSD-3-Clause 许可。 ![GitHub stars](https://img.shields.io/github/stars/meta-llama/PurpleLlama?style=social)
- [Garak (NVIDIA)](https://github.com/NVIDIA/garak) - LLM 漏洞扫描器。探测模型是否存在幻觉、数据泄漏、提示词注入、错误信息、有毒内容和越狱问题。拥有基于插件的庞大架构，包含 100+ 漏洞探针。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/garak?style=social)
- [Promptfoo](https://github.com/promptfoo/promptfoo) - 开源的 LLM 评估与红队测试框架。支持对提示词、agent 和 RAG 进行自动化安全漏洞扫描、模型并排对比以及 CI/CD 集成。现已并入 OpenAI。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/promptfoo/promptfoo?style=social)
- [DeepTeam (Confident AI)](https://github.com/confident-ai/deepteam) - 面向 LLM 系统的红队框架，含 50+ 漏洞类型、20+ 对抗攻击和生产就绪的护栏。包含 OWASP、NIST 和 MITRE ATLAS 框架映射。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/confident-ai/deepteam?style=social)
- [SkillSpector (NVIDIA)](https://github.com/NVIDIA/SkillSpector) - 面向 AI agent skills 的安全扫描器，检测漏洞、恶意模式和安全风险。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/SkillSpector?style=social)
- [ADR (Uber)](https://github.com/uber/ADR) - 面向 AI agent 的企业级安全系统，提供遥测采集、基准评估和威胁检测。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/uber/ADR?style=social)

---

## 9. 评估、基准与数据集

> 用于模型评估的基准、评估框架、数据集及配套工具。

#### 基准测试套件

- [LiveBench](https://github.com/LiveBench/LiveBench) - 无污染的 LLM 基准，采用客观的标准答案评分。ICLR 2025  spotlight 论文，题目取自近期来源并频繁更新。测试数学、编程、推理、语言、指令遵循和数据分析能力。 ![GitHub stars](https://img.shields.io/github/stars/LiveBench/LiveBench?style=social)
- [lm-evaluation-harness (EleutherAI)](https://github.com/EleutherAI/lm-evaluation-harness) - 生成式模型评估的事实标准。 ![GitHub stars](https://img.shields.io/github/stars/EleutherAI/lm-evaluation-harness?style=social)
- [HELM (Stanford)](https://github.com/stanford-crfm/helm) - 语言模型的整体性评估 (Holistic Evaluation of Language Models)。 ![GitHub stars](https://img.shields.io/github/stars/stanford-crfm/helm?style=social)
- [SWE-bench](https://github.com/SWE-bench/SWE-bench) - 在来自 15+ Python 仓库的真实 GitHub issue 上评估 LLM。 ![GitHub stars](https://img.shields.io/github/stars/SWE-bench/SWE-bench?style=social)
- [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) - 真实世界的多步 agentic 基准。
- [OpenCompass](https://github.com/open-compass/opencompass) - 面向语言模型与多模态模型、覆盖大型基准套件的评估平台。 ![GitHub stars](https://img.shields.io/github/stars/open-compass/opencompass?style=social)
- [MLPerf Inference](https://github.com/mlcommons/inference) - 行业标准的 ML 推理基准，附带面向 AI 加速器的参考实现。 ![GitHub stars](https://img.shields.io/github/stars/mlcommons/inference?style=social)
- [MLPerf Training](https://github.com/mlcommons/training) - 来自 MLCommons 的行业标准 ML 训练基准。提供在图像分类、目标检测、NLP 和推荐等任务上大规模训练 AI 模型的参考实现。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/mlcommons/training?style=social)
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) - 面向大型多模态模型 (LMMs) 的开源评估工具包。支持 220+ LMM 和 80+ 基准，包括 MMMU、MathVista 和 ChartQA。是 OpenVLM Leaderboard 的支撑工具。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/open-compass/VLMEvalKit?style=social)
- [Vectara Hallucination Leaderboard](https://github.com/vectara/hallucination-leaderboard) - 对比各主流 LLM 在摘要短文档时产生幻觉表现的排行榜。系统性地评估各模型的事实一致性。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/vectara/hallucination-leaderboard?style=social)
- [SWE-rebench (Nebius)](https://huggingface.co/datasets/nebius/SWE-rebench) - 持续更新的基准，含 21000+ 真实 SWE 任务，用于评估 agentic LLM。已做去污染处理，采掘自 GitHub。
- [MLE-bench (OpenAI)](https://github.com/openai/mle-bench) - 用于衡量 AI agent 在机器学习工程方面表现的基准。在覆盖多样 ML 任务的 75 个 Kaggle 竞赛上评估 agent。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/openai/mle-bench?style=social)
- [PinchBench](https://github.com/pinchbench/skill) - 用于将 LLM 模型评估为 OpenClaw coding agent 的基准测试系统。由 kilo.ai 团队使用 Rust 构建。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/pinchbench/skill?style=social)
- [ImageBench V1](https://imagebench.ai/imagebench-v1) - 面向文生图模型的提示词遵循度排行榜。在 60+ 前沿模型 (Flux 2 Max、Nano Banana Pro、Seedream V4、Reve 2.1、Imagen 4 Ultra、Qwen Image 2.0 Pro) 上运行相同的提示词，并由 VLM 评审团打分。提供[公开方法论](https://imagebench.ai/imagebench-v1/methodology)、[模型正面对比页面](https://imagebench.ai/imagebench-v1/compare)，以及 Hugging Face 上完整的[提示词与评审数据集](https://huggingface.co/datasets/dh7/imagebench)（CC0）。

#### 评估框架

- [DeepEval](https://github.com/confident-ai/deepeval) - “LLM 的 Pytest”。 ![GitHub stars](https://img.shields.io/github/stars/confident-ai/deepeval?style=social)
- [iFixAi](https://github.com/ifixai-ai/iFixAi) - Agent 审计框架，对 AI agent 执行对抗性与运维层面的检查，输出带评分的 JSON 和 Markdown 报告，适用于 CLI、CI 和 agent 插件工作流。Apache-2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/ifixai-ai/iFixAi?style=social)
- [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) - 英国 AI Security Institute 出品的大语言模型评估框架。 ![GitHub stars](https://img.shields.io/github/stars/UKGovernmentBEIS/inspect_ai?style=social)
- [Lighteval](https://github.com/huggingface/lighteval) - 跨多种后端评估 LLM 的工具包，提供可复用的任务、指标和结果追踪。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/lighteval?style=social)
- [Hugging Face Evaluate](https://github.com/huggingface/evaluate) - 标准化评估指标。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/evaluate?style=social)
- [OpenAI Evals](https://github.com/openai/evals) - 用于评估 LLM 与 LLM 系统的框架，内置开源注册表，收录 100+ 社区贡献的基准。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/openai/evals?style=social)
- [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) - 统一的多模态评估工具包，支持文本、图像、视频和音频任务，支持 100+ 基准。 ![GitHub stars](https://img.shields.io/github/stars/EvolvingLMMs-Lab/lmms-eval?style=social)
- [BrowserGym](https://github.com/ServiceNow/BrowserGym) - 用于 web 任务自动化与 agent 评估的 Gym 环境。包含 MiniWoB、WebArena、WorkArena 等。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/ServiceNow/BrowserGym?style=social)
- [TruLens](https://github.com/truera/trulens) - 面向 LLM 实验和 AI agent 的评估与追踪。提供反馈函数来衡量质量、相关性和依据支撑度，并集成 LangChain 与 LlamaIndex。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/truera/trulens?style=social)
- [OpenEvals](https://github.com/langchain-ai/openevals) - 面向 LLM 与 agent 应用的开源评估库。由 LangChain 构建，预置了涵盖 RAG、agent 和结构化输出校验等常见场景的评估器。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/langchain-ai/openevals?style=social)
- [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) - RAG AutoML 工具，自动寻找最优的 RAG 流水线。以 AutoML 式自动化针对你自己的数据和用例评估并优化检索增强生成。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Marker-Inc-Korea/AutoRAG?style=social)
- [E2B Code Interpreter](https://github.com/e2b-dev/code-interpreter) - 用于在安全的隔离沙箱中运行 AI 生成代码的 Python 与 JS/TS SDK。是评估代码生成类 LLM 的关键基础设施，提供安全的执行环境。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/e2b-dev/code-interpreter?style=social)
- [SimpleEvals (OpenAI)](https://github.com/openai/simple-evals) - 用于评估语言模型的轻量级库，提供透明的准确率数字。包含 MMLU、GPQA、MATH、HumanEval、MGSM、DROP 和 SimpleQA 基准的参考实现。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/openai/simple-evals?style=social)
- [EvalScope (ModelScope)](https://github.com/modelscope/evalscope) - 精简且可定制的大模型 (LLM、VLM、AIGC) 评估与性能基准框架。一站式评估方案，内含 80+ 基准。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/modelscope/evalscope?style=social)
- [Harbor](https://github.com/harbor-framework/harbor) - 用于运行 agent 评估以及创建/使用 RL 环境的框架。可评估 Claude Code、OpenHands、Codex CLI 等任意 agent。构建并共享基准与环境。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/harbor-framework/harbor?style=social)
- [YYLO Benchmark](https://github.com/yylo-dev/yylo-benchmark) - 面向 agent 运行的纵向评估：在私有的全新仓库尝试工作区中执行任务提示词，并保留尝试证据、评估器来源信息和报告，以备后续重新评估。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/yylo-dev/yylo-benchmark?style=social)
- [JevOss](https://github.com/mertkayacs/jevoss) - 面向实现 Jev API 的决策模型服务器的测试套件，报告准确率与概率校准情况，并探测提示词注入、选项顺序敏感性、无关填充和重复调用一致性。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/mertkayacs/jevoss?style=social)
- [Solo R Benchmark Harness](https://github.com/O-word/solo-r-benchmark-harness) - 让 SoloRoleplayer 应用以无头模式对接本地模型服务器运行，并对角色扮演回复按性格锁定、视角跳跃、重复和硬规则遵循度打分。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/O-word/solo-r-benchmark-harness?style=social)
- [lane_lint](https://github.com/O-word/lane-lint) - 零依赖检查器，标记角色扮演文本中的视角跳跃（即替对方角色写台词、动作或感受）。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/O-word/lane-lint?style=social)

#### 高质量开放数据集与数据工具

- [Hugging Face Datasets](https://github.com/huggingface/datasets) - 最大的开放数据集仓库。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/datasets?style=social)
- [OrcaPromptVault](https://github.com/Continuum-AI-Corp/OrcaPromptVault) - 开放语料库，收录实际投递给 AI agent 的指令与工具 schema：来自 43 个产品的 119 份制品，其中 44 份为线上抓取记录并附有可复现命令，每个文件均标注为 captured 或 reported。AGPL-3.0。 ![GitHub stars](https://img.shields.io/github/stars/Continuum-AI-Corp/OrcaPromptVault?style=social)
- [FineWeb / FineWeb-2 (Hugging Face)](https://huggingface.co/datasets/HuggingFaceFW/fineweb) - 面向预训练的 15T+ token 精选网络数据集。
- [OSWorld](https://github.com/xlang-ai/OSWorld) - 多模态 agent 基准数据集。 ![GitHub stars](https://img.shields.io/github/stars/xlang-ai/OSWorld?style=social)

---

<a id="section-10"></a>

## 10. AI 安全、对齐与可解释性

> 用于对齐、可解释性、安全评估与对抗性测试的工具。

#### 安全评估框架

- [AgentOps](https://github.com/AgentOps-AI/agentops) - 用于 AI agent 监控、LLM 成本追踪、基准测试和评估的 Python SDK。集成 CrewAI、Agno、OpenAI Agents SDK、LangChain、Autogen、AG2 和 CamelAI。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/AgentOps-AI/agentops?style=social)
- [Bloom](https://github.com/safety-research/bloom) - 面向前沿 AI 模型自动化行为评估的开源 agentic 框架。生成有针对性的评估套件，探测 LLM 的特定行为（谄媚、自我保存、政治偏见等）并给出定量的诱发率。来自 Anthropic 的安全研究团队。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/safety-research/bloom?style=social)
- [AgentShield Benchmark](https://github.com/doronp/agentshield-benchmark) - Apache-2.0 许可的语料库，含 537 个测试用例和一个 TypeScript 运行器，用于对 AI agent 护栏产品做基准测试；由一家将其自家产品纳入评分范围的厂商维护。 ![GitHub stars](https://img.shields.io/github/stars/doronp/agentshield-benchmark?style=social)

#### 对齐与 RLHF 工具

- [Alignment Handbook](https://github.com/huggingface/alignment-handbook) - 全栈对齐的完整配方。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/alignment-handbook?style=social)

#### 可解释性与可说明性

- [interpret (Microsoft)](https://github.com/interpretml/interpret) - 拟合可解释模型，并用包括可解释 boosting 机器 (Explainable Boosting Machines) 和基于 SHAP 的解释在内的前沿可解释性技术解释黑盒机器学习。 ![GitHub stars](https://img.shields.io/github/stars/interpretml/interpret?style=social)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) - 机制可解释性的金标准。 ![GitHub stars](https://img.shields.io/github/stars/TransformerLensOrg/TransformerLens?style=social)
- [SAELens](https://github.com/jbloomAus/SAELens) - 用于发现可解释特征的稀疏自编码器工具集。 ![GitHub stars](https://img.shields.io/github/stars/jbloomAus/SAELens?style=social)
- [nnsight](https://github.com/ndif-team/nnsight) - 用于大规模检视、追踪和干预神经网络内部的库。 ![GitHub stars](https://img.shields.io/github/stars/ndif-team/nnsight?style=social)
- [Captum](https://github.com/pytorch/captum) - PyTorch 官方可解释性库。 ![GitHub stars](https://img.shields.io/github/stars/pytorch/captum?style=social)
- [EasyEdit](https://github.com/zjunlp/EasyEdit) - 易于使用的 LLM 知识编辑框架。可精确修改模型知识与行为，以纠正幻觉或过时信息。ACL 2024。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/zjunlp/EasyEdit?style=social)
- [AIX360](https://github.com/Trusted-AI/AIX360) - 全面的 AI 可解释性工具包，提供用于数据和机器学习模型的可解释性算法。包含 TED、BRCG 和 ProtoNN 等方法，满足多样化的解释需求。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Trusted-AI/AIX360?style=social)
- [Shapash](https://github.com/MAIF/shapash) - 面向透明 ML 模型的易用可解释性库。提供配以明确标签的精美可视化，人人都能看懂。可生成 Web 报告，并与 SHAP/LIME 集成。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/MAIF/shapash?style=social)

#### 公平性与偏差缓解

- [AI Fairness 360](https://github.com/Trusted-AI/AIF360) - 用于检测、理解和缓解数据集与 ML 模型中非预期算法偏差的综合工具包。 ![GitHub stars](https://img.shields.io/github/stars/Trusted-AI/AIF360?style=social)
- [Fairlearn](https://github.com/fairlearn/fairlearn) - 用于评估和提升机器学习模型公平性的 Python 包。提供差异评估指标以及不公平性缓解算法，并集成 scikit-learn。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/fairlearn/fairlearn?style=social)

#### 对抗与红队测试工具

- [PyRIT (Microsoft)](https://github.com/microsoft/PyRIT) - 面向生成式 AI 的 Python 风险识别工具 (Python Risk Identification Tool)。微软开源的自动化红队框架，支持多模态攻击、crescendo 策略以及 100+ 操作体验。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/PyRIT?style=social)
- [Heretic](https://github.com/p-e-w/heretic) - 开源系统，用于自动移除语言模型输出中的审查与鲁棒性抑制。 ![GitHub stars](https://img.shields.io/github/stars/p-e-w/heretic?style=social)
- [Agentic Security](https://github.com/msoedov/agentic_security) - Agentic LLM 漏洞扫描器与 AI 红队工具包，具备多步攻击模拟和自动化安全探测。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/msoedov/agentic_security?style=social)
- [NeMo Guardrails (NVIDIA)](https://github.com/NVIDIA-NeMo/Guardrails) - 面向基于 LLM 的对话系统的可编程护栏工具包。使用 Colang DSL 定义安全规则、对话流程和内容边界。集成 LangChain、LangGraph 和 LlamaIndex 以支持生产部署。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/NVIDIA-NeMo/Guardrails?style=social)
- [Guardrails AI](https://github.com/guardrails-ai/guardrails) - 用于构建可靠 AI 应用的输入/输出校验框架。通过可组合的校验器检测和缓解 PII、有毒内容、提示词注入等风险，并校验结构化输出。提供 Guardrails Hub，内含 50+ 预置校验器。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/guardrails-ai/guardrails?style=social)
- [Detoxify](https://github.com/unitaryai/detoxify) - 训练好的模型与代码，可在全部 3 个 Jigsaw Toxic Comment Challenges 上预测有毒评论。基于 PyTorch Lightning 和 Transformers 构建，用于检测毒性、强毒性、脏话、威胁、侮辱、身份攻击和性露骨内容。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/unitaryai/detoxify?style=social)
- [RedAmon](https://github.com/samugit83/redamon) - AI 驱动的 agentic 红队框架，在零人工干预下完成从信息收集、漏洞利用到后渗透的完整攻防安全操作。集成多种安全工具以进行全面的渗透测试。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/samugit83/redamon?style=social)
- [CAI](https://github.com/aliasrobotics/cai) - 面向网络安全的 AI 框架，用于半自动化和全自动化的攻防安全任务。专为网络安全场景打造，采用基于 agent 的架构进行漏洞评估与安全运营。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/aliasrobotics/cai?style=social)
- [AI-Infra-Guard (Tencent)](https://github.com/Tencent/AI-Infra-Guard) - 全栈 AI 红队平台，通过 OpenClaw 安全扫描、Agent 扫描、Skills 扫描、MCP 扫描、AI 基础设施扫描和 LLM 越狱评估来守护 AI 生态。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Tencent/AI-Infra-Guard?style=social)
- [PentestAgent (GH05TCREW)](https://github.com/GH05TCREW/pentestagent) - 面向黑盒安全测试的 AI agent 框架，支持漏洞赏金、红队和渗透测试工作流。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/GH05TCREW/pentestagent?style=social)
- [Superagent](https://github.com/superagent-ai/superagent) - 保护 AI 应用免受提示词注入、数据泄露和有害输出的影响。将安全能力直接嵌入你的应用，并向客户证明合规性。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/superagent-ai/superagent?style=social)
- [Anthropic Sandbox Runtime](https://github.com/anthropic-experimental/sandbox-runtime) - 面向 AI agent 的可移植沙箱执行环境，限制文件系统与网络边界，旨在降低工具使用和 LLM 自主性测试中的影响半径。 ![GitHub stars](https://img.shields.io/github/stars/anthropic-experimental/sandbox-runtime?style=social)
- [SchemaBrain](https://github.com/Arun-kc/schemabrain) - 位于 AI agent 与数据库之间的开源 schema 智能与安全层（类似 Cloudflare）。提供只读的 Model Context Protocol (MCP) 接口，具备 PII 脱敏、结构化查询恢复和防篡改的 SHA-256 审计日志。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Arun-kc/schemabrain?style=social)
- [Anthropic Cybersecurity Skills](https://github.com/mukul975/Anthropic-Cybersecurity-Skills) - 结构化的网络安全技能库，含 754 项技能，映射到包括 MITRE ATT&CK 和 NIST CSF 2.0 在内的五个框架，兼容 Claude Code、Cursor 和 agentskills.io。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/mukul975/Anthropic-Cybersecurity-Skills?style=social)
- [Strix](https://github.com/usestrix/strix) - 自主的 AI 渗透测试 agent，能发现漏洞并用概念验证 (PoC) exploit 加以验证。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/usestrix/strix?style=social)


#### 负责任的 AI 开发

- [Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox) - 一套工具，提供模型与数据探查、评估界面和相关库，帮助理解 AI 系统。使开发者能以更负责任、更有数据支撑的方式开发和监控 AI。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/responsible-ai-toolbox?style=social)
- [Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit) - 面向自主 agent 的安全与合规工具包，通过策略强制执行、零信任身份和沙箱化执行来降低运营风险。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/agent-governance-toolkit?style=social)
- [Orca AI Incident Archive](https://github.com/Continuum-AI-Corp/Orca-AI-Incident-Archive) - 真实世界 AI agent 安全事件的开放数据库，每条来源记录均标注了已确认的危害、AI 参与情况与事件类型，并提供 JSON/CSV 导出。CC BY 4.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Continuum-AI-Corp/Orca-AI-Incident-Archive?style=social)

#### 隐私保护 AI

- [Presidio (Microsoft)](https://github.com/microsoft/presidio) - 用于在文本和图像中检测、涂黑、掩码和匿名化敏感个人身份信息 (PII) 的 SDK。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/presidio?style=social)

---
## 11. 专业领域

#### 气象与气候 AI

- [WeatherNext](https://github.com/google-deepmind/weathernext) - 来自 Google DeepMind 的全球天气与热带气旋预报框架，涵盖 WeatherNext 2、GraphCast 和 GenCast 模型。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/google-deepmind/weathernext?style=social)

#### 科学 AI 与物理机器学习

- [NVIDIA Modulus](https://github.com/NVIDIA/modulus) - 面向物理信息机器学习（Physics-ML）的开源深度学习框架。使用最先进的 SciML 方法，为 AI4science 与工程应用构建、训练和微调模型。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/NVIDIA/modulus?style=social)
- [TorchGeo](https://github.com/microsoft/torchgeo) - 面向地理空间数据的 PyTorch 领域库。提供多光谱卫星影像与遥感所需的数据集、采样器、变换和预训练模型。首个提供 Sentinel-2 波段预训练模型的库。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/microsoft/torchgeo?style=social)
- [Astropy](https://github.com/astropy/astropy) - Python 天文学与天体物理学核心库。为专业天文学提供天体坐标、FITS I/O、宇宙学计算和数据分析等全面工具。BSD-3-Clause 许可。![GitHub stars](https://img.shields.io/github/stars/astropy/astropy?style=social)

#### 科学 AI 与药物发现

- [Boltz](https://github.com/jwohlwend/boltz) - 开源生物分子相互作用预测模型。Boltz-1 是首个接近 AlphaFold3 精度的完全开源模型；Boltz-2 增加了用于药物发现的结合亲和力预测。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/jwohlwend/boltz?style=social)
- [Protenix](https://github.com/bytedance/Protenix) - 来自字节跳动的高精度开源生物分子结构预测模型。首个在多种基准测试上超越 AlphaFold3 的完全开源模型，采用 Apache 2.0 许可，学术与商业用途均可。![GitHub stars](https://img.shields.io/github/stars/bytedance/Protenix?style=social)
- [DeepChem](https://github.com/deepchem/deepchem) - 让药物发现、量子化学、材料科学和生物学的深度学习人人可用。提供高质量开源工具链，包含 50 多个模型和详尽教程。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/deepchem/deepchem?style=social)

#### 概率编程与贝叶斯机器学习

- [PyMC](https://github.com/pymc-devs/pymc) - 现代、全面的 Python 概率编程框架。提供基于高级 MCMC 采样、变分推断的贝叶斯建模能力，并与 ArviZ 无缝集成以实现可视化。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/pymc-devs/pymc?style=social)
- [ArviZ](https://github.com/arviz-devs/arviz) - 用 Python 对贝叶斯模型进行探索性分析。为概率模型提供全面的可视化与诊断，支持 PyMC、Pyro、Stan 及其他 PPL。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/arviz-devs/arviz?style=social)
- [Stanza](https://github.com/stanfordnlp/stanza) - Stanford NLP 的 Python 库，支持 100 多种人类语言。提供分词、NER、句法分析和情感分析等最先进神经网络流水线，并附带预训练模型。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/stanfordnlp/stanza?style=social)

#### 医学影像与医疗健康 AI

- [MONAI](https://github.com/Project-MONAI/MONAI) - 面向 AI 的医疗开放网络（Medical Open Network for AI）。面向医疗影像的端到端框架，提供最先进、可用于生产环境的训练工作流。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/Project-MONAI/MONAI?style=social)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) - 自配置的医学图像分割深度学习方法。无需手动调参即可自动适配任意数据集。已被广泛采用为生物医学分割挑战的标准基线。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/MIC-DKFZ/nnUNet?style=social)
- [OpenMed](https://github.com/maziyarpanahi/openmed) - 本地优先的医疗 AI 框架与应用，支持临床文本去标识化、实体抽取和端侧专用模型服务。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/maziyarpanahi/openmed?style=social)

#### 创意写作与叙事 AI

- [Alex](https://github.com/Alex663028/Alex-Novel-Platform) - 基于 Tauri、FastAPI、SQLite 并内置情节生成工作流的开源桌面小说创作引擎与 AI 故事脚手架。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/Alex663028/Alex-Novel-Platform?style=social)
- [AI Novel Writer](https://github.com/EthanYoQ/AI-Novel-Writer) - 本地优先的长篇小说桌面工作台，组织故事前提、人物、世界观设定、章节大纲、起草、审校与修订，支持 Ollama，并提供 Windows/macOS 版本。GPL-3.0 许可。![GitHub stars](https://img.shields.io/github/stars/EthanYoQ/AI-Novel-Writer?style=social)
- [Ozan](https://github.com/arbazsiddiqui/Ozan) - 可复现的训练配方与开放权重，基于 QLoRA 和反套话（anti-slop）DPO 对 Mistral-Nemo-12B 进行创意写作微调，提供 GGUF 量化版本，并在 EQ-Bench Creative Writing v3 测试的同尺寸 12B 模型中取得最低的 slop 分数。Apache-2.0 许可。![GitHub stars](https://img.shields.io/github/stars/arbazsiddiqui/Ozan?style=social)

#### HR 与招聘 AI

- [Aural](https://github.com/1146345502/aural-oss) - 可自托管的 AI 面试平台，支持语音、聊天和视频面试，具备自适应追问、自动评分和详细反馈报告。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/1146345502/aural-oss?style=social)

#### 游戏 AI 与仿真

- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) - 使用深度强化学习在游戏和仿真中训练智能体的工具包。支持 NPC 行为控制、自动化测试和游戏设计评估。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/Unity-Technologies/ml-agents?style=social)
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) - 面向 Stable Baselines3 强化学习智能体的训练框架，提供超参数优化、预训练智能体和大量基准环境。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/DLR-RM/rl-baselines3-zoo?style=social)
- [skrl](https://github.com/Toni-SM/skrl) - 模块化强化学习库，基于 PyTorch、JAX 和 NVIDIA Warp 实现，支持 Gymnasium、NVIDIA Isaac Lab、MuJoCo Playground 等环境。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/Toni-SM/skrl?style=social)
- [Finetrainers](https://github.com/huggingface/finetrainers) - 来自 Hugging Face 的可扩展、内存优化的扩散模型训练工具。支持视频和图像生成模型的 LoRA 与全量微调。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/huggingface/finetrainers?style=social)
- [OpenSpiel](https://github.com/google-deepmind/open_spiel) - 来自 Google DeepMind 的环境与算法集合，用于通用强化学习以及游戏中搜索/规划的研究。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/google-deepmind/open_spiel?style=social)

#### 金融与量化 AI

- [OpenBB](https://github.com/OpenBB-finance/OpenBB) - 面向分析师、量化研究员和 AI 智能体的金融数据平台。开源的投资研究基础设施，集成大量数据源。AGPL-3.0 许可。![GitHub stars](https://img.shields.io/github/stars/OpenBB-finance/OpenBB?style=social)
- [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) - 开源金融大语言模型。以数据为中心的训练管线和多个模型版本推动金融 AI 平民化，覆盖交易、分析和智能投顾。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/AI4Finance-Foundation/FinGPT?style=social)
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL) - 面向量化交易的金融强化学习框架。提供用于股票交易、资产配置和市场执行的深度 RL 库，内置环境和基准。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/AI4Finance-Foundation/FinRL?style=social)
- [Qlib](https://github.com/microsoft/qlib) - 微软推出的 AI 量化投资平台。支持多种机器学习建模范式，包括监督学习、市场动态建模和强化学习。现已配备 RD-Agent 以实现自动化研发流程。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/microsoft/qlib?style=social)
- [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) - 使用 LLM 进行金融分析的开源 AI 智能体平台。多智能体系统，包含交易、分析和研究等专用智能体。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/AI4Finance-Foundation/FinRobot?style=social)
- [Daily Stock Analysis](https://github.com/ZhuLinsen/daily_stock_analysis) - 由 LLM 驱动的多市场股票分析系统，提供自动化决策看板报告和实时市场洞察。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/ZhuLinsen/daily_stock_analysis?style=social)
- [stock-analysis](https://github.com/AdvancingTitans/stock-analysis) - 面向 A 股、港股和美股股票、基金及投资组合的证据优先市场研究 CLI 与 Agent Skill，提供可审计的 JSON 证据包并明确标注数据质量缺口。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/AdvancingTitans/stock-analysis?style=social)
- [Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) - 开源个人交易智能体与研究自动驾驶平台，具备多智能体集群、量化回测和券商连接器。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/HKUDS/Vibe-Trading?style=social)
- [AI Hedge Fund](https://github.com/virattt/ai-hedge-fund) - 模拟 AI 驱动对冲基金的多智能体系统，利用各具特色的投资者智能体人格做出交易决策。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/virattt/ai-hedge-fund?style=social)
- [etf-pattern-match-pybind11](https://github.com/redamancy231-create/etf-pattern-match-pybind11) - 高性能 ETF 形态匹配，使用 C++20/pybind11 加速，相比纯 Python 实现 DTW 提速 43 倍、形态匹配提速 58 倍。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/redamancy231-create/etf-pattern-match-pybind11?style=social)

#### 计算机视觉

- [OpenCV](https://github.com/opencv/opencv) - 全球使用最广泛的计算机视觉库。![GitHub stars](https://img.shields.io/github/stars/opencv/opencv?style=social)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 最先进的实时目标检测。![GitHub stars](https://img.shields.io/github/stars/ultralytics/ultralytics?style=social)
- [Detectron2](https://github.com/facebookresearch/detectron2) - 高性能目标检测库。![GitHub stars](https://img.shields.io/github/stars/facebookresearch/detectron2?style=social)
- [CVAT](https://github.com/cvat-ai/cvat) - 业界领先的计算机视觉数据标注平台。交互式视频与图像标注工具，被数万个团队用于各种规模的机器学习。![GitHub stars](https://img.shields.io/github/stars/cvat-ai/cvat?style=social)
- [SAM 2](https://github.com/facebookresearch/sam2) - 可提示的图像与视频分割模型，已发布检查点和训练代码。![GitHub stars](https://img.shields.io/github/stars/facebookresearch/sam2?style=social)
- [AI Segmentation by TerraLab](https://github.com/TerraLabAI/QGIS_AI-Segmentation) - QGIS 插件，可在卫星和无人机影像中通过点选方式使用 SAM 分割建筑物、树木及任意物体，并提供免费的纯 CPU 本地模式。![GitHub stars](https://img.shields.io/github/stars/TerraLabAI/QGIS_AI-Segmentation?style=social)
- [Kornia](https://github.com/kornia/kornia) - 可微分计算机视觉库。![GitHub stars](https://img.shields.io/github/stars/kornia/kornia?style=social)
- [Roboflow Supervision](https://github.com/roboflow/supervision) - 面向 Python 检测、跟踪和分割流水线的可复用计算机视觉工具集。![GitHub stars](https://img.shields.io/github/stars/roboflow/supervision?style=social)
- [torchaudio](https://github.com/pytorch/audio) - PyTorch 音频处理库。为音频 I/O、变换和深度学习提供全面工具包，支持语音识别、TTS 和音频分类。BSD-2-Clause 许可。![GitHub stars](https://img.shields.io/github/stars/pytorch/audio?style=social)
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) - 跨平台多模态流水线。![GitHub stars](https://img.shields.io/github/stars/google-ai-edge/mediapipe?style=social)
- [OpenEyes](https://github.com/mandarwagh9/openeyes) - 硬件无关的机器人视觉框架，内置世界模型，可在边缘设备上实现预测式智能。![GitHub stars](https://img.shields.io/github/stars/mandarwagh9/openeyes?style=social)
- [xdfdet](https://github.com/mertkayacs/xdfdet) - Deepfake 视频检测，包含八个在 FaceForensics++ 上训练的 EfficientNet-B4 模型，提供 Grad-CAM 热力图并测量八个面部区域的注意力分布。代码采用 MIT 许可，模型权重采用 CC BY-NC 4.0。![GitHub stars](https://img.shields.io/github/stars/mertkayacs/xdfdet?style=social)

#### 3D 视觉与点云处理

- [Open3D](https://github.com/isl-org/Open3D) - 现代 3D 数据处理库，提供 Python 和 C++ API。核心功能包括 3D 数据结构、处理算法、场景重建、表面配准、3D 可视化和 GPU 加速。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/isl-org/Open3D?style=social)
- [Point Cloud Library (PCL)](https://github.com/PointCloudLibrary/pcl) - 独立的大型开源项目，用于 2D/3D 图像与点云处理。提供滤波、特征估计、表面重建、配准、模型拟合和分割等全面算法。BSD 许可。![GitHub stars](https://img.shields.io/github/stars/PointCloudLibrary/pcl?style=social)
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) - FAIR 出品的 3D 数据深度学习可复用组件库。提供高效的 3D 算子、可微渲染和网格处理工具，并与 PyTorch 集成。BSD 许可。![GitHub stars](https://img.shields.io/github/stars/facebookresearch/pytorch3d?style=social)
- [RTAB-Map](https://github.com/introlab/rtabmap) - 面向 RGB-D、双目和 LiDAR SLAM 的实时基于外观建图库。基于图的 SLAM 方法，支持增量式基于外观的回环检测，适用于大规模和长时运行。BSD 许可。![GitHub stars](https://img.shields.io/github/stars/introlab/rtabmap?style=social)
- [MoveIt 2](https://github.com/ros-planning/moveit2) - ROS 2 的开源机器人操作框架。为机械臂提供运动规划、操作、3D 感知、运动学、控制和导航。BSD-3-Clause 许可。![GitHub stars](https://img.shields.io/github/stars/ros-planning/moveit2?style=social)
- [LingBot-Map](https://github.com/Robbyant/lingbot-map) - 用于流式 3D 场景变换与重建的前馈式 3D 基础模型。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/Robbyant/lingbot-map?style=social)

#### 强化学习与机器人

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - 可用于生产环境的强化学习算法。![GitHub stars](https://img.shields.io/github/stars/DLR-RM/stable-baselines3?style=social)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) - GPU 加速的机器人学习框架。![GitHub stars](https://img.shields.io/github/stars/isaac-sim/IsaacLab?style=social)
- [MuJoCo](https://github.com/google-deepmind/mujoco) - 面向机器人、生物力学和机器学习研究的通用物理仿真器。高保真接触动力学，提供原生 Python 和 C++ 绑定。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/google-deepmind/mujoco?style=social)
- [Gymnasium (ex-OpenAI Gym)](https://github.com/Farama-Foundation/Gymnasium) - 标准的强化学习环境 API。![GitHub stars](https://img.shields.io/github/stars/Farama-Foundation/Gymnasium?style=social)
- [OpenEnv](https://github.com/huggingface/OpenEnv) - 面向智能体强化学习训练的端到端沙箱执行环境框架，构建于兼容 Gymnasium 的 API 之上。![GitHub stars](https://img.shields.io/github/stars/huggingface/OpenEnv?style=social)
- [JaxMARL](https://github.com/FLAIROx/JaxMARL) - 多智能体强化学习库，提供 JAX 加速的环境和基线。![GitHub stars](https://img.shields.io/github/stars/FLAIROx/JaxMARL?style=social)
- [MicroDuck RL](https://github.com/pollen-robotics/microduck_rl) - 面向 MicroDuck 机器人平台的强化学习环境和训练工具。![GitHub stars](https://img.shields.io/github/stars/pollen-robotics/microduck_rl?style=social)

#### 时间序列与科学 AI

- [Time Series Library (TSLib)](https://github.com/thuml/Time-Series-Library) - 面向时间序列模型的综合性基准库。![GitHub stars](https://img.shields.io/github/stars/thuml/Time-Series-Library?style=social)
- [Chronos (Amazon)](https://github.com/amazon-science/chronos-forecasting) - 用于时间序列预测的预训练基础模型。![GitHub stars](https://img.shields.io/github/stars/amazon-science/chronos-forecasting?style=social)
- [GluonTS (AWS Labs)](https://github.com/awslabs/gluonts) - 基于深度学习的概率时间序列建模。为 Amazon SageMaker 预测提供支撑，基于 PyTorch 和 MXNet 后端。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/awslabs/gluonts?style=social)
- [AutoTS](https://github.com/winedarksea/AutoTS) - 自动化时间序列预测，提供广泛的模型选择、集成、异常检测和节假日效应处理。面向生产部署设计，几乎无需配置。![GitHub stars](https://img.shields.io/github/stars/winedarksea/AutoTS?style=social)
- [TimesFM (Google Research)](https://github.com/google-research/timesfm) - Google Research 开发的时间序列预测预训练仅解码器基础模型，支持最长 16k 上下文长度、协变量支持以及 Flax / PyTorch 后端。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/google-research/timesfm?style=social)

#### 边缘 / 端侧 AI

- [ExecuTorch](https://github.com/pytorch/executorch) - PyTorch 运行时与工具链，用于在移动、嵌入式和边缘设备上部署 AI 模型。![GitHub stars](https://img.shields.io/github/stars/pytorch/executorch?style=social)
- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel 的边缘部署工具包。![GitHub stars](https://img.shields.io/github/stars/openvinotoolkit/openvino?style=social)
- [Apache TVM](https://github.com/apache/tvm) - 开放式机器学习编译器框架。通用部署方案将模型转换为最小可部署模块，可嵌入并运行于从数据中心到边缘设备的各种环境。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/apache/tvm?style=social)
- [NCNN](https://github.com/Tencent/ncnn) - 针对移动平台优化的神经网络推理框架。无第三方依赖、跨平台，在移动 CPU 上比所有已知开源框架都更快。支撑腾讯的 QQ、微信和 Pitu 等应用。BSD-3-Clause 许可。![GitHub stars](https://img.shields.io/github/stars/Tencent/ncnn?style=social)
- [MNN](https://github.com/alibaba/MNN) - 由阿里巴巴实战检验的极速轻量推理引擎。支持推理和训练，具备业界领先的端侧性能。通过 MNN-LLM 运行时支撑高性能 LLM 和边缘 AI。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/alibaba/MNN?style=social)
- [RuView](https://github.com/ruvnet/RuView) - 利用普通 WiFi 信号和神经网络实现实时空间感知、生命体征监测和人体姿态估计。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/ruvnet/RuView?style=social)

#### 法律 AI 与合同分析

- [OpenContracts](https://github.com/Open-Source-Legal/OpenContracts) - 面向法律 AI 的自托管文档标注平台。提供语义搜索、合同分析、版本控制和 MCP 集成，用于构建法律知识库。AGPL-3.0 许可。![GitHub stars](https://img.shields.io/github/stars/Open-Source-Legal/OpenContracts?style=social)
- [Harvey LAB](https://github.com/harveyai/harvey-labs) - 用于评估 AI 智能体在 24 多个业务领域复杂法律工作上表现的基准数据集与执行框架。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/harveyai/harvey-labs?style=social)
- [Korean Law MCP](https://github.com/chrisryugj/korean-law-mcp) - 面向韩国官方法律数据库（法制处开放 API）的 MCP 服务器与 CLI，涵盖法令、判例、行政规则、地方条例和条约，并可对 LLM 输出中的法律引用进行存在性和内容校验。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/chrisryugj/korean-law-mcp?style=social)

#### 自动驾驶与机器人仿真器

- [CARLA](https://github.com/carla-simulator/carla) - 开源自动驾驶研究仿真器。以高保真度模拟城市环境，包含真实的物理、传感器和交通场景。广泛用于训练和验证自动驾驶算法。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/carla-simulator/carla?style=social)
- [Webots](https://github.com/cyberbotics/webots) - 开源多平台机器人仿真器，为机器人、车辆和机械系统的建模、编程与仿真提供完整开发环境。应用于教育、科研和工业。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/cyberbotics/webots?style=social)
- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) - 面向具身智能研究的高性能物理 3D 仿真器。支持室内/室外空间的 3D 扫描、CAD 模型和可配置传感器。支撑 Meta 的具身智能研究。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/facebookresearch/habitat-sim?style=social)
- [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world) - 面向具身智能和机器人研究的通用仿真平台，提供灵活的场景、传感器和智能体工作流。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/Genesis-Embodied-AI/genesis-world?style=social)
- [OpenPilot](https://github.com/commaai/openpilot) - 面向机器人的操作系统。目前已为 300 多款支持的车型升级驾驶辅助系统。端到端自动驾驶技术栈，软硬件均开源。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/commaai/openpilot?style=social)
- [Autoware](https://github.com/autowarefoundation/autoware) - 全球领先的开源自动驾驶软件项目。从定位、目标检测到路径规划和控制的完整技术栈。全球 50 多家公司在使用。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/autowarefoundation/autoware?style=social)

---

<a id="section-12"></a>

## 12. 用户界面与自托管平台

#### 本地 AI 聊天界面与个人助手

- [OpenClaw](https://github.com/openclaw/openclaw) - 本地优先的个人 AI 助手，具备多渠道集成和完整的智能体任务执行能力。![GitHub stars](https://img.shields.io/github/stars/openclaw/openclaw?style=social)
- [AstrBot](https://github.com/AstrBotDevs/AstrBot) - 多渠道个人 AI 智能体助手与集成框架，连接消息平台、LLM 和插件。AGPL-3.0 许可。![GitHub stars](https://img.shields.io/github/stars/AstrBotDevs/AstrBot?style=social)
- [Open WebUI](https://github.com/open-webui/open-webui) - 最受欢迎的自托管 ChatGPT 风格界面。![GitHub stars](https://img.shields.io/github/stars/open-webui/open-webui?style=social)
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui) - 用于运行本地 LLM 的 Web 界面，支持多种后端、扩展和模型格式。![GitHub stars](https://img.shields.io/github/stars/oobabooga/text-generation-webui?style=social)
- [LibreChat](https://github.com/danny-avila/LibreChat) - 功能丰富的多 LLM 界面。![GitHub stars](https://img.shields.io/github/stars/danny-avila/LibreChat?style=social)
- [HuggingChat (self-hosted)](https://github.com/huggingface/chat-ui) - HuggingChat 的官方开源代码库。![GitHub stars](https://img.shields.io/github/stars/huggingface/chat-ui?style=social)
- [Khoj](https://github.com/khoj-ai/khoj) - 可自托管的个人 AI 助手，可对本地和网络数据进行搜索、对话、自动化和工作流处理。![GitHub stars](https://img.shields.io/github/stars/khoj-ai/khoj?style=social)
- [Newelle](https://github.com/qwersyk/Newelle) - GNOME/Linux 桌面虚拟助手，内置文件编辑器、全局快捷键和配置档案管理器。![GitHub stars](https://img.shields.io/github/stars/qwersyk/Newelle?style=social)
- [NextChat](https://github.com/ChatGPTNextWeb/NextChat) - 轻量快速的 AI 助手，支持 Web、iOS、macOS、Android、Linux 和 Windows。一键部署，支持多模型。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/ChatGPTNextWeb/NextChat?style=social)
- [big-AGI](https://github.com/enricoros/big-AGI) - 面向高级用户的 AI 套件，支持多模型 "Beam" 对话、AI 人格、语音、文生图、代码执行和 PDF 导入。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/enricoros/big-AGI?style=social)
- [Morphic](https://github.com/miurla/morphic) - 带生成式界面的 AI 搜索引擎。支持多个 AI 提供商（OpenAI、Anthropic、Google、Ollama）和搜索提供商（Tavily、SearXNG、Brave）。具备智能搜索模式、小组件以及图片/视频搜索。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/miurla/morphic?style=social)
- [Leon](https://github.com/leon-ai/leon) - 你的开源个人助手。围绕工具、上下文、记忆和智能体执行构建。可自托管、注重隐私、可扩展。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/leon-ai/leon?style=social)
- [Willow](https://github.com/HeyWillow/willow) - 开源、本地、可自托管的语音助手替代方案，可对标 Amazon Echo/Google Home，并支持硬件。Apache-2.0 许可。![GitHub stars](https://img.shields.io/github/stars/HeyWillow/willow?style=social)
- [CoPaw](https://github.com/agentscope-ai/CoPaw) - 你的个人 AI 助手；易于安装，可部署在自己的机器或云端；支持多个聊天应用，能力易于扩展。Apache-2.0 许可。![GitHub stars](https://img.shields.io/github/stars/agentscope-ai/CoPaw?style=social)
- [Smart2Brain](https://github.com/your-papa/obsidian-Smart2Brain) - 注重隐私的 Obsidian 插件，提供 AI 驱动的第二大脑功能。可使用本地或远程 LLM（包括 Ollama 和 OpenAI）与笔记对话。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/your-papa/obsidian-Smart2Brain?style=social)
- [Casibase](https://github.com/casibase/casibase) - 开源企业级 AI 知识库与智能体管理平台。支持多种 LLM 提供商、RAG 和团队协作。Apache-2.0 许可。![GitHub stars](https://img.shields.io/github/stars/casibase/casibase?style=social)
- [BionicGPT](https://github.com/bionic-gpt/bionic-gpt) - 面向团队的企业内网 ChatGPT 替代方案，提供助手、RAG、访问控制、审计和企业部署功能。![GitHub stars](https://img.shields.io/github/stars/bionic-gpt/bionic-gpt?style=social)
- [OpenHuman](https://github.com/tinyhumansai/openhuman) - 本地优先的个人助手与记忆框架，具备 Obsidian 集成、模型路由和集成自动同步。GPL-3.0 许可。![GitHub stars](https://img.shields.io/github/stars/tinyhumansai/openhuman?style=social)
- [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber) - 通过免手语音交互与任意 LLM 对话，支持语音打断和本地跨平台运行的 Live2D 说话头像。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/Open-LLM-VTuber/Open-LLM-VTuber?style=social)
- [AIRI](https://github.com/moeru-ai/airi) - 可自托管的电子伴侣与 VTuber 平台，具备 Live2D 角色渲染、实时语音聊天和游戏集成能力。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/moeru-ai/airi?style=social)
- [Next AI Draw.io](https://github.com/DayuanJiang/next-ai-draw-io) - Next.js Web 应用，将 AI 能力与 draw.io 图表集成，可通过自然语言创建和编辑图表。Apache-2.0 许可。![GitHub stars](https://img.shields.io/github/stars/DayuanJiang/next-ai-draw-io?style=social)
- [lucinate](https://github.com/lucinate-ai/lucinate) - 终端原生 AI 聊天客户端，支持 OpenClaw、Hermes、Ollama 和任意兼容 OpenAI 的后端——在 TUI 中提供例程、多智能体切换和本地智能体技能。Apache-2.0 许可。![GitHub stars](https://img.shields.io/github/stars/lucinate-ai/lucinate?style=social)
- [nanobot](https://github.com/HKUDS/nanobot) - 轻量级个人 AI 智能体，内置 WebUI、模型路由、MCP 和多渠道聊天集成。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/HKUDS/nanobot?style=social)
- [LifeOS](https://github.com/danielmiessler/LifeOS) - 通用个人 AI 框架与助手框架，具备持久记忆、自定义技能和目标跟踪。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/danielmiessler/LifeOS?style=social)
- [ThoughtDAG](https://github.com/chenxiachan/thoughtdag) - 本地优先的可视化 LLM 工作台，通过图中的边决定发送给模型的上下文，支持分支、合并、文档提取以及 Ollama 或兼容 OpenAI 的端点。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/chenxiachan/thoughtdag?style=social)
- [Thursday](https://github.com/cgoinglove/thursday) - 通过 `npx thursday-agent` 启动、运行在用户电脑上的语音助手：一场 OpenAI GPT-Live 通话持续进行，同时后台机器人基于 OpenAI、Anthropic、Google 或 xAI 模型使用 shell、浏览器、Agent Skills 和 MCP 服务器，通话、记忆和文件保存在 `~/.thursday`。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/cgoinglove/thursday?style=social)
- [Arelis](https://github.com/sAnct1x/Arelis) - Windows 桌面研究助手，通过 Ollama 运行本地模型、搜索网络、驱动自己的浏览器，并在写入文件或发送消息前请求批准。AGPL-3.0 许可。![GitHub stars](https://img.shields.io/github/stars/sAnct1x/Arelis?style=social)

#### 完整的自托管 AI 平台

- [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) - 一体化的 RAG + 智能体平台。![GitHub stars](https://img.shields.io/github/stars/Mintplex-Labs/anything-llm?style=social)
- [Flowise](https://github.com/FlowiseAI/Flowise) - 拖拽式 LLM 应用构建器。![GitHub stars](https://img.shields.io/github/stars/FlowiseAI/Flowise?style=social)
- [LocalAI](https://github.com/mudler/LocalAI) - 开源 AI 引擎，可在任意硬件上运行 LLM、视觉、语音、图像和视频模型。自托管、兼容 OpenAI 的 API。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/mudler/LocalAI?style=social)
- [Onyx](https://github.com/onyx-dot-app/onyx) - 功能完备的 AI 平台，涵盖 Chat、RAG、Agents 和 Actions。支持 40 多种文档连接器和所有 LLM。MIT 许可（社区版）。![GitHub stars](https://img.shields.io/github/stars/onyx-dot-app/onyx?style=social)
- [biniou](https://github.com/Woolverine94/biniou) - 自托管 WebUI，支持 30 多种生成式 AI 模型。无需专用 GPU（最低 8GB 内存）即可在自己的电脑上用 AI 生成多媒体内容。部署后可离线运行。GPL-3.0 许可。![GitHub stars](https://img.shields.io/github/stars/Woolverine94/biniou?style=social)
- [Plane](https://github.com/makeplane/plane) - 开源的 Jira、Linear、Monday 和 ClickUp 替代方案。AI 驱动的项目管理平台，具备智能任务分派、迭代规划和自动化工作流。AGPL-3.0 许可。![GitHub stars](https://img.shields.io/github/stars/makeplane/plane?style=social)
- [Buzz](https://github.com/block/buzz) - 可自托管的工作区与 Nostr 中继实现，人类团队成员和 AI 智能体在共享频道、画布和工作流中协作。Apache-2.0 许可。![GitHub stars](https://img.shields.io/github/stars/block/buzz?style=social)
- [Cloudflare OS](https://github.com/cloudflare/cloudflare-os) - 开源智能体工作区，用于在沙箱化的 Cloudflare Workers 中构建和运行 AI 生成的应用、文档和智能体，具备基于能力的集成和可审查的操作。Apache-2.0 许可。![GitHub stars](https://img.shields.io/github/stars/cloudflare/cloudflare-os?style=social)
- [Macro](https://github.com/macro-inc/macro) - 统一的团队工作区，将邮件、消息、文档、任务、CRM 和 AI 智能体结合在一起，并共享记忆。AGPL-3.0 许可。![GitHub stars](https://img.shields.io/github/stars/macro-inc/macro?style=social)
- [ODS](https://github.com/Osmantic/ODS) - 采用 Apache-2.0 许可的自托管本地 AI 技术栈，涵盖推理、聊天、语音、智能体、工作流和 RAG。![GitHub stars](https://img.shields.io/github/stars/Osmantic/ODS?style=social)
- [Self-Hosted AI Stack](https://github.com/hwdsl2/self-hosted-ai-stack) - 采用 MIT 许可的 Docker Compose 私有 AI 技术栈，包含 Ollama、LiteLLM 网关、聊天/RAG、语音和 MCP 工具。![GitHub stars](https://img.shields.io/github/stars/hwdsl2/self-hosted-ai-stack?style=social)
- [Octop (Tencent Cloud)](https://github.com/TencentCloud/Octop) - 自托管的多用户、多智能体 AI 助手平台，具备长期记忆、MCP 工具集成和沙箱化执行。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/TencentCloud/Octop?style=social)
- [ENZO](https://github.com/theguysudo/ENZO) - 自托管的 AI 工作区，将智能体、技能和工具（Gmail、日历、文件转换）结合起来，完全使用你自己的提供商 API 密钥运行，密钥被封存在浏览器中，服务器不会保存任何密钥。Apache-2.0 许可。![GitHub stars](https://img.shields.io/github/stars/theguysudo/ENZO?style=social)
- [OpenBot](https://github.com/regnull/openbot) - 自托管平台，用于运行一支持久化 AI 机器人团队，它们使用工具和 MCP 服务器、保持长期记忆、在共享会话中相互交接工作，并可暂停以请求人类批准。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/regnull/openbot?style=social)
- [Tale](https://github.com/tale-project/tale) - 采用 MIT 许可的项目工作区，通过共享任务、管理者委派、持久化沙箱和人工评审交付成果来协调人类与 AI 智能体。![GitHub stars](https://img.shields.io/github/stars/tale-project/tale?style=social)

#### 桌面与移动端 AI 应用

- [InvoiceFlowAI](https://github.com/EthanYoQ/Invoice-Downloader) - 开源 Windows/macOS 桌面应用，收集通过邮件发送的 PDF/OFD/XML 发票，使用 OCR 并辅以人工复核，导出 Excel 汇总表；可选的模型提供商支持信息提取。![GitHub stars](https://img.shields.io/github/stars/EthanYoQ/Invoice-Downloader?style=social)
- [AI Language Partner](https://github.com/duct-tape2/ai-language-partner) - 本地优先的 Expo + FastAPI 日语练习应用，面向韩语使用者，使用兼容 Whisper 的 STT、本地 TTS 和基于嵌入的对话匹配，无需运行时 LLM。![GitHub stars](https://img.shields.io/github/stars/duct-tape2/ai-language-partner?style=social)
- [Jan](https://github.com/janhq/jan) - 本地优先的 AI 应用框架。![GitHub stars](https://img.shields.io/github/stars/janhq/jan?style=social)
- [Cherry Studio](https://github.com/CherryHQ/cherry-studio) - AI 生产力工作室，提供智能对话、自主智能体和 300 多个助手，统一接入前沿 LLM。AGPL-3.0 许可。![GitHub stars](https://img.shields.io/github/stars/CherryHQ/cherry-studio?style=social)
- [DeepChat](https://github.com/ThinkInAIXYZ/deepchat) - 将强大 AI 连接到你的个人世界的智能助手。内置 MCP 和 ACP 支持、多种搜索引擎，注重隐私并支持本地数据存储。Apache-2.0 许可。![GitHub stars](https://img.shields.io/github/stars/ThinkInAIXYZ/deepchat?style=social)
- [SillyTavern](https://github.com/SillyTavern/SillyTavern) - 高度可定制的角色扮演前端。![GitHub stars](https://img.shields.io/github/stars/SillyTavern/SillyTavern?style=social)
- [ChatALL](https://github.com/ai-shifu/ChatALL) - 同时与多个 AI 机器人对话以找到最佳答案。桌面应用，可并排比较 ChatGPT、Claude、Gemini 等 20 多个 LLM。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/ai-shifu/ChatALL?style=social)
- [aiFetchly](https://github.com/robertzengcn/aiFetchly) - 面向业务自动化的开源桌面 AI 智能体：在 Windows、macOS 和 Linux 上提供线索获取、知识库 RAG、外联和定时工作流。![GitHub stars](https://img.shields.io/github/stars/robertzengcn/aiFetchly?style=social)
- [Chatbox](https://github.com/chatboxai/chatbox) - 面向 ChatGPT、Claude 等 LLM 的强大桌面 AI 客户端。跨平台，界面现代。GPLv3 许可（社区版）。![GitHub stars](https://img.shields.io/github/stars/chatboxai/chatbox?style=social)
- [Maid](https://github.com/Mobile-Artificial-Intelligence/maid) - 免费开源的 Android 应用，可在本地对接 llama.cpp 模型以及远程 API（Anthropic、DeepSeek、Mistral、Ollama、OpenAI）。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/Mobile-Artificial-Intelligence/maid?style=social)
- [Jev for WhatsApp](https://github.com/jev-chat/jev-chat-jarvis) - WhatsApp 的 Android 回复助手，读取屏幕上的聊天内容，用你的 OpenRouter 密钥判断意图和风险，生成两条候选回复，并填入你选中的一条。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/jev-chat/jev-chat-jarvis?style=social)
- [Dive](https://github.com/OpenAgentPlatform/Dive) - 开源 MCP Host 桌面应用，采用 Tauri/Electron 双架构。可无缝对接任何支持函数调用的 LLM。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/OpenAgentPlatform/Dive?style=social)
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) - 开源应用，将小型语言模型直接带到你的手机上。在 iOS 和 Android 上 100% 私密运行 AI，无需云端。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/a-ghorbani/pocketpal-ai?style=social)
- [Scowld](https://github.com/apoorvdarshan/scowld) - 原生 iOS AI 语音伴侣，带有动画 VRM 角色、语音与文字聊天、本地对话历史、可选的摄像头上下文，以及自带密钥（BYOK）的 AI、语音转文字和文字转语音提供商。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/apoorvdarshan/scowld?style=social)
- [Hermes Desktop](https://github.com/fathah/hermes-desktop) - 用于安装、配置 Hermes Agent 并与之对话的桌面伴侣应用。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/fathah/hermes-desktop?style=social)
- [DSH Studio](https://github.com/Moresyl/dsh-studio) - 跨平台桌面应用，用于在本地安装、运行、健康检查和守护 DeepSeek Harness。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/Moresyl/dsh-studio?style=social)
- [Craft Agents](https://github.com/craft-ai-agents/craft-agents-oss) - 面向智能体工作流的开源桌面与 Web 界面，内置 MCP 支持、可自定义的 API 连接和会话共享。Apache-2.0 许可。![GitHub stars](https://img.shields.io/github/stars/craft-ai-agents/craft-agents-oss?style=social)
- [Meetily](https://github.com/Zackriya-Solutions/meetily) - 注重隐私的本地 AI 会议助手，支持实时转录和摘要生成。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/Zackriya-Solutions/meetily?style=social)
- [oats](https://github.com/ariso-ai/oats) - 开源本地优先的 macOS 会议记录应用，提供实时转录、说话人标签、AI 摘要和可选的完全离线端侧模式。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/ariso-ai/oats?style=social)
- [OpenSuperWhisper](https://github.com/Starmel/OpenSuperWhisper) - macOS 听写应用，使用 Whisper 和 Parakeet 模型提供实时音频转录和拖放文件转录。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/Starmel/OpenSuperWhisper?style=social)
- [akou](https://github.com/GeiserX/akou) - macOS 桌面应用，可录制通话、在端侧转录并标注说话人，还允许 Claude Code、Codex 或任意 MCP 客户端在通话进行中跟随并提问。GPL-3.0 许可。![GitHub stars](https://img.shields.io/github/stars/GeiserX/akou?style=social)
- [T3 Code](https://github.com/pingdotgg/t3code) - 用于与 Codex、Claude Code、Cursor、OpenCode 等编码智能体交互的极简 Web GUI 与桌面应用。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/pingdotgg/t3code?style=social)
- [OpenWork](https://github.com/different-ai/openwork) - 开源桌面应用，可在智能体之间共享 AI 工作流、技能和 MCP 能力。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/different-ai/openwork?style=social)
- [ego lite](https://github.com/citrolabs/ego-lite) - 桌面浏览器，设计用于在隔离工作区中并行运行 Web 自动化任务和 AI 智能体。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/citrolabs/ego-lite?style=social)
- [Off Grid AI Desktop](https://github.com/off-grid-ai/OGAD) - 本地优先的 macOS AI 应用，通过 llama.cpp 完全在端侧运行 LLM 对话、图像生成、语音转录和个人记忆/RAG——数据绝不离开本机。AGPL-3.0 许可。![GitHub stars](https://img.shields.io/github/stars/off-grid-ai/OGAD?style=social)
- [Clips Kitty](https://github.com/ColinGPT9/clips-studio) - 本地优先的 Windows 桌面应用，把长视频和直播转为竖屏短视频，使用 faster-whisper 生成逐词字幕，并通过 Ollama 使用本地 LLM 挑选精彩片段。AGPL-3.0 许可。![GitHub stars](https://img.shields.io/github/stars/ColinGPT9/clips-studio?style=social)
- [Voz](https://github.com/quietbin/voz-mac) - macOS 听写应用，可在任意应用的光标处输入转录后的语音，内置 whisper.cpp 和 Whisper 模型，无需配置即可离线运行。还可在端侧录制会议，并通过内置的 llama.cpp 生成摘要。GPL-3.0 许可。![GitHub stars](https://img.shields.io/github/stars/quietbin/voz-mac?style=social)
- [Codex-X](https://github.com/yynxxxxx/Codex-X) - 面向 OpenAI Codex Desktop 和 CLI 的跨平台桌面管理应用，提供提供商切换、本地故障转移路由、会话历史管理、MCP 配置和 token 用量分析。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/yynxxxxx/Codex-X?style=social)
- [PersonalJarvis](https://github.com/PersonalJarvis/PersonalJarvis) - 面向 Windows、macOS 和 Linux 的桌面语音与聊天助手，具备自定义唤醒短语、本地语音转文字，以及可在终端面板中并排运行 CLI 编码智能体的工作区，兼容任意单一模型提供商密钥或通过 Ollama 运行的本地模型。Apache-2.0 许可。![GitHub stars](https://img.shields.io/github/stars/PersonalJarvis/PersonalJarvis?style=social)

#### 智能体与语音基础设施

- [Speech to Speech](https://github.com/huggingface/speech-to-speech) - 低延迟、完全模块化的语音智能体流水线，通过兼容 OpenAI Realtime 的 WebSocket API 集成语音活动检测、语音转文字、LLM 和文字转语音。Apache 2.0 许可。![GitHub stars](https://img.shields.io/github/stars/huggingface/speech-to-speech?style=social)
- [LiveKit Agents](https://github.com/livekit/agents) - 构建实时语音 AI 智能体的框架，提供 WebRTC 传输、STT-LLM-TTS 流水线和生产级编排。被 Salesforce Agentforce 和 Tesla 使用。Apache-2.0 许可。![GitHub stars](https://img.shields.io/github/stars/livekit/agents?style=social)
- [Pipecat](https://github.com/pipecat-ai/pipecat) - 面向语音和多模态对话 AI 的开源框架。用于构建实时语音智能体，支持语音转文字、LLM、文字转语音和实时视频。BSD-2-Clause 许可。![GitHub stars](https://img.shields.io/github/stars/pipecat-ai/pipecat?style=social)
- [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui) - Web 应用，通过聊天界面与任意 LangGraph 智能体（Python 与 TypeScript）交互。支持流式消息、处理中断并查看智能体状态。MIT 许可。![GitHub stars](https://img.shields.io/github/stars/langchain-ai/agent-chat-ui?style=social)

---
## 13. 开发者工具与集成

#### AI 原生 IDE 与开发环境

- [Claude Pet](https://github.com/nikhilagrima/claude-pet) - Claude Code 的桌面伴侣，可对 hook 事件做出动画表情与音效响应，按项目维护本地 SQLite 图记忆并在会话启动时以精简回顾形式注入，将重复出现的模式沉淀为 SKILL.md 文件，同时监控 GitHub 仓库活动。支持 macOS/Windows/Linux，100% 本地运行。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/nikhilagrima/claude-pet?style=social)
- [Ralph](https://github.com/frankbria/ralph-claude-code) - 面向 Claude Code 的自主 AI 开发循环，具备智能退出检测。以自我监控能力自动化迭代式编码工作流。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/frankbria/ralph-claude-code?style=social)
- [Nimbalyst](https://github.com/nimbalyst/nimbalyst) - 桌面应用，可在并行 Git worktree 中同时运行多个 Codex 与 Claude Code 会话。在统一界面中测试、对比方案并管理 AI 辅助开发工作流。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/nimbalyst/nimbalyst?style=social)
- [Yardlet](https://github.com/zzunkie/yardlet) - 本地优先的终端工作台，通过确定性任务队列、路由层、评估器与交接记录来运行已有的 Codex、Claude Code 或自定义 agent CLI。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/zzunkie/yardlet?style=social)
- [Nezha](https://github.com/hanshuaikang/nezha) - 面向 AI agent 时代的代码编辑器。以直观界面在本机跨项目运行多个 Claude Code 与 Codex agent。GPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/hanshuaikang/nezha?style=social)
- [Aider Desk](https://github.com/hotovo/aider-desk) - 面向 AI 驱动的软件工程师的平台。以现代化 UI 增强 aider 终端体验的桌面应用。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/hotovo/aider-desk?style=social)
- [Zed](https://github.com/zed-industries/zed) - 高性能多人协作代码编辑器，内置 AI 特性。出自 Atom 与 Tree-sitter 创造者之手。原生 AI agent 式编辑，支持任意 LLM 提供方。GPL 许可。 ![GitHub stars](https://img.shields.io/github/stars/zed-industries/zed?style=social)
- [Code Server](https://github.com/coder/code-server) - 在任意机器上运行 VS Code 并通过浏览器访问。自托管云 IDE，完整支持扩展。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/coder/code-server?style=social)
- [Gitpod](https://github.com/gitpod-io/gitpod) - 云开发环境平台，提供自动化预构建、临时工作区，并支持任意 IDE。核心开源，可自托管。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/gitpod-io/gitpod?style=social)
- [Onlook](https://github.com/onlook-dev/onlook) - 开源的 AI 优先设计与 React 编辑环境，用于可视化地构建和修改前端应用。 ![GitHub stars](https://img.shields.io/github/stars/onlook-dev/onlook?style=social)
- [Daytona](https://github.com/daytonaio/daytona) - 用于运行 AI 生成代码的安全弹性基础设施。自托管的 GitHub Codespaces 替代方案，支持多种 IDE、预构建与任意云服务商。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/daytonaio/daytona?style=social)
- [AI Workdeck](https://github.com/zeweihan/aiworkdeck) - 面向法律与文档密集型工作流的开源 AI 原生 IDE 工作区——"律师的 VS Code"。可自托管，具备 MCP agent 编排、OCR、尽职调查风险标记、证据链管理、WPS WebOffice 集成与智能剪贴板。支持使用 Ollama + 本地存储的气隙部署。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/zeweihan/aiworkdeck?style=social)
- [Orca](https://github.com/stablyai/orca) - 代理式开发环境（ADE），可在桌面与移动端并行 Git worktree 中运行和编排编码 agent。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/stablyai/orca?style=social)
- [Terax](https://github.com/crynta/terax-ai) - 轻量级、终端优先的 AI 原生开发工作区（ADE），具备多标签终端、带 AI 编辑 diff 的代码编辑器、源码控制与 agent 式工作流。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/crynta/terax-ai?style=social)
- [Pi Web](https://github.com/agegr/pi-web) - 面向 pi coding agent 的 Web UI 与本地工作区，支持会话浏览、文件预览和模型配置。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/agegr/pi-web?style=social)
- [Garcon](https://github.com/cfal/garcon) - 浏览器与移动端工作区，用于运行和操控并行的 Claude Code、Codex、Cursor Agent、OpenCode、Amp、Droid 与 Pi 会话，集成终端、文件编辑、diff 审查、Git/PR 工作流、移动端审批、定时任务与跨 agent 传输。GPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/cfal/garcon?style=social)
- [screenshot-to-code](https://github.com/abi/screenshot-to-code) - MIT 许可的工具，将截图、原型和 Figma 设计转换为前端代码。 ![GitHub stars](https://img.shields.io/github/stars/abi/screenshot-to-code?style=social)
- [Atlas](https://github.com/pacifio/atlas) - 本地优先的代理式开发环境与源码控制系统，将 Git 提交与 agent 会话、工具调用、推理轨迹及共享记忆关联起来。 ![GitHub stars](https://img.shields.io/github/stars/pacifio/atlas?style=social)

#### AI 编码助手（开源）

- [Continue](https://github.com/continuedev/continue) - 面向 VS Code 与 JetBrains 的开源 AI 编码副驾。 ![GitHub stars](https://img.shields.io/github/stars/continuedev/continue?style=social)
- [Tabby](https://github.com/TabbyML/tabby) - 可自托管的 AI 编码助手。 ![GitHub stars](https://img.shields.io/github/stars/TabbyML/tabby?style=social)
- [Cline](https://github.com/cline/cline) - 开源 IDE 编码 agent，可在用户批准后编辑文件、执行命令并使用工具。 ![GitHub stars](https://img.shields.io/github/stars/cline/cline?style=social)
- [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter) - 让 LLM 在本地运行代码。 ![GitHub stars](https://img.shields.io/github/stars/OpenInterpreter/open-interpreter?style=social)
- [Aider](https://github.com/Aider-AI/aider) - 基于终端的 AI 结对编程工具。在本地编辑器中修改代码，由 aider 实现变更。支持多种 LLM、语音编程和自动 Git 提交。在 SWE Bench 上得分领先。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Aider-AI/aider?style=social)
- [Kimi Code CLI](https://github.com/MoonshotAI/kimi-code) - 运行在终端中的 AI 编码 agent，可编辑代码、执行 shell 命令并管理 Model Context Protocol（MCP）服务器与技能。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/MoonshotAI/kimi-code?style=social)
- [Qwen Code](https://github.com/QwenLM/qwen-code) - 面向终端的开源 AI agent，针对 Qwen 系列模型优化。支持 OpenAI、Anthropic、Gemini、阿里云、OpenRouter 等多种协议提供方。具备基于 Skills 与 SubAgents 的 agent 式工作流。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/QwenLM/qwen-code?style=social)
- [Forge](https://github.com/NorviaLabs/forge) - 开源终端 AI 编码 agent（Rust TUI），在单一键盘驱动的工作区中统一 agent、代码编辑器与 shell，支持持久化 SQLite 会话日志、MCP 支持以及感知审批的命令执行。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/NorviaLabs/forge?style=social)
- [DeepCode](https://github.com/HKUDS/DeepCode) - 将研究论文与自然语言转换为生产级代码。AI 驱动的"研究到代码"自动化工具。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/HKUDS/DeepCode?style=social)
- [OpenSpec](https://github.com/Fission-AI/OpenSpec) - 面向 AI 编码助手的规格驱动开发（SDD）框架与 CLI 工具，为提议、实现和归档代码变更提供结构化工作流。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Fission-AI/OpenSpec?style=social)
- [Spec Kit](https://github.com/github/spec-kit) - 用于规格驱动开发（Spec-Driven Development）的工具包与 CLI，可与 AI 编码 agent 集成，从可执行规格生成结构化的代码实现。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/github/spec-kit?style=social)
- [SpecJudge](https://github.com/JoaquinRuiz/SpecJudge) - 读取项目规格文件并推荐适合的 LLM 的 CLI，通过 Ollama 使用本地评判模型，并为每个需求等级标注所依据的规格片段。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/JoaquinRuiz/SpecJudge?style=social)
- [Open Code Review](https://github.com/alibaba/open-code-review) - AI 驱动的代码审查 CLI 工具，将确定性分析流水线与 LLM agent 结合，提供精确到行级的反馈。Apache-2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/alibaba/open-code-review?style=social)
- [CodeOtter](https://github.com/dharmeshgurnani/CodeOtter) - 面向 GitHub、Forgejo 与 Gitea 的可自托管 AI PR 评审工具，可为 PR 评分、以状态检查方式强制合并门禁并发布行内修复建议，使用经 llama.cpp 运行的本地 GGUF 模型或托管模型 API。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/dharmeshgurnani/CodeOtter?style=social)

#### 笔记本与交互式计算

- [Open Notebook](https://github.com/lfnovo/open-notebook) - Notebook LM 的开源实现，支持多模态内容（PDF、视频、音频、网页）。具备多说话人播客生成、18+ AI 提供方集成，以及全文 + 向量搜索。可自托管，数据完全自主。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/lfnovo/open-notebook?style=social)
- [Deta Surf](https://github.com/deta/surf) - 个人 AI 笔记本，用 AI 生成的笔记整理文件与网页。本地优先的数据存储、开放的数据格式与开放的模型选择（含本地模型）。面向研究与思考工作流的跨平台桌面应用。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/deta/surf?style=social)
- [FilePilot AI](https://github.com/cuiheng511/filepilot-ai) - 本地优先的文件智能应用、CLI 与 MCP 服务器，为 AI agent 工作流提供本地文件的搜索、摘要、打标、去重与安全整理。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/cuiheng511/filepilot-ai?style=social)
- [Quarto](https://github.com/quarto-dev/quarto-cli) - 基于 Pandoc 构建的开源科学与技术出版系统。用 Python、R、Julia 与 Observable 创建动态内容。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/quarto-dev/quarto-cli?style=social)
- [Drawdata](https://github.com/koaning/drawdata) - 在 Python 笔记本中绘制数据集。可在 Jupyter 环境中直接创建和编辑数据集的交互式数据可视化工具。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/koaning/drawdata?style=social)
- [Deepnote](https://github.com/deepnote/deepnote) - Jupyter 的替代方案，AI 优先的设计、精致 UI 与原生数据集成。本地使用 Python、R 与 SQL，随后可扩展到 Deepnote 云端进行协作与部署可交付的数据应用。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/deepnote/deepnote?style=social)
- [Zasper](https://github.com/zasper-io/zasper) - 使用 Go 构建的高性能 Jupyter Notebook IDE。相比 JupyterLab，CPU 占用最多降低 5 倍、内存减少 40 倍。实现了 Jupyter 的 wire protocol，并支持大规模并发。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/zasper-io/zasper?style=social)

#### IDE 插件与扩展

- [Parallel Search MCP](https://github.com/parallel-web/search-mcp) - MIT 许可的客户端包与安装技能，让编码 agent 无需 Parallel 账号或 API key 即可接入 Parallel 托管的网页搜索与页面抓取服务。 ![GitHub stars](https://img.shields.io/github/stars/parallel-web/search-mcp?style=social)
- [Archify](https://github.com/tt-a1i/archify) - 面向 Claude Code、Codex 与 OpenCode 的图表生成 agent 技能，将自然语言转换为交互式的系统图、工作流图、时序图、数据流图与生命周期图。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/tt-a1i/archify?style=social)
- [Graphify](https://github.com/Graphify-Labs/graphify) - AI 编码助手技能，将代码库与资源目录映射为可交互的知识图谱，支持本地查询与导航。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Graphify-Labs/graphify?style=social)
- [llama.vim](https://github.com/ggml-org/llama.vim) - 基于 llama.cpp 的本地 LLM 代码补全插件，适用于 Vim/Neovim。快速、隐私优先、无需 API key。 ![GitHub stars](https://img.shields.io/github/stars/ggml-org/llama.vim?style=social)
- [CodeCompanion.nvim](https://github.com/olimorris/codecompanion.nvim) - 面向 Neovim 的 AI 驱动编码助手。支持行内代码生成、对话、动作与工具使用，并兼容多种 LLM 提供方。 ![GitHub stars](https://img.shields.io/github/stars/olimorris/codecompanion.nvim?style=social)
- [ProxyAI](https://github.com/carlrobertoh/ProxyAI) - 面向 JetBrains IDE 的领先开源 AI 副驾。可在任意环境中连接任意模型，支持自动应用、图片对话、文件引用、网页搜索与可自定义人格。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/carlrobertoh/ProxyAI?style=social)
- [avante.nvim](https://github.com/yetone/avante.nvim) - 将 Cursor 风格的 AI IDE 特性带入 Vim 的 Neovim 插件。用自然语言编辑代码、从上下文生成代码，并与 AI 讨论你的代码库。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/yetone/avante.nvim?style=social)
- [Serena](https://github.com/oraios/serena) - 面向编码 agent 的强大 MCP 工具包，提供语义化检索与编辑能力。集成语言服务器以实现 IDE 级别的代码理解。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/oraios/serena?style=social)
- [Jupyter AI](https://github.com/jupyterlab/jupyter-ai) - 在笔记本内进行对话与代码生成。 ![GitHub stars](https://img.shields.io/github/stars/jupyterlab/jupyter-ai?style=social)
- [Minuet AI](https://github.com/milanglacier/minuet-ai.nvim) - Neovim 插件，通过 OpenAI、Gemini、Claude、Ollama、Llama.cpp、Codestral 等主流 LLM 提供边输入边补全的代码补全。GPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/milanglacier/minuet-ai.nvim?style=social)
- [Peekaboo](https://github.com/steipete/Peekaboo) - macOS CLI 与 MCP 服务器，让 AI agent 截取屏幕并自动化 UI 交互。通过本地或远程 AI 模型进行视觉问答。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/steipete/Peekaboo?style=social)
- [Skills](https://github.com/mattpocock/skills) - 面向 Claude Code 及其他 AI 编码 agent 的生产力与工程技能合集。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/mattpocock/skills?style=social)
- [planning-with-files](https://github.com/OthmanAdi/planning-with-files) - 面向 AI 编码 agent 的持久化文件规划技能，通过确定性的完成门槛在上下文丢失与重启后仍能延续工作。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/OthmanAdi/planning-with-files?style=social)
- [Understand Anything](https://github.com/Egonex-AI/Understand-Anything) - 将子目录、代码库或知识库转化为可交互、可由 AI 导航的知识图谱与仪表盘。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Egonex-AI/Understand-Anything?style=social)
- [Knowledge Work Plugins (Anthropic)](https://github.com/anthropics/knowledge-work-plugins) - 面向 Claude Cowork 与 Claude Code 的角色专属插件与子 agent 合集。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/anthropics/knowledge-work-plugins?style=social)
- [Cursor Plugins](https://github.com/cursor/plugins) - Cursor 官方插件，涵盖开发工具、工作流与 SaaS 集成。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/cursor/plugins?style=social)
- [Claude Plugins Marketplace](https://github.com/anthropics/claude-plugins-official) - 由 Anthropic 官方维护的 Claude Code 高质量插件目录。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/anthropics/claude-plugins-official?style=social)
- [Agent Toolkit for AWS](https://github.com/aws/agent-toolkit-for-aws) - AWS 官方支持的 MCP 服务器、技能与插件，帮助 AI 编码 agent 在 AWS 上构建、部署和管理应用。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/aws/agent-toolkit-for-aws?style=social)
- [Caveman](https://github.com/JuliusBrussee/caveman) - 面向 Claude Code、Codex、Gemini、Cursor 等编码 agent 的技能与安装脚本，通过要求 agent 使用简洁的"穴居人式"文风来减少 token 输出。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/JuliusBrussee/caveman?style=social)
- [Chrome DevTools MCP](https://github.com/ChromeDevTools/chrome-devtools-mcp) - Google 官方的 Model Context Protocol（MCP）服务器，让编码 agent 能控制和检查运行中的 Chrome 浏览器，用于自动化、调试与网页性能分析。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/ChromeDevTools/chrome-devtools-mcp?style=social)
- [Codex Plugin for Claude Code](https://github.com/openai/codex-plugin-cc) - Claude Code 的官方集成插件，让开发者运行 Codex 代码审查、委派任务并管理后台编码作业。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/openai/codex-plugin-cc?style=social)
- [Compound Engineering](https://github.com/EveryInc/compound-engineering-plugin) - 面向 Claude Code、Codex、Cursor 等编码 agent 的官方 AI 工程插件，实现"头脑风暴 → 计划 → 执行 → 评审 → 沉淀"的迭代工作流。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/EveryInc/compound-engineering-plugin?style=social)
- [Unity MCP](https://github.com/CoplayDev/unity-mcp) - Model Context Protocol（MCP）服务器，将 AI 助手与 Unity Editor 连接起来，自动化资源、场景与脚本工作流。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/CoplayDev/unity-mcp?style=social)
- [.NET Agent Skills](https://github.com/dotnet/skills) - 精选的技能、自定义 agent 与插件合集，协助 AI 编码 agent 进行 .NET 与 C# 开发。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/dotnet/skills?style=social)
- [Claude Video](https://github.com/bradautomates/claude-video) - 一项技能与插件，让 AI 编码助手能够观看、转录并分析视频内容。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/bradautomates/claude-video?style=social)
- [Stitch Design Skills](https://github.com/google-labs-code/stitch-skills) - 与 Stitch MCP 服务器协同工作的 agent 技能与插件库，遵循 Agent Skills 开放标准。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/google-labs-code/stitch-skills?style=social)
- [Impeccable](https://github.com/pbakaus/impeccable) - 设计指导技能、CLI 工具与检测规则，用于改进 AI 编码 agent 生成的用户界面设计。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/pbakaus/impeccable?style=social)
- [Hallmark](https://github.com/Nutlope/hallmark) - 面向 Claude Code、Cursor 与 Codex 的反"AI 味"设计技能与规则集，防止产出千篇一律的 AI 生成布局。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Nutlope/hallmark?style=social)
- [Code Review Graph](https://github.com/tirth8205/code-review-graph) - 本地优先的代码智能图谱，为代码库构建持久化映射，为 AI 编码工具提供 token 优化的上下文。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/tirth8205/code-review-graph?style=social)
- [CAD Skills](https://github.com/earthtojake/text-to-cad) - 模块化 agent 技能库，用于生成、检查、切片与导出 CAD 及机器人描述几何体（STEP、STL、URDF、SDF）。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/earthtojake/text-to-cad?style=social)
- [ADHD](https://github.com/UditAkhourii/adhd) - 面向 AI agent 的思维树式发散推理技能，派生并行的认知框架以剪除陷阱并对存活的路径评分。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/UditAkhourii/adhd?style=social)
- [MCP Lens](https://github.com/labmimors/dsh-mcp-lens) - DeepSeek Harness 插件，通过搜索界面和带有精确输入 schema 的显式服务器/工具调用界面暴露已配置的远程 MCP 工具。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/labmimors/dsh-mcp-lens?style=social)
- [jev-use](https://github.com/shitianfang/jev-use) - Claude Code、Codex 与 pi 的插件，将 agent 循环中无需文本输出的步骤路由给带类型的评判模型，把不该由它决定的内容交还给 LLM。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/shitianfang/jev-use?style=social)
- [Cockney](https://github.com/Kadajett/cockney) - 面向 omp、Claude Code 与 Codex 的技能、Claude Code 插件与安装脚本，让编码 agent 以 Cockney 风格简短作答并做出最小且正确的改动，同时对高风险操作给出通俗英文警告。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Kadajett/cockney?style=social)

#### UI 组件与聊天库

- [Assistant UI](https://github.com/assistant-ui/assistant-ui) - 用于构建生产级 AI 聊天界面的 React/TypeScript 库。提供开箱即用的流式消息、工具调用与多模态输入组件。 ![GitHub stars](https://img.shields.io/github/stars/assistant-ui/assistant-ui?style=social)
- [Deep Chat](https://github.com/OvidijusParsiunas/deep-chat) - 为你的网站提供完全可定制的 AI 聊天机器人组件。支持 OpenAI、直连 API 服务与自定义端点。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/OvidijusParsiunas/deep-chat?style=social)
- [Markstream](https://github.com/Simon-He95/markstream-vue) - 面向 AI 聊天界面的多框架流式 Markdown 渲染器，支持未完成 Markdown 处理、Mermaid、KaTeX、Shiki/Monaco 代码块、SSR，并提供 Vue、React、Svelte 与 Angular 的包。 ![GitHub stars](https://img.shields.io/github/stars/Simon-He95/markstream-vue?style=social)
- [CopilotKit](https://github.com/CopilotKit/CopilotKit) - 构建全栈 agent 式应用、生成式 UI 与聊天应用的一流 SDK。也是被 Google、LangChain、AWS 与 Microsoft 采用的 AG-UI Protocol 的创造者。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/CopilotKit/CopilotKit?style=social)
- [json-render](https://github.com/vercel-labs/json-render) - 生成式 UI 框架，用于渲染由 LLM 与 agent 流式输出的结构化 JSON 所生成的动态、类型安全界面。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/vercel-labs/json-render?style=social)

#### CLI 工具与 API 客户端

- [Ruler](https://github.com/intellectronica/ruler) - 集中式 AI agent 规则注册中心。为各项目中的 AI 编码 agent 管理和分发规则。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/intellectronica/ruler?style=social)
- [PR-Agent (Qodo)](https://github.com/qodo-ai/pr-agent) - 面向 GitHub、GitLab、Bitbucket 与 Azure DevOps 的 AI 驱动代码审查 agent。支持自动化 PR 分析、改进建议，以及通过 CLI、GitHub Actions 或 webhook 的多平台部署。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/qodo-ai/pr-agent?style=social)
- [LLM (Simon Willison)](https://github.com/simonw/llm) - 用于通过远程 API 或本地方式与数十种 LLM 交互的 CLI 工具与 Python 库。可扩展的插件生态，支持 SQLite 日志。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/simonw/llm?style=social)
- [aicommits](https://github.com/Nutlope/aicommits) - 用 AI 为你撰写 Git 提交信息的 CLI，让���再也不用手动写提交信息。支持多种提供方，包括 OpenAI、Groq、xAI、Ollama 与 LM Studio。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Nutlope/aicommits?style=social)
- [Codex CLI](https://github.com/openai/codex) - OpenAI 运行在终端中的轻量级编码 agent。支持代码生成、文件编辑与需审批的命令执行。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/openai/codex?style=social)
- [VibePod](https://github.com/VibePod/vibepod-cli) - 统一 CLI，在隔离的 Docker 或 Podman 容器中运行 Claude Code、Codex、OpenCode 等编码 agent，提供本地 HTTP 流量与 token 指标，并带并排对比仪表盘。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/VibePod/vibepod-cli?style=social)
- [Repomix](https://github.com/yamadashy/repomix) - 强大的工具，可将整个仓库打包为单个对 AI 友好的文件。非常适合把代码库喂给 LLM，具备智能过滤与 token 计数。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/yamadashy/repomix?style=social)
- [DevProjex](https://github.com/Avazbek22/DevProjex) - 开源 GUI、TUI、CLI 与只读 MCP 应用，用于选择项目文件、预览、脱敏密钥，并按 Git 作用域与语法感知压缩打包带 token 计数的上下文。Apache-2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Avazbek22/DevProjex?style=social)
- [GitIngest](https://github.com/cyclotruc/gitingest) - 把任意 GitHub URL 中的 'hub' 换成 'ingest'，即可获得对提示友好的代码库提取内容。针对 Python 生态与数据科学工作流做了优化。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/cyclotruc/gitingest?style=social)
- [GitDiagram](https://github.com/ahmedkhaleel2004/gitdiagram) - 使用 LLM 将任意 GitHub 仓库转换为交互式系统架构图，带可点击的源码链接与 Mermaid 导出。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/ahmedkhaleel2004/gitdiagram?style=social)
- [Instructor](https://github.com/jxnl/instructor) - 使用 Pydantic 模型从 LLM 中提取结构化、可校验数据的 Python 库。处理校验、重试与错误处理，支持 15+ 提供方。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/jxnl/instructor?style=social)
- [Mirascope](https://github.com/Mirascope/mirascope) - 用于构建 LLM 应用的 Python 工具包，内置自动版本管理、追踪与成本统计。是给希望掌控一切的开发者的"LLM 反框架"。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Mirascope/mirascope?style=social)
- [Context7](https://github.com/upstash/context7) - 为 LLM 与 AI 代码编辑器提供最新的代码文档。通过 MCP 将最新文档与代码示例直接拉取进 LLM 上下文。消除幻觉 API。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/upstash/context7?style=social)
- [Claude Squad](https://github.com/smtg-ai/claude-squad) - 像管理 Claude Code、Codex、OpenCode 与 Amp 一样管理多个 AI 终端 agent。面向 AI 编码 agent 的终端多路复用器，支持会话管理与并行执行。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/smtg-ai/claude-squad?style=social)
- [YYLO](https://github.com/yylo-dev/yylo) - 命令行编排器，在隔离的 git worktree 中并行运行 AI 编码 agent（Claude Code、Codex、Gemini CLI），并在看板上跟踪，带有带类型的合并与发布流程。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/yylo-dev/yylo?style=social)
- [Herdr](https://github.com/ogulcancelik/herdr) - 终端 agent 多路复用器与工作区管理器，具备鼠标原生的分屏与 agent 状态自动检测。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/ogulcancelik/herdr?style=social)
- [TUIOS](https://github.com/Gaurav-Gosain/tuios) - 面向编码 agent 的终端窗口管理器与多路复用器，支持跨机器持久会话、agent 状态上报、共享收件箱、harness 集成与 MCP 控制。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Gaurav-Gosain/tuios?style=social)
- [agent-manager](https://github.com/YoanWai/agent-manager) - 终端 UI，可并排运行 Claude Code、Codex、OpenCode、Gemini CLI 等编码 agent CLI，每个都在独立的持久 tmux 会话中，提供实时状态、可选的按会话 Git worktree，以及能把行级评论回传给 agent 的 diff 审查。Apache-2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/YoanWai/agent-manager?style=social)
- [agenttrace](https://github.com/luoyuctl/agenttrace) - 本地优先的 TUI，用于观测跨 Claude Code、Codex CLI、Gemini CLI、Aider、Cursor 导出、OpenCode 等工具的 AI 编码 agent 会话。 ![GitHub stars](https://img.shields.io/github/stars/luoyuctl/agenttrace?style=social)
- [agentsview](https://github.com/kenn-io/agentsview) - 本地优先的会话智能与成本分析仪表盘，面向 AI 编码 agent，支持 Claude Code、Codex 及其他工具。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/kenn-io/agentsview?style=social)
- [Uni-CLI](https://github.com/olo-dot-io/Uni-CLI) - 自我修复的 CLI 目录，将 Web、桌面、Electron 与桥接工具暴露为面向 AI agent 的确定性命令。 ![GitHub stars](https://img.shields.io/github/stars/olo-dot-io/Uni-CLI?style=social)
- [OpenChamber Mobile Bridge](https://github.com/Zaradacht/opencode-openchamber-mobile-bridge) - OpenCode/OpenChamber 辅助工具，通过宿主桥接把基于 devcontainer 的编码会话暴露给私有的 Tailscale 移动端访问。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/Zaradacht/opencode-openchamber-mobile-bridge?style=social)
- [Desktop Commander MCP](https://github.com/wonderwhy-er/DesktopCommanderMCP) - 面向 Claude 的 MCP 服务器，提供终端控制、文件系统搜索与基于 diff 的文件编辑能力。通过 Model Context Protocol 实现自主代码编辑。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/wonderwhy-er/DesktopCommanderMCP?style=social)
- [Mobile MCP](https://github.com/mobile-next/mobile-mcp) - MCP 服务器，让 AI agent 通过无障碍树、截图与结构化设备工具自动化和检查 iOS 与 Android 的模拟器、模拟运行环境及真机。Apache-2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/mobile-next/mobile-mcp?style=social)
- [REA](https://github.com/morluto/rea) - MCP 服务器与 CLI，让 AI agent 能以有证据支撑的逆向工程结果检查二进制文件、应用程序、网站与运行时行为。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/morluto/rea?style=social)
- [Claude Code Action](https://github.com/anthropics/claude-code-action) - 用于在 PR 与 issue 工作流中运行 Claude Code 的 GitHub Action，支持感知审批的自动化与编码辅助。 ![GitHub stars](https://img.shields.io/github/stars/anthropics/claude-code-action?style=social)
- [OfficeCLI](https://github.com/iOfficeAI/OfficeCLI) - 专为 AI agent 打造的 Office 套件，无需安装外部 Office 即可读取、编辑和自动化 Word、Excel 与 PowerPoint 文件。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/iOfficeAI/OfficeCLI?style=social)
- [CLI-Anything](https://github.com/HKUDS/CLI-Anything) - 将软件应用转换为面向 AI 编码 agent 的原生命令行界面的框架。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/HKUDS/CLI-Anything?style=social)
- [MulmoTerminal](https://github.com/receptron/mulmoterminal) - 通过一条 `npx` 命令启动的 Claude Code 与 Codex 实时会话浏览器网格。每个单元格都是真实的 PTY，带颜色编码状态、由 tmux 支撑的持久化，以及每个单元格独立的 git worktree。对 Claude Code 而言，needs-you 与 done 分开展示，读取自该 CLI 自身的 hooks。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/receptron/mulmoterminal?style=social)
- [Worktrunk](https://github.com/max-sixty/worktrunk) - 面向并行 AI 编码 agent 工作流设计的 Git worktree 管理与工作流自动化 CLI。 ![GitHub stars](https://img.shields.io/github/stars/max-sixty/worktrunk?style=social)
- [bestax-mcp](https://github.com/allxsmith/bestax/tree/main/bestax-mcp) - 离线 MCP 服务器与 Agent Skills，为编码 agent 提供 Bestax（一个面向 Bulma v1 的 React 组件库）的 props、示例与 CSS 变量。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/allxsmith/bestax?style=social)
- [KyttoMCP](https://github.com/heyitsjakub/KyttoMCP) - 面向 macOS（Swift）与 Windows（.NET）的桌面应用，在一个矩阵中展示 Claude Desktop、Claude Code、Cursor、VS Code 与 Codex 的全部 MCP 服务器，执行健康检查与工具描述安全扫描，并在写入配置前备份。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/heyitsjakub/KyttoMCP?style=social)
- [Archcore](https://github.com/archcore-ai/archcore) - 面向 AI 编码 agent、以 Git 为原生的上下文工程与规格驱动开发工具。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/archcore-ai/archcore?style=social)
- [Caprock](https://github.com/dspv/caprock) - 面向 Claude Code、Codex 与 OpenCode 会话的桌面应用与本地仪表盘：标签页与分屏中的 agent、按 API 目录价计算的每个仓库 token 成本、套餐额度窗口、循环告警与可搜索历史。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/dspv/caprock?style=social)
- [LynxPrompt](https://github.com/GeiserX/LynxPrompt) - 可自托管的 Web 应用与 `lynxp` CLI，把一套 AI 编码规则转换为 30 种工具的规则文件，其中包括 AGENTS.md、CLAUDE.md 与 Cursor 规则，并提供团队可共享的蓝图。AGPL-3.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/GeiserX/LynxPrompt?style=social)

#### SDK 与 API 开发工具

- [Vercel AI SDK](https://github.com/vercel/ai) - 与提供方无关的 TypeScript 工具包，用于构建 AI 驱动的应用与 agent。为 OpenAI、Anthropic、Google 及 20+ 提供方提供统一 API，一流支持流式输出、工具调用与结构化输出。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/vercel/ai?style=social)
- [GitHub Copilot SDK](https://github.com/github/copilot-sdk) - 用于将 GitHub Copilot Agent 集成进应用与服务的多平台 SDK。经生产验证的 agent 运行时，具备规划、工具调用与上下文管理能力。无需编写自己的编排逻辑即可构建 Copilot 风格的 agent。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/github/copilot-sdk?style=social)
- [IBM MCP Context Forge](https://github.com/IBM/mcp-context-forge) - MCP/A2A/REST API 的网关与注册中心，为生产级 agent 集成提供统一发现、路由与护栏。 ![GitHub stars](https://img.shields.io/github/stars/IBM/mcp-context-forge?style=social)
- [Fern](https://github.com/fern-api/fern) - 面向 REST API 的开源 SDK 生成器。可从 OpenAPI 规范生成 TypeScript、Python、Go、Java 等语言的类型安全 API 客户端。为 OpenAI、Anthropic、Cloudflare 等公司提供 SDK 支撑。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/fern-api/fern?style=social)
- [Cortex](https://github.com/cortex-docs/cortex) - 从 OpenAPI、AsyncAPI、GraphQL、Protocol Buffer、OpenRPC 与 Markdown 源生成带类型的 SDK、API 文档与 MCP 服务器。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/cortex-docs/cortex?style=social)

#### 测试与调试工具

- [no-mistakes](https://github.com/kunchenguid/no-mistakes) - 本地 Git 代理与校验流水线，在临时 worktree 中运行 AI 驱动的检查并应用修复，然后才转发推送并创建干净的 PR。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/kunchenguid/no-mistakes?style=social)
- [oai-smoke](https://github.com/airouter-dev/openai-compatible-api-smoke-test) - MIT 许可、仅依赖标准库的 Go CLI，用于校验 OpenAI 兼容 API 的 `/models` 及可选的 `/chat/completions` 行为，并输出有界诊断信息，不含凭据与响应体。 ![GitHub stars](https://img.shields.io/github/stars/airouter-dev/openai-compatible-api-smoke-test?style=social)
- [e2e](https://github.com/tester-army/e2e) - 面向 agent 的 Web 与移动端端到端测试框架，执行自然语言目标并记录已验证的动作，无需再次调用模型即可重放。Apache-2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/tester-army/e2e?style=social)

#### 提示工程与管理

- [GEPA](https://github.com/gepa-ai/gepa) - 使用自然语言反思与帕累托前沿学习的反思式提示演化优化器。在提示优化上优于强化学习。已与 DSPY 和 MLflow 集成。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/gepa-ai/gepa?style=social)
- [Entroly](https://github.com/juyterman1000/entroly) - 本地优先的 MCP 服务器，支持显式预算的上下文选择、内容寻址的精确恢复，以及可审计的上下文回执。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/juyterman1000/entroly?style=social)
- [Vibe-Coding Prompt Template](https://github.com/KhazP/vibe-coding-prompt-template) - 分阶段的提示工作流与 CLI，把产品想法转化为 PRD、技术设计以及面向 AI 编码 agent 的 `AGENTS.md` 指令文件。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/KhazP/vibe-coding-prompt-template?style=social)
- [Webcmd](https://github.com/agentrhq/webcmd) - 面向 AI agent 的自学习浏览器基础设施，将网站导航编译为按站点划分的确定性 CLI 命令（Apache-2.0，TypeScript）。 ![GitHub stars](https://img.shields.io/github/stars/agentrhq/webcmd?style=social)

---

## 14. 资源与学习

#### 附有开源实现的论文

- [Papers with Code](https://paperswithcode.com) - 将论文与开源代码、数据集关联起来的权威数据库。
- [Hugging Face Papers](https://huggingface.co/papers) - 每日更新的最新 arXiv 论文（含开放权重）信息流。
- [Open LLM Leaderboard (Hugging Face)](https://huggingface.co/spaces/open-llm-leaderboard) - 开放模型的实时排名。

#### 社区、论坛与 newsletter

- [Hugging Face Discussions](https://discuss.huggingface.co) - 最大的开放 AI 论坛。

#### 教育资源与课程

- [AI Engineering from Scratch (rohitg00)](https://github.com/rohitg00/ai-engineering-from-scratch) - 全面课程，通过从零实现来覆盖机器学习、深度学习、NLP、计算机视觉与 agent。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/rohitg00/ai-engineering-from-scratch?style=social)
- [AMD Strix Halo Local LLM Guide](https://github.com/hogeheer499-commits/strix-halo-guide) - 针对 Ryzen AI MAX+ 395 本地 AI 系统、可复现的 Ubuntu、Ollama、llama.cpp、Vulkan/RADV 与 ROCm 安装配置及基准测试证据。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/hogeheer499-commits/strix-halo-guide?style=social)
- [AI Agents in Depth](https://github.com/bojieli/ai-agent-book) - 开源教材与代码仓库，涵盖 AI agent 设计原则、架构与工程实践。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/bojieli/ai-agent-book?style=social)
- [Claude How To](https://github.com/luongnv89/claude-howto) - 面向 Claude Code 的全面学习路径与模板指南，涵盖安装配置、hooks、自定义技能与 MCP 服务器集成。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/luongnv89/claude-howto?style=social)
- [Maths, CS & AI Compendium](https://github.com/HenryNdubuaku/maths-cs-ai-compendium) - 一部开放且另辟蹊径的教材，从零开始讲解数学、计算机科学与人工智能。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/HenryNdubuaku/maths-cs-ai-compendium?style=social)
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA) - 探讨本地/开源 LLM 话题的首选 subreddit。

#### 课程与交互式练习场

- [Hugging Face Course](https://huggingface.co/learn) - 仅使用开放模型的免费动手课程。
- [ML For Beginners (Microsoft)](https://github.com/microsoft/ML-For-Beginners) - 面向初学者的经典机器学习课程，12 周、26 课、52 道测验。课程体系全面，涵盖回归、分类、聚类与 NLP，并配有实战项目。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/ML-For-Beginners?style=social)
- [AI For Beginners (Microsoft)](https://github.com/microsoft/AI-For-Beginners) - 12 周、24 课的人工智能课程体系。涵盖符号 AI、神经网络、计算机视觉、NLP 与强化学习，并配有动手实验。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/AI-For-Beginners?style=social)
- [Generative AI for Beginners (Microsoft)](https://github.com/microsoft/generative-ai-for-beginners) - 21 节课，涵盖生成式 AI 基础、提示工程、RAG 应用、微调与 LLM 应用部署，并配实战练习。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/generative-ai-for-beginners?style=social)
- [LangChain Academy](https://academy.langchain.com) - 关于 agent 与 RAG 的免费课程。
- [Data Science for Beginners (Microsoft)](https://github.com/microsoft/Data-Science-For-Beginners) - 10 周、20 课的数据科学基础课程体系。涵盖数据准备、可视化、建模与部署，并配实战项目。 ![GitHub stars](https://img.shields.io/github/stars/microsoft/Data-Science-For-Beginners?style=social)
- [The Incredible PyTorch](https://github.com/ritchieng/the-incredible-pytorch) - 为深度学习研究者精选的 PyTorch 教程、论文、项目与社区列表。 ![GitHub stars](https://img.shields.io/github/stars/ritchieng/the-incredible-pytorch?style=social)
- [Deep RL Class (Hugging Face)](https://github.com/huggingface/deep-rl-class) - 免费的深度强化学习课程，含动手练习，并可将训练好的 agent 发布到 Hugging Face Hub。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/deep-rl-class?style=social)
- [NLP Course (Yandex Data School)](https://github.com/yandexdataschool/nlp_course) - YSDA 的自然语言处理课程，采用 2025 年材料，涵盖文本分类、语言模型、Transformer 与现代 NLP 技术。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/yandexdataschool/nlp_course?style=social)
- [Large Language Model Notebooks Course](https://github.com/peremartra/Large-Language-Model-Notebooks-Course) - 关于大语言模型及其应用的实践动手课程。涵盖聊天机器人、代码生成、OpenAI API、Hugging Face、向量数据库、LangChain、微调、PEFT、LoRA、QLoRA。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/peremartra/Large-Language-Model-Notebooks-Course?style=social)
- [Transformers Tutorials (Niels Rogge)](https://github.com/NielsRogge/Transformers-Tutorials) - 使用 Hugging Face Transformers 库进行 NLP、视觉与多模态任务的全面教程与演示。 ![GitHub stars](https://img.shields.io/github/stars/NielsRogge/Transformers-Tutorials?style=social)
- [AI Engineering Hub](https://github.com/patchy631/ai-engineering-hub) - 93+ 个可直接投产的项目，附带关于 LLM、RAG 与真实场景 AI agent 应用的深入教程。覆盖从入门到高级各技能水平的全面资源。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/patchy631/ai-engineering-hub?style=social)
- [Complete Agentic AI Engineering Course](https://github.com/ed-donner/agents) - 6 周的 agent 式 AI 全面课程，涵盖自主 agent、多 agent 系统与实用的 agent 开发。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/ed-donner/agents?style=social)
- [OpenMAIC](https://github.com/THU-MAIC/OpenMAIC) - MIT 许可的多 agent 平台，用于交互式学习与课程生成。 ![GitHub stars](https://img.shields.io/github/stars/THU-MAIC/OpenMAIC?style=social)

#### 入门项目与示例

- [Awesome LLM Apps](https://github.com/Shubhamsaboo/awesome-llm-apps) - 一批开箱即用的人工智能模板与 agent 定制脚本，包含 100 多个检索增强生成（RAG）与智能 agent 蓝图。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/Shubhamsaboo/awesome-llm-apps?style=social)
- [Claude Cookbooks](https://github.com/anthropics/claude-cookbooks) - 官方配方与笔记本合集，演示如何用 Claude 实现工具使用、提示缓存、结构化输出与 agent 式工作流。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/anthropics/claude-cookbooks?style=social)
- [Hugging Face Transformers Notebooks](https://github.com/huggingface/notebooks) - 在 Colab 中运行 Transformers、Datasets 等。 ![GitHub stars](https://img.shields.io/github/stars/huggingface/notebooks?style=social)
- [TensorFlow Tutorials](https://github.com/tensorflow/docs) - 面向从初学者到高级用户的官方指南。 ![GitHub stars](https://img.shields.io/github/stars/tensorflow/docs?style=social)

#### 精选资源列表

- [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning) - 按语言组织的机器学习框架、库与软件的权威精选列表。全面覆盖 Python、C++、Java、JavaScript 等语言的 ML 生态。CC0-1.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/josephmisiti/awesome-machine-learning?style=social)
- [Awesome Artificial Intelligence](https://github.com/owainlewis/awesome-artificial-intelligence) - 为开发者与研究者精选的人工智能课程、书籍、视频讲座与论文列表。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/owainlewis/awesome-artificial-intelligence?style=social)
- [Andrej Karpathy Skills](https://github.com/forrestchang/andrej-karpathy-skills) - 一个用于改善 Claude Code 行为的 CLAUDE.md 文件，内容源自 Andrej Karpathy 对 LLM 编程陷阱的观察。原则：先思考再写代码、简单优先、外科手术式改动、目标驱动执行。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/forrestchang/andrej-karpathy-skills?style=social)
- [Awesome DESIGN.md](https://github.com/VoltAgent/awesome-design-md) - 精选的 DESIGN.md 文件合集，代表主流设计系统，指导 AI 编码 agent 生成风格统一的 UI。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/VoltAgent/awesome-design-md?style=social)
- [Awesome Claude Skills](https://github.com/ComposioHQ/awesome-claude-skills) - 精选的 Claude Skills、插件、资源与自定义命令列表，用于扩展终端与 API 工作流。Apache 2.0 许可。 ![GitHub stars](https://img.shields.io/github/stars/ComposioHQ/awesome-claude-skills?style=social)
- [Best of Open-Source AI](https://github.com/archestack/best-of-ai) - 排名的开源 AI 应用、agent、模型服务器与基础设施工具列表，每项基于公开 GitHub 数据评分，附书面优缺点分析，每天更新两次。MIT 许可。 ![GitHub stars](https://img.shields.io/github/stars/archestack/best-of-ai?style=social)

---

## 贡献指南

欢迎贡献。请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解具体指南。

- 仅接受 OSI 批准的许可证
- 无最低 GitHub star 数要求
- 项目需要有实用价值、相关切当前、文档完善，并有清晰的存在理由
- 保持为当前维护的列表，而非历史归档
