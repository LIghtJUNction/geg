<div align="center">

# geg
### Geg Embeds Gemma

**Search by meaning. Keep your code on your machine.**  
**像 grep 一样使用，用语义寻找代码与文本。**

[![crates.io](https://img.shields.io/crates/v/geg?style=flat-square&color=537a5a)](https://crates.io/crates/geg)
[![License: MIT](https://img.shields.io/badge/license-MIT-537a5a?style=flat-square)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2024-b7410e?style=flat-square)](https://www.rust-lang.org/)
[![Local inference](https://img.shields.io/badge/inference-100%25_local-537a5a?style=flat-square)](#模型与离线运行)

[快速开始](#快速开始) · [大型仓库索引](#大型仓库显式索引) · [安装 Agent 技能](#安装-agent-技能) · [grep 兼容边界](#grep-兼容边界) · [English](#english)

</div>

---

```sh
# 不必先知道函数叫什么
geg --code -nH --score 'where are failed requests retried?' src/

# 从描述找到实现，也可以直接搜索中文
geg --code --json '校验密码并拒绝未授权请求' src/

# 文本管道同样适用
cat notes.txt | geg '旅行计划'
```

**本地推理，没有 API key，没有外部服务。** 使用 EmbeddingGemma 2 的 **270M 文本骨干**，同时理解自然语言与代码；不加载视觉和音频编码器。Rust CLI 内置 CPU 推理引擎与 SQLite，不需要 Python、PyTorch、Ollama 或单独的推理服务器。

## 为什么用 geg？

- **按含义搜索** — 不知道符号名称，也能定位相关代码；`--code` 使用官方代码检索指令。
- **重复调用更省事** — 内容寻址向量缓存自动复用，新增、修改内容才重新推理；缓存全命中时不加载推理模型。
- **大仓库有显式索引** — 建好之后直接扫描磁盘向量，不再遍历和读取整个源码树。
- **内存有边界** — 流式分块、磁盘缓存、固定 SQLite 页缓存、有限 top-k；不把整仓库文本和全部向量塞进 RAM。
- **适合 agent** — JSONL 输出、常用 grep 风格参数、配套技能，以及 0 / 1 / 2 退出码。
- **不污染项目** — 默认写系统用户缓存，不强制创建 `.geg/`。

> **不是 GNU grep 的替代品。** geg 默认返回相似度最高的 10 个文本块，不是所有正则匹配行。精确符号、正则、完整引用列表请用 `rg` 或 `grep`。运行内存不等于 168 MiB 的模型文件大小；冷推理与缓存命中的资源开销也不同。

## 快速开始

### 1 · 安装命令

```sh
cargo install geg --locked
```

源码编译需要 Rust **1.88+**、CMake、C/C++ 编译器及 libclang（用于原生绑定生成）。锁文件用于保持已验证的依赖组合。Linux 示例：

```sh
# Debian / Ubuntu
sudo apt-get install build-essential cmake clang libclang-dev

# Arch Linux
sudo pacman -S --needed base-devel cmake clang
```

macOS 使用 Xcode Command Line Tools、CMake 与 LLVM；Windows 使用 MSVC C++ Build Tools、CMake、LLVM，必要时设置 `LIBCLANG_PATH`。项目面向 Linux、macOS 和 Windows；本版的实机验证范围见下文，不把未测试的平台声称为已验证。

### 2 · 准备本地模型

推荐 [unsloth/embeddinggemma-2-GGUF](https://huggingface.co/unsloth/embeddinggemma-2-GGUF) 的 **`embeddinggemma-2-UD-Q4_K_XL.gguf`**，约 **167.5 MiB**。只需要文本权重，**不要下载 `mmproj`**。

```sh
mkdir -p ~/.cache/geg/models
curl -L --fail --retry 3 \
  'https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/main/embeddinggemma-2-UD-Q4_K_XL.gguf' \
  -o ~/.cache/geg/models/embeddinggemma-2-UD-Q4_K_XL.gguf

export GEG_MODEL="$HOME/.cache/geg/models/embeddinggemma-2-UD-Q4_K_XL.gguf"
```

这是**用户显式进行的一次性下载**。geg 本身不自动下载模型，不调用外部网络 API。也可以在联网机器下载后离线拷贝。模型可放在任意目录，以 `--model /path/model.gguf` 或 `GEG_MODEL` 指定；macOS / Windows 尤其推荐显式设置路径。

PowerShell 示例：

```powershell
$env:GEG_MODEL = 'C:\Models\embeddinggemma-2-UD-Q4_K_XL.gguf'
geg --code -nH 'validate a password' .\src\
```

### 3 · 搜索

```sh
geg -nH --score --chunk-lines 1 'password security' notes.txt
geg --json --code -k 5 'resource cleanup on failure' src/
geg -e 'validate a login' -e 'check a password' src/
geg -f queries.txt src/                     # 每行一个语义查询，OR 匹配
geg -r --code 'retry a request'              # 未指定路径时递归搜索当前目录
```

JSONL 每个匹配对象含 `path`、起始 `line`（从 1 开始）、`score`、`text`。例如一次真实模型检索：

```json
{"path":"/tmp/geg-demo.txt","line":3,"score":0.8321745,"text":"Passwords are hashed using argon2 before storage.\n"}
```

相似度不是概率，阈值需要根据你的语料调整：

```sh
geg --threshold 0.75 --code 'authentication' src/
geg --threshold -1 -k 5 'authentication' src/   # 查看前五个候选，不按阈值过滤
```

## 大型仓库：显式索引

普通搜索每次仍会读取文件以检查内容。反复检索大型仓库时，建立独立的持久化索引：

```sh
# 默认保存在系统缓存目录；成功后输出 JSON，包含 index 路径和统计
geg index src/ docs/

# 或自己指定位置，不要求 .geg/ 文件夹
geg index src/ docs/ --output /private/path/project.sqlite3

# 使用打印的索引路径；查询不再读取源码
geg query 'retry failed network requests' \
  --index /private/path/project.sqlite3 --code --json -k 5

# 源码有变化后，重新执行同一个 index 命令
geg index src/ docs/ --output /private/path/project.sqlite3
```

- **增量推理**：重建会重新扫描输入并更新整个索引快照，但复用未变文本块的向量，不重复做模型推理。不是文件监听器，也不是只读变更文件的增量扫描。
- **一致性**：成功重建后移除已删文件；失败时事务回滚，保留上一份完整索引。查询读取同一个已提交快照。
- **格式校验**：模型内容哈希、向量维度及 `-i` 设置必须一致；`--code` 仅影响查询指令，可在查询时切换。
- **空间换时间**：索引保存向量、源码片段和绝对源文件路径。新建数据库在 Unix 上使用 `0600` 权限；已有文件权限不会被覆盖。仍请放在私有目录，不要上传或提交到 Git；原文变更后，旧索引仍可能返回旧片段。
- **不是 ANN**：索引查询目前是流式的精确余弦扫描，复杂度 O(文本块数量 × 维度)，不是近似最近邻数据库。百万级以上请先在目标机器测量。

普通搜索扫描到 **100 个文件**时，自动在 **stderr** 给出一次建索引建议，不污染 stdout/JSONL：

```sh
geg --no-index-hint 'query' src/  # 关闭建议
```

## 缓存与重复调用

默认向量维度 **256**；可选 `--dimensions 128|256|512|768`，截断后重新 L2 归一化。256 维每条向量的原始数据为 1 KiB（不含 SQLite 索引及其他开销）。

自动缓存的键包含：**模型文件内容哈希 + 推理格式版本 + 维度 + 完整任务指令与文本**。查询向量也会缓存，相同内容的不同文件可共享向量。

- Linux：`~/.cache/geg/`（遵循 `XDG_CACHE_HOME`）。
- macOS：`~/Library/Caches/geg/`。
- Windows：用户的 Local AppData 下的 geg 缓存目录。
- `--cache /path/vectors.sqlite3` / `GEG_CACHE`：指定自动缓存数据库。
- `--no-cache`：关闭自动向量缓存，不影响显式索引本身。

每个新进程仍会完整哈希模型一次，以避免替换权重后误用旧向量。缓存没有自动配额或过期清理；更换模型或频繁修改仓库会增加磁盘占用。缓存是可重建派生数据，不是备份；停止 geg 后，可删除相应缓存/索引数据库及其 SQLite sidecar 文件以回收空间。

### 常驻交互模式

很多查询连续发生时，可保留同一个进程，避免重新加载模型、词表和计算上下文：

```sh
printf 'authentication\nretry logic\n' | geg --interactive --code src/
```

每行查询对应一个立即刷新的 JSON 响应，包含 `query`、`hits`、`counts`、`stats`。`stats.cache_hits` / `stats.embedded` 可以检查是否发生了重复推理。单次查询失败会返回带 `error` 的 JSON，后续查询继续处理。

这是**前台 stdin 协议**，不是监听端口的后台服务；每个交互查询仍扫描指定文件。显式索引查询目前使用 `geg query`，不与 `--interactive` 合并。

## grep 兼容边界

**参数长得熟悉，不代表匹配语义完全相同。** 支持：

- `-e / --regexp`、`-f / --file`：提供一个或多个**语义查询**，并非正则。
- `-i / --ignore-case`：嵌入前将查询和文本都转为小写。
- `-v / --invert-match`：选中低于相似度阈值的块。
- `-r / -R`：递归；普通扫描中 `-R` 跟随符号链接，索引构建不跟随。
- `-n`：行号；`-H`：文件名；`-h`：不显示文件名。
- `-c`：每个文件的选中行数，在 top-k 截断**之前**统计。
- `-l / -L`：仅列出有/无选中行的文件。
- `-m NUM`：每个文件最多选中 NUM 行；多行块会在行边界截短输出。
- `-q / --quiet / --silent`：不输出常规结果；仍计算退出状态，不保证推理提前终止。
- `-s / --no-messages`：不显示错误信息；阈值仅使用 `--threshold`，不占用 `-s`。
- `--label`、`--line-buffered`、`--help`、`-V / --version`。

主要差异：默认返回 **top 10 块**，按相似度排序；默认块为 **8 行 / 最多 1024 UTF-8 字节**，`--chunk-lines 1` 更接近逐行语义过滤。目录无需 `-r` 也会遍历，默认遵守 `.gitignore` / `.ignore` 并跳过隐藏项。普通输出转义控制字符以避免终端转义注入；需要原始内容请读取 JSON 的 `text` 字段。

尚不支持 `-E/-F/-G/-P`、`-w/-x/-o`、`-A/-B/-C`、NUL 模式、字节偏移、glob 过滤或 GNU grep 的全部二进制/设备处理策略。不支持的参数会报错，而非被静默忽略。查找字面上的 `index` / `query` / `skill` 时用 `--`，避免被解析成子命令。

**退出码**：普通搜索 / 索引查询：`0` 有选中结果，`1` 没有结果，`2` 错误。`-L` 根据是否列出无匹配文件决定 0/1。建索引、导出技能成功返回 `0`；交互模式的单条查询错误在 JSON 中报告，协议或 I/O 错误返回 `2`。

## 安装 Agent 技能

仓库自带 [`skills/geg/SKILL.md`](skills/geg/SKILL.md)，命令行也内嵌了相同内容。技能教 agent 正确选用语义搜索、建立和更新索引、读取 JSON、核对源文件，并避免把 geg 当作正则工具。

### 安装到 Pi

```sh
mkdir -p ~/.pi/agent/skills/geg
geg skill > ~/.pi/agent/skills/geg/SKILL.md
```

### 安装到其他 Agent

将相同的 `SKILL.md` 放入该工具支持的技能目录。例如：

```sh
# 支持 ~/.agents/skills 的 agent
mkdir -p ~/.agents/skills/geg
geg skill > ~/.agents/skills/geg/SKILL.md

# Claude Code
mkdir -p ~/.claude/skills/geg
geg skill > ~/.claude/skills/geg/SKILL.md

# Cursor
mkdir -p ~/.cursor/skills/geg
geg skill > ~/.cursor/skills/geg/SKILL.md
```

这些目录是**备选安装位置**，不必全部执行。也可以从仓库复制整个 `skills/geg/` 目录。Windows PowerShell 示例：

```powershell
New-Item -ItemType Directory -Force "$HOME\.agents\skills\geg" | Out-Null
geg skill | Set-Content -Encoding utf8 "$HOME\.agents\skills\geg\SKILL.md"
```

安装后重新打开或按对应 agent 的方式刷新技能；更新 geg 后再次导出即可更新技能（会覆盖旧文件）。技能安装不需要模型，也不会访问网络。

可直接告诉 agent：

> 使用 geg 在当前仓库中查找“失败请求的重试逻辑”。如果仓库较大且要反复搜索，先建立索引；用 JSON 读取候选，再打开源文件核实。

## 模型与离线运行

本工具针对 [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) 的文本路径，而不是上一代 `embeddinggemma-300m`。官方 740M 多模态模型由 270M 文本、170M 视觉和 300M 音频部分组成；geg 只加载 GGUF 文本骨干。

使用 mean pooling，以及官方非对称检索前缀：

```text
查询：task: search result | query: {query}
代码：task: code retrieval | query: {query}
文档：title: none | text: {content}
```

当前 CPU 后端使用 `llama-cpp-4 0.9`，原生代码静态链接，关闭 OpenMP、GPU、多模态和动态后端等默认可选特性。模型通过 mmap 加载；**mmap 并不等于没有 RAM 开销**，词表、权重页和计算图仍占内存。为了限制工作集，单次推理窗口是 **512 tokens**，并非模型完整的 8192-token 上限。超长查询/文本会明确报错，不会静默截断；超 64 KiB 的单行、无效 UTF-8、stdin 中的 NUL 同样报错，磁盘文件开头看似二进制时跳过。

模型权重独立于 CLI 分发，权重与上游依赖遵循各自许可证。单文件 CLI 不需要另带 llama/SQLite 动态库，但仍依赖目标平台的标准系统运行库；不能把一个 Linux 二进制直接搬到 macOS 或 Windows。

## 实测：缓存命中与冷推理分开看

在 Linux x86_64、Intel i9-12900HX、8 个推理线程上，使用推荐 Q4 GGUF、256 维向量测试了 **100 个小文件 / 300 行 / 100 个块**：

- 发布构建二进制约 **9.2 MiB**；模型另计约 **167.5 MiB**。
- 空向量缓存的首次搜索：**64.53 秒**，峰值 RSS **294.3 MiB**。
- 相同查询的缓存命中搜索：5 次中位数 **0.206 秒**，峰值 RSS 最高 **16.1 MiB**。
- 文档向量已缓存时建索引：**0.117 秒**；再次重建 **0.134 秒**，100 个块均复用缓存、0 次推理。
- 查询向量也已缓存时的索引检索：5 次中位数 **0.131 秒**，峰值 RSS 最高 **16.2 MiB**。

“空缓存”指 geg 的向量缓存为空，**不是**清空操作系统文件页缓存。这是小型合成语料测试，不代表大型真实仓库，也不代表新查询无需推理。首次建库仍有明显 CPU 推理成本；缓存和索引主要优化反复使用的成本。另对 128 / 256 / 512 / 768 四种维度运行真实模型检索，均正确把密码哈希文本排在无关猫咪文本之前；这只是 smoke test，不是标准检索质量评测。

可在自己的机器复现：

```sh
python3 scripts/benchmark.py --binary target/release/geg --model "$GEG_MODEL"
```

目前完成实机验证的是 **Linux x86_64**。macOS / Windows 使用相应平台的源码构建，尚未在本项目中完成实机验证；此处不承诺相同速度与内存数据。

## 本地开发与验证

```sh
cargo fmt --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --locked
cargo doc --no-deps --locked

# 真实模型测试：四种向量维度的语义排序
GEG_TEST_MODEL=/path/embeddinggemma-2-UD-Q4_K_XL.gguf \
  cargo test --locked --test cli real_model_semantic_smoke -- --ignored

cargo install --path . --locked
```

普通测试不下载模型：用明确标注的预置缓存向量测试 CLI、管道、退出码、增量索引、回滚、源码移除后的索引查询和大量文件提示；真实 GGUF 另作独立实测，不把预置向量称作模型效果测试。

## English

**geg** is an offline semantic grep for source code and text, powered by the **270M text backbone of EmbeddingGemma 2**. It is a Rust CLI with an embedded CPU llama.cpp backend and bundled SQLite, not an HTTP API client.

```sh
cargo install geg --locked
export GEG_MODEL=/path/embeddinggemma-2-UD-Q4_K_XL.gguf
geg --code --json 'retry failed requests' src/
geg index src/ --output /private/path/repo.sqlite3
geg query 'validate a password' --index /private/path/repo.sqlite3 --code --json
```

The model must be provisioned separately. No network requests are made at runtime. Ordinary searches automatically cache query/document embeddings but still read files. Explicit indexes avoid rereading source files; rerun `index` after changes. Indexes contain source excerpts and should be kept private. Queries perform exact, streamed cosine scans, not ANN retrieval.

The default output is the **top 10 chunks**, ranked by similarity; this is **not full GNU grep compatibility**. Common grep-style flags are supported, but query patterns are semantic instructions, not regex. Use `rg` or `grep` for exact matching. Set `--chunk-lines 1` for line-level chunks and `--threshold -1` to inspect top candidates without threshold filtering.

Install the bundled agent skill with `geg skill > <agent-skill-directory>/geg/SKILL.md` after creating the directory, then reload your agent's skills. In Pi, use `~/.pi/agent/skills/geg/SKILL.md`. Exporting the skill requires neither weights nor network access.

## License

[MIT](LICENSE). Model weights and native dependencies retain their own licenses.
