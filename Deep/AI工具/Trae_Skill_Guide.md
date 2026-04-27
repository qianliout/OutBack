# 深入浅出 Trae (Claude) Skill：编写、使用与维护指南

## 1. 什么是 Skill？
Skill 是赋予 AI (如 Claude/Trae) 特定领域知识、工作流和工具集的一种“入职指南”。它能将通用的 AI 助手转化为某个垂直领域的专家，让其在执行特定任务时更加一致、可靠。

## 2. 怎么写 Skill？


### 2.1 核心结构
一个标准的 Skill 项目通常包含以下结构：
```text
skill-name/
├── SKILL.md           # 必填：定义工作流、指令、触发条件（不超过 500 行）
├── scripts/           # 可选：确定性、可重复的操作脚本（如 Python/Bash 脚本）
├── references/        # 可选：按需加载的领域知识库（如架构指南、规范文档）
└── assets/            # 可选：输出时使用的资源文件（如模板、图片），不加载到上下文中
```

### 2.2 编写步骤 (Workflow)
1. **明确目标 (Understand)**：确定这个 Skill 解决什么具体问题？列出至少 3 个具体的使用场景，并确定触发该 Skill 的关键词。
2. **规划架构 (Plan)**：梳理哪些是固定脚本，哪些是领域知识（放入 references）。
3. **编写描述 (Description)**：在 `SKILL.md` 头部定义元数据。描述（Description）决定了何时自动触发该 Skill。
4. **制定铁律 (Iron Law)**：在文件开头写下模型最容易犯的错误，并设定不可逾越的规则。
5. **设计工作流检查单 (Checklist)**：用具体的步骤引导 AI（例如：Step 1, Step 2），对于破坏性操作添加“确认门”（要求先询问用户）。
6. **提供反模式 (Anti-Patterns)**：明确告诉 AI“不要做什么”。
7. **交付前检查 (Pre-Delivery Checklist)**：列出具体的、可验证的输出检查项。

---

## 3. 怎么使用 Skill？
在 Trae 或 Claude Code 中使用 Skill 非常简单：
- **自动触发**：当你在对话中输入与 Skill `description` 中定义的关键词（Trigger keywords）高度匹配的话语时，AI 会自动加载该 Skill。
- **手动调用**：你可以直接通过命令 `/skill <skill-name>` 或者在提示词中明确说“请使用 XXX skill 来完成...”。

---

## 4. 怎么维护 Skill？
1. **控制大小**：`SKILL.md` 必须保持精简（< 500 行），任何超出的长篇领域知识都应该剥离到 `references/` 目录下，并配置按需加载。
2. **基于真实反馈迭代**：观察 AI 在使用 Skill 时哪里容易犯错或产生幻觉，将这些错误补充到 `SKILL.md` 的**反模式 (Anti-Patterns)** 中。
3. **保护 Token 成本**：每一行 Skill 指令都必须有其价值。如果不改善输出质量，就删掉它。

---

## 5. 示例详解

### 示例 1：约束 Markdown 格式的 Skill (markdown-formatter)

**场景**：强制 AI 在生成 Markdown 文档时遵循特定的排版规范（如中英文之间加空格、不使用复杂格式等）。

**文件：`markdown-formatter/SKILL.md`**
```markdown
---
name: markdown-formatter
description: 强制执行 Markdown 排版规范。当用户要求格式化文档、写文档、约束 markdown 格式、或触发 "format md", "markdown style" 时使用。
---

# Markdown 格式化专家 (Markdown Formatter)

**铁律 (IRON LAW)**：
- 绝对不要使用过于复杂的嵌套格式！
- 不使用嵌套的 `-` 或 `*` 层级（最多一层）！

## 工作流 (Workflow)

请严格按照以下步骤执行：

- [ ] **Step 1: 分析输入**
  - 阅读用户提供的文本。
  - 检查是否存在违反规范的嵌套列表或复杂的 Markdown 格式。
- [ ] **Step 2: 执行格式化规则**
  - 中英文之间必须增加一个半角空格。
  - 数字与中文之间必须增加一个半角空格。
  - 移除所有多层嵌套的列表，展平为单层列表。
  - 标题结构保持清晰，最多使用到 `###` 三级标题。
- [ ] **Step 3: 交付前检查 (Pre-Delivery Checklist)**
  - 确认是否已展平所有列表？(是/否)
  - 确认中英文之间是否都有空格？(是/否)

## 反模式 (Anti-Patterns - 绝对不要做)
- ❌ 不要写出这样的列表：
  - 第一层
    - 第二层
      - 第三层
- ❌ 不要忽略标点符号前后的空格规范。
```

**详细解释**：
1. **触发器 (description)**：明确列出了用户可能说的词，以便自动唤醒。
2. **铁律 (Iron Law)**：结合了你提供的 markdown 规则（不使用复杂格式，不嵌套），将其设为最高优先级。
3. **工作流 (Workflow)**：用 Checklist 的形式让 AI 一步步执行，防止遗漏。
4. **反模式 (Anti-Patterns)**：通过直接给出错误的示例（❌），让 AI 明确知道“什么叫嵌套过深”。

---

### 示例 2：写 Golang 代码的 Skill (golang-expert)

**场景**：让 AI 写 Go 代码时，严格遵守依赖注入、显式错误处理、Effective Go 等高级规范。

**文件：`golang-expert/SKILL.md`**
```markdown
---
name: golang-expert
description: 编写高质量、生产级别的 Golang 代码。当用户要求写 Go 代码、Golang 架构、Go 接口实现或触发 "写 golang", "go 代码" 时使用。
---

# Golang 资深开发专家 (Golang Expert)

**铁律 (IRON LAW)**：
- 必须显式处理错误，绝不忽略 `error`（禁止使用 `_ = xxx()`）！
- 业务逻辑中绝对禁止使用 `panic`！

## 工作流 (Workflow)

请严格按照以下步骤编写或重构 Go 代码：

- [ ] **Step 1: 架构设计与依赖注入**
  - 所有模块之间通过接口解耦。
  - 使用构造函数模式（如 `NewService(...)`）进行依赖注入。
  - 配置必须通过结构体传入，绝不硬编码密钥、地址或端口。
- [ ] **Step 2: 核心代码实现**
  - 遵循 `gofmt` 和 `Effective Go` 规范。
  - 命名清晰：使用 `camelCase`，利用首字母大小写控制可见性。
  - 优先使用 Go 标准库，避免过度依赖第三方包。
  - 所有可能阻塞的函数，第一个参数必须接受 `context.Context`，以便透传链路追踪 (trace ID)。
- [ ] **Step 3: 日志与注释规范**
  - **所有公开 API 必须带 godoc 格式的中文注释。**
  - **代码内的逻辑注释必须使用中文。**
  - **日志 (Log) 输出内容必须使用英文**，拒绝冗余，但必须包含核心排障信息（关键参数、错误堆栈）。
- [ ] **Step 4: 交付前检查 (Pre-Delivery Checklist)**
  - [ ] 检查所有 `error` 是否都被妥善处理？
  - [ ] 检查是否有硬编码的配置信息？
  - [ ] 检查日志输出是否全为英文？
  - [ ] 检查公开函数是否都有中文注释？

## 反模式 (Anti-Patterns - 绝对不要做)
- ❌ 忽略错误：`f, _ := os.Open("file")`
- ❌ 滥用协程：在 `for` 循环内无限制启动 `goroutine` 而不使用 WaitGroup 或 Channel 控制并发。
- ❌ 复杂初始化：使用 `init()` 函数做复杂的初始化操作。
- ❌ 全局变量：滥用全局变量，而不是通过依赖注入传递状态。
```

**详细解释**：
1. **融入记忆与规则**：该 Skill 完美整合了你的核心开发规则，包括“日志必须英文，注释必须中文”、“依赖注入”、“显式错误处理”、“context 传递”等。
2. **工作流分离**：将“架构设计”、“代码实现”、“日志规范”分为了三个独立的步骤，强制 AI 在写代码前先考虑架构，保证了“高标准架构”。
3. **交付前检查 (Pre-Delivery Checklist)**：通过一系列的反问句，强制 AI 在输出代码前进行自我审查。
4. **反模式防线**：直接列出常见的 Golang 坏味道（如 `init()` 滥用、忽略 `error`），在 AI 生成代码时直接阻断这些坏习惯。
