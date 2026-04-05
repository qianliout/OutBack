---
title: "2026 年，万物皆 Coding Agent 的平台工程新范式（A2A / ACP / MCP / Skill）"
source: "https://mp.weixin.qq.com/s/WFRT5EHN-ir-nrEVIT5VNw"
author:
  - "[[Phodal]]"
published:
created: 2026-03-08
description: "过去几年，DevOps 工具链越来越复杂：CI/CD、Issue、测试、文档、PR、IDE……每一个环节都像一个独立角色。如今，AI 正在重塑这一切，让工具不再只是被动执行脚本，而是能自主行动的 Coding Agent。"
tags:
  - "clippings"
---
原创 Phodal *2026年3月3日 12:03*

PS：就着最近的研究，我想大概写一篇今年的趋势

过去几年，DevOps 工具链越来越复杂：CI/CD、Issue、测试、文档、PR、IDE……每一个环节都像一个独立角色。如今，AI 正在重塑这一切，让工具不再只是被动执行脚本，而是能自主行动的 **Coding Agent** 。

![图片](https://mmbiz.qpic.cn/mmbiz_png/8EGPLjlFPVK0sVCp8ibNAGMPfQGy8aEibAK7tawSd4DjXXQ7KwzmibUiaQv0YFlC2lUiclAQWGPTtGUUZxOXZg3fMgDDo2wuZVe1iciaYa99DdDTqQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

借助 **A2A、ACP、MCP 和 Skill** ，我们可以把散落在工具链里的能力统一标准化、可调度、可追踪，让平台工程成为真正的 **Agent 网络** 。每个 Agent 各司其职、互相协作，形成可演进、可治理的整体系统。

![图片](https://mmbiz.qpic.cn/mmbiz_png/8EGPLjlFPVJicI5ZOhrXuhIEoKulZsvkCdfeflWYWuR8uo2cntNEZlucsiaul0PaHp6wKnmuFoEgr6yfycHZgC3C0gXGx6etTzQGDZhKCb9ibk/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=1)

这不是单体 AI 助手的简单叠加，而是一场 **平台工程的新范式升级** ——2026 年，智能 Agent 的网络化协作将成为企业 DevOps 的核心趋势。

## 一、平台工程的碎片化困境与演进

## 1.1 Agent 孤岛化趋势

根据 2025 年 Platform Engineering Survey，78% 的企业在过去 18 个月内引入了至少 3 种不同 AI Agent，但仅 23% 实现了工具间有效协作。DevOps 工具链的每个环节都在长出自己的 Agent：

![图片](https://mmbiz.qpic.cn/mmbiz_png/8EGPLjlFPVLr80gdcG3ibJKO2fQlXvaITKJECMqbkibzwibQ9KMXIiaeYuCGSpsx65hMKlYfaoK579nOKvm1N6OXAicgdC418ibe5tl8pibuRiaQv1U/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2)

| 平台/厂商 | Agent | 能力范围 | 市场占有率 |
| --- | --- | --- | --- |
| **GitLab** | Duo Agent Platform | CI/CD 编排、代码审查、安全扫描、Issue 自动处理 | 34% |
| **GitHub** | Copilot Coding Agent | PR 自动创建、代码实现、Issue 修复 | 67% |
| **Cursor/Claude Code** | 编码 Agent | 编辑器内全栈编码、终端自主编程 | 28% |
| **Kubernetes** | K8sGPT/Copilot | 集群诊断、资源优化、故障自愈 | 19% |
| **DataDog/New Relic** | AIOps Agent | 监控告警、根因分析、性能调优 | 42% |
| **PagerDuty/Splunk** | 运维 Agent | 事件响应、自动化运维、容量规划 | 25% |
| **Vercel/Netlify** | 部署 Agent | 自动部署、预览环境、回滚 | 31% |

虽然每个 Agent 都在“帮你搞定一切”，但现实是：没有一个 Agent 能覆盖全流程——CI/CD、代码、部署、监控、业务逻辑各自孤立。企业因此付出 **成本激增、效率悖论、技能债务** 的代价：

- • 平均每个 Agent $50-200/月/用户，40% 预算浪费
- • 工具切换和上下文丢失延长整体交付周期 25%
- • 开发者需掌握 6-8 种交互模式，认知负荷高

**问题本质** ：当平台工程每个环节都是 Agent，谁来统一编排它们？

## 1.2 从需求到交付的 AI 演进阶段

![图片](https://mmbiz.qpic.cn/mmbiz_png/8EGPLjlFPVLlK3DY7BqISuyJWKf6dqZ4lZFqCBychS2KsSlsV1iaMUZsTbyAbO9iaxXJklcNXppWmicrX7va4WqPTgtGgUGLrTWsDPmOd8BRNw/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=3)

平台工程 AI 化不是一步到位，而是经历了 0 → 4 阶段的演进：

| 阶段 | 特征 | 典型工具/Agent | 核心问题 |
| --- | --- | --- | --- |
| **0：需求分析与规划（传统阶段）** | 人工收集需求、评估工期、跨团队沟通 | Jira、Confluence、架构文档 | 需求变更频繁（40%）、工期预估误差大（50-200%）、沟通链条长 |
| **1：脚本自动化（2015-2020）** | 人写脚本，机器按序执行 | Jenkins Pipeline、GitLab CI、Terraform | YAML 地狱、异常处理差、新场景需重写脚本 |
| **2：单点 AI 助手（2022-2023）** | AI 提供建议，人工执行 | Copilot、ChatGPT、Tabnine | AI 仅建议、上下文需复制粘贴、无法跟踪任务 |
| **3：自主 Coding Agent（2024-2025）** | AI 可读写文件、自主决策、提交代码 | Cursor/Claude Code、Devin、OpenCode | 单 Agent 能力有限、多 Agent 不互通、缺乏协作机制 |
| **4：Agent 网络/联邦（2025-）** | 多 Agent 分工协作，统一协议层，编排引擎调度全局 | Google A2A 生态、Agent Teams、Intent | 协议标准化、任务分解调度、容错与同步 |

我们正处在 **阶段 3 → 4 的过渡期** 。协议是基础设施，编排是核心能力。谁先建立有效的 Agent 编排体系，谁就能在下一代平台工程中占据先机。

## 三、协议：连接 Agent 孤岛的三层标准

![图片](https://mmbiz.qpic.cn/mmbiz_png/8EGPLjlFPVL0ZzgZ0YP7AAOFjU1KlzbyBgdWfhkERSxQW0vBZ9YYT3nbTwpKiaz9FXGH43vBFhcNibkCjKAJdZPOBvErjNqBZSLhgG6O24H50/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=4)

要让 Agent 互通，需要三层协议解决不同层次的问题：

| 协议 | 解决问题 | 类比 | 主导 |
| --- | --- | --- | --- |
| **A2A (Agent-to-Agent Protocol)** | 跨平台 Agent 发现与协作 | 互联网 HTTP — Agent 间怎么对话？ | Google (2025.04)、Microsoft、Anthropic、LangChain 支持 |
| **ACP (Agent Client Protocol)** | Coding Agent 进程生命周期管理 | 操作系统进程管理 — Agent 怎么启动/运行/停止？ | JetBrains + Zed (2025.10) |
| **MCP (Model Context Protocol)** | AI 获取工具与上下文 | USB 接口 — Agent 怎么插入新能力？ | Anthropic (2024.11)，Linux Foundation 托管 |

## 3.1 MCP：Agent 的能力插座

**Model Context Protocol** (MCP) 由 Anthropic 于 2024 年 11 月发布，由 Linux Foundation 托管。

**核心问题** ：AI 如何标准化地调用工具？

MCP 统一了不同 AI 客户端的工具调用接口（Claude 的 tool *use、OpenAI 的 function* calling 等），提供：

- • **Tools** ：可调用函数（createfile、runcommand、search…）
- • **Resources** ：可读取的数据源（文件系统、数据库、API 响应）
- • **Prompts** ：可复用的提示模板（代码审查、重构指南）

**价值** ：一个 MCP Server，可被所有支持 MCP 的 Agent 使用，实现工具共享。

**编排视角** ：

MCP 是能力插座，可注入协作工具，例如：

- • create\_agent — 创建子 Agent
- • delegate\_task\_to\_agent — 委派任务
- • send\_message\_to\_agent — Agent 间通信
- • report\_to\_parent — 向上级汇报

这些操作本身就是 MCP Tools，任何支持 MCP 的 Agent 都可调用。

## 3.2 ACP：Agent 的进程管理器

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/8EGPLjlFPVKZaXGJ6icyz9ZVZWm72WQ451xugpcIUFJa6bicHIh3Qjia7L7aOticVDBiaJYow7jroo3fnWib70ZbeHn3B4q6pkrMpCEoM4veFMRtY/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=5)

**Agent Client Protocol** (ACP) 由 JetBrains 和 Zed 于 2025 年 10 月发布。

**核心问题** ：IDE 如何统一管理 Coding Agent 生命周期？

ACP 定义 JSON-RPC 协议，包括：

- • initialize — 能力协商，声明支持 features
- • session/new — 创建新会话，传入初始 prompt
- • session/prompt — 发送后续消息
- • session/update — 流式推送执行进度（SSE）
- • session/cancel — 取消当前执行

**价值** ：任何 Coding Agent 可在任意 ACP 支持的 IDE 中运行，实现跨 IDE 使用。

**编排视角** ：

ACP 是进程管理器，编排引擎可通过 ACP：

1. 1\. 启动不同 Agent（Claude 做规划、OpenCode 做实现）
2. 2\. 管理会话状态，跟踪执行进度
3. 3\. 按需调度任务（强模型处理复杂任务，廉价模型处理重复任务）

## 3.3 A2A：Agent 的互联网

**Agent-to-Agent Protocol** (A2A) 由 Google 于 2025 年 4 月发布，Microsoft、Anthropic、LangChain 等支持。

**核心问题** ：跨平台 Agent 如何发现与协作？

MCP 解决工具调用，ACP 解决进程管理，但都局限在单一平台。A2A 提供跨平台能力，包括：

- • **AgentCard** — 能力声明（类似 OpenAPI）
- • **Skill** — 擅长的任务类型（code-review、test-generation）
- • **Task** — 任务状态流（submitted → working → completed）

**价值** ：Agent 可跨平台协作，编排系统能：

- • 发现 GitLab 安全 Agent
- • 委派任务给 GitHub Coding Agent
- • 接收 Vercel 部署 Agent 的完成通知

## 3.4 Skill：Agent 的能力简历

**Skill** 不是独立协议，而是 A2A 和 Coding Agent 系统的子概念，用于结构化描述 Agent 能力：

- • **A2A Skill** ：对外宣告能力，让其他 Agent 知道“我能做什么”
- • **内部 Skill** ：Agent 的角色指令与行为边界

*示例文件（ `.agents/skills/frontend-design.md` ）* ：

```
name:"Frontend Design"description:"Implements UI components following design system"tags:["frontend","react","design-system"]---## InstructionsYou are a frontend specialist.When implementing UI:1.Follow/src/design design system2.UseTypeScriptwith strict mode3.WriteStorybook stories4.Ensure accessibility (WCAG 2.1 AA)
```

**编排视角** ：Skill 是能力简历，编排引擎根据 Skill 匹配任务：

- • 前端任务 → 匹配 frontend-design Skill 的 Agent
- • 安全审查 → 匹配 security-review Skill 的 Agent
- • API 设计 → 匹配 `api-design` Skill 的 Agent 明白了，你的意思是：Workspace Agent 并不是单进程独立做任务，而是一个 **宏观协调的 ACP Agent** ，通过内置逻辑管理其它 ACP Agent（比如 Claude、OpenCode 等）完成子任务。也就是说，Workspace Agent 是“微编排器”，把 ACP Agent 当作可调度的执行单元，而不是自己执行具体代码。

## 四、两种编排模式：外部编排 vs Workspace Agent

协议让 Agent 能互通，但“能互通 ≠ 会协作”。真正的挑战是 **如何编排** ——谁来调度、谁来分配、谁来跟踪任务执行。

![图片](https://mmbiz.qpic.cn/mmbiz_png/8EGPLjlFPVKA0Kiay5OLGbbfpUaK1SQEQXtF3cqJctibyTic71VQoU6EXzWYjFChahcia6Yv16G3FKub1N5e38GJd2hhz0ibBNz54CuP8aDYNnb4/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=6)

## 4.1 两种主流模式概览

| 维度 | 外部编排（Orchestrator + ACP Agent） | Workspace Agent（原生 ACP 宏观协调） |
| --- | --- | --- |
| 核心架构 | Orchestrator + 多 ACP 子进程 | 单个 Workspace Agent 管理子 ACP Agent |
| Agent 进程 | 每个子任务独立 ACP Agent | Workspace Agent + 可调度 ACP Agent |
| 通信机制 | ACP JSON-RPC / EventBus | 内部调度 + ACP Tools |
| 状态管理 | 外部持久化（DB + EventBus） | Workspace Agent 内部管理子 Agent 状态 |
| 适用场景 | 复杂企业流程、多团队协作 | 快速原型、轻量级任务宏观协调 |
| 典型实现 | Routa Orchestrator、Agent Teams | Intent Workspace Agent |

## 4.2 外部编排（ACP 进程模式）

**核心思想** ：编排引擎在外部，Coding Agent 是被调度的执行单元。

```
用户需求："实现用户登录功能"
            ↓
       ┌───────────────┐
       │  Coordinator  │  规划、拆解（ACP #1）
       └───────┬───────┘
               ↓ create_agent / delegate
   ┌───────────┼───────────┐
   ↓           ↓           ↓
┌──────┐   ┌──────┐   ┌──────┐
│ API  │   │ 前端 │   │ 测试 │
└───┬──┘   └───┬──┘   └───┬──┘
    ↓           ↓           ↓
  Crafter     Crafter     Crafter
 (ACP #2)   (ACP #3)   (ACP #4)
    │           │           │
    └───────────┼───────────┘
                ↓ report_to_parent
         ┌───────────────┐
         │   EventBus    │  状态同步、依赖检查
         └───────────────┘
```

**工作流程** ：

1. 1\. Orchestrator 接收任务 → 启动 Coordinator ACP Agent
2. 2\. Coordinator 拆解任务 → 调用 MCP Tools 创建子任务
3. 3\. 子任务启动独立 ACP Agent（Claude、OpenCode 等）
4. 4\. 子 Agent 执行 → 调用 report\_to\_parent 上报结果
5. 5\. Orchestrator 根据依赖图触发下游任务

## 4.3 Workspace Agent（ACP 宏观协调模式）

**核心思想** ：Workspace Agent 是一个 **宏观协调者 ACP Agent** ，管理多个子 ACP Agent 来完成复杂任务，而不是自己执行具体代码。它内置“Agentic Loop”逻辑，用于任务拆解、调度、监控与依赖管理。

```
用户需求："实现用户登录功能"
            ↓
     ┌─────────────────────┐
     │   Workspace Agent   │  (ACP 宏观协调)
     └─────────┬───────────┘
               ↓ create_agent / delegate
    ┌───────────┼───────────┐
    ↓           ↓           ↓
┌──────┐   ┌──────┐   ┌──────┐
│ ACP  │   │ ACP  │   │ ACP  │
│ API  │   │ 前端 │   │ 测试 │
└──────┘   └──────┘   └──────┘
```

**工作流程** ：

1. 1\. Workspace Agent 接收任务 → 拆解子任务
2. 2\. 调度不同 ACP Agent 执行子任务
3. 3\. 通过 MCP Tools 管理 Agent 间通信、任务委派、状态上报
4. 4\. 监控子任务执行 → 汇总结果 → 输出整体执行状态

## 多 Agent 编排模式对比

| 维度 | 外部编排（ACP 进程模式） | Workspace Agent（宏观 ACP Agent） |
| --- | --- | --- |
| **架构** | Orchestrator + 多 ACP 子进程 | Workspace Agent 管理多个 ACP Agent |
| **通信** | ACP JSON-RPC / EventBus | 内部调度 + MCP Tools |
| **状态管理** | 外部持久化（DB / EventBus） | 内部统一管理子 Agent 状态 |
| **任务拆解** | Orchestrator / Coordinator 拆解 | Workspace Agent 内置拆解逻辑 |
| **优势** | Provider 可插拔，进程隔离，可审计，可扩展 | 宏观调度多 Agent，Provider 可插拔，状态统一，可灵活处理复杂任务 |
| **劣势** | 进程开销高，通信延迟，系统复杂 | 单点崩溃风险，内部逻辑复杂，长任务需合理切分 |
| **适用场景** | 多团队协作、复杂任务、状态持久化 | 快速原型、轻量任务、Workspace 宏观协调 |

## 五、实践参考：Routa 编排架构

基于上述两种模式，Routa 实现了一个多 Agent 编排平台，采用 **外部编排 + Workspace Agent 混合模式** ：

```
┌─────────────────────────────────────────────────────────────┐
│                      协议网关层                               │
│   A2A（联邦）· MCP（工具）· ACP（进程）· REST（管理）           │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│   编排引擎层：Orchestrator · EventBus · Skill Matcher        │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│   状态管理层：Tasks · Agents · Notes · Traces                │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│   ACP Process Manager：spawn / prompt / cancel / subscribe  │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│   Coding Agent 层：Claude Code · OpenCode · Gemini · Codex  │
└─────────────────────────────────────────────────────────────┘
```

**核心设计** ：

| 组件 | 职责 | 对应第四部分 |
| --- | --- | --- |
| **Orchestrator** | 任务拆解、依赖管理、Agent 调度 | 外部编排引擎 |
| **EventBus** | 事件驱动通信、 `report_to_parent` 处理 | 通信与同步 |
| **Skill Matcher** | 根据任务特征匹配 Agent | Agent 选型与调度 |
| **ACP Process Manager** | 管理 Coding Agent 生命周期 | ACP 进程模式 |
| **Workspace Agent** | 宏观协调子 Agent（可选模式） | Workspace Agent |

**角色分工** ：

| 角色 | 职责 | 边界 | 典型 Provider |
| --- | --- | --- | --- |
| **Coordinator** | 规划、拆解、汇总 | 不直接写代码 | Claude (smart) |
| **Implementor** | 按任务实现功能 | 不扩大任务范围 | OpenCode (fast) |
| **Verifier** | 验收、审查 | 不修改代码 | Claude (smart) |

**实践要点** ：

1. 1\. **协议融合** ：A2A / MCP / ACP 统一接入，外部系统通过 Webhook 或 A2A 触发任务
2. 2\. **Provider 可插拔** ：同一角色可用不同 Coding Agent，运行时可切换
3. 3\. **事件驱动** ：所有状态变更通过 EventBus 传递，支持异步协作
4. 4\. **Skill 定义** ：通过 Markdown + YAML frontmatter 定义角色指令与行为边界

## 六、总结：万物皆 Coding Agent

当每个 DevOps 环节都长出自己的 Agent，孤岛化问题显而易见：工具多、重复多、效率低。协议（MCP、ACP、A2A）打通了能力接口、进程管理和跨平台协作，为 Agent 网络铺路。

但协议只是道路， **编排才是交通系统** 。通过任务拆解、Agent 调度、事件同步和状态管理，零散的 Agent 被组织成链条：规划者规划，实现者实现，验证者验收，每个角色都有 Skill 明确能力边界。

多 Agent 编排让平台工程不再是工具的集合，而是 **可调度、可追踪、可演进的智能体网络** 。单体 AI 助手只是工具，而真正的升级，是整个网络协作——这就是新范式下的“万物皆 Coding Agent”。

## 附录：相关资源

- • **MCP 官方文档** ：https://modelcontextprotocol.io/
- • **ACP 协议说明** ：https://zed.dev/acp
- • **A2A 协议发布** ：https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/
- • **Routa 开源项目** ：https://github.com/phodal/routa-js

## 参考文献

- • \[1\] Platform Engineering Survey 2025. (2025). State of Platform Engineering Report. Platform Engineering Foundation. https://platformengineering.org/survey-2025
- • \[2\] McKinsey & Company. (2025). The Economic Impact of AI Agent Fragmentation. McKinsey Global Institute. https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/ai-agent-fragmentation-report
- • \[3\] Gartner Research. (2025). Platform Engineering Efficiency Paradox: When More Tools Mean Less Productivity. Gartner
- • Inc. https://www.gartner.com/en/documents/platform-engineering-efficiency-paradox
- • \[4\] Stack Overflow. (2025). Developer Experience Survey: The Cognitive Load of Multi-Agent Workflows. Stack Overflow Insights. https://insights.stackoverflow.com/survey/2025/developer-experience
- • \[5\] Standish Group. (2024). CHAOS Report 2024: Requirements Volatility in Agile Development. The Standish Group International. https://www.standishgroup.com/sampleresearchfiles/CHAOSReport2024.pdf
- • \[6\] IEEE Software Engineering Institute. (2024). Software Estimation Accuracy: A 20-Year Retrospective. Carnegie Mellon University. https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=software-estimation-accuracy-2024

作者提示: 个人观点，仅供参考

[阅读原文](https://mp.weixin.qq.com/s/)

继续滑动看下一个

phodal

向上滑动看下一个

剪存为飞书云文档