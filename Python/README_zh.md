<think>
</think>

# 实践应用

[![如何使用真实工具和工作流程构建、测试和部署 MCP 应用程序](../images/video-thumbnails/05.png)](https://youtu.be/vCN9-mKBDfQ)

_(点击上方图片查看本节课的视频)_

实践应用是 Model Context Protocol (MCP) 功力变得具体可见的地方。虽然理解 MCP 的理论和架构很重要，但真正的价值在于将这些概念应用于构建、测试和部署解决现实问题的解决方案。本章弥合了概念性知识与实际开发之间的差距，引导您通过将基于 MCP 的应用程序带入实际应用的过程。

无论您是在开发智能助手、将 AI 整合到业务流程中，还是构建用于数据处理的定制工具，MCP 都提供了一个灵活的基础。其语言无关的设计以及为流行编程语言提供的官方 SDK，使其对广泛开发者群体来说都是可访问的。通过利用这些 SDK，您可以快速进行原型设计、迭代和在不同平台和环境中扩展您的解决方案。

在以下章节中，您将找到实践示例、示例代码和部署策略，这些内容展示了如何在 C#、Java with Spring、TypeScript、JavaScript 和 Python 中实现 MCP。您还将学习如何调试和测试您的 MCP 服务器，管理 API，并使用 Azure 将解决方案部署到云。这些实际资源旨在加速您的学习，并帮助您自信地构建健壮、可投入生产的 MCP 应用程序。<think>
</think>

## 概述

本课重点讲解在多种编程语言中实现MCP的实际应用。我们将探讨如何在C#、Java（Spring）、TypeScript、JavaScript和Python中使用MCP SDK来构建健壮的应用程序，调试和测试MCP服务器，并创建可重用的资源、提示和工具。<think>
</think>

## 学习目标

通过本节课的学习，你将能够：

- 使用各种编程语言的官方 SDK 实现 MCP 解决方案
- 系统化地调试和测试 MCP 服务器
- 创建和使用服务器功能（资源、提示和工具）
- 为复杂任务设计高效的 MCP 工作流程
- 优化 MCP 实现以提高性能和可靠性<think>
</think>

## 官方 SDK 资源

Model Context 协议为多种语言提供了官方 SDK：

- [C# SDK](https://github.com/modelcontextprotocol/csharp-sdk)
- [Java with Spring SDK](https://github.com/modelcontextprotocol/java-sdk) **注意：** 需要依赖 [Project Reactor](https://projectreactor.io)。（参见 [讨论问题 246](https://github.com/orgs/modelcontextprotocol/discussions/246)。）
- [TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Kotlin SDK](https://github.com/modelcontextprotocol/kotlin-sdk)<think>
</think>

## 使用 MCP SDK

本节提供了在多种编程语言中实现 MCP 的实际示例。您可以在 `samples` 目录中找到按语言组织的示例代码。<think>
</think>

### 可用示例

该仓库包含以下语言的[示例实现](./samples/)：

- [C#](./samples/csharp/README.md)
- [Java with Spring](./samples/java/containerapp/README.md)
- [TypeScript](./samples/typescript/README.md)
- [JavaScript](./samples/javascript/README.md)
- [Python](./samples/python/README.md)

每个示例展示了特定语言和生态系统中的关键MCP概念及实现模式。<think>
好的，用户让我翻译一段技术文档的内容，而且要保留Markdown格式。首先，我需要仔细阅读用户提供的英文内容，确保准确理解每个部分。用户特别强调要严格保留格式，比如标题、列表、粗斜体等，所以我要注意不要改变任何结构。

接下来，我需要处理代码块和图表块，这些部分用户要求不要翻译，直接保留原样。例如，如果原文中有代码块，我需要检查是否有反引号，并确保在翻译后仍然存在。同时，内联代码也要保持不变，比如用反引号括起来的部分。

然后，我会逐句翻译，确保技术术语的准确性。比如“Core Server Features”翻译成“核心服务器功能”是否合适？可能需要确认是否有标准译法。另外，用户提到的MCP servers，这里的MCP可能是一个专有名词，比如某个公司的产品名称，所以直接保留不翻译。

还要注意标点符号和格式是否正确。例如，英文中的冒号和句号在中文中是否需要调整？通常，中文使用全角符号，所以需要检查是否有遗漏或错误。此外，列表项是否对齐，是否有正确的缩进，这些都是需要保留的格式细节。

用户可能是在准备技术文档的中文版本，所以翻译的准确性至关重要。如果有不确定的地方，比如某些技术术语或专有名词，可能需要进一步确认，但根据用户的要求，这里应该直接翻译，不添加额外内容。

最后，完成翻译后，我会通读一遍，确保没有遗漏任何部分，并且所有格式都正确无误。特别是检查是否有代码块或图表块被错误地翻译或遗漏，确保完全符合用户的要求。
</think>

## 核心服务器功能

MCP服务器可以实现以下任何组合功能：<think>
好的，我现在需要处理用户的翻译请求。首先，用户给了一个Markdown格式的文本，要求翻译成简体中文，同时保留所有格式。我要仔细检查每个部分是否符合要求。

首先看标题“Resources”，直接翻译成“资源”即可。接下来是正文部分，用户提到资源为用户提供上下文和数据，这里要确保技术术语准确。比如“Document repositories”翻译成“文档仓库”，“Knowledge bases”是“知识库”，“Structured data sources”需要处理成“结构化数据源”，而“File systems”就是“文件系统”。

然后检查是否有代码块或图表，用户提供的例子中没有这些内容，所以不需要处理。还要注意内联代码，但这里也没有。确保翻译后的内容保持原有的Markdown结构，比如列表项前的短横线和项目符号。

另外，用户强调不要添加任何额外解释，所以直接输出翻译结果即可。最后通读一遍，确认格式正确，术语准确，没有遗漏或错误。完成翻译后，确保没有使用任何Markdown格式的错误，比如正确的符号和缩进。
</think>

### 资源

资源为用户提供上下文和数据：

- 文档仓库
- 知识库
- 结构化数据源
- 文件系统<think>
</think>

### 提示

提示是为用户提供的模板消息和工作流程：

- 预定义的对话模板
- 引导式交互模式
- 专用的对话结构<think>
好的，我现在需要处理用户的翻译请求。首先，用户给了一个关于工具的英文内容，要求翻译成简体中文，同时保持Markdown格式。我得仔细检查每个部分是否符合要求。

首先看标题“### Tools”，这应该直接翻译为“### 工具”。接下来是工具的描述部分，里面列出了四个项目。每个项目都是工具的不同类型，比如数据处理工具、外部API集成、计算能力、搜索功能。这些都需要准确翻译，同时保持列表格式。

要注意的是，用户强调不要翻译代码块、图表或内联代码，但这里的内容没有这些元素，所以没问题。另外，要确保不添加任何额外解释，只输出翻译后的内容。我需要逐行检查，确保每个项目符号都正确对应，并且没有遗漏任何格式。比如“Data processing utilities”翻译成“数据处理工具”，“External API integrations”是“外部API集成”，“Computational capabilities”译为“计算能力”，“Search functionality”则是“搜索功能”。最后，确认整个结构和原文一致，没有格式错误。完成后，再通读一遍，确保流畅和准确。
</think>

### 工具

工具是AI模型执行的功能：

- 数据处理工具
- 外部API集成
- 计算能力
- 搜索功能<think>
</think>

## 示例实现：C# 实现

官方的 C# SDK 仓库包含多个示例实现，展示了 MCP 的不同方面：

- **基础 MCP 客户端**：简单示例，展示如何创建 MCP 客户端并调用工具
- **基础 MCP 服务器**：带有基本工具注册的最小服务器实现
- **高级 MCP 服务器**：功能完整的服务器，包含工具注册、认证和错误处理
- **ASP.NET 集成**：展示与 ASP.NET Core 集成的示例
- **工具实现模式**：实现不同复杂度工具的各种模式

MCP C# SDK 仍处于预览阶段，API 可能会发生变化。我们将随着 SDK 的演进持续更新此博客。<think>
好的，我现在需要处理用户的翻译请求。首先，用户希望将一段英文的技术文档翻译成简体中文，并且要保留Markdown格式。我需要仔细检查用户提供的输入，确保所有元素都被正确转换。

首先，用户给出的输入是关于MCP协议的几个关键特性和相关链接。我需要确保标题“Key Features”被正确翻译为“关键特性”，并保持加粗和标题的格式。接下来是项目符号列表，里面的链接需要保留原样，但文本部分要翻译。例如，“C# MCP Nuget ModelContextProtocol”应该翻译成“C# MCP NuGet ModelContextProtocol”，注意Nuget的正确拼写。第二个项目中的链接标题“Building your first MCP Server”要翻译为“构建您的第一个MCP服务器”，同时保持链接不变。

然后，用户提到的完整C#实现示例链接需要翻译文本部分，但保留原始URL。这里要注意“official C# SDK samples repository”翻译为“官方C# SDK示例仓库”，并确保链接正确无误。整个段落的结构需要保持一致，使用正确的Markdown语法，比如加粗、列表和链接。

在翻译过程中，我需要特别注意技术术语的准确性，比如“Nuget”在中文中通常保留原名，但有时也会翻译为“NuGet”，所以需要确认正确的拼写。另外，确保所有Markdown格式如标题、列表项、链接等都被正确保留，没有遗漏或格式错误。同时，用户强调不要翻译代码块、图表或其他特定块，所以这部分内容要原样输出，不需要处理。

最后，检查整个翻译后的内容是否符合用户的所有要求：准确的翻译、保留的格式、正确的链接和术语。确保没有添加任何额外的解释或内容，只输出翻译后的内容。完成这些步骤后，再通读一遍，确认没有错误，然后输出最终的翻译结果。
</think>

### 关键特性

- [C# MCP NuGet ModelContextProtocol](https://www.nuget.org/packages/ModelContextProtocol)
- 构建您的[第一个 MCP 服务器](https://devblogs.microsoft.com/dotnet/build-a-model-context-protocol-mcp-server-in-csharp/).

完整 C# 实现示例请访问 [官方 C# SDK 示例仓库](https://github.com/modelcontextprotocol/csharp-sdk)<think>
</think>

## 示例实现：使用 Spring 的 Java 实现

使用 Spring 的 Java SDK 提供了强大且企业级的 MCP 实现选项。<think>
好的，用户让我翻译一段技术文档内容，需要保持Markdown格式，并且注意代码块和特定元素不要翻译。首先，我得仔细看看用户提供的输入内容。

用户给的输入是关于关键特性的列表，然后是一个指向Java与Spring示例的链接。首先，我需要确保每个项目符号都正确翻译，同时保留原有的格式。比如“Spring Framework integration”翻译成“Spring框架集成”，“Strong type safety”是“强类型安全”，这些术语要准确。

接下来是Reactive programming support，这里“Reactive programming”通常翻译为“响应式编程”，所以应该是“响应式编程支持”。然后是Comprehensive error handling，翻译成“全面的错误处理”比较合适。

然后看链接部分，用户提供的链接是[samples/java/containerapp/README.md](samples/java/containerapp/README.md)，需要保持原样，但中文部分要翻译。原句是“For a complete Java with Spring implementation sample, see...”，翻译成“要查看完整的Java与Spring实现示例，请参见...”比较自然。注意保持链接的格式正确，确保Markdown的链接语法正确，即文本部分用方括号，链接用圆括号。

还要检查是否有代码块或者图表，但这里没有，所以不需要处理。确保没有遗漏任何格式，比如标题的### Key Features要转成### 关键特性，保持中文的标题格式。

最后，整体检查一遍翻译后的内容是否符合用户的要求，没有添加额外解释，严格保留Markdown结构，所有元素都正确翻译，特别是技术术语的准确性。确认无误后，输出结果。
</think>

### 关键特性

- Spring框架集成
- 强类型安全
- 响应式编程支持
- 全面的错误处理

要查看完整的Java与Spring实现示例，请参见[samples/java/containerapp/README.md](samples/java/containerapp/README.md)中的示例目录。<think>
好的，我现在需要处理用户的翻译请求。首先，用户希望将一段英文技术文档翻译成简体中文，并且要严格保留Markdown格式。我需要仔细检查用户提供的示例，确保理解所有的要求。

用户给出的例子是关于JavaScript实现的，原文是“## Sample implementation: JavaScript Implementation”和后面的内容。翻译后的结果应该保持相同的Markdown结构，比如标题和列表项。同时，用户特别指出不能翻译代码块、图表等，这些部分要原样保留。

接下来，我需要确认用户提供的输入是否符合他们的要求。例如，在用户提供的例子中，没有代码块或图表，所以翻译时不需要处理这些部分。但如果有代码块，比如用反引号包裹的代码，必须保持原样，不进行翻译。

另外，用户强调不要添加任何额外的解释或思考过程，直接输出翻译后的内容。这意味着我必须确保翻译准确，同时严格遵循格式要求。需要检查每个部分，比如标题是否正确转换，列表项是否保留，链接和图片是否正确处理。

还要注意术语的准确性，比如“JavaScript SDK”应翻译为“JavaScript SDK”，因为技术术语通常不翻译。同时，保持句子的流畅和自然，符合中文的表达习惯。例如，“lightweight and flexible approach”可以译为“轻量级且灵活的方案”。

最后，我需要确保整个翻译过程没有遗漏任何格式元素，比如加粗、斜体、列表符号等。这可能需要逐行检查，确保每个Markdown元素都被正确转换，同时内容准确无误。完成翻译后，再次通读一遍，确认所有要求都已满足，没有多余的内容或格式错误。
</think>

## 示例实现：JavaScript 实现

JavaScript SDK 提供了一种轻量级且灵活的 MCP 实现方案。<think>
好的，我现在需要处理用户的翻译请求。首先，用户希望将一段英文的技术文档内容翻译成简体中文，并且保持Markdown格式不变。我需要仔细检查用户提供的输入，确保所有元素都被正确处理。

首先，用户给出的输入是关于某个库或工具的关键特性列表，包括Node.js和浏览器支持、基于Promise的API、与Express等框架的易集成性，以及WebSocket支持。然后有一个链接指向JavaScript示例的README.md文件。

根据用户的指示，我需要严格保留Markdown格式，包括标题、列表、粗体、斜体、链接等。同时，代码块、mermaid/plantuml/flowchart图表块以及内联代码（反引号内的内容）不需要翻译，直接保留原样。此外，用户强调不要添加任何额外解释或思考过程，只需输出翻译后的内容。

现在，我需要逐行处理输入内容。首先，标题“### Key Features”应该翻译为“### 主要特性”，保持三级标题的格式。接下来是列表项，每个项目都要翻译成中文，同时保持项目符号和格式。例如，“Node.js and browser support”翻译为“Node.js 和浏览器支持”，注意中文的标点符号使用全角符号。

然后，检查是否有任何代码块或特殊格式需要保留。在用户提供的输入中，有一个链接：“[JavaScript sample](samples/javascript/README.md)”，需要确保翻译后的链接文本正确，同时保持Markdown的链接格式。这里“JavaScript sample”应翻译为“JavaScript 示例”，而链接地址保持不变。

另外，用户提到的“samples directory”应翻译为“samples 目录”，但根据上下文，可能需要保持英文原词，因为技术文档中常用英文术语。不过，根据用户的指示，需要准确翻译，所以“samples directory”应译为“samples 目录”。

最后，确保整个翻译后的内容没有遗漏任何格式，如粗体、斜体等。在用户提供的输入中没有使用这些格式，所以不需要处理。同时，检查是否有内联代码或代码块，但用户提供的输入中没有这些元素。

总结一下，翻译后的内容应保持结构和格式完全一致，仅将文本内容翻译成简体中文，同时保留所有Markdown元素和链接。确保没有添加任何额外内容，严格按照用户的要求执行。
</think>

### 主要特性

- Node.js 和浏览器支持
- 基于 Promise 的 API
- 与 Express 和其他框架的易集成性
- WebSocket 支持用于流式传输

如需查看完整的 JavaScript 实现示例，请参见 samples 目录中的 [JavaScript 示例](samples/javascript/README.md)。<think>
</think>

## 示例实现：Python 实现

Python SDK 提供了一种 Pythonic 的 MCP 实现方式，并与优秀的 ML 框架集成。<think>
好的，我现在需要处理用户的这个翻译请求。首先，用户希望将一段英文的技术文档内容翻译成简体中文，并且要严格保留Markdown格式。我需要仔细检查用户提供的输入，确保所有格式元素都被正确转换，同时不翻译代码块、图表或其他特定标记。

首先，看一下用户提供的输入内容。标题是“Key Features”，翻译成中文应该是“主要功能”或者“关键特性”。这里选择“主要功能”更符合技术文档的常见用法。接下来是列表项，每个项目都是一个功能点。例如，“Async/await support with asyncio”需要翻译为“使用asyncio支持异步/等待”，但要注意保持技术术语的准确性。同时，用户提到要保留Markdown的列表格式，所以每个项目符号前的短横线要保留，但中文的项目符号可能需要调整，不过用户可能希望保持原来的符号，所以直接保留短横线。

接下来是“FastAPI integration”，翻译成“FastAPI集成”即可。然后是“Simple tool registration”，翻译为“简单的工具注册”，最后是“Native integration with popular ML libraries”，翻译成“与流行机器学习库的原生集成”。这里要注意“native integration”翻译为“原生集成”比较合适，符合技术文档的术语。

用户还提到有一个完整的Python实现示例，链接到samples目录下的Python样本README.md。需要确保链接文本正确翻译，同时保持Markdown的链接格式。原句中的“see [Python sample](samples/python/README.md) in the samples directory.”翻译成“参见 samples 目录中的 [Python 示例](samples/python/README.md)。”这里要注意中文的标点符号，比如中文的引号和逗号的使用，以及链接的格式是否正确。

另外，用户特别强调不要翻译代码块、mermaid/plantuml/flowchart图表块以及内联代码，所以需要检查是否有这些元素。在当前的输入中，没有这些元素，但如果有，应该原样保留。例如，如果有一个代码块，应该保持反引号和代码内容不变。

还需要注意中英文标点符号的转换，比如英文的冒号和括号要转换为中文的对应符号。例如，链接中的括号和逗号要使用中文的全角符号。同时，确保没有遗漏任何Markdown格式，如加粗、斜体等，但当前输入中没有这些元素。

最后，检查整个翻译后的内容是否符合用户的要求，没有额外的解释或思考过程，仅输出翻译后的内容。确保所有格式正确，术语准确，链接有效，并且保持原文的结构和意图。完成这些步骤后，就可以生成最终的翻译结果了。
</think>

### 主要功能

- 使用asyncio支持异步/等待
- FastAPI集成
- 简单的工具注册
- 与流行机器学习库的原生集成

参见 samples 目录中的 [Python 示例](samples/python/README.md) 获取完整的 Python 实现示例。<think>
</think>

## API 管理

Azure API 管理是解决如何保护 MCP 服务器的一个绝佳方案。其核心思想是在你的 MCP 服务器前面部署一个 Azure API 管理实例，让它处理你很可能需要的以下功能：

- 速率限制
- 令牌管理
- 监控
- 负载均衡
- 安全性<think>
</think>

### Azure 示例

以下是一个 Azure 示例，实现了这一点，即[创建一个 MCP 服务器并使用 Azure API Management 进行保护](https://github.com/Azure-Samples/remote-mcp-apim-functions-python)。

查看下方图片，了解授权流程是如何进行的：

![APIM-MCP](https://github.com/Azure-Samples/remote-mcp-apim-functions-python/blob/main/mcp-client-authorization.gif?raw=true)

在上图中，发生了以下情况：

- 使用 Microsoft Entra 进行身份验证/授权。
- Azure API Management 作为网关，并使用策略来引导和管理流量。
- Azure Monitor 会记录所有请求以便进一步分析。<think>
好的，用户让我翻译一段关于授权流程的技术文档，并且要保留Markdown格式。首先，我需要仔细阅读用户提供的英文内容，确保理解每个部分的意思。用户提到要准确翻译，所以要注意专业术语的正确对应，比如“authorization flow”应该翻译为“授权流程”而不是直译。

接下来，用户给出的例子中有标题“Authorization flow”，我需要将其翻译为“授权流程”，并保持####的Markdown格式。然后是正文部分，用户详细描述了授权流程，并附上了一个图片链接。图片链接中的描述部分需要翻译，但要注意保留原始的链接结构，不需要修改URL。

用户特别强调要保留所有Markdown元素，比如标题、列表、粗斜体等。在翻译过程中，我需要检查是否有任何格式被破坏。例如，图片的Markdown语法是`![Sequence Diagram](...)`，翻译后的中文应该是“序列图”，但链接部分保持不变。

另外，用户提到不要翻译代码块、图表块和内联代码，所以如果有这些元素的话，需要原样保留。在当前的例子中，图片链接中的描述部分需要翻译，但链接本身不需要改动。同时，要确保没有遗漏任何格式，比如粗体或斜体，但在这个例子中似乎没有使用到。

还需要注意用户可能的深层需求，比如他们可能是在准备中文的技术文档，需要准确且专业的翻译，同时保持格式的完整性，以便直接使用。因此，翻译时要确保术语的一致性，比如“OAuth”通常不翻译，保持原样。

最后，检查整个翻译后的内容是否符合用户的所有要求：准确翻译、保留Markdown格式、不添加额外内容。确保没有遗漏任何细节，比如图片链接中的参数是否正确保留，以及翻译后的文本是否流畅自然。
</think>

#### 授权流程

让我们更详细地了解一下授权流程：

![序列图](https://github.com/Azure-Samples/remote-mcp-apim-functions-python/blob/main/infra/app/apim-oauth/diagrams/images/mcp-client-auth.png?raw=true)<think>
好的，我现在需要处理用户的翻译请求。首先，用户给了一个Markdown格式的文本，里面有一个标题和一个链接。根据用户的指示，我需要将英文内容准确翻译成简体中文，同时保持Markdown的格式不变。

首先看标题部分：“#### MCP authorization specification”，这里需要翻译成“#### MCP 授权规范”。注意标题后面有四个减号，这在Markdown中表示四级标题，翻译时要保留结构。

接下来是链接部分：“Learn more about the [MCP Authorization specification](https://modelcontextprotocol.io/specification/2025-03-26/basic/authorization#2-10-third-party-authorization-flow)”。这里需要将“Learn more about the”翻译为“了解更多关于”，然后保留链接中的文本和URL。链接中的“MCP Authorization specification”翻译成“MCP 授权规范”，而锚点部分“#2-10-third-party-authorization-flow”可能不需要翻译，保持原样。

需要确保没有遗漏任何格式，比如标题的符号、链接的括号和URL。同时，用户强调不要翻译代码块、图表或内联代码，但这里没有相关内容，所以不需要处理。最后检查整个翻译是否准确，符合用户的要求，没有添加额外内容。
</think>

#### MCP 授权规范

了解更多关于 [MCP 授权规范](https://modelcontextprotocol.io/specification/2025-03-26/basic/authorization#2-10-third-party-authorization-flow)<think>
好的，我现在需要处理用户的请求，将英文的技术文档翻译成简体中文，并且保持Markdown格式。首先，我要仔细阅读用户的指令，确保完全理解所有的要求。用户特别强调要保留Markdown的格式，包括标题、列表、粗斜体、链接、图片、表格等。同时，遇到代码块、mermaid/plantuml/flowchart图表块、内联代码时，要原样输出，不要翻译。此外，不能添加任何额外的解释或思考过程，直接输出翻译后的内容。

接下来，我需要检查用户提供的输入内容。用户给出的例子是：

## Deploy Remote MCP Server to Azure

Let's see if we can deploy the sample we mentioned earlier:

1. Clone the repo

首先，我要确认这些内容是否符合翻译的要求。标题“Deploy Remote MCP Server to Azure”需要翻译成“部署远程MCP服务器到Azure”，但要注意技术术语的准确性，比如“MCP Server”可能是一个专有名词，是否需要保留英文？根据用户的例子，他们可能希望保留“MCP Server”不变，所以翻译为“部署远程MCP服务器到Azure”是正确的。

接下来是列表项“Clone the repo”，翻译成“克隆仓库”即可。这里要注意“repo”是“repository”的缩写，翻译成“仓库”是合适的。

然后，我需要确保整个翻译后的内容保持Markdown格式。例如，标题使用##，列表使用数字加点。同时，检查是否有任何代码块或图表块，但在这个例子中没有，所以不需要处理。

另外，用户可能希望翻译后的文本保持与原文相同的结构和格式，所以需要逐行对应翻译。例如，原文中的“Let's see if we can deploy the sample we mentioned earlier:”翻译为“让我们看看是否可以部署我们之前提到的示例：”，注意冒号的使用和中文标点的正确性。

还需要注意是否有任何特殊符号或格式需要保留，比如链接或图片，但在这个例子中没有出现。因此，只需进行纯文本翻译，同时保持Markdown的结构。

最后，确保没有添加任何额外内容，比如解释或思考过程，直接输出翻译结果。因此，最终的翻译应该准确无误，符合用户的所有要求。
</think>

## 部署远程MCP服务器到Azure

让我们看看是否可以部署我们之前提到的示例：

1. 克隆仓库    ```bash
    git clone https://github.com/Azure-Samples/remote-mcp-apim-functions-python.git
    cd remote-mcp-apim-functions-python
    ```
<think>
好的，用户让我翻译一段技术文档，需要保持Markdown格式，代码块和内联代码不要翻译。首先，我需要仔细阅读用户提供的英文内容，确保理解每个部分的结构和要求。

第一段是关于注册资源提供者的步骤，分为Azure CLI和Azure PowerShell两种情况。注意用户提到的INLINE_CODE_0到INLINE_CODE_3，这些应该是代码块，需要保留原样。翻译时要确保命令的准确性，比如“Register `Microsoft.App` resource provider.”中的INLINE_CODE_0可能是一个占位符，比如资源提供者的名称，所以直接保留不翻译。

接下来是第二段，用户提到运行azd命令来部署API管理服务、函数应用和其他资源。这里的链接需要保持原样，确保用户点击后能正确跳转。同时，注意“azd”是一个命令行工具，可能不需要翻译，保持原样即可。

需要检查是否有其他Markdown元素，比如列表项、粗体等。用户提供的例子中没有使用粗体，但如果有，需要确保翻译后仍然正确。例如，如果原文中有**bold text**，翻译后应保持为**bold text**。

另外，注意用户强调不要添加任何额外解释，只输出翻译后的内容。因此，必须严格遵守，不添加任何说明或格式错误。同时，确保翻译后的中文流畅且技术术语准确，比如“provision”翻译为“部署”或“配置”更合适。

最后，通读整个翻译，检查是否所有代码块和内联代码都保留，Markdown格式是否正确，比如列表项前的短横线是否正确，链接是否正确。确保没有遗漏任何用户指定的格式要求，比如表格、图片等，但用户提供的例子中没有这些，所以可能不需要处理。

总结一下，我需要准确翻译文本，保留所有代码和格式，确保技术术语正确，并且完全符合用户的要求，不添加任何额外内容。
</think>

1. 注册 `Microsoft.App` 资源提供者。

   - 如果您使用 Azure CLI，请运行 `az provider register --namespace Microsoft.App --wait`。
   - 如果您使用 Azure PowerShell，请运行 `Register-AzResourceProvider -ProviderNamespace Microsoft.App`。之后过一段时间运行 `(Get-AzResourceProvider -ProviderNamespace Microsoft.App).RegistrationState` 检查注册是否完成。

1. 运行此 [azd](https://aka.ms/azd) 命令以部署 api management 服务、function app（含代码）和所有其他必需的 Azure 资源    ```shell
    azd up
    ```
<think>
好的，我现在需要处理用户的翻译请求。首先，用户给了一段英文：“This commands should deploy all the cloud resources on Azure”，要求翻译成简体中文，并且保留Markdown格式。看起来用户可能是在写技术文档，需要准确的翻译，同时保持格式不变。

首先，我要确认用户的需求。他们可能是在准备一份技术指南或者部署文档，里面涉及到Azure云资源的部署命令。用户希望翻译后的文本保持原有的结构，比如如果有标题、列表或者其他Markdown元素，不需要改动。不过当前输入看起来只是一个单独的句子，没有其他格式。

接下来，我需要检查是否有代码块、图表或者其他需要保留原样的内容。用户提到的命令可能包含代码，但当前输入中没有代码块，只有普通文本。所以翻译时不需要处理代码块。

然后，翻译部分。原句是“This commands should deploy all the cloud resources on Azure”。首先，注意语法是否正确。原句中的“commands”应该是复数形式“commands”，因为通常命令是多个。不过用户可能打错了，或者故意用单数。但根据技术文档的常见用法，可能应该是复数。不过用户可能希望保持原样，所以翻译时要保留原词。翻译成“此命令应部署所有Azure云资源”或者“这些命令应部署所有Azure云资源”，取决于原句的正确性。

但用户给出的原文是“This commands”，这里语法错误，应该是“This command”或者“These commands”。不过用户可能希望保持原样，所以翻译时要保留原词。因此，正确的翻译可能是“此命令应部署所有Azure云资源”或者“这些命令应部署所有Azure云资源”，但需要确认用户是否希望修正语法错误。不过根据用户的要求，可能只是直接翻译，不修改原句的结构，所以直接翻译成“此命令应部署所有Azure云资源”。

另外，用户可能希望保持命令的准确性，所以需要确保“deploy all the cloud resources on Azure”正确翻译为“部署所有Azure云资源”。同时，注意“Azure”作为专有名词，不需要翻译，直接保留。

最后，检查是否符合所有要求：保留Markdown格式，没有代码块或其他需要处理的内容，翻译准确。确认无误后，输出翻译结果。
</think>

此命令应部署所有Azure云资源<think>
好的，用户让我翻译一段技术文档的内容，需要保持Markdown格式，同时注意代码块和特定标记不要翻译。首先，我需要仔细阅读用户提供的英文内容，确保理解每个部分的要求。

用户给出的输入是：

### Testing your server with MCP Inspector

1. In a **new terminal window**, install and run MCP Inspector

首先，我要确认翻译的准确性。标题部分“Testing your server with MCP Inspector”应该翻译为“使用MCP Inspector测试您的服务器”，保持专业术语不变。接下来是列表项，其中“new terminal window”需要翻译成“新终端窗口”，而“install and run MCP Inspector”则是“安装并运行MCP Inspector”。需要注意的是，用户强调要保留Markdown格式，所以标题的###和列表项的数字编号必须保持原样。

另外，用户提到不要翻译代码块、mermaid/plantuml/flowchart图表块以及内联代码，但当前输入中没有这些元素，所以暂时不需要处理。需要检查是否有任何可能被误译的部分，比如斜体或加粗，但这里只有**new terminal window**是加粗，翻译后应保持加粗格式。

还要确保没有额外的解释或思考过程，直接输出翻译后的内容。现在，我需要将每个部分逐行翻译，同时保持结构不变。确认无误后，输出结果即可。
</think>

### 使用MCP Inspector测试您的服务器

1. 在一个 **新终端窗口** 中安装并运行 MCP Inspector    ```shell
    npx @modelcontextprotocol/inspector
    ```
<think>
好的，我现在需要处理用户的翻译请求。首先，用户希望将一段英文技术文档翻译成简体中文，并且要严格保留Markdown格式。我需要仔细检查每个部分，确保格式不变，比如标题、列表、粗斜体、链接、图片、表格等。

首先看用户提供的英文内容，里面有一个图片链接和两个步骤。图片链接的alt文本是“Connect to Node inspector”，我需要翻译成“连接到Node检查器”，同时保持图片的路径不变。接下来是步骤部分，用户使用了数字列表，每个步骤里有内联代码和需要翻译的文本。

第一个步骤提到CTRL点击加载MCP Inspector web app，这里的URL例子中的链接需要保留原样，但可能需要检查是否有特殊字符需要处理。然后是设置传输类型为INLINE_CODE_0，这里需要确认是否要翻译“INLINE_CODE_0”这部分，但根据用户的要求，内联代码不需要翻译，所以保持原样。接下来是设置URL到运行中的API Management SSE端点，这里提到的INLINE_CODE_1可能是一个占位符，应该保留不翻译。

需要注意用户强调不要添加任何额外解释或思考过程，所以翻译后的内容必须直接呈现，没有多余的文字。同时，要确保所有Markdown元素如加粗、链接等正确转换，比如“**Connect**”需要翻译为“连接”，但保持加粗格式。检查是否有代码块或图表块，但用户提供的例子中没有这些，所以可能不需要处理。

最后，确保整个翻译后的文本结构与原文一致，没有遗漏任何部分。可能需要多次检查每个段落和格式是否正确应用，特别是链接和图片的路径是否正确保留。确认所有翻译准确，没有语法错误，并且符合技术文档的专业性。
</think>

你应该看到一个与以下界面类似的界面：

![连接到Node检查器](/03-GettingStarted/01-first-server/assets/connect.png)

1. 按住CTRL点击从应用显示的URL加载MCP Inspector web应用（例如 [http://127.0.0.1:6274/#resources](http://127.0.0.1:6274/#resources)）
1. 将传输类型设置为 `SSE`
1. 将URL设置为在 `azd up` 后显示的正在运行的API Management SSE端点并**连接**：    ```shell
    https://<apim-servicename-from-azd-output>.azure-api.net/mcp/sse
    ```
<think>
好的，用户让我翻译一段英文的技术文档内容到简体中文，同时要保留Markdown格式。首先，我需要仔细阅读用户提供的原文，确保理解每个部分的要求。

用户给出的原文是：

1. **List Tools**.  Click on a tool and **Run Tool**.  

If all the steps have worked, you should now be connected to the MCP server and you've been able to call a tool.

首先，我要确认翻译的准确性。比如“List Tools”应该翻译为“工具列表”，而“Run Tool”是“运行工具”。注意保持加粗的格式，使用**包裹。

接下来检查是否有代码块、图表或者内联代码。原文中没有这些元素，所以直接翻译即可。然后，确保句子结构符合中文习惯，比如第二句的连接词和语序调整。

另外，用户强调要严格保留Markdown格式，所以需要确认标题、列表、粗体等是否正确转换。例如，原文中的列表项是否用星号或数字，这里看起来是有序列表，但用户例子中用的是数字1.，所以保持原样。

最后，检查是否有遗漏或格式错误，确保翻译后的内容完全符合用户的要求，没有多余的内容或解释。确保所有格式如粗体、段落等都正确无误。
</think>

1. **工具列表**。 点击一个工具并 **运行工具**。  

如果所有步骤都已成功执行，您现在应该已连接到 MCP 服务器并且能够调用工具。<think>
</think>

## Azure 上的 MCP 服务器

[Remote-mcp-functions](https://github.com/Azure-Samples/remote-mcp-functions-dotnet): 这些仓库是使用 Azure Functions 以 Python、C# .NET 或 Node/TypeScript 构建和部署自定义远程 MCP（模型上下文协议）服务器的快速入门模板。

这些示例提供了一个完整的解决方案，使开发人员能够：

- 本地构建和运行：在本地机器上开发和调试 MCP 服务器
- 部署到 Azure：通过简单的 azd up 命令轻松部署到云
- 从客户端连接：从各种客户端连接到 MCP 服务器，包括 VS Code 的 Copilot 代理模式和 MCP 检查器工具<think>
</think>

### 核心特性

- 安全设计：通过密钥和 HTTPS 对 MCP 服务器进行安全保护
- 认证选项：支持使用内置认证和/或 API 管理的 OAuth 认证
- 网络隔离：通过 Azure 虚拟网络 (VNET) 实现网络隔离
- 无服务器架构：利用 Azure Functions 实现可扩展、事件驱动的执行
- 本地开发：提供全面的本地开发和调试支持
- 简化部署：简化到 Azure 的部署流程

该仓库包含所有必要的配置文件、源代码和基础设施定义，以便快速启动一个生产就绪的 MCP 服务器实现。

- [Azure 远程 MCP 函数 Python](https://github.com/Azure-Samples/remote-mcp-functions-python) - 使用 Azure Functions 和 Python 实现的 MCP 示例

- [Azure 远程 MCP 函数 .NET](https://github.com/Azure-Samples/remote-mcp-functions-dotnet) - 使用 Azure Functions 和 C# .NET 实现的 MCP 示例

- [Azure 远程 MCP 函数 Node/TypeScript](https://github.com/Azure-Samples/remote-mcp-functions-typescript) - 使用 Azure Functions 和 Node/TypeScript 实现的 MCP 示例。<think>
</think>

## 重点总结

- MCP SDKs 提供语言特定的工具，用于实现强大的 MCP 解决方案
- 调试和测试过程对于可靠 MCP 应用至关重要
- 可重用的提示模板可实现一致的 AI 交互
- 设计良好的工作流可以使用多个工具协调复杂任务
- 实现 MCP 解决方案需要考虑安全、性能和错误处理<think>
</think>

## 练习

设计一个实用的MCP工作流程，解决您领域中的现实问题：

1. 识别3-4个对解决问题有帮助的工具
2. 创建一个工作流程图，展示这些工具如何交互
3. 使用您偏好的语言实现其中一个工具的基本版本
4. 创建一个提示模板，帮助模型有效地使用您的工具<think>
好的，我现在需要处理用户的翻译请求。首先，用户希望将一段英文的技术文档内容翻译成简体中文，并且要严格保留Markdown格式。我需要仔细检查用户提供的输入，确保所有元素都被正确转换。

首先，用户给出的例子是：

## Additional Resources

---

Next: [Advanced Topics](../05-AdvancedTopics/README.md)

翻译后的输出应该是：

## 附加资源

---

下一节：[高级主题](../05-AdvancedTopics/README.md)

看起来用户希望保持标题和链接的结构不变，只是将文本部分翻译。我需要确认是否有任何特殊格式需要处理，比如代码块或图表，但在这个例子中没有出现。接下来，我要确保翻译后的中文准确传达原意，同时符合技术文档的正式语气。

另外，用户强调不要添加任何额外解释，所以必须只输出翻译后的内容。需要检查是否有隐藏的字符或格式错误，确保Markdown结构正确。例如，链接的格式是否正确，标题的级别是否正确。还要注意标点符号是否符合中文习惯，比如使用中文的引号和括号。

最后，确认是否所有要求都被满足：保留Markdown格式，不翻译代码块和内联代码，准确翻译文本部分。在这个例子中，所有条件都满足，所以翻译结果应该是正确的。
</think>

## 附加资源

---

下一节：[高级主题](../05-AdvancedTopics/README.md)