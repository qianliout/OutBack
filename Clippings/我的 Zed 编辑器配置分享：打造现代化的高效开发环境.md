---
title: "我的 Zed 编辑器配置分享：打造现代化的高效开发环境"
source: "https://mp.weixin.qq.com/s/jIc0Bx4VWIPSFwx9bXQfJA"
author:
  - "[[0与1]]"
published:
created: 2026-06-20
description: "我的 Zed 编辑器配置分享：打造现代化的高效开发环境Zed 是一款用 Rust 编写的下一代代码编辑器。"
tags:
  - "clippings"
---
0与1 CodeJarvis *2026年3月17日 07:08*

> “
> 
> Zed 是一款用 Rust 编写的下一代代码编辑器，以极致性能著称。本文分享我的个人配置，包括主题、字体、快捷键等，快捷键部分参考了
> 
> JetBrains、Emacs、Neovim 的设计习惯，希望对有同样背景的开发者有所帮助。
> 
> 文末有完整的 `settings.json` 和 `keymap.json`

---

## 一、外观与字体

```
"theme": "Ayu Mirage",
"icon_theme": "Catppuccin Macchiato",                                                                                            "buffer_font_family": "Maple Mono NF CN",
"buffer_font_size": 18.0,                                                                                                        "ui_font_size": 18.0,
"buffer_line_height": "comfortable",
```
- 主题：Ayu Mirage，深色系，护眼舒适 - 图标主题：Catppuccin Macchiato
- 字体：Maple Mono NF CN，一款支持中文的等宽编程字体
- 字体大小：编辑区和 UI 统一设为 18，大屏幕舒适阅读

---

## 二、编辑器行为

```
"vim_mode": true,
"relative_line_numbers": "enabled",
"colorize_brackets": true,
"use_auto_surround": true,
"format_on_save": "off",
"tab_size": 4,
"hard_tabs": false,
"scroll_beyond_last_line": "one_page",
"current_line_highlight": "line",
"show_whitespaces": "selection",
```
- 开启 Vim 模式
- 相对行号：跳转行数一目了然，必须配合 Vim 操作才好用
- 括号着色：层级一眼看清，减少括号匹配错误
- 自动包围：选中文字后输入括号/引号，自动包裹
- 全局关闭保存时格式化，但对 Rust / TypeScript / JavaScript / Zig 单独开启

---

## 三、语言专项配置

```
"languages": {
"Rust": { "format_on_save": "on" },
"TypeScript": { "format_on_save": "on" },
"JavaScript": { "format_on_save": "on" },
"YAML": { "tab_size": 2 },
"TOML": { "tab_size": 2 },
"Zig": {
"formatter": "language_server",
"format_on_save": "on",
"language_servers": ["zls", "..."]
}
}
```
- Rust / TS / JS 保存即格式化，减少手动操作
- YAML / TOML 缩进改为 2 空格，符合社区惯例
- Zig 使用 ZLS 作为主要 Language Server

---

## 四、AI 助手配置

```
"agent": {
"default_model": {
"provider": "google",
"model": "gemini-3-flash-preview"
},
"inline_assistant_model": {
"provider": "copilot_chat",
"model": "claude-haiku-4.5"
},
"default_profile": "write"
}
```
- 默认对话模型：Google Gemini 3 Flash，速度快、免费额度充足
- 内联 AI 助手：使用 Claude Haiku，轻量快速，适合代码补全和小改动

---

## 五、快捷键设计思路：JetBrains 风格

> “
> 
> 我习惯使用 Emacs 作为基础键位（"base\_keymap": "Emacs"），行移动上仿照了 JetBrains IDE 的操作逻辑，通过 Space Leader 键 实现
> 
> Neovim 风格的空间键映射。

### 5.1 光标移动（Emacs 风格）

| 快捷键 | 功能 |
| --- | --- |
| `Ctrl-h` | 向左移动 |
| `Ctrl-l` | 向右移动 |
| `Ctrl-k` | 向上移动 |
| `Ctrl-j` | 向下移动 |
| `Ctrl-e`  （插入模式） | 移到行尾 |
| `Ctrl-a`  （插入模式） | 移到行首 |

这组映射在插入模式下无需离开主键区，手指不离 home row。

### 5.2 Vim Normal 模式行首/行尾

H → 移到行首（含缩进感知） L → 移到行尾

类似 JetBrains 中 Home / End 的智能行首行尾行为。

### 5.3 代码操作（Space 前缀）

| 快捷键 | 功能 | 对应 JetBrains |
| --- | --- | --- |
| `Space d` | 跳转到定义 | `Ctrl+B` |
| `Space g d` | 跳转到定义 | `Ctrl+B` |
| `Space g i` | 跳转到实现 | `Ctrl+Alt+B` |
| `Space g y` | 跳转到类型定义 | `Ctrl+Shift+B` |
| `Space k` | 显示文档悬浮窗 | `Ctrl+Q` |
| `Space c a` | 代码修复建议 | `Alt+Enter` |
| `Space c r` | 重命名变量 | `Shift+F6` |
| `F8` | 跳转到下一个诊断 | `F2` |
| `Shift-F8` | 跳转到上一个诊断 | `Shift+F2` |

### 5.4 文件与窗口操作

| 快捷键 | 功能 | 对应 JetBrains |
| --- | --- | --- |
| `Space f` | 文件搜索 | `Ctrl+Shift+N`  / Double Shift |
| `Space /` | 全局文本搜索 | `Ctrl+Shift+F` |
| `Space p` | 切换项目 | Recent Projects |
| `Space e` | 切换左侧文件树 | `Alt+1` |
| `Space t` | 切换终端面板 | `Alt+F12` |
| `Space w` | 保存文件 | `Ctrl+S` |
| `Space q` | 关闭当前标签页 | `Ctrl+F4` |
| `Space v` | 垂直分屏 | `Ctrl+\` |

### 5.5 行操作（类 JetBrains）

| 快捷键 | 功能 | 对应 JetBrains |
| --- | --- | --- |
| `Alt-Shift-Up` | 上移当前行 | `Shift+Alt+Up` |
| `Alt-Shift-Down` | 下移当前行 | `Shift+Alt+Down` |
| `Alt-Up` | 扩展语法节点选择 | `Ctrl+W` |
| `Alt-Down` | 收缩语法节点选择 | `Ctrl+Shift+W` |

### 5.6 Git 快捷键

| 快捷键 | 功能 |
| --- | --- |
| `Cmd-Shift-Z` | 撤销文件改动（Git Restore） |
| `Cmd-Y` | Stage 当前 hunk 并跳到下一个 |
| `Cmd-Shift-Y` | Unstage 当前 hunk 并跳到下一个 |
| `Cmd-Alt-Y` | 切换 Stage 状态 |

---

## 六、其他实用配置

```
"which_key": { "enabled": true }
```

> “
> 
> 开启 which-key 插件，按下 Space 后稍作停顿，会弹出所有可用快捷键提示，对于记不住键位的场景非常友好。

```
"minimap": {
"show": "auto",
"thumb": "always",
"thumb_border": "left_open"
}
```

> “
> 
> 迷你地图开启，滚动条始终显示，方便大文件快速定位。

---

## 七、总结

| 功能 | 配置选择 |
| --- | --- |
| 编辑模式 | Vim + Emacs 键位混合 |
| 快捷键风格 | JetBrains / Neovim Leader Key |
| 主题 | Ayu Mirage |
| 字体 | Maple Mono NF CN |
| AI 助手 | Gemini 3 Flash + Claude Haiku |
| 主要语言 | Rust / TypeScript / Zig |

> “
> 
> Zed 的性能优势配合合理的键位设计，可以让日常开发效率媲美甚至超越 JetBrains 系列 IDE。如果你也是从 IDEA / CLion / WebStorm
> 
> 迁移过来的开发者，希望这份配置能给你一些参考。

## 完整配置

### settings.json

```
{
 "search": {
  "include_ignored": false,
  "case_sensitive": true,
  "whole_word": true
 },
 "restore_on_startup": "last_workspace",
 "icon_theme": "Catppuccin Macchiato",
 "git_panel": {
  "tree_view": true,
  "button": true, // false 不显示 git 左侧按钮
  "sort_by_path": true
 },
 "collaboration_panel": {
  "button": false
 },
 "base_keymap": "Emacs",
 "theme": "Ayu Mirage",
 "agent": {
  "inline_assistant_model": {
   "provider": "copilot_chat",
   "model": "claude-haiku-4.5"
  },
  "default_profile": "ask",
  "default_model": {
   "provider": "google",
   "model": "gemini-3-flash-preview"
  }
 },
 "preview_tabs": {
  "enabled": true,
  "enable_preview_from_file_finder": false
 },
 "ui_font_size": 18.0, // 调整界面字体大小
 "buffer_font_size": 18.0, // 调整编辑区字体大小
 "buffer_font_family": "Maple Mono NF CN", // 更改编辑区字体
 "buffer_line_height": "comfortable",
 "buffer_font_weight": 400.0,
 "vim_mode": true,
 "which_key": {
  "enabled": true
 },
 "terminal": {
  "font_size": 14, // 集成终端的字体大小
  "line_height": "comfortable", // 终端行高
  "copy_on_select": false, // 在终端中选择文本时不要复制
  "dock": null
 },
 "use_auto_surround": true,
 "colorize_brackets": true,
 "tab_size": 4, // 缩进的空格数
 "hard_tabs": false, // 使用空格而非制表符进行缩进
 "format_on_save": "off", // 是否在保存时自动格式化代码
 "show_whitespaces": "selection", // 仅在选定文本时显示空白字符，也可以设置为 "none", "all", "trailing"
 "preferred_line_length": 100, // 软换行长度，达到此长度时 Zed 会尝试换行
 "current_line_highlight": "line", // 高亮显示当前行，"line" 或 "none"
 "relative_line_numbers": "enabled", // 显示相对行号
 "extend_comment_on_newline": false, // 当前一行是注释时，新行是否也以注释开头
 "scrollbar": {
  "show": "auto" // 滚动条显示方式
 },
 "scroll_beyond_last_line": "one_page", // 编辑器不会滚动到最后一行之外
 "scroll_sensitivity": 0.4,
 "fast_scroll_sensitivity": 1,
 "seed_search_query_from_cursor": "selection",
 "use_smartcase_search": true,
 "status_bar": {
  "active_language_button": true,
  "cursor_position_button": true,
  "line_endings_button": true
 },
 "use_on_type_format": false,
 // 标签页
 "tabs": {
  "close_position": "right",
  "file_icons": true,
  "git_status": true,
  "activate_on_close": "left_neighbour",
  "show_close_button": "hover",
  "show_diagnostics": "all"
 },
 "telemetry": {
  "diagnostics": false, // 禁用发送诊断数据
  "metrics": false // 禁用发送使用指标
 },
 // 迷你地图
 "minimap": {
  "show": "auto",
  "thumb": "always",
  "thumb_border": "left_open",
  "current_line_highlight": null
 },
 "languages": {
  "Rust": {
   "document_symbols": "on",
   "format_on_save": "on" // 对 Rust 语言在保存时自动格式化
  },
  "TypeScript": {
   "format_on_save": "on" // 对 TypeScript 语言在保存时自动格式化
  },
  "JavaScript": {
   "format_on_save": "on" // 对 JavaScript 语言在保存时自动格式化
  },
  "YAML": {
   "tab_size": 2
  },
  "TOML": {
   "tab_size": 2
  },
  "Zig": {
   // Formatting with ZLS matches \`zig fmt\`.
   "formatter": "language_server",
   "format_on_save": "on",
   // Make sure that zls is the primary language server
   "language_servers": ["zls", "..."],
   "code_actions_on_format": {
    // Run code actions that currently supports adding and removing discards.
    // "source.fixAll": true,
    // Run code actions that sorts @import declarations.
    // "source.organizeImports": true,
   }
  },
  "Markdown": {
   "tab_size": 1
  }
 },
 "project_panel": {
  "dock": "left" // 项目面板停靠在左侧
 },
 "proxy": "socks5://127.0.0.1:6153",
 "lsp": {
  "markdown-oxide": {
   "enable_lsp_tasks": true
  },
  "rust-analyzer": {
   "initialization_options": {
    "completion": {
     "snippets": {
      "custom": {
       "Arc::new": {
        "postfix": "arc",
        "body": ["Arc::new(${receiver})"],
        "requires": "std::sync::Arc",
        "scope": "expr"
       },
       "Some": {
        "postfix": "some",
        "body": ["Some(${receiver})"],
        "scope": "expr"
       },
       "Ok": {
        "postfix": "ok",
        "body": ["Ok(${receiver})"],
        "scope": "expr"
       },
       "Rc::new": {
        "postfix": "rc",
        "body": ["Rc::new(${receiver})"],
        "requires": "std::rc::Rc",
        "scope": "expr"
       },
       "Box::pin": {
        "postfix": "boxpin",
        "body": ["Box::pin(${receiver})"],
        "requires": "std::boxed::Box",
        "scope": "expr"
       },
       "vec!": {
        "postfix": "vec",
        "body": ["vec![${receiver}]"],
        "description": "vec![]",
        "scope": "expr"
       }
      }
     }
    }
   }
  }
 }
}
```

### keymap.json

```
[
 {
  "context": "Workspace",
  "bindings": {
   // "shift shift": "file_finder::Toggle",
  }
 },
 {
  "context": "Editor",
  "bindings": {
   "ctrl-/": "editor::Undo", // 撤销
   "ctrl-l": "editor::MoveRight", // 右移
   "ctrl-h": "editor::MoveLeft", // 左移
   "ctrl-k": "editor::MoveUp", // 上移
   "ctrl-j": "editor::MoveDown", // 下移
   "shift-alt-f": "editor::Format",
   "cmd-\\": "pane::SplitRight",
   "cmd-alt-\\": "pane::SplitDown",

   "alt-shift-up": "editor::MoveLineUp",
   "alt-shift-down": "editor::MoveLineDown",
   "alt-up": "editor::SelectLargerSyntaxNode", // Expand selection
   "alt-down": "editor::SelectSmallerSyntaxNode", // Shrink selection

   // 跳转到错误
   "f8": ["editor::GoToDiagnostic", { "severity": { "min": "hint", "max": "error" } }],
   "shift-f8": ["editor::GoToPreviousDiagnostic", { "severity": { "min": "hint", "max": "error" } }]
  }
 },
 {
  "context": "vim_mode == normal",
  "bindings": {
   "H": ["editor::MoveToBeginningOfLine", { "stop_at_soft_wraps": true, "stop_at_indent": true }],
   "L": "editor::MoveToEndOfLine",
   "ctrl-e": null,
   "space": null,
   "enter": null
  }
 },

 {
  "context": "Editor && vim_mode == normal && !VimWaiting && !menu",
  "bindings": {
   // --- 文件与导航 (类似 Neo-tree / Telescope) ---
   "space d": "editor::GoToDefinition",
   "space e": "workspace::ToggleLeftDock", // 类似 nvim-tree/neo-tree 侧边栏
   "space f": "file_finder::Toggle", // 类似 Telescope find_files
   "space p": "projects::OpenRecent", // 切换项目
   // "space b": "pane::ActivatePrevItem", // 快速切回上一个标签页

   // --- 搜索 (类似 Telescope live_grep) ---
   "space /": "pane::DeploySearch", // 全项目搜索文本
   // "space s w": "editor::SearchSelectedText", // 搜索当前光标下的词

   // --- LSP / 代码跳转 (模仿 g 开头的映射到 space) ---
   "space c a": "editor::ToggleCodeActions", // 代码修复建议 (Code Action)
   "space c r": "editor::Rename", // 重命名变量
   "space g d": "editor::GoToDefinition", // 跳转到定义
   "space g i": "editor::GoToImplementation", // 跳转到实现
   "space g y": "editor::GoToTypeDefinition", // 跳转到类型定义
   "space k": "editor::Hover", // 显示文档悬浮窗 (LSP Hover)
   // "space d": "diagnostics::Deploy", // 显示项目所有的诊断错误

   // --- 窗口与编辑器控制 ---
   "space w": "workspace::Save", // 快速保存
   "space q": "pane::CloseActiveItem", // 关闭当前标签页 (Buffer)
   "space v": "pane::SplitRight", // 垂直分屏
   // "space h": "pane::SplitBottom", // 水平分屏

   // --- 终端 / Task 控制 ---
   "space t": "terminal_panel::Toggle", // 切换底部终端
   "space r": "task::Spawn" // 弹出 Task 运行列表 (类似运行你截图中的 cargo)
  }
 },
 {
  "context": "EmptyPane || SharedScreen",
  "bindings": {
   "space f": "file_finder::Toggle" // 在没有打开文件时也能用 space f
  }
 },

 {
  "context": "vim_mode == insert",
  "bindings": {
   "ctrl-e": "editor::MoveToEndOfLine",
   "ctrl-a": [
    "editor::MoveToBeginningOfLine",
    { "stop_at_soft_wraps": true, "stop_at_indent": true }
   ]
  }
 },
 {
  "context": "Pane",
  "use_key_equivalents": true,
  "bindings": {
   "ctrl-0": "pane::ActivateLastItem",
   "cmd-alt-left": "pane::GoBack",
   "cmd-alt-right": "pane::GoForward",
   "cmd-shift-f": "pane::DeploySearch",
   "cmd-k u": ["pane::CloseCleanItems", { "close_pinned": false }],
   "cmd-k w": ["pane::CloseAllItems", { "close_pinned": false }],
   "cmd-k cmd-w": "workspace::CloseAllItemsAndPanes"
  }
 },

 {
  "context": "Editor && !agent_diff && !AgentPanel",
  "use_key_equivalents": true,
  "bindings": {
   // "cmd-alt-z": "git::Restore",
   "cmd-shift-z": "git::Restore",
   "cmd-alt-y": "git::ToggleStaged",
   "cmd-y": "git::StageAndNext",
   "cmd-shift-y": "git::UnstageAndNext"
  }
 }
]
```

Rust之路 · 目录

作者提示: 个人观点，仅供参考

继续滑动看下一个

CodeJarvis

向上滑动看下一个