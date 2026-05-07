配置
```
// Zed settings
//
// For information on how to configure Zed, see the Zed
// documentation: https://zed.dev/docs/configuring-zed
//
// To see all of Zed's default settings without changing your
// custom settings, run `zed: open default settings` from the
// command palette (cmd-shift-p / ctrl-shift-p)
{
  "base_keymap": "JetBrains",
  "soft_wrap": "prefer_line",
  "which_key": {
    "enabled": true
  },
  "terminal": {
    "font_family": "MesloLGS NF"
  },
  "prettier": {
    "allowed": false
  },
  "autosave": "on_focus_change",
  "vim_mode": true,
  "icon_theme": {
    "mode": "system",
    "light": "Zed (Default)",
    "dark": "Catppuccin Mocha"
  },
  "edit_predictions": {
    "provider": "zed",
    "mode": "eager"
  },
  "ui_font_size": 16,
  "buffer_font_size": 15,
  "theme": {
    "mode": "system",
    "light": "One Light",
    "dark": "Ayu Mirage"
  },
  "buffer_font_features": {
    "liga": false, // 禁用 liga：!= 显示为 != 而非 ≠
    "calt": false // 禁用 calt：<= 显示为 <= 而非 ≤，--> 显示为 --> 而非 →→
  },
}


```

快捷键
```
[
  // {
  //   "context": "Workspace",
  //   "bindings": {
  //     "cmd-o": "project_panel::ToggleFocus",
  //     "cmd-shift-enter": "workspace::ToggleZoom",
  //     "cmd-j": "workspace::NewSearch",
  //     "ctrl-g": "workspace::CloseAllDocks"
  //   }
  // },
  // {
  //   "context": "ProjectPanel && not_editing",
  //   "bindings": {
  //     "enter": "project_panel::Open"
  //   }
  // },
  {
    "context": "Editor",
    "bindings": {
      "cmd-o": "project_panel::ToggleFocus",
      "cmd-shift-enter": "workspace::ToggleZoom",
      "cmd-j": "workspace::NewSearch",
      "ctrl-g": "workspace::CloseAllDocks",
      "ctrl-b": "editor::GoToDefinition",
      "cmd-b": "editor::GoToImplementation",
      "ctrl-n": "pane::GoBack",
      "ctrl-[": "editor::FoldAll",
      "ctrl-]": "editor::UnfoldAll",
      "ctrl-i": "editor::MoveToBeginningOfLine",
      "ctrl-o": "editor::MoveToEndOfLine",
      "ctrl-backspace": "editor::DeleteToPreviousWordStart",
      "cmd-shift-n": "editor::Rename",
      "cmd-k": "editor::FindAllReferences",
    }
  },
  {
    "context": "Terminal||Editor||Workspace",
    "bindings": {
      "ctrl-t": "terminal_panel::Toggle",
      "ctrl-g": "workspace::CloseAllDocks",
    }
  },
  {
    "context": "Editor && !menu",
    "bindings": {
      "ctrl-f": "buffer_search::Deploy",
      "ctrl-c": "editor::Copy",
      "ctrl-x": "editor::Cut",
      "ctrl-v": "editor::Paste",
      "ctrl-a": "editor::SelectAll",
      "ctrl-y": "editor::Undo",
      "ctrl-s": "workspace::Save"
    }
  },
  {
    "context": "Editor && VimControl && !VimWaiting && !menu",
    "bindings": {
      "space f f": [
        "workspace::SendKeystrokes",
        "cmd-shift-i cmd-s"
      ]
    }
  },
]

```