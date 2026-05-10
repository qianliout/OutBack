#  mcp-feedback-collector

## github地址
https://github.com/sanshao85/mcp-feedback-collector-web

## 用法

augment的会员是按对话次数计费的，这个mcp可以在augment一次对话结束之后询问用户并等待用户反馈 ，如果用户反馈之后就会继续执行，这样就可以在一次对话中做更多的事

## 配置
```json
{
  "mcpServers": {
    "mcp-feedback-collector": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-feedback-collector@latest"
      ],
      "env": {
        "MCP_API_KEY": "your_api_key_here",
        "MCP_API_BASE_URL": "https://api.ssopen.top",
        "MCP_DEFAULT_MODEL": "grok-3",
        "MCP_WEB_PORT": "5050",
        "MCP_DIALOG_TIMEOUT": "60000",
        "MCP_ENABLE_IMAGE_TO_TEXT": "true"
      }
    }
  }
}
```

直接这样配置的话，当一次对话的任务执行完成之后，就会在浏览器中打开5050端口的界面，让用户输入反馈

