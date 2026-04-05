# OAuth 2.0 完整学习指南

## 一、什么是 OAuth 2.0？

OAuth 2.0 是一个 **授权框架（Authorization Framework）**，不是认证协议（Authentication）。它的核心目标是：

> **让第三方应用在用户授权的前提下，安全地访问用户在资源服务器上的受保护资源，而无需获取用户的用户名和密码。**

### 常见场景举例：
- 使用"微信登录"登录某电商网站
- 某记账 App 请求访问你的支付宝交易记录（需你授权）
- GitHub Actions 需要访问你的代码仓库

> ⚠️ 注意：OAuth 2.0 本身不提供身份认证（Authentication），但常与 OpenID Connect（OIDC）结合实现单点登录（SSO）。

---

## 二、OAuth 2.0 的四大角色

| 角色 | 说明 |
|------|------|
| **Resource Owner（资源所有者）** | 通常是用户（User），拥有受保护资源的控制权 |
| **Client（客户端）** | 请求访问资源的应用（Web App、Mobile App、SPA 等） |
| **Authorization Server（授权服务器）** | 验证用户身份并颁发访问令牌（Access Token）的服务 |
| **Resource Server（资源服务器）** | 托管受保护资源的服务，接受带有效 Token 的请求 |

> 💡 示例：你在用"网易云音乐"登录"豆瓣"，那么：
> - 资源所有者 = 你
> - 客户端 = 豆瓣
> - 授权服务器 = 网易云音乐的 OAuth 服务
> - 资源服务器 = 网易云音乐的 API（如获取你的歌单）

---

## 三、核心概念

### 1. Access Token（访问令牌）
- 一个字符串（通常为 JWT 或 opaque token）
- 代表对特定资源的访问权限
- 有作用域（Scope）、有效期（Expires In）
- 客户端用它向资源服务器请求数据

### 2. Refresh Token（刷新令牌）
- 用于在 Access Token 过期后获取新的 Access Token
- 通常只在 **Confidential Client**（如后端 Web 应用）中使用
- 必须安全存储（不能暴露给前端）

### 3. Scope（作用域）
- 定义客户端请求的权限范围（如 `read:profile`, `write:email`）
- 用户授权时可看到具体权限
- 资源服务器根据 Scope 决定是否放行请求

### 4. Redirect URI（重定向 URI）
- 授权服务器完成授权后跳转回客户端的地址
- 必须预先注册，防止开放重定向攻击

---

## 四、OAuth 2.0 的授权模式（Grant Types）

OAuth 2.0 定义了多种授权流程，适用于不同客户端类型。

### 1. Authorization Code（授权码模式）✅【最安全、最常用】
- 适用于 **Web 应用（有后端）**
- 流程：
  1. 用户点击"使用 Google 登录"
  2. 跳转到 Google 授权页（带 `client_id`, `redirect_uri`, `scope`）
  3. 用户同意授权 → Google 重定向回你的网站，附带 **authorization code**
  4. 你的后端用 code + `client_secret` 向 Google 换取 `access_token`
  5. 用 access_token 调用 Google API

- **`client_secret` 的作用与来源**
  - **作用**: 它扮演着**应用程序的“密码”**角色。当后端用 `authorization_code` 换取 `access_token` 时，必须同时附上 `client_secret`，以此向授权服务器证明自己是合法的应用，而不是伪装的攻击者。这是防止 `authorization_code` 在中途被截获后遭滥用的关键安全措施。
  - **来源**: `client_id` 和 `client_secret` 都来自于**预先注册**。开发者必须在服务提供商（如 Google Cloud Platform）的控制台中注册自己的应用程序，并提供 `redirect_uri` 等信息。注册成功后，服务提供商会生成这对凭证，并将其与你的应用绑定。

> ✅ 优点：code 只在浏览器中短暂存在，token 通过后端交换，更安全  
> 🔒 支持 PKCE（见下文）增强安全性

---

### 2. Authorization Code with PKCE（Proof Key for Code Exchange）
- 专为 **公共客户端（Public Clients）** 设计，如 SPA（React/Vue）、移动 App
- 解决传统授权码模式在无 `client_secret` 场景下的安全问题
- 流程增加：
  - 客户端生成 `code_verifier`（随机字符串）和 `code_challenge = SHA256(code_verifier)`
  - 授权请求时带上 `code_challenge`
  - 换 token 时必须提供原始 `code_verifier`
- 防止授权码被截获后冒用

> ✅ 现代 SPA 应优先使用此模式（替代已废弃的 Implicit 模式）

---

### 3. Client Credentials（客户端凭证模式）
- 适用于 **服务间通信（Machine-to-Machine）**
- 客户端直接用 `client_id` + `client_secret` 换取 token
- 不涉及用户（Resource Owner）
- 例如：后台微服务 A 调用微服务 B 的 API

---

### 4. Resource Owner Password Credentials（密码模式）⚠️【已不推荐】
- 客户端直接收集用户账号密码，发给授权服务器换 token
- 仅适用于高度信任的场景（如自家 App 调自家 API）
- **违反 OAuth 原则**（不应接触用户密码）
- RFC 6749 已明确不鼓励使用

---

### ❌ Implicit Grant（隐式模式）【已废弃】
- 曾用于 SPA，token 直接通过 URL fragment 返回
- 存在严重安全风险（token 暴露在浏览器历史、Referer 中）
- **已被 PKCE + Authorization Code 取代**

---

## 五、Token 类型

| 类型                      | 特点          | 使用场景                 |
| ----------------------- | ----------- | -------------------- |
| **Opaque Token**        | 随机字符串，无意义   | 授权服务器需验证（查数据库）       |
| **JWT（JSON Web Token）** | 自包含、可签名、可加密 | 资源服务器可本地验证（无需查授权服务器） |

JWT 示例结构：
  Header.Payload.Signature
``` jsoon
  { "sub": "user123", "scope": "read:profile", "exp": 1735689600, "iss": "[https://auth.example.com](https://auth.example.com/)" } 
 ```


## 六、安全最佳实践

1. **始终使用 HTTPS**  
    所有通信必须加密，防止 token 被窃听。
    
2. **严格校验 Redirect URI**  
    防止开放重定向（Open Redirect）导致 token 泄露。
    
3. **短期 Access Token + 长期 Refresh Token**  
    减少 token 泄露影响范围。
    
4. **SPA 使用 PKCE**  
    即使没有 `client_secret`，也能保证授权码安全。
    
5. **避免在前端存储 Refresh Token**  
    若必须（如移动端），应使用安全存储（Keychain/Keystore）。
    
6. **使用 state 参数防 CSRF**  
    在授权请求中加入随机 `state`，回调时校验，防止跨站请求伪造。
    
7. **最小权限原则（Least Privilege）**  
    只申请必要的 Scope。
    

---

## 七、OAuth 2.0 vs OpenID Connect（OIDC）

|特性|OAuth 2.0|OpenID Connect|
|---|---|---|
|目的|**授权**（Access Delegation）|**认证**（User Authentication）|
|输出|Access Token|ID Token（JWT） + Access Token|
|标准|RFC 6749|基于 OAuth 2.0 的扩展|
|用途|"你能帮我做事吗？"|"你是谁？"|

> OIDC = OAuth 2.0 + 身份层  
> 实际项目中，登录功能通常用 **OIDC**，API 访问用 **OAuth 2.0**

---

## 八、详解授权流程：以网易云音乐登录豆瓣为例

### 场景设定

- **用户**：你（Resource Owner）
- **客户端（Client）**：豆瓣网站（`https://www.douban.com`）
- **授权服务器 & 资源服务器**：网易云音乐（`https://music.163.com`）
    - 授权服务器地址：`https://auth.music.163.com`
    - 资源 API 地址：`https://api.music.163.com/v1/user/profile`

### 前提条件（开发阶段配置）

1. 豆瓣开发者在网易云音乐开放平台注册应用：
    - `client_id = douban_web_2025`
    - `redirect_uri = https://www.douban.com/auth/netease/callback`

### 完整授权流程（带 PKCE 的 Authorization Code Flow）

#### 步骤 1️⃣：用户点击"使用网易云音乐登录"

豆瓣前端生成 PKCE 参数（仅 SPA 或无后端 secret 时需要）：

```js
// 1. 生成 code_verifier（高熵随机字符串，43~128 字符）
const code_verifier = generateRandomString(64); // 如 "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"

// 2. 计算 code_challenge = BASE64URL(SHA256(code_verifier))
const code_challenge = base64UrlEncode(sha256(code_verifier));

// 3. 将 code_verifier 临时存入 sessionStorage（后续换 token 用）
sessionStorage.setItem('code_verifier', code_verifier);
```

#### 构造授权请求 URL 并跳转：

```
GET https://auth.music.163.com/oauth/authorize?
  response_type=code&
  client_id=douban_web_2025&
  redirect_uri=https%3A%2F%2Fwww.douban.com%2Fauth%2Fnetease%2Fcallback&
  scope=profile&
  state=xyz123abc&          // 防 CSRF 的随机值
  code_challenge=dbjftjez4cvp-mb92k27uhbuj1p1r_ww1gfwfoejxk&
  code_challenge_method=S256
```

> ✅ `state` 必须由豆瓣生成并存储（如存入 session），回调时校验，防止 CSRF  
> ✅ `code_challenge_method=S256` 表示使用 SHA256（推荐），也可用 `plain`（不安全，不推荐）

#### 步骤 2️⃣：网易云音乐授权页面

- 用户被重定向到网易云音乐的登录/授权页
- 如果未登录，先输入账号密码登录
- 然后看到授权提示：
    
    > "豆瓣想要访问你的网易云音乐个人资料（昵称、头像）"
    
- 用户点击 **"同意"**

#### 步骤 3️⃣：网易云音乐重定向回豆瓣（携带授权码）

用户同意后，网易云音乐将浏览器重定向回豆瓣的 `redirect_uri`，附带 `code` 和 `state`：

```
HTTP/1.1 302 Found
Location: https://www.douban.com/auth/netease/callback?
  code=SplxlOBeZQQYbYS6WxSbIA&
  state=xyz123abc
```

> ⚠️ `code` 是一次性、短期有效的（通常 1~10 分钟过期）

#### 步骤 4️⃣：豆瓣后端用 code 换取 Access Token

> 📌 关键点：**这一步必须由豆瓣的后端完成！不能在前端做！**

豆瓣后端收到请求后：

1. 校验 `state` 是否与之前存储的一致（防 CSRF）
2. 从 `sessionStorage` 或前端传来的安全通道获取 `code_verifier`（如果是 SPA，前端需通过 POST 发给后端）
3. 向网易云音乐授权服务器发起 **POST 请求** 换取 token：

```http
POST /oauth/token HTTP/1.1
Host: auth.music.163.com
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&
code=SplxlOBeZQQYbYS6WxSbIA&
redirect_uri=https%3A%2F%2Fwww.douban.com%2Fauth%2Fnetease%2Fcallback&
client_id=douban_web_2025&
code_verifier=dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk
```

#### 步骤 5️⃣：网易云音乐返回 Access Token

如果验证通过（code 有效、code_verifier 匹配、redirect_uri 一致），返回：

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxxx",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "def50200a1b2c3d4e5f6...",  // 可选
  "scope": "profile"
}
```

> 💡 `access_token` 可能是 JWT 或 opaque token。如果是 JWT，豆瓣可自行解析；否则需调用 userinfo endpoint。

#### 步骤 6️⃣：豆瓣用 Access Token 获取用户信息

豆瓣后端携带 token 调用网易云音乐的资源 API：

```http
GET /v1/user/profile HTTP/1.1
Host: api.music.163.com
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxxx
```

网易云音乐验证 token 有效且 scope 包含 `profile` 后，返回：

```json
{
  "id": "13829102",
  "nickname": "音乐爱好者小张",
  "avatar": "https://p1.music.126.net/xxx.jpg"
}
```

#### 步骤 7️⃣：豆瓣完成本地登录

- 豆瓣根据网易云音乐用户 ID（如 `13829102`）查找本地是否已有账户
    - 若有：直接登录
    - 若无：创建新账户，绑定网易云音乐 ID
- 设置豆瓣自己的会话 Cookie（如 `sessionid=abc123`）
- 重定向用户到豆瓣首页

✅ 至此，**整个 OAuth 2.0 授权登录流程完成**！

---

## 九、为什么必须由后端换 Token？

### 情况一：传统 Web 应用（有后端 + 有 `client_secret`）

#### ❌ 如果前端去换 token 会发生什么？

假设豆瓣前端（JavaScript）直接向网易云音乐发请求换 token：

```js
// ⚠️ 危险！绝对不要这样做！
fetch("https://auth.music.163.com/oauth/token", {
  method: "POST",
  body: new URLSearchParams({
    grant_type: "authorization_code",
    code: "SplxlOBeZQQYbYS6WxSbIA",
    redirect_uri: "https://www.douban.com/auth/netease/callback",
    client_id: "douban_web_2025",
    client_secret: "SECRET_12345"   // ← 问题在这里！
  })
})
```

#### 风险：

1. **`client_secret` 被写死在前端代码里**
    - 任何用户打开浏览器开发者工具 → Sources / Network，都能看到这个密钥
2. **攻击者拿到 `client_secret` 后可以：**
    - 伪造豆瓣应用，向网易云音乐发起任意授权请求
    - 如果配合钓鱼网站，诱导用户授权，就能窃取大量用户数据
    - 甚至可能被网易云音乐封禁整个豆瓣的 `client_id`

> 💡 `client_secret` 的作用就是证明："我是真正的豆瓣服务器，不是冒牌货"。一旦公开，就失去了意义。

### 情况二：单页应用 SPA（无后端 or 无 `client_secret`，用 PKCE）

#### ⚠️ 但如果不用 PKCE，只在前端换 token，会怎样？

假设没有 PKCE，前端直接用 `code` 换 token：

```js
// 没有 PKCE 的危险做法
fetch("https://auth.music.163.com/oauth/token", {
  method: "POST",
  body: { grant_type: "authorization_code", code: "...", ... }
})
```

#### 风险：**授权码拦截攻击（Authorization Code Interception Attack）**

1. 攻击者在用户点击"网易云登录"后，通过恶意软件或网络劫持，**截获了回调 URL 中的 `code`**
2. 因为没有 `client_secret` 也没有 PKCE，攻击者可以直接用这个 `code` 去换 `access_token`
3. 于是攻击者就拿到了用户的网易云音乐访问权限！

> 🌰 举例：你在咖啡店连 Wi-Fi，黑客在同一网络嗅探流量，看到你访问 `https://douban.com/callback?code=ABC123`，立刻用这个 code 换 token —— 成功冒充你！

### ✅ PKCE 如何解决这个问题？

- 豆瓣前端生成 `code_verifier` 并存起来
- 授权请求时只发送 `code_challenge`（不可逆）
- 换 token 时必须提供原始 `code_verifier`
- **即使攻击者拿到 `code`，没有 `code_verifier` 也无法换 token**

### 总结表格

|场景|能否前端换 token？|原因|
|---|---|---|
|**有后端 + 有 `client_secret`**|❌ 绝对不行|会泄露 `client_secret`|
|**纯前端 SPA + 无 secret**|✅ 可以，但**必须用 PKCE**|防止 code 被盗用|
|**最佳安全实践**|👉 尽量让后端处理 token 交换|减少前端暴露敏感信息|

---

## 十、常见实现方案

|方案|说明|
|---|---|
|**Auth0**|商业化 Identity Platform，支持 OAuth 2.0 / OIDC|
|**Keycloak**|开源身份和访问管理（IAM）系统|
|**Spring Security OAuth**|Java 生态常用（注意：旧版已弃用，新项目用 Spring Authorization Server）|
|**Passport.js**|Node.js 的认证中间件，支持多种 OAuth 策略|
|**Django OAuth Toolkit**|Python/Django 的 OAuth 2.0 实现|

---

## 十一、动手实验建议

1. 使用 [Postman](https://www.postman.com/) 配置 OAuth 2.0（支持所有 grant type）
2. 注册一个 Google Cloud 项目，尝试"Google Login"流程
3. 用 [oauth.tools](https://oauth.tools/) 可视化调试 OAuth 流程
4. 搭建本地 Keycloak，模拟授权服务器

---

## 十二、总结

- OAuth 2.0 是 **授权框架**，不是认证协议
- **Authorization Code + PKCE** 是现代 Web/移动应用的黄金标准
- 安全是核心：HTTPS、PKCE、state、最小权限
- 实际登录场景多用 **OpenID Connect**（基于 OAuth 2.0）
- **后端换 token 是安全的关键步骤**，避免泄露 `client_secret` 或防止授权码被盗用

---

> 📚 推荐阅读：
> 
> - [RFC 6749 - The OAuth 2.0 Authorization Framework](https://datatracker.ietf.org/doc/html/rfc6749)
> - [OAuth 2.0 Simplified (by Aaron Parecki)](https://aaronparecki.com/oauth-2-simplified/)
> - [OAuth 2.0 Security Best Current Practice (RFC 6819)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-security-topics)

```
</title>
```