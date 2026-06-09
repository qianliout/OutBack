 


# bug1：场景图查询出现了角色参考图
 资源库-图片资源-场景图 查询出现了角色参考图

上传后出现两条 `sta_resources`

## bug原因 
同一文件先调 `POST /api/v1/storage/upload`（或 `…/image/upload`），成功后会自动 `AddResource`（`source=storage_upload`）；再调 `POST /api/v1/resources/uploads` 登记时又会 `AddResource`（`source=user_upload`）。两条链互不感知，同一 `file_url` 产生两行。

**涉及代码**  
`internal/server/http/controllers/v1/storage.go`（`Upload`、`UploadImage`）。

`AddResource` 在未显式传入 `category`/`sub_kind` 时依赖 `inferCategorySubKindFromLegacy`：对 `type=image`，除 `scene`、`prop_reference`、`character_outfit` 等少数 `source` 外，**默认一律推断为 `sub_kind=scene`**。通用上传曾使用 `storage_upload` 等未枚举 `source`，会被标成场景图；与「场景图」列表筛选条件一致，因而混入。


## 修复方式
上传接口仅负责 OSS 与团队空间用量，**不再**写入 `sta_resources`。资源库统一由 `POST /api/v1/resources/uploads` 登记；上传响应中不再返回 `resource_id`。


# bug2: 关键帧图-未返回section_num

分镜资产-关键帧图-未返回section_num
## Bug 原因

资源库接口 `GET /api/v1/resources` 在筛选 分镜类关键帧（`category=board`、`sub_kind=keyframe`、`source=keyframe` 等）时，前端需要 `section_num`（本集内第几段），与 `episode_num` / `shot_num` 一起唯一定位到「某集 · 某段 · 某镜」。

实际情况是：

1. 列表 最初没有在响应里体现段号  
    `LibraryResourceItem` 没有 `section_num` 字段，只把数据库里的 `extra` 原样返回；若 `extra` 里本身就没有段号，接口整体就不会出现 `section_num`。
    
2. `sta_resources.extra` 在写入关键帧资源时未写入段号
    
    - MQ 里关键帧生成成功后 `AddResource` 只写了 `video_frame_id`、`board_version_id`、`shot_num`、`episode_num` 等，未写入 `section_num`，尽管同一条流水里已通过 `ResolveBoardSectionShot` 拿到了 `Section.SectionIndex`。
    - 管理端从 `sta_video_frames` 回补资源时同样只设了集号/镜号，没有把解析结果里的段落序号写入 `extra`。
    - 跨项目复制关键帧 路径里 `extra` 更瘦，同样缺少与线上一致的段落、集号等信息。

## 修复方式

1. 写入侧：在 `extra` 中补齐 `section_num`
    
    - `handler_generate_video_frame.go`：关键帧落库资源时，在 `resolved.Section != nil` 时设置 `extra["section_num"] = resolved.Section.SectionIndex`。
    - `internal/services/admin/resources.go`：回补关键帧资源时，先保留 `ResolveBoardSectionShot` 的完整结果，再组装 `extra`，在 `resolved.Section != nil` 时写入 `section_num`。
    - `internal/services/resource/copy.go`：`copyBoardKeyframeTx` 与主路径对齐，写入 `board_version_id`、`episode_num`，并在有 section 时写入 `section_num`。
2. 读取侧：列表显式返回 `section_num`
    
    - `LibraryResourceItem` 增加 `section_num` 字段。
    - `internal/utils/tools.go` 增加 `GetInt64FromAny`，从 `extra["section_num"]`（兼容 JSON 常见的 `float64` / `json.Number` / 整型）解析后赋给列表项，保证 `GET /api/v1/resources` 的 `list[].section_num` 与 `extra` 一致。



# bug3:角色图片没有版本号

资源库-角色图片-点击查看详情没有版本区分

- 详情页面是有version字段的，前端没有展示
- 列表页面没有返回version制度

## 原因

- 列表形状：`GET /api/v1/resources` 返回的 `LibraryResourceItem` 原先没有顶层 `version`，版本信息若存在只会出现在 `extra` 里。
- 数据不一致（根因）：主形象在 `character_image` MQ 成功写入 `sta_resources` 时，`extra` 里带有 `version`（数值版号）；妆造（`source=character_outfit`）在两条路径入库时 `extra` 未写 `version`：
    - MQ 成功分支里 `outfitRow != nil` 的 `AddResource`；
    - 创建/更新妆造并带图时的 `indexOutfitResource`。
- 因此前端若读 `extra.version` 或期望与主形象字段一致时，妆造类条目会表现为没有版本号；这不是列表层「删掉字段」，而是写入侧漏字段。

## 修复说明

1. MQ 入库（[`handler_character_generate.go`](vscode-file://vscode-app/Applications/Cursor.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/internal/process/runners/mq/handler_character_generate.go)）  
    在 `character_outfit` 的 `Extra` 中增加 `"version": ver.Version`，与主形象语义一致：`extra.version` = 对应 `sta_character_versions.version`。
    
2. 妆造索引入库（[`internal/services/character/outfit.go`](vscode-file://vscode-app/Applications/Cursor.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/internal/services/character/outfit.go)）  
    `indexOutfitResource` 内按 `CharacterVersionID` 查询 `sta_character_versions`，将 `cv.Version` 写入 `extra["version"]`；索引失败时 向上返回错误（不再静默 `_ =`），避免「妆造已保存但资源索引不完整」无人知晓。
    
3. 列表兼容前端（[`internal/services/resource/list.go`](vscode-file://vscode-app/Applications/Cursor.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/internal/services/resource/list.go)）  
    为 `LibraryResourceItem` 增加 `version`（`json:"version,omitempty"`），在 `extra["version"]` 能解析为正整数时 填入顶层字段








# bug5 片段视频查询出错
资源库-剪辑视频-片段视频/完整正片 查询出来有分镜视频（分镜视频应该出现在分镜资产就可以了）

```
curl 'https://testapi.storyaai.com/api/v1/resources?page=1&page_size=16&category=video&sub_kind=clip' 
```

代码位置：
```
@internal/server/http/controllers/v1/resource.go:51-107
```

原因及修复方式
## Bug原因

`source=board_video`、`type=video` 在 `inferCategorySubKindFromLegacy` 里被当成普通视频，一律落成 `video` + `clip`，所以分镜生成的 MP4 会混进「剪辑视频 → 片段」。
## 修复说明

1. `internal/services/resource/catalog.go`
    - 增加 `SubKindShotVideo = "shot_video"`。
    - `source` 为 `board_video` 时归为 `board` + `shot_video`。
    - `deriveTypeFromCategorySubKind`：`(board, shot_video)` → `type=video`。
2. `internal/process/runners/mq/handler_generate_board_video_phase.go`
    - 写入资源时显式 `Category: board`、`SubKind: shot_video`，并在 `extra` 里增加 `generate_video_id`。
3. `internal/services/resource/copy.go`
    - `board` + `shot_video` 走 `copyBoardShotVideoTx`；与 `copyVideoClipTx` 共用 `copyGenerateVideoBackedBoardShotTx`（目标分类不同）







场景图查询出现了角色参考图
 资源库-图片资源-场景图 查询出现了角色参考图
# 资源库：重复记录与场景图误归类（原因与修复）

## 问题一：上传后出现两条 `sta_resources`

**原因**  
同一文件先调 `POST /api/v1/storage/upload`（或 `…/image/upload`），成功后会自动 `AddResource`（`source=storage_upload`）；再调 `POST /api/v1/resources/uploads` 登记时又会 `AddResource`（`source=user_upload`）。两条链互不感知，同一 `file_url` 产生两行。

**修复**  
上传接口仅负责 OSS 与团队空间用量，**不再**写入 `sta_resources`。资源库统一由 `POST /api/v1/resources/uploads` 登记；上传响应中不再返回 `resource_id`。

**涉及代码**  
`internal/server/http/controllers/v1/storage.go`（`Upload`、`UploadImage`）。

---

## 问题二：「图片 → 场景图」列表里出现非场景图

**原因**（历史/并存逻辑）  
`AddResource` 在未显式传入 `category`/`sub_kind` 时依赖 `inferCategorySubKindFromLegacy`：对 `type=image`，除 `scene`、`prop_reference`、`character_outfit` 等少数 `source` 外，**默认一律推断为 `sub_kind=scene`**。通用上传曾使用 `storage_upload` 等未枚举 `source`，会被标成场景图；与「场景图」列表筛选条件一致，因而混入。

**与修复一的关系**  
去掉上传自动落库后，**不再**因「只上传」就新增一条误标为场景图的记录；但若仍有其它路径以 `image` + 未覆盖 `source` 且无类目入库，推断逻辑仍可能影响数据（需单独治理历史数据或扩展 `infer…` / 显式传参）。

---

## 前端/调用方注意

- 上传拿到 `url` 后，需再调用 `POST /api/v1/resources/uploads` 传入 `category`、`sub_kind`、`file_url` 等完成入库。
- 勿再依赖上传接口返回的 `resource_id`。

## Bug：资源库列表里「角色图片」没有版本号

### 原因

- 列表形状：`GET /api/v1/resources` 返回的 `LibraryResourceItem` 原先没有顶层 `version`，版本信息若存在只会出现在 `extra` 里。
- 数据不一致（根因）：主形象在 `character_image` MQ 成功写入 `sta_resources` 时，`extra` 里带有 `version`（数值版号）；妆造（`source=character_outfit`）在两条路径入库时 `extra` 未写 `version`：
    - MQ 成功分支里 `outfitRow != nil` 的 `AddResource`；
    - 创建/更新妆造并带图时的 `indexOutfitResource`。
- 因此前端若读 `extra.version` 或期望与主形象字段一致时，妆造类条目会表现为没有版本号；这不是列表层「删掉字段」，而是写入侧漏字段。

### 修复说明

1. MQ 妆造入库（[`handler_character_generate.go`](vscode-file://vscode-app/Applications/Cursor.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/internal/process/runners/mq/handler_character_generate.go)）  
    在 `character_outfit` 的 `Extra` 中增加 `"version": ver.Version`，与主形象语义一致：`extra.version` = 对应 `sta_character_versions.version`。
    
2. 妆造索引入库（[`internal/services/character/outfit.go`](vscode-file://vscode-app/Applications/Cursor.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/internal/services/character/outfit.go)）  
    `indexOutfitResource` 内按 `CharacterVersionID` 查询 `sta_character_versions`，将 `cv.Version` 写入 `extra["version"]`；索引失败时 向上返回错误（不再静默 `_ =`），避免「妆造已保存但资源索引不完整」无人知晓。
    
3. 列表兼容前端（[`internal/services/resource/list.go`](vscode-file://vscode-app/Applications/Cursor.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/internal/services/resource/list.go)）  
    为 `LibraryResourceItem` 增加 `version`（`json:"version,omitempty"`），在 `extra["version"]` 能解析为正整数时 填入顶层字段；缺失或非正数则不输出，避免误当成 `0`。
    
4. 历史数据（[`deploy/migrations/099_backfill_character_outfit_resource_extra_version.sql`](vscode-file://vscode-app/Applications/Cursor.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/deploy/migrations/099_backfill_character_outfit_resource_extra_version.sql)）  
    对已有 `category='character'` 且 `source='character_outfit'`、`extra` 含 `character_version_id` 且尚无 `version` 键的行，按 `sta_character_versions.id` 关联回填 `extra.version`（不覆盖已有 `version`）。
    

### 验证要点

- 新生成妆造图、`indexOutfitResource` 触发的条目：`extra.version` 与对应版本行一致，`GET /resources` 顶层 `version` 与 `extra.version` 一致（在有值时）。
- 执行迁移后：旧妆造资源 `extra.version` 被补齐；列表顶层 `version` 随之可见（解析逻辑不变）。

---

以上内容可直接粘贴进 changelog / PR / 缺陷单「原因」与「修复」栏使用。


# 未返回section_num
分镜资产-关键帧图-未返回section_num
## Bug 原因

资源库接口 `GET /api/v1/resources` 在筛选 分镜类关键帧（`category=board`、`sub_kind=keyframe`、`source=keyframe` 等）时，前端需要 `section_num`（本集内第几段），与 `episode_num` / `shot_num` 一起唯一定位到「某集 · 某段 · 某镜」。

实际情况是：

1. 列表 DTO 最初没有在响应里体现段号  
    `LibraryResourceItem` 长期没有 `section_num` 字段，只把数据库里的 `extra` 原样返回；若 `extra` 里本身就没有段号，接口整体就不会出现 `section_num`。
    
2. `sta_resources.extra` 在写入关键帧资源时未写入段号
    
    - MQ 里关键帧生成成功后 `AddResource` 只写了 `video_frame_id`、`board_version_id`、`shot_num`、`episode_num` 等，未写入 `section_num`，尽管同一条流水里已通过 `ResolveBoardSectionShot` 拿到了 `Section.SectionIndex`。
    - 管理端从 `sta_video_frames` 回补资源时同样只设了集号/镜号，没有把解析结果里的段落序号写入 `extra`。
    - 跨项目复制关键帧 路径里 `extra` 更瘦，同样缺少与线上一致的段落、集号等信息。

以上均未改表结构即可避免，因为段号可以落在现有 JSONB `extra` 中；根因是写入与列表映射不完整，不是缺列。

---

## 修复说明

1. 写入侧：在 `extra` 中补齐 `section_num`
    
    - `handler_generate_video_frame.go`：关键帧落库资源时，在 `resolved.Section != nil` 时设置 `extra["section_num"] = resolved.Section.SectionIndex`。
    - `internal/services/admin/resources.go`：回补关键帧资源时，先保留 `ResolveBoardSectionShot` 的完整结果，再组装 `extra`，在 `resolved.Section != nil` 时写入 `section_num`。
    - `internal/services/resource/copy.go`：`copyBoardKeyframeTx` 与主路径对齐，写入 `board_version_id`、`episode_num`，并在有 section 时写入 `section_num`。
2. 读取侧：列表显式返回 `section_num`
    
    - `LibraryResourceItem` 增加 `section_num` 字段。
    - `internal/utils/tools.go` 增加 `GetInt64FromAny`，从 `extra["section_num"]`（兼容 JSON 常见的 `float64` / `json.Number` / 整型）解析后赋给列表项，保证 `GET /api/v1/resources` 的 `list[].section_num` 与 `extra` 一致。
3. 存量数据  
    已在库里且 `extra` 无 `section_num` 的旧记录，不会自动变更；新生成、回补新建、复制产生的新记录会带上段号。若要对齐历史数据，需对 `sta_resources.extra` 做数据修补（仍不涉及表结构变更）。