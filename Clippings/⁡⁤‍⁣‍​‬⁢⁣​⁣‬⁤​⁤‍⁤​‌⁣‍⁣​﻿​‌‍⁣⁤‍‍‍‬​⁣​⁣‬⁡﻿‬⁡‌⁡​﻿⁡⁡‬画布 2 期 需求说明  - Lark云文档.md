---
title: "⁡⁤‍⁣‍​‬⁢⁣​⁣‬⁤​⁤‍⁤​‌⁣‍⁣​﻿​‌‍⁣⁤‍‍‍‬​⁣​⁣‬⁡﻿‬⁡‌⁡​﻿⁡⁡‬画布 2 期 需求说明  - Lark云文档"
source: "https://chengduduck2.jp.larksuite.com/docx/SoWWd9RwSofToUxtpagjCAVjp2c"
author:
published:
created: 2026-06-01
description:
tags:
  - "clippings"
---
2026年6月1日

分阶段：6.7日前，完成新建画布、空白画布、画布引导、思考模块视觉改造

1\. 功能范围

画布 2 期提供多模态自由创作工作台。用户从临时画布开始，创建并上传文本、图片、视频、文档、音频等多模态内容，添加世界观、角色、场景、大纲、剧本、镜头、片段组、合成等业务节点，通过连线、拖拽和框选组织上下文，使用 Agent 围绕当前节点、当前画布、框选集合、@节点或@资产执行创作任务。自由画布生产的内容在接入根节点后填充到原有框架的结构化容器。

<table><tbody><tr><td rowspan="1" colspan="1"><p></p><p>模块</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>功能</p><p></p></td></tr></tbody></table>

<table><colgroup><col width="146"> <col width="674"></colgroup><tbody><tr><td rowspan="1" colspan="1"><p></p><p>入口与临时画布</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>新建入口、进入空白画布、临时画布保存为临时画布、再次打开恢复</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p>空白引导</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>空白画布展示引导话术，点击话术回填输入框；思考模块固定在左上并带动效</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p>左侧工具栏</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>添加节点、上传文件、打开资产库、切换选择与连线模式</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p>多模态节点</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>创建文本、图片、视频、音频、文档节点，支持空状态与素材填充</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p>业务节点</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>创建世界观、角色、场景、大纲、剧本、镜头、片段组、合成、完整视频节点</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p>上传</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>上传图片、视频、文档、音频，展示加载态，结果加入画布或资产库</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p>资产库</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>轻量资产库，浏览多模态素材，加入画布或填充当前节点</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p>拖拽填充</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>拖拽多模态产物或业务节点填充空状态或替换内容</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p>框选</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>框选多个节点，集合引用到输入框</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p>连线与依赖</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>建立、取消连线，按节点类型判断关系并校验依赖</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p>镜头与合成</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>片段组选择镜头合成视频节点，视频节点经合成器合成完整视频</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p>节点详情</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>节点详情弹窗，镜头节点提供右侧详情面板</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p>Agent 输入框</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>按作用域切换模型与可选参数，支持 @引用与框选引用</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p>工作台同步</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>仅同步接入根节点的结构化内容，画布独有内容不同步</p><p></p></td></tr></tbody></table>

2\. 核心对象

2.1 画布项目

画布项目承载一次创作会话，包含项目名、画布上的节点与连线、引用的资产、任务记录，以及是否已保存。

<table><tbody><tr><td rowspan="1" colspan="1"><p></p><p>对象</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>说明</p><p></p></td></tr></tbody></table>

<table><colgroup><col width="165"> <col width="634"></colgroup><tbody><tr><td rowspan="1" colspan="1"><p></p><p>项目名</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>默认未命名，用户可修改</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p>根节点</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>判断哪些内容可同步到原框架</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p>节点与连线</p><p></p></td><td rowspan="1" colspan="1"><p></p><p>当前画布上的全部节点和关系</p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p></p></td><td rowspan="1" colspan="1"><p></p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p></p></td><td rowspan="1" colspan="1"><p></p><p></p></td></tr><tr><td rowspan="1" colspan="1"><p></p><p></p></td><td rowspan="1" colspan="1"><p></p><p></p></td></tr></tbody></table>

评论（0）

跳转至首条评论

0 字

- 上传日志

- 联系客服

- 功能更新

- 帮助中心

- 效率指南

当前文档通知