---
layout: post
toc: true
title: "Pavo 3D-BOX 深度研究：LLM Agent 驱动浏览器白模预演的逆向分析与开源对照"
excerpt: "Pavo 3D-BOX 是国内产品 Pavo AI 的'3D 预演台'：用户输入一句场景描述，DeepSeek Agent 通过流式工具调用在浏览器里实时搭出一个可编辑的 Three.js 白模场景与分镜运镜。本文不走'产品评测'路线，而是把它当作黑盒做了一次完整的动态逆向：真实注册态触发一次白模生成、在页面上下文安装 fetch/XHR/WS/SSE 观测器捕获完整 API 流水线、对约 10MB 生产环境 JS 产物做内存级正则取证（不落盘），从压缩产物中还原出 18 个 Zod schema 工具注册表、glTF 对齐的场景文档 schema、操作日志 + revision 乐观并发的同步协议、以及白模默认材质的原始代码。再用开源界两条流派（代码生成流 vs 工具调用流）做逐项对照——最硬的一条证据是：Pavo 的 Agent 工具词汇与 three.js 官方编辑器 Add 菜单源码逐字对齐，连 Box 默认尺寸 1,1,1 都一致。最后诚实列出本次研究未能验证的边界。"
categories: AI
tags: [AI, Agent, Three.js, WebGL, 逆向工程, 白模, Previz, AG-UI, glTF, DeepSeek, OpenSource]
author:
  - vortezwohl
  - 吴子豪
  - ZCode
---

> 本文成文于 2026-11-18。研究对象是 Pavo AI 的 3D-BOX（白模预演）功能[^1]。信息来自四条研究通道：①**动态实测**——在已登录的真实会话中触发一次白模生成，在页面上下文安装网络观测器（fetch/XHR/WebSocket/SSE hook）捕获 API 流水线；②**运行时取证**——读取页面资源清单、localStorage、认证机制；③**静态产物分析**——抓取约 10MB 生产环境 JS chunk 在内存中做正则取证（全程不写盘）；④**开源对照**——对六组同类开源项目与 two 项标准规范（glTF 2.0、AG-UI）做源码级交叉比对。文中论断分两档标注：【实锤】= 有原始代码/请求为证；【推断】= 由间接证据推得。逆向结论仅反映 2026-11-18 时点的产品形态，引用前请自行复核。

## 一、研究对象与一句话结论

Pavo 3D 预演台（3D-BOX）的官方描述是"Agent 智能搭建场景，精准编排镜头，让创意在制作前清晰可见，把灵感拍成下一幕"[^1]。用户输入一句中文场景描述（如"音箱桌面产品展示：白色桌面上放着一台深灰色音箱，镜头从远处缓慢推进并环绕音箱一周"），选择一个大模型（实测时为 DeepSeek V4.1 Flash 1017），系统自动生成一个**可编辑的 3D 白模场景 + 分镜运镜**——即影视行业所说的 previz（预演/动态分镜）的 AI 化。

本研究的一句话结论：

> **Pavo 3D-BOX 的白模 = DeepSeek Agent 通过 18 个 Zod schema 约束的场景工具（词汇表几乎逐字对齐 three.js 官方编辑器的 Add 菜单），以 AG-UI 风格的 SSE 事件流驱动浏览器内原生 Three.js 渲染的程序化几何与 GLB 资产，配合纯色 PBR 材质与阴影实时渲染；场景以"实体快照 + 操作日志 + revision 检查点"的协同编辑式协议持久化。**

它不是文生视频，不是 NeRF/3D 高斯泼溅，也不是"LLM 写代码生成场景"——它是**结构化工具调用流**的 LLM 场景构建，与开源界的 blender-mcp、Holodeck、InteriorAgent 同派，但把整个"编辑器即运行时"完全搬进了浏览器。

## 二、研究方法：四条证据通道与零信任原则

本次研究的方法论核心是零信任：不轻信任何单点输出，每条论断都要求至少一条可复核的证据链。

### 2.1 动态实测（真实操作）

- 使用浏览器自动化在**已登录的真实账号**（右上角可见积分余额）中打开 `app.pavo-ai.cn/3dbox`[^1]；
- 在页面 JS 上下文中安装观测器：包装 `window.fetch`、`XMLHttpRequest.prototype.open/send`、`WebSocket`、`EventSource`，将请求 URL、方法、请求体、状态码与响应体片段记录到 `window.__netLog`（上限截断防止内存膨胀）；
- **真实点击发送**一次生成任务，捕获从项目创建到 Agent 流式工作的完整请求链；
- 对 3D 视口做 WebGL 原型 hook（`getUniformLocation`/`drawElements`/`uniformMatrix4fv` 等调用计数与 uniform 名采集），尝试渲染指纹鉴定【实锤——结果是该编辑器按需渲染，构建期间无帧产出，指纹采集为空，改走产物取证】。

### 2.2 运行时取证

- `performance.getEntriesByType("resource")` 列出全部已加载 JS 资源（初始 19 个 → 编辑器水合后 43 个）；
- localStorage 全量检查：发现 JWT 令牌、场景修订号键（`3d-box:server-revision:<uuid>`）、以及一个 26KB 的 `errorMessage`（后证实为腾讯云 IM SDK 错误码表静态缓存，属干扰项，排除）[^16]；
- 用页面内 JWT 复现认证请求（`Authorization: Bearer`），确认 API 认证方式。

### 2.3 生产产物内存级取证

对 43+4 个 Next.js chunk（合计约 10MB）按特征正则做两轮筛选：第一轮按引擎特征（`WebGLRenderer`/`MeshStandardMaterial`/`GLTFLoader`）与业务特征（`scene3d`/`toolName`/`baseRevision`）定位核心模块；第二轮对命中 chunk 做上下文级提取（命中点前后 300–600 字节窗口），还原出工具注册表、场景 schema、同步协议的原始代码。全程在内存中进行，不落盘。

### 2.4 开源对照

对同类开源项目做源码级取证：three.js 官方编辑器的 `Menubar.Add.js`、AG-UI 协议事件规范、Holodeck/InteriorAgent/blender-mcp/three-live/BlenderGPT 的架构文档与工具定义，再对 glTF 2.0 规范做字段级对齐。对照的目的不是"找相似"，而是**用开源实现的公开源码反推闭源产品的工程决策理由**。

## 三、研究过程实录

### 3.1 触发一次真实生成【实锤】

在 `/3dbox` 输入提示词并发送后，捕获到完整 API 流水线（观测器实录，有舍入）[^2]：

```
1. POST /api/studio/create
   body: {name:"", config:"{\"animationPreview\":{\"version\":1,
          \"prompt\":\"音箱桌面产品展示：…\",\"generationSettings\":{
          \"aspectRatio\":\"16:9\",\"durationSeconds\":\"auto\",
          \"agentMode\":\"deepseek-v4.1-flash-1017\",
          \"reasoningModel\":\"deepseek-v4.1-flash-1017\"}}}"}
2. GET  /api/studio/detail?uuid=<projectId>          （轮询项目详情）
3. GET  /api/v1/pixa/model_options?scene_code=scene3d （拉取可选模型列表）
4. POST /api/v1/chat/conversation                     （创建会话，body:{}）
5. POST /api/studio/update                            （更新 config，带 revision）
6. POST /api/v1/chat/stream                           ← 核心：Agent 启动
   body: {"conversation_id":"381355910114719168",
          "prompt":"音箱桌面产品展示：…",
          "mode":"scene3d",
          "model":"deepseek-v4.1-flash-1017",
          "ratio":"16:9","duration":"auto","files":[],
          "extra_context":{"scene3d":{
              "projectUuid":"9cc717af…",
              "baseRevision":2,
              "selectedIds":[]}}}
```

随后页面 SPA 跳转到 `/3dbox-view?projectId=…` 编辑器，左侧出现网格地面视口（"正在构建场景……"），右侧 Agent 对话流依次输出："我先划块掉空间，并编写场景量化代码"→"已完成 1 steps"→"现在并行构建六个基础模块"→"思考中"。**Agent 跑完 6 个步骤后弹出版"积分不足"**（账户余额 1,300 积分不足以完成该任务）——生成中断。这个中断本身就是证据：**计费粒度是 Agent step 级**【推断：步数与积分的换算比例未获得】。

### 3.2 钩子存亡与取证路线修正【实锤】

研究过程中踩到的坑值得记录：该站的卡片导航是**原生 `<a>` 硬导航**而非 SPA 路由，每次硬导航都会重建页面 JS 世界，导致已安装的观测器连同捕获数据一并丢失（发生过两次）[^17]。因此后期改为：①主动拉取（用页面内 JWT 直接调 `studio/detail`）+ ②静态产物取证两条不依赖页面存活期的路线。这一教训对同类研究的可复现性有直接意义：**观测器必须在每次硬导航后重装，或改用 CDP 层的网络捕获**。

### 3.3 产物定位：三步锁定编辑器核心【实锤】

- 第一轮（43 个 chunk）按特征命中：`6345-fa7014acb4f2ace8.js`（244KB，`toolCall`×101、`api`×16、`scene3d`×7）= API 客户端 + SSE 解析器；`e8b8ea12`（351KB，`WebGLRenderer`×35、材质类×43）= Three.js 核心库；`e31cea27`（366KB）= Three.js 材质/扩展模块；`9767`（43KB，`GLTFLoader`×24）= GLTF 加载器。
- 第二轮补扫首轮 HTML 未含的 4 个动态加载 chunk，命中 `1184.3b801f5e5a5cda52.js`（266KB，`sceneConversationId`×5、`3D-BOX`×3、`WebGLRenderer`×1）= **编辑器核心**[^4]。
- 第三轮对 `1184` 与 `6345` 做上下文窗口提取，得到下文第四节的全部关键代码。

### 3.4 一次被排除的干扰项【实锤】

localStorage 中 26KB 的 `errorMessage` 一度疑似"生成中断的完整管线错误"，解析后证实为**腾讯云 IM SDK 的错误码表中英文案**（该产品同时内置 IM 协作能力），与 3D 管线无关。记录于此以防后续研究者重复踩坑。

## 四、研究结果：系统架构全解

### 4.1 总体架构

```
提示词 ──► Next.js/React 前端 (app.pavo-ai.cn)
             │ POST /api/studio/create          创建项目（config 存 prompt/模型/比例/时长）
             │ POST /api/v1/chat/conversation   创建会话
             │ POST /api/v1/chat/stream         mode:"scene3d" + baseRevision + selectedIds
             ▼
       服务端 Agent（DeepSeek V4.1 Flash，可切换模型档位）
             │  SSE 流式事件: AgentThinking/Reasoning/TaskList/
             │  ToolCallStart(toolName+toolInput)/toolCallResult/
             │  ContextCompacting/MemoryUpdate/GenerationArtifact
             ▼
       浏览器 Web Worker 解析 → Zod 校验 → 执行到 Three.js 场景图
             ▼
       WebGL2 白模渲染（原生 Three.js，非 react-three-fiber）
             ▼
       操作日志回传 POST /api/studio/update（baseRevision 乐观并发）
```

### 4.2 Agent 工具注册表：18 个工具全表【实锤】

从 chunk `1184` 中提取的工具注册表（原始变量 `M={…}`，schema 构建器为 Zod 风格：`d.Ik`=`z.object`、`d.k5`=`z.enum`、`d.Yj`=`z.string`、`d.zM`=`z.boolean`、`d.PV`=`z.tuple`）：

| 工具名 | 参数 schema | 作用 |
|---|---|---|
| `timeline.shots.set` | `{timeline: {version:1, duration≥0, shots:[{cameraId, start, end}]}}` | 设置分镜时间线 |
| `world.settings.set` | `{settings: {type:"solid", color:[f,f,f], strength≥0}}` | 世界环境（纯色背景+强度）|
| `object.create.primitive` | `{primitive: enum[box,plane,uvSphere,cylinder,cone], transform, name?}` | **程序化基元（白模主体）** |
| `object.create.from_glb` | `{assetId, transform, name?}` | 从 GLB 资产库创建 |
| `camera.create` | `{projection: perspective{verticalFov,near,far} \| orthographic{verticalSize,near,far}, transform, name?}` | 创建相机 |
| `camera.projection.set` | `{projection}` | 改投影 |
| `camera.transform.set` / `object.transform.set` / `entity.transform.set` | `{transform: {position:[3], rotation:[4]四元数, scale:[3]}}` | 变换（三套别名共存，见 4.6）|
| `camera.active.set` | `{}` | 激活某相机为当前视角 |
| `camera.rig.set` | `{rig: {aspect, dofFocusName, path:{waypoints, anchor{origin,focus,radius}, aim, focus, focusParent:"aim"\|"path"}, animation:{times, positions(3N), quaternions(4N)}}}` | 相机 rig：路径/锚点/DOF/关键帧动画 |
| `light.create` | `{properties:{type: enum[directional,point,spot], color:[3], intensity, castShadow}, transform}` | 灯光 |
| `light.properties.set` | 同上 properties | 改灯光 |
| `object.material.set` | `{material:{baseColor:[f,f,f,f], metallic, roughness}}` | **纯色 PBR 材质** |
| `entity.rename` / `object.rename` | `{name}` | 重命名 |
| `entity.delete` / `object.delete` | `{}` | 删除 |

`object.material.set` 的参数 `{baseColor, metallic, roughness}` 与 glTF 2.0 的 `pbrMetallicRoughness {baseColorFactor, metallicFactor, roughnessFactor}` **逐字段对齐**；`camera perspective {verticalFov, near, far}` 即 glTF 的 `{yfov, znear, zfar}`；`transform` 的四元数旋转即 glTF node 的 TRS；`animation {times, positions(3N), quaternions(4N)}` 即 glTF animation sampler 的平移轨道+旋转轨道语义。**场景文档是 glTF 对齐的活体场景图**【实锤——字段级对照】[^8]。

### 4.3 白模的本质：默认材质原始代码【实锤】

编辑器内新建基元的工厂函数（chunk `1184` 原文，模块别名还原为 Three.js 原名）：

```js
eq = (e) => new THREE.Mesh(
  "box"===e       ? new THREE.BoxGeometry(1, 1, 1)
: "uvSphere"===e  ? new THREE.SphereGeometry(0.65, 40, 24)
:                   new THREE.ConeGeometry(0.65, 1, 4),
  new THREE.MeshStandardMaterial({ color: 8176895 /* #7cc4ff */, roughness: 0.45 })
)
```

即：**无贴图、单一漫反射色、中等粗糙度的 MeshStandardMaterial**——这就是"白模/灰模"预演外观的来源（新摆对象默认 `#7cc4ff` 浅蓝，Agent 生成对象经由 `object.material.set` 或 GLB 自带材质覆盖）。手动添加工具同样创建了 sphere/box/pyramid(→cone)、相机（`PerspectiveCamera(45, 1, .01, 1e4)`）、太阳光（`DirectionalLight(0xffffff, 3)`）、点光（`PointLight(0xffffff, 50)`）、聚光（`SpotLight(0xffffff, 75, 0, 45°, .25, 2)`），对象挂 `userData.hf_id = crypto.randomUUID()` 与 `userData.hf_asset`（资产溯源）。

### 4.4 渲染层【实锤】

chunk `1184` 中的渲染器初始化原文（别名还原）：

```js
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;   // E.QP0
scene.background = new THREE.Color("#343434");       // 编辑器深灰底
// 自定义网格地面着色器 uniform:
//   uCellColor / uSectionColor(0xc4c4c4) / uAxisXColor(0xd96a63 红) / uAxisZColor(蓝)
```

要点：**原生 Three.js 命令式渲染，未使用 react-three-fiber**（运行时无 r3f 特征，全页面仅有 2 处 react-three 字符串且位于无关 chunk）；类 Blender 的自定义网格地面着色器；开启阴影；各向异性按设备上限自动设置（`getMaxAnisotropy`）。渲染引擎本体由三个 chunk 组成：核心库（351KB）、材质/扩展模块（366KB）、GLTFLoader（43KB）[^5][^6]。

### 4.5 SSE 事件协议与 Web Worker【实锤】

chunk `6345` 的 SSE 解析器暴露的块类型全集：

```
AgentThinking / Reasoning / Message / File / ToolCallStart / toolCallResult
TaskList / MemoryUpdate / ContextCompacting / GenerationArtifact
GenerationRetryParams / ShortDramaPhaseStart / ShortDramaVisualBatch
```

关键细节：`ToolCallStart` 携带 `{toolCallId, toolName, toolInput}`；对 `load_skill` 工具有专门的二进制/JSON 双格式解析分支（**Agent 具备技能热加载机制**，服务端配套 `/v1/skills` 系列 API）；解析器内部维护 `ttftMs`（首 token 延迟）统计；事件被映射为 Worker 协议（`{WorkerStart, WorkerMessage, WorkerToolCall, WorkerToolResult…}`）后进入 UI 层——**Agent 流处理运行在 Web Worker 中**，避免阻塞渲染线程。流式接口支持断线续传：`POST /v1/chat/stream/resume {conversation_id, from_seq}`【实锤】[^3]。

### 4.6 场景同步协议：操作日志 + revision 乐观并发【实锤】

chunk `1184` 中的两个关键 schema（别名还原）：

```js
// 单个操作事件
Operation = { operationId: string, entityId: string,
              sequence: int≥0, status: string, kind: string,
              payload: map(string, any) }
// 场景文档
SceneDoc  = { entities: [{ id, kind, deleted?, state: map }],
              operations: [Operation],
              checkpoint: { sequence } | null }
```

配套机制：API 层的 `studio/update` 封装函数带**乐观并发重试**——提交携带 `baseRevision`，失败后重读 `studioMeta.revision`，若远端 revision 已前进则重试更新（原始代码注释级还原：`if(r===e.baseRevision) throw n;`）；`localStorage["3d-box:server-revision:<uuid>"]` 记录本地已见修订号；请求侧 `extra_context.scene3d.baseRevision` 与 `selectedIds`（用户当前选中对象，截断至 100 个）随每次对话上报——**Agent 知道用户当前选中了什么**。这套"实体快照 + 有序操作日志 + 序号检查点 + 版本检查"是协同编辑领域（op-based CRDT / Yjs 一族）的标准架构；即便在单人会话里也必须存在，因为 **Agent 与用户是两个并发写者**【机制实锤；并发冲突场景未实测】。

### 4.7 资产体系【实锤】

- GLB 上传强校验（chunk `1184` 原文逻辑）：扩展名必须 `.glb`、大小 ≤ `0x6400000`（100MB）、二进制魔数校验（`glTF` = `0x46546c67`、版本 = 2、长度字段 = 实际字节数）；
- 本地资产库：IndexedDB 数据库 **`3d-box-local-assets`**（objectStore: `assets`，keyPath: `id`，记录含 `{id, name, bytes, savedAt}`）；
- 资产元数据 schema 含 `glb_sha256` 与 `triangles`（三角形数）字段；
- 外部资源代理：`/api/proxy-file?url=…`（规避第三方资源 CORS）；
- "项目资产"面板（实测时为空态"空场景"）承担资产入库入口【GLB 上传→资产库→`from_glb` 引用的完整链路未端到端实测——见第六节】。

### 4.8 API 端点全景【实锤】

从 chunk `6345` 提取的全部端点（按功能分组）：

```
3D 会话:  /v1/chat/stream · /v1/chat/stream/resume · /v1/chat/stream/cancel · /v1/chat/conversation
项目:     /studio/create · /studio/detail · /studio/update · /studio/delete · /studio/list
          /project/create · /project/<id> · /folder/entries · /fork
模型:     /v1/pixa/model_options?scene_code=scene3d · /v1/pixa/studio/assets/import
技能:     /v1/skills · /v1/skills/<id> · /v1/skills/tabs · /v1/skills/favorites
资产:     /asset-extract · /assets · /assets/characters(/states) · /assets/props(/states) · /assets/scenes(/states)
短剧:     /outline(/confirm/retry) · /episodes(/storyboard/retry/resume) · /shots · /segments(/reorder)
          /storyboard/detail · /storyboard/generate · /media/upload · /keyframe/generate
          /video/generate · /compose · /reference-images · /auto-sync-assets · /stop · /subtitle
文件:     /api/proxy-file?url=…
```

可见 3D-BOX 只是 Pavo 平台的一个场景域（`scene3d`），同一套 Agent 会话协议还服务于短剧/分镜/成片域（`ShortDramaPhaseStart`/`ShortDramaVisualBatch` 事件即来自该域）。

## 五、开源对照：用公开源码反推闭源决策

### 5.1 两条流派与 Pavo 的站位

开源界"LLM → 3D 场景"明确分成两派[^13][^14][^12][^10][^11][^15]：

| 流派 | 代表 | 机制 | 缺陷 |
|---|---|---|---|
| 代码生成流 | google/three-live（LLM 写 Three.js 代码→沙箱编译→GLB）、BlenderGPT（LLM 写 bpy Python） | 代码即中间表示 | 场景不可结构化编辑/回放；沙箱内执行任意生成代码 |
| 工具调用流 | blender-mcp（粗粒度 MCP 工具）、Holodeck（CVPR 2024，LLM 语义约束→DFS/MILP 求解器布局）、InteriorAgent（检索→DSL→lint→VLM 批评循环） | 受 schema 约束的场景操作 | 工具设计成本高；表达力受工具集限制 |

Pavo 是纯粹的**工具调用流**。这个选择的根本原因在浏览器语境下是刚性的：**不能在用户浏览器里执行 LLM 生成的任意代码**（three-live 为此自研了 Rust 沙箱编译器，而 Pavo 直接绕开了这个问题），且白模生成后用户要继续手动编辑——只有结构化场景图才能支持可编辑、可回放、可 diff。

### 5.2 最硬的一条证据：工具词汇 = three.js 官方编辑器 Add 菜单【实锤】

取证 three.js 官方编辑器源码 `editor/js/Menubar.Add.js`（dev 分支）[^7]：

```js
// three.js 编辑器 Add 菜单（源码原文摘录）
new THREE.BoxGeometry( 1, 1, 1, 1, 1, 1 )        // ← Pavo: "box"，默认尺寸 1,1,1 完全一致
new THREE.SphereGeometry( 1, 32, 16, … )          // ← Pavo: "uvSphere"
new THREE.CylinderGeometry( 1, 1, 1, 32, … )      // ← Pavo: "cylinder"
new THREE.PlaneGeometry( 1, 1, 1, 1 )             // ← Pavo: "plane"
// 灯光: DirectionalLight / PointLight / SpotLight  ← Pavo: directional/point/spot + castShadow
// 相机: PerspectiveCamera / OrthographicCamera     ← Pavo: perspective/orthographic
// 新建网格默认材质: new THREE.Mesh( geometry, new THREE.MeshStandardMaterial() )  ← Pavo 同款
```

Pavo 的 `object.create.primitive` 五种基元、灯光三件套、双投影相机、MeshStandardMaterial——**正是 three.js 编辑器 Add 菜单的子集**（刻意舍弃了 torus/knot/多面体等装饰性几何，只留白模预演所需的占位基元；Sphere/Cone 参数改为低多边形档：`SphereGeometry(.65,40,24)`、`ConeGeometry(.65,1,4)`）。结论：**Pavo 的 Agent 工具集本质上是"把 three.js 编辑器的手动操作菜单 schema 化后交给 LLM"**。

### 5.3 协议对齐：AG-UI 家族【实锤对照，使用与否未证实】

对照 AG-UI 协议（CopilotKit 开源的 Agent-User 交互协议，17 种事件）[^9]：

| Pavo 的 SSE 块 | AG-UI 对应 |
|---|---|
| `ToolCallStart` / `toolCallResult` | `ToolCallStart` / `ToolCallResult`（**名字相同**）|
| `Reasoning` + `AgentThinking` 并存 | `ReasoningMessage*`（AG-UI 1.0 刚把 `THINKING_*` 更名为 `REASONING_*`，Pavo 两代并存像过渡期设计）|
| `TaskList`（"并行构建六个基础模块"）| `ActivitySnapshot {activityType:"PLAN"}` |
| `MemoryUpdate` / `ContextCompacting` | State/Messages 同步事件族的扩展（AG-UI 无同名事件）|
| `ShortDrama*` | AG-UI 的 `Custom` 扩展事件位 |

无法证实 Pavo 直接使用了 AG-UI，但事件命名、Start/Content/End 三段式、Worker 端 chunk 展开等设计与 AG-UI 同族。**对读者的可移植价值：要复刻这类系统，AG-UI 是现成的开放协议底座。**

### 5.4 布局策略的共同智慧：不让 LLM 直接输出坐标

Holodeck（CVPR 2024）的核心发现是：LLM 提语义约束、确定性求解器（DFS/MILP）出数值布局，资产靠大规模资产库特征检索而非生成，收益显著优于 LLM 裸坐标输出[^10]。InteriorAgent 更进一步：毫秒级 AST lint 拦截 LLM 编造的 DSL 动词/参数、几何 lint 检查物体悬浮/下沉、VLM 批评循环[^11]。回看 Pavo Agent 的自述——"**我先划块掉空间，并编写场景量化代码**"——同样在做**空间划分 + 量化布局**的中间层，而非让模型直接吐 `position:[x,y,z]`。`baseRevision`+`selectedIds` 随请求上报、`TaskList` 驱动并行构建，正对齐这一流派的工程共识。

### 5.5 Pavo 的差异化与可能短板

**差异化**：①全浏览器闭环——Holodeck 依赖 AI2-THOR/Unity，InteriorAgent/blender-mcp 依赖本地 Blender，three-live 需自研跨端沙箱编译器，只有 Pavo 把"编辑器即运行时"完全放进浏览器；②影视 previz 语义——`timeline.shots.set {cameraId, start, end}` + 相机 rig（waypoints/aim/focus 锚点/景深焦点）是分镜级相机编排，六组开源对标项目无一实现；③商业化闭环（积分按 step 计费、订阅升级）。

**开源对照暴露的可能短板**【推断】：blender-mcp 有 `look` 工具（多视角渲染自检）[^12]、three-live 有 `render`/视觉反馈闭环与 agent gym[^13]、InteriorAgent 有 VLM 批评循环[^11]——Pavo 的 SSE 事件集中未见视觉反馈类事件（`GenerationArtifact` 可能承载图片工件，未完全证实），推断其 Agent 目前主要依赖文本推理链推进，生成质量的闭环校验弱于开源前沿。

## 六、研究边界与未验证项（诚实清单）

1. **服务端 Agent 的系统提示词与编排逻辑不可见**——工具 schema 全部从前端校验层反推，服务端可能有额外的工具包装/重试/规划层。
2. **完整场景持久化格式未端到端实测**——因积分耗尽，未能观察到一个完整生成完毕的项目；`studio/detail` 返回的 config 此刻仅含 `animationPreview` 设置（prompt/模型/会话 ID），实体快照的落库格式（是否内嵌于 config 或独立存储）是高置信推断而非实锤。
3. **Agent 每步的具体 toolInput 因整页导航丢失 SSE 快照**，未能留档（协议结构已从客户端解析器完整还原，但原始流样本未留存）。
4. **资产库端到端链路未实测**——"项目资产"面板为空态，GLB 上传→`assetId` 分配→`from_glb` 加载的完整闭环未走通。
5. **渲染指纹采集为空**——编辑器按需渲染导致 hook 窗口内无帧，渲染层结论全部来自产物取证（证据等级仍为实锤，但缺运行时行为佐证）。
6. **模型名的时点性**——实测时选项为"DeepSeek V4.1 Flash 1017"（`model_options` 接口动态下发），该产品显然支持多模型热切换，其他档位未测。

## 七、结语

把这次逆向的结果压缩成一张可复用的工程蓝图：**"受约束的工具集（编辑器菜单词汇的 schema 化）+ 流式事件协议（AG-UI 家族）+ glTF 对齐的场景文档 + 操作日志/版本检查的并发控制 + 浏览器端声明式渲染"**——五件套就是"LLM Agent 在浏览器里搭 3D 白模"的完整配方。Pavo 没有用到什么黑魔法：引擎是开源的 Three.js，协议设计有开放标准 AG-UI 可对标，场景格式有 glTF 兜底，布局策略有 Holodeck/InteriorAgent 的公开方法论可循。它真正做对的是三件组合层面的决策：把工具词汇锚定在 LLM 训练语料里已大量存在的 three.js 编辑器语义上（让模型"天生会用"）；把并发控制按协同编辑标准做（Agent 与用户双写者的必然要求）；把整个运行时塞进浏览器（交付即编辑，无本地依赖）。对想复刻这类系统的团队，第六节的未验证清单同时就是风险清单；对研究者，本文的四通道取证方法（动态实测 / 运行时取证 / 产物内存分析 / 开源对照）可以整体迁移到任何同类黑盒产品上。

## 参考文献

[^1]: Pavo AI. *3D 预演台（PAVO 3D 预演台）*. app.pavo-ai.cn/3dbox（产品页面，2026-11-18 实测；页面口号"Agent智能搭建场景，精准编排镜头"、示例提示词、模型选择器"DeepSeek V4.1 Flash 1017"、积分余额与积分不足提示均来自实测截图与 DOM 快照）. <https://app.pavo-ai.cn/3dbox>

[^2]: 本研究实测数据. *API 流水线捕获*（2026-11-18）. 页面上下文 hook `fetch/XHR/EventSource` 所录：`POST /api/studio/create`、`GET /api/studio/detail`、`POST /api/v1/chat/conversation`、`POST /api/studio/update`、`POST /api/v1/chat/stream`（含完整请求体：`mode:"scene3d"`、`extra_context.scene3d.{projectUuid, baseRevision, selectedIds}`）；`GET /api/v1/pixa/model_options?scene_code=scene3d`、`GET /api/v3/subscription/assets-balance`.

[^3]: Pavo 生产环境 JS 产物 `6345-fa7014acb4f2ace8.js`（244KB，API 客户端与 SSE 解析器）. 内存取证提取：SSE 块类型全集（`AgentThinking/Reasoning/Message/File/ToolCallStart/toolCallResult/TaskList/MemoryUpdate/ContextCompacting/GenerationArtifact/GenerationRetryParams/ShortDramaPhaseStart/ShortDramaVisualBatch`）、`load_skill` 专用解析分支、`ttftMs` 统计、Worker 协议映射（`WorkerStart/WorkerMessage/WorkerToolCall/WorkerToolResult`）、`/v1/chat/stream/resume {from_seq}` 断线续传、`/v1/chat/stream/cancel`、`studio/update` 的 `baseRevision` 乐观并发重试逻辑、全部 API 端点清单. <https://app.pavo-ai.cn/_next/static/chunks/6345-fa7014acb4f2ace8.js>

[^4]: Pavo 生产环境 JS 产物 `1184.3b801f5e5a5cda52.js`（266KB，3D-BOX 编辑器核心）. 内存取证提取：18 工具注册表 `M={…}`（Zod 风格 schema）、场景文档 schema（`entities/operations/checkpoint`、`Operation{operationId, entityId, sequence, status, kind, payload}`）、`timeline/animation/rig` schema（`times`/`positions(3N)`/`quaternions(4N)` refine 校验）、白模默认材质工厂 `eq()`、渲染器初始化 `tC()`、GLB 校验函数 `C()`（魔数/版本/长度/100MB）、IndexedDB `3d-box-local-assets`、手动添加工具 `addElement`、对象→操作反序列化 `eH()`. <https://app.pavo-ai.cn/_next/static/chunks/1184.3b801f5e5a5cda52.js>

[^5]: Pavo 生产环境 JS 产物 `e8b8ea12-445eaf9087d59f60.js`（351KB，`WebGLRenderer`×35、材质类×43）与 `e31cea27-d50a54f9c3470845.js`（366KB，材质/扩展模块）. 特征计数来自正则检索，判定为 Three.js 核心库与扩展的分包.

[^6]: Pavo 生产环境 JS 产物 `9767-ab4e78649f3891ff.js`（43KB，`GLTFLoader`×24）与 `2039.5c0f863b99fedcce.js`（109KB，`MeshStandardMaterial`×5）. 判定为 three.js GLTF 加载器与材质模块的分包.

[^7]: mrdoob et al. *three.js* 官方仓库 `editor/js/Menubar.Add.js`（dev 分支）. 源码实读：Add 菜单几何词汇（`BoxGeometry(1,1,1)`、`SphereGeometry(1,32,16)`、`CylinderGeometry`、`PlaneGeometry` 等 14 种）、灯光五类、双投影相机、新建网格默认 `new THREE.MeshStandardMaterial()`. <https://github.com/mrdoob/three.js/blob/dev/editor/js/Menubar.Add.js>

[^8]: Khronos Group. *glTF 2.0 Specification*（core spec：node TRS 四元数变换、`pbrMetallicRoughness {baseColorFactor, metallicFactor, roughnessFactor}`、camera `perspective {yfov, znear, zfar}` / `orthographic {xmag, ymag, znear, zfar}`、animation sampler `{input, output}`）. Pavo schema 与之逐字段对齐的对照见本文 4.2 节. <https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html>

[^9]: CopilotKit. *AG-UI Protocol — Events*（17 种事件类型：RunStarted/StepStarted/TextMessage*/ToolCallStart/Args/End/Result/StateSnapshot/StateDelta(JSON Patch)/MessagesSnapshot/ActivitySnapshot(PLAN)/ReasoningMessage*/Subagent*/Custom 等；THINKING_* 已废弃更名为 REASONING_*）. <https://docs.ag-ui.com/concepts/events>

[^10]: Yang, Y. et al. *Holodeck: Language Guided Generation of 3D Embodied AI Environments*. CVPR 2024；GitHub 仓库 allenai/Holodeck（LLM 语义约束 + DFS/MILP 求解器布局、Objaverse 资产特征检索、AI2-THOR/Unity 渲染）. <https://github.com/allenai/Holodeck>

[^11]: KunalMGupta. *InteriorAgent*（GitHub：MCP 工具集 `retrieve/inspect/pin/ingest_glbs`、687MB embedding 资产检索（FutureHSSD 约 29k 资产）、IDSDL DSL、`lint_program` AST/几何 lint、VLM 批评循环、"plan → retrieve → author → build + critic loop → design judge" 流水线、Blender 4.5 渲染）. <https://github.com/KunalMGupta/interioragent>

[^12]: ahujasid. *BlenderMCP*（GitHub：MCP 工具 `execute_blender_code / look / get_scene_info / generate_3d / search_assets / import_asset` 等 9 个；粗粒度工具 + AI 自写 Python 的混合架构；多视角渲染自检）. <https://github.com/ahujasid/blender-mcp>

[^13]: Google. *three-live*（GitHub：AI-first Native 3D；LLM 直接生成 15–30 行 Three.js 代码→Rust Boa/QuickJS 沙箱编译→GLB→Filament/RealityKit/WebGL2；`editScene` 编译失败自动修复；MCP 工具 `validate/render/convert/from_image/gym`；视觉反馈闭环 REFERENCE\|RENDER\|OVERLAY）. <https://github.com/google/three-live>

[^14]: gd3kr. *BlenderGPT*（GitHub：GPT-4/3.5 英语指令→bpy Python 脚本→Blender 场景，代码生成流原型）. <https://github.com/gd3kr/BlenderGPT>

[^15]: zhuyansen. *awesome-3d-skills*（GitHub：Three.js/Web3D agent 技能与 MCP 收录清单，34 个 Web 3D 项目条目，含 CloudAI-X/threejs-skills、img2threejs、pascalorg/editor 等）. <https://github.com/zhuyansen/awesome-3d-skills>

[^16]: 本研究实测数据. *运行时取证*（2026-11-18）：`performance.getEntriesByType("resource")` 资源清单（初始 19 → 水合后 43 个 chunk）、localStorage（JWT `token`、`3d-box:server-revision:<uuid>`=2、`userinfo`、腾讯云 IM 错误码表缓存 `errorMessage`）、页面 `textarea` 输入与"积分不足"弹窗、Agent 对话流文案（"我先划块掉空间，并编写场景量化代码"/"已完成 1 steps"/"现在并行构建六个基础模块"）.

[^17]: 本研究方法说明. *观测器实现与失效记录*：fetch/XHR/WS/SSE 包装器写入 `window.__netLog`；WebGL 原型 hook（`getUniformLocation/drawElements/useProgram/uniformMatrix4fv`）采集窗口内零帧（编辑器按需渲染）；两次整页硬导航导致钩子与捕获数据丢失的经过与路线修正（转 JWT 主动拉取 + 静态产物取证）.
