---
layout: post
toc: true
title: "Codex + Blender 自动 3D 建模实践"
categories: Agent
tags: [Codex, Blender, MCP, 3D, Agent]
author:
  - vortezwohl
  - 吴子豪
  - Codex
---

如果你想让 `Codex` 直接控制 `Blender` 做 3D 建模，最简单的方式是接入 `blender-mcp`。整个流程并不复杂，本质上只有两部分：一部分是在 Blender 里安装 `addon.py`，另一部分是在 Codex 里配置 MCP server。两边都完成后，Codex 就可以直接读取场景、创建物体、修改模型。

> 本文环境：Windows + Codex + Blender 5.x

## 先安装什么

先准备下面这几样：

- `Blender`
- `Python`
- `uv`
- `blender-mcp`

安装 `uv` 可以直接用：

```bash
python -m pip install -U uv
```

安装完成后，通常就会同时得到 `uvx`。

## 第一步：获取 blender-mcp

先把仓库拉到本地：

```bash
git clone https://github.com/vortezwohl/blender-mcp.git
```

这里拉仓库的目的，主要是拿到根目录下的 `addon.py`。

## 第二步：在 Blender 里安装插件

打开 Blender 后，点击：

`编辑 -> 偏好设定`

如下图所示：

![打开 Blender 偏好设定](/images/CodexBlender自动3D建模实践/image-1.png)

然后进入“附加元件”页面，点击右上角下拉菜单中的“从磁碟安装”：

![从磁碟安装 Blender MCP 插件](/images/CodexBlender自动3D建模实践/image-2.png)

接着选择你刚才 clone 下来的 `blender-mcp` 仓库根目录中的 `addon.py`，完成安装并启用 `Blender MCP`。

## 第三步：在 Codex 里接入 MCP

打开 Codex 的配置文件，在其中加入下面这段：

```toml
[mcp_servers.blender]
command = "cmd"
args = ["/c", "uvx", "blender-mcp"]
```

配置完成后，重启 Codex。

这一步的作用是让 Codex 通过 `uvx blender-mcp` 启动对应的 MCP server。

## 第四步：连接 Blender

完成前面两步后，打开 Blender，在右侧面板中找到 `Blender MCP`。

如果插件已经正常启用，并且 Codex 也已经重启完成，那么此时 Blender 和 Codex 就可以建立连接。

## 如何判断是否接入成功

判断方式很简单，只要满足下面几点，基本就说明已经通了：

- Blender 中已经启用 `Blender MCP`
- Codex 已经重启并加载了 MCP 配置
- Codex 可以读取 Blender 当前场景
- Codex 可以执行简单操作，比如创建一个立方体

例如，如果 Codex 已经能读到场景中的 `Cube`、`Light`、`Camera`，就说明链路已经正常。

## 一句话总结

最小接入流程就是：

1. clone `blender-mcp` 仓库
2. 用 `python -m pip install -U uv` 安装 `uv`
3. 在 Blender 中从磁碟安装仓库里的 `addon.py`
4. 在 Codex 中配置 `uvx blender-mcp`
5. 重启 Codex
6. 让 Codex 连接 Blender

完成后，Codex 就可以直接控制 Blender 做自动 3D 建模。
