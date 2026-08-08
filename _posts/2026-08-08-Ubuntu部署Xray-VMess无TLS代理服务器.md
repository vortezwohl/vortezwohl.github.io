---
layout: post
toc: true
title: "Ubuntu 云服务器部署 Xray/VMess 实践笔记"
categories: Network
tags: [Ubuntu, Xray, V2Ray, VMess, v2rayN, v2rayNG, Cloud Server]
author:
  - vortezwohl
  - 吴子豪
excerpt: >-
  本教程面向完全没有代理服务器经验的初学者，手把手演示如何在 Ubuntu 22.04/24.04 云服务器上部署 Xray VMess 代理。内容从购买并登录云服务器开始，依次讲解系统更新、chrony 时间同步、使用 Xray 官方安装脚本安装服务、生成 UUID、编写和校验 JSON 配置、使用 systemd 启停服务、配置 Ubuntu UFW、防火墙安全组和云厂商网络规则，再分别说明如何在 Windows 的 v2rayN 和 Android 的 v2rayNG 中手动添加 VMess 节点。本文默认使用 TCP/RAW 传输并关闭 TLS，同时解释新版 Xray 的 method/raw 写法与旧教程 network/tcp 写法的对应关系，最后提供端口测试、日志查看、连接失败、握手失败、时间不同步等常见问题的逐项排查方法。无 TLS 配置更适合学习和受控环境，长期使用前应理解其流量特征和安全边界。
---

# Ubuntu 云服务器部署 Xray/VMess 无 TLS 代理：从零开始的完整教程

> 本文按截至 2026-08-08 的 Xray、v2rayN、v2rayNG 官方资料整理。
>
> 请只在你拥有或获授权管理的服务器上使用。代理服务器会转发网络流量，必须遵守云厂商服务条款和所在地法律法规。

## 1. 先理解这次要搭建什么

最终结构如下：

```text
v2rayN / v2rayNG
        |
        | VMess + TCP/RAW，security=none
        v
Ubuntu 云服务器:10086
        |
        | freedom 出站
        v
公网
```

本文选择的参数是：

| 参数 | 示例值 | 说明 |
| --- | --- | --- |
| 系统 | Ubuntu 22.04/24.04 LTS | 推荐使用仍在维护的 LTS |
| 服务端核心 | Xray-core | Project X 的持续维护实现 |
| 入站协议 | VMess | 客户端和服务端必须一致 |
| 传输 | TCP/RAW | 旧教程通常写作 TCP |
| TLS | 关闭 | `security: "none"` |
| 监听端口 | `10086/tcp` | 可以换成其他未占用 TCP 端口 |
| 用户凭据 | UUID | 类似密码，不能公开 |

### 1.1 无 TLS 到底意味着什么

关闭 TLS 并不表示 VMess 完全没有加密。VMess 仍然会对代理协议载荷进行保护，但 Xray 官方文档明确指出：VMess 不具备 TLS 1.3 式的前向保密，也没有正常 HTTPS 的流量外观，流量更容易被分类或识别。

因此本文的无 TLS 方案适合：

- 学习 Xray 配置和云服务器防火墙；
- 内部测试或受控网络；
- 你明确知道无 TLS 的安全边界的场景。

它不适合被误认为“隐身”或“万能安全方案”。长期使用前，应研究 TLS、REALITY 或其他带传输层安全的配置。

## 2. 准备工作

你需要准备：

1. 一台有公网 IPv4 的 Ubuntu 22.04/24.04 云服务器。
2. 云厂商控制台的安全组、防火墙或网络规则管理权限。
3. 服务器公网 IP，例如 `203.0.113.10`。
4. 可以 SSH 登录服务器的电脑。
5. Windows 上的 v2rayN，或 Android 上的 v2rayNG。

建议先在纸上记下这些变量：

```text
SERVER_IP=203.0.113.10       # 换成你的真实公网 IP
PORT=10086                   # 本文使用的 TCP 端口
UUID=稍后生成                 # 不要自行编造格式错误的 UUID
```

## 3. 第一步：通过 SSH 登录 Ubuntu

如果云厂商给你的登录用户是 `root`：

```bash
ssh root@203.0.113.10
```

如果登录用户是 `ubuntu`：

```bash
ssh ubuntu@203.0.113.10
sudo -i
```

确认当前用户：

```bash
whoami
```

看到 `root` 后继续。若没有 root 权限，后面的系统命令都需要加 `sudo`。

## 4. 第二步：更新系统并安装基础工具

先更新软件索引和已安装的软件：

```bash
apt update
apt -y upgrade
```

安装本教程需要的工具：

```bash
apt install -y curl unzip chrony ufw
```

这些工具的用途是：

| 工具 | 用途 |
| --- | --- |
| `curl` | 下载 Xray 官方安装脚本 |
| `unzip` | 解压 Xray 发布包 |
| `chrony` | 同步服务器系统时间 |
| `ufw` | 管理 Ubuntu 主机防火墙 |

## 5. 第三步：同步服务器时间

VMess 依赖系统时间。Project X 官方要求系统 UTC 时间误差保持在 120 秒以内，时区本身不重要。

启用 chrony：

```bash
systemctl enable --now chrony
timedatectl set-ntp true
timedatectl status
```

检查服务状态：

```bash
systemctl is-active chrony
chronyc tracking
```

`systemctl is-active chrony` 应输出：

```text
active
```

如果系统没有 `chrony` 服务而使用 `systemd-timesyncd`，也可以查看：

```bash
timedatectl show-timesync --all
```

只要系统已经同步，不需要同时运行多个时间同步服务。

## 6. 第四步：使用 Xray 官方脚本安装

Xray 官方安装仓库是：

```text
https://github.com/XTLS/Xray-install
```

为便于审查，先下载脚本到临时目录，不直接盲目执行远程内容：

```bash
curl -fL -o /tmp/install-release.sh \
  https://github.com/XTLS/Xray-install/raw/main/install-release.sh
```

可以查看脚本内容：

```bash
less /tmp/install-release.sh
```

按 `q` 退出 `less`。确认来源无误后安装：

```bash
bash /tmp/install-release.sh @ install
```

如果当前用户不是 root：

```bash
sudo bash /tmp/install-release.sh @ install
```

检查版本和 systemd 服务：

```bash
/usr/local/bin/xray version
systemctl status xray --no-pager
```

官方脚本默认使用这些路径：

```text
/usr/local/bin/xray
/usr/local/etc/xray/config.json
/etc/systemd/system/xray.service
/usr/local/share/xray/geoip.dat
/usr/local/share/xray/geosite.dat
```

## 7. 第五步：生成 UUID

执行：

```bash
/usr/local/bin/xray uuid
```

输出类似：

```text
a3f1d6b2-7d4c-4e9f-9f4e-1a2b3c4d5e6f
```

把实际输出复制到安全位置。客户端和服务端的 UUID 必须完全一致，包括连字符。

也可以让当前 shell 暂存一个 UUID：

```bash
UUID=$(/usr/local/bin/xray uuid)
echo "$UUID"
```

注意：关闭 SSH 会话后这个 shell 变量会消失，所以最终仍要把 UUID 写入配置并保存到密码管理器。

## 8. 第六步：编写 Xray 服务端配置

打开配置文件：

```bash
nano /usr/local/etc/xray/config.json
```

删除文件原内容，粘贴下面的完整配置，并把 `REPLACE_WITH_YOUR_UUID` 换成实际 UUID：

```json
{
  "log": {
    "loglevel": "warning"
  },
  "inbounds": [
    {
      "listen": "0.0.0.0",
      "port": 10086,
      "protocol": "vmess",
      "settings": {
        "users": [
          {
            "id": "REPLACE_WITH_YOUR_UUID",
            "level": 0,
            "email": "beginner-vmess"
          }
        ]
      },
      "streamSettings": {
        "method": "raw",
        "security": "none",
        "rawSettings": {
          "header": {
            "type": "none"
          }
        }
      }
    }
  ],
  "outbounds": [
    {
      "protocol": "freedom",
      "settings": {}
    }
  ]
}
```

### 8.1 配置逐项解释

`listen`：

```json
"listen": "0.0.0.0"
```

表示监听服务器所有 IPv4 网卡。不能写成 `127.0.0.1`，否则只有服务器本机能连接。

`port`：

```json
"port": 10086
```

表示 Xray 监听 TCP 10086 端口。云安全组、UFW、客户端配置必须使用同一个端口。

`protocol`：

```json
"protocol": "vmess"
```

表示这个入站使用 VMess。

`users[].id`：

```json
"id": "你的 UUID"
```

这是客户端身份凭据。客户端必须使用同一个 UUID。

`streamSettings.method`：

```json
"method": "raw"
```

当前 Xray 文档把原来的 TCP 传输称作 RAW。客户端界面通常仍显示 `TCP`。

`security`：

```json
"security": "none"
```

表示关闭 TLS 和 REALITY，符合本文的无 TLS 目标。

`outbounds`：

```json
"protocol": "freedom"
```

表示服务器把通过代理收到的请求直接发往公网。

### 8.2 旧版 Xray 兼容写法

一些旧版核心或旧教程使用如下字段：

```json
"streamSettings": {
  "network": "tcp",
  "security": "none",
  "tcpSettings": {
    "header": {
      "type": "none"
    }
  }
}
```

两套写法不要同时放入同一个配置。优先使用本文的 `method: raw` 写法；只有 `xray run -test` 明确提示字段不兼容时，才替换为旧写法。

## 9. 第七步：校验配置并启动服务

先只测试 JSON 和 Xray 配置：

```bash
/usr/local/bin/xray run \
  -test \
  -config /usr/local/etc/xray/config.json
```

成功输出：

```text
Configuration OK.
```

如果出现错误，常见原因是：

- UUID 没有替换，仍是 `REPLACE_WITH_YOUR_UUID`；
- JSON 少了逗号或多了逗号；
- 引号使用了中文全角引号；
- `method` 和 `network` 两套字段混用；
- 端口不是整数。

校验成功后：

```bash
systemctl daemon-reload
systemctl enable --now xray
systemctl restart xray
systemctl is-active xray
```

预期输出：

```text
active
```

确认 Xray 正在监听端口：

```bash
ss -lntp | grep 10086
```

查看最近日志：

```bash
journalctl -u xray -n 50 --no-pager
```

## 10. 第八步：配置 Ubuntu UFW 防火墙

云厂商安全组和 Ubuntu UFW 是两道独立防火墙，必须同时允许流量。

先设置默认策略：

```bash
ufw default deny incoming
ufw default allow outgoing
```

### 10.1 放行 SSH

将 `YOUR_ADMIN_PUBLIC_IP` 替换为你管理电脑当前的公网 IPv4，例如 `198.51.100.25`：

```bash
ufw allow from YOUR_ADMIN_PUBLIC_IP to any port 22 proto tcp
```

如果你不知道当前公网 IP，先临时放行 SSH：

```bash
ufw allow 22/tcp
```

确认重新 SSH 登录成功后，再收紧为固定 IP。

### 10.2 放行 VMess 端口

```bash
ufw allow 10086/tcp
```

本文使用 TCP，不需要开放 UDP：

```text
不需要开放 10086/udp
```

启用防火墙：

```bash
ufw enable
ufw status verbose
```

检查规则编号：

```bash
ufw status numbered
```

如果之前临时开放了 SSH，可以删除宽泛规则：

```bash
ufw delete allow 22/tcp
ufw allow from YOUR_ADMIN_PUBLIC_IP to any port 22 proto tcp
```

## 11. 第九步：配置云厂商安全组

在云厂商控制台找到类似“安全组”“云防火墙”“入站规则”“网络安全组”的页面。

添加以下入站规则：

| 协议 | 端口 | 来源 | 用途 |
| --- | ---: | --- | --- |
| TCP | 22 | 管理电脑公网 IP/32 | SSH 登录 |
| TCP | 10086 | 客户端公网 IP/32 或 `0.0.0.0/0` | VMess |

如果只在固定办公室或家里使用，推荐把 10086 的来源限制为固定公网 IP。

如果手机使用移动网络，公网 IP 可能变化。此时可以临时使用 `0.0.0.0/0`，但要注意：

- 这代表全球 IPv4 都可以访问该端口；
- 端口号不是密码；
- 必须依赖强 UUID；
- 应该定期升级系统和 Xray。

出站规则通常保留云厂商默认的“允许全部出站”。如果云厂商还启用了网络 ACL、子网防火墙或路由策略，也要确认出站 DNS 和公网 TCP 没有被拦截。

如果不使用 IPv6，不要为了完整而开放 `::/0`。如果服务器有 IPv6，并且客户端通过 IPv6 访问，则需要单独配置 IPv6 安全组和 UFW 规则。

## 12. 第十步：测试外部端口

### 12.1 Windows

在本地 PowerShell 执行：

```powershell
Test-NetConnection 203.0.113.10 -Port 10086
```

重点查看：

```text
TcpTestSucceeded : True
```

### 12.2 Linux 或 macOS

```bash
nc -vz 203.0.113.10 10086
```

### 12.3 结果含义

| 结果 | 说明 |
| --- | --- |
| `True` 或 `succeeded` | TCP 端口可以到达，继续检查客户端参数 |
| `Connection refused` | Xray 没监听、服务没启动或 UFW 拒绝 |
| `Timed out` | 云安全组、网络 ACL、UFW 或公网 IP 有问题 |

## 13. 第十一步：在 Windows 使用 v2rayN

官方仓库和下载地址：

```text
https://github.com/2dust/v2rayN
https://github.com/2dust/v2rayN/releases
```

官方 README 说明 v2rayN 支持 Xray core，详细使用方式以 Wiki 为准：

```text
https://github.com/2dust/v2rayN/wiki
```

操作步骤：

1. 下载与你的 Windows 架构匹配的 v2rayN 发布包。
2. 解压并启动 v2rayN。
3. 点击“服务器”菜单。
4. 选择“添加 VMess 服务器”或类似的手动 VMess 入口。
5. 按下面表格填写。

| v2rayN 字段 | 填写值 |
| --- | --- |
| 地址/Address | Ubuntu 服务器公网 IP |
| 端口/Port | `10086` |
| 用户 ID/UUID | 服务端生成的 UUID |
| AlterId | `0` |
| 加密/Encryption | `auto` |
| 传输协议/Network | `TCP` |
| TCP Header | `none` |
| TLS | 关闭、`none` 或“不加密” |
| SNI | 留空 |
| Host | 留空 |
| Path | 留空 |
| Flow | 留空 |

6. 保存节点。
7. 右键节点，选择“设为活动服务器”。
8. 打开系统代理或按需要选择 PAC 模式。
9. 访问下面的网址查看出口 IP：

```text
https://api.ipify.org
```

如果显示服务器公网 IP，说明基本连接成功。

### 13.1 为什么客户端仍然有 AlterId

很多 v2rayN 界面为了兼容旧版 V2Ray 仍显示 AlterId。当前 Xray 官方 VMess 用户配置重点是 UUID，本文服务端不配置旧式 AlterId；客户端把 AlterId 填 `0` 即可。

## 14. 第十二步：在 Android 使用 v2rayNG

官方仓库和下载地址：

```text
https://github.com/2dust/v2rayNG
https://github.com/2dust/v2rayNG/releases
```

操作步骤：

1. 安装 v2rayNG。
2. 打开应用，点击右下角 `+`。
3. 选择“手动设置”或“VMess”。
4. 填写以下值。

| v2rayNG 字段 | 填写值 |
| --- | --- |
| 地址/Address | Ubuntu 服务器公网 IP |
| 端口/Port | `10086` |
| 用户 ID/UUID | 服务端生成的 UUID |
| Alter ID | `0` |
| 加密方式 | `auto` |
| 网络/Network | `tcp` |
| 伪装类型/Header | `none` |
| TLS | 关闭 |
| SNI | 留空 |
| Host | 留空 |
| Path | 留空 |

5. 保存配置。
6. 点选该节点。
7. 点击启动按钮。
8. 第一次运行时允许 Android VPN 权限。
9. 用浏览器打开 `https://api.ipify.org` 检查出口 IP。

无 TLS 配置不需要填写证书、不需要域名、不需要 SNI，也不需要打开“允许不安全连接”来代替 TLS。

## 15. 可选：生成 VMess 分享链接

如果不想在客户端逐项填写，可以生成一个 `vmess://` 链接。下面是在 Windows PowerShell 中运行的示例，把 IP 和 UUID 换成真实值：

```powershell
$json = '{"v":"2","ps":"Ubuntu-VMess-NoTLS","add":"203.0.113.10","port":"10086","id":"REPLACE_WITH_YOUR_UUID","aid":"0","scy":"auto","net":"tcp","type":"none","host":"","path":"","tls":"","sni":""}'
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($json)).TrimEnd('=').Replace('+','-').Replace('/','_')
"vmess://$encoded"
```

导入方式：

- v2rayN：复制链接后选择“从剪贴板导入服务器”。
- v2rayNG：点击 `+`，选择“从剪贴板导入”。

这个链接包含服务器地址和 UUID，不要贴到公开群组、Issue 或截图中。

## 16. 第十三步：完整故障排查

### 16.1 先确认 systemd 服务

```bash
systemctl is-active xray
systemctl status xray --no-pager
```

如果不是 `active`，查看详细日志：

```bash
journalctl -u xray -n 100 --no-pager
```

### 16.2 确认端口监听

```bash
ss -lntp | grep 10086
```

没有输出表示 Xray 没有监听该端口。检查配置端口、配置校验结果和服务状态。

### 16.3 确认 UFW

```bash
ufw status numbered
```

至少应该看到允许 SSH 和 `10086/tcp` 的规则。

### 16.4 确认云安全组

重点检查：

- 规则是入站规则，而不是只配置了出站；
- 协议是 TCP；
- 端口是 `10086`；
- 规则绑定到了正确的实例或网卡；
- 使用的是当前实例的公网 IP；
- 来源 IP 没有误填成内网地址。

### 16.5 端口通但 VMess 握手失败

依次核对：

1. 服务端 UUID 与客户端 UUID 完全一致。
2. 客户端端口是 `10086`。
3. 客户端协议是 VMess，不是 VLESS/Trojan。
4. 客户端网络是 TCP。
5. 客户端 TLS 是关闭状态。
6. 客户端 Header/伪装类型是 `none`。
7. 客户端 AlterId 是 `0`。
8. 服务器和客户端设备时间正确。

### 16.6 检查时间

```bash
timedatectl status
systemctl is-active chrony
chronyc tracking
```

如果时间明显错误：

```bash
systemctl restart chrony
timedatectl set-ntp true
```

### 16.7 检查 Xray 服务使用的配置文件

```bash
systemctl cat xray
```

查看 `ExecStart` 指向的配置路径，确认它和你编辑的 `/usr/local/etc/xray/config.json` 一致。

## 17. 维护和停用命令

查看服务：

```bash
systemctl status xray --no-pager
```

重启服务：

```bash
systemctl restart xray
```

停止服务：

```bash
systemctl stop xray
```

禁止开机启动：

```bash
systemctl disable xray
```

升级 Xray：

```bash
bash /tmp/install-release.sh @ install
```

如果临时停用，除了停止服务，也应在云安全组和 UFW 中删除 `10086/tcp` 入站规则。

## 18. 安全加固建议

无 TLS 方案至少应做到：

- 使用随机 UUID，不使用示例 UUID；
- SSH 仅允许固定管理 IP；
- 代理端口尽量限制为客户端 IP；
- Ubuntu 定期执行 `apt update && apt upgrade`；
- 不开放不需要的 UDP 端口；
- 不启用 Xray API、统计接口和管理面板；
- 定期查看 `journalctl -u xray`；
- 不公开 VMess 分享链接；
- 长期使用前迁移到带 TLS/REALITY 的传输安全方案。

“换一个冷门端口”只能减少非常简单的扫描，不能替代身份认证和安全组限制。

## 19. 资料来源与版本说明

本文优先采用官方资料，社区教程只用于补充操作背景。旧教程中的字段可能已经过时，因此配置以当前 Xray 文档和实际 `xray run -test` 结果为准。

### Project X / Xray 官方

- VMess 入站配置：<https://xtls.github.io/config/inbounds/vmess.html>
- 入站对象：<https://xtls.github.io/config/inbound.html>
- 出站对象：<https://xtls.github.io/config/outbound.html>
- 传输配置：<https://xtls.github.io/config/transport.html>
- RAW 传输：<https://xtls.github.io/config/transports/raw.html>
- 官方安装脚本：<https://github.com/XTLS/Xray-install>
- 配置测试参数：<https://github.com/XTLS/Xray-core/blob/main/main/run.go>

### 客户端官方资料

- v2rayN 仓库：<https://github.com/2dust/v2rayN>
- v2rayN Wiki：<https://github.com/2dust/v2rayN/wiki>
- v2rayNG 仓库：<https://github.com/2dust/v2rayNG>
- v2rayNG Wiki：<https://github.com/2dust/v2rayNG/wiki>

### Ubuntu 和云厂商资料

- Ubuntu Server 防火墙文档：<https://documentation.ubuntu.com/server/how-to/security/firewalls/>
- DigitalOcean UFW 教程：<https://www.digitalocean.com/community/tutorials/how-to-set-up-a-firewall-with-ufw-on-ubuntu>
- AWS Security Groups：<https://docs.aws.amazon.com/vpc/latest/userguide/security-group-rules.html>
- Google Cloud VPC Firewall：<https://cloud.google.com/firewall/docs/firewalls>
- Azure NSG：<https://learn.microsoft.com/en-us/azure/virtual-network/network-security-groups-overview>
- V2Fly 新手指南：<https://www.v2fly.org/guide/start.html>
- V2Fly VMess 文档：<https://www.v2fly.org/config/protocols/vmess.html>
