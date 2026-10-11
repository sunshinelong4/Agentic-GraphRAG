# 共享知识库：鉴权与并发升级

这一版将项目升级为需要登录的共享知识库。管理员负责导入、抽取、删除、重建及创建账号；普通用户可以查询、检索和问答。默认启用鉴权，没有默认账号或默认密码。

## 本地启动

在 `backend/` 使用现有虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m app.security admin
.\.venv\Scripts\python.exe -m app.security reader --role user
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 100
```

创建账号会交互输入并确认密码，至少 12 个字符。用户名只允许字母、数字、点、下划线和短横线，长度 3–64。重复用户名不会覆盖原账号。

前端在 `frontend/` 执行 `npm run dev`，打开页面后登录。令牌只保存在浏览器内存，刷新后需要重新登录。管理员也可以通过鉴权后的 `POST /api/auth/users` 创建账号，`GET /api/auth/users` 列出账号。

Swagger 调试时，先请求 `POST /api/auth/login`，将返回的 `access_token` 粘贴到 Authorize 按钮中；命令行请求则添加 `Authorization: Bearer <access_token>`。鉴权由 ASGI 中间件统一执行。不要把真实令牌写进报告或提交到仓库。

Docker Compose 也保留单 worker，新增前端构建和 nginx 转发配置。使用 `docker compose --env-file backend/.env up --build -d` 启动，再执行 `docker compose exec backend python -m app.security admin` 创建账号。打开 `http://localhost:3000` 登录，`http://localhost:3000/docs` 打开 API 文档。数据库、图谱、向量和缓存通过命名卷保存，镜像构建排除了 `.env` 和本地数据。nginx 关闭 SSE 缓冲。

Compose 的后端 8000 端口仅在私有容器网络开放，由 nginx 转发并覆盖客户端转发头，因此后端信任该私有网络入口的代理头。若之后直接对外发布后端端口，必须同时将 `FORWARDED_ALLOW_IPS` 收紧为可信代理地址。该配置用于本地演示，公网部署需配置 HTTPS。当前未在 Docker 引擎内执行镜像构建。

## 权限和会话

| 路径/操作 | 匿名 | 普通用户 | 管理员 |
|---|---|---|---|
| `/api/health`、登录 | 可用 | 可用 | 可用 |
| 查询场景、图谱、知识库统计、问答、检索 | 401 | 可用 | 可用 |
| 导入/删除文档、信息抽取、解析 PDF、构建/重置图谱、清缓存 | 401 | 403 | 可用 |
| 创建/列出账号、`GET /api/system/metrics` | 401 | 403 | 可用 |
| `/api/auth/me`、退出登录 | 401 | 可用 | 可用 |

业务查询的 POST 接口采用明确白名单，其余修改操作默认要求管理员。根介绍页和 API 文档公开，业务数据接口受保护。密码使用独立随机盐的 scrypt 哈希；令牌为 256 bit 随机值，数据库只保存 SHA-256 摘要。默认有效期 8 小时，退出后在服务端立即撤销。每次请求检查账号状态和过期时间。

账号/会话存储使用 SQLite WAL，每次调用创建并关闭独立连接。部署通过 HTTPS 反向代理访问；这版不提供第三方登录、自助注册、找回密码或密码重置页面。`AUTH_ENABLED=false` 仅用于明确需要匿名访问的本地开发，设置后所有 API 都按开发管理员处理。

## 并发控制的实际行为

| 配置 | 默认值 | 作用 |
|---|---|---|
| `MAX_INFLIGHT_REQUESTS` | 64 | API 在途请求总上限，满时返回 503 |
| `MAX_EXPENSIVE_REQUESTS` | 8 | 问答、检索、抽取、PDF、导入等重请求的并发上限 |
| `MAX_USER_CONCURRENCY` | 2 | 单个用户的重请求并发上限，满时返回 429 |
| `USER_REQUESTS_PER_MINUTE` | 30 | 单个用户重请求的固定分钟窗口配额 |
| `LOGIN_REQUESTS_PER_MINUTE` | 10 | 每个来源 IP 的登录尝试配额 |
| `MAX_AUTH_CONCURRENCY` | 2 | 登录哈希运算并发上限，避免占满线程池 |
| `AUTH_SESSION_TTL_SECONDS` | 28800 | 会话寿命 |
| `LLM_TIMEOUT_SECONDS` | 45 | Agent/旧问答每次 SDK 调用的超时，SDK 最多重试一次 |

这里采用有界准入与立即拒绝，不让请求无限排队。429/503 带 `Retry-After`，前端显示错误，不将这些错误转换成 mock 回答。分钟窗口在边界可能允许两倍的短时突发，由并发上限进一步约束。登录 IP 以 ASGI 收到的客户端地址为准；通过反向代理部署时，仅让 Uvicorn 信任代理的地址，不能信任客户端自行填写的转发头。

所有执行同步 SDK、磁盘和图计算的业务路由使用 FastAPI 同步路径，让框架在线程池执行。旧 SSE 生成器也改为同步生成器。长连接直到整个 ASGI 响应结束或断开才释放名额，而不是只在发出响应头时释放。容量限制针对请求，不等同于精确的外部 LLM 调用配额：一个请求仍可能执行多轮模型调用。SDK 超时也不等同于整个 Agent 的总 deadline。

每个 Agent 请求创建独立实例、工具闭包和图谱快照；hops/top-k 等参数不会串用，客户端不能把重试/工具轮数提高到服务器上限之外。图谱 JSON 写入采用锁内快照与临时文件原子替换。抽取缓存共享单实例，读写加锁、原子落盘，返回深拷贝以避免调用方修改共享结果。

共享知识库管理操作一次只放行一个，避免建图、重置和抽取统计等共享可变状态互相覆盖，其余管理请求返回 503。只读图谱接口读取快照，避免遍历时被写入打断。

管理员可查看 `/api/system/metrics` 的在途计数、重请求数量、管理操作及拒绝原因。日志记录请求 ID、路径、状态与总耗时，响应返回 `X-Request-ID`。计数器是进程内累计，重启后清零。

## 验证与压测

```powershell
# 无模型成本的权限、并发、SSE 断开与 Agent 参数隔离测试
.\.venv\Scripts\python.exe -m pytest tests/ -q

# 本地 HTTP 压测：默认只调用路由预览，不调用模型
.\.venv\Scripts\python.exe -m evaluation.load_test --username reader --requests 100 --concurrency 20 --output evaluation/results/load_metadata.json

# 真实模型压测：明确启用，会产生模型调用费用；先用小批量
.\.venv\Scripts\python.exe -m evaluation.load_test --username reader --mode qa --requests 10 --concurrency 5 --output evaluation/results/load_qa.json
```

密码交互输入，也可通过 `LOAD_TEST_PASSWORD` 注入。报告包含状态码、业务失败数、全体/成功请求 P50 与 P95、成功吞吐和压测期间健康检查。拒绝请求很快，不能把包含大量 429/503 的总 RPS 宣传成问答吞吐。脚本使用一个账号，因此主要验证单用户保护；模拟多用户总容量需使用多个普通账号独立运行，并记录模型限额及机器配置。

自动化并发用例通过事件屏障制造 20 个真正重叠的请求，在容量为 2 时断言 2 个成功、18 个 503；同时确认健康检查可用、执行后名额归零。另一个用例模拟 SSE 断开，断言名额归还。这些验证机制正确性，不是线上吞吐测量。

## 部署范围与下一阶段

当前只支持单 API 进程、单实例。不要直接加 `--workers 4` 或扩成多个副本：内存图谱、Chroma 本地目录和限流计数都不是分布式共享状态。默认的 8 个重请求名额是起点，需根据上游模型的 RPM/TPM 和本机资源压测调整。

真正需要横向扩容时，下一阶段依次迁移：

1. 账号、会话、文档元数据到 PostgreSQL；图谱使用共享持久化存储或版本化只读快照。
2. Redis 原子脚本实现跨进程配额和并发租约，租约带过期与心跳，异常时不能静默放行。
3. PDF 解析、抽取和建图迁到持久任务队列：任务 ID、幂等、重试、失败状态和独立 worker。
4. 在共享状态具备一致性后增加 API 副本，并加入上游熔断、端到端 deadline、指标监控与多用户持续压测。

面试时可准确表述为“完成共享知识库 RBAC、可撤销会话、同步阻塞修复、请求级 Agent 隔离和有界并发保护，并用并发测试验证过载和流式断开行为”。真实 QPS/P95 应在真实模型压测后填入，不能从单元测试推算。
