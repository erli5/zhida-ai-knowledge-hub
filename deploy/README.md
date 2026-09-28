# 云运维模块（部署与 CI/CD）

本模块对应目标岗位方向"云运维"，展示把一个多服务应用**容器化、编排、反向代理、监控、自动化发布**的完整闭环。

## 目录内容

```
deploy/
├── docker-compose.yml     # 一键编排 backend + ai_app + nginx
├── nginx/nginx.conf        # Nginx 反向代理（统一入口）
├── monitoring/prometheus.yml  # Prometheus 抓取后端 /metrics
└── README.md
.ai_app/Dockerfile         # 前端镜像
backend/Dockerfile         # 后端镜像
.github/workflows/ci.yml   # CI：语法检查 + 接口测试
.github/workflows/cd.yml   # CD：构建并推送镜像到 GHCR
```

> 说明：GitHub Actions 工作流必须放在仓库根的 `.github/workflows/` 才能被 GitHub 识别，
> 因此 CI/CD 文件位于项目根的 `.github/workflows/`，这里在 README 中集中说明。

## 1. 本地一键编排

```bash
cd deploy
docker compose up --build
# 访问 http://localhost  （Nginx 入口，内含前端与后端 API）
```

Nginx 把 `/api`、`/health`、`/metrics` 转发到后端，其余转发到 Streamlit 前端。

## 2. 监控

后端暴露 Prometheus 格式的指标端点 `/metrics`（请求总数、服务存活）。
启动 Prometheus 抓取：

```bash
docker run -p 9090:9090 -v ${PWD}/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
# 打开 http://localhost:9090 查询 zhida_requests_total
```

## 3. CI / CD

- **CI**（`ci.yml`）：每次 push/PR 自动安装依赖、做语法检查、跑 pytest，保证主分支可发布。
- **CD**（`cd.yml`）：push 到 main 时自动构建后端与前端镜像并推送到 GitHub Container Registry（GHCR）。
  首次使用需在仓库 Secrets 中确认 `GITHUB_TOKEN` 权限（默认已具备 `packages: write`）。

## 面试可讲的点

- 容器化解决了"环境不一致"；多阶段构建可减小镜像体积。
- Nginx 为什么在最前面：统一入口、HTTPS、负载均衡、静态资源。
- CI 保证"合进来的代码是绿的"，CD 让发布可重复、可回滚。
- 可观测性：指标（Prometheus）+ 日志 + 告警，第一时间发现线上问题。
