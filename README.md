# 公交出行比例两轮重抽分析（FastAPI + DuckDB）

对同一批抽样片区的两轮住户调查，按区计算"乘公交比例"的两轮差值，并用重抽权重
（replicate weights）按编号配对生成差值分布，报告其线性插值 2.5% / 97.5% 分位数。

## 1. 数据口径

每轮上传一对文件：

| 文件 | 格式 | 列 |
|---|---|---|
| 样本 | CSV（含表头） | `sample_id, district, bus, main_weight` |
| 重抽权重 | Parquet | `sample_id, rep_id, weight` |

* `bus` 只能是 0/1；两类权重必须为有限非负数；`sample_id`、`district`、`rep_id` 非空。
* 重抽权重只能引用**本轮**样本 CSV 中的 `sample_id`（外键校验）。
* 每个样本在每次重抽中**恰有一条**权重（缺漏、重复均拒绝）；每次重抽的权重总和
  必须为正（全零重抽无法形成比例，拒绝）。
* 两轮的同编号 `rep_id` 表示同一次重抽；两轮编号集合不同时无法配对。

计算（选区 D，轮次 t）：

* 主权重比例 `p_t = Σ(main_weight × bus) / Σ(main_weight)`（仅 D 内样本）
* 比例差点估计 `p_2 − p_1`
* 第 k 次重抽比例 `p_t,k = Σ(weight_k × bus) / Σ(weight_k)`，配对差值
  `d_k = p_2,k − p_1,k`
* 区间端点：`d_k` 的 type-7 线性插值分位数（与 DuckDB `quantile_cont`、numpy 默认
  分位数同口径），取 0.025 与 0.975。

区间留空的三种情况（均返回中文原因）：

1. 任一轮主权重分母为零 → 点估计与区间均留空；
2. 两轮重抽编号集合不同 → 点估计保留，区间留空并列出各自独有的编号；
3. 某次重抽在该区的权重分母为零 → 点估计保留，区间留空并列出问题编号。

## 2. 运行

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

Compose 约束：`mem_limit: 384m`、`cpus: 2`；数据放在命名卷 `duckdb_data`
（容器内 `/data`），DuckDB 数据文件、WAL 与临时落盘目录都在该卷上。
DuckDB 自身 `memory_limit=180MB`、`threads=2`，超限走磁盘临时区，不靠堆内存硬扛。

本地无 Docker 时（开发）：

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
DATA_DIR=./data uvicorn app.main:app --port 8000
```

## 3. HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 单页界面（上传、状态、选区查询） |
| POST | `/api/rounds/{1\|2}` | multipart：`sample_csv`、`replicate_parquet`；成功返回代际号与行数，失败 422 |
| GET | `/api/status` | 两轮当前代际、样本/重抽/权重行数、区列表 |
| GET | `/api/districts` | 区列表 |
| GET | `/api/result?district=...` | 比例差与区间（留空时附原因） |

## 4. 实现要点

* **代际（generation）模型**：每次成功上传在同一 DuckDB 文件内新建
  `s_gen_<g>` / `r_gen_<g>` 两张表并在事务中改名发布（`meta_current` 指向当前代际）。
  校验全部在暂存表 + 单事务内完成，任何一步失败即回滚——**换表失败时旧代际原样可查**。
* **查询快照**：查询开始时通过 `meta_refcount` 钉住两轮当前代际；换表只改指针，
  被钉住的旧代际延迟到查询结束、引用计数归零后才删除。因此进行中的查询始终使用
  起始时刻的两套输入，不受并发换表影响。
* **单进程单 uvicorn worker**：上传由写锁串行化；查询走独立短连接。DuckDB 同一
  进程不允许只读/读写连接混用，读连接使用同配置、由应用层保证不写。
* **流式上传**：multipart 以 1 MiB 分块直接落临时目录，避免整文件驻留内存。
* 所有用户输入只作为**数据值**参数化或写入临时文件路径，表名/列名均为服务端
  固定生成，不存在 SQL 标识符注入面。

## 5. 测试与实测

```bash
pytest -q                      # 21 个单元/接口测试
python scripts/make_demo.py    # 生成手工可算的小样本（含错配套件）
python scripts/scale_test.py --rows 5000000
```

单元测试覆盖：手算基准的比例与分位数、缺轮、编号错配、主权重零分母、重抽零分母、
外键引用、缺漏/重复权重、非法 bus/权重、重复样本号、列结构不符、换表失败保旧、
换表成功发新代际、快照钉住与延迟删除，以及 Python 分位数与 `quantile_cont` 的
多规模交叉一致性。

### 500 万行实测（每轮 10,000 样本 × 500 次重抽 = 5,000,000 权重行，两轮共 1,000 万行）

多轮实测区间（`scripts/scale_test.py --rows 5000000`）：

| 指标 | 实测 |
|---|---|
| 单轮上传（含全部校验、建表发布） | 约 2.6–4.9 秒 |
| 分区查询（含配对、分位数） | 0.1–0.4 秒 |
| 服务进程峰值 RSS | 161–237 MiB（上限 384 MiB） |
| DuckDB 数据文件（两轮合计） | 约 77 MiB |
| 分位数与独立 DuckDB 重算 | 逐位一致（1e-9 容差内） |
| 换表失败后旧数据 | 代际、行数、结果均不变 |
| 换表成功 | 新代际发布、旧代际释放 |

> 内存说明：本沙箱 cgroup 文件系统只读，无法用 `memory.max` 实测容器级 OOM 行为，
> 脚本以进程峰值 RSS 佐证；`scale_test.py` 保留 cgroup v2 分支，在有权限的环境
> （或直接 `docker compose up`）会按 384 MiB 限制执行并读取 `memory.peak`。
> 另经验证，同进程内多个连接并发重查询时 DuckDB 的 `memory_limit` 由共享缓冲管理器
> 统一约束（两连接并发的 RSS 增量约 31 MiB，而非两个 120 MB 上限相加），并发查询
> 不会使内存预算翻倍。

## 6. 遗留事项 / 已知边界

* 环境内无 Docker 守护进程，Compose 文件仅做了 YAML 与配置校验，未实际构建镜像；
  384 MiB 约束以同参数的进程实测佐证，建议在有 Docker 的环境再跑一次
  `docker compose up` + `scale_test.py`。
* 单 worker 设计：上传串行，查询并发受 DuckDB 单实例能力限制；面向多用户高并发
  需另做队列或多实例分片。
* 零/负权重按"非负"校验，未做权重设计层面的口径审计（如权数和与样本量关系）。
* 上传文件大小没有应用层显式上限（依赖反代/容器磁盘），超大文件会先完整落盘再校验。
* 分位数基于重抽差值的经验分布，未做正态近似或有限总体校正。
