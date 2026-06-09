# PIMU-Sim

PIMU-Sim 是一个面向 NPU ISA 的仿真器项目。当前分支为 `new_spm`，主线是**性能仿真**：从 128-bit 十六进制指令文件出发，按周期推进前端流水、指令队列、六路 DMA、功能单元、片上 SPM/ACC SRAM、DDR 模型，并输出周期数、模块利用率、带宽和可选功耗统计。

代码中保留了 `FUNCTIONAL` / `PERFORMANCE` 两种模式选择，但当前 `FuncSim::run()` 仍是占位实现，实际可用和持续维护的是 `PERFORMANCE` 路径。

---

## 当前能力

### 指令类型

性能模型当前由 `Decoder` 识别以下 opcode：

| Opcode | 指令 | 指令行数 | 目标队列 |
| --- | --- | ---: | --- |
| `0x01` | `MVIN` | 2 | `DMAInstQueue` |
| `0x02` | `MVOUT` | 2 | `DMAInstQueue` |
| `0x03` | `SA` | 3 | `SAInstQueue` |
| `0x04` | `MATADD` | 3 | `SAInstQueue` |
| `0x05` | `SFU` | 2 | `SFUInstQueue` |

已建模的主要语义包括：

- `MVIN` / `MVOUT`：DRAM 与 SPM/ACC 之间的数据搬运；当前解码路径支持 SPM `MVIN/MVOUT` 和 ACC `MVIN`。
- `SA`：脉动阵列路径，支持 GEMM / CONV 建模，并可配合 ACC 做累加。
- `MATADD`：从 ACC Data SRAM 读取两个矩阵，输出到 SPM。
- `SFU`：支持 softmax、gelu、layernorm、pool_max、transpose。

### 周期级前端

性能仿真前端按如下路径推进：

```text
Instruction file -> InstCache -> Decoder(fetch/decode/dispatch)
                 -> DMA/SA/SFU InstQueue -> Functional Units
```

每条有效指令行会从文件读取到 `InstCache`，再由 `Decoder` 识别 opcode、聚合多行指令并分发到对应队列。`PerfNpuCore::tick()` 每个核心周期执行中断处理、清空 SPM/ACC 端口状态、推进功能单元、连接 DMA/DDR 消息、采样利用率/功耗、发射队列、推进解码器、加载指令和 retire。

### new_spm 存储与 DMA 策略

当前分支的核心变化是固定 bank 角色和固定 DMA 路由：

| 区域 | Bank | 用途 | DMA 路由 |
| --- | --- | --- | --- |
| SPM A | `bank0/1` | SA 输入 A / activation | `MVIN` 走 `DMA1` |
| SPM B | `bank2/3` | SA 输入 B / weight | `MVIN` 走 `DMA2` |
| SPM Output | `bank4+` | SFU、SA/MATADD 输出、中间结果 | `MVIN` 走 `DMA4`，`MVOUT` 走 `DMA5/6` |
| ACC Data | `ACC bank0/1` | psum、MATADD 输入 | ACC `MVIN` 走 `DMA3` |
| ACC Bias | bias 区 | SA bias/psum bias 路径 | 由 ACC SRAM bias 端口建模 |

`DMA5` / `DMA6` 对 SPM output bank 的 `MVOUT` 做 round-robin 分发。`EventSyncUnit` 同时检查 DMA 路由合法性、bank valid/busy 状态、SA 输入 bank 角色、输出 bank 是否空闲，以及 `MATADD` 两个 ACC 输入是否落在不同 bank。

### 功能单元

当前性能核心会创建并采样以下模块：

- `DMA1` / `DMA2` / `DMA3` / `DMA4` / `DMA5` / `DMA6`
- `SystolicArray`
- `Accumulator`
- `SFU`
- `GEMV`
- `SPM`
- `ACC Bias SRAM`
- `ACC Data SRAM`

`GEMV` 单元目前已实例化并纳入利用率类型定义，但当前 opcode 分发主路径仍由 DMA、SA、MATADD 和 SFU 覆盖。

### SPM / ACC 建模

SPM 当前按 bank 建模：

- bank 数量和容量由 `sram_bank_num`、`sram_bank_bytes` 决定。
- 每次 SPM 访问宽度为 `systolic_array_size * sram_data_width / 8` 字节。
- 访问必须完全落在单个 bank 内，跨 bank 会触发错误。
- 每个 bank 每周期只能接受一次访问，`PerfNpuCore` 每周期开始清空 bank 端口状态。

ACC SRAM 当前分为：

- `ACC Bias SRAM`：独立 bias 容量，由 `bias_bytes` 管理。
- `ACC Data SRAM`：data 容量由 `acc_bytes` 管理。
- bias/data 各有两个端口状态；读写路径会检查边界和端口冲突。

### DDR 模型

支持两种 DDR 模式：

- `AnalyticalDDR`：默认使用的解析模型，按 AXI beat、burst、4KB 边界、基础时序参数和请求队列估算访问延迟。
- `DetailDDR`：接入 `DRAMSim3`，用于更细粒度 DDR 时序实验；当前事务拆分仍较简化，常规 benchmark 默认使用解析模式。

仿真器会根据 `npu_freq` 与 `ddr_freq` 计算统一 base tick，使核心和 DDR 可按各自频率推进。

### 统计与分析

仿真结束后会在标准输出打印，并在 `enable_log = 1` 时写入日志文件：

- 总周期数、最大周期限制、仿真耗时
- 已读取指令行数、已模拟完整指令数、模拟 IPS
- `DMA1` 至 `DMA6`、`SPM`、`SystolicArray`、`Accumulator`、`ACC Bias/Data SRAM`、`SFU`、`GEMV` 的 busy cycle 与利用率
- DDR、SPM、ACC Bias、ACC Data 的读写字节数、平均带宽、峰值 B/cycle
- 可选功耗报告：六路 DMA、ACC、SFU、SA 的能量、平均功耗和模式周期占比

---

## 目录结构

```text
PIMU-Sim/
├── include/
│   ├── config.hpp
│   ├── func_sim.hpp
│   ├── logger.hpp
│   ├── regfile.hpp
│   ├── sim_base.hpp
│   ├── util.hpp
│   └── perf/
│       ├── acc_sram.hpp
│       ├── analytical_ddr.hpp
│       ├── base_module.hpp
│       ├── decoder.hpp
│       ├── detail_ddr.hpp
│       ├── event_sync_unit.hpp
│       ├── hw_utilization.hpp
│       ├── inst_cache.hpp
│       ├── inst_queue.hpp
│       ├── perf_func_unit.hpp
│       ├── perf_npu_core.hpp
│       ├── perf_sim.hpp
│       ├── power_analyzer.hpp
│       └── scratchpad.hpp
├── src/
│   ├── main.cpp
│   ├── config.cpp
│   ├── func_sim.cpp
│   ├── logger.cpp
│   ├── regfile.cpp
│   └── perf/
│       ├── acc_sram.cpp
│       ├── analytical_ddr.cpp
│       ├── decoder.cpp
│       ├── detail_ddr.cpp
│       ├── event_sync_unit.cpp
│       ├── hw_utilization.cpp
│       ├── inst_cache.cpp
│       ├── inst_queue.cpp
│       ├── perf_func_unit.cpp
│       ├── perf_npu_core.cpp
│       ├── perf_sim.cpp
│       ├── power_analyzer.cpp
│       ├── scratchpad.cpp
│       └── func_units/
│           ├── accumulator.cpp
│           ├── dma.cpp
│           ├── gemv.cpp
│           ├── sfu.cpp
│           └── systolic_array.cpp
├── config/
│   ├── power.txt
│   ├── resnet_block_cfg.txt
│   ├── smolvlm2_decode_cfg.txt
│   └── smolvlm2_prefill_cfg.txt
├── benchmarks/tb_test/
│   ├── gen_resnet_block.py
│   ├── gen_smolvlm2_cases.py
│   ├── tb_resnet_block.txt
│   ├── tb_smolvlm2_decode.txt
│   └── tb_smolvlm2_prefill.txt
├── log/
├── external/DRAMsim3/
├── plot_timeline.py
├── Makefile
├── Env.cfg
├── Simulator.md
└── README.md
```

---

## 环境依赖

推荐在 Linux 或 WSL/MSYS2 这类 POSIX shell 环境下使用。当前 `Makefile` 和 `Env.cfg` 使用了 `find`、`mkdir -p`、`rm -rf`、`source`、`LD_LIBRARY_PATH` 等 Linux/POSIX 习惯；原生 Windows PowerShell 下通常需要额外适配。

基础依赖：

- `g++`，支持 C++17
- `make`
- `bash`
- `pthread`
- 仓库内置 `external/DRAMsim3/libdramsim3.so`

可视化脚本依赖：

- Python 3
- `plotly`
- 可选：`kaleido`，用于 `plot_timeline.py` 导出 PNG

---

## 构建

进入仓库后建议先加载环境变量：

```bash
source Env.cfg
```

编译：

```bash
make build
```

生成的可执行文件：

```bash
./build/npu_sim
```

手动运行时必须传入一个配置文件：

```bash
./build/npu_sim ./config/smolvlm2_decode_cfg.txt
```

清理构建目录：

```bash
make clean
```

---

## 运行

`Makefile` 当前默认目标为：

```makefile
TOP = smolvlm2_decode
```

因此默认运行：

```bash
make run
```

等价于：

```bash
./build/npu_sim ./config/smolvlm2_decode_cfg.txt
```

也可以通过命令行覆盖 `TOP`：

```bash
make run TOP=resnet_block
make run TOP=smolvlm2_prefill
make run TOP=smolvlm2_decode
```

当前仓库提供的主要 benchmark：

| TOP | 配置文件 | 指令文件 | 默认日志 |
| --- | --- | --- | --- |
| `smolvlm2_decode` | `config/smolvlm2_decode_cfg.txt` | `benchmarks/tb_test/tb_smolvlm2_decode.txt` | `log/tb_smolvlm2_decode_log.txt` |
| `smolvlm2_prefill` | `config/smolvlm2_prefill_cfg.txt` | `benchmarks/tb_test/tb_smolvlm2_prefill.txt` | `log/tb_smolvlm2_prefill_log.txt` |
| `resnet_block` | `config/resnet_block_cfg.txt` | `benchmarks/tb_test/tb_resnet_block.txt` | `log/tb_resnet_block_log.txt` |

注意：`smolvlm2_decode_cfg.txt` 和 `smolvlm2_prefill_cfg.txt` 当前默认 `enable_log = 0`，只输出控制台统计；如需生成 timeline 输入日志，需要改成 `enable_log = 1`。`resnet_block_cfg.txt` 当前默认开启日志。

---

## 配置文件

配置文件采用简单的 `key = value` 格式，支持 `#` 注释。当前 `Config::from_file()` 实际解析的主要字段如下。

仿真与文件：

- `simulator_type`：`PERFORMANCE` / `FUNCTIONAL`，当前建议使用 `PERFORMANCE`
- `max_sim_cycle`
- `inst_file`
- `log_file`
- `enable_log`

核心与接口：

- `systolic_array_size`
- `npu_freq`
- `axi_data_width`
- `*_reg_addr`：如 `iar_reg_addr`、`mer_reg_addr`、`ier_reg_addr`、`isr_reg_addr`、`ipr_reg_addr`

DDR：

- `ddr_mode`：`1` 为 `AnalyticalDDR`，`0` 为 `DetailDDR`
- `dramsim_cfg_path`
- `ddr_freq`
- `ddr_bandwidth`
- `tCAS`
- `tRCD`
- `tRP`
- `tBURST`
- `BL`
- `ddr_bus_width`
- `ddr_channel_num`
- `ddr_request_queue_depth`

SPM：

- `sram_bank_num`
- `sram_bank_bytes`
- `sram_data_width`

ACC：

- `bias_bank_num`
- `bias_bytes`
- `acc_bank_num`
- `acc_bytes`
- `acc_data_width`

指令系统：

- `inst_cache_size`
- `inst_queue_depth`

功耗：

- `enable_power`
- `power_file`

当前配置文件中还保留了 `core_mode`、`sram_ports`、`acc_ports` 等字段，但当前解析器没有消费这些字段；修改它们不会影响现有仿真行为。

---

## Benchmark 生成

仓库内置两个生成脚本：

```bash
python3 benchmarks/tb_test/gen_resnet_block.py
python3 benchmarks/tb_test/gen_smolvlm2_cases.py
```

- `gen_resnet_block.py` 生成固定 bank / 固定 DMA 策略下的 ResNet-like block 指令流：`D = A * B + A`。SPM bank0/1 放 A，bank2/3 放 B，bank4/5 放临时 C，bank6/7 放最终 D；ACC bank0/1 用于 `MATADD` 输入。
- `gen_smolvlm2_cases.py` 生成 SmolVLM2-500M-Video-Instruct 单 block 的 prefill/decode 指令流，采用 new_spm 固定映射：bank0/1 为 A，bank2/3 为 B，bank4/5 为 SFU 区域，bank6/7 为 SA/MATADD 输出；同时会写出对应配置文件。

运行生成脚本会覆盖同名 `.txt` 或配置文件。`gen_smolvlm2_cases.py` 生成的配置模板中 `max_sim_cycle = 30000000`、`inst_cache_size = 1024`、`inst_queue_depth = 2`；如只想保留当前手工配置，运行脚本前需要注意这些字段会被重写。

---

## 可视化

`plot_timeline.py` 会从日志中解析 DMA、SA、ACC、SFU、DDR 的执行区间、等待区间、预测结束点和依赖关系，输出交互式 HTML。

```bash
python3 plot_timeline.py
```

脚本顶部的 `CONFIG` 当前默认读取：

```text
log/tb_resnet_block_log.txt
```

默认 lane 包括：

```text
DMA1, DMA2, DMA3, DMA4, DMA5, DMA6, SA, ACC, SFU, DDR
```

如需分析其他 workload，先开启对应配置文件的 `enable_log = 1` 并运行仿真，再修改脚本顶部的 `CONFIG["log"]` 和输出路径。

---

## 当前项目状态

已经完成或当前可用：

- 性能仿真主路径
- 指令文件驱动的前端流水
- 多行指令聚合与 DMA/SA/SFU 队列分发
- 六路 DMA 固定路由策略
- new_spm bank 访问冲突与跨 bank 边界检查
- SPM、ACC Bias/Data SRAM 建模
- `EventSyncUnit` 资源依赖同步和固定 bank 角色检查
- `AnalyticalDDR` 与 `DetailDDR` 双模式
- 模块利用率、带宽和可选功耗统计
- ResNet-like block、SmolVLM2 prefill/decode workload
- 六路 DMA timeline 可视化脚本

仍需继续完善：

- `FUNCTIONAL` 模式目前为空实现
- `DetailDDR` 的事务拆分和时序细节仍简化
- 独立 `GEMV` 单元尚未接入当前 opcode 分发主路径
- 配置中的部分保留字段尚未解析
- 缺少自动化回归测试入口

---

## 开发入口

建议优先阅读以下文件：

- `Simulator.md`：开发记录与设计说明
- `src/main.cpp`：程序入口与模式选择
- `src/config.cpp`：配置解析
- `src/perf/perf_sim.cpp`：性能仿真顶层循环
- `src/perf/perf_npu_core.cpp`：核心周期推进、六路 DMA 和模块连接
- `src/perf/decoder.cpp`：opcode 分类、指令行数和固定 DMA 分发策略
- `src/perf/event_sync_unit.cpp`：bank valid/busy、固定 bank 角色和 DMA 路由合法性检查
- `src/perf/scratchpad.cpp`：new_spm bank 访问模型
- `src/perf/acc_sram.cpp`：ACC Bias/Data SRAM 端口模型
- `src/perf/inst_queue.cpp`：指令队列、发射、retire 和 ISR 更新
- `src/perf/func_units/`：DMA、SA、ACC、SFU、GEMV 功能单元
- `src/perf/hw_utilization.cpp`：利用率与带宽统计
- `src/perf/power_analyzer.cpp`：六路 DMA 功耗采样和报告

---

## License

本项目使用 Apache License 2.0，详见 `LICENSE`。
