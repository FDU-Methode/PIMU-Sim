# PIMU-Sim

PIMU-Sim 是一个面向 NPU ISA 的仿真器项目，支持两类运行模式：

- 功能仿真：用于验证指令语义与执行流程
- 性能仿真：用于周期级建模、模块并行、带宽与利用率统计

当前项目为性能仿真路径，已经具备较完整的前端流水、功能单元、存储系统和统计模块，可用于单算子测试、卷积/矩阵计算测试以及残差块等组合指令流验证。

---

## 项目特性

### 1. 支持的主要指令类型

当前性能模型支持以下指令类型：

- `MVIN`
- `MVOUT`
- `SA`
- `MATADD`
- `SFU`

其中：

- `SA` 支持 GEMM / CONV 路径
- `SFU` 已覆盖 GELU、SOFTMAX、LAYERNORM、POOLMAX、TRANSPOSE
- `MATADD` 支持从 ACC SRAM 读取两个输入矩阵并输出到 SPM
- `DMA` 支持 DRAM / SPM / ACC 之间的数据搬运

### 2. 周期级前端流水

性能模型采用如下前端组织：

- `InstFile -> InstCache -> Decoder -> InstQueue -> Functional Units`

其中：

- `InstCache`：每周期从指令文件按行读取指令
- `Decoder`：执行 `fetch / decode / dispatch`
- `InstQueue`：按完整指令组包并排队
- `EventSyncUnit`：负责基于 valid / busy 的同步控制
- `PerfNpuCore`：统一推进整个核心各阶段

### 3. 功能单元建模

当前主要功能单元包括：

- `DMA1 / DMA2 / DMA3`
- `SystolicArray`
- `Accumulator`
- `SFU`
- `GEMV`

当前代码中，**DMA 调度策略仍在持续迭代**；项目已经支持多 DMA 结构，但为了避免复杂乱序死锁，当前DMA数量设置为1。

### 4. 片上存储建模

项目已实现两类片上存储：

- `SPM`
  - 多 bank 建模
  - bank 数量与每个 bank 容量由配置文件决定
  - 当前模型按 bank 做 valid / busy 管理
- `ACC SRAM`
  - 分为 `bias` 区和 `data` 区
  - `bias` 和 `data` 分开建模
  - 支持分别统计利用率与带宽

### 5. DDR 建模

支持两种 DDR 模式：

- `AnalyticalDDR`
  - 用解析模型估算访存延迟与带宽
  - 适合快速测试
- `DetailDDR`
  - 接入 `DRAMSim3`
  - 适合更细粒度的外存时序分析

### 6. 统计与分析

仿真结束后会输出：

- 总周期数
- 指令行数
- 完整指令数
- 每个模块的 busy cycle 和利用率
- DDR / SPM / ACC Bias / ACC Data 的带宽统计

---

## 项目结构

```text
PIMU-Sim/
├── include/                  # 公共头文件
│   ├── config.hpp
│   ├── logger.hpp
│   ├── regfile.hpp
│   ├── sim_base.hpp
│   └── perf/
│       ├── perf_sim.hpp
│       ├── perf_npu_core.hpp
│       ├── decoder.hpp
│       ├── inst_cache.hpp
│       ├── inst_queue.hpp
│       ├── event_sync_unit.hpp
│       ├── scratchpad.hpp
│       ├── acc_sram.hpp
│       ├── hw_utilization.hpp
│       └── perf_func_unit.hpp
├── src/
│   ├── main.cpp
│   ├── config.cpp
│   ├── logger.cpp
│   ├── regfile.cpp
│   ├── func_sim.cpp
│   └── perf/
│       ├── perf_sim.cpp
│       ├── perf_npu_core.cpp
│       ├── decoder.cpp
│       ├── inst_cache.cpp
│       ├── inst_queue.cpp
│       ├── event_sync_unit.cpp
│       ├── scratchpad.cpp
│       ├── acc_sram.cpp
│       ├── analytical_ddr.cpp
│       ├── detail_ddr.cpp
│       └── func_units/
│           ├── dma.cpp
│           ├── systolic_array.cpp
│           ├── accumulator.cpp
│           ├── sfu.cpp
│           └── gemv.cpp
├── config/                   # 仿真配置文件
├── benchmarks/tb_test/       # 指令测试文件与生成脚本
├── log/                      # 仿真输出日志
├── external/DRAMsim3/        # DRAMSim3 依赖
├── Makefile
├── Env.cfg
├── README.md
└── Simulator.md              # 开发记录与设计说明
```

---

## 环境依赖

推荐环境：

- `g++`  with C++ 17
- `make`
- Linux
- 项目内已包括`DRAMSim3`

如需详细DDR仿真，需要能够正常加载：

- `external/DRAMSim3/libdramsim3.so`

---

## 构建方法

首次进入新终端后建议先加载环境：

```bash
source Env.cfg
```

编译：

```bash
make build
```

生成可执行文件：

```bash
./build/npu_sim
```

---

## 运行方法

直接使用`Makefile`运行默认测试：

```bash
make run
```

默认运行的配置由`Makafile`中的`TOP`决定，例如：

```makefile
TOP = resnet_block
```

对应运行：

```bash
./build/npu_sim ./config/resnet_block_cfg.txt
```

清理构建文件：

```bash
make clean
```

---

## 配置文件说明

配置文件位于`./config`，当前主要字段包括：

仿真模式

- simulator_type
    - FUNCTIONAL
    - PERFORMANCE

核心配置

- max_sim_cycle
- core_mode
- systolic_array_size
- npu_freq

DDR 配置

- ddr_mode
- dramsim_cfg_path
- ddr_freq
- ddr_bandwidth
- tCAS / tRCD / tRP / tBURST
- BL
- ddr_bus_width
- ddr_channel_num
- ddr_request_queue_depth

SPM 配置

- sram_bank_num
- sram_bank_bytes
- sram_ports
- sram_data_width

ACC 配置

- acc_bank_num
- bias_bank_num
- acc_bytes
- acc_ports
- acc_data_width

指令系统配置

- inst_cache_size
- inst_queue_depth

文件路径

- inst_file
- log_file
- enable_log

---

## 测试

当前`benchmarks/tb_test`与`config`中已经包含多类测试：

算子测试：

- dma
- gemm
- conv
- matadd
- gelu
- softmax
- layernorm
- poolmax
- transpose

模型测试：

- cnn_raw
- resnet_block

模型生成python脚本：

- gen_resnet_block.py

其中：

- tb_xxx.txt：原始指令文件
- xxx_cfg.txt：对应配置文件
- tb_xxx_log.txt：运行日志

---

## 当前性能模型执行流程

在性能模型下，`PerfSim`会推进两个主要模块：

- `PerfNpuCore`
- `DDR`

`PerfNpuCore`每周期主要执行：

- 1.中断处理
- 2.清空 SPM / ACC SRAM 端口状态
- 3.推进各功能单元
- 4.处理 DMA 与 DDR 的消息收发
- 5.采样硬件利用率
- 6.发射 InstQueue
- 7.推进 Decoder
- 8.从指令文件继续加载指令
- 9.执行 retire 和完成判断

以上执行顺序根据硬件执行时序进行调整

---

## 当前项目状态

当前项目已经完成：

- ISA 指令文件驱动
- 周期级前端流水
- 多功能单元并行推进
- PM / ACC SRAM 建模
- EventSyncUnit 同步控制
- Analytical DDR / DRAMSim3 DDR 双模式
- 模块利用率与带宽统计
- 基础算子和组合测试用例

当前仍在持续优化的部分包括：

- 多 DMA 分发策略
- 更复杂组合 workload 下的队列死锁规避
- 指令流生成与调度策略
- 更系统的回归测试

---

## 开发说明

如需继续扩展项目，建议优先查看：

- `Simulator.md`：开发记录
- `src/perf/perf_npu_core.cpp`：核心调度入口
- `src/perf/decoder.cpp`：前端流水
- `src/perf/inst_queue.cpp`：完整指令组包与发射
- `src/perf/event_sync_unit.cpp`：valid / busy 同步控制
- `src/perf/func_units/`：各功能单元建模