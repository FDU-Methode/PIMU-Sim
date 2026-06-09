# PIMU-Sim

PIMU-Sim 是一个面向 NPU ISA 与编译器运行时的仿真器项目。当前分支为 `dev_compiler`，主线是**由 ONNX / onnx-mlir 编译产物驱动的性能仿真**：运行编译后的模型共享库，通过 `npux-mlir` runtime 向仿真器外部喂入 NPU 指令，并按周期推进 NPU Core、DMA、片上存储和 DDR 模型。

代码中保留了 `FUNCTIONAL` / `PERFORMANCE` 两种模式选择，但当前 `FuncSim::run()` 仍为空实现；实际可用和持续维护的是 `PERFORMANCE` 路径。

---

## 当前能力

### 编译器运行时路径

`src/main.cpp` 在 `PERFORMANCE` 模式下采用当前分支的默认执行路径：

```text
ONNX model 或 compiled .so
    -> onnx-mlir ExecutionSession
    -> npux-mlir npu_runtime
    -> PerfSim / PerfNpuCore external instruction feed
    -> output_*.bin + output_*.meta.txt
```

主要流程：

- 创建 `PerfSim`。
- 通过 `npu_bind_simulator_backend(perf_sim)` 绑定 runtime 后端。
- 调用 `npu_init()` 初始化 NPU runtime。
- 调用 `PerfNpuCore::enableExternalInstFeed()`，切换为外部指令喂入模式。
- 如果 `compiled_model_lib` 非空，直接加载已有模型共享库。
- 如果 `compiled_model_lib` 为空，则使用 `onnx_model_file`、`onnx_mlir_bin`、`onnx_mlir_flags` 在 `compile_work_dir` 下编译生成 `.so`。
- 从 `model_input_dir` 读取输入张量二进制文件。
- 将输出写入 `model_output_dir`。

注意：当前 `src/main.cpp` 会在 `PERFORMANCE` 模式启动时清空 `inst_file`，因此默认可执行路径不是原始指令文件驱动。`PerfNpuCore` 中仍保留了指令文件读取能力，但它不是当前 main 的默认运行方式。

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

- `MVIN` / `MVOUT`：DRAM 与 SPM/ACC 之间的数据搬运。
- `SA`：脉动阵列路径，支持 GEMM / CONV 建模，并可配合 ACC 做累加。
- `MATADD`：从 ACC Data SRAM 读取两个矩阵，输出到 SPM。
- `SFU`：支持 softmax、gelu、layernorm、pool_max、transpose。

### 前端与 DMA 分发

性能仿真核心仍使用周期级前端：

```text
InstCache -> Decoder(fetch/decode/dispatch)
          -> DMA/SA/SFU InstQueue -> Functional Units
```

在当前分支中，指令通常由 runtime 外部写入仿真器，而不是由 `InstFile` 加载。`Decoder` 会识别 opcode、聚合多行指令，并分发到对应队列。

当前 DMA 分发策略：

- ACC 相关 DMA 指令固定走 `DMA1`。
- SPM `MVIN` 根据队列负载在 `DMA2` / `DMA3` 间选择；负载相同则 round-robin，必要时回退到 `DMA1`。
- SPM `MVOUT` 在 `DMA1` 空闲且无负载时可借用 `DMA1`，否则在 `DMA2` / `DMA3` 间按负载和 round-robin 分发。
- `EventSyncUnit` 基于资源 valid/busy 状态阻止存在数据相关风险的指令提前发射。

### 功能单元与存储

当前性能核心会创建并采样以下模块：

- `DMA1` / `DMA2` / `DMA3`
- `SystolicArray`
- `Accumulator`
- `SFU`
- `GEMV`
- `SPM`
- `ACC Bias SRAM`
- `ACC Data SRAM`

片上存储：

- `SPM`：按 bank 建模，bank 数和容量由 `sram_bank_num`、`sram_bank_bytes` 配置。
- `ACC SRAM`：分为 bias 区和 data 区，由 `bias_bytes`、`acc_bytes` 配置，并分别统计访问与利用率。

`GEMV` 单元目前已实例化并纳入利用率统计，但当前 opcode 分发主路径仍由 DMA、SA、MATADD 和 SFU 覆盖。

### DDR 模型

支持两种 DDR 模式：

- `AnalyticalDDR`：解析模型，按 AXI beat、burst、基础时序参数和请求队列估算访问延迟。
- `DetailDDR`：接入 `DRAMSim3`，用于更细粒度 DDR 时序实验；当前事务拆分仍较简化。

仿真器会根据 `npu_freq` 与 `ddr_freq` 计算统一 base tick，使 NPU Core 和 DDR 可按各自频率推进。

### 统计与输出

仿真结束后会在标准输出打印，并在 `enable_log = 1` 时写入日志文件：

- 总周期数、最大周期限制、仿真耗时
- 已读取指令行数、已模拟完整指令数、模拟 IPS
- `DMA1/2/3`、`SPM`、`SystolicArray`、`Accumulator`、`ACC Bias/Data SRAM`、`SFU`、`GEMV` 的 busy cycle 与利用率
- DDR、SPM、ACC Bias、ACC Data 的读写字节数、平均带宽、峰值 B/cycle

当前分支不包含功耗分析模块。

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
│       ├── scratchpad.cpp
│       └── func_units/
│           ├── accumulator.cpp
│           ├── dma.cpp
│           ├── gemv.cpp
│           ├── sfu.cpp
│           └── systolic_array.cpp
├── config/
│   ├── mnist_cfg.txt
│   └── resnet_block_cfg.txt
├── benchmarks/tb_test/
│   ├── gen_resnet_block.py
│   └── tb_resnet_block.txt
├── external/DRAMsim3/
├── npux-mlir/                 # 当前分支预期的 runtime / model 接入目录
├── log/
├── Makefile
├── Env.cfg
├── Simulator.md
└── README.md
```

`Makefile` 会编译 `npux-mlir/model_test/runtime/`、`npux-mlir/src/Runtime/` 和 `npux-mlir/src/Support/` 下的运行时源文件。若当前工作区中的 `npux-mlir/` 为空或缺少这些路径，需要先补齐对应工程内容，否则 `make build` 会找不到 runtime 源文件。

---

## 环境依赖

推荐在 Linux、WSL 或 MSYS2 这类 POSIX shell 环境下使用。当前 `Makefile` 和 `Env.cfg` 使用了 `find`、`mkdir -p`、`rm -rf`、`source`、`LD_LIBRARY_PATH` 等 POSIX 习惯；原生 Windows PowerShell 下通常需要额外适配。

基础依赖：

- `g++`，支持 C++17
- `gcc`
- `make`
- `bash`
- `pthread`
- `dl`
- `external/DRAMsim3/libdramsim3.so`

编译器运行时路径还需要：

- `npux-mlir` / `onnx-mlir` 运行时源文件，放在 `npux-mlir/` 目录下。
- 可执行的 `onnx-mlir`，用于从 ONNX 编译 `.so`。
- 对应模型目录，包含 ONNX 文件和输入张量二进制文件。

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

如果构建时报 `npux-mlir/.../npu_runtime.cpp`、`ExecutionSession.cpp` 或 `OMTensor.cpp` 等文件不存在，说明当前分支依赖的 `npux-mlir/` 内容尚未补齐。

清理构建目录：

```bash
make clean
```

清理 `run_onnx` 生成的编译工作目录和临时配置：

```bash
make clean_onnx
```

---

## 运行

`Makefile` 当前默认目标为：

```makefile
TOP = mnist
```

因此默认运行：

```bash
make run
```

等价于：

```bash
./build/npu_sim ./config/mnist_cfg.txt
```

`config/mnist_cfg.txt` 默认使用：

```text
onnx_model_file = ./npux-mlir/model_test/models/mnist/model.onnx
onnx_mlir_bin = ./onnx_mlir_bin
model_input_dir = ./npux-mlir/model_test/models/mnist
model_output_dir = ./sim_output/mnist
```

如果 `./onnx_mlir_bin` 不存在或不可执行，需要修改配置中的 `onnx_mlir_bin`，或使用 `make run_onnx` 显式传入编译器路径。

### ONNX 驱动运行

`run_onnx` 会先调用 `onnx-mlir` 编译模型，再生成临时配置并运行仿真：

```bash
make run_onnx \
  TOP=mnist \
  MODEL_ONNX=./npux-mlir/model_test/models/mnist/model.onnx \
  MODEL_INPUT_DIR=./npux-mlir/model_test/models/mnist \
  ONNX_MLIR_BIN=/path/to/onnx-mlir
```

可选参数：

- `MODEL_OUTPUT_DIR`：默认 `./sim_output/$(TOP)`
- `ONNX_MLIR_FLAGS`：传给 `onnx-mlir` 的额外参数
- `COMPILER_WORK_DIR`：默认 `./build/compiler_run/$(TOP)`

`run_onnx` 生成的临时配置位于：

```text
build/generated/<TOP>_run_cfg.txt
```

### 使用已有共享库

如果配置文件中设置了 `compiled_model_lib`，仿真器会直接加载该 `.so`，不再调用 `onnx-mlir` 编译。例如 `config/resnet_block_cfg.txt` 当前设置了：

```text
compiled_model_lib = ./npux-mlir/model_test/build/NPU_conv/model.so
```

注意：该配置文件中仍保留了 `inst_file = ./benchmarks/tb_test/tb_resnet_block.txt`，但当前 `src/main.cpp` 会清空 `inst_file` 并启用外部指令喂入，因此运行时仍以编译模型路径为主。

### 输入与输出文件

输入张量从 `model_input_dir` 读取，支持的候选文件名包括：

- `input_<index>.bin`
- 单输入模型的 `input.bin`
- `<input_name>.bin`
- `<model_dir_name>_<input_name>.bin`
- 单输入模型的 `<model_dir_name>_input.bin`

输出写入 `model_output_dir`：

```text
output_0.bin
output_0.meta.txt
output_1.bin
output_1.meta.txt
...
```

`.meta.txt` 中包含 `dtype`、`rank`、`shape` 和 `buffer_size`。

---

## 配置文件

配置文件采用简单的 `key = value` 格式，支持 `#` 注释。当前 `Config::from_file()` 实际解析的主要字段如下。

仿真、日志与运行时：

- `simulator_type`：`PERFORMANCE` / `FUNCTIONAL`，当前建议使用 `PERFORMANCE`
- `max_sim_cycle`
- `inst_file`
- `log_file`
- `enable_log`
- `compiled_model_lib`
- `onnx_model_file`
- `onnx_mlir_bin`
- `onnx_mlir_flags`
- `compile_work_dir`
- `model_input_dir`
- `model_output_dir`

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

当前配置文件中还保留了 `core_mode`、`sram_ports`、`acc_ports` 等字段，但当前解析器没有消费这些字段；修改它们不会影响现有仿真行为。

---

## 模型与 benchmark

当前分支保留的配置和 benchmark：

| 名称 | 配置文件 | 当前用途 |
| --- | --- | --- |
| `mnist` | `config/mnist_cfg.txt` | 默认 ONNX 编译运行示例 |
| `resnet_block` | `config/resnet_block_cfg.txt` | 保留旧指令文件字段，同时指向 conv 模型 `.so` / ONNX 路径 |
| `tb_resnet_block` | `benchmarks/tb_test/tb_resnet_block.txt` | 旧式原始指令流测试数据 |

`benchmarks/tb_test/gen_resnet_block.py` 可生成旧式 ResNet-like block 指令流。该路径仍可作为低层指令格式和功能单元调试参考，但不是当前 `src/main.cpp` 的默认性能仿真入口。

---

## 当前项目状态

已经完成或当前可用：

- `PERFORMANCE` 性能仿真主路径。
- ONNX / compiled `.so` 驱动的模型执行入口。
- `npux-mlir` runtime 到 `PerfSim` 的后端绑定。
- 外部指令喂入模式。
- 三路 DMA、SA、ACC、SFU、SPM、ACC SRAM、DDR 的周期级建模。
- `AnalyticalDDR` 与 `DetailDDR` 双模式。
- 输出张量 dump 到 `sim_output/`。
- 模块利用率与带宽统计。

仍需继续完善：

- `FUNCTIONAL` 模式目前为空实现。
- 当前 main 默认不走原始指令文件路径；如果要恢复该路径，需要调整入口逻辑。
- 当前工作区需要补齐 `npux-mlir/` 运行时和模型内容后才能完整构建运行。
- `DetailDDR` 的事务拆分和时序细节仍较简化。
- `GEMV` 尚未接入当前 opcode 分发主路径。
- 当前分支不包含功耗分析模块。
- 缺少自动化回归测试入口。

---

## 开发入口

建议优先阅读以下文件：

- `Simulator.md`：开发记录与设计说明
- `Makefile`：构建规则、runtime 源文件、`run_onnx` 临时配置生成
- `src/main.cpp`：程序入口、ONNX 编译、ExecutionSession 执行、输入输出 dump
- `src/config.cpp`：配置解析
- `src/perf/perf_sim.cpp`：性能仿真顶层、base tick、外部执行计时
- `src/perf/perf_npu_core.cpp`：核心周期推进、外部指令喂入、DMA/DDR 连接
- `src/perf/decoder.cpp`：opcode 分类、指令行数和 DMA 分发策略
- `src/perf/inst_queue.cpp`：指令队列、发射、retire 和 ISR 更新
- `src/perf/event_sync_unit.cpp`：资源同步与相关性控制
- `src/perf/func_units/`：DMA、SA、ACC、SFU、GEMV 功能单元
- `src/perf/hw_utilization.cpp`：利用率与带宽统计
- `npux-mlir/model_test/runtime/`：当前分支预期的 NPU runtime 适配层

---

## License

本项目使用 Apache License 2.0，详见 `LICENSE`。
