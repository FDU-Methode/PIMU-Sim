# PIMU-Sim 软件栈实现方案

本文档规定 PIMU-Sim 软件栈的目标架构、ISA、模型前端、编译器、运行时、验证方法与实施顺序。
**目标是通用的多模型软件栈，首个交付实例为完整 Qwen3 模型的 Prefill 和多步 Decode 性能仿真，后续通过统一接口接入其他模型。**
适用后端为 **`dev` 系列分支的 Parallel PIMU**：单 NPU 核内部的流水线、多队列、
三 DMA 并行性能仿真。软件栈代码和运行产物统一放在 `PIMU-Sim/sw/`。
**DDR 统一使用 AnalyticalDDR**，所有模型、Prefill/Decode 阶段及验收均采用该后端。
为覆盖 Qwen3，需要补充必要的算子性能模型和版本化指令能力，同时保留上述 Parallel PIMU 架构。
仿真器接入主线采用 **Python Runtime + 可复用 C++ 仿真器库 + 独立 CLI**，由持久会话
连续执行 Prefill 和多步 Decode；具体重构方案见第 6 节，所列接口均为待实现要求。
openTPU 仅作为软件分层和 API 设计参考，不修改其代码，也不采用其指令编码、存储布局或板卡接口作为 PIMU 的硬件契约。

本文区分三类内容：**当前行为**描述兼容基线的实际实现；**实现要求**约束后续软件栈；
**后续扩展**需要新增接口或后端能力，不能视为已经可用。
已有三个 `.txt` 仅用于底层 ISA/后端兼容回归，不是软件栈的模型输入、支持列表或验收上限。

## 目录

- [1. 设计目标与总体方案](#1-设计目标与总体方案)
- [2. 目标后端与参考实现](#2-目标后端与参考实现)
- [3. ISA 详细规范](#3-isa-详细规范)
- [4. PIMU-Sim 与 openTPU 的概念映射](#4-pimu-sim-与-opentpu-的概念映射)
- [5. 推荐的软件栈目录](#5-推荐的软件栈目录)
- [6. Parallel PIMU backend 接入契约](#6-parallel-pimu-backend-接入契约)
- [7. Python ISA 与编译器设计](#7-python-isa-与编译器设计)
- [8. Runtime 和内存模型](#8-runtime-和内存模型)
- [9. Qwen3 模型接入与通用算子库](#9-qwen3-模型接入与通用算子库)
- [10. Qwen3 必需的性能模型与 ISA 扩展](#10-qwen3-必需的性能模型与-isa-扩展)
- [11. Decode 和控制流策略](#11-decode-和控制流策略)
- [12. 验证体系和验收标准](#12-验证体系和验收标准)
- [13. 分阶段实施计划](#13-分阶段实施计划)
- [14. 第一轮实际编码顺序](#14-第一轮实际编码顺序)

## 1. 设计目标与总体方案

### 1.1 软件栈分层

PIMU-Sim 应采用“Python 编译器/runtime + C++ performance backend”的结构：

```text
Qwen3 checkpoint/config + request(prompt length, decode steps)
        |
        v
ModelAdapter registry -> model graph -> generic operator IR
        |
        v
kernel lowering -> tiler/bank allocator/scheduler -> PIMU instruction stream
        |                                      |
        |                                      +--> manifest + memory plan
        v
Python runtime -> PerformanceBackend -> C ABI -> C++ PerfSession
                                                   |
                                                   v
                                      Core / queues / units / DDR
                                                   |
                                                   +--> status/cycles/bytes/power/trace
```

其中：

1. Python 层负责模型/权重元数据、张量布局、量化元数据、分块、指令生成、runtime 生命周期和性能结果回收。
2. C++ 层只负责周期级性能模型、队列/资源冲突、DDR、利用率、带宽、功耗和 trace；不建设功能解释器或数值结果回收路径。
3. Python compiler 的输入是模型的 shape、dtype、layout、权重大小和执行计划，输出是 PIMU 指令、内存布局和性能仿真 manifest；性能仿真结果是周期和硬件统计。
4. 首版采用单 NPU core、静态编译的 Prefill/Decode 程序和显式 W8A8/ACC32 性能映射假设，保留 DMA1/2/3、SA、SFU 并行。该映射不代表已经实现原始 checkpoint 的数值量化或 BF16 性能支持。
5. 主线从 Qwen3 模型配置和权重元数据构建真实算子图，补齐所需性能模型，再完成单层、全层和多步 Decode。RMSNorm、Q/K Norm、RoPE、SwiGLU、attention scale/mask 是首版必需项。
6. compiler 必须针对硬件配置生成指令顺序、bank 生命周期和复用位，并验证有限深度队列下的执行进展；不能只生成合法 hex，也不能靠全局 barrier 串行化来替代正确的依赖分析。
7. 同一请求复用 C++ 会话，Runtime 按完整 Prefill/chunk 或 Decode step 提交程序；阶段切换保留约定的设备状态及模拟时间。txt/hex 是程序的导入导出格式，CLI 是独立入口，不作为模型执行的唯一控制通道。

性能模式中的“运行模型”是：导入模型结构、参数形状和请求，自动生成并执行覆盖完整模型的
计算、访存、KV cache 和调度指令，输出周期、带宽、利用率与能耗。不建设 functional model，
也不要求 simulator 计算 token 数值；多步请求由指定长度或外部 token trace 驱动，不能将
性能结果描述为已经生成了正确文本。默认用 Qwen3-0.6B 验证链路，参数不写死在编译器中。

### 1.2 `pimu.language` 语言层

`pimu.language` 作为通用 kernel 入口；模型入口则由 ModelAdapter 自动组图。语言先提供
静态、硬件感知的 IR builder，不实现完整 Triton 级 DSL。既有五类指令和 Qwen3 必需的
扩展内核共享同一 lowering/调度路径，不通过复制其他架构的线程、布局和控制流语义隐藏
PIMU 的 bank/队列约束。

推荐分三步：

1. **先定义 IR 和显式 builder**：提供 `TensorDesc`、`WeightDesc`、`TilePlan` 以及 `gemm()`、`matadd()`、`sfu()`、`load()`、`store()` 等 API。它们直接生成 IR，由 compiler 做 tiling、bank allocation 和 scheduling。
2. **在同一层加入 `pimu.language`**：推荐使用 `import pimu.language as pl`，并以 `@pl.kernel` 作为 kernel 入口。装饰器接收 symbolic descriptor，调用 kernel 函数时只允许发射静态 PIMU op；`pl.static_range`、`pl.tiles` 等构造器展开编译期循环。这样 `pl` 是 builder/IR 的易用语法层，不再额外维护一套语义。
3. **后续再增加语法糖**：可以提供类似 `ol.load`、`ol.dot`、`tl.load` 的短写法，但必须继续降低到同一套 IR，并保留显式 layout、stride、bank 和资源约束。

因此结论是：**可以引入 `pimu.language`，建议以 `pl` 作为短别名并使用 `@pl.kernel`；实现轻量、静态、硬件感知的版本，不实现完整 Triton 语言，也不支持任意 Python 程序自动变成 PIMU 指令。**

`@pimu` 的可行性来自三点：Python 本身适合承载模型配置和编译期循环；PIMU 指令是固定格式，容易由 builder 生成；PIMU 的 bank/queue 约束可以在 IR verifier 中提前检查。它的主要限制是：不能有数据相关分支，不能依赖运行时循环次数，不能隐藏 DMA 或 bank 选择，也不提供 GPU 式线程/warp 抽象。

建议的最小形式如下：

```python
import pimu.language as pl

@pl.kernel
def gemm(a, b, c, M, N, K):
    for m, n in pl.tiles(M, N):
        for k in pl.static_range(0, K, pl.K_TILE):
            pl.mvin(a.tile(m, k), pl.spm("A", k & 1))
            pl.mvin(b.tile(k, n), pl.spm("B", k & 1))
            pl.sa("A", "B", c.tile(m, n), accumulate=(k != 0))
        pl.mvout(c.tile(m, n))
```

这个例子只是拟议语法，不是当前可执行 API。`pl.spm("A"/"B", ...)` 是编译器分配的逻辑 bank 角色，
不是 Banked PIMU 的硬件功能分区；`pl.sa` 还须 lowering 为首块/中间块/末块的 ACC 控制。
kernel 发射 IR，compiler 决定完整指令流，仿真器继续根据自身队列和 EventSyncUnit 动态发射；
Python 不给硬件注入预定开始周期，也不把每次 `pl.mvin` 当作 host 同步点。
这里 `pl` 是模块别名，正式入口使用 `@pl.kernel`，后续可增加 `@pl.program` 等入口。

## 2. 目标后端与参考实现

### 2.1 PIMU-Sim 当前已经具备的能力

PIMU-Sim 当前主线是性能模型，入口是 `src/main.cpp` -> `PerfSim` -> `PerfNpuCore`。
本文采用 AnalyticalDDR 配置的核心流水线为：

```text
hex text file
  -> InstCache
  -> Decoder(fetch/decode/dispatch)
  -> DMA/SA/SFU InstQueue
  -> EventSyncUnit
  -> DMA/SystolicArray/Accumulator/SFU/GEMV
  -> AnalyticalDDR
```

当前 PIMU 指令以一条 128-bit 十六进制行表示：

| opcode | 语义 | 行数 | 当前目标队列 |
|---|---|---:|---|
| `0x01` | MVIN | 2 | DMA |
| `0x02` | MVOUT | 2 | DMA |
| `0x03` | SA | 3 | SA |
| `0x04` | MATADD | 3 | SA |
| `0x05` | SFU | 2 | SFU |

已建模的主要算子是 GEMM/CONV、MATADD、softmax、GELU、layernorm、pool_max、transpose 以及 DMA 的 SPM/ACC 搬运。`Decoder` 会按 opcode 聚合多行指令，`EventSyncUnit` 依据 SPM、ACC data、ACC bias 的 bank valid/busy 状态控制发射。

PIMU-Sim 已有几个对软件栈很有价值的设计：

- DMA 请求有 row/column、DRAM stride、local stride 和 AXI burst 规划，适合由编译器直接生成二维 tile 搬运。
- DMA1/2/3 分发和 SRAM bank scoreboard 已经存在，编译器可以把 bank 约束作为显式资源约束。
- DDR 统一使用 `AnalyticalDDR`，Runtime 接收其有效配置与性能统计，不提供 DDR 后端选择流程。
- 配置文件已经包含频率、SRAM/ACC 容量、cache/queue 深度、寄存器地址和日志/功耗选项。

### 2.2 PIMU-Sim 当前阻碍实际模型运行的缺口

当前缺口以“完整 Qwen3 能否自动编译并执行”为优先级：

1. PIMU 指令文件解析只提供文本输入，缺少稳定的指令 AST/encoder/disassembler、版本号和 manifest。
2. `Message` 只描述 DMA 请求及少量属性，性能 workload 缺少统一的 request id、算子名和 trace metadata。
3. 当前没有 host runtime：没有 buffer/address allocator、权重布局、程序生命周期或多次 launch API。
4. 缺少模型注册、Qwen3 adapter、模型级 IR、算子 lowering、tile planner、bank allocator 和依赖调度器；现有手写 generator 不构成模型前端。
5. 当前 LayerNorm/GeLU 不能替代 Qwen3 的 RMSNorm/SwiGLU；必须覆盖 Q/K 按 head 归一化、RoPE、scale/mask、辅助参数读取与对应性能契约。KV append/read 优先降低成现有 DMA/transpose。
6. 当前没有 LOOP/WAIT/HALT/host mailbox 协议。静态程序通过 EOF 加后端排空完成；第一版 host 生成/重复 launch decode workload 即可，不需要增加终止 opcode。
7. 统计输出主要面向人读日志，缺少机器可读的 JSON、每条指令的 trace schema、算子级 cycle breakdown 和编译期资源报告。

### 2.3 openTPU 软件栈的可复用思想

openTPU 的完整链路是：

```text
model kernel
  -> @ol.jit tracing
  -> Tensor/QTensor/Stationary/Affine layout
  -> compiler Builder
  -> 8 x 32-bit ISA
  -> bit-exact ISA simulator / RTL / Board backend
  -> runtime place/launch/collect
  -> model engine / host service
```

应复用的不是具体 ISA，而是以下边界：

- **ISA 描述集中化**：每种指令有结构化构造器、编码、解码、字段检查和反汇编。
- **显式数据布局**：Tensor/Weight/KV cache 都有地址、shape、stride、dtype 和 ownership，而不是 kernel 中散落裸地址。
- **编译时可见的数据搬运**：load/store/量化/矩阵乘/elementwise 都降低为可观察指令。
- **kernel tracing 与 runtime placement 分离**：kernel 描述计算，runtime 决定 memory plan、权重分片和输出位置。
- **统一 backend contract**：性能 simulator、未来 RTL 和板卡 backend 的输入是同一份 compiled program、memory plan 和 manifest。
- **性能可解释**：compiler 产生算子/tiling 标识，backend 返回周期、DDR/SPM/ACC 字节、队列等待、bank 冲突和模块利用率。
- **模型层和硬件层解耦**：attention、MLP、norm、KV cache 通过 kernel library 组合，模型适配器只负责权重和层参数。
- **程序提交与设备执行分离**：openTPU 的 Engine 通过 Backend 提交已编译程序，ISA Backend 持有持久 Machine，并用 load/run 执行阶段程序、保留存储状态。PIMU 采用相同的生命周期边界，以 C ABI 连接 Python 与 C++；性能模式不照搬真实 tensor/logits 的读写接口。

不应直接照搬的部分：

- openTPU 的 8 x 32-bit 指令、TMEM、ACT RAM、MX 格式和 register/loop 语义。
- openTPU 的 `GATHER`、`WAITW`、HALT CHAIN、多 slice collective；PIMU 当前没有对应硬件。
- openTPU host 的 PCIe/XDMA、板卡寄存器和双 DDR channel 映射。
- openTPU 以 VPU 指令表达的任意 elementwise；PIMU 当前只有 SFU 的固定 optype，需显式扩展或拆成现有指令。

### 2.4 Parallel PIMU 架构约束

| 架构特性 | 软件栈实现要求 |
|---|---|
| 前端与异构队列解耦 | 保留逐 128-bit 行输入、完整包入队、各队列独立发射的周期行为；compiler 输出一条全局指令流，不直接提供各 DMA 的独立程序。 |
| 三 DMA、一个 SA 队列、一个 SFU 队列 | 理论最大发射宽度为 5。SA 与 MATADD 共享 SA 队列和 Accumulator，SA→Accumulator 是流式耦合，不是第六条独立发射流水线。 |
| 仅 DMA1 支持 ACC 转换路径 | ACC 搬运固定 DMA1；SPM 搬运由 decoder 轮询/空闲判定选择，ISA 不增加 channel 字段，目标不新增 DMA4。 |
| bank valid/busy 隐式同步 | IR 跟踪 producer/consumer、数据代次和生命周期，编码成地址、顺序及 A/B/bias keep 位，不插入未支持的硬件 WAIT/FLAG。 |
| 同质多 bank SPM | A/B/SFU/output 区域是可调整的软件分配策略，硬件没有固定 A/B/O bank 类型或专用 DMA 连接。 |
| DDR 与 core 独立时钟 | 使用后端实际调度结果；compiler cost model 只用于选择方案，不以算子周期求和替代仿真。 |
| 有限队列与反压 | queue_depth 是编译与验证输入，必须检查执行进展；加深队列不保证性能提升。 |
| 配置驱动的分块 | DSE 同时记录硬件配置、编译策略和最终指令；改变容量后重新 tiling/placement，区分固定程序和重新编译两类实验。 |

### 2.5 当前实现限制与兼容规则

当前 dev 源码决定可运行契约。后端修复必须单独形成版本和性能基线，避免将硬件行为变化
计入软件栈优化收益。

| 主题 | 当前实现 | 软件栈处理 |
|---|---|---|
| 输入 busy | `issueUpdate()` 主要锁输出，未完整锁住 SA/MATADD/SFU 输入 | verifier 检查实际读生命周期与端口，不假定全部输入已被锁定。 |
| 前端吞吐 | `decode()` 成功置 decode_valid 后，同周期不继续 fetch；`fetch()` 也检查 decode_valid | 复现实际门控：无阻塞时相邻 word 的分发最早间隔 2 个 core 周期；后端最大发射宽度 5 不等于前端每周期接受 5 条指令。 |
| 存储冲突 | SPM/ACC 返回 false 后，多处调用立即 assert | 编译出无冲突程序；不能将非法端口访问当作自动重试的性能停顿。 |
| 端口配置 | 示例 config 中有 sram_ports/acc_ports，当前 Config parser 不读取；SPM 是每 bank 单周期一次访问，ACC 固定两 data/bias port | 以有效代码能力验证，不能根据文本中的 `ports=2` 假设 SPM 同 bank 同时可读写。 |
| DMA 轮询 | SPM MVIN 在 DMA2/3 轮询；SPM MVOUT 仅在 DMA1 完全空闲时借用 | 验证器复现相同路由规则，不静态指定 DMA 通道。 |
| 缓存容量 | `full()` 比较的是 `Inst` 个数 | inst_cache_size 以 128-bit word 为单位，256 项对应 4096 字节编码容量；队列深度以完整包计。 |
| 队列顺序 | 每个队列只检查队首 | 建模为队内 FIFO、跨队列可超越，同队列 ready 指令不能跳过队首。 |
| MATADD | 从两路 ACC 输入执行加法，结果写 SPM | 定义为 ACC+ACC→SPM 矩阵加法。 |
| CONV 尺寸 | timing 行数截成 6 bit，且存在位域重叠 | 保留第 3 节实际公式和位重叠，未修复前标记 experimental。 |
| 大阵列 DSE | A 行数/B 列数字段只有 5 bit | SA 单 tile 的 M/N≤32、K≤2048；模型维度可更大，compiler 必须分块。64 行/列单 tile 需另行修订 ISA。 |
| DDR 响应身份 | `tryConsumeResp` 只按 LD_RESP/ST_RESP 类型匹配 | 多 DMA 程序先采用当前行为兼容验证；增强并发前验证错配风险，request_id 修复后独立建立基线。 |

### 2.6 首个模型范围与回归样例定位

| 对象 | 作用与范围 |
|---|---|
| Qwen3 Dense | 首个模型家族。默认 Qwen/Qwen3-0.6B，配置驱动适配同类 Dense 变体；从 embedding 到全部 decoder layers、final norm、lm_head，覆盖 Prefill 和多步 Decode。 |
| 后续模型 | 通过 ModelAdapter 注册、通用 IR 和 kernel capability 接入，不复制模型专用 ISA/compiler/runtime。 |
| MoE、VL 等不同架构 | 需要专家路由、视觉编码器等独立图与性能契约；首版不因名称相似就宣称已支持。 |
| 现有三个 txt | 用于既有 encoder/parser、单元和后端行为回归；可直接复用，不要求先重写其 generator，也不要求 Qwen3 指令数或周期与它们相同。 |

模型性能基线绑定 model revision、配置/权重元数据、请求、执行精度映射、硬件 profile 和
后端版本。完整模型验收不能缩小层数、换成简化激活、遗漏 lm_head 或用旧单层负载替代。
正式交付必须从模型入口自动生成程序；手写 Qwen3 hex 或按旧 txt 修改尺寸仅可用于调试。

## 3. ISA 详细规范

本节独立列出 ISA 的字段位置、位宽、名称与当前 dev 性能行为。C++ 决定哪些字段影响
周期、访存和同步，generator 用于对照现有编码。
“已定义但模型未消费的字段”与“保留位”必须区分，不将前者作为可自由复用的新位。
量化数值语义和未细分 config bit 不作推断；bit 19 的 CONV 重叠保留源码事实。

### 3.1 公共编码约定

- 一条指令行是 128 bit 十六进制文本，格式为 `high(64 bit)` 拼接 `low(64 bit)`，即文本最左侧对应 bit 127，最右侧对应 bit 0。
- C++ 的 `Inst.low` 对应指令 bit `[63:0]`，`Inst.high` 对应 bit `[127:64]`；下表中的 `low[x:y]`、`high[x:y]` 均按各自 64-bit word 的 bit 编号书写。
- 只有第 0 行的 `low[7:0]` 是 opcode。当前 decoder 按 opcode 固定聚合后续行：MVIN/MVOUT/SFU 为 2 行，SA/MATADD 为 3 行。续行的 `low[7:0]` 是普通数据位，例如 DMA 的 local stride 和 SA 的 config2，**不要求为 0**；decoder 给续行设置的内部 `opcode=0` 只是队列接口标记，不会改写指令数据。
- 地址字段均为 32-bit 无符号**字节地址**。SPM、ACC data、ACC bias 是独立地址空间：SPM/ACC data 地址相对各自空间起点，bias 地址相对 bias 空间起点。DRAM 地址也是字节地址。DMA 的 `sram_stride/dram_stride` 按元素数解释并乘以解码出的元素字节数；SA 的输入 stride 直接表示字节增量，不能套用 DMA 的换算规则。
- `rows`/`cols` 的字段存储值是 `实际数量 - 1`。C++ 在 DMA、SA、MATADD、SFU 中都通过 `+1` 恢复数量。generator 的两个 helper 对参数的命名不完全一致，但生成的最终编码遵循这一约定。

取位公式为 `field = (word >> lsb) & ((1 << width) - 1)`，其中
`width = msb - lsb + 1`。`high[x:y]` 对应整行 bit `[x+64:y+64]`，
`low[x:y]` 对应整行 bit `[x:y]`。`inst0/inst1/inst2` 是同一指令组的第 0/1/2 行，
每行均重新从 bit 0 编号，不把 3 行当成一个由 C++ 直接解码的 384-bit 整数。

文中的“未消费”表示当前值不影响该指令的性能行为；“保留”表示字段表中标注为 reserved 的区域，
并不表示现有 simulator 已经实现保留位必须为零的校验。已命名的量化字段，其定点
小数位、移位方向、有符号性和溢出规则均未由性能模型定义，不能仅凭字段名补全。

### 3.2 opcode 和指令组

| opcode | 指令 | 行数 | 当前执行单元 | 性能模型状态 |
|---:|---|---:|---|---|
| `0x01` | MVIN | 2 | DMA1/2/3 | 已实现 |
| `0x02` | MVOUT | 2 | DMA1/2/3 | 已实现 |
| `0x03` | SA | 3 | SystolicArray + Accumulator | GEMM/CONV 已实现；GEMV 路径不完整 |
| `0x04` | MATADD | 3 | Accumulator | 已实现 |
| `0x05` | SFU | 2 | SFU | optype 0/1/2/6/8 有 timing path |

### 3.3 MVIN/MVOUT：DMA 搬运指令

MVIN 和 MVOUT 共用位布局。字段 `source` 在两种指令中的方向不同：MVIN 中表示
目的地，MVOUT 中表示源端。

#### 第 0 行

| 字内位段 | 位宽 | 整行 bit | 编码字段 | 当前语义 |
|---|---:|---|---|---|
| `low[7:0]` | 8 | `[7:0]` | `opcode` | `0x01` 为 MVIN，`0x02` 为 MVOUT。 |
| `low[9:8]` | 2 | `[9:8]` | `data_type` | 2-bit 类型字段。当前 MVIN 只使用 bit 1，MVOUT 只使用 bit 0；未形成完整枚举。 |
| `low[11:10]` | 2 | `[11:10]` | `precision` | 2-bit 精度字段，和 `data_type`/`is_quant` 一起决定 DRAM/local 元素字节数。 |
| `low[12]` | 1 | `[12]` | `is_quant` | 量化/扩展路径标志；MVIN 参与 size decode，MVOUT 记录到 request 的 `quant`。 |
| `low[13]` | 1 | `[13]` | `source` | MVIN：`0=SPM, 1=ACC` 目的地；MVOUT：`0=SPM, 1=ACC` 源端。也用于 decoder 将 ACC 流量固定分到 DMA1。 |
| `low[14]` | 1 | `[14]` | `is_bias` | MVIN 写 ACC 时，`1` 表示写 ACC bias bank；当前 MVOUT 强制按 `is_bias=false` 处理。 |
| `low[31:15]` | 17 | `[31:15]` | 保留/未消费 | 当前 C++ 不消费；软件栈新生成的指令建议写 0，simulator 不检查其是否为零。 |
| `low[47:32]` | 16 | `[47:32]` | `col_num_m1` | 列数减一；C++ 使用 `(field + 1)`。 |
| `low[63:48]` | 16 | `[63:48]` | `row_num_m1` | 行数减一；C++ 使用 `(field + 1)`。 |
| `high[31:0]` | 32 | `[95:64]` | local address | MVIN 的 SPM/ACC 目的地址；MVOUT 的 SPM/ACC 源地址。 |
| `high[63:32]` | 32 | `[127:96]` | DRAM address | MVIN 的 DRAM 源地址；MVOUT 的 DRAM 目的地址。 |

DMA `config` 总范围是整行 `[14:8]`，对应上表 data_type、precision、
is_quant、source、is_bias 共 7 bit；不增加额外 config word。

当前 size decode 的实际组合如下；没有列出的组合会落到默认
`{dram_bytes=1, local_bytes=1}`，不能在 compiler 中假定为其他数据类型：

| 指令 | 有效组合 | DRAM 元素字节 | local 元素字节 |
|---|---|---:|---:|
| MVIN | `data_type[1]=0, precision=1, is_quant=0`（int8 -> int8） | 1 | 1 |
| MVIN | `data_type[1]=0, precision=2, is_quant=0`（fp16 -> fp16） | 2 | 2 |
| MVIN | `data_type[1]=1, precision=2, is_quant=0`（bias/fp16 路径） | 2 | 2 |
| MVIN | `data_type[1]=1, precision=1, is_quant=0`（int32 -> int32） | 4 | 4 |
| MVIN | `data_type[1]=0, precision=1, is_quant=1`（int8 -> int32 扩展） | 1 | 4 |
| MVOUT | `data_type[0]=0, precision=2`（fp16） | 2 | 2 |
| MVOUT | `data_type[0]=0, precision=1`（int8） | 1 | 1 |
| MVOUT | `data_type[0]=1, precision=1`（int32） | 4 | 4 |

表中的类型由当前 C++ 注释和 size decode 解释；未使用的类型组合不赋予额外语义，
`data_type` 两个 bit 的完整硬件命名仍需另行确认。MVIN 的原始 `low[8]` 和 MVOUT 的原始 `low[9]`
不影响各自的 size decode。`source`、`is_bias` 只决定空间/端口选择，不自动决定元素大小。
MVOUT 的 `is_quant` 不参与 size decode，AnalyticalDDR 也不据此增加量化周期。

#### 第 1 行

| 字内位段 | 位宽 | 整行 bit | 编码字段 | 当前语义 |
|---|---:|---|---|---|
| `low[15:0]` | 16 | `[15:0]` | `sram_stride` | local 每行 stride，按元素数传入 C++，再乘 local 元素字节数。 |
| `low[31:16]` | 16 | `[31:16]` | `dram_stride` | DRAM 每行 stride，按元素数传入 C++，再乘 DRAM 元素字节数。 |
| `low[47:32]` | 16 | `[47:32]` | `qscale` | 量化 scale；generator 支持写入，当前 DMA C++ 不读取。 |
| `low[63:48]` | 16 | `[63:48]` | `qshift` | 量化 scaleshift；当前 DMA C++ 不读取。 |
| `high[31:0]` | 32 | `[95:64]` | `qzp` | 量化 zeropoint；当前 DMA C++ 不读取。 |
| `high[63:32]` | 32 | `[127:96]` | 保留/未消费 | 当前没有语义。 |

### 3.4 SA：GEMM/CONV/GEMV 计算指令

SA 是 3 行一组。第 0 行保存 opcode、卷积/量化元数据；第 1 行保存操作类型、
累加控制、ACC 地址和 stride；第 2 行保存两个 SPM 输入的地址、尺寸和 stride。

#### 第 0 行

| 字内位段 | 位宽 | 整行 bit | generator/代码中的名称 | 当前语义 |
|---|---:|---|---|---|
| `low[7:0]` | 8 | `[7:0]` | `opcode` | 固定为 `0x03`。 |
| `low[8]` | 1 | `[8]` | `config1` 的 bit 0 | 属于 `config1`，子语义未细分；C++ 不消费。 |
| `low[10:9]` | 2 | `[10:9]` | `pad_left` | 仅 CONV 路径读取，2 bit，直接取值。 |
| `low[11]` | 1 | `[11]` | `config1` bit 3 | 子语义未定义，当前没有消费。 |
| `low[13:12]` | 2 | `[13:12]` | `pad_right` | 仅 CONV 路径读取，2 bit，直接取值。 |
| `low[15:14]` | 2 | `[15:14]` | `pad_top` | 仅 CONV 路径读取，2 bit，直接取值。 |
| `low[17:16]` | 2 | `[17:16]` | `pad_bottom` | 仅 CONV 路径读取，2 bit，直接取值。 |
| `low[18]` | 1 | `[18]` | `pad_mode` bit 0 | SystolicArray 读取并记录；当前 timing 计算不使用。 |
| `low[19]` | 1 | `[19]` | `pad_mode` bit 1，同时也是 `weight_shape` bit 0 | 当前代码存在位域重叠，不能视作两个独立字段；SystolicArray 读取并记录 pad_mode，但 timing 计算不使用。 |
| `low[22:20]` | 3 | `[22:20]` | `weight_shape` bits 3:1 | 完整 shape 编码是 `low[22:19]`，实际值为 `((low >> 19) & 0xf) + 1`；bit 19 与 `pad_mode` 重叠，不能仅对本行的 3 bit 做 `+1`。 |
| `low[24:23]` | 2 | `[24:23]` | `weight_stride` | CONV 路径读取，实际 stride 为 `field + 1`。 |
| `low[29:25]` | 5 | `[29:25]` | `weight_dilation` | CONV 路径读取，实际 dilation 为 `field + 1`。 |
| `low[31:30]` | 2 | `[31:30]` | `config1` bits 23:22 | 属于 `config1`，子语义未定义；C++ 不消费，不可作为保留位重新分配。 |
| `low[47:32]` | 16 | `[47:32]` | `input_a_zp` | generator 会写入；当前性能模型不读取。 |
| `low[63:48]` | 16 | `[63:48]` | `input_b_zp` | generator 会写入；当前性能模型不读取。 |
| `high[15:0]` | 16 | `[79:64]` | `output_scale` | generator 会写入；当前性能模型不读取。 |
| `high[31:16]` | 16 | `[95:80]` | `output_shift` | generator 会写入；当前性能模型不读取。 |
| `high[63:32]` | 32 | `[127:96]` | `output_zp` | generator 会写入；当前性能模型不读取。 |

`config1` 为整行 `[31:8]`，共 24 bit。
ResNet generator 的写法是 `config1 << 8`，没有掩码和参数范围校验；当前 GEMM 调用传入 0。
未使用的 `config1` 子位段不赋予额外语义，CONV 已使用子字段的边界以 C++ 提取式为准。
`pad_mode = (low >> 18) & 3` 仅写日志，未参与延迟计算；
`weight_shape = ((low >> 19) & 15) + 1` 实际参与延迟计算。bit 19 的双重读取是现有行为，
这里保留这一事实，不擅自重新分配字段。

#### 第 1 行

| 字内位段 | 位宽 | 整行 bit | 编码字段 | 当前语义 |
|---|---:|---|---|---|
| `low[0]` | 1 | `[0]` | `config2` bit 0 | 当前没有消费。 |
| `low[2:1]` | 2 | `[2:1]` | `optype` | `0=GEMM`，`1=CONV`。`2/3` 进入 SystolicArray 的固定 10 周期 GEMV 占位分支，不启动 Accumulator，当前不能正常完成 SA 队列退休，不能视为可用 ISA。 |
| `low[3]` | 1 | `[3]` | `acc_dest` | `1`：SA 输出写 ACC data；`0`：SA 输出写 SPM。 |
| `low[4]` | 1 | `[4]` | `a_keep` / `a_valid` | SA 退休时将 A bank 的 valid 赋为该 bit：`1` 保留可读，`0` 消耗失效；不表示跳过输入读取。 |
| `low[5]` | 1 | `[5]` | `b_keep` / `b_valid` | SA 退休时将 B bank 的 valid 赋为该 bit：`1` 保留可读，`0` 消耗失效。 |
| `low[6]` | 1 | `[6]` | `is_acc` | `1`：读取 ACC 累加输入；`0`：不读取 ACC 累加输入。 |
| `low[10:7]` | 4 | `[10:7]` | `config2` bits 10:7 | 属于 `config2`，子语义未定义；C++ 不消费。 |
| `low[11]` | 1 | `[11]` | `is_bias` | `is_acc=1` 时，`1` 选择 ACC bias bank，`0` 选择 ACC data bank。 |
| `low[12]` | 1 | `[12]` | `bias_keep` / `bias_valid` | SA 退休时无条件将首个 bias bank 的 valid 赋为该 bit，包括 `is_acc=0` 或 `is_bias=0` 的情况。 |
| `low[15:13]` | 3 | `[15:13]` | 保留/未消费 | 当前没有消费。 |
| `low[23:16]` | 8 | `[23:16]` | `bias_w` | bias 宽度字段；generator 对应 bias_w；C++ 不提取，也不定义是否为减一编码。 |
| `low[31:24]` | 8 | `[31:24]` | `bias_h` | bias 高度字段；generator 对应 bias_h；C++ 不提取，也不定义是否为减一编码。 |
| `low[47:32]` | 16 | `[47:32]` | `bias_stride` | Accumulator 解码并保存，但当前 timing 代码没有用它计算地址。 |
| `low[63:48]` | 16 | `[63:48]` | `output_stride` | Accumulator 用于计算 SA 输出写入地址；单位按 ACC/SPM 元素 stride 解释。 |
| `high[31:0]` | 32 | `[95:64]` | `biaspsum_addr` | `is_acc=1` 时的 ACC data/bias 读取基地址。 |
| `high[63:32]` | 32 | `[127:96]` | `output_addr` | SA 输出基地址；`acc_dest=0` 时为 SPM 地址，`acc_dest=1` 时为 ACC 地址。 |

`a_keep/b_keep/bias_keep` 表示**本条 SA 完成并退休后**相应输入是否保持有效：需要后续复用
则设为 1，最后使用后通常设为 0，使 bank 可以加载新数据。它们不控制本条是否读取输入，
也不表示清零物理数据。当前 bias valid 的无条件赋值是实现行为：未使用 bias 的 SA 也必须
保持已有 bias 的正确状态；不能在没有有效生产者时随意设 1，或将仍待复用的 bias 清为 0。

#### 第 2 行

| 字内位段 | 位宽 | 整行 bit | 编码字段 | 当前语义 |
|---|---:|---|---|---|
| `low[31:0]` | 32 | `[31:0]` | `a_addr` | A/activation/IFM 的 SPM 基地址。 |
| `low[42:32]` | 11 | `[42:32]` | `a_col_num_m1` | 11 bit，实际 A 列数为 `field + 1`；GEMM 中是 K，CONV 中是 IFM width。 |
| `low[47:43]` | 5 | `[47:43]` | `a_row_num_m1` | 5 bit，实际 A 行数为 `field + 1`；GEMM 中是 M，CONV 中是 IFM height。 |
| `low[63:48]` | 16 | `[63:48]` | `a_stride` | A/activation SPM 读请求之间的字节增量；当前模型不额外乘元素字节数。 |
| `high[31:0]` | 32 | `[95:64]` | `b_addr` | B/weight 的 SPM 基地址。 |
| `high[36:32]` | 5 | `[100:96]` | `b_col_num_m1` | 5 bit，实际 B 列数为 `field + 1`；GEMM 中是 N，CONV 中是 output channels。 |
| `high[47:37]` | 11 | `[111:101]` | `b_row_num_m1` | 11 bit，实际 B 行数为 `field + 1`；GEMM 中必须等于 A 的 K，CONV 中是 input channels。 |
| `high[63:48]` | 16 | `[127:112]` | `b_stride` | B/weight SPM 读请求之间的字节增量；当前模型不额外乘元素字节数。 |

### 3.5 MATADD：ACC 双输入到 SPM

MATADD 固定为 3 行。当前性能模型只使用第 0 行的矩阵地址/尺寸；第 1 行的量化
字段虽然由 ResNet generator 写出，但没有参与 timing 或地址计算；第 2 行全为保留位。

| 字内位段 | 位宽 | 整行 bit | 编码字段 | 当前语义 |
|---|---:|---|---|---|
| `inst0.low[7:0]` | 8 | `[7:0]` | `opcode` | 固定为 `0x04`。 |
| `inst0.low[15:8]` | 8 | `[15:8]` | `col_num_m1` | 8 bit，实际列数为 `field + 1`。 |
| `inst0.low[23:16]` | 8 | `[23:16]` | `row_num_m1` | 8 bit，实际行数为 `field + 1`。 |
| `inst0.low[31:24]` | 8 | `[31:24]` | 保留/未消费 | 当前没有消费。 |
| `inst0.low[63:32]` | 32 | `[63:32]` | `a_addr` | ACC data 输入 A 基地址。 |
| `inst0.high[31:0]` | 32 | `[95:64]` | `b_addr` | ACC data 输入 B 基地址。 |
| `inst0.high[63:32]` | 32 | `[127:96]` | `out_addr` | SPM 输出基地址。 |
| `inst1.low[15:0]` | 16 | `[15:0]` | `output_scale` | generator 会写入；共用 Accumulator 前导代码会按 SA config 读取其中若干 bit，但不影响 MATADD 的读写地址、周期或 bank 更新。 |
| `inst1.low[31:16]` | 16 | `[31:16]` | `output_shift` | generator 会写入；不影响 MATADD 性能行为。 |
| `inst1.low[63:32]` | 32 | `[63:32]` | `output_zp` | 共用代码按 SA stride 解码并保存；不影响 MATADD 性能行为。 |
| `inst1.high[63:0]` | 64 | `[127:64]` | 保留/未消费 | 共用代码按 SA 地址解码并保存；MATADD 分支不用这些值，generator 为 0。 |
| `inst2.low[63:0]` | 64 | `[63:0]` | 保留/未消费 | MATADD 分支不用；generator 全部写 0。 |
| `inst2.high[63:0]` | 64 | `[127:64]` | 保留/未消费 | 共用代码提取其中的 B 列数，但 MATADD 分支不用；generator 全部写 0。 |

### 3.6 SFU：固定功能向量操作

SFU 固定为 2 行。第 1 行当前 generator 写 0，C++ 不校验它是否为零；当前 C++ 没有读取显式 stride 或额外地址字段，
SFU 的行 stride 和 vector 访问模式由 C++ 根据 shape、`systolic_array_size` 和 optype
推导。第 1 行定义输入/输出量化字段，当前 C++ 不使用这些字段，详见下表。

#### 第 0 行

| 字内位段 | 位宽 | 整行 bit | 编码字段 | 当前语义 |
|---|---:|---|---|---|
| `low[7:0]` | 8 | `[7:0]` | `opcode` | 固定为 `0x05`。 |
| `low[13:8]` | 6 | `[13:8]` | `optype` | 6 bit：`0=softmax`，`1=GELU`，`2=layernorm`，`6=pool_max`，`8=transpose`。其他值当前没有有效 timing path。 |
| `low[15:14]` | 2 | `[15:14]` | `config` bits 7:6 | 属于 `config`，子语义未定义；C++ 不消费。 |
| `low[16]` | 1 | `[16]` | `is_quant` | 量化路径标志；当前 SFU 主要用它增加 modeled cycle，不改变真实数据格式。 |
| `low[18:17]` | 2 | `[18:17]` | `config` bits 10:9 | 属于 `config`，子语义未定义；C++ 不消费。 |
| `low[31:19]` | 13 | `[31:19]` | 保留/未消费 | 保留区域；C++ 不消费。 |
| `low[47:32]` | 16 | `[47:32]` | `col_num_m1` | 16 bit，实际列数为 `field + 1`。 |
| `low[63:48]` | 16 | `[63:48]` | `row_num_m1` | 16 bit，实际行数为 `field + 1`。 |
| `high[31:0]` | 32 | `[95:64]` | `input_addr` | SPM 输入基地址。 |
| `high[63:32]` | 32 | `[127:96]` | `output_addr` | SPM 输出基地址。 |

#### 第 1 行

| 字内位段 | 位宽 | 整行 bit | 编码字段 | 当前语义 |
|---|---:|---|---|---|
| `low[31:0]` | 32 | `[31:0]` | `input_zeropoint` | 已定义字段；C++ 不读取，generator 写 0。 |
| `low[47:32]` | 16 | `[47:32]` | `input_scale` | 已定义字段；C++ 不读取。 |
| `low[63:48]` | 16 | `[63:48]` | `input_scaleshift` | 已定义字段；C++ 不读取。 |
| `high[15:0]` | 16 | `[79:64]` | `output_zeropoint` | 输出 zeropoint 为 16 bit，输入 zeropoint 为 32 bit；C++ 不读取。 |
| `high[31:16]` | 16 | `[95:80]` | `output_scale` | 已定义字段；C++ 不读取。 |
| `high[47:32]` | 16 | `[111:96]` | `output_scaleshift` | 已定义字段；C++ 不读取。 |
| `high[63:48]` | 16 | `[127:112]` | 保留/未消费 | 保留区域；C++ 不读取。 |

SFU `config` 为整行 `[18:8]` 共 11 bit；其中 optype 和 is_quant 的位置由
当前 C++ 确认，其余 config bit 未定义子语义，不能自行赋予其他算子或量化模式。

当前 optype 的性能路径是：softmax、GELU、layernorm、pool_max、transpose。SFU 的
`3/4/5/7` 等值尚未在当前 simulator 中定义，不能直接用于 `@pl.kernel`；后续扩展必须
同时定义输入输出布局、stride、字节模型和 latency 模型。

### 3.7 指令组、bank 状态和软件编码约束

位域之外，指令是否能发射还取决于 EventSyncUnit 的隐含协议：

- MVIN 使目标 SPM/ACC bank busy；MVOUT 要求源 bank valid，并在完成后清除 valid。
- SA 要求 A/B 输入 bank valid；`is_acc/is_bias` 决定额外的 ACC data/bias bank；`acc_dest` 决定输出写 ACC 还是 SPM。
- MATADD 的两个 ACC 输入必须位于不同 ACC bank，输出必须是空闲 SPM bank。
- SFU 要求输入 SPM bank valid、输出 SPM bank 无 valid；完成后输入失效、输出有效。
- 指令组的行数是固定的，compiler 必须原子生成完整的 2 行或 3 行，不能在组内插入其他指令。
- 当前 ISR 位不是指令字段：MVIN bit 0、MVOUT bit 1、SA bit 2、MATADD bit 3、SFU bit 5。它们是完成事件的粘滞位，不是每条指令的完成计数器；core 会根据 MER/IER/IPR 自动写 IAR 确认中断，runtime 不能仅依赖一次轮询推断所有指令已完成。

因此，`sw/pimu/isa.py` 的第一版应严格实现上述已确认字段；新指令将无命名的保留位清零，
有命名但不影响性能的字段可以保留 benchmark 值以支持原样 round-trip。SA CONV 的重叠字段
应显式标记为 experimental，不应由 compiler 猜测新的编码。以上是软件栈的拟定策略，
不是声称当前 C++ 已经实现这些校验。

### 3.8 字段范围、地址换算和性能含义

#### 可编码范围

下表是字段容量，不是任意组合都能运行的承诺；实际还受 SRAM 容量、端口、bank、
整数运算范围和 timing 模型约束。所有参与地址、shape、stride 解码的字段均按无符号值提取。

| 字段 | 编码位宽 | 编码值 | 解码值/单位 |
|---|---:|---|---|
| 任意地址 | 32 | `0..0xffffffff` | byte offset，实际必须位于所选空间范围内 |
| DMA/SFU 行数、列数 | 16 | `0..65535` | `1..65536` 个元素，无零维编码 |
| DMA 两种 stride | 16 | `0..65535` | 元素数；零 stride 会重复同一行地址，不表示自动连续布局 |
| SA A 列数 / B 行数 | 11 | `0..2047` | `1..2048`；GEMM 要求两者相等 |
| SA A 行数 / B 列数 | 5 | `0..31` | `1..32`，不会自动随配置中的阵列尺寸扩大位宽 |
| SA 输入 stride | 16 | `0..65535` | 字节；0 表示重复读同一地址 |
| SA output stride | 16 | `0..65535` | 由下面的 GEMM/CONV 输出公式换算，0 表示重复写同一地址 |
| MATADD 行数、列数 | 8 | `0..255` | `1..256` 个元素，无独立 stride 字段 |
| CONV 四侧 padding | 各 2 | `0..3` | 直接值，没有减一编码 |
| CONV kernel shape | 4 | `0..15` | `1..16`，单个值同时表示 kernel 宽和高 |
| CONV weight stride | 2 | `0..3` | `1..4` |
| CONV dilation | 5 | `0..31` | `1..32` |
| SFU optype | 6 | `0..63` | 当前仅 0、1、2、6、8 有分支，其他值抛出异常 |

量化 scale/shift/zp 的位宽见第 3.3–3.6 节字段表；其数值解释没有在当前性能模型
中定义。DMA 的 `qzp` 是 32 bit，但 generator 的 `high1=qzp` 没有掩码，
不能把高 32 bit 当成已经验证的 signed extension 协议。现有 generator 大多不做范围检查；
超宽或负整数参数可能污染相邻字段，软件栈 encoder 应先检查再编码。

#### 当前地址递推

设 `P=cfg.systolic_array_size`、`S=cfg.sram_data_width/8`、
`A=cfg.acc_data_width/8`，均不是指令 bit。

| 路径 | 当前 C++ 使用的地址公式 |
|---|---|
| DMA DRAM 第 r 行 | `dram_base + r * dram_stride * dram_elem_bytes` |
| DMA local 第 r 行 | `local_base + r * sram_stride * local_elem_bytes` |
| DMA 行内 local 偏移 | `dram_payload_offset * local_elem_bytes / dram_elem_bytes`，按 2 的幂移位换算 |
| SA A/B 第 t 次读 | `a_addr + t*a_stride` / `b_addr + t*b_stride`；每次 SPM 访问宽度为 `P*S` 字节 |
| SA 累加输入第 t 次读 | `biaspsum_addr + t*P*A`；**不使用 bias_stride**，每次宽度为 `P*A` |
| GEMM 输出第 t 次写 | `output_addr + t*output_stride*E`，`E=A`（ACC）或 `S`（SPM） |
| CONV 输出第 t 次写 | `output_addr + t*output_stride*cout*E`，比 GEMM 多乘 `cout` |
| MATADD 第 t 次读/写 | 两个 ACC 输入各加 `t*P*A`，SPM 输出加 `t*P*S` |
| SFU softmax/layernorm/pool_max | 行地址加 `r*cols*S`，行内第 b 次访问再加 `b*P*S` |
| SFU GELU | 输入/输出第 g 组基地址分别加 `g*4*S`，每次 SPM 访问仍计 `P*S` 字节 |
| SFU transpose | 输入 row stride 为 `cols*S`；输出 row stride 为 `rows*S`，以 `P×P` tile 读写 |

SPM 每次访问检查完整 `P*S` 字节是否位于同一 bank、该 bank 本周期是否空闲；
ACC data/bias 每次访问检查完整 `P*A` 字节和对应端口。当前实现没有统一的“地址必须
按向量宽度对齐”校验，不能在规范中虚构这一 ISA 约束。compiler 可以选择对齐布局，
但必须为尾块、transpose padding 和 SA 流水线额外读请求预留可访问空间。

额外访问须按实际 `tick()` 条件评估：GEMM 的 A stream 发出 `M+K+N+2` 次向量读，
B stream 发出 `K+N+2` 次；CONV 的 A stream 次数为 `6+K_timing+cout+1`，
B stream 次数为 `K_timing`。Accumulator 的 SA 累加读取条件包含边界，若有效输出行数
为 M，则读取 M+1 次；MATADD 设 `G=ceil(R*C/P)`，当前读两个输入各 G+2 次、写输出
G+1 次。这些是当前循环的实际请求数，说明紧贴逻辑矩阵大小分配存储可能越界，不能把
它们改写成理想化的矩阵 payload。CONV 的派生量定义见下面的周期公式。

#### 与指令字段直接相关的周期公式

这些是功能单元 `execute()` 初始化的周期计数，不能直接相加当作程序总周期；
总周期还包括 fetch/decode/排队、依赖等待、单元重叠和 DDR 响应。以下 `q` 为 is_quant，
`R/C` 为解码后的行/列数。

| 指令/路径 | 当前局部周期计数 |
|---|---|
| SA GEMM | `M+K+N+2`；启动 Accumulator 的倒计数为 `K+N+1` |
| Accumulator GEMM | `M+tail`，ACC 输出 `tail=3`，SPM 输出 `tail=4` |
| MATADD | `ceil(R*C/P)+5` |
| SFU softmax | `1 + R*(2*ceil(C/P)+15) + q` |
| SFU GELU | `4 + 60*ceil(R*C/4) + q`；当前固定使用最慢 GELU 路径 |
| SFU layernorm | `1 + R*(2*ceil(C/P)+28) + q` |
| SFU pool_max | `3 + R*ceil(C/P) + q`；没有 kernel/stride/pooling 输出 shape 字段 |
| SFU transpose | `(P+2)*(ceil(R/P)*ceil(C/P)+1) + q` |

CONV 的精确派生式为：

```text
k_eff = dilation * (kernel_shape - 1) + 1
ofm_w = (ifm_w + pad_left + pad_right - k_eff) / weight_stride + 1
ofm_h = (ifm_h + pad_top + pad_bottom - k_eff) / weight_stride + 1
M_timing = (ofm_w * ofm_h) & 0x3f
K_timing = kernel_shape * kernel_shape * P
SA cycles = 6 + M_timing + cout + K_timing + 2
Accumulator start countdown = 6 + K_timing + cout + 1
Accumulator cycles = M_timing + tail
```

除法是当前 C++ 整数除法，负值向零截断；代码随后要求 ofm_w/ofm_h 大于零。`cin` 被解码
并记录，但当前 `K_timing` 固定乘 `P`，没有乘实际 cin；输出空间大小按 6 bit 截断，
64 的倍数会变为 0。这些是现有模型行为，不能在软件栈中悄悄替换为一般卷积公式。

DMA 没有仅由指令字段决定的固定周期公式：DRAM 每行 payload 为
`cols*dram_elem_bytes`，根据 AXI beat 对齐、4 KiB 边界和最大 256 beats 拆 burst，
等待 local transfer 和对应 DDR response 都完成才退休。统计中的 DRAM bytes 按整 beat
计算，可能大于逻辑 tensor payload；`qscale/qshift/qzp` 不参与这个过程。

### 3.9 EventSyncUnit 的精确行为与已知限制

令 `V[b]` 为 valid，`B[b]` 为 busy，`n=cfg.sram_bank_num`、
`a=cfg.acc_bank_num`。SPM bank ID 为 `addr / sram_bank_bytes`；ACC data bank ID
为 `n + (addr*a)/acc_bytes`（整数除法）；bias scoreboard 使用固定位置 `n+a`。
valid/busy 初始全 0。当前 SA 实现显式处理 ACC data 的两个位置 `n` 和 `n+1`，
**不是根据 biaspsum_addr 只选择一个 bank，也没有泛化到任意 acc_bank_num**。

| 操作 | 发射条件/动作 | 退休动作 |
|---|---|---|
| MVIN | 目标 bank `B=0,V=0`，发射后置 `B=1`；bias 目的地固定用 `n+a` | 目标 `B=0,V=1` |
| MVOUT | 源 bank `B=0,V=1`，发射后置 `B=1` | 源 `B=0,V=0` |
| SA 输入 | A/B bank 均 `B=0,V=1`；发射时不将输入置 busy | 依次设置 `V[A]=a_keep`、`V[B]=b_keep`；若同 bank，后者覆盖前者 |
| SA 累加输入 | `is_acc=0` 不检查；`is_bias=1` 检查 `n+a` 的 `B=0,V=1`；否则要求 `n`、`n+1` 都 `B=0,V=1` | 受输出路径和 bias_keep 更新影响，见下面两行 |
| SA 输出 ACC | allowIssue 不单独检查输出 ACC 是否空闲；issueUpdate 对输出地址做 ACC 范围检查并置 `B[n]=B[n+1]=1` | `B[n]=B[n+1]=0`，`V[n]=V[n+1]=1` |
| SA 输出 SPM | 输出 bank `B=0,V=0`，发射置输出 `B=1` | 输出 `B=0,V=1`，同时清 `V[n]`、`V[n+1]` |
| SA bias 状态 | 仅在 is_acc/is_bias 路径检查可读，不设置输入 busy | 无条件设置 `V[n+a]=bias_keep`，与本条是否实际读 bias 无关 |
| MATADD | A/B 在不同 ACC bank 且都 `B=0,V=1`，SPM 输出 `B=0,V=0`；只置输出 `B=1` | 输出 `B=0,V=1`，两个输入 `V=0` |
| SFU | 输入 `B=0,V=1`、输出 `B=0,V=0`；只置输出 `B=1` | 输出 `B=0,V=1`，输入 `V=0` |

同一 SA 的上述退休更新顺序为：先处理输出/两个 ACC data 状态，再设置 bias valid，最后
设置 A valid 和 B valid。SPM 输出与输入同 bank 无法同时满足发射条件；SFU 也不能原地
覆盖输入。MATADD 的“两个 ACC bank 不同”是 scoreboard 要求，底层 ACC 实际用两个
data port 建模，并不在 `accessDataPort()` 中按 bank 自动仲裁。

scoreboard 只按基地址标记 bank，没有按 shape/stride 标记覆盖到的所有 bank，也没有将
SA/SFU/MATADD 输入全部锁 busy。因此 encoder 的字段合法不等于程序无资源冲突；
compiler 应对完整访问区间和端口时序检查。每个队列内部保持 FIFO，但不同队列可并发；
core 发射顺序为 SA、SFU、DMA1、DMA2、DMA3，不能将文本顺序当作全局执行屏障。

DMA 通道不是额外指令字段：`source=1` 固定走 DMA1；`source=0` 的 MVIN 在 DMA2/3
轮询；`source=0` 的 MVOUT 优先使用完全空闲的 DMA1，否则在 DMA2/3 轮询。

### 3.10 可复核的十六进制编码示例

以下各行都为 `high || low` 的 32 个十六进制字符。示例说明编码，不是可以任意独立运行
的完整程序，执行前仍须满足 3.9 的 bank valid/busy 条件。

**MVIN int8：** DRAM `0x01000000` 到 SPM `0x00000000`，32×32，
local stride 32 元素，DRAM stride 256 元素，data_type=0、precision=1、q=0。
下面两行对应现有 ResNet benchmark 的第一条 MVIN：

```text
0100000000000000001f001f00000401
00000000000000000000000001000020
```

**MVOUT int32：** ACC `0x00080000` 到 DRAM `0x03000000`，32×32，
两侧 stride 都为 32 元素，`data_type=1, precision=1, source=1`。

```text
0300000000080000001f001f00002502
00000000000000000000000000200020
```

**SA GEMM：** A 为 SPM `0x0`，B 为 SPM `0x00080000`，均 32×32，
输入 stride 均 32 字节；首个 K tile 不累加，结果写 ACC `0x00080000`，
`config2=0x8`，output/bias stride 均 32，output_scale=1。
这三行对应现有 ResNet benchmark 的第一条 SA：

```text
00000000000000010000000000000003
00080000000000000020002000000008
002003ff000800000020f81f00000000
```

**MATADD：** A/B 分别位于 ACC `0x0`/`0x00080000`，输出 SPM `0x00180000`，
32×32，沿用 ResNet 的 output_scale=1。

```text
001800000008000000000000001f1f04
00000000000000000000000000000001
00000000000000000000000000000000
```

**SFU softmax：** 输入 SPM `0x00100000`，输出 `0x00140000`，1×32，q=0。

```text
00140000001000000000001f00000005
00000000000000000000000000000000
```

一个需要避免的现有 helper 陷阱：`gen_smolvlm2_cases.py::emit_mvout(from_acc=True)`
写入 `data_type=2`。当前 MVOUT 只取 data_type 的 bit 0，因此该组合实际上按 1-byte
int8 建模，**不会**因为 source=ACC 自动变为 4-byte int32；要编码 int32 应按 3.3 的
表设置 `data_type[0]=1`。兼容模式保留原始编码；验证模式按实际 dtype 显式选择字段。

### 3.11 源码依据和软件栈的实现边界

下列路径均相对本文档所在的 `sw/` 目录：

| 内容 | 当前实现依据 |
|---|---|
| high/low 定义与文本 parser | [inst_cache.hpp](../include/perf/inst_cache.hpp) 的 `Inst`；[perf_npu_core.cpp](../src/perf/perf_npu_core.cpp) 的 `parse_inst()` |
| opcode、行数、DMA 路由 | [decoder.hpp](../include/perf/decoder.hpp) 的 `getOpcode/getInstLineCount`；[decoder.cpp](../src/perf/decoder.cpp) 的 `selectDMATarget` |
| DMA 字段、类型、burst/地址换算 | [dma.cpp](../src/perf/func_units/dma.cpp) 的 `decode_mvin_sizes/decode_mvout_sizes/execute/tick` |
| SA shape、stride、CONV 重叠位与 GEMV 占位 | [systolic_array.cpp](../src/perf/func_units/systolic_array.cpp) 的 `execute/tick` |
| SA 累加、输出地址、MATADD | [accumulator.cpp](../src/perf/func_units/accumulator.cpp) 的 `execute(uint8_t)/tick` |
| SFU bit 字段、算子周期和访存 | [sfu.cpp](../src/perf/func_units/sfu.cpp) 的 `execute/tick` |
| valid/busy 协议 | [event_sync_unit.cpp](../src/perf/event_sync_unit.cpp) 的 `allowIssue/issueUpdate/retireUpdate` |
| byte 地址范围和端口检查 | [scratchpad.cpp](../src/perf/scratchpad.cpp) 的 `access`；[acc_sram.cpp](../src/perf/acc_sram.cpp) 的 `accessDataPort/accessBiasPort` |
| 指令队列退休/ISR 位 | [inst_queue.cpp](../src/perf/inst_queue.cpp) 的各队列 `inst_retire` |
| generator 字段名与基线 hex | [gen_resnet_block.py](../benchmarks/tb_test/gen_resnet_block.py)、[gen_smolvlm2_cases.py](../benchmarks/tb_test/gen_smolvlm2_cases.py)、[tb_resnet_block.txt](../benchmarks/tb_test/tb_resnet_block.txt) |

文本 parser 支持空行/`#` 注释、`0x`/`0X` 前缀、`128'h` 风格前缀和下划线，短行左补
零，超过 32 个 hex 字符报错。它未通过 `stoull` 的结束位置检查每个字符；软件栈应输出
严格的 32 位 hex 行，并验证指令组完整性，不能依赖当前 parser 发现所有非法输入。

基线 encoder/disassembler 应覆盖 5 个 opcode 的全部 128-bit 行并保留原始字段；
完整 Qwen3 目标还需实现第 10 节冻结后的扩展格式。
verifier 则另行检查：字段范围、减一编码、DMA dtype 组合、实际访问跨度、bank/端口、
保留位策略及不支持的 SA/SFU optype。量化定点语义、未细分 config bit、CONV pad_mode 的完整含义、
GEMV 的完整执行协议和任何新 opcode 仍需单独定义；这些缺口不妨碍先实现已经明确的
performance workload 编译，但不能以文档推测替代模型实现。

## 4. PIMU-Sim 与 openTPU 的概念映射

| openTPU | PIMU-Sim 对应 | 实现决策 |
|---|---|---|
| `@ol.jit` kernel | `@pl.kernel` 静态 kernel/trace | 先静态展开，禁止隐式调度 |
| `Tensor` / `Affine` | `TensorDesc` / `BufferDesc` | 保留 shape、dtype、stride、空间和地址 |
| `QTensor` | `WeightDesc` | 第一版 int8 per-tile/per-row；量化元数据单独布局 |
| `KVDesc` | `KVCacheDesc` | 按模型层/请求/位置规划 K/V，首版可用转置 K / sequence-major V，容量由请求决定 |
| `ol.load` | MVIN DRAM -> SPM/ACC | compiler 分配 bank 和二维 stride |
| `ol.store` | MVOUT SPM/ACC -> DRAM | 输出写回的地址和字节数由 memory plan 记录 |
| `ol.dot` / MM | SA + 可选 ACC 累加 | K 维 tile 展开；首块/中间块/末块配置不同 |
| `ol.quantize` | MVIN 转换字段/量化 metadata 及扩展转换内核 | 不建设数值量化路径；模型必需的转换性能按第 8.4/10 节实现，数据相关动态量化另设契约 |
| VOP/outer/reduce | 既有 SFU + 第 10 节扩展内核 | 为 Qwen3 补齐 Norm/RoPE/SwiGLU/mask，不复制 openTPU 指令编码 |
| `all_gather` / `all_reduce` | 无直接硬件对应 | 单 core 先禁止；多 core 时增加 collective engine |
| `WAITW` mailbox | 无直接对应 | 第一版 host polling；后续增加 mailbox/WAIT opcode |
| `BAR` | EventSyncUnit + 静态依赖 | 不把 bank scoreboard 当作通用程序屏障 |
| `LOOP` / `HALT CHAIN` | 无直接对应 | 第一版 host 多次 launch；后续扩展控制流 |
| ISA simulator / RTL / board | PIMU performance backend / future RTL backend | 先统一 performance simulator contract |

## 5. 推荐的软件栈目录

软件栈代码统一放在 `PIMU-Sim/sw/`，C++ 仿真器的现有 `include/` 和 `src/` 保持原位置：

```text
PIMU-Sim/
├── sw/
│   ├── pimu/                     # Python package
│   │   ├── isa.py                # 128-bit PIMU instruction AST/encode/decode
│   │   ├── target.py             # Parallel PIMU dev profile + capability/config fingerprint
│   │   ├── language/             # pimu.language as pl, @pl.kernel, static IR builders
│   │   ├── formats.py            # dtype/quantization/serialization
│   │   ├── precision.py          # checkpoint dtype -> modeled storage/compute precision
│   │   ├── memory.py             # DRAM/SPM/ACC layout and allocators
│   │   ├── compiler/
│   │   │   ├── ir.py             # TensorDesc, BufferDesc, Op, Program
│   │   │   ├── graph.py          # model-level semantic graph and shape inference
│   │   │   ├── lowering.py       # generic operators -> executable kernels
│   │   │   ├── kernels.py        # primitive lowering and kernel decorators
│   │   │   ├── tiling.py         # SA/ACC/SFU tile plans
│   │   │   ├── scheduler.py      # dependency + bank-aware static scheduler
│   │   │   ├── verifier.py       # field/range/bank/dependency checks
│   │   │   ├── progress.py       # bounded FIFO/dispatch/scoreboard progress verification
│   │   │   └── emit.py           # text/JSON/manifest output
│   │   ├── runtime.py            # Device, Buffer, Weight, KVCache, launch
│   │   ├── backend/
│   │   │   ├── sim.py            # persistent C ABI PerformanceBackend
│   │   │   ├── cli.py            # file/subprocess compatibility and offline backend
│   │   │   └── perf.py           # performance stats/timeline API
│   │   ├── kernels/
│   │   │   ├── gemm.py
│   │   │   ├── matadd.py
│   │   │   ├── norm.py
│   │   │   ├── attention.py
│   │   │   ├── mlp.py
│   │   │   ├── activation.py
│   │   │   ├── rope.py
│   │   │   ├── embedding.py
│   │   │   └── kv_cache.py
│   │   ├── models/
│   │   │   ├── base.py           # ModelAdapter, ModelSpec, RequestSpec
│   │   │   ├── registry.py       # model_type -> adapter
│   │   │   ├── qwen3.py          # first complete model adapter
│   │   │   └── hf.py             # config/checkpoint metadata importer
│   │   └── cli.py                # pimu-compile / pimu-run / pimu-profile
│   ├── tests/                    # ISA, compiler, scheduling, performance tests
│   ├── configs/                  # device profiles, Qwen3 requests and precision policies
│   ├── baselines/                # source/config/program/power fingerprints + measured goldens
│   ├── runs/                     # isolated per-launch config/program/report artifacts
│   ├── native/                   # C ABI boundary and Python bindings for PerfSession
│   └── examples/                 # qwen3_prefill.py, qwen3_decode.py and small kernels
└── ...
```

目录名可以调整，但层次必须保留：ISA、IR/compiler、memory placement、runtime/backend、kernel library、model adapter 不应混成一个脚本。
C++ 核心库、PerfSession、Program/指令源及周期推进逻辑在现有 `include/`、`src/` 中重构；
`src/main.cpp` 作为 CLI 薄入口。构建规则分别生成仿真器库和 CLI，供 `sw/native/` 绑定复用。
重构后的核心库和 CLI 仅构建 AnalyticalDDR 路径；按源文件清单选择依赖，避免全量扫描
源文件引入非目标 DDR 实现及其外部库。

## 6. Parallel PIMU backend 接入契约

### 6.1 统一输入 manifest

由 `sw` Runtime 管理 manifest，解析为配置、Program 和 memory plan，经 Backend 提交给
C++ 会话；同一 artifact 也可导出 config/hex 供 CLI 离线执行。不要求 C++ 核心直接解析整个
模型 manifest。建议的软件包格式如下（新增字段是拟定契约）：

```json
{
  "version": 1,
  "backend": "performance",
  "target_profile": "parallel_pimu_qwen3_v1",
  "simulator_revision": "<backend-commit-with-required-kernels>",
  "model_id": "Qwen/Qwen3-0.6B",
  "model_revision": "<immutable-model-revision>",
  "adapter": "qwen3",
  "request": {"batch_size": 1, "prompt_length": 128, "decode_steps": 16, "logits_positions": "last"},
  "precision": {"weights": "int8", "activations": "int8", "accumulator": "int32", "kv_cache": "int8", "aux": "fp16"},
  "config": "run/simulator.cfg",
  "program": "run/program.hex",
  "memory_plan": "run/memory.json",
  "model": "run/model.json",
  "coverage": "run/coverage.json",
  "kernel_models": "run/kernel_models.json",
  "trace": "run/trace.jsonl",
  "stats": "run/stats.json"
}
```

`target_profile` 名称是拟定标识，不是现有 C++ 配置项。profile 包含 3 DMA/5 队列、
同质 bank SPM、SA/MATADD 共享队列、实际 ISA 位宽及未实现能力；另记录完整有效配置、
源码 dirty 状态、二进制/指令/功耗表哈希和编译策略。不能只用 branch 名作缓存键。

manifest 由模型编译生成，不要求用户先提供 program.hex。完整模型 artifact 可包含 Prefill
程序、Decode 模板与必要的分段程序清单；`program` 是示意入口，不能把一小段 kernel 冒充
整个模型。模型导入、编译、执行和各步性能统计都要能够追踪到同一请求与覆盖报告。

主线由 PerformanceBackend 在同一 PerfSession 中提交阶段程序；`program` 路径是可复现
产物的位置，不要求仿真器每次都从磁盘取指。两种入口都将产物写入独立的
`sw/runs/<run_id>/`，不覆盖原 config、benchmark 或 log，并保留
word offset→完整指令→算子/tile 的 metadata 映射。
CLI 继续支持 `./build/npu_sim <generated_config>`，由 `sw` 兼容适配器设置仓库根目录为
工作目录；它用于迁移验证、离线执行和文件回归，不能替代持久会话的模型级验收。

### 6.2 Performance backend contract

C++ backend 保留当前周期推进和并行行为。接入层必须落实以下契约：

- `sw` 先检查指令组完整性、profile 和字段范围；C++ 仍按当前前端、DMA、SA/Accumulator、SFU、SPM、ACC、DDR 推进，不调用 GEMV 占位路径；
- 保持 core 内顺序：清端口→tick 现有功能单元→发送 DDR 请求→采样→发射队列→decoder→每周期载入一行→退休；不能把退休提前或把输入一次性塞满 cache；
- 多频率用 `PerfSim` 和 `compute_base_tick_and_multipliers()` 的实际时钟推进；当前有整数 ps 舍入和最小 10 ps base tick，profile 记录名义频率与实际 tick，cost model 不假定完美实数频率比；
- C ABI 与 CLI 共用结构化状态/统计，timeline 增加稳定的 `session_id/launch_id/program_id/instruction_index/opcode/issue/finish/retire` 与算子/tile 映射。当前 inst_id 按队列局部编号，不能直接当全局 ID；
- 输出总周期、各模块 busy cycle、DDR/SPM/ACC 字节数、峰值/平均带宽、队列 occupancy、bank conflict 和功耗；
- 明确区分 `DONE/TIMEOUT/INVALID/RESOURCE_CONFLICT/BACKEND_ERROR`；完成状态来自完整排空检查，进程退出码为 0 或打印统计不等于成功完成。迁移期间旧 CLI 只能用 `Stats.isDone` 或可靠完成日志判定；未取得证据时标记 UNKNOWN，不发布成功结果。

Python 产出统一字段 schema，用现有 C++ 解码建立差分验证；可以后续生成 C++ 常量，
不要求先重构整个 decoder。backend 不保存真实 tensor 数值，也不负责 logits 或模型输出。
当前 C++ 不在所有配置下向 stdout 输出 isDone；迁移验证保存包含 Final state 的日志，
重构后的主接口直接返回结构化终态与统计，关闭逐周期日志仍须能可靠判断完成。

新增 JSON/trace 等观测接口应保证原程序周期不变。`request_id` 响应匹配、输入 bank 锁和
冲突行为必须在 Qwen3 多 DMA 集成前审计，必要修复独立记录并重建 golden。新增 Qwen3
kernel/指令能力应有明确版本；保留底层兼容回归，不以不修改旧二进制为由省略模型算子。
持久会话是完整模型主线的前置能力；独立进程仍可提供冷启动参考，不能据此宣称已经验证
同会话连续执行、片上数据复用或设备侧循环。

### 6.3 仿真器重构范围与控制流程

将当前“创建即打开 txt、run 一次、进程退出”的结构重构为以下三层：

| 层次 | 修改要求 |
|---|---|
| C++ 仿真器核心库 | 提供 PerfSession、Program、指令源和周期推进；Core、队列、功能单元、EventSync、SPM/ACC、AnalyticalDDR 由核心库模拟。会话固定创建 AnalyticalDDR，构造时不要求存在指令文件。 |
| Python Runtime/Backend | 模型适配和编译后，以完整阶段提交程序，管理请求/KV descriptor、地址绑定、程序缓存和统计回收；C ABI 及 Python 绑定放在 `sw/native/`。 |
| 独立 CLI | `src/main.cpp` 解析参数后调用同一核心库；文件输入、离线执行与旧配置兼容复用同一加载和运行实现。 |

主线控制流程为：

```text
model adapter -> compiler -> prefill/chunk + decode programs
                                 |
Python Runtime -> PerformanceBackend -> C ABI -> PerfSession
                                                 |
       create -> load memory plan -> load program -> run -> status/stats
                                        ^                    |
                                        +-- next stage ------+
                                  request ends -> destroy
```

Python 不逐个算子驱动功能单元，也不指定指令的发射时刻。Prefill 可按 chunk 切分，Decode
每步包含完整模型 forward；核内部的流水线、有限队列、多 DMA 和 bank 同步继续决定时序。

重构按六项交付：核心库与 CLI 分离、文件/内存指令源解耦、持久会话与状态规则、统一 Backend
接口、Runtime 阶段控制、结构化统计及回归。C++ 改动覆盖必要的生命周期与时钟推进重构，
不限定为少量观测 hooks；ISA/算子性能扩展与接口重构分别记录和验收。

### 6.4 Program 与指令加载机制

`Program` 包含有序 128-bit word/完整包信息、ISA/profile 版本、program fingerprint 和可选
调试映射。`Inst` 的 high/low 顺序及字节序由接口明确规定；不能依赖编译器结构体布局。
文件解析和内存输入都转换到统一指令源，包完整性、字段范围及 profile 校验规则一致。

`load_program()` 仅准备程序和取指位置；执行时仍由前端按第 6.2 节规则逐 word 送入
InstCache，不能因内存输入而一次性塞满缓存、绕开容量或缩短取指周期。长程序支持分块
持有或映射存储，host 读文件/分配内存耗时与模拟的前端周期分开记录。

初版位置、KV 长度和地址参数由 Python 编译器绑定/重定位后生成实际指令；模板复用不
代表现有 ISA 已经具有 openTPU 的运行时参数寄存器。每个已绑定程序保留独立指纹和来源。

每段 Program 另带入口/出口状态契约：读取的 DRAM 已发布范围、预期 SPM/ACC bank 数据
代次与布局、keep 租约、写回范围和 KV 更新。Runtime/Backend 在加载前校验前一段 DONE
后的实际状态是否满足入口；不满足时重新规划搬运或拒绝，不能靠 metadata 把 bank 标成
有效。原始 hex 的首次兼容执行从设备初始态开始；若要用于跨阶段执行，必须补齐状态契约。

### 6.5 PerfSession 生命周期与统一接口

以下为拟议 C ABI，落地时固定 ABI/schema 版本、参数布局、缓冲区所有权及错误返回方式。
异常必须在 C++ 边界转换成状态与诊断信息，不穿越 C ABI；pybind11 不作为硬依赖。
当前多处 `assert` 会直接终止进程，不能靠 try/catch 转换；对外承诺可恢复的非法输入和
资源错误需改为受检返回/异常。不可恢复错误使会话失效，不能继续执行或伪装成 DONE。

```text
pimu_sim_create(config)
pimu_sim_get_effective_config_json(handle, dst, size)
pimu_sim_get_capabilities_json(handle, dst, size)
pimu_sim_load_program(handle, program_words, count, program_metadata)
pimu_sim_load_memory_plan(handle, json, size)
pimu_sim_run(handle, max_cycles)
pimu_sim_get_status(handle)
pimu_sim_get_stats_json(handle, dst, size)
pimu_sim_reset(handle)
pimu_sim_destroy(handle)
```

Python Backend 对应提供 `create/load_program/run/get_status/get_stats/reset/close`，可封装
为上下文管理器。`load_memory_plan` 传递容量、地址范围、布局与逻辑状态描述，不向性能
模型上传真实权重或 tensor 数值，也不能直接把尚未执行的 KV 写入标记为已完成。

生命周期定义为：create 后 IDLE；加载合法程序后 READY；run 时 RUNNING；完全排空后 DONE。
仅在 IDLE/DONE 的安全边界加载下一程序，READY 可在尚未执行时替换程序；RUNNING 不能
替换输入或 memory plan。非法加载返回 INVALID 并保留原安全状态；TIMEOUT、运行期
RESOURCE_CONFLICT 或 BACKEND_ERROR 记录现场，初版要求 reset/destroy 后再执行。
`max_cycles` 是本次 launch 的 core 周期预算；初版不把 TIMEOUT 当作支持任意暂停/恢复。

`load_program()` 与 `reset()` 的职责严格区分：

| 状态 | create/reset | DONE 后加载下一程序 |
|---|---|---|
| 程序、PC、EOF、decoder 部分包、launch 完成标志 | 初始化到空闲态 | 清前一程序的执行控制信息，设置新入口；加载前必须无残留部分包。 |
| 队列、功能单元、busy/锁、DDR 请求、inbox/outbox | 清空/重建为初始态 | 必须已按正常执行路径排空和解锁，不能直接清除未完成请求以伪造完成。 |
| SPM/ACC 建模状态、EventSync valid/数据代次 | 初始化无效 | 按已验证的 keep/lifetime 规则保留；编译器不能假定整个模型权重常驻。 |
| DRAM/KV 地址与逻辑有效范围 | 清除前一请求状态，重新导入 memory plan | 保留请求地址空间及已完成写入；Runtime 在 DONE 后发布新 KV valid_length。 |
| 寄存器与持久微架构状态 | 使用设备复位值 | 按寄存器和模块逐项规定保留策略，不能为了匹配冷启动而隐式复位。 |
| 模拟时间、core/DDR 相位与 DDR 时序状态 | 回到初始状态 | 连续保留；累计时钟、模块 cycle 和统计计数使用足够宽的整数。 |
| 统计 | 清累计值并建立新 session 标识 | 建立 launch 起止快照，返回阶段增量和 session 累计值。 |

当前 `openInstFile()` 只清部分前端状态，`PerfSim::run()` 的局部 base tick/core/DDR 计数也
不足以支持上述语义。需将连续时间和时钟相位归属于会话，并审计 decoder、各队列、
EventSync、功能单元、寄存器、DDR 模型及其回调；reset 可通过重建内部模块保证与新建
会话等价。每个状态项的保留/清除规则应有实现和回归证据。

首版阶段边界采用完全排空：程序源耗尽、无不完整包，InstCache/decoder/所有队列为空，
功能单元及退休处理结束，且 DDR 和全部模块间请求/响应均已完成后才能返回 DONE。
仅 core 空闲或 EOF 不足以完成；残留不完整包应报告 INVALID。跨 launch 的计算/访存重叠
作为后续调度扩展，不能把初版完全排空边界的结果解释为没有 host 边界的设备循环。

host 编译、绑定、FFI 调用、文件 I/O 的墙钟时间单独记录；初版 host 间隔不自动推进
device 时间，也不隐式收取程序加载或 PCIe 延迟。未来建模 launch/传输开销时使用显式策略
与版本；指令本身产生的前端、搬运及计算周期照常计入。

### 6.6 结构化输出与重构验收

运行结果包含 ABI/schema、session/launch/program 标识、完成/错误状态、模拟时间起止、
core/DDR/base tick、指令数、各模块 busy cycles、DRAM/SPM/ACC bytes、利用率及可用功耗
统计。阶段指标由起止快照导出，累计值跨 launch 连续；错误输出包含 PC/指令、部分包、
队列/事件与未完成请求的可诊断信息。观测接口不能要求开启逐周期日志才能获取终态。

起止快照相减只适用于周期、字节、指令数和能量等可加计数；峰值带宽、最大 occupancy
需在每个 launch 单独采样，不能对累计最大值做差。平均值/利用率用该阶段分子分母重算。
算子区间存在跨队列重叠，分别报告 issue→retire 延迟和资源 busy/等待，不能直接求和当作
全模型时间；程序/请求总时间由会话的全局时间边界给出。
当前 `simulated_inst_num` 在完整 packet 入等待队列时递增，并非退休计数。新 schema 分开
记录 fetched_words、assembled_packets、issued_packets、retired_packets；只有 DONE 且
完整排空时才可预期它们的相应数量一致，超时报告不能把已入队数量称为已执行完成数量。

验收至少覆盖：

- 同一配置/程序的旧 CLI、重构 CLI 与新会话首次执行的周期、流量及 busy cycles 对照；
- 文件和内存指令源等价，包括 cache/队列反压、续行与 EOF；
- 同会话连续 launch 的 SPM/ACC/EventSync/DDR 状态及 core/DDR 相位连续，reset 与全新会话等价；
- 完成排空、预算超时、不完整包、非法加载、错误后 reset，以及关闭逐周期日志后的结果获取；
- 一个会话实际完成全部 Prefill/chunk 和多个 Decode，逐层 KV 范围正确增长，阶段增量与累计统计一致。

尚未实现接口或未执行测试不得写成已有能力。迁移期间保留 CLI 回归路径，完整模型主线
验收必须使用同会话 Backend。

### 6.7 有效配置、能力查询与编译一致性

软件栈依赖 PIMU-Sim 的有效硬件参数。ModelAdapter 只解析模型配置，编译器和 Backend
通过统一的 DeviceConfig/TargetProfile 使用设备参数，不能在不同模型文件中各自解析
simulator config 或复制一套 C++ 默认值。

```text
simulator config + C++ defaults + backend implementation
 -> create/validate session -> effective config + capabilities
 -> compiler TargetProfile -> bound Program -> same-session launch validation

model config + checkpoint metadata -> ModelSpec
request lengths/token trace        -> RequestSpec
```

重构后由 C++ 的配置解析与校验层导出 effective config 和 capabilities；CLI 提供同等查询
入口，Python 通过第 6.5 节接口读取。所有查询结果附 schema/backend/ISA 版本，明确
requested、effective、ignored/unsupported 字段及单位。首版必须显式选择 PERFORMANCE，
因为当前 `Config` 的默认 simulator_type 是 FUNCTIONAL；缺失 mode 不能静默进入空功能路径。

| 参数组 | 软件栈用途及实际约束 |
|---|---|
| systolic_array_size、sram_data_width、acc_data_width | 决定向量访问字节数和受支持的 tile；存储位宽不是 BF16/FP16 计算能力声明。 |
| sram_bank_num、sram_bank_bytes | 决定整 bank 租约、容量、端口冲突和 guard 边界；总 SPM 容量为两者乘积。 |
| acc_bytes、acc_bank_num、bias_bytes、bias_bank_num | acc_bytes/bias_bytes 是各自空间的总字节数，不是每 bank 容量；旧 SA scoreboard 的模型主线限定 acc_bank_num=2、bias_bank_num=1，其他值先修复并验证。 |
| inst_cache_size、inst_queue_depth | 分别按 128-bit word 和等待中的完整 packet 计；影响有限队列进展，执行包/部分包另计。 |
| axi_data_width、ddr_request_queue_depth | 决定 AXI beat/burst 及反压；校验正值、整字节位宽及当前按位对齐实现需要的 2 的幂 beat，禁止零容量。 |
| ddr_mode、tCAS/tRCD/tRP/tBURST | DDR 固定为 AnalyticalDDR，兼容配置使用 ddr_mode=1；软件栈不提供其他模式，导入配置若不为 1 则明确报错。tBURST 当前按 DDR tick 间隔使用，不能按 Config 注释中的 ns 解释。 |
| npu_freq、ddr_freq | 同时导出名义频率、整数 ps 周期、base_tick_ps、ticks_per_period 和有效周期；验证 tick 数大于 0，core/DDR 相位跨 launch 连续保留。 |
| max_sim_cycle、日志/产物选项、功耗表 | 属于执行预算与观测/功耗配置；预算不定义模型层数，日志路径不改变程序合法性，功耗表哈希影响能量结果。 |
| iar/mer/ier/isr/ipr_reg_addr | 当前 RegFile 构造所需的五个配置项，缺少会抛错；Backend 校验它们存在、地址不冲突，不从任意模型路径推导寄存器配置。 |

当前 parser 会忽略未知 key；`core_mode/sram_ports/acc_ports` 等示例字段不生效。
`ddr_bandwidth/BL/ddr_bus_width/ddr_channel_num` 虽被解析，当前 AnalyticalDDR 并未用它们
改变服务过程，不能据此推导多通道带宽。导出接口应说明这种差异，Python 用户配置中的
未知/不支持能力应明确诊断，不得默默生效。SA/SFU 支持项、DMA 转换、端口规则和字段
位宽由实现导出 capabilities，不能仅凭用户填写一个 profile 名称启用新操作。

示例配置与 C++ 默认值分开记录：

| 字段 | 当前 Config 默认值 | 现有 smolvlm2_decode_cfg.txt 示例值 |
|---|---|---|
| simulator_type | FUNCTIONAL | PERFORMANCE |
| acc_bytes | 262144（256 KiB 总量） | 1048576（1 MiB 总量） |
| inst_queue_depth | 8 | 3 |
| ddr_freq | 2.13 GHz | 3.2 GHz |
| bias_bytes | 262144 | 未指定，使用默认值 |

这些是当前源码快照，不是 Python 应永久硬编码的默认配置；第 12.5 节采用的是显式验证
preset。profile 必须包含实际值，编译产物加载时与目标会话比对：

- ISA/capability、阵列/位宽、bank/容量和布局变化，需要重新编译或严格证明原计划仍合法；
- queue/cache/DDR/频率等调度条件变化，至少重新进行资源与进展验证；重新 autotune 的产物单独记录；
- 只改变日志路径无需重编译；执行预算改变影响是否完成，功耗表改变则重新计算能量。

分别记录程序兼容性、调度/优化和运行结果指纹，完整 provenance 保留所有有效配置和外部
配置文件内容哈希，不用一个包含日志路径的粗粒度哈希决定所有缓存失效。

## 7. Python ISA 与编译器设计

### 7.1 `sw/pimu/isa.py`

为每类 PIMU 指令定义 dataclass 和构造器：

```python
Inst(opcode, high, low)
mvin(...)
mvout(...)
sa(...)
matadd(...)
sfu(...)
vec(...)  # 仅在具备第 10 节扩展能力的 target 上可用
assemble(program) -> list[str]
decode(line) -> Inst
disassemble(program) -> str
```

字段使用第 3 节的位域布局与 dev 行为，集中写成 schema/常量；禁止在 kernel 中直接使用
`<< 32` 等裸位操作。区分 `InstructionWord(high,low)` 与包含 2/3 个 word 的完整
`InstructionPacket`，disassembler 保留已定义但未消费的字段。构造器必须检查：

- opcode 和行数；
- rows/cols/stride 是否适合字段宽度；
- DRAM/SPM/ACC 字节地址范围和选定布局策略的对齐；不将软件对齐策略误称为已有硬件检查；
- `source/is_bias/is_acc/acc_dest` 组合是否合法；
- 维度是否满足 SA GEMM/CONV 的约束。

提供两种明确模式：`baseline_compatible` 原样复现现有 benchmark（含已记录的 dtype、
量化 metadata 和保留值），`validated` 拒绝不支持/有歧义的组合并使用已验证模板。
`baseline_compatible` 只服务旧 ISA 样例；模型编译默认 `validated`，并绑定第 10 节扩展后的
capability。若后端缺少必需指令，编译失败而不是自动降级成简化模型。纠正已有 helper dtype
造成的流量变化单独报告，不以旧样例周期仍相同作为 Qwen3 的验收目标。

### 7.2 IR 和布局

`TensorDesc` 至少包含：`name, shape, dtype, address, strides, space, layout, producer, consumers`。
`space` 使用 `DRAM/SPM/ACC_DATA/ACC_BIAS`；stride 明确标注 byte/element 单位，
`layout` 使用 `row_major/tile_major/kv_sequence/kv_transposed` 等枚举。

在 tile IR 之前增加模型语义图：`ModelSpec`、`RequestSpec`、`ModelGraph` 包含层、
Linear/Norm/Attention/MLP/Embedding、权重绑定、head 分组、归约域和 KV 语义。
先做 config/shape/算子能力校验，再进行精度映射、kernel selection、tiling、bank allocation、
依赖调度与 ISA emit。`pl` 是 kernel IR builder，模型入口直接使用 ModelAdapter，
用户无需为每层/每个 head 手写 `@pl.kernel`。

增加 `BufferVersion(buffer_id,generation)`、`BankLease(bank_id,version,live_range)`、
`Op(reads,writes,resource_class,operator_id,tile_id)` 和 `TargetProfile`。复用同一物理 bank
时生成新 generation；一个 valid bit 不携带张量身份，verifier 必须防止读到旧代次但仍 valid
的数据。IR 同时记录 DRAM 的 RAW/WAR/WAW、ACC 两 bank 共享约束和 SA/MATADD 的互斥。

`WeightDesc` 额外包含 `quant_format, logical_bytes, stored_dtype, scale_address, zero_point,
block_size`。首版以 symmetric int8 格式/大小 metadata 描述权重，无需读取所有权重值；
ISA 中的量化字段只作为参数保存，不实现数值 quantize/dequantize。4-bit 和动态量化需要独立
格式及带宽契约，不能复用 int8 的隐式解释。

### 7.3 tile planner

SA GEMM 的标准 lowering：

```text
for output tile (m0, n0):
    for k0 in K tiles:
        MVIN A[m0:m1, k0:k1] -> A ping/pong SPM bank
        MVIN B[k0:k1, n0:n1] -> B ping/pong SPM bank
        SA(A, B, first/middle/last accumulate mode)
    optional MVOUT C temporary
    optional MVIN residual/other operand -> ACC banks
    MATADD or elementwise fusion
    MVOUT result -> DRAM
```

planner 必须同时满足：

- PIMU 的 SPM/ACC 容量和 bank 不能跨界；
- 首版 SA tile 的 M/N 不超过 min(systolic_array_size, 32)，K 不超过 2048；不能因 5-bit 字段可写 32 就让 16×16 目标隐式执行 32×32 tile；
- MVIN 必须先于使用它的 SA，SA 输出不得覆盖仍被读取的 bank；
- 首个、中间、最后 K tile 的 `acc_dest/is_acc` 配置正确；
- DMA queue depth 和 DMA1/2/3 分发策略可接受；
- 同一 bank 的访问不在一个周期冲突；
- 对失败的布局给出可读诊断，而不是让 EventSyncUnit 在运行时死锁。

首版采用经过验证的静态流和 ping-pong bank，保留已存在的多 DMA/计算重叠。
对 K tile 明确编码：单 tile `is_acc=0,acc_dest=0`；多 tile 的首块 `0,1`，中间块 `1,1`，
末块 `1,0`。按数据代次的后续消费者决定 A/B/bias keep：需要复用的本条设 1，最后消费
后通常设 0；若下一消费者是 MVOUT/SFU 等消耗型操作，也要纳入生命周期。不能每条都默认
keep=1。bias 状态即使在本条未读取时也不能被错误覆盖，keep=1 不能凭空创建有效数据。
ACC 同时供 SA 部分和与 MATADD 使用，两者的跨队列搬运生命周期也要检查。
K=0/M=0/N=0 在 IR 层处理为空操作或报错，不能用减一字段生成负数。

优化阶段再选择跨 tile 的 prefetch、SA/SFU 重叠和 keep 复用，但合法化不等于序列化：
compiler 通过地址、复用位和完整指令组的排列暴露并行，五个队列继续由原后端动态执行。

### 7.4 依赖与调度

以第 3.9 节的隐式 bank 同步规则为硬件契约，将完整数据依赖保留在 compiler IR 中；
openTPU 仅提供依赖图组织方式的参考，不移植它的硬件事件指令：

- 资源依赖：SPM bank、ACC data/bias 端口、decoder 自动选择的 DMA channel、SA/MATADD 共享队列、SFU；
- 数据依赖：producer output valid 后才能被 consumer 读取；
- 生命周期：最后一个 consumer 完成后 bank 才能复用；
- 指令组依赖：2/3 行必须原子生成，不能被其他指令插入；
- 终止依赖：所有队列、执行单元和 DMA 请求排空才认为程序完成；最后一条 hex 被读入不是终点。

compiler 不能只验证 DAG 无环或逐条串行更新 valid/busy；一个有向无环的模型图，映射到
有限 FIFO 和循环复用 bank 后仍可能无进展。编译缓存键包含 ISA/profile、配置、shape/layout、
tile/bank/scheduling 策略；queue_depth 或 bank 容量改变后必须重新验证。

### 7.5 Parallel PIMU 的有界队列与进展验证

检查器根据 target profile 复现前端/队列状态转移，不另造功能数值模拟器。第 3 节定义旧指令
行为，第 10 节新增指令实现后同步加入检查器，不用旧双 word SFU packet 代替新包的发射过程：

1. 128-bit word 逐行输入，前端 `Dispatch→Decode→Fetch`，保留当前 decode_valid/fetch_valid 门控和由此产生的空泡；完整包才入等待队列。缓存按 word、等待队列按 packet 计，正在执行包和部分重组包分开计数。
2. 复现 DMA 首行选路、续行锁定 channel、两个 SPM round-robin 游标和 DMA1 的 `queued+busy+partial` 负载；实际选路影响不同 DMA FIFO 的相对顺序，不能提前硬编码。
3. 只从队首发射，依次应用 SA→SFU→DMA1→DMA2→DMA3 的即时状态更新；SA 与 MATADD 同时依赖 Accumulator 空闲。保留“本周期完成、先 issue 后 retire、下一周期才可再次 issue”的边界。
4. 按第 3 节当前单元计数和完整访存区间检查占用；建模 core/DDR 相位、outbox/request queue 反压和响应类型匹配。先覆盖现有模板，未支持的 timing/config 显式拒绝，不用理想延迟给出通过结论。
5. 跟踪每个 bank 的数据 generation、读者生命周期和最后消费者；下一次 MVIN/覆盖之前必须有可验证的消费/失效路径，否则会被残留 valid 阻塞。程序末尾或跨阶段有意保留的有效数据可以无本段消费者，但须在状态契约中声明，不把单纯残留 valid 判为死锁。
6. 无发射但仍有 BUSY/DDR 进行中属于等待；所有可能的完成事件均消失且仍有队首阻塞才报告死锁候选，输出等待链、queue head、bank 代次及最早冲突指令。以真实 C++ 回归确认检查器预测，不将 max_sim_cycle 等同于数学死锁证明。

DRAM 数据依赖需要额外处理：EventSyncUnit 不检查 DRAM 地址，文本中先写 MVOUT、后写
MVIN 不保证它们位于不同 DMA 时按此完成顺序执行。planner 必须证明相应发布/读取顺序，
优先保留片上 producer→consumer 或选择已验证的队列顺序；若当前 ISA 无法可靠表达所需依赖，
可在相关子图边界完整排空后分段 launch，并在报告中明确分段及丧失的重叠。不能编造免费
设备 barrier，也不能把跨 launch 的周期直接当作单段驻留程序的周期。

现有 SmolVLM2 脚本中有 `emit_acc_barrier()`，其形式是额外 MVIN/MATADD/MVOUT，
不是 ISA barrier。仅在旧基线实际调用时按原工作量保留；新 compiler 不用人为增加矩阵
计算来掩盖依赖问题，任何额外搬运/计算均计入指令和流量。

### 7.6 模型到指令的编译流水线

```text
model metadata + RequestSpec
 -> Qwen3Adapter.build_graph (all layers, embedding, final norm, lm_head)
 -> shape / weight binding / operator coverage validation
 -> precision mapping + logical layouts + per-layer KV planning
 -> kernel selection and capability check (including required extensions)
 -> reduction-preserving tiling / transpose / explicit data movement
 -> DRAM placement + SPM/ACC allocation + parameter/temporary buffers
 -> dependency scheduling + keep flags + bounded progress/resource verification
 -> instruction emit + manifest + coverage + operator/layer trace mapping
```

prefill、decode 和 logits selection 使用统一图生成规则，根据请求派生不同 shape。
kernel 注册表按 operation、dtype、layout、target capability 匹配，不按模型名切换裸位编码。
候选融合必须给出真实 kernel 与成本；不能只删 IR 节点后沿用原始语义覆盖率。

### 7.7 完整模型的指令规模与编译缓存

完整模型可能产生大量 128-bit word，emit 使用可流式写出的 InstructionBlock/迭代器，
metadata 按范围索引，避免要求一次性构建全部 Python 字符串。前端通过统一指令源逐 word
读取最终字节流，控制流展开的真实指令与前端时间都计入；模板复用不等于免费硬件 LOOP。

可缓存通用 kernel 的编译结果和不同层共享的 tile 模板，再绑定不同权重、KV、scratch
地址。缓存键包含 model/config revision、请求 shape、precision、ISA/profile、辅助参数、
bank/队列配置与 schedule。所有层和每个 Decode step 均实际发射/仿真；不通过“测单层再乘层数”
或“测一步再乘生成长度”替代完整调度，跨层复用和增长的 KV 访问会影响结果。

## 8. Runtime 和内存模型

### 8.1 Device/Buffer API

模型 API 是主入口，descriptor/kernel API 保留用于自定义算子。以下为拟议接口：

```python
model = pimu.Model.from_pretrained(
    "sw/checkpoints/Qwen3-0.6B", adapter="qwen3", revision=model_revision
)
request = pimu.Request(batch_size=1, prompt_length=128,
                       decode_steps=16, logits_positions="last")
with pimu.Device(config="sw/configs/device.cfg",
                 target="parallel_pimu_qwen3_v1", backend="performance") as device:
    program = pimu.compile(model, request=request, device=device,
                           precision=pimu.PrecisionPolicy.w8a8(kv_dtype="int8"))
    result = device.run(program)
```

`Device` 通过 PerformanceBackend 管理 C++ PerfSession，一个请求内复用会话执行各阶段，
结束后显式 close 或由上下文管理器释放；独立 CLI backend 作为兼容入口。`Buffer` 暴露
descriptor，模型层无需管理绝对地址；可从 host tensor 或
checkpoint header 提取 metadata，但不上传/执行真实 tensor 数值。`result` 返回完成状态、
目标/配置指纹、cycles、bytes、利用率、energy 和 trace，不包含 tensor 数值。
from_pretrained 在本方案中导入结构与元数据，不表示调用 CPU/GPU 的完整 forward 代替
NPU 仿真。可另接受 prompt/token IDs，经 host tokenizer 得到真实长度与 embedding 行地址；
仅传长度时使用有标识的 token trace，便于确定数据无关执行计划。

### 8.2 Memory plan

采用 openTPU runtime 的“先 placement、后 compile binding”思想，但只生成地址和容量计划，不复制或解释真实数据：

1. `DramAllocator` 按对齐规则分配 weights、inputs、intermediates、outputs 和 KV cache，记录每个 buffer 的 bytes 和 lifetime。指令/manifest 属于 host 控制面；当前指令文件进入 InstCache，不把它虚构成经 DMA 从 DRAM 搬入的 tensor 流量。
2. `Layout` 保存 host object 到 `(address, shape, stride, bytes)` 的映射。
3. 每次 launch 更新动态 shape、position、KV base/stride 等 descriptor；当前没有硬件 mailbox，不分配或写入虚构 mailbox。每步仍需按计划计入权重/KV 搬运，权重固定地址不等于权重已常驻 SPM。
4. memory plan 同时记录每个算子的 DRAM read/write bytes，供 compiler 估算和 backend 对账。

### 8.3 SPM/ACC placement

SPM/ACC 是 compiler 的 scratch allocator，不是 host 可见的长期 buffer。为每个 kernel 生成 bank plan：

- A/B ping-pong banks；
- SFU input/output banks；
- SA/MATADD output banks；
- ACC data 双 bank；
- bias bank；
- 每个 buffer 的 byte base、stride、tile shape 和复用区间。

首版硬件配置可先使用 `sram_bank_num=8, acc_bank_num=2, bias_bank_num=1`，
bank plan 由 Qwen3 算子、辅助参数和生命周期生成，不要求复制旧 generator 的固定 bank 地址。

这只是一份软件分配模板，允许后续按 workload 调整；不可把 bank0/1 只能接 DMA1、bank2/3
只能接 DMA2 等 Banked PIMU 限制加入当前 target。每个 bank 初始无效；同一 bank 的不同
offset 不会得到独立 scoreboard，首版按整 bank 分配活跃租约，并为实际额外向量访问留 guard。
不将“总 SRAM 有剩余”视为可以突破 11/5-bit shape 字段或同时保存任意多个有效 tile。

### 8.4 模型精度映射与地址容量

区分 `checkpoint_dtype`、`modeled_weight_dtype`、`activation/kv/aux_dtype`、
`accumulator_dtype` 和 kernel 内部计算精度。首版 W8A8/ACC32 是面向当前数据通路的
性能部署假设，aux 默认 fp16；原始模型元数据不被改写。归一化/非线性/attention 的宽精度
临时量、重定标及必要参数读写需要显式成本，不声称该 profile 等价于 BF16 数值推理。

per-tensor/per-channel scale、norm gamma、RoPE 系数和量化 metadata 进入 memory plan；
runtime 重量化若发生，必须有 kernel/访存成本。模型静态参数常驻 DRAM 和常驻 SPM 是两种
不同状态；“权重只导入一次”不能删除每次推理所需的 DMA。

dtype 转换还必须落实到每条边的生产者/消费者契约，而非仅在 manifest 写 W8A8：

| 数据边界 | 必须表达的性能行为 |
|---|---|
| SA int32 部分和 → int8 激活 | 区分 SA/Accumulator 的已建模输出路径与新增重定标内核；scale 参数、读取/输出和延迟有明确来源。 |
| int8 QK 输出 → scale/mask/Softmax | 明确 scores 的存储与归约精度、转换位置及临时区；不能用 int8 字节数冒充 fp16/fp32 工作量。 |
| Softmax 概率 → PV GEMM | 如果 PV 使用 int8 SA，概率必须有对应格式转换和 scale 流量；现有 SA 不能因 descriptor 标注 fp16 就获得 fp16 GEMM 能力。 |
| Norm/RoPE/SiLU/SwiGLU → 后续 SA | 显式声明输入/输出 dtype、内部宽精度和量化/转换是否融合，匹配下一输入的实际存储格式。 |
| residual → MATADD | 两输入按 ACC32 布局与搬运字段准备，或使用已实现的向量加法能力；不能把 fp16 SPM 地址直接作为 ACC32 输入。 |

新增 VEC 的向量元素数由实际端口字节宽度和该操作 dtype 推导，fp16/fp32 参数不能沿用
int8 的元素吞吐率。当前 MVOUT 的 is_quant 不增加转换周期，旧 SFU 的 is_quant 只影响
既有 timing；这两个 bit 均不能作为任意 dtype 转换已经实现的证据。需要额外转换时，将其
纳入第 10 节内核与 coverage，能力不足则拒绝完整支持声明。

当前所有地址字段只有 32 bit。allocator 检查权重、KV、激活、临时区及 guard 的每个
`base+span` 不超过可寻址空间，并同时检查 SPM/ACC 实际容量。默认小模型先在该边界内完成
完整部署；更大模型无法容纳时显式拒绝，或在单独版本中加入有成本记录的地址窗口/外存调度。
不得截断地址、溢出回绕或把全模型权重压成一个小 tile 来规避容量。

### 8.5 每层 KV cache 与连续请求状态

每个请求、每层各有 K/V descriptor，记录 `capacity, valid_length, position, Hkv, D,
dtype, layout, strides, base`。容量由 `prompt_length+decode_steps` 和有效模型限制决定，
不固定为 33。首版可用 K `[B,Hkv,D,capacity]`、V `[B,Hkv,capacity,D]`；
layout 选择对应真实 transpose/写回/读取成本，不默认为免费 view。

非压缩缓存的逻辑容量为 `2*layers*B*Hkv*capacity*D*kv_element_bytes`，另加对齐/metadata。
物理 KV head 按 Hkv 分配，GQA 的 Hq 个查询通过分组映射访问，不将持久 cache 扩大 R 倍。
各 step 只追加当前 K/V 并读取其需要的历史范围，不重复追加整段历史，也不读取 capacity
中尚未有效的位置。同会话运行保留请求的 DRAM/KV 状态描述，片上有效位由实际指令和
keep/lifetime 规则更新，不能用 host 的 KV 地址表直接恢复 SPM/ACC valid。
独立进程回归仅传递逻辑状态描述，并明确标注冷启动，不声称继承片上驻留状态。

Prefill 分块时保留跨块 position、有效 cache 长度与 causal mask，不能把每块当作独立序列。
默认先使用可容纳的完整行归约与显式外存中间结果；长上下文需要多遍/online attention
时，临时最大值/归约和重读成本同时纳入，不能将 Softmax 简单按 tile 拆开。

## 9. Qwen3 模型接入与通用算子库

### 9.1 首个模型与支持边界

默认首个实例为 **Qwen/Qwen3-0.6B 的 Dense 文本模型**；选择小规格用于控制指令量和仿真耗时，
完整层数和真实结构均保留。adapter 读取模型配置而非硬编码规格，相同架构的其他 Dense
变体在容量和能力检查通过后复用同一路径。不同 model_type 分别注册，MoE、VL 或其他变体
不自动继承 Dense 支持状态。[Qwen3 官方模型说明](https://huggingface.co/Qwen/Qwen3-0.6B)

首版范围为 batch=1、推理模式、完整 Prefill 和指定步数 Decode；默认验证 prompt 长度
32/128/512，另测 1/31/33/129 等边界。超长上下文、批处理和其他 RoPE/attention 变体按
capability 逐项开放，不把默认测试长度写成模型本身的最大上下文限制。

### 9.2 配置、参数与维度推导

adapter 读取 `config.json` 中的层数、hidden/intermediate/head 维度、Q/KV head 数、词表、
norm epsilon、RoPE 参数、attention 模式和权重共享规则，并锁定模型 revision。
首个配置快照如下；实际导入值优先于本表，配置变化需重新生成图与内存计划。

| 参数 | Qwen3-0.6B 的配置值 |
|---|---:|
| num_hidden_layers | 28 |
| hidden_size | 1024 |
| intermediate_size | 3072 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| vocab_size | 151936 |
| hidden_act | silu |
| attention_bias | false |
| rms_norm_eps | 1e-6 |
| rope_theta | 1000000 |
| rope_scaling | null（默认 RoPE） |
| max_position_embeddings | 40960 |
| tie_word_embeddings | true |
| use_sliding_window | false |
| checkpoint dtype | bfloat16 |

以上数值来自[官方配置](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json)。
首次集成时记录实际配置哈希与不可变 revision，不依赖远端 main 始终不变。

定义 `H=hidden_size`、`I=intermediate_size`、`D=head_dim`、`Hq/Hkv` 为 Q/KV head 数，
`R=Hq/Hkv`。Q 宽度为 `Hq*D`，K/V 宽度为 `Hkv*D`，O projection 输入为 `Hq*D`。
**不能使用 `hidden_size/num_attention_heads` 覆盖显式 head_dim，也不能假定 Q 宽度等于 H。**
首个模型的 Q 宽度是 2048，K/V 宽度是 1024，GQA 分组为 2。
这些维度由配置及投影定义推导。[Qwen3 配置定义](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/configuration_qwen3.py)

权重导入校验 HF Linear 的 `[out_features,in_features]` 与 NPU GEMM layout 的转换关系：
Q/O 投影、K/V 投影、三路 MLP 和 embedding/lm_head 均独立检查形状。绑定权重共享只合并
存储 allocation，不删除 embedding 读取或 lm_head 计算；不得按参数名猜测不存在的 bias。
权重共享时 checkpoint 可以只保存 embedding 一份；允许缺少独立 lm_head tensor，但需
验证配置、绑定关系及等价形状，不能将合法的 tied weights 判成缺失权重。

adapter 校验 Hq/Hkv 为正且整除、D 与旋转布局兼容、投影和 gamma 形状匹配。
首版明确支持默认 RoPE、无 sliding window 的 Dense 路径；attention_bias、其他激活或
RoPE 变体须有对应 lowering/capability，否则明确拒绝，不能忽略配置项后继续套默认图。

### 9.3 完整模型计算图

模型前端构建如下语义图，然后展开全部层并进行算子 lowering；层实例与权重/KV 地址必须
区分，不能仅模拟一层后将周期乘以层数。

```text
token IDs -> embedding lookup
for each decoder layer:
    r0 = hidden
    x = RMSNorm(hidden)
    Q, K, V = LinearQ(x), LinearK(x), LinearV(x)
    Q = RMSNormPerHead(Q)
    K = RMSNormPerHead(K)
    Q, K = RoPE(Q, K, positions)
    append current K/V to this layer's KV cache
    scores = (Q @ K_cache^T) * (head_dim ** -0.5)
    probs = Softmax(scores + causal/padding mask, over all visible keys)
    context = probs @ V_cache                  # GQA head mapping
    hidden = r0 + LinearO(concat_heads(context))
    r1 = hidden
    x = RMSNorm(hidden)
    hidden = r1 + LinearDown(SiLU(LinearGate(x)) * LinearUp(x))
hidden = FinalRMSNorm(hidden)
logits workload = LMHead(selected hidden positions)
```

Q/K Norm 按每个 head 的 D 维处理，RoPE 位于 Q/K Norm 之后；MLP 包含 gate/up/down
三路线性层和逐元素乘法，不能替换为单路 GeLU FFN。
[Qwen3 模型实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py)、
[Qwen3 Attention 定义](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modular_qwen3.py)

generation 性能请求默认 Prefill 仅对最后一个位置执行 lm_head，Decode 对当前单 token
执行 lm_head；另支持 all_positions 口径。manifest 必须记录 `logits_positions`，不能把全序列
投影与末位置投影的性能混为一谈，也不能完全省略 lm_head。tokenizer/chat template 与采样为
host 控制面；它们不在 NPU scope 时单列说明。[logits_to_keep 接口](https://huggingface.co/docs/transformers/model_doc/qwen3#transformers.Qwen3ForCausalLM.forward)

### 9.4 Qwen3 必需算子的 lowering

| 语义算子 | 执行方案 | 验收要求 |
|---|---|---|
| embedding lookup | host 已知 token ID→对应权重行的 DMA 读取；布局转换按需物化 | 记录行地址、字节和重复访问；未知生成 token 使用外部/合成 trace 并注明，不将查表成本记零。 |
| Q/K/V/O、gate/up/down、lm_head | SA GEMM + ACC 累加 + 显式搬运 | 使用真实维度和每层权重；M=1 Decode 仍走已实现 GEMM，不调用 GEMV 占位路径。 |
| RMSNorm、Q/K Norm、final norm | 新增 RMSNorm 性能内核/扩展指令，包含 gamma 读取和完整 reduction | hidden norm 与 head norm 使用各自归约宽度；禁止使用 LayerNorm 周期冒充。 |
| RoPE | 新增位置相关内核，输入 Q 或 K、position、旋转系数参数 | 指定系数预生成/存储及读取成本；共享同一位置表不等于全部读取免费。 |
| GQA attention | 显式 head→KV head 映射、QK/PV GEMM | 不默认复制 R 份持久 KV；确有物化才增加流量。 |
| scale、causal/padding mask | 扩展向量内核，或有完整成本契约的 fused masked-softmax | 基于 query/key position 和有效长度，不能省略；跳过未来位置计算需合法 mask-aware schedule。 |
| Softmax | 完整 key 维归约；可复用现有 optype 0 的已验证访问路径 | 不能对每个 key tile 各自 Softmax 后直接拼接；容量不足需多遍/online 算法和对应读写成本。 |
| SiLU、逐元素乘法/SwiGLU | 扩展向量内核；gate/up 两输入明确可见 | 独立算子或真实 fused 内核均可，参数、读写、占用和延迟均计入。 |
| residual add | 可先使用 ACC 搬运→MATADD，或经验证的双输入向量加法 | 保留两条 residual 路径和真实 dtype/转换开销。 |
| KV append/read、transpose、head layout | DMA/transpose/显式 layout lowering | 区分零成本 view 与需要搬运的数据重排，不因 IR reshape 就默认已完成物理转置。 |

所有 norm/attention 的 reduction domain 必须保留在 IR 中；分块只是物理执行方式，不能
改变语义上的全局归约范围。新增内核先完成第 10 节的可执行性能契约，再宣称 Qwen3 全覆盖。

### 9.5 ModelAdapter 与模型扩展接口

统一接口建议为：

```text
ModelAdapter.detect(config) -> supported model_type/variant
ModelAdapter.load_metadata(model_dir, revision) -> ModelSpec + WeightManifest
ModelAdapter.build_graph(ModelSpec, RequestSpec) -> ModelGraph
ModelAdapter.validate_weights(WeightManifest) -> shape/tie/dtype report
compiler.lower(ModelGraph, TargetProfile, PrecisionPolicy) -> ExecutableProgram
runtime.run(ExecutableProgram) -> PerformanceResult
```

首版直接实现 `Qwen3Adapter` 的显式语义图，可用 PyTorch meta/FX 做 shape 对照，不要求
任意 Python forward 自动编译。加载优先读取 config、safetensors index/header 和参数名/形状；
性能模式不必将全部权重数值常驻 host。仅有 config 时可导出结构分析结果，但必须标注
`metadata_validation=config_only`，不能伪装成已完成 checkpoint 一致性校验。

ModelGraph 只保留数学语义、权重绑定、KV 行为和请求信息；通用 IR/kernel 不含 Qwen3
模型路径或固定层号。后续模型通过新 adapter 组合既有 Linear/Norm/Attention/MLP 算子，
缺失能力在 kernel registry 增加，不复制整个软件栈。MoE 等数据相关执行未来需要 routing trace
或明确分布模型，不能仅凭静态 shape 估计真实专家选择。

### 9.6 覆盖报告与模型支持状态

`coverage.json` 为每个原始语义节点记录 `native/lowered/fused/unsupported`、kernel、
instruction range、权重/辅助参数和测量口径。fusion 保留原始节点到真实内核的映射。
声明 `qwen3_full_model_supported` 时必须包含 embedding、所有真实层、Q/K Norm、RoPE、
attention、SwiGLU、两路 residual、final norm、lm_head 和每层 KV 更新，且 unsupported=0。
每条 dtype/布局转换和附加参数搬运同样纳入覆盖。coverage 分别记录 modeled、executable、
validated 状态：有 IR 或手填一个延迟不等于后端可执行，后端执行 DONE 也不自动证明模型
已经校准。完整支持至少要求每个节点/数据转换实际可执行并通过相应资源与行为验收。

单层/缩小配置/局部子图用于开发测试，并标记 partial；不能作为完整模型的最终验收。
在性能模式中“全覆盖”指计算、访存和依赖行为都有非遗漏的模型，不表示数值推理精度已验证。

### 9.7 后续模型的接入规则

模型适配代码位于 `sw/pimu/models/`；配置/检查点目录由用户提供，不将目录名当作模型
架构标识。注册表负责 model_type/variant→adapter 的确定性选择，未知或重复注册明确
报错；新增 Python 文件还须经显式导入/发现并注册，不能假定任意文件会自动生效。

| 扩展情形 | 最小改动范围 |
|---|---|
| 同架构、不同层数/维度 | 若配置、布局、精度与容量检查通过，提供模型目录即可复用已有 adapter，通常无需新增 Python。 |
| 新架构、全部算子和状态机制已支持 | 增加 model adapter、注册入口和模型级验证用例；复用通用 compiler/runtime/backend。 |
| 新算子、dtype、布局或缓存/路由状态 | 额外补充 kernel/lowering/state contract，确需新硬件能力时版本化扩展后端/ISA。 |

adapter 负责元数据、权重绑定、语义图和模型状态，不自行拼裸 ISA、固定物理 bank 或
逐算子调用 simulator。后续以第二个组合已有能力的 adapter 验证扩展边界：接入时无需修改
通用编译器、Runtime 或 ISA；不要求首个交付同时支持任意 Hugging Face 模型。

## 10. Qwen3 必需的性能模型与 ISA 扩展

### 10.1 基线能力与模型目标的分离

第 3 节是当前五类指令的实际规范，应完整保留。它不足以直接表达完整 Qwen3，补齐缺失
算子属于首版工作，而不是完整模型完成后的可选项。新增能力继续使用现有 Parallel PIMU
的三 DMA、同质 bank、SA/Accumulator、SFU 及队列结构，不转向 Banked PIMU。

建议保留 `parallel_pimu_dev_v1` 兼容旧程序，新增继承其架构的
`parallel_pimu_qwen3_v1` capability profile；这是拟定名称。编译完整 Qwen3 时检查后端能力，
缺失则报错，不把新程序交给未识别扩展指令的旧二进制。新增版本不等于更换成另一种硬件架构。

### 10.2 扩展指令的承载方式

推荐为多输入/带参数向量操作新增 **VEC 指令组**，候选 opcode 为当前未占用的 `0x06`，
长度固定 3 个 128-bit word，分发到现有 SFU 执行队列。其格式在实施时单独冻结，
decoder、packet/queue、EventSyncUnit、SFU、Python schema 与 trace 同步实现。
这项格式是设计建议，不能混入第 3 节冒充已经实现的位域。

VEC 必须显式携带/绑定：operation kind、输入/输出 byte address、辅助参数地址、
shape/stride、dtype、归约轴长度、position/有效 KV 长度、scale/epsilon 等所需参数。
不同 operation 使用不同受检布局；若 3-word 字段容量不足，先调整并冻结扩展格式，
不得用未被 simulator 解码的 manifest metadata 代替真实操作数。原 SFU 第 1 行已定义的
量化位也不能随意改为第二输入地址。

| 必需操作 | 最少可见信息 | 性能建模要求 |
|---|---|---|
| RMSNorm | x、gamma、out、rows、完整归约宽度、epsilon、dtype | 数据/参数读取、归约与逆平方根路径、乘 gamma、输出；Q/K Norm 和 final norm 复用同一内核。 |
| RoPE | x、packed sin/cos 参数、out、head_dim、position、layout | 旋转配对、系数流量、多个 head 的复用策略和写回；position 不只保存在 Python 中。 |
| SiLU | x、out、elements、dtype | 非线性流水周期、向量读取和写回。 |
| elementwise multiply/add | a、b、out、shape/stride、dtype | 两输入依赖、逐次端口访问；支持 SwiGLU 和可选 residual。 |
| scale/mask | scores、out、scale、query/key positions、有效长度 | 全 key 维的有效区域、因果/padding 边界、必要读取/写回；可融合但不能删除工作量。 |
| masked-softmax / 多遍归约 | scores、out、归约域、有效长度、临时区 | 完整 reduction 的阶段依赖与流量；不能把独立 tile Softmax 当作全局 Softmax。 |
| convert/requantize（需要时） | 输入/输出 dtype、scale/zero-point 参数、布局、元素数 | 对第 8.4 节每条精度边界给出已实现路径；可与其他内核融合，但读写、参数和转换吞吐不能遗漏。 |

仅给旧 `optype` 新增一个名字或延迟常数不够：新增 gamma/系数/第二输入都会影响 DMA、
SPM 端口、scoreboard 和总周期。若后端暂不实现某 operation，不得以 CPU 数值算子或
零成本 host callback 绕过；partial 分析结果必须列出缺失项。

### 10.3 延迟、流量和功耗校准

每个内核定义 setup、vector/reduction batch 数、pipeline latency、尾部周期、每周期端口
访问及 modeled bytes。SPM 访问始终按实际端口字节宽度计费；例如辅助 fp16 参数存储需要的
字节数不同于 int8 激活，不能因为共用同一 bank 就按相同元素数计算访问次数。

优先使用已有算术单元/设计资料可支持的分解模型，再以独立 microbenchmark 校验；
没有硬件依据的常数标成 estimate，报告模型来源和适用范围。新增功耗状态需要校准；
未校准时可交付 cycles/bytes，同时将对应 energy 标记 unavailable/estimated，不能默认记 0
后给出确定的完整模型节能结论。性能参数可配置，但不能按期望模型时延反向调参。

### 10.4 模型集成前必须审计的并行行为

在扩展单元接入前处理 request_id 响应身份、输入读生命周期和端口冲突行为。修复需独立记录
版本及旧 ISA 回归差异，但不能以保持旧错误周期为理由让 Qwen3 依赖不可靠的多 DMA 同步。
VEC 的多输入/辅助参数要纳入 bank valid/busy 和 compiler lease；完整 reduction/双输入
不能在未获取资源时开始，也不能在参数未加载时用默认值完成。

## 11. Decode 和控制流策略

### 11.1 主线：持久会话中的 host-driven step

完整请求先执行一次覆盖全部层的 Prefill，再执行指定次数的完整 Decode forward：

```text
load Qwen3 -> compile full prefill + decode templates -> create PerfSession
 -> run embedding + all layers + final norm/lm_head for prompt
 -> publish all layers' KV state
for step = 1..decode_steps:
    bind token trace / position / visible KV length
    run embedding + all layers + final norm/lm_head for this token
    publish each layer's appended KV and collect step statistics
close PerfSession after the request
```

该路径用于生成每个 decode step 的静态 workload；token 数值、采样和 logits 正确性不属于
performance backend。同一请求的 Prefill/chunk 和 Decode 使用同一个 C++ PerfSession，
遵循第 6.5 节的完全排空、状态保留及连续时钟规则。历史 KV 由 descriptor 和 memory plan
描述，当前步实际读写照常计费；片上复用必须有此前指令执行和编译器 lifetime 的证据。
时间分别记录连续设备模拟时间、各 launch 增量和 host 编译/绑定/提交耗时。
持久仿真器会话不等于设备自行执行 token 循环：初版仍由 host 每步提交程序，阶段边界
完全排空，不建模跨 launch 的执行重叠。独立进程仅用于冷启动/兼容对照，不能替代主线验收。

`decode_steps=G` 明确定义为 Prefill 后实际执行 G 次 Decode forward，不直接等同于某个
文本生成框架的 max_new_tokens。若 prompt 长度为 P，第 i 次 Decode 处理绝对位置
`P+i-1`，追加后每层有效 KV 长度为 `P+i`。报告分别给出 device_prefill_time、
每步 decode_time、device_request_time、请求/上下文长度，以及 host 时间和冷/热启动语义。
设备侧 TTFT 若报告，只覆盖本 scope 的 Prefill+末位置 lm_head，排除的 tokenizer/采样/
排队时间必须注明。请求上下文不能在层或 step 之间错误复位。

首版请求检查 batch=1、P>0、G≥0，初始序列从位置 0 开始，P+G 不超过已支持的模型位置
范围与 KV 容量；G=0 表示仅 Prefill。外部 token trace 覆盖所需输入位置且 token ID 在词表内，
合成 trace 记录生成规则/种子。性能后端不产生下一 token 或 EOS，固定步数请求不会自行
提前终止；按 EOS/数据相关路径结束需要外部 trace 给出，不能声称模拟器完成了实际采样。

### 11.2 后续：设备侧循环与 resident program

当单步正确后，再增加：

- `HALT`/`LOOP`/`JUMP` 或等价控制流；
- host mailbox 和 `WAIT` 指令；
- workload/event ring buffer；
- device-side scheduling metadata；
- program chaining、跨 launch 重叠和更长期的权重/中间结果驻留策略。

这些属于持久会话和模型链路完成后的独立架构扩展。设备控制流、mailbox 与 program
chaining 需更新 decoder、performance pipeline、manifest、runtime ABI 和状态规则；
现有 keep 位支持的跨阶段数据保留在主线中验证，不以增加 LOOP 为前提。
不能只在 Python 层伪造时序，也不能把 host 重复 launch 当作实现了硬件 LOOP。

## 12. 验证体系和验收标准

### 12.1 单元测试

- ISA encode/decode round trip；
- 每个位域边界、负例和非法组合；
- 128-bit hex parser 与现有 benchmark 完全兼容；
- DRAM/SPM/ACC allocator 对齐、容量和 bank boundary；
- int8/int32 dtype、元素字节数和量化字段序列化 metadata；不建设数值 quantize/dequantize 验证路径；
- tile planner 的整除、尾块和 K 累加模式；
- EventSyncUnit 等价 scoreboard verifier。

### 12.2 Performance compiler tests

每个 kernel 至少检查：

1. 生成的 opcode、字段和指令行数；
2. tile shape、stride、memory plan 和 bank allocation；
3. EventSyncUnit 等价 verifier 是否接受依赖图；
4. 真实 Parallel backend 的完成状态、cycle、DDR/SPM/ACC bytes 和模块 busy cycles 是否符合 golden；
5. 不同队列并发时的 bank generation、最后读者、复用位和 ACC/MATADD 共享占用；
6. 队列深度改变、DMA1 被占用、目标队列满/部分包在途情况下的路由和进展；
7. 验证器拒绝无效程序的诊断，不通过插入伪 barrier、关闭 DMA 或改成串行执行规避错误。

### 12.3 后端兼容回归与模型回归

- 现有 `tb_resnet_block.txt`、`tb_smolvlm2_prefill.txt`、`tb_smolvlm2_decode.txt` 周期和统计回归；
- 既有 hex 的 encode/decode 保留字段、行数和既有语义；不要求先由新模型前端重新生成三个旧文件；
- Qwen3 kernel、真实单层、完整模型 Prefill 和多步 Decode 的版本化 golden；
- 多次 launch 的地址/lifetime/状态计划不发生重叠；
- 第 6.6 节核心库/CLI、文件/内存取指、持久会话、时钟相位、统计增量和 reset 回归；
- effective config/capability 导出与实际行为一致，未知配置、缺失寄存器、错误 mode、配置不匹配和未实现 dtype 转换均拒绝；
- 超时、非法指令、容量不足、bank 冲突、DDR backpressure 可诊断且不会静默死锁。

未改变执行语义时，旧指令兼容回归要求周期、整数流量和忙周期一致；修复和扩展造成的变化
按版本单独解释。Qwen3 指令不与旧文件比较字节数或周期，而与其完整语义图、参数和独立
kernel 基线对照。调度优化允许 hex 和周期变化，但模型算子图、
dtype/layout、目标配置及有效 payload 保持可对照，输出搬运数/复用收益/guard 流量差异报告。
修复 dtype、同步或响应匹配造成的变化单独归类为契约修复，不算作纯 compiler 加速。

### 12.4 模型级验收

首个交付必须完成以下闭环：

```text
Qwen3 model directory + prompt/request + precision/device profile
 -> adapter -> full semantic graph -> kernels -> scheduled ISA program(s)
 -> Parallel PIMU backend -> full prefill + per-step decode reports
 -> coverage + memory/traffic/layer/operator/cycle breakdown
```

验收报告应同时给出：模型配置、权重格式、输入 token/shape、指令数、总周期、DRAM bytes、SPM/ACC bytes、各模块利用率、队列等待和每算子统计。

必须满足以下条件：

1. 模型入口不接受已有模型 txt 作为必要输入；从真实 Qwen3 config/checkpoint metadata 自动生成完整程序，不要求用户编写每层 kernel 或手工转换指令。
2. 默认实例保留 28 层和真实维度，涵盖 embedding、全部 decoder、final norm、lm_head 与每层 KV；coverage 的 unsupported=0。
3. 融合前语义图中每层有 7 个 Linear、4 个 RMSNorm 域、2 条 residual；最终另有 final norm 和 lm_head。按 head 展开的 kernel 数可以更多，但不能遗漏 Q/K Norm；Q 宽度 2048、KV 宽度 1024 的首个配置应由规则推导而非补丁。
4. 所有 Prefill 和 Decode step 都由真实 backend 执行并 DONE，context 从 P 递增至 P+G，各层 KV 地址和有效长度独立；不以测一层或一步再乘次数替代。
5. 有效 payload、AXI bytes、辅助参数、临时区、模型范围和精度映射可对账；LMHead 使用 last/all_positions 的范围明确；无零成本未建模算子。
6. 结果定位为性能仿真，不宣称生成了正确 token 或验证了量化精度。未校准内核能耗单列，不能把缺失值当作完整模型 energy=0。
7. 首版使用 Qwen3 Dense adapter，但核心 IR/kernel/runtime 不依赖某个固定模型名称；第二个 adapter 可以复用同一编译与执行路径。
8. 同一请求的全部 Prefill/chunk 与 Decode 在同一 PerfSession 完成；会话生命周期、状态保留、core/DDR 相位及阶段/累计统计通过第 6.6 节验收。逐步重建进程的结果仅列为冷启动对照。

### 12.5 配置与调度回归矩阵

| 维度 | 实施要求 |
|---|---|
| 显式 Parallel 验证 preset | `P=32`、8×256 KiB SPM、ACC data 总量=1 MiB/2 bank、bias 总量=256 KiB/1 bank、队列深度 3、inst cache 256 words、core 1 GHz、DDR 名义 3.2 GHz、AnalyticalDDR。这是配置示例选取的验证点，不是 Config 全部默认值；以有效配置和实际 tick 为准。 |
| Qwen3 Prefill | batch=1，完整模型，P=32/128/512；额外覆盖 P=1/31/33/129 尾块；分别记录 last/all_positions logits 范围。 |
| Qwen3 Decode | 从已完成 Prefill 的逻辑 KV 状态开始，G=1/4/16；每步真实执行全层，并检查 cache_position、visible length 和增长的流量。 |
| 架构 shape | 显式 head_dim、Hq≠Hkv、Q 宽度≠hidden_size、绑定 embedding/lm_head、K 维超过 2048 和 M/N 超过 32 的多 tile lowering。 |
| 新 kernel | norm 整行归约、head norm、RoPE 系数/position、双输入 SwiGLU、causal/padding mask、Softmax 跨 tile；每项有独立延迟/访问/资源基线。 |
| 队列深度 | Qwen3 kernel、真实层和完整请求覆盖深度 1..5；每个配置重新验证调度。TIMEOUT/资源冲突均需诊断，不能视为模型支持成功。 |
| 阵列尺寸 | 16/32/64 分别重新选 tile 和 guard；首版 M/N tile≤min(P,32)，64×64 配置不等于可编码 64×64 tile。记录相同程序与重新 tiling 的区别；未经验证的配置标记 unsupported。 |
| SPM bank 容量 | 64/128/256 KiB 分别运行固定指令和重新编译两组；后一组记录 K tile、DMA 数、bank 地址变化，不将 tiling 收益全归因于容量。 |
| DMA 交互 | ACC MVIN/MVOUT 与 SPM 写回竞争 DMA1，DMA2/3 轮询，队列反压、同类型响应交错；request_id 或 stall 修复后单独再建基线。 |
| 前端/存储 | 混合旧/扩展指令、cache/队列满、续行保持队列、非对齐 AXI、4 KiB/256-beat 边界、尾块/跨 bank、32-bit 地址溢出负例。 |
| 状态边界 | EOF 后 core/DDR 全部排空、达到周期预算、异常、不完整包；同会话重复加载、keep 数据代次、时钟相位、reset 与新建会话等价，不能复用过期 valid 位。 |
| keep/bias | A/B/bias 重复消费、最后一次释放、未读取 bias 的 SA 保持状态、无生产者的伪 valid、跨 launch 入口不匹配、残留 valid 阻塞下一次加载。 |
| AnalyticalDDR | 所有阶段固定使用该后端；验证二维 stride、burst/4 KiB 边界、流量、请求排队/响应匹配、完全排空和同会话重入。 |

以上为实施阶段的验收要求。各项结果须附运行配置、程序指纹、完成状态与测量产物；
尚未执行的测试不得标记为通过。RTL 对照作为独立验证记录，不与软件栈回归结果混用。

### 12.6 指标定义与优化目标

优化首先满足无冲突、依赖合法和 DONE，再比较总完成时间、流量与能耗。
各项指标采用以下定义，并明确模型覆盖范围：

- `module_util=busy_cycles/core_cycles` 表示模块忙比例，SA busy 不等于 PE 有效 MAC 利用率。DMA 利用率分别报告，三个 DMA 的利用率总和可超过 100%；需要平均值时显式除以 3。
- 时间以会话导出的模拟时间边界和有效时钟为准：`effective_core_period_ps=base_tick_ps*ticks_per_period[core]`，名义 GHz 不能覆盖整数 ps/tick 舍入。平均带宽为 `bytes/device_time_ns`，以十进制 GB/s 报告；只有有效周期与名义频率一致时才等于 `bytes/core_cycles*npu_freq_GHz`。DDR cycle/base tick/core cycle 不混用；logical payload 与 AXI modeled bytes 分列。
- 当前 DDR read/write 字节计数来自 DMA local-side beat 的记账，不是直接观察 AnalyticalDDR 物理 channel 服务；瞬时峰值是该模型的计数口径，不能据此声称真实 DRAM 获得三倍通道带宽。AnalyticalDDR 当前一次处理一个 active request，多 DMA 不等于多 DDR request 同时服务。
- 分别输出 `decoder_backpressure`、queue full、队首数据等待、资源等待和 DMA 等响应周期，明确分母及是否可重叠。队列阻塞比例高与较短总周期可能同时出现，不能只最小化 queue_full_ratio。
- `energy=Σ power_mW(state)*cycle_ns`，单位 pJ；`energy_uJ=energy_pJ/1e6`。当前 PowerAnalyzer 使用名义 `cycle_ns=1/npu_freq`，重构后的主报告应使用有效 core 周期；二者不一致时分列 legacy 与有效时钟口径，不能默默混算。包括已建模模块的 idle 功耗；当前覆盖三 DMA、SA、ACC、SFU，不自动等于含 DRAM/SPM/互联的整芯片功耗。
- `EDP=energy*time`；和同配置基线比较 `speedup=T_base/T_new`、`NEDP=(E_new*T_new)/(E_base*T_base)`。跨频率比较使用时间而非裸周期，跨阵列比较须注明功耗表的校准/缩放策略，不能假定原表自动适用于新硬件。
- Roofline 标明逻辑 MAC/FLOP 计数约定（例如 1 MAC=2 ops）、访问量口径和理论/实测带宽来源；SFU 不混为等价矩阵 MAC。分别分析 Qwen3 的 RMSNorm/RoPE/SwiGLU、矩阵计算和 Decode 小 M/KV 流量，不用旧 GeLU 负载预设瓶颈。

同一图可选 latency-first、energy-first 或 EDP 目标，但首版先使用固定可复现策略，
后续通过真实 backend 反馈选择候选 tiling/placement/schedule，不把静态估算当作最终性能。
未启用或未校准的功耗项用 unavailable/null 及原因表示，不能将其填为 0 后计算完整模型
EDP。细分指标附单位、计数来源和覆盖范围；累计峰值与各阶段峰值分别采样。

## 13. 分阶段实施计划

### 13.1 阶段 0：Qwen3 模型前端与范围冻结

- 建立 `sw/pimu`、模型注册接口、Qwen3Adapter 和 config/checkpoint 元数据导入；
- 用真实 Qwen3-0.6B 配置构建完整语义图，覆盖 28 层、head_dim 推导、Q/K Norm、RoPE、SwiGLU、final norm、lm_head、权重共享和每层 KV；
- 定义 RequestSpec、precision policy、coverage schema、模型/后端版本指纹及资源限制；
- 冻结 Program、PerfSession、Backend、launch 边界和状态保留契约，建立原仿真器周期/流量基线；
- 输出 model.json、weight manifest、shape/算子覆盖报告和缺失 kernel 清单。

完成标准：无需输入任何旧 txt 即可从模型目录生成完整 Qwen3 图；图中没有简化替代或遗漏。

### 13.2 阶段 1：仿真器核心库与持久会话重构

- 分离核心库和 CLI，使构造 PerfSession 不依赖指令文件；
- 引入 Program 与统一指令源，支持内存加载、文件导入导出和长指令流，保留前端取指时序；
- 将 base tick、core/DDR 相位、模块状态及累计计数归属于会话，定义安全程序切换、reset 和完全排空条件；
- 实现第 6 节 C ABI、有效配置/能力导出、`sw/native/` Python 绑定、PerformanceBackend 和结构化状态/统计；
- 完成旧/新 CLI、内存输入、连续 launch、超时/错误、reset、AnalyticalDDR 和日志关闭时的回归；DDR 模式固定为 AnalyticalDDR。

完成标准：Python 可以在同一会话中连续提交多个合法程序，状态和时钟连续、统计可对账；
reset 后与新建会话等价。纯接口重构保持原周期基线；必要的行为修复单独记录版本。

### 13.3 阶段 2：通用 compiler/runtime 基础链路

- 实现 ISA word/packet、schema、encoder/disassembler/validator 和 target capability；
- 建立通用模型/算子 IR、`pimu.language` builder、tiler、DRAM/SPM/ACC allocator 和调度器；
- Runtime 接入持久会话 Backend，提供程序绑定/缓存、独立产物目录和阶段/累计报告，保留 CLI 兼容适配器；
- 用小型 Qwen3 shape 的 Linear/attention GEMM/residual 验证实际发射；已有 txt 直接用于兼容回归，不重写其模型生成器。

完成标准：模型节点能降低为可执行的 Parallel PIMU kernel，并可追踪算子→指令→统计；不宣称此时已覆盖完整模型。

### 13.4 阶段 3：Qwen3 必需 kernel 与后端能力

- 冻结并实现第 10 节扩展格式和 decoder/queue/EventSync/SFU 的全链路支持；
- 实现 RMSNorm/head norm、RoPE、SiLU、二元逐元素、scale/mask 和完整 reduction 性能内核；
- 落实第 8.4 节的存储/计算 dtype 与转换路径，将必要 convert/requantize 纳入内核和 coverage；
- 校验 gamma、系数、两输入、临时区、dtype、SPM 端口和 DMA 流量，记录 latency/energy 模型依据；
- 审计并修复会影响 Qwen3 正确调度的响应身份和资源保护，保留独立变更记录；
- 引入有限队列、数据代次、跨 DMA/DRAM 依赖和多输入进展检查。

完成标准：首个 Qwen3 配置所需 kernel 全部可执行，capability 真实反映能力；不靠 LayerNorm、GeLU 或零成本 host 回调补洞。

### 13.5 阶段 4：真实 Qwen3 单层集成

- 从完整模型提取真实一层配置、参数绑定及 Prefill/Decode 图用于集成测试；
- 包含 Q/K Norm、RoPE、GQA、SwiGLU、两路 residual 和该层 KV append/read；
- 覆盖完整归约、head/layout 变化、tail tile、不同上下文长度和队列深度；
- 逐算子、逐 tile 对照 shape、有效字节、指令范围和实际后端完成状态；
- 在同一会话中串接该层 Prefill/Decode，验证真实 bank 状态、KV 更新和阶段边界。

完成标准：真实结构单层在两种阶段完成，报告标记 layer_only/partial；这是全模型集成步骤，不是最终交付。

### 13.6 阶段 5：完整 Qwen3 Prefill 与多步 Decode

- 展开并实际仿真全部层，加入 embedding、final norm、lm_head，按配置复用 tied weights；
- 实现每请求/每层 KV、position、visible length、增长的 Decode shape 和模板重绑定；
- 支持长指令流流式 emit、层/算子 trace、coverage 与全请求统计；
- 在同一会话执行第 12 节完整请求矩阵，审计地址容量、非空功耗覆盖、阶段边界与冷/热启动范围。

完成标准：从本地 Qwen3 模型目录与请求参数自动生成并在同一会话执行完整性能 workload，
unsupported=0，所有阶段 DONE；不以单层/单步周期乘法替代。

### 13.7 阶段 6：性能优化与后续模型

- 在相同完整 Qwen3 图/precision 下优化 keep/prefetch、bank reuse、跨层调度、GQA/KV 与实际 kernel fusion；
- 通过 ModelAdapter registry 接入其他模型，新增能力优先复用通用 IR/kernel；
- 用第二个仅组合已有能力的 adapter 验证新增模型不需要修改通用 compiler/runtime/ISA；
- 在已验收的持久会话上扩展设备侧循环、program chaining、跨 launch 重叠、batching 和更长上下文；
- 4-bit、地址窗口、MoE 或多模态能力使用独立格式/性能契约和测试，不静默改变既有 scope。

完成标准：Qwen3 全模型回归持续通过；新增模型有明确 coverage、能力边界和独立验收。

## 14. 第一轮实际编码顺序

开始实现时按以下顺序提交小变更：

1. 冻结模型/请求范围及 Program/PerfSession/Backend 契约，保存原配置、指令与周期基线；
2. 拆分核心库/CLI，引入统一指令源，完成文件与内存输入等价回归；
3. 实现持久会话、时钟相位保留、安全程序切换、reset、可靠完成状态与结构化统计；
4. 提供 C ABI、Python 绑定及 Backend，通过同会话连续运行和错误恢复验收；
5. 定义 ModelSpec/RequestSpec/ModelAdapter registry，加入 Qwen3 config/weight metadata importer、完整语义图及 precision/capability/coverage 校验；
6. 实现 ISA encoder、通用 IR/`pl`、allocator、tiler/scheduler 和会话 Runtime，跑通模型导出的基础 kernel；
7. 实现 Qwen3 所需的性能单元/扩展指令、参数搬运与资源验证，处理并行响应/依赖问题；
8. 集成真实 Qwen3 单层 Prefill/Decode，再展开全部层、embedding、final norm 与 lm_head；
9. 在同会话完成多步 Decode/KV 状态管理、全请求指标和第 12 节验收矩阵；
10. 在完整模型基础上做并行优化，并通过 adapter/kernel registry 接入后续模型。

软件栈、绑定代码和运行产物位于 `PIMU-Sim/sw/`；C++ 核心库、会话、取指与时钟推进在
现有 `include/`、`src/` 及构建规则中重构。必需的算子/指令性能扩展及并行修复独立记录并
重建基线。保留旧配置、旧 benchmark 和现有单参数 CLI 的兼容路径，但不将它们的
生成器重写作为模型支持的前置交付。
