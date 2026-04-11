# H20 预训练集群 — 实习生上手文档

> 更新: 2026-04-11 | 目标读者: 需要在 H20 集群上开发初始化算法的实习生

---

## 1. 集群概览

两台火山引擎 H20 机器，共享 `/physis` 存储 (49T GPFS, 剩 23T)。

| 名称 | SSH 命令 | GPU 状态 | 用途 |
|------|---------|---------|------|
| **H20-pretrain** | `ssh -i ~/.ssh/volcengine_h20 -p 45650 root@101.47.156.239` | 8×H20 空闲 | **你的开发机** |
| **H20-midtrain** | `ssh -i ~/.ssh/volcengine_h20 -p 34098 root@101.47.156.239` | 训练中 (勿动) | Mid-training 实验 |

> SSH 必须带 `-i ~/.ssh/volcengine_h20` (ed25519 密钥)

### 硬件
- 8× NVIDIA H20 (98GB HBM3 每卡)
- 1.7 TiB 系统内存
- `/physis` 共享存储 (两台机器都能访问)

---

## 2. 快速开始: 克隆 VeOmni 代码

VeOmni 是我们的预训练框架 (基于 ByteDance VeOmni fork), 代码仓库在 GitHub:

```bash
# 1. SSH 登录 pretrain 机器
ssh -i ~/.ssh/volcengine_h20 -p 45650 root@101.47.156.239

# 2. 克隆代码到 /physis (大磁盘, / 根分区只有 6.7GB)
cd /physis
git clone git@github.com:Gaiejj/VeOmni.git
cd VeOmni

# 3. 搭建 Python 环境 (需要 Python 3.11)
python3.11 -m venv /physis/veomni_venv
source /physis/veomni_venv/bin/activate

# 4. 安装依赖
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -e .
pip install flash-attn transformers diffusers accelerate sentencepiece protobuf

# 5. 验证
python -c "import veomni; import torch; print('OK', torch.__version__)"
```

### 仓库结构 (重点文件)

```
VeOmni/
├── veomni/
│   ├── models/
│   │   ├── transformers/wan/
│   │   │   ├── modeling_wan.py            # ★ Wan DiT 模型 + 初始化代码
│   │   │   └── config_wan.py
│   │   └── diffusers/wan_t2v/
│   │       ├── wan_transformer/           # WanTransformer3DModel (diffusers格式)
│   │       └── wan_condition/             # 条件模型 (VAE + T5 + Scheduler)
│   ├── trainer/
│   │   ├── dit_trainer.py                 # ★ DiT 训练器主逻辑
│   │   ├── callbacks/
│   │   │   └── ema_callback.py            # EMA (FSDP2 兼容)
│   │   └── stage_controller.py            # 多阶段渐进式训练
│   ├── data/
│   │   └── multimodal/dit/               # DiT 数据加载 (online/offline)
│   └── distributed/                       # FSDP2/HSDP 分布式
├── configs/
│   ├── dit/                               # ★ 训练配置 (yaml)
│   │   ├── wan_pretrain_80h20.yaml        # 80卡预训练配置
│   │   ├── wan_1.3b_real_data_8gpu.yaml   # 单机 8 卡真实数据
│   │   └── wan_smoke_test_8gpu.yaml       # smoke test
│   └── model_configs/wan/                 # 模型规模配置
│       ├── want2v_0.3b.json               # 0.3B (dim=1024, 20层)
│       ├── want2v_0.6b.json               # 0.6B (dim=1536, 24层)
│       ├── want2v_1.3b.json               # 1.3B (dim=2560, 32层)
│       ├── want2v_3b.json                 # 3B
│       └── want2v_7b.json                 # 7B (dim=3584, 36层)
└── tasks/
    └── train_dit.py                       # ★ 训练入口脚本
```

### 启动训练

```bash
# 单机 8 卡 smoke test
cd /physis/VeOmni
source /physis/veomni_venv/bin/activate
torchrun --standalone --nproc-per-node=8 tasks/train_dit.py configs/dit/wan_smoke_test_8gpu.yaml

# 真实数据训练 (需先下载模型和数据)
torchrun --standalone --nproc-per-node=8 tasks/train_dit.py configs/dit/wan_1.3b_real_data_8gpu.yaml
```

---

## 3. 初始化体系 (你的开发重点)

### 3.1 VeOmni 中已实现的初始化

**文件**: `veomni/models/transformers/wan/modeling_wan.py` (lines 558-607)

```python
# 两阶段初始化:
# Stage 1: _init_weights() — 全局 xavier_uniform_ + 零偏置
#   - 所有 nn.Linear: xavier_uniform_
#   - 所有 nn.Conv3d: xavier_uniform_
#   - 所有 bias: zeros_

# Stage 2: _zero_init_residual_outputs() — 零初始化残差输出:
#   - self_attn.o.weight/bias → 0
#   - cross_attn.o.weight/bias → 0
#   - ffn[2].weight/bias → 0 (FFN 输出层)
#   - head.head.weight/bias → 0 (最终头)
# 原理: 每个 DiTBlock 初始为恒等映射 (DiT 论文, Peebles & Xie 2023)
```

**Timestep 采样** (`veomni/models/diffusers/wan_t2v/wan_condition/modeling_wan_condition.py`):
- `uniform`: 均匀采样离散 scheduler timesteps
- `logit_normal`: SD3 风格, 集中在中等噪声水平
- `cosmap`: 余弦映射采样

**Loss 权重** (同上文件):
- `none`: 均匀权重
- `min_snr`: 信噪比加权, gamma=5.0
- `cosmap`: 逆 sigma 加权

### 3.2 DreamDojo 中的初始化 (参考)

H20-midtrain 机器上的 DreamDojo 代码有更丰富的初始化模式可以参考:

**基座 DiT** (`minimal_v4_dit.py`):
- `trunc_normal_` with dimension-dependent std
- GPT-2 风格深度缩放 (layer2 std 随深度递减)

**Action 模块** (`temporal_action_dit.py`):
- **零初始化门控 (Zero-Init Gates)**: 新模块通过门控连接主干, 初始值=0
- kaiming_uniform_ for Linear, normal_(std=0.02) for Embedding

**MotionFieldAdapter** (`motion_field_adapter.py`):
- gate = 1e-4 (非零, 避免死梯度)
- kaiming_normal_ for Conv2d (fan_out)
- Final conv: kaiming then mul_(0.01) 缩小初始输出

### 3.3 关键设计问题 (待你研究)

| 问题 | 说明 |
|------|------|
| trunc_normal vs xavier vs kaiming | 哪种适合从零预训练 Wan2.1 DiT? |
| 零初始化门控的开放策略 | 线性 vs 学习 vs cosine? |
| Timestep 采样 | uniform vs logit_normal 在预训练阶段的效果差异 |
| Loss weighting | min-SNR vs cosmap 在大规模训练的稳定性 |
| 深度缩放 | 是否需要 GPT-2 风格的 depth-dependent std? |

---

## 4. 可用资源 (H20 集群)

### 模型权重 (`/physis/models/`)

| 模型 | 大小 | 说明 |
|------|------|------|
| Cosmos-Predict2.5-2B | 70G | 2B 基座 |
| Cosmos-Predict2.5-14B | 54G | 14B 基座 |
| Wan-AI (Wan2.1 models) | 139G | Wan2.1 T2V/I2V 全系列 |
| t5-11b | 85G | 文本编码器 |

### 数据集 (`/physis/datasets/`)

| 数据集 | 类型 | 数量 |
|--------|------|------|
| EgoDex4DreamDojo | 机器人自我中心 | 338K mp4 |
| Physics-aware-videos | 物理现象 | 83K mp4 |
| WISA-80K | 世界交互 | 81K mp4 |
| Kinetics-700 | 通用动作 | 174K mp4 |
| GR1/G1/AgiBotWorld | 机器人操作 | 各数万 |

---

## 5. 注意事项

- **不要动 midtrain 机器** — 正在跑训练
- `/` 根分区只有 6.7GB — 所有大文件放 `/physis/`
- 输出目录设为 `/physis/cosmos_logs/` (不要用默认 overlay)
- HuggingFace 离线模式 — 模型已在 `/physis/models/`
- 开发时建议用小模型 (0.3B/0.6B) + smoke test 快速迭代

---

## 6. 参考文献

- DiT 初始化: Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023)
- GPT-2 深度缩放: Radford et al. (2019), arxiv 1908.11365
- Zero-init residual: 每个 block 初始为恒等映射, 训练更稳定
- Gate=1e-4 (非零): 避免死梯度, 来自 DreamDojo MotionFieldAdapter 的实践经验
- Flow Matching shift: `u = shift * u / (1 + (shift - 1) * u)` 控制噪声分布
