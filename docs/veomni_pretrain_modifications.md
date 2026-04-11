# VeOmni DiT 预训练改造说明

> 更新: 2026-04-11 | 面向: 参与预训练初始化研究的开发者

---

## 0. 背景

VeOmni 是字节跳动 Seed 团队的多模态分布式训练框架 (AAAI 2026)。原始版本**不支持从零预训练 DiT**——只支持微调场景, 缺少以下关键能力:

- 图+视频混合训练 (多分辨率)
- 多阶段渐进式训练 (256px → 480p → 720p)
- 从零训练的权重初始化
- Timestep 采样策略 & Loss 加权
- EMA (指数移动平均)

我们在 2 天内用 Claude Code 完成了改造, 涉及 34 个文件 (+1475/-64 行), 并在 lyg 集群 8×A100 上验证了端到端训练。

---

## 1. 改动全景

### 1.1 改动文件索引

| 类别 | 文件 | 改动类型 | 验证状态 |
|------|------|---------|---------|
| **模型初始化** | `veomni/models/transformers/wan/modeling_wan.py:558-607` | 新增 | **已验证** |
| **Timestep 采样 + Loss 权重** | `veomni/models/diffusers/wan_t2v/wan_condition/modeling_wan_condition.py:136-203` | 新增 | **uniform 已验证**, logit_normal/cosmap 未验证 |
| **条件模型配置** | `veomni/models/diffusers/wan_t2v/wan_condition/configuration_wan_condition.py:30-33` | 新增字段 | 已验证 |
| **Diffusers 模型 forward** | `veomni/models/diffusers/wan_t2v/wan_transformer/modeling_wan_transformer.py:314` | Bug fix | **已验证** |
| **视频解码** | `veomni/data/multimodal/video_utils.py:27-41` | 新增 decord 后备 | **已验证** |
| **序列长度计算** | `veomni/utils/helper.py:118-122, 247-249` | Bug fix | **已验证** |
| **训练器** | `veomni/trainer/dit_trainer.py` (多处) | 集成新功能 | 部分已验证 |
| **EMA 回调** | `veomni/trainer/callbacks/ema_callback.py` | **全新文件** | **未验证** |
| **多阶段控制器** | `veomni/trainer/stage_controller.py` | **全新文件** | **未验证** |
| **Bucket 采样器** | `veomni/data/diffusion/bucket_sampler.py` | **全新文件** | **未验证** |
| **训练配置** | `configs/dit/*.yaml` | 全新 | smoke test 已验证 |
| **模型规模配置** | `configs/model_configs/wan/*.json` | 全新 | 0.3B-7B |

### 1.2 Git 提交历史

```
fca7d9c [docs] add lyg cluster development guide
e3d8e28 [docs] add H20 pretrain cluster onboarding guide
129413b [dit, data, config] feat: add smoke test infra, decord video backend, and online training fixes
c07b1fb [dit, trainer, data] fix: resolve 14 bugs in pretrain infrastructure
89ac327 Merge branch 'pretrain/multi-node-and-configs'
```

---

## 2. 已验证可用的功能

以下功能经过端到端训练验证, 可以直接使用:

### 2.1 从零训练 (Xavier + Zero-Init)

**文件**: `veomni/models/transformers/wan/modeling_wan.py:558-607`

两阶段初始化:
1. **Stage 1** `_init_weights()`: 所有 `nn.Linear` 和 `nn.Conv3d` 用 `xavier_uniform_`, 偏置归零
2. **Stage 2** `_zero_init_residual_outputs()`: 残差路径输出层全部归零
   - `self_attn.o` (自注意力输出)
   - `cross_attn.o` (交叉注意力输出)
   - `ffn[2]` (FFN 第二层)
   - `head.head` (最终输出头)

**原理**: DiT 论文 (Peebles & Xie, 2023) 的标准做法——每个 block 初始时是恒等映射。

**验证结果**:
- toy 模型 (2层): loss 1.51 → 1.00 (理论最优), 正常收敛
- 1.3B 模型 (dummy data): loss 1.51 → 1.00, 50步, 31s/步
- 1.3B 模型 (真实视频): loss 0.92 → 0.11, 50步, 11s/步

### 2.2 Online Training (在线编码)

**流程**: 原始 MP4 → decord 解码 → UMT5-XXL 文本编码 + VAE 视频编码 → Flow Matching → DiT 训练

**关键组件**:
- `WanTransformer3DConditionModel.get_condition()`: 编码 prompt (T5) + 视频 (VAE)
- `WanTransformer3DConditionModel.process_condition()`: 添加噪声 + 采样 timestep
- `WanTransformer3DModel.forward()`: MSE loss 计算

**验证**: 用 crush-smol 数据集 (47 个液压机视频) 完成端到端训练。

### 2.3 FSDP2 分布式训练

- 8×A100 上验证通过
- 必须设置 `init_device: meta` (FSDP2 硬性要求)
- Flash Attention 正常工作
- Gradient checkpointing 正常

### 2.4 Decord 视频解码

**文件**: `veomni/data/multimodal/video_utils.py:27-41`

原始 torchcodec 与 CUDA 12.4 不兼容, 增加了 decord 后备方案:
```python
_USE_DECORD = True  # 自动检测, torchcodec 不可用时 fallback
```

### 2.5 Bug Fixes (14 个, 全部已验证)

| Bug | 文件 | 修复 |
|-----|------|------|
| `forward()` 不接受 `loss_weight` | `modeling_wan_transformer.py:314` | 添加 `**kwargs` |
| online 训练 `latents` key 不存在 | `helper.py:118-122` | 添加 key 存在性检查 |
| 指标计算除零 | `helper.py:247-249` | `max(denominator, 1)` |
| min-SNR 分母为零 | `modeling_wan_condition.py:195,197` | `sigma.clamp(min=1e-6)` |
| cosmap epsilon 太小 | `modeling_wan_condition.py:200` | `1e-6` → `1e-3` |
| loss weight 爆炸 | `modeling_wan_condition.py:201` | `clamp(max=10.0)` |
| `init_device: cuda` 崩溃 | configs | 改为 `init_device: meta` (FSDP2 要求) |

---

## 3. 未验证 / 潜在有问题的功能

> **重要**: 以下功能代码已写好, 但未经过实际训练验证。使用前需要仔细测试。

### 3.1 EMA 回调 (高风险)

**文件**: `veomni/trainer/callbacks/ema_callback.py` (163 行, 全新)

- 维护模型参数的指数移动平均 shadow copy
- 通过 FSDP2 DTensor 的 `to_local()` 操作本地 shard
- 支持线性 warmup + 恒定 decay, 可配置更新频率

**风险点**:
- `to_local()` 在 DTensor 上的行为未实测
- EMA state 的 checkpoint save/load 未验证
- 内存占用翻倍 (每个参数存两份)

### 3.2 多阶段渐进训练 (高风险)

**文件**: `veomni/trainer/stage_controller.py` (117 行, 全新)

- 5 个阶段: 256px 图片 → 480p 图片 → 480p 短视频 → 480p 长视频 → 720p 视频
- 每阶段可配分辨率、帧数、学习率、batch size

**风险点**:
- 阶段切换时的 dataloader 重建逻辑未验证
- 学习率 scheduler 重置是否正确未知
- 与 FSDP2 的交互 (batch size 变化时 shard 行为) 未测

### 3.3 Bucket 采样器 (中风险)

**文件**: `veomni/data/diffusion/bucket_sampler.py` (186 行, 全新)

- 按分辨率分桶, 同 batch 内分辨率一致
- 支持 256/480/720p 及多种宽高比
- 分布式安全: per-rank 截断, drop 不完整 batch

**风险点**:
- 跨 rank 的 bucket 分配一致性未验证
- 极端宽高比 (如 16:9 vs 1:1 混合) 的 padding/效率未测

### 3.4 非 Uniform Timestep 采样

**文件**: `modeling_wan_condition.py:157-176`

- `logit_normal`: SD3 风格, 集中在中等噪声水平
- `cosmap`: 余弦映射, 集中在中间区域

**注意**: 只有 `uniform` 在训练中实际跑过。`logit_normal` 和 `cosmap` 的代码已实现, 逻辑看起来正确, 但未验证其对训练收敛的影响。

### 3.5 Loss Weighting

**文件**: `modeling_wan_condition.py:180-203`

- `min_snr`: 信噪比加权, gamma=5.0
- `cosmap`: 逆 sigma 加权

同上, 只有 `none` (均匀权重) 经过验证。

---

## 4. 你需要修改的地方 (初始化 & 架构)

### 4.1 关键认知: 两个 "WanTransformer3DModel"

VeOmni 中存在两个模型格式:

| 格式 | 文件 | 用途 |
|------|------|------|
| **transformers 格式** | `veomni/models/transformers/wan/modeling_wan.py` | 初始化代码在这里, 但**不直接参与训练** |
| **diffusers 格式** | `veomni/models/diffusers/wan_t2v/wan_transformer/modeling_wan_transformer.py` | **实际训练用的模型** |

当 `init_from_pretrained: null` (从零训练) 时:
1. `modeling_wan.py` 中的 `_init_weights()` 和 `_zero_init_residual_outputs()` 被调用
2. 初始化后的权重被加载到 diffusers 格式的模型中
3. 训练循环使用 diffusers 格式的 `forward()`

**结论**: 修改初始化策略, 主要改 `modeling_wan.py`; 修改模型架构, 需要同时改两个文件。

### 4.2 初始化修改入口

**文件**: `veomni/models/transformers/wan/modeling_wan.py`

```python
# === 你需要修改的函数 ===

def _init_weights(self, module):                    # line 566
    """全局初始化策略 — 替换 xavier_uniform_ 为你的方案"""
    # 当前: xavier_uniform_ for Linear/Conv3d
    # 可选: trunc_normal_, kaiming_uniform_, 带深度缩放的 normal_ 等
    pass

def _zero_init_residual_outputs(self):              # line 582
    """残差路径初始化 — 可改为非零门控"""
    # 当前: 全零初始化
    # 可选: 1e-4 小值初始化 (DreamDojo 经验), 可学习门控
    pass
```

**实验建议**:

1. **trunc_normal vs xavier**: trunc_normal 在 ViT 中效果好, 但 DiT 的标准是 xavier
2. **深度缩放**: GPT-2 风格 `std = base_std / sqrt(2 * num_layers)`, 可让深层输出更小
3. **非零门控**: 用 `1e-4` 而非 `0` 初始化残差输出, 避免死梯度
4. **模块级初始化**: 对 attention, FFN, embedding 分别用不同策略

### 4.3 Timestep 采样 & Loss 权重修改入口

**文件**: `veomni/models/diffusers/wan_t2v/wan_condition/modeling_wan_condition.py`

```python
def _sample_timesteps(self, batch_size, device, dtype):     # line 136
    """添加新的采样策略"""
    # 在 elif 链中加新分支即可

def _compute_loss_weight(self, timestep):                    # line 180
    """添加新的 loss 加权策略"""
    # 同上
```

**配置**: `veomni/models/diffusers/wan_t2v/wan_condition/configuration_wan_condition.py`
```python
# line 30-33, 添加新的配置字段
timestep_sampling: str = "uniform"
logit_normal_mean: float = 0.0
logit_normal_std: float = 1.0
loss_weighting: str = "none"
```

### 4.4 架构修改入口

如果需要修改 DiT block 结构 (如加 gate, 加 adapter):

**Diffusers 格式** (训练实际用这个):
- `modeling_wan_transformer.py` 中的 `WanTransformer3DModel.__init__()` (line 284)
- 可在 `self.blocks` 循环中 patch 每个 block

**Transformers 格式** (初始化在这里):
- `modeling_wan.py` 中的 `WanModel.__init__()` (line ~520-560)
- `WanDiTBlock` 类定义 (line ~200-350)

### 4.5 模型规模配置

| 配置 | 参数量 | 推荐用途 |
|------|--------|---------|
| `configs/model_configs/wan/want2v_0.3b.json` | 0.3B | 快速迭代初始化实验 |
| `configs/model_configs/wan/want2v_0.6b.json` | 0.6B | 中等规模验证 |
| `configs/model_configs/wan/want2v_1.3b.json` | 1.3B | 完整验证 |

**建议**: 用 0.3B/0.6B 做初始化实验 (训练快, 显存小), 确认效果后再上 1.3B。

---

## 5. 快速实验流程

```bash
# 1. 在 modeling_wan.py 中修改 _init_weights / _zero_init_residual_outputs

# 2. 用 toy 模型 smoke test (5 分钟)
torchrun --standalone --nproc-per-node=8 \
  tasks/train_dit.py configs/dit/wan_smoke_test_8gpu.yaml
# 期望: loss 从 ~1.5 降到 ~1.0 (随机数据理论最优)
# 如果 NaN 或不收敛 → 初始化有问题

# 3. 用 1.3B 模型验证规模效应 (25 分钟)
torchrun --standalone --nproc-per-node=8 \
  tasks/train_dit.py configs/dit/wan_1.3b_smoke_test_8gpu.yaml

# 4. 用真实视频验证端到端 (10 分钟)
torchrun --standalone --nproc-per-node=8 \
  tasks/train_dit.py configs/dit/wan_1.3b_real_data_8gpu.yaml
```

### 如何对比不同初始化方法

1. 在 yaml 中添加自定义字段 (如 `init_method: "trunc_normal"`)
2. 在 `modeling_wan.py` 的 `_init_weights` 中读取 config 选择策略
3. 固定随机种子, 跑同样步数, 比较 loss 曲线
4. 关注: 初始 loss 值、收敛速度、是否出现 NaN

---

## 6. 参考文献

| 主题 | 来源 | 说明 |
|------|------|------|
| DiT 初始化 | Peebles & Xie (2023) | Xavier + zero-init residual, 当前实现的基础 |
| GPT-2 深度缩放 | Radford et al. (2019) | `std / sqrt(2N)`, 待实验 |
| Flow Matching | Lipman et al. (2023) | Wan2.1 的训练范式 |
| SD3 Timestep | Esser et al. (2024) | logit_normal 采样策略 |
| min-SNR | Hang et al. (2023) | 信噪比 loss 加权 |
| Zero-init Gates | DreamDojo 经验 | gate=1e-4 避免死梯度 |
| Wan2.1 | Wan-AI (2025) | 1.3B/14B Text-to-Video DiT |
