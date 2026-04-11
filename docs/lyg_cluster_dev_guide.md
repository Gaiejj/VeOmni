# VeOmni DiT 预训练 — lyg 集群开发指南

> 更新: 2026-04-11 | 面向: 在 lyg 集群上开发初始化算法的实习生

---

## 1. 服务器选择

| 服务器 | GPU | 状态 | 推荐 |
|--------|-----|------|------|
| **lyg0203** | 8× A100-80GB | 空闲, venv 就绪 | **首选开发机** |
| **lyg0375** | 8× A100-80GB | 空闲, venv 就绪 | 备选 |
| lyg0337 | 8× A100-80GB | 被占用 (他人训练中) | 不可用 |
| lyg0378 | 8× A100-80GB | 被占用, 无 venv | 不可用 |

> 注: GPU 占用情况会变化, 用前先 `nvidia-smi` 确认。

---

## 2. 快速开始

```bash
# 1. SSH 到开发机
ssh lyg0203   # 或 lyg0375

# 2. 代码已在共享存储 (所有 lyg 机器都能访问)
cd /mnt/users/jiayi/jiayi/vla/VeOmni

# 3. 激活环境 (venv 在本地磁盘, 每台服务器独立)
source /tmp/jiayi_veomni_venv/bin/activate

# 4. 验证环境
python -c "
import torch; print('torch:', torch.__version__)
import flash_attn; print('flash-attn:', flash_attn.__version__)
import veomni; print('veomni OK')
import diffusers; print('diffusers:', diffusers.__version__)
"

# 5. 跑个 smoke test (从零训练 toy 模型, ~5分钟)
torchrun --standalone --nproc-per-node=8 \
  tasks/train_dit.py configs/dit/wan_smoke_test_8gpu.yaml
```

### 如果 venv 不存在 (首次在新服务器上)

```bash
# 创建 venv (放 /tmp 是因为本地磁盘比 NFS 快 10x)
/mnt/users/jiayi/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11 \
  -m venv /tmp/jiayi_veomni_venv
source /tmp/jiayi_veomni_venv/bin/activate

# 安装核心包
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu129
pip install transformers==4.57.3 diffusers==0.36.0 accelerate sentencepiece protobuf rich ftfy
pip install -e /mnt/users/jiayi/jiayi/vla/VeOmni

# flash-attn (有预编译备份, 比从源码编译快很多)
cp -r /mnt/users/jiayi/.cache/veomni/flash_attn_backup/flash_attn* \
  /tmp/jiayi_veomni_venv/lib/python3.11/site-packages/

# decord (视频解码, 替代 torchcodec)
pip install decord av
```

---

## 3. 已有模型和数据

所有 lyg 服务器共享 `/mnt/users/jiayi/` (CephFS, 30T, 剩 9.2T)。

### 模型权重

| 模型 | 路径 | 大小 | 说明 |
|------|------|------|------|
| **Wan2.1-T2V-1.3B** (diffusers) | `/mnt/users/jiayi/.cache/veomni/Wan2.1-T2V-1.3B-Diffusers/` | 28 GB | **已下载, 可直接用** |
| ├─ Transformer | `transformer/` (2 shards) | 5.3 GB | DiT 主模型 (12头, 30层) |
| ├─ Text Encoder | `text_encoder/` (5 shards) | 21.4 GB | UMT5-XXL |
| ├─ VAE | `vae/` | 485 MB | AutoencoderKLWan |
| ├─ Tokenizer | `tokenizer/` | 21 MB | T5TokenizerFast |
| └─ Scheduler | `scheduler/` | 1 KB | FlowMatchEulerDiscrete |

> 14B 模型未下载。如需要, HF ID: `Wan-AI/Wan2.1-T2V-14B`

### 数据集

| 数据集 | 路径 | 大小 | 说明 |
|--------|------|------|------|
| **crush-smol** (原始视频) | `/mnt/users/jiayi/.cache/veomni/crush-smol/` | 53 MB | 47 个液压机视频 + 文本描述 |
| **crush-smol** (parquet) | `/mnt/users/jiayi/.cache/veomni/crush-smol-parquet/` | 53 MB | 转换后的训练格式 (prompt + video_bytes) |
| dummy 1.3B 数据 | `/mnt/users/jiayi/.cache/veomni/wan_t2v_1.3b/` | 9.8 GB | 随机生成的 parquet (smoke test 用) |
| dummy toy 数据 | `/mnt/users/jiayi/.cache/veomni/wan_t2v/` | 257 MB | toy 模型 smoke test |

### 模型规模配置

| 配置文件 | 参数量 | dim | heads | layers |
|----------|--------|-----|-------|--------|
| `configs/model_configs/wan/want2v_0.3b.json` | 0.3B | 1024 | 8 | 20 |
| `configs/model_configs/wan/want2v_0.6b.json` | 0.6B | 1536 | 12 | 24 |
| `configs/model_configs/wan/want2v_1.3b.json` | 1.3B | 2560 | 20 | 32 |
| `configs/model_configs/wan/want2v_3b.json` | 3B | 2560 | 20 | 32 |
| `configs/model_configs/wan/want2v_7b.json` | 7B | 3584 | 28 | 36 |

> 0.3B 和 0.6B 推荐用于快速迭代初始化实验, 显存占用小, 训练速度快。

---

## 4. 训练配置

| 配置文件 | 模型 | 数据 | 用途 |
|----------|------|------|------|
| `configs/dit/wan_smoke_test_8gpu.yaml` | toy (2层) | dummy | 5分钟验证管线 |
| `configs/dit/wan_1.3b_smoke_test_8gpu.yaml` | 1.3B (从零) | dummy 1.3B | 50步验证大模型训练 |
| `configs/dit/wan_1.3b_real_data_8gpu.yaml` | 1.3B (预训练权重) | crush-smol (真实视频) | 真实数据端到端验证 |
| `configs/dit/wan_pretrain_80h20.yaml` | 1.3B | (需配置) | 80卡生产预训练 (参考) |

### 运行示例

```bash
cd /mnt/users/jiayi/jiayi/vla/VeOmni
source /tmp/jiayi_veomni_venv/bin/activate

# smoke test: toy 模型, dummy 数据, ~5 分钟
torchrun --standalone --nproc-per-node=8 \
  tasks/train_dit.py configs/dit/wan_smoke_test_8gpu.yaml

# 1.3B 模型从零训练, dummy 数据, ~25 分钟 (50步, 31s/步)
torchrun --standalone --nproc-per-node=8 \
  tasks/train_dit.py configs/dit/wan_1.3b_smoke_test_8gpu.yaml

# 1.3B 预训练权重 + 真实视频, ~10 分钟 (50步, 11s/步)
torchrun --standalone --nproc-per-node=8 \
  tasks/train_dit.py configs/dit/wan_1.3b_real_data_8gpu.yaml
```

---

## 5. 初始化代码导读

### 你需要改的文件

| 文件 | 内容 | 行号 |
|------|------|------|
| `veomni/models/transformers/wan/modeling_wan.py` | **DiT 权重初始化** (xavier + zero-init) | 558-607 |
| `veomni/models/diffusers/wan_t2v/wan_condition/modeling_wan_condition.py` | Timestep 采样 + Loss 权重 | 136-201 |
| `veomni/models/diffusers/wan_t2v/wan_condition/configuration_wan_condition.py` | 配置参数 | 30-33 |

### 当前初始化策略

```python
# veomni/models/transformers/wan/modeling_wan.py

def _init_weights(self, module):
    """Stage 1: Xavier uniform + 零偏置"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv3d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def _zero_init_residual_outputs(self):
    """Stage 2: 零初始化残差输出 → 每个 DiTBlock 初始为恒等映射"""
    for block in self.blocks:
        nn.init.zeros_(block.self_attn.o.weight)
        nn.init.zeros_(block.cross_attn.o.weight)
        nn.init.zeros_(block.ffn[2].weight)  # FFN 输出层
    nn.init.zeros_(self.head.head.weight)     # 最终输出头
```

### Timestep 采样 (已实现 3 种)

```python
# modeling_wan_condition.py
# 在 config 中设置: timestep_sampling: "uniform" / "logit_normal" / "cosmap"

# uniform: 均匀随机采样
# logit_normal: SD3 风格, 集中在中等噪声 (mean=0, std=1 可调)
# cosmap: 余弦映射, 集中在中间区域
```

### Loss 权重 (已实现 3 种)

```python
# 在 config 中设置: loss_weighting: "none" / "min_snr" / "cosmap"

# none: 均匀权重
# min_snr: 信噪比加权, clamp at gamma=5.0
# cosmap: 逆 sigma 加权 1/(1-sigma+eps)
```

---

## 6. 开发工作流建议

```bash
# 1. 拉最新代码
cd /mnt/users/jiayi/jiayi/vla/VeOmni
git pull

# 2. 开新分支
git checkout -b pretrain/your-feature-name

# 3. 改初始化代码
#    主要修改 veomni/models/transformers/wan/modeling_wan.py

# 4. 用 toy 模型快速验证 (改 configs/dit/wan_smoke_test_8gpu.yaml)
torchrun --standalone --nproc-per-node=8 \
  tasks/train_dit.py configs/dit/wan_smoke_test_8gpu.yaml

# 5. 看 loss 曲线是否正常
#    - 随机数据理论最优 loss ≈ 1.0
#    - 从 ~1.5 降到 ~1.0 说明训练正常
#    - NaN 或不收敛说明初始化有问题

# 6. 确认无误后, 换 1.3B 模型验证规模效应
torchrun --standalone --nproc-per-node=8 \
  tasks/train_dit.py configs/dit/wan_1.3b_smoke_test_8gpu.yaml
```

### 自定义初始化实验

如果想测试新的初始化方法, 可以:

1. 在 `modeling_wan.py` 中添加新的初始化函数
2. 在 yaml 配置中加一个 `init_method` 字段
3. 在 `build_foundation_model` 时根据配置选择初始化

或者更简单: 直接在 `_init_weights` / `_zero_init_residual_outputs` 里改, 用 smoke test 快速对比。

---

## 7. 注意事项

- **共享存储**: `/mnt/users/jiayi/` 所有 lyg 服务器共享, 改代码一台改全部看到
- **venv 在 /tmp**: 本地磁盘, 每台服务器独立, 重启会丢失 (需重建)
- **GPU 礼仪**: 用前 `nvidia-smi` 确认空闲, 用几张 kill 几张, 不要清空全部
- **不提交数据**: 模型权重、checkpoint、parquet 等不进 git
- **输出目录**: 训练输出放 `/tmp/` (本地) 或 `/mnt/users/jiayi/` (持久化)
- **下载用镜像**: `export HF_ENDPOINT=https://hf-mirror.com` (如果需要下载新模型)
