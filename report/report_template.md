# Mini-BLIP2 图像描述生成复现实验报告

## 1. 论文信息

- 论文名称：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
- 论文地址：https://arxiv.org/abs/2301.12597

## 2. 任务说明

本实验复现的任务是图像描述生成 Image Captioning。

输入：图片  
输出：英文 caption

## 3. 数据集

- 数据集名称：Flickr8k
- 数据集地址：https://www.kaggle.com/datasets/adityajn105/flickr8k
- 实际使用数据量：500 张图片（代码中 `load_data(num_images=500)`），前 200 张即可满足最低要求

数据加载流程：
1. 读取 `captions.txt`，按图片名聚合所有 caption（每张图片通常有 5 条描述）
2. 取前 N 张唯一图片，每张保留全部 caption
3. 训练时每个 batch 随机从每张图片的多条 caption 中选一条，增强数据多样性

## 4. 模型结构

```text
Image → Frozen Vision Encoder (CLIP ViT-B/32) → Mini Q-Former → Projection Layer → Frozen Language Decoder (OPT-125M) → Caption
```

### 4.1 Vision Encoder

- 模型：`openai/clip-vit-base-patch32`
- 输入：224×224 RGB 图片
- 输出：(batch, 50, 768) — 1 个 CLS token + 49 个 patch token（7×7 grid, patch_size=32）
- 状态：冻结（`requires_grad = False`）

### 4.2 Mini Q-Former

自己实现的 Mini Q-Former 结构：

| 参数 | 值 |
|---|---|
| query token 数量 | 32 |
| hidden size (embed_dim) | 768 |
| num_heads | 8 |
| FFN 中间维度 (ff_dim) | 2048 |
| Transformer 层数 | 4 |
| 是否使用 cross-attention | 是 |

每一层 QFormerBlock 由三部分组成（均采用 Pre-Norm + 残差连接）：

1. **Self-Attention**：32 个 query token 之间的自注意力交互（`nn.MultiheadAttention`）
2. **Cross-Attention**：query 作为 Q，视觉特征作为 K 和 V，提取图像信息
3. **FFN**：Linear(768→2048) → GELU → Linear(2048→768)

可学习参数：`nn.Parameter(torch.randn(32, 768))` 作为 learnable query tokens。

### 4.3 Projection Layer

- 结构：`nn.Linear(768, 768)`
- 作用：将 Q-Former 输出（已在视觉语义空间）映射到 OPT-125M 的词嵌入空间，使 prefix embeddings 与 text embeddings 可直接拼接

### 4.4 Language Decoder

- 模型：`facebook/opt-125m`
- 词嵌入维度：768（与 Q-Former 输出对齐）
- 状态：冻结（`requires_grad = False`）
- Tokenizer：OPT tokenizer，`pad_token` 设为 `eos_token`

## 5. 训练设置

| 参数 | 值 |
|---|---|
| 训练数据量 | 500 张图片（每张随机选 1 条 caption / batch） |
| epoch | 100 |
| batch size | 4 |
| learning rate | 5e-5 |
| optimizer | AdamW（weight_decay = 0.05） |
| LR scheduler | CosineAnnealingLR（T_max=100, eta_min=0） |
| loss function | Cross Entropy Loss（OPT 内部计算） |
| 图像预处理 | Resize(224,224) → ToTensor → Normalize(mean=0.5, std=0.5) |

- **冻结的模块**：Vision Encoder (CLIP ViT-B/32) + Language Decoder (OPT-125M)
- **训练的模块**：Mini Q-Former+ Projection Layer


## 6. 训练过程

Epoch 1/100, Avg Loss: 2.8630, LR: 5.00e-05, Time: 10.8s

  Batch 10/125, Loss: 2.9537, it/s: 13.62
  Batch 20/125, Loss: 2.5056, it/s: 13.08
  Batch 30/125, Loss: 2.1004, it/s: 13.22
  Batch 40/125, Loss: 3.2370, it/s: 13.19
  Batch 50/125, Loss: 2.1510, it/s: 13.31
  Batch 60/125, Loss: 2.2512, it/s: 13.02
  Batch 70/125, Loss: 2.1878, it/s: 13.12
  Batch 80/125, Loss: 2.7349, it/s: 12.18
  Batch 90/125, Loss: 1.9838, it/s: 13.50
  Batch 100/125, Loss: 2.6029, it/s: 13.59
  Batch 110/125, Loss: 1.9298, it/s: 12.92
  Batch 120/125, Loss: 1.9281, it/s: 13.57
Epoch 2/100, Avg Loss: 2.3186, LR: 5.00e-05, Time: 9.5s

  Batch 10/125, Loss: 2.0750, it/s: 12.85
  Batch 20/125, Loss: 2.0198, it/s: 12.65
  Batch 30/125, Loss: 2.0456, it/s: 12.99
  Batch 40/125, Loss: 2.5308, it/s: 12.59
  Batch 50/125, Loss: 2.3041, it/s: 13.27
  Batch 60/125, Loss: 1.9746, it/s: 12.99
  Batch 70/125, Loss: 2.8278, it/s: 13.09
  Batch 80/125, Loss: 2.0202, it/s: 13.12
  Batch 90/125, Loss: 2.1070, it/s: 13.42
  Batch 100/125, Loss: 1.9707, it/s: 13.29
  Batch 110/125, Loss: 1.9315, it/s: 12.97
  Batch 120/125, Loss: 2.4830, it/s: 12.75

  展示前两个epoch和损失量的下降情况

## 7. 生成结果展示

[1] 1286408831_05282582ed.jpg
    GT:  A boy with his mouth open and tongue sticking out clinging to a bar next to a platform .
    GEN: The young boy climbs a large tree while holding on to a tree .

[2] 1095580424_76f0aa8a3e.jpg
    GT:  A Corgi runs out of a tunnel .
    GEN: A dog is running out of a tunnel on a course .

[3] 1022975728_75515238d8.jpg
    GT:  A black dog running in the surf .
    GEN: A black dog splashes in the water .

[4] 140377584_12bdbdf2f8.jpg
    GT:  A climber in an orange helmet is ascending attached to a rope whilst climbing a rock face .
    GEN: A rock climber ascends .

[5] 1075716537_62105738b4.jpg
    GT:  A child with a helmet on his head rides a bike .
    GEN: A young boy in a helmet and helmet rides a bike on a dirt race .

## 8. 总结

使用AI进行辅助，在复现中能够极大提高效率，对于像我这样的初学者来说也能够完成基本的复现工作，在深度学习的相关知识点缺乏的情况下，AI对于复现任务理解和代码思路的帮助是巨大的，是一次很好的AI与科研相结合的训练。

## 9. AI 对话过程记录

请填写本次复现过程中与 AI 工具的对话记录（对应 requirements.md 第 9.1 节）。

- 对话链接：暂无对话链接
- 使用的 AI 模型：DeepSeek v4pro 网页版
- 累计对话时长 / 会话数：约十小时

简要说明 AI 在哪些环节给了帮助、哪些地方是自己独立完成或推翻了 AI 的建议（2—4 句话即可）：
AI在pytorch环境配置，两个冻结层和中间miniQformer的搭建提供了帮助，在图片读取中有大部分代码为我独立完成和修改，AI的代码思路的建议最初是依次读取caption.txt中的关于图片的描述，这会导致读取出现问题，我提出的修改思路是依次历遍caption的每一行，找出单独的每一张图片的解释，而不是一张图片的多个解释。在Qformer的搭建中AI提供了基本思路和原理解释，同时也帮助我完成代码，最后的训练参数的设置为我自己结合电脑性能设置。
最后在任务书阶段，AI在我写的基础上根据训练代码进行了补充和完善，提高了工作效率和质量。