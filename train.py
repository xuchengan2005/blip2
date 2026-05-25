import os
import random
import time
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPModel, OPTForCausalLM, AutoTokenizer

# 项目根目录，所有路径基于此，确保任意工作目录下运行都正确
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================== 数据加载 ====================

def read_captions(file_path, num=200):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # 跳过标题行
    start = 0
    first_line = lines[0].lower()
    if "image" in first_line or "comment" in first_line:
        start = 1

    img_names = []
    captions = []
    for line in lines[start:start + num]:
        line = line.strip()
        if not line:
            continue
        # 只按第一个逗号分割，避免描述中的逗号被误切
        img_name, caption = line.split(",", 1)
        img_names.append(img_name.strip())
        captions.append(caption.strip())

    return img_names, captions


def build_paths(img_names, img_dir=None):
    if img_dir is None:
        img_dir = os.path.join(BASE_DIR, "data/archive/Images")
    # 将文件名与目录拼接为完整路径
    full_paths = []
    for name in img_names:
        full_paths.append(os.path.join(img_dir, name))
    return full_paths


def load_data(num_images=200):
    # 读取描述文件，每张唯一图片保留所有描述
    with open(os.path.join(BASE_DIR, "data/archive/captions.txt"), "r") as f:
        lines = f.readlines()

    # 跳过标题行
    start = 0
    first_line = lines[0].lower()
    if "image" in first_line or "comment" in first_line:
        start = 1

    img_captions = {}       # {图片名: [描述列表]}
    img_order = []          # 保持图片出现的先后顺序
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        img_name, caption = line.split(",", 1)
        img_name = img_name.strip()
        caption = caption.strip()

        if img_name in img_captions:
            # 已收集的图片，追加新描述
            img_captions[img_name].append(caption)
        elif len(img_captions) < num_images:
            # 新图片且未凑够数量，加入
            img_captions[img_name] = [caption]
            img_order.append(img_name)

    # 按出现顺序构建返回列表
    image_paths = build_paths(img_order, img_dir=os.path.join(BASE_DIR, "data/archive/Images"))
    all_captions = [img_captions[name] for name in img_order]
    return image_paths, all_captions, len(image_paths)


# ==================== 模型加载 ====================

# 使用国内镜像源，解决 HuggingFace 下载不稳定的问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Loading vision encoder from local: models/clip-vit-base-patch32 ...")
full_clip = CLIPModel.from_pretrained(os.path.join(BASE_DIR, "models/clip-vit-base-patch32"))
vision_encoder = full_clip.vision_model  # 从完整 CLIP 中提取视觉编码器
vision_encoder.to(device)
for param in vision_encoder.parameters():
    param.requires_grad = False
print("Vision encoder loaded and frozen.")

print("Loading language model from local: models/opt-125m ...")
tokenizer = AutoTokenizer.from_pretrained(os.path.join(BASE_DIR, "models/opt-125m"))
language_model = OPTForCausalLM.from_pretrained(os.path.join(BASE_DIR, "models/opt-125m"))
language_model.to(device)
for param in language_model.parameters():
    param.requires_grad = False
print("Language model and tokenizer loaded and frozen.")

print("All models loaded successfully.")


# ==================== MiniQFormer ====================

class QFormerBlock(nn.Module):
    """单层 Q-Former：自注意力 + 交叉注意力 + 前馈网络"""
    def __init__(self, embed_dim=768, num_heads=8, ff_dim=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, queries, visual_features):
        # 自注意力：query 之间交互
        q_norm = self.norm1(queries)
        attn_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + attn_out

        # 交叉注意力：query → 视觉特征
        q_norm = self.norm2(queries)
        attn_out, _ = self.cross_attn(q_norm, visual_features, visual_features)
        queries = queries + attn_out

        # 前馈网络
        q_norm = self.norm3(queries)
        ffn_out = self.ffn(q_norm)
        queries = queries + ffn_out

        return queries


class MiniQFormer(nn.Module):
    def __init__(self, num_queries=32, embed_dim=768, num_heads=8, ff_dim=2048, num_layers=4):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_queries, embed_dim))
        self.layers = nn.ModuleList([
            QFormerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])

    def forward(self, visual_features):
        # visual_features: (batch_size, 197, 768)
        batch_size = visual_features.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            queries = layer(queries, visual_features)
        # out: (batch_size, num_queries, 768)
        return queries


# ==================== 实例化 Q-Former ====================

qformer = MiniQFormer(num_queries=32, embed_dim=768)
qformer.to(device)

# 投影层：将 Q-Former 输出映射到语言模型的词嵌入维度
projection = nn.Linear(768, 768)
projection.to(device)

# 统计可训练参数
trainable_params = sum(p.numel() for p in qformer.parameters()) + \
                   sum(p.numel() for p in projection.parameters())
print(f"Q-Former + Projection 可训练参数: {trainable_params:,}")


# ==================== 测试：Q-Former + Projection 形状验证 ====================

print("\n--- 测试 MiniQFormer + Projection ---")
with torch.no_grad():
    dummy_visual = torch.randn(4, 197, 768).to(device)
    query_out = qformer(dummy_visual)
    print(f"Q-Former 输出形状: {query_out.shape}")          # 期望 (4, 32, 768)
    proj_out = projection(query_out)
    print(f"Projection 输出形状: {proj_out.shape}")          # 期望 (4, 32, 768)
print("--- 测试通过 ---\n")


# ==================== DataLoader ====================

# OPT tokenizer 默认没有 pad_token，设为 eos_token 用于 padding
tokenizer.pad_token = tokenizer.eos_token

# 图像预处理：缩放 + 转 tensor + 归一化到 [-1, 1]
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class CaptionDataset(Dataset):
    def __init__(self, image_paths, all_captions):
        self.image_paths = image_paths
        self.all_captions = all_captions

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 返回图片路径和该图片的所有描述列表
        return self.image_paths[idx], self.all_captions[idx]


def collate_fn(batch):
    images = []
    captions = []
    for path, caption_list in batch:
        # 加载图片并预处理
        img = Image.open(path).convert("RGB")
        img = image_transform(img)
        images.append(img)
        # 从描述列表中随机选一个
        captions.append(random.choice(caption_list))

    images = torch.stack(images, dim=0)
    tokenized = tokenizer(
        captions, padding=True, truncation=True, return_tensors="pt"
    )
    return images, tokenized.input_ids, tokenized.attention_mask


def create_dataloader(image_paths, all_captions, batch_size=4, shuffle=True):
    dataset = CaptionDataset(image_paths, all_captions)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return loader


# ==================== 训练逻辑 ====================

def train_step(images, input_ids, attention_mask, scaler=None):
    """
    images:         (batch, 3, 224, 224)
    input_ids:      (batch, text_len)
    attention_mask: (batch, text_len)
    scaler:         GradScaler（GPU 混合精度时使用）
    """
    use_amp = device.type == "cuda"

    # 1. 冻结的视觉编码器提取特征（无梯度）
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp):
            vision_outputs = vision_encoder(images)
            visual_features = vision_outputs.last_hidden_state  # (batch, 197, 768)

    # 2. Q-Former：query 与视觉特征做交叉注意力
    with torch.amp.autocast("cuda", enabled=use_amp):
        query_out = qformer(visual_features)  # (batch, 32, 768)

        # 3. 投影到语言模型嵌入空间
        prefix_embeds = projection(query_out)  # (batch, 32, 768)

        # 4. 将文本 token 转为嵌入
        text_embeds = language_model.get_input_embeddings()(input_ids)  # (batch, T, 768)

        # 5. 拼接 prefix + 文本嵌入
        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)  # (batch, 32+T, 768)

    # 6. 构造完整的 attention_mask：prefix 部分全 1
    batch_size = images.shape[0]
    prefix_mask = torch.ones(
        batch_size, prefix_embeds.shape[1], device=device, dtype=attention_mask.dtype
    )
    full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # (batch, 32+T)

    # 7. 构造 labels：prefix 部分设为 -100（忽略），文本部分用 input_ids
    ignore_labels = torch.full(
        (batch_size, prefix_embeds.shape[1]), -100,
        device=device, dtype=torch.long
    )
    labels = torch.cat([ignore_labels, input_ids.to(torch.long)], dim=1)  # (batch, 32+T)

    # 8. OPT 前向
    with torch.amp.autocast("cuda", enabled=use_amp):
        outputs = language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
        )
    return outputs.loss


# ==================== 优化器与超参数 ====================

optimizer = torch.optim.AdamW(
    list(qformer.parameters()) + list(projection.parameters()),
    lr=5e-5,
    weight_decay=0.05,
)

# 学习率调度器：余弦退火
num_epochs = 100
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=0
)

scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

epoch_losses = []  # 记录每个 epoch 的平均 loss


# ==================== 训练循环 ====================

if __name__ == "__main__":
    # 加载数据
    image_paths, all_captions, n = load_data(num_images=500)
    print(f"读取到 {n} 张唯一图片\n")

    # 创建 DataLoader
    loader = create_dataloader(image_paths, all_captions, batch_size=4, shuffle=True)

    # 设置训练模式
    qformer.train()
    projection.train()
    vision_encoder.eval()
    language_model.eval()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = len(loader)
        epoch_start = time.time()
        batch_start = time.time()
        for batch_idx, (images, input_ids, attn_mask) in enumerate(loader):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            optimizer.zero_grad()
            loss = train_step(images, input_ids, attn_mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # 每 10 个 batch 打印一次当前 loss 和 it/s
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - batch_start
                it_per_sec = 10 / elapsed if elapsed > 0 else 0
                batch_start = time.time()
                print(f"  Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}, it/s: {it_per_sec:.2f}")

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.2e}, Time: {epoch_time:.1f}s\n")

    print("训练完成。\n")

    # ==================== 推理 ====================

    def generate_caption(image_path, max_new_tokens=30):
        # 加载并预处理图片
        img = Image.open(image_path).convert("RGB")
        img_tensor = image_transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

        with torch.no_grad():
            # 视觉编码器
            vision_outputs = vision_encoder(img_tensor)
            visual_features = vision_outputs.last_hidden_state  # (1, 197, 768)

            # Q-Former + 投影 -> prefix embeddings
            query_out = qformer(visual_features)  # (1, 32, 768)
            prefix_embeds = projection(query_out)  # (1, 32, 768)

            # attention_mask 与 prefix 长度一致
            prefix_attn_mask = torch.ones(
                1, prefix_embeds.shape[1], device=device, dtype=torch.long
            )

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                output_ids = language_model.generate(
                    inputs_embeds=prefix_embeds,
                    attention_mask=prefix_attn_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=5,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

        # inputs_embeds 模式下 generate 只返回新生成的 token，直接解码即可
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()

    # 随机选取 5 张图片生成描述
    num_samples = min(5, len(image_paths))
    sample_indices = random.sample(range(len(image_paths)), num_samples)

    print("=== 生成描述示例 ===")
    for i, idx in enumerate(sample_indices):
        path = image_paths[idx]
        img_name = os.path.basename(path)
        gt_caption = all_captions[idx][0]
        gen_caption = generate_caption(path)

        print(f"[{i+1}] {img_name}")
        print(f"    GT:  {gt_caption}")
        print(f"    GEN: {gen_caption}\n")

