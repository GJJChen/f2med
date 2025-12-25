import copy
import csv
import os
from PIL import Image, ImageDraw
from torch.amp import autocast
from torchvision.transforms import v2

# 环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datetime
import pickle
import torch
# —— 分布式导入（安全）——
try:
    import torch.distributed as dist
    _DIST_IMPORTED = True
except Exception:
    dist = None
    _DIST_IMPORTED = False

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import (
    resnet50, resnet18, resnet34, resnet101, resnet152,
    ResNet50_Weights, ResNet18_Weights, ResNet34_Weights,
    ResNet101_Weights, ResNet152_Weights,
    densenet121, DenseNet121_Weights
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Dict, List, Tuple, Optional, Callable
import logging
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score, cohen_kappa_score
)
import timm
from sklearn.model_selection import train_test_split
import json
import lmdb
import msgpack
import math

# --- 全局配置 ---
BERT_MODEL = 'xlm-roberta-base'

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- 分布式安全辅助 ----------
def _dist_available() -> bool:
    return _DIST_IMPORTED and hasattr(dist, "is_available") and dist.is_available()

def _dist_initialized() -> bool:
    return _dist_available() and dist.is_initialized()

def is_distributed_world() -> bool:
    # 只有在 torchrun 设置 WORLD_SIZE>1 且进程组已初始化时，才视为分布式
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return world > 1 and _dist_initialized()
def want_distributed() -> bool:
    # 只看环境是否要求多进程，不依赖 dist 是否已经初始化
    return int(os.environ.get("WORLD_SIZE", "1")) > 1
def get_rank_safe() -> int:
    return dist.get_rank() if _dist_initialized() else 0
def unwrap_ddp(m):
    return m.module if isinstance(m, DDP) else m

def unwrap_compile(m):
    # PyTorch 2.x compile 后的原始模块
    return getattr(m, "_orig_mod", m)
def unwrap_model(m: nn.Module) -> nn.Module:
    """从外到内拆壳：DDP / torch.compile(OptimizedModule) → 原始模型"""
    while True:
        if isinstance(m, DDP):
            m = m.module
            continue
        if hasattr(m, "_orig_mod"):  # PyTorch 2.x compile 外壳
            m = m._orig_mod
            continue
        break
    return m
def get_base_model(m):
    return unwrap_compile(unwrap_ddp(m))
def is_main_process() -> bool:
    # 导入阶段/未初始化时，安全地把当前进程当作 rank0
    return get_rank_safe() == 0
def set_epoch_if_needed(dataloader, epoch):
    if is_distributed_world():
        try:
            if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)
        except Exception:
            pass
def _is_dist():
    return dist.is_available() and dist.is_initialized()

def _is_main():
    return (not _is_dist()) or dist.get_rank() == 0

def _reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if _is_dist():
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        y /= dist.get_world_size()
        return y
    return x

def _reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if _is_dist():
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y
    return x

def build_dataloader(
    dataset,
    batch_size,
    shuffle,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    **kwargs
):
    """在 DDP 下自动使用 DistributedSampler；其余参数透传给 DataLoader。"""
    sampler = None
    if is_distributed_world():
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=shuffle)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **kwargs
    )

# Enable TF32 on Ampere+ for better throughput
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

def ddp_setup():
    """Initialize torch.distributed and pick local device."""
    if _dist_available() and not _dist_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def ddp_cleanup():
    if _dist_initialized():
        dist.destroy_process_group()
def norm_img_id(x: str) -> str:
    # 统一：basename + 小写 + 保留扩展名（如需可改成去扩展名）
    return os.path.basename(str(x)).lower()
# 这里可以安全打印（在未初始化时也不会触发 dist.get_rank）
if is_main_process():
    print(f"[Info] Using HF endpoint: {os.environ['HF_ENDPOINT']}")

# ===== 从这里开始是你原来的数据集定义 =====
class LMDBDataset(Dataset):
    def __init__(self, lmdb_path: str, transform=None, max_text_length: int = 64, split: str | None = None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.max_text_length = max_text_length
        self.split = split

        # --- one-shot open to read meta/keys, then close immediately ---
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, meminit=False, max_readers=4096)
        with env.begin(write=False) as txn:
            meta_raw = txn.get(b"__meta__")
            if not meta_raw:
                raise RuntimeError("LMDB缺少 __meta__")
            self.meta = json.loads(meta_raw.decode("utf-8"))
            keys_raw = txn.get(b"__keys__")
            if not keys_raw:
                raise RuntimeError("LMDB缺少 __keys__")
            all_keys = keys_raw.decode("utf-8").splitlines()
        env.close()

        if split is not None and split != "":
            prefix = f"{split}:"
            self.keys = [k for k in all_keys if k.startswith(prefix)]
            if is_main_process():
                logger.info(f"Loaded {len(self.keys)} samples for split: {split}")
        else:
            self.keys = all_keys
            if is_main_process():
                logger.info(f"Loaded {len(self.keys)} samples (all splits)")

        # gather label list to compute mapping & y
        label_list = []
        # reopen briefly to scan labels; closes immediately after
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, meminit=False, max_readers=4096)
        with env.begin(write=False) as txn:
            for k in self.keys:
                v = txn.get(k.encode("utf-8"))
                if v is None:
                    continue
                item = msgpack.unpackb(v, raw=False)
                label_list.append(int(item["label_index"]))
        env.close()

        if not label_list:
            raise RuntimeError("选择的 split 为空或 LMDB 无有效样本")

        vc = pd.Series(label_list).value_counts().sort_index()
        if is_main_process():
            logger.info(f"Class distribution for split '{split}':\n{vc}")

        self.labels = sorted(pd.unique(label_list).tolist())
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
        self.num_classes = len(self.labels)
        # important: make y available for stratified splits & other utilities
        self.y = torch.tensor([int(lbl) for lbl in label_list], dtype=torch.long)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None) or self.tokenizer.sep_token

        # --- defer env creation to workers ---
        self._env = None  # will be opened lazily in each worker

    def __len__(self):
        return len(self.keys)

    # ensure we have a per-worker env
    def _ensure_env(self):
        if self._env is None:
            # Each worker opens its own read-only environment
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=4096,
            )
        return self._env

    def __getstate__(self):
        # drop lmdb env before pickling (sent to worker processes)
        state = self.__dict__.copy()
        state['_env'] = None
        return state

    def __del__(self):
        try:
            if getattr(self, "_env", None) is not None:
                self._env.close()
        except Exception:
            pass

    @staticmethod
    def _decode_image(image_bytes: bytes):
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("cv2.imdecode failed")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx: int):
        env = self._ensure_env()
        key = self.keys[idx]
        with env.begin(write=False) as txn:
            raw = txn.get(key.encode("utf-8"))
            if raw is None:
                raise IndexError(f"Key not found in LMDB: {key}")
            item = msgpack.unpackb(raw, raw=False)

        image = self._decode_image(item["image_bytes"])
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']

        # text = item.get('description', '') or ''
        # enc = self.tokenizer(text, padding='max_length', truncation=True,
        #                      max_length=self.max_text_length, return_tensors='pt')
        # text_input_ids = enc['input_ids'].squeeze(0)
        # text_attention_mask = enc['attention_mask'].squeeze(0)

        raw_label = int(item['label_index'])
        label_idx = self.label_to_idx[raw_label]

        sample = {
            'image': image,
            'text_input_ids': 0,
            'text_attention_mask': 0,
            'label_index': torch.tensor(label_idx, dtype=torch.long),
            'class_name': raw_label,
            'image_id': item.get('id', key),
        }
        return sample

class BaseDataset(Dataset):
    """视网膜OCT数据集类"""

    def __init__(self, csv_path: str, dataset_base_path: str, transform=None, max_text_length=512, split=None):
        """
        Args:
            csv_path: CSV文件路径
            dataset_base_path: 数据集的基础路径
            transform: 图像变换
            max_text_length: 最大文本长度
            split: 数据集划分 ('train', 'val', 'test', None)
        """
        # 读取CSV文件
        self.df = pd.read_csv(csv_path)

        # 如果指定了split，则只使用对应的数据
        if split is not None:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)
            if is_main_process():
                logger.info(f"Loaded {len(self.df)} samples for split: {split}")

        if not self.df.empty:
            if is_main_process():
                logger.info(f"Class distribution for split '{split}':")
            if is_main_process():
                print(self.df['label_index'].value_counts())

        self.dataset_base_path = dataset_base_path

        # 获取类别信息
        self.labels = sorted(self.df['label_index'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        self.transform = A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()  # 注意这里是 ToTensorV2
        ])
        self.max_text_length = max_text_length

        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.num_classes = len(self.labels)
        # 预分词
        texts = self.df['description'].fillna("").tolist()
        enc = self.tokenizer(texts, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
        self.input_ids = enc['input_ids']
        self.attention_mask = enc['attention_mask']
        labels = sorted(self.df['label_index'].unique())
        self.label2idx = {l:i for i,l in enumerate(labels)}
        self.y = torch.tensor([self.label2idx[l] for l in self.df['label_index']], dtype=torch.long)

        if is_main_process():

            logger.info(f"Loaded {len(self.df)} samples with {self.num_classes} classes")
        if is_main_process():
            logger.info(f"Classes: {self.labels}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 加载图像
        image_path = os.path.join(self.dataset_base_path, row['image_path'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return {
            'image': image,
            'text_input_ids': self.input_ids[idx],
            'text_attention_mask': self.attention_mask[idx],
            'label_index': self.y[idx],
            'class_name': row['label_index'],
            'image_id': row['id']
        }

class MultiModalEncoder(nn.Module):
    """
    多模态编码器（A 方案）：
    - 用 `use_text` / `use_fusion` 控制是否启用文本与融合分支
    - 在构造函数中就对“未启用”的分支执行 requires_grad_(False) + eval()
    - 这样每个 step 所有可训练参数都会参与图，DDP 可用 find_unused_parameters=False
    """
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 512,
        backbone: str = "resnet50",
        text_model_name: str = "xlm-roberta-base",
        image_pretrained: bool = True,
        text_pretrained: bool = True,
        use_text: bool = False,           # 关/开 文本分支
        use_fusion: bool = False          # 关/开 多模态融合分支
    ):
        super().__init__()
        self.backbone_name   = backbone
        self.image_pretrained = image_pretrained
        self.text_pretrained  = text_pretrained
        self.hidden_dim       = hidden_dim
        self.use_text         = use_text
        self.use_fusion       = use_fusion

        # --- 图像编码器 (timm) ---
        self.image_encoder = timm.create_model(backbone, pretrained=image_pretrained)
        in_features = self.image_encoder.get_classifier().in_features
        self.image_encoder.reset_classifier(num_classes=hidden_dim)

        # --- 文本编码器 (Transformers) ---
        if text_pretrained:
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
        else:
            try:
                config = AutoConfig.from_pretrained(text_model_name)
            except Exception:
                from transformers import BertConfig
                config = BertConfig()
            self.text_encoder = AutoModel.from_config(config)

        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)

        # --- 多模态融合层（concat 后再映射） ---
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # --- 单模态“融合”（只走图像分支时的 MLP）---
        self.single_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # --- 分类器 ---
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # ===== 关键：在构造阶段就冻结“未启用”的分支 =====
        if not self.use_text:
            self.text_encoder.requires_grad_(False)
            self.text_projection.requires_grad_(False)
            self.text_encoder.eval()
            self.text_projection.eval()

        if not self.use_fusion:
            self.fusion_layer.requires_grad_(False)
            self.fusion_layer.eval()

    def forward(self, image, text_input_ids=None, text_attention_mask=None):
        # 图像特征
        image_features = self.image_encoder(image)  # shape: [B, hidden_dim]

        # 文本特征（仅在启用时计算）
        if self.use_text:
            assert text_input_ids is not None and text_attention_mask is not None, \
                "use_text=True 时需要提供 text_input_ids 与 text_attention_mask"
            text_outputs  = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            )
            # pooler_output 或者 last_hidden_state 的 [CLS]
            text_features = self.text_projection(text_outputs.pooler_output)  # [B, hidden_dim]
        else:
            # 不启用文本分支时，你可以选择：
            # 1) 直接复用 image_features；或
            # 2) 置零占位；任选其一即可
            text_features = image_features

        # 融合：启用融合则 concat，再走 fusion_layer；否则走单模态 MLP
        if self.use_fusion:
            combined_features = torch.cat([image_features, text_features], dim=1)  # [B, 2*hidden]
            fused_features = self.fusion_layer(combined_features)                  # [B, hidden]
        else:
            fused_features = self.single_layer(image_features)                     # [B, hidden]

        logits = self.classifier(fused_features)

        return {
            "logits": logits,
            "image_features": image_features,
            "text_features": text_features,
            "fused_features": fused_features,
        }



class LocalModel:
    """本地模型类"""

    def __init__(self, model_id: int, num_classes: int, device: torch.device,
                 backbone_options: List[str] = None, hidden_dim_options: List[int] = None,
                 use_data_parallel: bool = False, gpu_ids: List[int] = None, local_epochs: int = 15,
                 use_pretrained: bool = True):  # 新增use_pretrained
        self.model_id = model_id
        self.device = device
        self.use_data_parallel = use_data_parallel
        self.gpu_ids = gpu_ids if gpu_ids is not None else [device.index] if device.type == 'cuda' else [0]

        # 设置本地模型的骨干网络和隐藏维度选项
        # Use EfficientNet backbones as requested
        if backbone_options is None:
            backbone_options = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b2', 'efficientnet_b5']

        if hidden_dim_options is None:
            hidden_dim_options = [256, 256, 384, 384]

        # 根据model_id选择对应的骨干网络和隐藏维度
        backbone_idx = model_id % len(backbone_options)
        hidden_dim_idx = model_id % len(hidden_dim_options)

        self.backbone = backbone_options[backbone_idx]
        self.hidden_dim = hidden_dim_options[hidden_dim_idx]

        if is_main_process():

            logger.info(f"Local Model {model_id} using backbone: {self.backbone}, hidden_dim: {self.hidden_dim}")

        self.model = MultiModalEncoder(
            num_classes=num_classes,
            hidden_dim=self.hidden_dim,
            backbone=self.backbone,
            image_pretrained=use_pretrained,
            text_pretrained=use_pretrained
        ).to(device)

        # 在分布式场景下，用 DDP 包装（每进程绑定自己那张卡）
        if _dist_initialized():
            dev_idx = self.device.index if hasattr(self.device, "index") else torch.cuda.current_device()
            self.model = DDP(
                self.model,
                device_ids=[dev_idx],
                output_device=dev_idx,
                gradient_as_bucket_view=True,
                static_graph=False
            )

        try:
            # 注意：DataParallel模型不需要torch.compile
            if not self.use_data_parallel:
                self.model = torch.compile(self.model)
                if is_main_process():
                    logger.info("Local Model compiled with torch.compile.")
            else:
                if is_main_process():
                    logger.info("Local Model using DataParallel, skipping torch.compile.")
        except Exception as e:
            logger.warning(f"torch.compile failed for Local Model: {e}")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=local_epochs)
    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = correct = total = 0
        for batch in tqdm(dataloader, desc=f"Training Local Model {self.model_id}"):
        # for batch in dataloader:
            images = batch['image'].to(self.device, non_blocking=True)
            ids = batch['text_input_ids'].to(self.device, non_blocking=True)
            mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
            labels = batch['label_index'].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=(self.device.type == 'cuda')):
                outputs = self.model(images, ids, mask)
                loss = self.criterion(outputs['logits'], labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            preds = outputs['logits'].argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            del outputs, loss, preds
        self.lr_scheduler.step()
        return total_loss / len(dataloader), 100 * correct / total
    def prepare_ddp(self):
        """在进入本地训练前调用一次：迁移到本 rank 的 GPU 并包 DDP。"""
        if not torch.cuda.is_available():
            return
        local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")
        self.model.to(self.device)
        # 建议在包 DDP 之前就完成冻结/解冻
        self.model = DDP(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=False,
            static_graph=False,
            find_unused_parameters=False,
        )
        # 用包后的参数建优化器（或重建）
        if hasattr(self, "optimizer") and self.optimizer is not None:
            del self.optimizer
        self.optimizer = torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=getattr(self, "lr", 1e-4)
        )

    def train_epoch_ddp(self, dataloader: DataLoader):
        """4 卡并行训练 1 个 epoch；dataloader 必须是 DistributedSampler。"""
        self.model.train()
        running = 0.0
        correct = torch.tensor(0, device=self.device, dtype=torch.long)
        total = torch.tensor(0, device=self.device, dtype=torch.long)

        pbar = tqdm(dataloader, desc=f"Training Local Model {self.model_id}",
                    disable=not is_main_process(), leave=True)

        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)
            ids = batch['text_input_ids'].to(self.device, non_blocking=True)
            mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
            labels = batch['label_index'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=torch.float16, enabled=(self.device.type == 'cuda')):
                out = self.model(images, ids, mask)
                logits = out['logits'] if isinstance(out, dict) else out
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 统计
            loss_avg = float(_reduce_mean(loss))
            running += loss_avg

            pred = logits.argmax(1)
            correct += (pred == labels).sum()
            total += labels.numel()

            if is_main_process():
                pbar.set_postfix_str(f"Loss: {loss_avg:.4f}")

            del out, logits, loss, pred

        # 规约准确率
        correct = _reduce_sum(correct)
        total = _reduce_sum(total)
        acc = (correct.float() / total.clamp_min(1)).item()

        epoch_loss = running / max(1, len(dataloader))
        return epoch_loss, acc

    @torch.no_grad()
    def evaluate_ddp(self, dataloader: DataLoader):
        """
        [修改后版本]
        4卡并行验证：汇总所有预测和标签，计算 Accuracy 和 QWK。
        """
        self.model.eval()

        # 每个进程收集自己的预测和标签
        local_preds = []
        local_labels = []

        for batch in dataloader:
            images = batch['image'].to(self.device, non_blocking=True)
            ids = batch['text_input_ids'].to(self.device, non_blocking=True)
            mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
            labels = batch['label_index'].to(self.device, non_blocking=True)

            out = self.model(images, ids, mask)
            logits = out['logits'] if isinstance(out, dict) else out
            pred = logits.argmax(1)

            local_preds.append(pred)
            local_labels.append(labels)

        # 将列表中的Tensor拼接成一个大Tensor
        local_preds_tensor = torch.cat(local_preds)
        local_labels_tensor = torch.cat(local_labels)

        acc = 0.0
        qwk = 0.0

        if is_distributed_world():
            # --- 分布式环境下，收集所有进程的数据到 rank0 ---
            world_size = dist.get_world_size()

            # 准备接收列表
            gathered_preds_list = [torch.empty_like(local_preds_tensor) for _ in range(world_size)]
            gathered_labels_list = [torch.empty_like(local_labels_tensor) for _ in range(world_size)]

            # 执行 all_gather
            dist.all_gather(gathered_preds_list, local_preds_tensor)
            dist.all_gather(gathered_labels_list, local_labels_tensor)

            if is_main_process():
                # 只有主进程 (rank0) 进行最终计算
                y_true = torch.cat(gathered_labels_list).cpu().numpy()
                y_pred = torch.cat(gathered_preds_list).cpu().numpy()

                acc = accuracy_score(y_true, y_pred)
                qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        else:
            # --- 单卡环境下，直接计算 ---
            y_true = local_labels_tensor.cpu().numpy()
            y_pred = local_preds_tensor.cpu().numpy()

            acc = accuracy_score(y_true, y_pred)
            qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

        # 返回两个指标
        return acc, qwk

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on given dataloader"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                text_input_ids = batch['text_input_ids'].to(self.device)
                text_attention_mask = batch['text_attention_mask'].to(self.device)
                labels = batch['label_index'].to(self.device)

                outputs = self.model(images, text_input_ids, text_attention_mask)
                _, predicted = torch.max(outputs['logits'].data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    def extract_knowledge(self, dataloader: DataLoader) -> Dict:
        self.model.eval()
        all_soft_preds = []
        all_image_ids = []  # <-- 新增一个列表来收集ID
        with torch.inference_mode():
            # 这里的dataloader必须是 shuffle=False，以保证顺序
            for batch in tqdm(dataloader, desc=f"Extract knowledge from model {self.model_id}"):
                images = batch['image'].to(self.device, non_blocking=True)
                ids = batch['text_input_ids'].to(self.device, non_blocking=True)
                mask = batch['text_attention_mask'].to(self.device, non_blocking=True)

                # 从batch中获取 image_ids
                image_ids = batch['image_id']  # image_id通常是list of strings/ints

                outputs = self.model(images, ids, mask)
                soft = F.softmax(outputs['logits'] / 3.0, dim=1).float().detach().cpu().half()

                all_soft_preds.append(soft)
                all_image_ids.extend(image_ids)  # 使用 extend 添加ID列表

                del outputs, soft

        # 将所有批次的预测结果拼接起来
        final_soft_preds = torch.cat(all_soft_preds, dim=0)

        return {
            'soft_predictions': final_soft_preds,  # float16 CPU Tensor
            'image_ids': all_image_ids  # List of IDs
        }

    def evaluate_roc(self, dataloader: DataLoader) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """计算 ROC 曲线所需的 TPR 和 FPR"""
        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                text_input_ids = batch['text_input_ids'].to(self.device)
                text_attention_mask = batch['text_attention_mask'].to(self.device)
                labels = batch['label_index'].to(self.device)

                outputs = self.model(images, text_input_ids, text_attention_mask)
                probs = F.softmax(outputs['logits'], dim=1)

                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # 计算每个类别的 ROC
        fprs = []
        tprs = []
        from sklearn.metrics import roc_curve

        for i in range(all_probs.shape[1]):  # 对每个类别
            fpr, tpr, _ = roc_curve(
                (all_labels == i).astype(int),
                all_probs[:, i]
            )
            fprs.append(fpr)
            tprs.append(tpr)

        return fprs, tprs



class GlobalModel:
    """全局模型类 - 使用更强大的模型结构"""

    def __init__(self, num_classes: int, device: torch.device, use_data_parallel: bool = False, gpu_ids: List[int] = None,
                 use_pretrained: bool = True):  # 新增use_pretrained
        self.device = device
        self.use_data_parallel = use_data_parallel
        self.gpu_ids = gpu_ids if gpu_ids is not None else [device.index] if device.type == 'cuda' else [0]

        # 使用更强大的骨干网络和更大的隐藏维度
        self.model = MultiModalEncoder(
            num_classes=num_classes,
            hidden_dim=768,
            backbone='vit_base_patch16_224',
            # backbone='resnet50',
            text_model_name='xlm-roberta-large',
            image_pretrained=use_pretrained,
            text_pretrained=use_pretrained
        ).to(device)

        # 在分布式场景下，用 DDP 包装（每进程绑定自己那张卡）
        if _dist_initialized():
            dev_idx = self.device.index if hasattr(self.device, "index") else torch.cuda.current_device()
            self.model = DDP(
                self.model,
                device_ids=[dev_idx],
                output_device=dev_idx,
                gradient_as_bucket_view=True,
                static_graph=False
            )

        if is_main_process():

            logger.info(f"Global Model initialized with backbone: vit_base_patch16_224, hidden_dim: 768")
        try:
            # 注意：DataParallel模型不需要torch.compile
            if not self.use_data_parallel:
                self.model = torch.compile(self.model)
                if is_main_process():
                    logger.info("Global Model compiled with torch.compile.")
            else:
                if is_main_process():
                    logger.info("Global Model using DataParallel, skipping torch.compile.")
        except Exception as e:
            logger.warning(f"torch.compile failed for Global Model: {e}")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))


    def _prepare_partial_distillation(
        self,
        last_bert_layers: int = 2,
        last_vit_blocks: int = 2,
        unfreeze_cnn_stage: str = "layer4"
    ):
        core = self._core()
        # 1. 冻结全部
        for p in core.parameters():
            p.requires_grad = False

        # 2. 文本编码器：最后 N 层 + pooler
        te = core.text_encoder
        if hasattr(te, "encoder") and hasattr(te.encoder, "layer"):
            for layer in te.encoder.layer[-last_bert_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        if hasattr(te, "pooler"):
            for p in te.pooler.parameters():
                p.requires_grad = True

        # 3. 图像编码器
        ie = core.image_encoder
        # ViT (timm) 模式
        if hasattr(ie, "blocks") and isinstance(ie.blocks, (list, nn.ModuleList)):
            for blk in ie.blocks[-last_vit_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
            # head (分类头)
            if hasattr(ie, "head"):
                for p in ie.head.parameters():
                    p.requires_grad = True
        else:
            # CNN：解冻指定 stage + fc/head
            if hasattr(ie, unfreeze_cnn_stage):
                for p in getattr(ie, unfreeze_cnn_stage).parameters():
                    p.requires_grad = True
            # ResNet fc
            if hasattr(ie, "fc"):
                for p in ie.fc.parameters():
                    p.requires_grad = True
            # DenseNet classifier
            if hasattr(ie, "classifier"):
                for p in ie.classifier.parameters():
                    p.requires_grad = True

        # 4. 融合层与分类器
        for p in core.fusion_layer.parameters():
            p.requires_grad = True
        for p in core.classifier.parameters():
            p.requires_grad = True

        # 5. 仅以可训练参数重建优化器
        trainable = [p for p in core.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable, lr=5e-5)
        # 可根据需要设置不同 lr：示例（注释）
        # self.optimizer = optim.Adam([
        #     {"params": trainable_text, "lr": 3e-5},
        #     {"params": trainable_vision, "lr": 3e-5},
        #     {"params": trainable_fusion_head, "lr": 5e-5},
        # ], lr=5e-5)

    def distill_knowledge(
        self,
        teacher_map: Dict, # <-- 接收ID->预测的查找表
        dataloader,
        alpha: float = 0.3,
        temperature: float = 3.0,
        partial_layers: bool = False,
        last_bert_layers: int = 4,
        last_vit_blocks: int = 4
    ):
        self.model.train()
        # 首次执行部分层蒸馏准备
        if partial_layers and not getattr(self, "_partial_distill_ready", False):
            self._prepare_partial_distillation(
                last_bert_layers=last_bert_layers,
                last_vit_blocks=last_vit_blocks
            )
            self._partial_distill_ready = True

        total_loss = 0.0

        pbar = tqdm(dataloader, desc="Knowledge Distillation",
                    disable=not _is_main(), leave=True)

        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)
            ids = batch['text_input_ids'].to(self.device, non_blocking=True)
            mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
            labels = batch['label_index'].to(self.device, non_blocking=True)

            # 取 teacher targets
            batch_image_ids = batch['image_id']
            targets_list = [teacher_map[img_id] for img_id in batch_image_ids]
            tgt = torch.stack(targets_list).to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=torch.float16,
                          enabled=(self.device.type == "cuda")):
                outputs = self.model(images, ids, mask)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                hard = self.criterion(logits, labels)
                soft = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    tgt,
                    reduction="batchmean"
                )
                loss = alpha * hard + (1 - alpha) * soft

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # loss 平均后再显示
            loss_avg = float(_reduce_mean(loss))
            total_loss += loss_avg
            if _is_main():
                pbar.set_postfix(loss=f"{loss_avg:.4f}")

            del outputs, logits, hard, soft, tgt, loss

        return total_loss / max(1, len(dataloader))

    def evaluate(self, dataloader: DataLoader):
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                text_input_ids = batch['text_input_ids'].to(self.device)
                text_attention_mask = batch['text_attention_mask'].to(self.device)
                labels = batch['label_index'].to(self.device)

                outputs = self.model(images, text_input_ids, text_attention_mask)
                _, predicted = torch.max(outputs['logits'].data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total
    @torch.no_grad()
    def evaluate_ddp(self, dataloader: DataLoader):
        """
        [修改后版本]
        4卡并行验证：汇总所有预测和标签，计算 Accuracy 和 QWK。
        """
        self.model.eval()

        # 每个进程收集自己的预测和标签
        local_preds = []
        local_labels = []

        for batch in dataloader:
            images = batch['image'].to(self.device, non_blocking=True)
            ids = batch['text_input_ids'].to(self.device, non_blocking=True)
            mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
            labels = batch['label_index'].to(self.device, non_blocking=True)

            out = self.model(images, ids, mask)
            logits = out['logits'] if isinstance(out, dict) else out
            pred = logits.argmax(1)

            local_preds.append(pred)
            local_labels.append(labels)

        # 将列表中的Tensor拼接成一个大Tensor
        local_preds_tensor = torch.cat(local_preds)
        local_labels_tensor = torch.cat(local_labels)

        acc = 0.0
        qwk = 0.0

        if is_distributed_world():
            # --- 分布式环境下，收集所有进程的数据到 rank0 ---
            world_size = dist.get_world_size()

            # 准备接收列表
            gathered_preds_list = [torch.empty_like(local_preds_tensor) for _ in range(world_size)]
            gathered_labels_list = [torch.empty_like(local_labels_tensor) for _ in range(world_size)]

            # 执行 all_gather
            dist.all_gather(gathered_preds_list, local_preds_tensor)
            dist.all_gather(gathered_labels_list, local_labels_tensor)

            if is_main_process():
                # 只有主进程 (rank0) 进行最终计算
                y_true = torch.cat(gathered_labels_list).cpu().numpy()
                y_pred = torch.cat(gathered_preds_list).cpu().numpy()

                acc = accuracy_score(y_true, y_pred)
                qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        else:
            # --- 单卡环境下，直接计算 ---
            y_true = local_labels_tensor.cpu().numpy()
            y_pred = local_preds_tensor.cpu().numpy()

            acc = accuracy_score(y_true, y_pred)
            qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

        # 返回两个指标
        return acc, qwk
    def evaluate_roc(self, dataloader: DataLoader) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """计算 ROC 曲线所需的 TPR 和 FPR"""
        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                text_input_ids = batch['text_input_ids'].to(self.device)
                text_attention_mask = batch['text_attention_mask'].to(self.device)
                labels = batch['label_index'].to(self.device)

                outputs = self.model(images, text_input_ids, text_attention_mask)
                probs = F.softmax(outputs['logits'], dim=1)

                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # 计算每个类别的 ROC
        fprs = []
        tprs = []
        from sklearn.metrics import roc_curve

        for i in range(all_probs.shape[1]):  # 对每个类别
            fpr, tpr, _ = roc_curve(
                (all_labels == i).astype(int),
                all_probs[:, i]
            )
            fprs.append(fpr)
            tprs.append(tpr)

        return fprs, tprs

    def _core(self) -> nn.Module:
        return unwrap_model(self.model)

    def adapt_num_classes(self, new_num_classes: int):
        core = self._core()
        old_num = core.classifier.out_features
        if old_num == new_num_classes:
            return
        in_f = core.classifier.in_features
        new_head = nn.Linear(in_f, new_num_classes).to(self.device)
        # 可选：拷贝重叠权重
        with torch.no_grad():
            keep = min(old_num, new_num_classes)
            new_head.weight[:keep] = core.classifier.weight[:keep]
            new_head.bias[:keep] = core.classifier.bias[:keep]
        core.classifier = new_head
        # 重新构建优化器（只针对可训练参数）
        trainable = [p for p in core.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=1e-5)
        if is_main_process():
            print(f"[adapt_num_classes] classifier changed {old_num} -> {new_num_classes}")

    def fine_tune(self, dataloader: DataLoader, val_dataloader: DataLoader,
                  epochs: int = 3, lr: float = 1e-5, patience: int = 10):
        """
        DDP-稳态版微调：
        - 只在 rank0 打 tqdm / logger
        - 训练损失按全局平均规约
        - 前向/反向 用 self.model（可能是 DDP 包裹），解冻/冻结用 core
        """
        import torch.distributed as dist

        def ddp_avg(t: torch.Tensor) -> torch.Tensor:
            """在 DDP 下对标量做全局平均；单卡原样返回。"""
            if dist.is_available() and dist.is_initialized():
                tt = t.clone().detach()
                dist.all_reduce(tt, op=dist.ReduceOp.SUM)
                tt /= dist.get_world_size()
                return tt
            return t

        core = self._core()  # 解包真实模型用于解冻/保存
        core.train()

        # 1) 冻结全部参数
        for p in core.parameters():
            p.requires_grad = False

        # 2) 解冻分类器与（可选）融合层
        if hasattr(core, "classifier"):
            for p in core.classifier.parameters():
                p.requires_grad = True
        if hasattr(core, "fusion_layer"):
            for p in core.fusion_layer.parameters():
                p.requires_grad = True

        # 3) 文本编码器：解冻最后两层 + pooler（若存在）
        if hasattr(core, "text_encoder") and hasattr(core.text_encoder, "encoder") and hasattr(
                core.text_encoder.encoder, "layer"):
            for layer in core.text_encoder.encoder.layer[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True
        if hasattr(core, "text_encoder") and hasattr(core.text_encoder, "pooler"):
            for p in core.text_encoder.pooler.parameters():
                p.requires_grad = True

        # 4) 图像编码器：解冻最后若干块 + norm/head（若存在）
        if hasattr(core, "image_encoder") and hasattr(core.image_encoder, "blocks"):
            for block in core.image_encoder.blocks[-4:]:  # 4 可按需调整
                for p in block.parameters():
                    p.requires_grad = True
        if hasattr(core, "image_encoder") and hasattr(core.image_encoder, "norm"):
            for p in core.image_encoder.norm.parameters():
                p.requires_grad = True
        if hasattr(core, "image_encoder") and hasattr(core.image_encoder, "head"):
            for p in core.image_encoder.head.parameters():
                p.requires_grad = True

        # 只把需要训练的参数交给优化器
        trainable_params = [p for p in core.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        scaler = self.scaler

        local_rank = torch.cuda.current_device()
        if isinstance(self.model, DDP):
            # 旧壳子丢掉，防止状态不一致
            self.model = self.model.module

        # self.model.to(local_rank)
        self.model = DDP(
            core,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # <-- 解决未参与loss的参数
            gradient_as_bucket_view=False,  # <-- 也顺便去掉你之前的stride警告
            broadcast_buffers=False,  # 若BN等缓冲区不需要同步，关掉省通信
            static_graph=False
        )

        best_val_acc = 0.0
        patience_counter = 0
        history = []
        best_model_state = None

        for epoch in range(epochs):
            # 让 DistributedSampler 可复现实验
            running_loss = 0.0
            set_epoch_if_needed(dataloader, epoch)
            # —— 对齐各 rank 的 dataloader 步数（可选但推荐）——
            if dist.is_available() and dist.is_initialized():
                steps = torch.tensor([len(dataloader)], device=self.device)
                dist.all_reduce(steps, op=dist.ReduceOp.MIN)
                max_steps = int(steps.item())
            else:
                max_steps = len(dataloader)
            self.model.train()  # 前向/反向请用 self.model（DDP 包裹）

            pbar = tqdm(dataloader, desc=f"Fine-tuning Epoch {epoch + 1}/{epochs}",
                        disable=not is_main_process(), leave=True)

            # data_iter = iter(dataloader)

            for batch in pbar:
                images = batch['image'].to(self.device, non_blocking=True)
                ids = batch['text_input_ids'].to(self.device, non_blocking=True)
                mask = batch['text_attention_mask'].to(self.device, non_blocking=True)
                labels = batch['label_index'].to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(self.device.type == 'cuda')):
                    outputs = self.model(images, ids, mask)  # 用 DDP 包裹的模型
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    loss = self.criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # 全局平均的 step loss（显示更稳定）
                loss_avg = ddp_avg(loss.detach())
                running_loss += float(loss_avg)

                if is_main_process():
                    pbar.set_postfix_str(f"Loss: {float(loss_avg):.4f}")

                # 释放临时显存
                del outputs, logits, loss

            avg_loss = running_loss / max(1, len(dataloader))

            # —— 同步一下再做验证（可选，但更整齐）——
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            # 每 3 个 epoch 或最后一个 epoch 做一次验证（只打印一次）
            evaluate_interval = 3
            do_eval = ((epoch + 1) % evaluate_interval == 0) or (epoch == epochs - 1)
            if do_eval:
                val_acc, _ = self.evaluate_ddp(val_dataloader)  # 确保 evaluate 内部做了 DDP 规约
                device = torch.device(
                    f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
                stop_flag = torch.tensor([0], dtype=torch.int32, device=device)
                best_val_t = torch.tensor([best_val_acc], dtype=torch.float32, device=device)

                if is_main_process():
                    logger.info(f"Fine-tuning Epoch {epoch + 1} finished. Val Acc: {val_acc*100:.2f}%, ")
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                        best_val_t.fill_(best_val_acc)
                        best_model_state = {k: v.cpu() for k, v in core.state_dict().items()}
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            stop_flag.fill_(1)

                # 广播 stop_flag 和 best_val_acc
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(stop_flag, src=0)
                    dist.broadcast(best_val_t, src=0)
                    best_val_acc = float(best_val_t.item())

                if int(stop_flag.item()) == 1:
                    if is_main_process():
                        logger.info(f"Early stopping during fine-tuning at epoch {epoch + 1}.")
                    break
            else:
                if is_main_process():
                    logger.info(f"Fine-tuning Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")
                history.append({'epoch': epoch + 1, 'loss': avg_loss, 'val_acc': None})

        # 恢复最佳模型（仅 rank0 保存/恢复权重）
        if is_main_process() and best_model_state is not None:
            core.load_state_dict(best_model_state)

        return history

class FederatedKnowledgeDistillation:
    """联邦知识蒸馏主类"""

    def __init__(self, csv_path_a2: str, dataset_base_path_a2: str, dataset_name_a2: str,
                 csv_path_a1: str, dataset_base_path_a1: str, dataset_name_a1: str,
                 num_local_models: int = 3, local_epochs: int = 15, batch_size: int=32, device: torch.device = None,
                 data_distribution: str = 'iid', non_iid_alpha: float = 0.5,
                 use_data_parallel: bool = False, gpu_ids: List[int] = None, num_workers: int = 8,
                 local_init_mode: str = 'pretrained',   # 新增
                 global_init_mode: str = 'pretrained',   # 新增
                mode: str = 'train'  # 新增 mode 参数
                 ):
        """
        Args:
            csv_path_a2: 数据集A2的CSV文件路径（用于本地训练和蒸馏）
            dataset_base_path_a2: 数据集A2的基础路径
            dataset_name_a2: 数据集A2的名称
            csv_path_a1: 数据集A1的CSV文件路径（用于微调和测试）
            dataset_base_path_a1: 数据集A1的基础路径
            dataset_name_a1: 数据集A1的名称
            num_local_models: 本地模型数量
            device: 计算设备
            data_distribution: 数据分布类型 ('iid', 'quantity', 'class')
            non_iid_alpha: 非IID分布的alpha参数
            use_data_parallel: 是否使用DataParallel多GPU训练
            gpu_ids: GPU设备ID列表
            num_workers: 数据加载的工作线程数
            local_init_mode: 本地模型初始化方式 ('pretrained' 或 'random')
            global_init_mode: 全局模型初始化方式 ('pretrained' 或 'random')
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.use_data_parallel = use_data_parallel
        self.gpu_ids = gpu_ids if gpu_ids is not None else [device.index] if device.type == 'cuda' else [0]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name_a1 = dataset_name_a1
        self.num_local_models = num_local_models
        self.local_epochs = local_epochs
        self.data_distribution = data_distribution
        self.non_iid_alpha = non_iid_alpha
        self.mode = mode
        self.global_init_mode = global_init_mode
        self.local_init_mode = local_init_mode

        # --- New, enhanced data augmentation pipeline as requested ---
        if is_main_process():
            logger.info("Initializing with enhanced data augmentation pipeline for retinal images.")

        self.transform = A.Compose([
            # A.Resize(256, 256),
            # A.RandomCrop(224, 224),
            # A.HorizontalFlip(p=0.5),
            # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            # 1. 随机旋转：在-15度到+15度之间随机旋转。
            # border_mode=0 表示用黑色填充旋转后产生的边界，适合眼底图。
            # A.Rotate(limit=15, p=0.5, border_mode=0),
            # # 2. 随机水平翻转
            # A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()  # 注意这里是 ToTensorV2
        ])

        # self.full_train_dataset_a2 = BaseDataset(csv_path_a2, dataset_base_path_a2, self.transform, split='train')
        self.full_train_dataset_a2 = LMDBDataset(lmdb_path="/home/hhj/ZongShuJu/YanBing/DR_grading_AUG_LMDB",
                                                transform=self.transform,
                                                max_text_length=64,
                                                split="train")
        # 使用A2数据集的类别数量初始化模型
        self.num_classes = self.full_train_dataset_a2.num_classes
        # --- 根据模式加载数据和初始化模型 ---
        if self.mode == 'train':
            # 用 LMDB 的标签做分层切分
            indices = list(range(len(self.full_train_dataset_a2)))
            labels = self.full_train_dataset_a2.y.cpu().numpy()  # 见“改动 2”
            train_idx, val_idx = train_test_split(
                indices, test_size=0.1, random_state=42, stratify=labels,
            )
            self.train_dataset_a2 = self.full_train_dataset_a2
            self.val_dataset_a2 = LMDBDataset(lmdb_path="/home/hhj/ZongShuJu/YanBing/DR_grading_AUG_LMDB",
                                                transform=self.transform,
                                                max_text_length=64,
                                                split="val")

            # 蒸馏集加载完整 distill split
            self.distill_dataset_a2 = LMDBDataset(
                lmdb_path="/home/hhj/ZongShuJu/YanBing/DR_grading_AUG_LMDB",
                transform=self.transform,
                max_text_length=64,
                split='distill'
            )


            if is_main_process():
                logger.info(f"Dataset A2 loaded - Train: {len(self.train_dataset_a2)}, Val: {len(self.val_dataset_a2)}, Distill: {len(self.distill_dataset_a2)}, Distill Val: {len(self.val_dataset_a2)}")


            # 划分A2训练数据给本地模型
            self.local_datasets = self._split_train_dataset_a2()
            # 初始化本地模型
            self.local_models = [
                LocalModel(
                    i, self.num_classes, device, use_data_parallel=self.use_data_parallel, gpu_ids=self.gpu_ids,
                    local_epochs=self.local_epochs, use_pretrained=(local_init_mode == 'pretrained')
                ) for i in range(num_local_models)
            ]
            logger.info(
                f"Dataset A2 loaded - Train: {len(self.train_dataset_a2)}, Distill: {len(self.distill_dataset_a2)}")
        else: # 'finetune' 或 'test' 模式
            self.local_models = []
            self.local_datasets = []
            self.train_dataset_a2 = None
            self.val_dataset_a2 = None
            self.distill_dataset_a2 = None
            if is_main_process():
                logger.info(f"Running in '{self.mode}' mode. Skipping local models and dataset A2 loading.")

        # 加载数据集A1（用于微调和测试）
        self.finetune_dataset_a1 = BaseDataset(csv_path_a1, dataset_base_path_a1, self.transform, split='finetune')
        self.test_dataset_a1 = BaseDataset(csv_path_a1, dataset_base_path_a1, self.transform, split='test')




        if is_main_process():
            logger.info(f"Dataset A1 loaded - Finetune: {len(self.finetune_dataset_a1)}, Test: {len(self.test_dataset_a1)}")
        if is_main_process():
            logger.info(f"Number of classes: {self.num_classes}")



        # 初始化模型
        self.global_model = GlobalModel(
            self.num_classes,
            device,
            use_data_parallel=self.use_data_parallel,
            gpu_ids=self.gpu_ids,
            use_pretrained=(self.global_init_mode == 'pretrained')
        )

        self.local_models = [
            LocalModel(
                i,
                self.num_classes,
                device,
                use_data_parallel=self.use_data_parallel,
                gpu_ids=self.gpu_ids,
                local_epochs=self.local_epochs,
                use_pretrained=(self.local_init_mode == 'pretrained')
            )
            for i in range(num_local_models)
        ]


        self.metrics = {
            'local_models': [[] for _ in range(num_local_models)],
            'global_model': []
        }
        self.local_knowledge_list = [] # 存储本地知识

        # 在 FederatedKnowledgeDistillation 的 __init__ 方法中
        if is_main_process():
            print(f"Number of classes for distillation (from dataset A2): {self.num_classes}")
        if is_main_process():
            print(f"Global model classifier output features: {self.global_model._core().classifier.out_features}")

    def _split_train_dataset_a2(self):
        """将A2的训练数据分配给本地模型

        数据分配策略：
        - A2数据集的train部分用于本地模型训练
        - A2数据集的distill部分用于云端模型蒸馏
        """
        # 获取A2训练数据的索引
        total_train_size = len(self.train_dataset_a2)
        indices = list(range(total_train_size))
        random.shuffle(indices)

        # 根据数据分布模式分配数据给本地模型
        if self.data_distribution == 'iid':
            # --- 【IID 逻辑 (已修改)】---
            # 每个本地模型随机抽取总训练数据的40%，允许数据重叠
            logging.info(
                "Using IID data distribution: Each local model randomly samples 40% of the total training data (with overlap).")
            datasets = []
            # 计算每个本地模型应获取的样本数量（总数的40%）
            samples_per_model = int(total_train_size * 0.5)

            for i in range(self.num_local_models):
                # 从所有索引中随机抽取指定数量的样本，允许重复抽取
                local_indices = random.sample(indices, samples_per_model)
                local_dataset = torch.utils.data.Subset(self.train_dataset_a2, local_indices)
                datasets.append(local_dataset)
                logging.info(f"Local model {i}: {len(local_dataset)} samples (40% of total)")


        elif self.data_distribution == 'quantity':
            # 数量不平衡分布（Dirichlet分布）
            datasets = self._create_quantity_imbalanced_split(indices)

        elif self.data_distribution == 'class':
            # 类别不平衡分布
            datasets = self._create_class_imbalanced_split()

        else:
            raise ValueError(f"Unknown data distribution: {self.data_distribution}")

        return datasets

    def _create_quantity_imbalanced_split(self, indices):
        """创建数量不平衡的数据分割"""
        # 定义4个模型的分配比例：10%:15%:35%:40%
        # 如果模型数量不是4，则使用Dirichlet分布
        if self.num_local_models == 4:
            distribution_ratios = [0.10, 0.15, 0.35, 0.40]
            proportions = (np.array(distribution_ratios) * len(indices)).astype(int)
            # 确保所有数据都被分配（将余数加到最后一个模型）
            proportions[-1] = len(indices) - proportions[:-1].sum()
            if is_main_process():
                logger.info("Using fixed unbalanced distribution ratios for 4 models: 10%:15%:35%:40%")
        else:
            # 使用Dirichlet分布创建不平衡的数据分配
            proportions = np.random.dirichlet([self.non_iid_alpha] * self.num_local_models)
            proportions = (proportions * len(indices)).astype(int)
            # 确保所有数据都被分配
            proportions[-1] = len(indices) - proportions[:-1].sum()
            if is_main_process():
                logger.info(f"Using Dirichlet distribution (alpha={self.non_iid_alpha}) for {self.num_local_models} models")

        datasets = []
        start_idx = 0

        for i, prop in enumerate(proportions):
            if prop > 0:
                local_indices = indices[start_idx:start_idx + prop]
                local_dataset = torch.utils.data.Subset(self.train_dataset_a2, local_indices)
                datasets.append(local_dataset)
                start_idx += prop
                if is_main_process():
                    actual_ratio = len(local_dataset) / len(indices) * 100
                    logger.info(f"Local model {i}: {len(local_dataset)} samples ({actual_ratio:.1f}%)")
                # 打印本地模型的类别分布
                # 直接从 LMDBDataset 的 y 属性获取标签
                subset_labels = torch.tensor(
                    [self.train_dataset_a2.y[idx].item() for idx in local_indices],
                    dtype=torch.long
                )
                # 打印分布
                values, counts = torch.unique(subset_labels, return_counts=True)
                logger.info(f"Class distribution for local model {i}: " +
                            ", ".join([f"{int(v)}:{int(c)}" for v, c in zip(values, counts)]))
            else:
                # 如果分配的数据为0，给一个最小的数据集
                local_indices = indices[start_idx:start_idx + 1]
                local_dataset = torch.utils.data.Subset(self.train_dataset_a2, local_indices)
                datasets.append(local_dataset)
                start_idx += 1
                if is_main_process():
                    logger.info(f"Local model {i}: {len(local_dataset)} samples")
                # 打印本地模型的类别分布
                if local_indices:
                    # 直接从 LMDBDataset 的 y 属性获取标签
                    subset_labels = torch.tensor(
                        [self.train_dataset_a2.y[idx].item() for idx in local_indices],
                        dtype=torch.long
                    )
                    # 打印分布
                    values, counts = torch.unique(subset_labels, return_counts=True)
                    logger.info(f"Class distribution for local model {i}: " +
                                ", ".join([f"{int(v)}:{int(c)}" for v, c in zip(values, counts)]))

        return datasets

    def _create_class_imbalanced_split(self):
        """创建类别不平衡的数据分割

        策略:
        - 模型0优先分配类别0的50%,其余模型均分剩余50%
        - 模型1优先分配类别1的50%,其余模型均分剩余50%
        - 以此类推(循环分配)
        """
        # 按类别组织数据
        class_indices = defaultdict(list)
        for idx in range(len(self.train_dataset_a2)):
            label = self.train_dataset_a2.y[idx].item()
            class_indices[label].append(idx)

        num_classes = len(class_indices)
        datasets = [[] for _ in range(self.num_local_models)]

        if is_main_process():
            logger.info(f"Creating class imbalanced split for {self.num_local_models} models with {num_classes} classes")

        # 为每个类别分配数据
        for class_id in sorted(class_indices.keys()):
            class_data = class_indices[class_id]
            random.shuffle(class_data)
            total_samples = len(class_data)

            # 计算50%的样本数
            primary_size = total_samples // 2
            remaining_size = total_samples - primary_size

            # 确定该类别的主要模型(循环分配)
            primary_model_id = class_id % self.num_local_models

            # 主要模型获得50%
            datasets[primary_model_id].extend(class_data[:primary_size])

            # 其余模型均分剩余50%
            other_models = [i for i in range(self.num_local_models) if i != primary_model_id]
            samples_per_other = remaining_size // len(other_models)
            remainder = remaining_size % len(other_models)

            start_idx = primary_size
            for i, model_id in enumerate(other_models):
                end_idx = start_idx + samples_per_other + (1 if i < remainder else 0)
                datasets[model_id].extend(class_data[start_idx:end_idx])
                start_idx = end_idx

            if is_main_process():
                logger.info(f"Class {class_id}: Model {primary_model_id} gets {primary_size} samples (50%), "
                           f"others get ~{samples_per_other} samples each")

        # 创建Subset对象并打印统计信息
        result_datasets = []
        for i, indices in enumerate(datasets):
            local_dataset = torch.utils.data.Subset(self.train_dataset_a2, indices)
            result_datasets.append(local_dataset)

            if is_main_process():
                logger.info(f"Local model {i}: {len(local_dataset)} samples")

                # 打印类别分布
                if indices:
                    subset_labels = torch.tensor(
                        [self.train_dataset_a2.y[idx].item() for idx in indices],
                        dtype=torch.long
                    )
                    values, counts = torch.unique(subset_labels, return_counts=True)
                    dist_str = ", ".join([f"{int(v)}:{int(c)}" for v, c in zip(values, counts)])
                    logger.info(f"  Class distribution: {dist_str}")

        return result_datasets

    def train_local_models(self, epochs: int = 5, batch_size: int = 64, patience: int = 20):
        """
        [修改后版本]
        训练本地模型 - 使用QWK作为早停标准
        """
        if is_main_process():
            logger.info("Training local models on dataset A2...")

        for i, (local_model, local_dataset) in enumerate(zip(self.local_models, self.local_datasets)):
            if is_main_process():
                logger.info(f"Training Local Model {i + 1}/{len(self.local_models)}")

            train_dataloader = build_dataloader(local_dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
            val_dataloader = build_dataloader(self.val_dataset_a2, batch_size=batch_size, shuffle=False,
                                              num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

            model_metrics = []
            best_val_qwk = -1.0  # QWK的范围是[-1, 1]，初始值设为-1
            patience_counter = 0
            best_model_state = None

            for epoch in range(epochs):
                set_epoch_if_needed(train_dataloader, epoch)
                train_loss, train_accuracy = local_model.train_epoch_ddp(train_dataloader)

                # 验证，现在会返回 acc 和 qwk
                val_acc, val_qwk = local_model.evaluate_ddp(val_dataloader)

                if is_main_process():
                    # 打印所有关键指标
                    logger.info(
                        f"Local Model {i + 1} Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy * 100:.2f}%, Val Acc={val_acc * 100:.2f}%, Val QWK={val_qwk:.4f}")

                # === 同步早停逻辑 (基于QWK) ===
                device = self.device
                stop_flag = torch.tensor([0], dtype=torch.int32, device=device)
                best_val_t = torch.tensor([best_val_qwk], dtype=torch.float32, device=device)

                if is_main_process():
                    # 主进程根据 QWK 更新早停状态
                    if val_qwk > best_val_qwk:
                        best_val_qwk = val_qwk
                        patience_counter = 0
                        best_val_t.fill_(best_val_qwk)
                        # 保存模型状态到CPU，避免显存占用
                        best_model_state = {k: v.cpu() for k, v in unwrap_model(local_model.model).state_dict().items()}
                    else:
                        patience_counter += 1

                    if patience_counter >= patience and epoch > 5:
                        stop_flag.fill_(1)

                # 将早停标志和最佳分数广播给所有进程
                if is_distributed_world():
                    dist.broadcast(stop_flag, src=0)
                    dist.broadcast(best_val_t, src=0)

                best_val_qwk = float(best_val_t.item())
                if int(stop_flag.item()) == 1:
                    if is_main_process():
                        logger.info(
                            f"Early stopping for Local Model {i + 1} at epoch {epoch + 1} with best Val QWK: {best_val_qwk:.4f}")
                    break

                model_metrics.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_accuracy,
                    'val_acc': val_acc,
                    'val_qwk': val_qwk
                })

            # 主进程恢复最佳模型权重
            if is_main_process() and best_model_state is not None:
                unwrap_model(local_model.model).load_state_dict(best_model_state)

            self.metrics['local_models'][i] = model_metrics
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def perform_knowledge_distillation(self, epochs: int = 10, batch_size: int = 32, patience: int = 3):
        """A 版：teacher_probs 常驻 GPU，一次广播，训练阶段显存内索引"""
        import os
        import torch
        import torch.distributed as dist
        from torch.utils.data import DataLoader

        # --- helper: 统一 image_id 规范 ---
        def norm_img_id(x: str) -> str:
            return os.path.basename(str(x)).lower()  # 取 basename + 小写，保留扩展名

        # --- helper: GPU 上的“字典”包装（兼容旧的 teacher_map[img_id] 写法）---
        class TeacherLookupGPU:
            def __init__(self, probs_gpu: torch.Tensor, id2idx: dict[str, int]):
                self.probs = probs_gpu  # [N, C] (GPU, fp16/32均可)
                self.id2idx = id2idx  # str -> int（CPU小字典）

            def __contains__(self, k: str) -> bool:
                return norm_img_id(k) in self.id2idx

            def __getitem__(self, k: str) -> torch.Tensor:
                # 返回该 id 对应一行 [C]（GPU Tensor）
                idx = self.id2idx[norm_img_id(k)]
                return self.probs[idx]

            # 可选：批量取（若你后续想改 distill_knowledge 用它会更快）
            def gather(self, id_list: list[str]) -> torch.Tensor:
                idx = [self.id2idx[norm_img_id(x)] for x in id_list]
                idx = torch.as_tensor(idx, device=self.probs.device, dtype=torch.long)
                return self.probs.index_select(0, idx)

        if is_main_process():
            logger.info("Performing knowledge distillation on dataset A2...")

        # ------------------------------------------------------------
        # A) rank0 构建全量 teacher_probs 与 id_to_index
        #    用“非分布式”的顺序 DataLoader（避免分片/打乱差异）
        # ------------------------------------------------------------
        need_dist = dist.is_available() and dist.is_initialized()
        teacher_probs_cpu = None  # [N, C] (CPU, float16)
        id_to_index = None  # {norm_id: row_index}

        if (not need_dist) or dist.get_rank() == 0:
            seq_loader = DataLoader(
                self.distill_dataset_a2,
                batch_size=batch_size,
                shuffle=False,  # 必须顺序
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True
            )

            local_knowledge_list = []
            for i, local_model in enumerate(self.local_models):
                knowledge = local_model.extract_knowledge(
                    seq_loader)  # {'soft_predictions':[N,C], 'image_ids':list[str]}
                local_knowledge_list.append(knowledge)
                # 释放本地模型显存
                local_model.model.to("cpu")
                del local_model.model
                torch.cuda.empty_cache()

            # 平均 teacher 预测
            teacher_preds = [k['soft_predictions'].float() for k in local_knowledge_list]  # list of [N, C] (CPU)
            avg_teacher = torch.stack(teacher_preds).mean(0).contiguous()  # [N, C] float32 CPU
            teacher_probs_cpu = avg_teacher.half()

            # 统一化 ID，建立映射
            all_image_ids = [norm_img_id(x) for x in local_knowledge_list[0]['image_ids']]  # len N
            id_to_index = {img_id: idx for idx, img_id in enumerate(all_image_ids)}

            if is_main_process():
                logger.info(f"[distill] Built teacher_map with {len(id_to_index)} images.")

            # （如果后续 fine-tune 用得到）
            self.local_knowledge_list = local_knowledge_list
            # 及时释放临时内存
            del local_knowledge_list, teacher_preds, avg_teacher
            import gc
            gc.collect()

        # ------------------------------------------------------------
        # B) 把整块 teacher_probs 搬到各 GPU，并用 dist.broadcast 广播
        # ------------------------------------------------------------
        import torch.distributed as dist

        dev = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
        probs_gpu = None  # <- 先占位，避免 UnboundLocalError

        if need_dist and dist.is_initialized():
            # 先让所有 rank 知道形状
            if dist.get_rank() == 0:
                assert teacher_probs_cpu is not None, "rank0 必须先构建 teacher_probs_cpu"
                shape_tensor = torch.tensor(teacher_probs_cpu.shape, dtype=torch.long, device=dev)
            else:
                shape_tensor = torch.empty(2, dtype=torch.long, device=dev)
            dist.broadcast(shape_tensor, src=0)
            N = int(shape_tensor[0].item());
            C = int(shape_tensor[1].item())

            # 分配/设置 GPU 缓冲，然后广播整块概率
            if dist.get_rank() == 0:
                probs_gpu = teacher_probs_cpu.to(dev, non_blocking=True)
            else:
                probs_gpu = torch.empty((N, C), dtype=torch.float16, device=dev)
            dist.broadcast(probs_gpu, src=0)

            # 广播 ID 列表（bytes）
            if dist.get_rank() == 0:
                ids_bytes = "\n".join(list(id_to_index.keys())).encode("utf-8")
                ids_tensor_cpu = torch.frombuffer(bytearray(ids_bytes), dtype=torch.uint8)  # CPU
                ids_tensor = ids_tensor_cpu.to(dev, non_blocking=True)
                length_t = torch.tensor([ids_tensor.numel()], dtype=torch.long, device=dev)
            else:
                ids_tensor = torch.empty(0, dtype=torch.uint8, device=dev)
                length_t = torch.empty(1, dtype=torch.long, device=dev)

            dist.broadcast(length_t, src=0)
            if dist.get_rank() != 0:
                ids_tensor = torch.empty(int(length_t.item()), dtype=torch.uint8, device=dev)
            dist.broadcast(ids_tensor, src=0)

            ids_list = bytes(ids_tensor.tolist()).decode("utf-8").split("\n")
            id_to_index = {img_id: idx for idx, img_id in enumerate(ids_list)}

            # rank0 不再需要 CPU 副本
            if dist.get_rank() == 0:
                del teacher_probs_cpu
            torch.cuda.synchronize()

        else:
            # 单进程/未初始化分布式：直接搬到 GPU
            assert teacher_probs_cpu is not None, "单卡/未分布式时 teacher_probs_cpu 应当已构建"
            probs_gpu = teacher_probs_cpu.to(dev, non_blocking=True)
            del teacher_probs_cpu

        # 最后防御性检查，避免未赋值使用
        if probs_gpu is None:
            raise RuntimeError("probs_gpu 未成功创建，请检查 need_dist / rank / teacher_probs_cpu 的构建流程。")

        # 把“GPU 版 teacher_map”准备好（兼容旧接口）
        teacher_map = TeacherLookupGPU(probs_gpu, id_to_index)

        # ------------------------------------------------------------
        # C) DDP 训练/验证 DataLoader
        # ------------------------------------------------------------
        distill_dataloader = build_dataloader(
            self.distill_dataset_a2, batch_size=batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=True
        )
        val_dataloader = build_dataloader(
            self.val_dataset_a2, batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=True
        )

        # ------------------------------------------------------------
        # D) 训练循环（保持你的早停/验证逻辑）
        # ------------------------------------------------------------
        global_metrics = []
        best_val_acc = 0.0
        best_val_qwk = 0.0
        patience_counter = 0
        best_model_state = None

        device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device(
            "cpu")

        for epoch in range(epochs):
            set_epoch_if_needed(distill_dataloader, epoch)
            train_loss = self.global_model.distill_knowledge(teacher_map, distill_dataloader)
            val_acc, val_qwk = self.global_model.evaluate_ddp(val_dataloader)  # 内部已做 all_reduce

            # rank0 决策 early stop
            stop_flag = torch.tensor([0], dtype=torch.int32, device=device)
            best_val_t = torch.tensor([best_val_qwk], dtype=torch.float32, device=device)

            if _is_main():
                logger.info(f"[distill] epoch={epoch} loss={train_loss:.6f} val_acc={val_acc:.4f} val_qwk={val_qwk:.4f}")
                if val_qwk > best_val_qwk:
                    best_val_qwk = val_qwk
                    patience_counter = 0
                    best_val_t.fill_(best_val_qwk)
                    best_model_state = {k: v.cpu() for k, v in self.global_model._core().state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        stop_flag.fill_(1)

            # 广播 stop_flag 与 best_val
            if _is_dist():
                dist.broadcast(stop_flag, src=0)
                dist.broadcast(best_val_t, src=0)
                best_val_qwk = float(best_val_t.item())

            if int(stop_flag.item()) == 1:
                if _is_main():
                    logger.info(f"[distill] early stop at epoch {epoch}")
                break

        # rank0 恢复最佳模型
        if _is_main() and best_model_state is not None:
            self.global_model._core().load_state_dict(best_model_state)

        return best_val_qwk

    def perform_fine_tuning(self, batch_size: int = 16, epochs: int = 3, patience: int = 10):
        """在A1数据集的finetune部分进行微调，并在A1的测试集上验证"""
        if is_main_process():
            logger.info("Performing fine-tuning on dataset A1...")

        # 若 A1 类别数不同，适配分类头
        a1_num_classes = self.finetune_dataset_a1.num_classes
        if a1_num_classes != self.num_classes:
            logger.warning(f"A1 num_classes {a1_num_classes} != distilled model num_classes {self.num_classes}, adapting classifier.")
            self.global_model.adapt_num_classes(a1_num_classes)
            self.num_classes = a1_num_classes

        # 使用A1数据集的finetune部分进行微调
        fine_tune_dataloader = build_dataloader(self.finetune_dataset_a1, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

        # 使用A1的测试集作为微调阶段的验证集
        test_dataloader_a1 = build_dataloader(self.test_dataset_a1, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

        # 执行微调，并传入验证集
        fine_tune_history = self.global_model.fine_tune(
            fine_tune_dataloader,
            val_dataloader=test_dataloader_a1,
            epochs=epochs,
            patience=patience
        )

        self.metrics['fine_tuning_history'] = fine_tune_history

        if is_main_process():

            logger.info("Fine-tuning completed.")

    def save_global_model(self, save_dir: str = "saved_models", suffix: str = ""):
        """保存全局模型"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(save_dir, f'global_model_{self.dataset_name_a1}_{self.data_distribution}_{timestamp}{suffix}.pth')

        # Save model state
        torch.save({
            'model_state_dict': self.global_model.model.state_dict(),
            'num_classes': self.num_classes,
            'class_names': self.test_dataset_a1.labels,
            'timestamp': timestamp,
            'dataset_name': self.dataset_name_a1,
            'data_distribution': self.data_distribution
        }, model_path)

        if is_main_process():

            logger.info(f"Global model saved to {model_path}")

    def load_global_model(self, model_path: str):
        """加载全局模型"""
        if not os.path.exists(model_path):
            logger.error(f"Model path not found: {model_path}")
            raise FileNotFoundError(f"Model path not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device,weights_only=False)

        # 检查类别数是否匹配
        if checkpoint['num_classes'] != self.num_classes:
            logger.error(
                f"Model was trained with {checkpoint['num_classes']} classes, but dataset has {self.num_classes} classes.")
            raise ValueError("Number of classes in model and dataset do not match.")

        self.global_model.model.load_state_dict(checkpoint['model_state_dict'])
        if is_main_process():
            logger.info(f"Global model loaded from {model_path}")

    def plot_metrics(self, plots_dir: str = "training_plots"):
        """绘制训练和蒸馏过程的指标图"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(plots_dir, exist_ok=True)

        # Plot local models training metrics
        plt.figure(figsize=(12, 8))
        for i, model_metrics in enumerate(self.metrics['local_models']):
            epochs = [m['epoch'] for m in model_metrics]
            val_acc = [m['val_acc'] for m in model_metrics]
            plt.plot(epochs, val_acc, label=f'Local Model {i + 1}')

        plt.title('Local Models Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{plots_dir}/local_models_accuracy_{self.dataset_name_a1}_{timestamp}.png')
        plt.close()

        plt.figure(figsize=(12, 8))
        epochs = [m['epoch'] for m in self.metrics['global_model']]
        accuracy = [m['accuracy'] for m in self.metrics['global_model']]
        loss = [m['loss'] for m in self.metrics['global_model']]

        plt.subplot(2, 1, 1)
        plt.plot(epochs, accuracy, 'b-', label='Test Accuracy')
        plt.title('Global Model Distillation Metrics')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(epochs, loss, 'r-', label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'{plots_dir}/global_model_metrics_{self.dataset_name_a1}_{timestamp}.png')
        plt.close()

        # ROC曲线
        plt.figure(figsize=(12, 8))
        test_dataloader = build_dataloader(self.test_dataset_a1, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True,
            pin_memory=True)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.local_models) + 1))

        # 绘制全局模型的 ROC
        fprs, tprs = self.global_model.evaluate_roc(test_dataloader)
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)

        for fpr, tpr in zip(fprs, tprs):
            mean_tpr += np.interp(mean_fpr, fpr, tpr)

        mean_tpr /= len(fprs)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0

        plt.plot(mean_fpr, mean_tpr,
                 color=colors[-1],
                 label='Global Model',
                 linewidth=2)

        # 添加对角线
        plt.plot([0, 1], [0, 1], 'k--', label='Random')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True)

        plt.savefig(f'{plots_dir}/roc_curves_{self.dataset_name_a1}_{timestamp}.png')
        plt.close()

    def run_experiment(self, mode: str = 'train', model_path: Optional[str] = None, local_epochs: int = 5,
                       distill_epochs: int = 10, batch_size: int = 32, fine_tune_epochs: int = 3):
        """运行实验并保存结果"""
        # 创建结果保存目录结构：f2med/数据集名称/参数组合/
        # 创建基于参数的子目录名
        param_subdir = f"nodes_{self.num_local_models}_{self.data_distribution}"

        # 完整路径
        base_save_dir = os.path.join("f2med_logs", self.dataset_name_a1, param_subdir)
        os.makedirs(base_save_dir, exist_ok=True)

        if mode == 'train':
            if is_main_process():
                logger.info("Starting Federated Knowledge Distillation Experiment on Retinal OCT Dataset in TRAIN mode")
            # 先本地训练本地模型
            if local_epochs > 0:
                self.train_local_models(epochs=local_epochs, batch_size=batch_size, patience=25)
            if distill_epochs > 0:
                # 然后执行知识蒸馏
                self.perform_knowledge_distillation(epochs=distill_epochs, batch_size=batch_size, patience=25)
                # 保存蒸馏后的模型
                if is_main_process():
                    logger.info("Saving the distilled global model...")
                self.save_global_model(save_dir=os.path.join(base_save_dir, "models"), suffix="_distilled")
            # 最后执行微调
            self.perform_fine_tuning(epochs=fine_tune_epochs, batch_size=batch_size, patience=25) # 使用较小的batch_size
            # 保存模型到参数特定目录
            self.save_global_model(save_dir=os.path.join(base_save_dir, "models"), suffix="_finetuned")
            # 绘制指标图到参数特定目录
            self.plot_metrics(plots_dir=os.path.join(base_save_dir, "plots"))
        elif mode == 'finetune':
            if model_path is None:
                logger.error("必须为微调模式提供模型路径。")
                return 0.0
            if is_main_process():
                logger.info(f"在微调模式下使用模型在数据集A1上开始微调: {model_path}")
            self.load_global_model(model_path)
            # 直接执行微调
            self.perform_fine_tuning(epochs=fine_tune_epochs, batch_size=batch_size, patience=10)
            # 保存微调后的模型
            self.save_global_model(save_dir=os.path.join(base_save_dir, "models"), suffix="_finetuned")
        elif mode == 'test':
            if model_path is None:
                logger.error("Model path must be provided for testing mode.")
                return 0.0
            if is_main_process():
                logger.info(f"Starting evaluation on Retinal OCT Dataset in TEST mode with model: {model_path}")
            self.load_global_model(model_path)
        else:
            logger.error(f"Invalid mode: {mode}. Choose 'train', 'test', or 'finetune'.")
            return 0.0

        test_dataloader = build_dataloader(self.test_dataset_a1, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
        test_acc, test_precision, test_recall, test_f1, test_specificity, test_auprc, test_auroc, y_true, y_pred, y_prob = evaluate(
            self.global_model.model, test_dataloader, self.device, self.num_classes,
            self.test_dataset_a1.labels, base_save_dir, epoch=0, model_name="global_model"
        )

        # 计算每个类别的指标
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        # 为每个类别计算指标
        per_class_metrics = []
        for class_idx in range(self.num_classes):
            class_name = self.test_dataset_a1.labels[class_idx]

            # 为当前类创建二分类标签
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)

            # 计算每个类的指标
            class_precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            class_recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            class_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

            # 计算特异性
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            class_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # 计算AUPRC
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_true_binary,
                y_prob[:, class_idx]
            )
            class_auprc = auc(recall_curve, precision_curve)

            # 计算AUROC
            try:
                class_auroc = roc_auc_score(y_true_binary, y_prob[:, class_idx])
            except:
                class_auroc = -1

            # 计算准确率
            class_accuracy = accuracy_score(y_true_binary, y_pred_binary)

            # 存储该类的所有指标
            class_metrics = {
                'class_name': class_name,
                'accuracy': class_accuracy,
                'precision': class_precision,
                'recall': class_recall,
                'f1': class_f1,
                'specificity': class_specificity,
                'auprc': class_auprc,
                'auroc': class_auroc
            }
            per_class_metrics.append(class_metrics)

        # 整体指标
        overall_metrics = {
            'class_name': 'Overall',
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'specificity': test_specificity,
            'auprc': test_auprc,
            'auroc': test_auroc
        }

        # 将整体指标添加到每类指标列表中
        per_class_metrics.append(overall_metrics)

        # 转换为DataFrame并保存
        df_results = pd.DataFrame(per_class_metrics)
        save_path = os.path.join(base_save_dir, 'detailed_metrics.csv')
        df_results.to_csv(save_path, index=False)

        if is_main_process():

            logger.info(f"Detailed metrics saved to {save_path}")
        if is_main_process():
            logger.info(f"Final Global Model Test Accuracy: {test_acc:.4f}")
        if is_main_process():
            logger.info(f"Test Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        if is_main_process():
            logger.info(f"Specificity: {test_specificity:.4f}, AUPRC: {test_auprc:.4f}, AUROC: {test_auroc:.4f}")
        if is_main_process():
            logger.info(f"Results saved to {base_save_dir}")

        return test_acc

def evaluate(model, dataloader, device, num_classes, class_names=None, save_dir=None, epoch=None, model_name="model"):
    """评估函数"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['label_index'].to(device)

            outputs = model(images, text_input_ids, text_attention_mask)
            probs = torch.softmax(outputs['logits'], dim=1)
            preds = outputs['logits'].argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算基本指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # 计算特异性
    cm = confusion_matrix(all_labels, all_preds)
    specificity_list = []
    for i in range(num_classes):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_list.append(specificity)
    specificity = np.mean(specificity_list)

    # 计算AUPRC
    y_true_onehot = np.zeros((len(all_labels), num_classes))
    y_true_onehot[np.arange(len(all_labels)), all_labels] = 1
    auprc_list = []
    for class_idx in range(num_classes):
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true_onehot[:, class_idx],
            np.array(all_probs)[:, class_idx]
        )
        class_auprc = auc(recall_curve, precision_curve)
        auprc_list.append(class_auprc)
    auprc = np.mean(auprc_list)

    # 计算AUROC
    try:
        auroc = roc_auc_score(pd.get_dummies(all_labels), all_probs, average='macro', multi_class='ovr')
    except:
        auroc = -1

    # 如果提供了保存目录和类别名称，则保存混淆矩阵和ROC数据
    if save_dir and epoch is not None:
        os.makedirs(save_dir, exist_ok=True)

        # 保存原始预测数据
        np.save(os.path.join(save_dir, f'{model_name}_y_true.npy'), np.array(all_labels))
        np.save(os.path.join(save_dir, f'{model_name}_y_pred.npy'), np.array(all_preds))
        np.save(os.path.join(save_dir, f'{model_name}_y_prob.npy'), np.array(all_probs))

        # 保存混淆矩阵原始数据
        np.save(os.path.join(save_dir, f'{model_name}_confusion_matrix.npy'), cm)

        # 计算并保存ROC曲线数据
        fpr_dict = {}
        tpr_dict = {}
        roc_auc_dict = {}

        # 对每个类别计算ROC
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(
                (np.array(all_labels) == i).astype(int),
                np.array(all_probs)[:, i]
            )
            fpr_dict[i] = fpr
            tpr_dict[i] = tpr
            roc_auc_dict[i] = auc(fpr, tpr)

        # 保存ROC数据
        with open(os.path.join(save_dir, f'{model_name}_roc_data.pkl'), 'wb') as f:
            pickle.dump({
                'fpr': fpr_dict,
                'tpr': tpr_dict,
                'roc_auc': roc_auc_dict,
                'auroc': auroc
            }, f)

        # 绘制并保存混淆矩阵
        if class_names:
            cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            plt.figure(figsize=(10, 8))
            plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix (Normalized) - Epoch {epoch}')
            plt.colorbar()
            tick_marks = np.arange(num_classes)
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            # 添加文本标注
            thresh = cm_normalized.max() / 2.
            for i in range(cm_normalized.shape[0]):
                for j in range(cm_normalized.shape[1]):
                    plt.text(j, i, f"{cm_normalized[i, j]:.2f}",
                            ha="center", va="center",
                            color="white" if cm_normalized[i, j] > thresh else "black")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix_epoch_{epoch}.png'))
            plt.close()

            # 绘制并保存ROC曲线
            plt.figure(figsize=(10, 8))
            colors = plt.cm.rainbow(np.linspace(0, 1, num_classes + 1))

            # 绘制每个类别的ROC曲线
            for i, color in zip(range(num_classes), colors):
                plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=2,
                         label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc_dict[i]:.2f})')

            # 绘制对角线
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves - Epoch {epoch}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{model_name}_roc_curves_epoch_{epoch}.png'))
            plt.close()


    return acc, precision, recall, f1, specificity, auprc, auroc, all_labels, all_preds, all_probs

# 使用示例
def main():
    # DDP setup —— 注意先用 want_distributed()，不依赖已初始化
    if want_distributed():
        local_rank = ddp_setup()
    else:
        local_rank = 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Federated Knowledge Distillation for Retinal OCT Classification with Two Datasets")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'finetune'],
                        help='Operating mode: train, test, or finetune')
    parser.add_argument('--model_path', type=str, default=f"/home/hhj/BiomedGPT_HF/f2med/Augemnted_ocular_diseases_dataset/nodes_4_iid/models/global_model_Augemnted_ocular_diseases_dataset_iid_20250825_175021_distilled.pth",
                        help='Path to a pre-trained global model for testing or fine-tuning')

    # 数据集A2相关参数（用于本地训练和蒸馏）
    parser.add_argument('--dataset_type_a2', type=str, default='YanBing', choices=['YanBing', 'WeiBing'],
                        help='Type of dataset A2 to use for local training and distillation')
    parser.add_argument('--dataset_name_a2', type=str, default='Eyepacs_dataset',
                        help='Name of the dataset A2 folder')
    parser.add_argument('--dataset_subfolder_a2', type=str, default=None,
                        help='Subfolder for dataset A2, used for WeiBing datasets')
    parser.add_argument('--csv_path_a2', type=str, default=None, help='Path to the dataset A2 CSV file')

    # 数据集A1相关参数（用于微调和测试）
    parser.add_argument('--dataset_type_a1', type=str, default='YanBing', choices=['YanBing', 'WeiBing'],
                        help='Type of dataset A1 to use for fine-tuning and testing')
    parser.add_argument('--dataset_name_a1', type=str, default='APTOS2019_dataset',
                        help='Name of the dataset A1 folder')
    parser.add_argument('--dataset_subfolder_a1', type=str, default=None,
                        help='Subfolder for dataset A1, used for WeiBing datasets')
    parser.add_argument('--csv_path_a1', type=str, default=None, help='Path to the dataset A1 CSV file')

    # 训练参数
    parser.add_argument('--num_local_models', type=int, default=4, help='Number of local models')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--local_epochs', type=int, default=0, help='Number of local training epochs')
    parser.add_argument('--distill_epochs', type=int, default=1, help='Number of distillation epochs')
    parser.add_argument('--fine_tune_epochs', type=int, default=300, help='Number of fine-tuning epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    parser.add_argument('--distribution_mode', type=str, default='iid', choices=['iid', 'quantity', 'class'],
                        help='Data distribution mode: iid, quantity (imbalanced quantity, balanced classes), class (balanced quantity, imbalanced classes)')
    parser.add_argument('--imbalance_alpha', type=float, default=0.5,
                        help='Alpha parameter for controlling imbalance (Dirichlet)')
    # 新增：模型初始化方式
    parser.add_argument('--local_init_mode', type=str, default='pretrained', choices=['pretrained','random'],
                        help='本地模型参数初始化方式: pretrained 或 random')
    parser.add_argument('--global_init_mode', type=str, default='pretrained', choices=['pretrained','random'],
                        help='全局模型参数初始化方式: pretrained 或 random')

    # 多GPU参数
    parser.add_argument('--use_data_parallel', action='store_true', default=False,
                        help='Whether to use DataParallel for multi-GPU training')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,5,6,7',
                        help='Comma-separated list of GPU IDs to use (e.g., "6,7")')

    args = parser.parse_args()


    # 创建基于实验参数的日志文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"f2med_logs/experiment_{timestamp}_{args.dataset_name_a1}_{args.distribution_mode}_nodes{args.num_local_models}.log"

    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    # 包含文件输出
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if is_main_process():

        logger.info(f"Experiment log will be saved to: {log_filename}")

    # 数据集A2
    base_path_a2 = f"/home/hhj/ZongShuJu/{args.dataset_type_a2}/{args.dataset_name_a2}"
    # if args.dataset_type_a2 == 'WeiBing' and args.dataset_subfolder_a2:
    #     base_path_a2 = os.path.join(base_path_a2, args.dataset_subfolder_a2)
    csv_path_a2 = args.csv_path_a2 if args.csv_path_a2 else os.path.join(base_path_a2, 'dataset.csv')

    # 数据集A1
    base_path_a1 = f"/home/hhj/ZongShuJu/{args.dataset_type_a1}/{args.dataset_name_a1}"
    # if args.dataset_type_a1 == 'WeiBing' and args.dataset_subfolder_a1:
    #     base_path_a1 = os.path.join(base_path_a1, args.dataset_subfolder_a1)
    csv_path_a1 = args.csv_path_a1 if args.csv_path_a1 else os.path.join(base_path_a1, 'dataset.csv')

    # 打印数据集信息
    if is_main_process():
        logger.info(f"Dataset A2 (for local training & distillation): {args.dataset_name_a2}")
    if is_main_process():
        logger.info(f"Dataset A2 path: {base_path_a2}")
    if is_main_process():
        logger.info(f"Dataset A2 CSV: {csv_path_a2}")
    if is_main_process():
        logger.info(f"Dataset A1 (for fine-tuning & testing): {args.dataset_name_a1}")
    if is_main_process():
        logger.info(f"Dataset A1 path: {base_path_a1}")
    if is_main_process():
        logger.info(f"Dataset A1 CSV: {csv_path_a1}")

    # 联邦学习系统
    fed_system = FederatedKnowledgeDistillation(
        csv_path_a2=csv_path_a2,
        dataset_base_path_a2=base_path_a2,
        dataset_name_a2=args.dataset_name_a2,
        csv_path_a1=csv_path_a1,
        dataset_base_path_a1=base_path_a1,
        dataset_name_a1=args.dataset_name_a1,
        num_local_models=args.num_local_models,
        local_epochs = args.local_epochs,
        batch_size=args.batch_size,
        device=device,
        data_distribution=args.distribution_mode,
        non_iid_alpha=args.imbalance_alpha,
        use_data_parallel=args.use_data_parallel,
        gpu_ids=[0,1,2,3],
        num_workers=args.num_workers,
        local_init_mode=args.local_init_mode,
        global_init_mode=args.global_init_mode,
        mode=args.mode,
    )

    # 运行实验
    final_accuracy = fed_system.run_experiment(
        mode=args.mode,
        model_path=args.model_path,
        local_epochs=args.local_epochs,
        distill_epochs=args.distill_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size
    )

    if is_main_process():

        print(f"Experiment completed. Final accuracy: {100*final_accuracy:.2f}%")


if __name__ == "__main__":
    main()

