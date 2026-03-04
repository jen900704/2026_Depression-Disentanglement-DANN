"""
Speaker Probe — 24 個模型 Speaker Accuracy 統一評估
=====================================================

對應論文 Table 1 的三組 × 4 variants × 2 scenarios：

  Group 1: Wav2Vec-Linear Probing (G1)
    Orig/Frozen  → sklearn LogReg on frozen W2V，768 維，無模型檔
    Orig/FT      → HF Trainer checkpoint，768 維，mean pool
    DANN/Frozen  → manual PyTorch loop，無 torch.save → ⚠️ 無法 probe
    DANN/FT      → HF Trainer checkpoint，768 維，mean pool（有 spk_classifier 但 probe 用 pooled_output）

  Group 2: Wav2Vec-SLS (G2)
    Orig/Frozen  → HF Trainer checkpoint，128 維，SLS+down_proj
    Orig/FT      → HF Trainer checkpoint + .pth，128 維，SLS+down_proj
    DANN/Frozen  → HF Trainer .pth（down_proj.state_dict()），128 維
    DANN/FT      → HF Trainer checkpoint + .pth，128 維，SLS+down_proj

  Group 3: XLSR-eGeMAPS (G3)
    Orig/Frozen  → HF Trainer checkpoint，256 維，concat+down_proj（eGeMAPS 補零）
    Orig/FT      → HF Trainer checkpoint，256 維，concat+down_proj（eGeMAPS 補零）
    DANN/Frozen  → HF Trainer .pth（down_proj.state_dict()），256 維
    DANN/FT      → HF Trainer checkpoint + .pth，256 維

Probe 邏輯：
  Scenario A：probe train = 151 control（CSV_A_TRAIN），test = 38 unseen（CSV_A_TEST）→ 預期 ≈ 0%
  Scenario B：probe train = 38 target Historical（CSV_B_TRAIN 中篩出 target speakers），
              test = 38 target Current（CSV_B_TEST）→ 越高代表 leakage 越嚴重

注意：
  G1 DANN/Frozen A/B 的 manual loop 腳本不存模型，無法 probe，自動標記 N/A。
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from dataclasses import dataclass
from typing import Optional, Set, Tuple
from tqdm import tqdm
from torch.autograd import Function
from transformers import (
    Wav2Vec2Processor, Wav2Vec2FeatureExtractor,
    Wav2Vec2Model, Wav2Vec2Config,
    Wav2Vec2PreTrainedModel, AutoConfig,
)
from transformers.file_utils import ModelOutput
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
#  路徑設定（執行前請確認）
# ============================================================

AUDIO_ROOT = ""   # CSV 內已是絕對路徑，保持空字串

CSV_A_TRAIN = "./experiment_sisman_scientific/scenario_A_screening/train.csv"
CSV_A_TEST  = "./experiment_sisman_scientific/scenario_A_screening/test.csv"
CSV_B_TRAIN = "./experiment_sisman_scientific/scenario_B_monitoring/train.csv"
CSV_B_TEST  = "./experiment_sisman_scientific/scenario_B_monitoring/test.csv"

# ── G1: Wav2Vec-Linear Probing ───────────────────────────────
G1_ORIG_FT_A = "output_scenario_A_v3/run_1/best_model"
G1_ORIG_FT_B = "output_scenario_B_v2/run_1/best_model"

# G1 DANN/FT (Fixed v6 path)
G1_DANN_FT_A = "output_dann_finetune_A_v6/run_1/best_model"
G1_DANN_FT_B = "output_dann_finetune_B_v6/run_1/best_model"

# ── G2: Wav2Vec-SLS ──────────────────────────────────────────
G2_ORIG_FROZEN_A = "output_huang_sls_A/run_1/best_model"
G2_ORIG_FROZEN_B = "output_huang_sls_B/run_1/best_model"

G2_ORIG_FT_A_PTH = "output_huang_sls_ft_A/sls_ft_A_shared_encoder_run_1.pth"
G2_ORIG_FT_B_PTH = "output_huang_sls_ft_B/sls_ft_B_shared_encoder_run_1.pth"

# Missing DANN variables added back:
G2_DANN_FROZEN_A_PTH = "output_huang_sls_dann_A/huang_sls_dann_A_shared_encoder_run_1.pth"
G2_DANN_FROZEN_B_PTH = "output_huang_sls_dann_B/huang_sls_dann_B_shared_encoder_run_1.pth"

G2_DANN_FT_A_PTH = "output_huang_sls_dann_finetune_A/sls_dann_finetune_A_shared_encoder_run_1.pth"
G2_DANN_FT_B_PTH = "output_huang_sls_dann_finetune_B/sls_dann_finetune_B_shared_encoder_run_1.pth"

# ── G3: XLSR-eGeMAPS ─────────────────────────────────────────
G3_ORIG_FROZEN_A = "output_xlsr_egemaps_A/run_1/best_model"
G3_ORIG_FROZEN_B = "output_xlsr_egemaps_B/run_1/best_model"

# Fixed ft path
G3_ORIG_FT_A = "output_xlsr_egemaps_ft_A/run_1/best_model"
G3_ORIG_FT_B = "output_xlsr_egemaps_ft_B/run_1/best_model"

G3_DANN_FROZEN_A_PTH = "output_xlsr_egemaps_dann_A/xlsr_egemaps_dann_A_shared_encoder_run_1.pth"
G3_DANN_FROZEN_B_PTH = "output_xlsr_egemaps_dann_B/xlsr_egemaps_dann_B_shared_encoder_run_1.pth"

G3_DANN_FT_A_PTH = "output_xlsr_egemaps_dann_finetune_A/xlsr_egemaps_dann_finetune_A_shared_encoder_run_1.pth"
G3_DANN_FT_B_PTH = "output_xlsr_egemaps_dann_finetune_B/xlsr_egemaps_dann_finetune_B_shared_encoder_run_1.pth"

MODEL_NAME_W2V  = "facebook/wav2vec2-base"
MODEL_NAME_XLSR = "facebook/wav2vec2-xls-r-300m"
TOTAL_RUNS      = 1
EGEMAPS_DIM     = 88

# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  裝置: {DEVICE}")


# ============================================================
#  模型結構定義
# ============================================================

# ── G1 Orig/FT：標準 Wav2Vec2 fine-tune（768 維 mean pool）──

class Wav2Vec2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense    = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout  = nn.Dropout(config.final_dropout if hasattr(config, "final_dropout") else 0.1)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x); x = self.dense(x); x = torch.tanh(x)
        x = self.dropout(x); x = self.out_proj(x)
        return x


class G1_OrigFT_Model(Wav2Vec2PreTrainedModel):
    """G1 Orig/FT：mean pool → dep_classifier（768 維 probe）"""
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2   = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.init_weights()

    def get_embedding(self, input_values, attention_mask=None):
        out = self.wav2vec2(input_values, attention_mask=attention_mask)
        return torch.mean(out.last_hidden_state, dim=1)  # [B, 768]


# ── G1 DANN/FT：mean pool → spk_classifier（仍 768 維，無 down_proj）

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad): return grad.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def forward(self, x, alpha=1.0): return GradientReversalFn.apply(x, alpha)


class G1_DANN_FT_Model(Wav2Vec2PreTrainedModel):
    """G1 DANN/FT：CNN frozen + Transformer FT，mean pool → dep+spk（768 維 probe）"""
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2       = Wav2Vec2Model(config)
        self.wav2vec2.feature_extractor._freeze_parameters()
        self.dropout        = nn.Dropout(0.1)
        self.classifier     = nn.Linear(config.hidden_size, config.num_labels)
        self.spk_classifier = nn.Linear(config.hidden_size,
                                         getattr(config, "num_speakers", 151))
        self._alpha = 0.0
        self.init_weights()

    def get_embedding(self, input_values, attention_mask=None):
        out = self.wav2vec2(input_values, attention_mask=attention_mask, return_dict=True)
        last = out.last_hidden_state
        if attention_mask is not None:
            lengths = attention_mask.sum(-1)
            pooled = torch.stack([
                last[i, :lengths[i]].mean(0) for i in range(last.size(0))
            ])
        else:
            pooled = last.mean(1)
        return self.dropout(pooled)  # [B, 768]

    def forward(self, input_values, attention_mask=None, labels=None,
                speaker_labels=None, **kwargs):
        pooled     = self.get_embedding(input_values, attention_mask)
        logits     = self.classifier(pooled)
        spk_logits = self.spk_classifier(GradientReversalFn.apply(pooled, self._alpha))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.config.num_labels), labels.view(-1))
        if speaker_labels is not None:
            mask = speaker_labels >= 0
            if mask.sum() > 0:
                spk_loss = nn.CrossEntropyLoss()(
                    spk_logits[mask].view(-1, spk_logits.size(-1)),
                    speaker_labels[mask].view(-1),
                )
                loss = loss + spk_loss if loss is not None else spk_loss
        return (loss, logits) if loss is not None else logits


# ── G2：SLS + down_proj（128 維）────────────────────────────

class G2_SLS_Model(Wav2Vec2PreTrainedModel):
    """
    G2 通用：SLS weighted sum + down_proj（128 維 probe）
    涵蓋 Orig/Frozen、Orig/FT、DANN/Frozen、DANN/FT 四種 variant。
    有 DANN 的只是多了 spk_classifier，probe 只用 down_proj 輸出即可。
    """
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2    = Wav2Vec2Model(config)
        num_layers       = config.num_hidden_layers + 1
        self.sls_weights = nn.Parameter(torch.ones(num_layers))
        self.down_proj   = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )
        self.dep_classifier = nn.Linear(128, config.num_labels)
        # spk_classifier 可能存在（DANN 版），用 ignore_mismatched_sizes 處理
        self.spk_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, getattr(config, "num_speakers", 151)),
        )
        self._alpha = 0.0
        self.init_weights()

    def get_embedding(self, input_values, attention_mask=None):
        out     = self.wav2vec2(input_values, attention_mask=attention_mask,
                                output_hidden_states=True, return_dict=True)
        hs      = torch.stack(out.hidden_states)            # [L, B, T, H]
        weights = torch.softmax(self.sls_weights, dim=0)
        fused   = (hs * weights.view(-1, 1, 1, 1)).sum(0)  # [B, T, H]
        return self.down_proj(fused.mean(1))                 # [B, 128]

    def forward(self, input_values, attention_mask=None, labels=None,
                speaker_labels=None, **kwargs):
        shared     = self.get_embedding(input_values, attention_mask)
        dep_logits = self.dep_classifier(shared)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                dep_logits.view(-1, self.config.num_labels), labels.view(-1))
        return (loss, dep_logits) if loss is not None else dep_logits


# ── G2 .pth 專用 down_proj（凍結骨幹 + 外部 .pth）────────────

class G2_DownProj(nn.Module):
    """載入 G2 DANN/Frozen 或 G2 Orig/FT 存下來的 down_proj.state_dict()"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(768, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )
    def forward(self, x): return self.encoder(x)


# ── G3：XLSR + eGeMAPS concat + down_proj（256 維）───────────

class G3_XLSR_Model(Wav2Vec2PreTrainedModel):
    """
    G3 通用：XLSR mean pool + eGeMAPS concat → down_proj（256 維 probe）
    probe 時 eGeMAPS 補零。
    涵蓋 Orig/Frozen、Orig/FT、DANN/Frozen、DANN/FT。
    """
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2  = Wav2Vec2Model(config)
        combined_dim   = config.hidden_size + EGEMAPS_DIM  # 1024 + 88
        self.down_proj = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        )
        self.dep_classifier = nn.Linear(256, config.num_labels)
        self.spk_classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, getattr(config, "num_speakers", 151)),
        )
        self._alpha = 0.0
        self.init_weights()

    def get_embedding(self, input_values, attention_mask=None):
        out       = self.wav2vec2(input_values, attention_mask=attention_mask, return_dict=True)
        xlsr_feat = out.last_hidden_state.mean(1)           # [B, 1024]
        zero_pad  = torch.zeros(xlsr_feat.size(0), EGEMAPS_DIM,
                                dtype=xlsr_feat.dtype, device=xlsr_feat.device)
        return self.down_proj(torch.cat([xlsr_feat, zero_pad], dim=-1))  # [B, 256]

    def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
        shared     = self.get_embedding(input_values, attention_mask)
        dep_logits = self.dep_classifier(shared)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                dep_logits.view(-1, self.config.num_labels), labels.view(-1))
        return (loss, dep_logits) if loss is not None else dep_logits


# ── G3 .pth 專用 down_proj ──────────────────────────────────

class G3_DownProj(nn.Module):
    """載入 G3 DANN/Frozen 或 G3 DANN/FT 存下來的 down_proj.state_dict()"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1024 + EGEMAPS_DIM, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        )
    def forward(self, x): return self.encoder(x)


# ============================================================
#  工具函式
# ============================================================

def extract_speaker_id(filepath: str) -> str:
    return os.path.basename(str(filepath)).split('_')[0]


def get_target_speakers(test_csv: str) -> Set[str]:
    df = pd.read_csv(test_csv)
    return set(df['path'].apply(extract_speaker_id))


def _find_checkpoint(run_dir: str) -> Optional[str]:
    """best_model > 最新 checkpoint-XXXX > run_dir 本身"""
    if not os.path.isdir(run_dir):
        return None
    best = os.path.join(run_dir, "best_model")
    if os.path.isdir(best) and _has_weights(best):
        return best
    ckpts = sorted(
        [d for d in os.listdir(run_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1]),
    )
    if ckpts:
        return os.path.join(run_dir, ckpts[-1])
    if os.path.exists(os.path.join(run_dir, "config.json")) and _has_weights(run_dir):
        return run_dir
    return None


def _has_weights(d: str) -> bool:
    return (os.path.exists(os.path.join(d, "pytorch_model.bin")) or
            os.path.exists(os.path.join(d, "model.safetensors")))


def _load_pth(raw_sd: dict, prefix: str = "encoder") -> dict:
    """如果 key 沒有 prefix 就補上，對應 nn.Sequential 存的 state_dict"""
    if any(k.startswith(f"{prefix}.") for k in raw_sd.keys()):
        return raw_sd
    return {f"{prefix}.{k}": v for k, v in raw_sd.items()}


def load_wav(path: str, processor):
    try:
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        return processor(wav.squeeze().numpy(), sampling_rate=16000,
                         return_tensors="pt", padding=True)
    except Exception:
        return None


# ============================================================
#  Embedding 抽取函式
# ============================================================

def _iter_csv(csv_path, processor, filter_spk=None, desc=""):
    """讀 CSV，yield (inputs, spk_id)"""
    df = pd.read_csv(csv_path)
    for _, row in tqdm(df.iterrows(), total=len(df), leave=False, desc=f"  {desc}"):
        spk = extract_speaker_id(row['path'])
        if filter_spk is not None and spk not in filter_spk:
            continue
        inp = load_wav(os.path.join(AUDIO_ROOT, str(row['path'])), processor)
        if inp is None:
            continue
        yield inp, spk


def extract_frozen_w2v(csv_path, processor, w2v, filter_spk=None):
    """G1 Orig/Frozen：frozen W2V → 768 維 mean pool"""
    feats, spks = [], []
    w2v.eval()
    with torch.no_grad():
        for inp, spk in _iter_csv(csv_path, processor, filter_spk, "FrozenW2V"):
            emb = w2v(**{k: v.to(DEVICE) for k, v in inp.items()}).last_hidden_state.mean(1)
            feats.append(emb.squeeze().cpu().numpy()); spks.append(spk)
    return np.array(feats), np.array(spks)


def extract_from_checkpoint(csv_path, processor, ckpt_dir, ModelClass,
                            model_kwargs=None, filter_spk=None, desc=""):
    # 1. Force Official Config Skeleton
    official_name = "facebook/wav2vec2-xls-r-300m" if "G3" in ModelClass.__name__ else "facebook/wav2vec2-base"
    cfg = AutoConfig.from_pretrained(official_name)
    cfg.num_labels = 2
    if model_kwargs:
        for k, v in model_kwargs.items(): setattr(cfg, k, v)

    # 2. Build Empty Model
    model = ModelClass(cfg).to(DEVICE)
    
    # 3. Robust Weight Search (Key Fix)
    possible_files = [
        os.path.join(ckpt_dir, "pytorch_model.bin"),
        os.path.join(ckpt_dir, "model.safetensors"),
        os.path.join(os.path.dirname(ckpt_dir), "pytorch_model.bin"), # Try parent dir
    ]
    
    weights_path = None
    for f in possible_files:
        if os.path.exists(f):
            weights_path = f
            break
            
    if not weights_path:
        # Last resort: scan run_1 folder
        import glob
        found = glob.glob(os.path.join(os.path.dirname(ckpt_dir), "checkpoint-*/pytorch_model.bin"))
        if found: weights_path = found[0]

    if weights_path:
        print(f"   🎯 Found weights: {weights_path}")
        if weights_path.endswith(".bin"):
            sd = torch.load(weights_path, map_location=DEVICE)
        else:
            from safetensors.torch import load_file
            sd = load_file(weights_path)
        model.load_state_dict(sd, strict=False)
    else:
        print(f"   ❌ Weights not found for {ckpt_dir}, skipping.")
        return np.array([]), np.array([]) 

    model.eval()
    
    feats, spks = [], []
    with torch.no_grad():
        for inp, spk in _iter_csv(csv_path, processor, filter_spk, desc):
            emb = model.get_embedding(**{k: v.to(DEVICE) for k, v in inp.items()})
            feats.append(emb.squeeze().cpu().numpy()); spks.append(spk)
    del model; torch.cuda.empty_cache(); gc.collect()
    return np.array(feats), np.array(spks)

def extract_g2_from_pth(csv_path, processor, w2v, pth_path, filter_spk=None):
    from safetensors.torch import load_file
    down_proj = G2_DownProj().to(DEVICE)
    down_proj.load_state_dict(_load_pth(torch.load(pth_path, map_location=DEVICE)), strict=False)
    
    # Auto-detect SLS weights from nearby checkpoint
    sls_weights = torch.ones(13).to(DEVICE)
    parent_dir = os.path.dirname(pth_path)
    import glob
    # Look for model.safetensors in any subfolder (e.g., run_1/checkpoint-5360)
    ckpts = glob.glob(os.path.join(parent_dir, "run_1/checkpoint-*/model.safetensors"))
    if ckpts:
        try:
            sd_full = load_file(ckpts[0])
            if "sls_weights" in sd_full:
                sls_weights = sd_full["sls_weights"].to(DEVICE)
                print(f"   ✅ SLS weights loaded from: {os.path.basename(os.path.dirname(ckpts[0]))}")
        except Exception as e:
            print(f"   ⚠️ Failed to load SLS weights: {e}")

    down_proj.eval(); w2v.eval()
    feats, spks = [], []
    with torch.no_grad():
        for inp, spk in _iter_csv(csv_path, processor, filter_spk, "G2-SLS-Fusion"):
            out = w2v(**{k: v.to(DEVICE) for k, v in inp.items()}, output_hidden_states=True)
            # Perform Weighted Sum Fusion
            w = torch.softmax(sls_weights, dim=0)
            fused = (torch.stack(out.hidden_states) * w.view(-1,1,1,1)).sum(0)
            emb = down_proj(fused.mean(1))
            feats.append(emb.squeeze().cpu().numpy()); spks.append(spk)
    return np.array(feats), np.array(spks)


def extract_frozen_xlsr(csv_path, xlsr_proc, xlsr, filter_spk=None):
    """G3 Orig/Frozen（無 down_proj .pth）：frozen XLSR → 1024 維 mean pool（直接 probe）"""
    feats, spks = [], []
    xlsr.eval()
    with torch.no_grad():
        for inp, spk in _iter_csv(csv_path, xlsr_proc, filter_spk, "FrozenXLSR"):
            emb = xlsr(**{k: v.to(DEVICE) for k, v in inp.items()}).last_hidden_state.mean(1)
            feats.append(emb.squeeze().cpu().numpy()); spks.append(spk)
    return np.array(feats), np.array(spks)


def extract_g3_from_pth(csv_path, xlsr_proc, xlsr, pth_path, filter_spk=None):
    """
    G3 DANN/Frozen, G3 DANN/FT：
    frozen XLSR → concat zero eGeMAPS → down_proj .pth → 256 維。
    """
    down_proj = G3_DownProj().to(DEVICE)
    down_proj.load_state_dict(_load_pth(torch.load(pth_path, map_location=DEVICE)), strict=False)
    down_proj.eval()
    feats, spks = [], []
    xlsr.eval()
    with torch.no_grad():
        for inp, spk in _iter_csv(csv_path, xlsr_proc, filter_spk, "G3-pth"):
            xlsr_feat = xlsr(**{k: v.to(DEVICE) for k, v in inp.items()}).last_hidden_state.mean(1)
            zero_pad  = torch.zeros(1, EGEMAPS_DIM, dtype=xlsr_feat.dtype, device=DEVICE)
            emb       = down_proj(torch.cat([xlsr_feat, zero_pad], dim=-1))
            feats.append(emb.squeeze().cpu().numpy()); spks.append(spk)
    del down_proj; torch.cuda.empty_cache(); gc.collect()
    return np.array(feats), np.array(spks)


# ============================================================
#  Probe 評估（LogisticRegression）
# ============================================================

def probe(X_train, y_train, X_test, y_test) -> float:
    clf = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))


# ============================================================
#  主程式
# ============================================================

if __name__ == "__main__":

    # ── 預載骨幹（多個 group 共用）─────────────────────────────
    print("\n🧠 載入 frozen Wav2Vec2-base...")
    w2v_proc   = Wav2Vec2Processor.from_pretrained(MODEL_NAME_W2V)
    w2v_frozen = Wav2Vec2Model.from_pretrained(MODEL_NAME_W2V).to(DEVICE)
    w2v_frozen.eval()

    print("🧠 載入 frozen XLS-R-300m...")
    xlsr_proc   = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME_XLSR)
    xlsr_frozen = Wav2Vec2Model.from_pretrained(MODEL_NAME_XLSR).to(DEVICE)
    xlsr_frozen.eval()

    # Scenario B target speakers（38 位）
    target_spk_B = get_target_speakers(CSV_B_TEST)
    print(f"\n🎯 Scenario B target speakers: {len(target_spk_B)} 位")

    # ── G1 Orig/Frozen：預先抽好（骨幹不變，5 runs 全同）──────
    print("\n📦 預抽 G1 Orig/Frozen 特徵（frozen W2V，5 runs 共用）...")
    g1_frozen_A_train_X, g1_frozen_A_train_y = extract_frozen_w2v(CSV_A_TRAIN, w2v_proc, w2v_frozen)
    g1_frozen_A_test_X,  g1_frozen_A_test_y  = extract_frozen_w2v(CSV_A_TEST,  w2v_proc, w2v_frozen)
    g1_frozen_B_hist_X,  g1_frozen_B_hist_y  = extract_frozen_w2v(CSV_B_TRAIN, w2v_proc, w2v_frozen,
                                                                    filter_spk=target_spk_B)
    g1_frozen_B_test_X,  g1_frozen_B_test_y  = extract_frozen_w2v(CSV_B_TEST,  w2v_proc, w2v_frozen)

    # ── 結果容器 ────────────────────────────────────────────────
    # key = (group, variant, scenario)
    results = {k: [] for k in [
        "G1_Orig_Frozen_A", "G1_Orig_Frozen_B",
        "G1_Orig_FT_A",     "G1_Orig_FT_B",
        "G1_DANN_Frozen_A", "G1_DANN_Frozen_B",   # manual loop，無法 probe
        "G1_DANN_FT_A",     "G1_DANN_FT_B",
        "G2_Orig_Frozen_A", "G2_Orig_Frozen_B",
        "G2_Orig_FT_A",     "G2_Orig_FT_B",
        "G2_DANN_Frozen_A", "G2_DANN_Frozen_B",
        "G2_DANN_FT_A",     "G2_DANN_FT_B",
        "G3_Orig_Frozen_A", "G3_Orig_Frozen_B",
        "G3_Orig_FT_A",     "G3_Orig_FT_B",
        "G3_DANN_Frozen_A", "G3_DANN_Frozen_B",
        "G3_DANN_FT_A",     "G3_DANN_FT_B",
    ]}

    # G1 DANN/Frozen 無模型存檔，用特殊標記
    CANT_PROBE = "CANT_PROBE"
    results["G1_DANN_Frozen_A"] = CANT_PROBE
    results["G1_DANN_Frozen_B"] = CANT_PROBE

    # ── G1 Orig/Frozen：probe 每 run 結果相同（骨幹不變），跑一次 ──
    acc = probe(g1_frozen_A_train_X, g1_frozen_A_train_y,
                g1_frozen_A_test_X,  g1_frozen_A_test_y)
    results["G1_Orig_Frozen_A"] = [acc] * TOTAL_RUNS
    print(f"\n[G1 Orig/Frozen A]  Spk Acc = {acc:.4f}  (共用 {TOTAL_RUNS} runs)")

    acc = probe(g1_frozen_B_hist_X, g1_frozen_B_hist_y,
                g1_frozen_B_test_X, g1_frozen_B_test_y)
    results["G1_Orig_Frozen_B"] = [acc] * TOTAL_RUNS
    print(f"[G1 Orig/Frozen B]  Spk Acc = {acc:.4f}  (共用 {TOTAL_RUNS} runs)")

    # ── 逐 run 評估（需要模型的 12 個 variant）──────────────────
    for i in range(1, TOTAL_RUNS + 1):
        print(f"\n{'='*65}\n🎬 Run {i} / {TOTAL_RUNS}\n{'='*65}")

        # ──────────────────────────────────────────────────────
        # G1 Orig/FT A（checkpoint，768 維）
        ckpt = _find_checkpoint(G1_ORIG_FT_A)
        if ckpt and _has_weights(ckpt):
            print(f"\n  [G1 Orig/FT A] ← {ckpt}")
            Xtr, ytr = extract_from_checkpoint(CSV_A_TRAIN, w2v_proc, ckpt,
                                                G1_OrigFT_Model, desc="G1FT-A-train")
            Xte, yte = extract_from_checkpoint(CSV_A_TEST,  w2v_proc, ckpt,
                                                G1_OrigFT_Model, desc="G1FT-A-test")
            acc = probe(Xtr, ytr, Xte, yte)
            results["G1_Orig_FT_A"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [A unseen]")
        else:
            print(f"\n  [G1 Orig/FT A]  ⏳ Run {i} 不存在，跳過")

        # G1 Orig/FT B
        ckpt = _find_checkpoint(G1_ORIG_FT_B)
        if ckpt and _has_weights(ckpt):
            print(f"\n  [G1 Orig/FT B] ← {ckpt}")
            Xhi, yhi = extract_from_checkpoint(CSV_B_TRAIN, w2v_proc, ckpt,
                                                G1_OrigFT_Model, filter_spk=target_spk_B,
                                                desc="G1FT-B-hist")
            Xte, yte = extract_from_checkpoint(CSV_B_TEST,  w2v_proc, ckpt,
                                                G1_OrigFT_Model, desc="G1FT-B-test")
            acc = probe(Xhi, yhi, Xte, yte)
            results["G1_Orig_FT_B"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [B target-only]")
        else:
            print(f"\n  [G1 Orig/FT B]  ⏳ Run {i} 不存在，跳過")

        # ── G1 DANN/FT A（checkpoint，768 維）─────────────────
        ckpt = _find_checkpoint(G1_DANN_FT_A)
        if ckpt and _has_weights(ckpt):
            print(f"\n  [G1 DANN/FT A] ← {ckpt}")
            Xtr, ytr = extract_from_checkpoint(CSV_A_TRAIN, w2v_proc, ckpt,
                                                G1_DANN_FT_Model,
                                                model_kwargs={"num_speakers": 151},
                                                desc="G1DANN-FT-A-train")
            Xte, yte = extract_from_checkpoint(CSV_A_TEST,  w2v_proc, ckpt,
                                                G1_DANN_FT_Model,
                                                model_kwargs={"num_speakers": 151},
                                                desc="G1DANN-FT-A-test")
            acc = probe(Xtr, ytr, Xte, yte)
            results["G1_DANN_FT_A"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [A unseen]")
        else:
            print(f"\n  [G1 DANN/FT A]  ⏳ Run {i} 不存在，跳過")

        # G1 DANN/FT B
        ckpt = _find_checkpoint(G1_DANN_FT_B)
        if ckpt and _has_weights(ckpt):
            print(f"\n  [G1 DANN/FT B] ← {ckpt}")
            Xhi, yhi = extract_from_checkpoint(CSV_B_TRAIN, w2v_proc, ckpt,
                                                G1_DANN_FT_Model,
                                                model_kwargs={"num_speakers": 38},
                                                filter_spk=target_spk_B,
                                                desc="G1DANN-FT-B-hist")
            Xte, yte = extract_from_checkpoint(CSV_B_TEST,  w2v_proc, ckpt,
                                                G1_DANN_FT_Model,
                                                model_kwargs={"num_speakers": 38},
                                                desc="G1DANN-FT-B-test")
            acc = probe(Xhi, yhi, Xte, yte)
            results["G1_DANN_FT_B"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [B target-only]")
        else:
            print(f"\n  [G1 DANN/FT B]  ⏳ Run {i} 不存在，跳過")

        # ──────────────────────────────────────────────────────
        # G2 Orig/Frozen A（HF checkpoint，128 維）
        ckpt = _find_checkpoint(G2_ORIG_FROZEN_A)
        if ckpt and _has_weights(ckpt):
            print(f"\n  [G2 Orig/Frozen A] ← {ckpt}")
            Xtr, ytr = extract_from_checkpoint(CSV_A_TRAIN, w2v_proc, ckpt,
                                                G2_SLS_Model, desc="G2Orig-A-train")
            Xte, yte = extract_from_checkpoint(CSV_A_TEST,  w2v_proc, ckpt,
                                                G2_SLS_Model, desc="G2Orig-A-test")
            acc = probe(Xtr, ytr, Xte, yte)
            results["G2_Orig_Frozen_A"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [A unseen]")
        else:
            print(f"\n  [G2 Orig/Frozen A]  ⏳ Run {i} 不存在，跳過")

        # G2 Orig/Frozen B
        ckpt = _find_checkpoint(G2_ORIG_FROZEN_B)
        if ckpt and _has_weights(ckpt):
            print(f"\n  [G2 Orig/Frozen B] ← {ckpt}")
            Xhi, yhi = extract_from_checkpoint(CSV_B_TRAIN, w2v_proc, ckpt,
                                                G2_SLS_Model, filter_spk=target_spk_B,
                                                desc="G2Orig-B-hist")
            Xte, yte = extract_from_checkpoint(CSV_B_TEST,  w2v_proc, ckpt,
                                                G2_SLS_Model, desc="G2Orig-B-test")
            acc = probe(Xhi, yhi, Xte, yte)
            results["G2_Orig_Frozen_B"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [B target-only]")
        else:
            print(f"\n  [G2 Orig/Frozen B]  ⏳ Run {i} 不存在，跳過")

        # ── G2 Orig/FT A（.pth，128 維）──────────────────────
        pth = G2_ORIG_FT_A_PTH
        if os.path.exists(pth):
            print(f"\n  [G2 Orig/FT A] ← {pth}")
            Xtr, ytr = extract_g2_from_pth(CSV_A_TRAIN, w2v_proc, w2v_frozen, pth)
            Xte, yte = extract_g2_from_pth(CSV_A_TEST,  w2v_proc, w2v_frozen, pth)
            acc = probe(Xtr, ytr, Xte, yte)
            results["G2_Orig_FT_A"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [A unseen]")
        else:
            print(f"\n  [G2 Orig/FT A]  ⏳ Run {i} .pth 不存在，跳過")

        # G2 Orig/FT B
        pth = G2_ORIG_FT_B_PTH
        if os.path.exists(pth):
            print(f"\n  [G2 Orig/FT B] ← {pth}")
            Xhi, yhi = extract_g2_from_pth(CSV_B_TRAIN, w2v_proc, w2v_frozen, pth,
                                             filter_spk=target_spk_B)
            Xte, yte = extract_g2_from_pth(CSV_B_TEST,  w2v_proc, w2v_frozen, pth)
            acc = probe(Xhi, yhi, Xte, yte)
            results["G2_Orig_FT_B"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [B target-only]")
        else:
            print(f"\n  [G2 Orig/FT B]  ⏳ Run {i} .pth 不存在，跳過")

        # ── G2 DANN/Frozen A（.pth，128 維）──────────────────
        pth = G2_DANN_FROZEN_A_PTH
        if os.path.exists(pth):
            print(f"\n  [G2 DANN/Frozen A] ← {pth}")
            Xtr, ytr = extract_g2_from_pth(CSV_A_TRAIN, w2v_proc, w2v_frozen, pth)
            Xte, yte = extract_g2_from_pth(CSV_A_TEST,  w2v_proc, w2v_frozen, pth)
            acc = probe(Xtr, ytr, Xte, yte)
            results["G2_DANN_Frozen_A"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [A unseen]")
        else:
            print(f"\n  [G2 DANN/Frozen A]  ⏳ Run {i} .pth 不存在，跳過")

        # G2 DANN/Frozen B
        pth = G2_DANN_FROZEN_B_PTH
        if os.path.exists(pth):
            print(f"\n  [G2 DANN/Frozen B] ← {pth}")
            Xhi, yhi = extract_g2_from_pth(CSV_B_TRAIN, w2v_proc, w2v_frozen, pth,
                                             filter_spk=target_spk_B)
            Xte, yte = extract_g2_from_pth(CSV_B_TEST,  w2v_proc, w2v_frozen, pth)
            acc = probe(Xhi, yhi, Xte, yte)
            results["G2_DANN_Frozen_B"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [B target-only]")
        else:
            print(f"\n  [G2 DANN/Frozen B]  ⏳ Run {i} .pth 不存在，跳過")

        # ── G2 DANN/FT A（.pth，128 維）──────────────────────
        pth = G2_DANN_FT_A_PTH
        if os.path.exists(pth):
            print(f"\n  [G2 DANN/FT A] ← {pth}")
            Xtr, ytr = extract_g2_from_pth(CSV_A_TRAIN, w2v_proc, w2v_frozen, pth)
            Xte, yte = extract_g2_from_pth(CSV_A_TEST,  w2v_proc, w2v_frozen, pth)
            acc = probe(Xtr, ytr, Xte, yte)
            results["G2_DANN_FT_A"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [A unseen]")
        else:
            print(f"\n  [G2 DANN/FT A]  ⏳ Run {i} .pth 不存在，跳過")

        # G2 DANN/FT B
        pth = G2_DANN_FT_B_PTH
        if os.path.exists(pth):
            print(f"\n  [G2 DANN/FT B] ← {pth}")
            Xhi, yhi = extract_g2_from_pth(CSV_B_TRAIN, w2v_proc, w2v_frozen, pth,
                                             filter_spk=target_spk_B)
            Xte, yte = extract_g2_from_pth(CSV_B_TEST,  w2v_proc, w2v_frozen, pth)
            acc = probe(Xhi, yhi, Xte, yte)
            results["G2_DANN_FT_B"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [B target-only]")
        else:
            print(f"\n  [G2 DANN/FT B]  ⏳ Run {i} .pth 不存在，跳過")

        # ──────────────────────────────────────────────────────
        # G3 Orig/Frozen A（HF checkpoint，256 維）
        ckpt = _find_checkpoint(G3_ORIG_FROZEN_A)
        if ckpt and _has_weights(ckpt):
            print(f"\n  [G3 Orig/Frozen A] ← {ckpt}")
            Xtr, ytr = extract_from_checkpoint(CSV_A_TRAIN, xlsr_proc, ckpt,
                                                G3_XLSR_Model, desc="G3Orig-A-train")
            Xte, yte = extract_from_checkpoint(CSV_A_TEST,  xlsr_proc, ckpt,
                                                G3_XLSR_Model, desc="G3Orig-A-test")
            acc = probe(Xtr, ytr, Xte, yte)
            results["G3_Orig_Frozen_A"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [A unseen]")
        else:
            print(f"\n  [G3 Orig/Frozen A]  ⏳ Run {i} 不存在，跳過")

        # G3 Orig/Frozen B
        ckpt = _find_checkpoint(G3_ORIG_FROZEN_B)
        if ckpt and _has_weights(ckpt):
            print(f"\n  [G3 Orig/Frozen B] ← {ckpt}")
            Xhi, yhi = extract_from_checkpoint(CSV_B_TRAIN, xlsr_proc, ckpt,
                                                G3_XLSR_Model, filter_spk=target_spk_B,
                                                desc="G3Orig-B-hist")
            Xte, yte = extract_from_checkpoint(CSV_B_TEST,  xlsr_proc, ckpt,
                                                G3_XLSR_Model, desc="G3Orig-B-test")
            acc = probe(Xhi, yhi, Xte, yte)
            results["G3_Orig_Frozen_B"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [B target-only]")
        else:
            print(f"\n  [G3 Orig/Frozen B]  ⏳ Run {i} 不存在，跳過")

        # ── G3 Orig/FT A（checkpoint，256 維）────────────────
        ckpt = _find_checkpoint(G3_ORIG_FT_A)
        if ckpt and _has_weights(ckpt):
            print(f"\n  [G3 Orig/FT A] ← {ckpt}")
            Xtr, ytr = extract_from_checkpoint(CSV_A_TRAIN, xlsr_proc, ckpt,
                                                G3_XLSR_Model, desc="G3FT-A-train")
            Xte, yte = extract_from_checkpoint(CSV_A_TEST,  xlsr_proc, ckpt,
                                                G3_XLSR_Model, desc="G3FT-A-test")
            acc = probe(Xtr, ytr, Xte, yte)
            results["G3_Orig_FT_A"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [A unseen]")
        else:
            print(f"\n  [G3 Orig/FT A]  ⏳ Run {i} 不存在，跳過")

        # G3 Orig/FT B
        ckpt = _find_checkpoint(G3_ORIG_FT_B)
        if ckpt and _has_weights(ckpt):
            print(f"\n  [G3 Orig/FT B] ← {ckpt}")
            Xhi, yhi = extract_from_checkpoint(CSV_B_TRAIN, xlsr_proc, ckpt,
                                                G3_XLSR_Model, filter_spk=target_spk_B,
                                                desc="G3FT-B-hist")
            Xte, yte = extract_from_checkpoint(CSV_B_TEST,  xlsr_proc, ckpt,
                                                G3_XLSR_Model, desc="G3FT-B-test")
            acc = probe(Xhi, yhi, Xte, yte)
            results["G3_Orig_FT_B"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [B target-only]")
        else:
            print(f"\n  [G3 Orig/FT B]  ⏳ Run {i} 不存在，跳過")

        # ── G3 DANN/Frozen A（.pth，256 維）──────────────────
        pth = G3_DANN_FROZEN_A_PTH
        if os.path.exists(pth):
            print(f"\n  [G3 DANN/Frozen A] ← {pth}")
            Xtr, ytr = extract_g3_from_pth(CSV_A_TRAIN, xlsr_proc, xlsr_frozen, pth)
            Xte, yte = extract_g3_from_pth(CSV_A_TEST,  xlsr_proc, xlsr_frozen, pth)
            acc = probe(Xtr, ytr, Xte, yte)
            results["G3_DANN_Frozen_A"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [A unseen]")
        else:
            print(f"\n  [G3 DANN/Frozen A]  ⏳ Run {i} .pth 不存在，跳過")

        # G3 DANN/Frozen B
        pth = G3_DANN_FROZEN_B_PTH
        if os.path.exists(pth):
            print(f"\n  [G3 DANN/Frozen B] ← {pth}")
            Xhi, yhi = extract_g3_from_pth(CSV_B_TRAIN, xlsr_proc, xlsr_frozen, pth,
                                             filter_spk=target_spk_B)
            Xte, yte = extract_g3_from_pth(CSV_B_TEST,  xlsr_proc, xlsr_frozen, pth)
            acc = probe(Xhi, yhi, Xte, yte)
            results["G3_DANN_Frozen_B"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [B target-only]")
        else:
            print(f"\n  [G3 DANN/Frozen B]  ⏳ Run {i} .pth 不存在，跳過")

        # ── G3 DANN/FT A（.pth，256 維）──────────────────────
        pth = G3_DANN_FT_A_PTH
        if os.path.exists(pth):
            print(f"\n  [G3 DANN/FT A] ← {pth}")
            Xtr, ytr = extract_g3_from_pth(CSV_A_TRAIN, xlsr_proc, xlsr_frozen, pth)
            Xte, yte = extract_g3_from_pth(CSV_A_TEST,  xlsr_proc, xlsr_frozen, pth)
            acc = probe(Xtr, ytr, Xte, yte)
            results["G3_DANN_FT_A"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [A unseen]")
        else:
            print(f"\n  [G3 DANN/FT A]  ⏳ Run {i} .pth 不存在，跳過")

        # G3 DANN/FT B
        pth = G3_DANN_FT_B_PTH
        if os.path.exists(pth):
            print(f"\n  [G3 DANN/FT B] ← {pth}")
            Xhi, yhi = extract_g3_from_pth(CSV_B_TRAIN, xlsr_proc, xlsr_frozen, pth,
                                             filter_spk=target_spk_B)
            Xte, yte = extract_g3_from_pth(CSV_B_TEST,  xlsr_proc, xlsr_frozen, pth)
            acc = probe(Xhi, yhi, Xte, yte)
            results["G3_DANN_FT_B"].append(acc)
            print(f"    Spk Acc = {acc:.4f}  [B target-only]")
        else:
            print(f"\n  [G3 DANN/FT B]  ⏳ Run {i} .pth 不存在，跳過")

    # ============================================================
    #  彙總輸出
    # ============================================================

    # 論文 Table 1 的順序：每組 Original(A/B) → FT(A/B) → DANN(A/B) → DANN-FT(A/B)
    TABLE_ORDER = [
        # label_table, key, scenario_label
        ("G1 | Original | Frozen Wav2Vec2 | A",  "G1_Orig_Frozen_A", "A"),
        ("G1 | Original | Frozen Wav2Vec2 | B",  "G1_Orig_Frozen_B", "B"),
        ("G1 | Original | Fine-tuned W2V  | A",  "G1_Orig_FT_A",     "A"),
        ("G1 | Original | Fine-tuned W2V  | B",  "G1_Orig_FT_B",     "B"),
        ("G1 | DANN     | Frozen Wav2Vec2 | A",  "G1_DANN_Frozen_A", "A"),
        ("G1 | DANN     | Frozen Wav2Vec2 | B",  "G1_DANN_Frozen_B", "B"),
        ("G1 | DANN     | Fine-tuned W2V  | A",  "G1_DANN_FT_A",     "A"),
        ("G1 | DANN     | Fine-tuned W2V  | B",  "G1_DANN_FT_B",     "B"),
        ("G2 | Original | Frozen Wav2Vec2 | A",  "G2_Orig_Frozen_A", "A"),
        ("G2 | Original | Frozen Wav2Vec2 | B",  "G2_Orig_Frozen_B", "B"),
        ("G2 | Original | Fine-tuned W2V  | A",  "G2_Orig_FT_A",     "A"),
        ("G2 | Original | Fine-tuned W2V  | B",  "G2_Orig_FT_B",     "B"),
        ("G2 | DANN     | Frozen Wav2Vec2 | A",  "G2_DANN_Frozen_A", "A"),
        ("G2 | DANN     | Frozen Wav2Vec2 | B",  "G2_DANN_Frozen_B", "B"),
        ("G2 | DANN     | Fine-tuned W2V  | A",  "G2_DANN_FT_A",     "A"),
        ("G2 | DANN     | Fine-tuned W2V  | B",  "G2_DANN_FT_B",     "B"),
        ("G3 | Original | Frozen XLS-R    | A",  "G3_Orig_Frozen_A", "A"),
        ("G3 | Original | Frozen XLS-R    | B",  "G3_Orig_Frozen_B", "B"),
        ("G3 | Original | Fine-tuned XLSR | A",  "G3_Orig_FT_A",     "A"),
        ("G3 | Original | Fine-tuned XLSR | B",  "G3_Orig_FT_B",     "B"),
        ("G3 | DANN     | Frozen XLS-R    | A",  "G3_DANN_Frozen_A", "A"),
        ("G3 | DANN     | Frozen XLS-R    | B",  "G3_DANN_Frozen_B", "B"),
        ("G3 | DANN     | Fine-tuned XLSR | A",  "G3_DANN_FT_A",     "A"),
        ("G3 | DANN     | Fine-tuned XLSR | B",  "G3_DANN_FT_B",     "B"),
    ]

    print(f"\n{'='*80}")
    print(f"📊 Speaker Probe 彙總  (24 個模型)")
    print(f"  Scenario A → probe train=151 control，test=38 unseen  → 預期 ≈ 0%")
    print(f"  Scenario B → probe train=38 target hist，test=38 target current")
    print(f"{'='*80}")
    print(f"  {'#':<3} {'模型':<42} {'Scen':<5} {'有效runs':<10} {'平均 Spk Acc':<16} 標準差")
    print(f"  {'─'*80}")

    summary_rows = []
    for idx, (label, key, scen) in enumerate(TABLE_ORDER, 1):
        val = results[key]
        if val == CANT_PROBE:
            print(f"  {idx:<3} {label:<42} {scen:<5} {'N/A':<10} {'manual loop 無模型存檔'}")
            continue
        if len(val) == 0:
            print(f"  {idx:<3} {label:<42} {scen:<5} {'0':<10} N/A")
            continue
        arr = np.array(val)
        mean, std = arr.mean(), arr.std()
        runs_str  = str(len(val)) if len(val) == TOTAL_RUNS else f"⚠️{len(val)}/{TOTAL_RUNS}"
        print(f"  {idx:<3} {label:<42} {scen:<5} {runs_str:<10} {mean:.4f}{'':>10} ± {std:.4f}")
        summary_rows.append({
            "idx": idx, "label": label, "key": key, "scenario": scen,
            "valid_runs": len(val),
            "spk_acc_mean": round(mean, 4),
            "spk_acc_std":  round(std,  4),
        })

    if summary_rows:
        out_csv = "speaker_probe_24models.csv"
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
        print(f"\n✅ 彙總已儲存至 {out_csv}")

    pending = [(label, key, len(results[key]))
               for label, key, _ in TABLE_ORDER
               if results[key] != CANT_PROBE and len(results[key]) < TOTAL_RUNS]
    if pending:
        print(f"\n⏳ 尚未完成所有 {TOTAL_RUNS} runs 的模型：")
        for label, key, done in pending:
            print(f"   {label}: {done}/{TOTAL_RUNS} runs")
        print("   → 訓練完成後重新執行此腳本即可自動補上")

    print("\n🏁 完成！")
