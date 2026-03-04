"""
XLS-R + eGeMAPS + DANN（Fine-tune Transformer，Scenario B）
===========================================================
對齊 A 腳本成功邏輯：
  - 修正：CSV 指向 scenario_B_monitoring
  - 修正：Output 目錄為 _B
  - 修正：TrainingArguments 重複參數與最佳模型指標
  - 修正：F1 score 統一為 binary 並處理 DANN Tuple 輸出
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import opensmile
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union, Any
from math import sqrt, exp

from torch.autograd import Function
from datasets import Dataset as HFDataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Config,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
)
from transformers.file_utils import ModelOutput
from packaging import version
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc
)

# ============================================================
#  設定區 (已對齊 Scenario B)
# ============================================================
TRAIN_CSV  = "/home/xzhao117/hyeh_project/experiment_sisman_scientific/scenario_B_monitoring/train.csv"
TEST_CSV   = "/home/xzhao117/hyeh_project/experiment_sisman_scientific/scenario_B_monitoring/test.csv"
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"

MODEL_NAME  = "facebook/wav2vec2-xls-r-300m"
OUTPUT_DIR  = "./output_xlsr_egemaps_dann_finetune_B"  # 確保是 B
EGEMAPS_DIM = 88

SEED             = 103
TOTAL_RUNS       = 1
NUM_EPOCHS       = 10
LEARNING_RATE    = 1e-5
BATCH_SIZE       = 1
GRAD_ACCUM       = 4
EVAL_STEPS       = 200
SAVE_STEPS       = 200
LOGGING_STEPS    = 10
SAVE_TOTAL_LIMIT = 2
FP16             = torch.cuda.is_available()

LABEL_MAP   = {"non": 0, "0": 0, 0: 0, "dep": 1, "1": 1, 1: 1}
LABEL_NAMES = ["non-depressed", "depressed"]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

SMILE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# ============================================================
#  ModelOutput & GRL
# ============================================================
@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss:           Optional[torch.FloatTensor] = None
    logits:         torch.FloatTensor           = None
    speaker_logits: Optional[torch.FloatTensor] = None

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

# ============================================================
#  模型定義
# ============================================================
class XLSR_eGeMaps_DANN_FT(Wav2Vec2PreTrainedModel):
    def __init__(self, config, egemaps_dim: int = EGEMAPS_DIM):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.wav2vec2.feature_extractor._freeze_parameters()

        combined_dim = config.hidden_size + egemaps_dim
        self.down_proj = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),  # 配合 Batch Size 1 增加穩定性
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.dep_classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, config.num_labels)
        )
        self.spk_classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, getattr(config, "num_speakers", 189)) # B 預設較多 speaker
        )
        self._alpha = 0.0
        self.init_weights()

    def forward(self, input_values, attention_mask=None, egemaps_feat=None, labels=None, speaker_labels=None, **kwargs):
        outputs   = self.wav2vec2(input_values, attention_mask=attention_mask, return_dict=True)
        xlsr_feat = torch.mean(outputs.last_hidden_state, dim=1)

        if egemaps_feat is not None:
            egemaps_feat = egemaps_feat.to(xlsr_feat.dtype)
            combined = torch.cat([xlsr_feat, egemaps_feat], dim=-1)
        else:
            zero_pad = torch.zeros(xlsr_feat.size(0), EGEMAPS_DIM, dtype=xlsr_feat.dtype, device=xlsr_feat.device)
            combined = torch.cat([xlsr_feat, zero_pad], dim=-1)

        shared     = self.down_proj(combined)
        dep_logits = self.dep_classifier(shared)
        spk_logits = self.spk_classifier(GradientReversalFn.apply(shared, self._alpha))

        loss = None
        if labels is not None:
            loss_dep = nn.CrossEntropyLoss()(dep_logits.view(-1, self.config.num_labels), labels.view(-1))
            loss = loss_dep
        if speaker_labels is not None:
            mask = speaker_labels >= 0
            if mask.sum() > 0:
                loss_spk = nn.CrossEntropyLoss()(spk_logits[mask].view(-1, spk_logits.size(-1)), speaker_labels[mask].view(-1))
                loss = loss + self._alpha * loss_spk if loss is not None else loss_spk

        return SpeechClassifierOutput(loss=loss, logits=dep_logits, speaker_logits=spk_logits)

# ============================================================
#  Data Processing & Metrics (對齊 A 成功邏輯)
# ============================================================
@dataclass
class DataCollatorWithEGeMAPSAndSpeaker:
    processor: Wav2Vec2FeatureExtractor
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.processor.pad([{"input_values": f["input_values"]} for f in features], padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        batch["speaker_labels"] = torch.tensor([f["speaker_labels"] for f in features], dtype=torch.long)
        batch["egemaps_feat"] = torch.tensor(np.stack([f["egemaps_feat"] for f in features]), dtype=torch.float32)
        return batch

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    true_labels = p.label_ids[0] if isinstance(p.label_ids, tuple) or (isinstance(p.label_ids, np.ndarray) and p.label_ids.ndim == 2) else p.label_ids
    return {
        "accuracy": accuracy_score(true_labels, preds),
        "f1": f1_score(true_labels, preds, average="binary"),
    }

class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        # 計算總步數 (保持原本邏輯)
        total_steps = self.args.max_steps if self.args.max_steps > 0 else (
            len(self.train_dataset) // (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps) * int(self.args.num_train_epochs)
        )
        
        # 計算進度 p (0.0 -> 1.0)
        p = float(self.state.global_step) / max(total_steps, 1)
        
        # ============================================================
        # 🔑 關鍵修改：降低 Alpha 強度
        # ============================================================
        # 原本公式 (0 -> 1.0)：
        # alpha = 2.0 / (1.0 + exp(-10.0 * p)) - 1.0
        
        # 修正後 (0 -> 0.1)：乘以 0.1 防止 Speaker Loss 喧賓奪主
        alpha = (2.0 / (1.0 + exp(-10.0 * p)) - 1.0) * 0.1
        
        model._alpha = alpha
        
        return super().training_step(model, inputs)

# ============================================================
#  資料讀取與提取
# ============================================================
def extract_speaker_id(filepath: str) -> str:
    return os.path.basename(filepath).split("_")[0]

def extract_egemaps(wav_path: str) -> np.ndarray:
    try:
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != 16000: waveform = torchaudio.functional.resample(waveform, sr, 16000)
        feat = SMILE.process_signal(waveform.squeeze().numpy(), 16000).values.flatten().astype(np.float32)
        return np.nan_to_num(feat, nan=0.0)
    except Exception: return np.zeros(EGEMAPS_DIM, dtype=np.float32)

def load_audio_dataset(csv_path: str, speaker_to_idx: dict = None, is_train: bool = True):
    df = pd.read_csv(csv_path)
    if is_train and speaker_to_idx is None:
        all_speakers = sorted(set(extract_speaker_id(p) for p in df["path"].tolist()))
        speaker_to_idx = {spk: idx for idx, spk in enumerate(all_speakers)}
    
    records = []
    for _, row in df.iterrows():
        wav_path = os.path.join(AUDIO_ROOT, row["path"])
        if not os.path.exists(wav_path): continue
        records.append({
            "path": wav_path, "label": LABEL_MAP[str(row["label"]).strip().lower()],
            "speaker_labels": speaker_to_idx.get(extract_speaker_id(wav_path), -1),
        })
    return HFDataset.from_dict({k: [r[k] for r in records] for k in ["path", "label", "speaker_labels"]}), speaker_to_idx

def speech_file_to_array_fn(batch, processor):
    speech, sr = torchaudio.load(batch["path"])
    if speech.shape[0] > 1: speech = torch.mean(speech, dim=0, keepdim=True)
    speech = speech.squeeze().numpy()
    if sr != 16000:
        import librosa
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
    batch["speech"] = speech
    batch["egemaps_feat"] = extract_egemaps(batch["path"])
    return batch

def preprocess_function(batch, processor):
    result = processor(batch["speech"], sampling_rate=16000, return_tensors="np", padding=False)
    batch["input_values"] = result.input_values[0]
    batch["labels"] = batch["label"]
    return batch

# ============================================================
#  評估與主程式
# ============================================================
def full_evaluation(trainer, test_dataset, output_dir, run_i):
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
    y_pred = np.argmax(preds, axis=1)
    y_true = predictions.label_ids
    
    results_path = os.path.join(output_dir, f"results_run{run_i}")
    os.makedirs(results_path, exist_ok=True)
    pd.DataFrame(classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)).transpose().to_csv(os.path.join(results_path, "clsf_report.csv"), sep="\t")
    
    fpr, tpr, _ = roc_curve(y_true, y_pred); roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.2f})"); plt.legend(); plt.savefig(os.path.join(results_path, "roc_curve.png")); plt.close()
    return {"accuracy": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred, average="binary"), "auc": roc_auc}

if __name__ == "__main__":
    print("🚀 XLS-R + eGeMAPS + DANN FT - Scenario B")
    set_seed(SEED)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

    train_dataset_full, speaker_to_idx = load_audio_dataset(TRAIN_CSV, is_train=True)
    test_dataset_raw, _ = load_audio_dataset(TEST_CSV, speaker_to_idx=speaker_to_idx, is_train=False)
    
    train_dataset_full = train_dataset_full.map(speech_file_to_array_fn, fn_kwargs={"processor": processor})
    test_dataset_raw = test_dataset_raw.map(speech_file_to_array_fn, fn_kwargs={"processor": processor})
    train_dataset_full = train_dataset_full.map(preprocess_function, fn_kwargs={"processor": processor})
    test_dataset = test_dataset_raw.map(preprocess_function, fn_kwargs={"processor": processor})

    all_results = []
    for run_i in range(1, TOTAL_RUNS + 1):
        set_seed(SEED + run_i - 1)
        config = Wav2Vec2Config.from_pretrained(MODEL_NAME, num_labels=2, num_speakers=len(speaker_to_idx), final_dropout=0.1)
        model = XLSR_eGeMaps_DANN_FT.from_pretrained(MODEL_NAME, config=config)
        
        run_output_dir = os.path.join(OUTPUT_DIR, f"run_{run_i}")
        os.makedirs(run_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=run_output_dir, per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM, evaluation_strategy="steps",
            num_train_epochs=NUM_EPOCHS, fp16=FP16, gradient_checkpointing=True,
            save_steps=SAVE_STEPS, eval_steps=EVAL_STEPS, logging_steps=LOGGING_STEPS,
            learning_rate=LEARNING_RATE, save_total_limit=SAVE_TOTAL_LIMIT,
            load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
            remove_unused_columns=False, report_to="none"
        )

        trainer = CTCTrainer(
            model=model, data_collator=DataCollatorWithEGeMAPSAndSpeaker(processor=processor),
            args=training_args, compute_metrics=compute_metrics,
            train_dataset=train_dataset_full, eval_dataset=test_dataset
        )

        trainer.train()
        
        # 存檔對齊 B
        pth_path = os.path.join(OUTPUT_DIR, f"xlsr_egemaps_dann_finetune_B_shared_encoder_run_{run_i}.pth")
        torch.save(trainer.model.down_proj.state_dict(), pth_path)
        all_results.append(full_evaluation(trainer, test_dataset, OUTPUT_DIR, run_i))

    print("\n🏁 Scenario B 完成！")