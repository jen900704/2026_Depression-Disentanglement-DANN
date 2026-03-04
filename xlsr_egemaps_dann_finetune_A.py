"""
XLS-R + eGeMAPS + DANN（Fine-tune Transformer，Scenario A）
===========================================================
修正說明：
  - 修正 CSV 路徑指向 Scenario A
  - 修正 Output 目錄為 _A
  - 修正 Alpha 為 0.01 (防止訓練崩潰)
  - 修正 TrainingArguments 設定 (以 F1 挑選最佳模型)
  - 修正 F1 score 為 binary
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
    confusion_matrix, roc_curve, auc, mean_squared_error,
)

# ============================================================
#  設定區 (已修正為 Scenario A)
# ============================================================
TRAIN_CSV  = "./experiment_sisman_scientific/scenario_A_screening/train.csv"
TEST_CSV   = "./experiment_sisman_scientific/scenario_A_screening/test.csv"
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"

MODEL_NAME  = "facebook/wav2vec2-xls-r-300m"
OUTPUT_DIR  = "./output_xlsr_egemaps_dann_finetune_A"  # 修正為 A
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
    hidden_states:  Optional[Tuple[torch.FloatTensor]] = None
    attentions:     Optional[Tuple[torch.FloatTensor]] = None


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# ============================================================
#  模型定義（Fine-tune 版：CNN 凍結，Transformer 可訓練）
# ============================================================
class XLSR_eGeMaps_DANN_FT(Wav2Vec2PreTrainedModel):
    def __init__(self, config, egemaps_dim: int = EGEMAPS_DIM):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)

        # 只凍結 CNN，Transformer 可訓練
        self.wav2vec2.feature_extractor._freeze_parameters()

        combined_dim = config.hidden_size + egemaps_dim
        self.down_proj = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.dep_classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, config.num_labels)
        )
        self.spk_classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, getattr(config, "num_speakers", 151))
        )

        self._alpha = 0.0
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def get_embedding(self, input_values, egemaps_feat=None, attention_mask=None):
        outputs  = self.wav2vec2(input_values, attention_mask=attention_mask, return_dict=True)
        xlsr_feat = torch.mean(outputs.last_hidden_state, dim=1)
        if egemaps_feat is not None:
            egemaps_feat = egemaps_feat.to(xlsr_feat.dtype)
            combined = torch.cat([xlsr_feat, egemaps_feat], dim=-1)
        else:
            zero_pad = torch.zeros(xlsr_feat.size(0), EGEMAPS_DIM, dtype=xlsr_feat.dtype, device=xlsr_feat.device)
            combined = torch.cat([xlsr_feat, zero_pad], dim=-1)
        return self.down_proj(combined)

    def forward(
        self,
        input_values,
        attention_mask=None,
        egemaps_feat=None,
        labels=None,
        speaker_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
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
                loss_spk = nn.CrossEntropyLoss()(
                    spk_logits[mask].view(-1, spk_logits.size(-1)),
                    speaker_labels[mask].view(-1)
                )
                loss = loss + self._alpha * loss_spk if loss is not None else loss_spk

        return SpeechClassifierOutput(
            loss=loss,
            logits=dep_logits,
            speaker_logits=spk_logits,
        )


# ============================================================
#  DataCollator
# ============================================================
@dataclass
class DataCollatorWithEGeMAPSAndSpeaker:
    processor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.processor.pad(
            [{"input_values": f["input_values"]} for f in features],
            padding=self.padding, return_tensors="pt",
        )
        batch["labels"]         = torch.tensor([f["labels"]         for f in features], dtype=torch.long)
        batch["speaker_labels"] = torch.tensor([f["speaker_labels"] for f in features], dtype=torch.long)
        batch["egemaps_feat"]   = torch.tensor(
            np.stack([f["egemaps_feat"] for f in features]), dtype=torch.float32
        )
        return batch


# ============================================================
#  compute_metrics (修正版)
# ============================================================
def compute_metrics(p: EvalPrediction):
    # Fix: 解開 DANN Tuple
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    
    true_labels = p.label_ids[0] if isinstance(p.label_ids, tuple) or (isinstance(p.label_ids, np.ndarray) and p.label_ids.ndim == 2) else p.label_ids
    return {
        "accuracy": accuracy_score(true_labels, preds),
        "f1":       f1_score(true_labels, preds, average="binary"), # Fix: 統一用 binary
    }


# ============================================================
#  CTCTrainer
# ============================================================
class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        # 1. 計算總訓練步數 (用於控制 Alpha 的成長曲線)
        if self.args.max_steps > 0:
            total_steps = self.args.max_steps
        else:
            # 估算總 step 數： (資料量 / batch_size) * epochs
            num_update_steps_per_epoch = len(self.train_dataset) // (
                self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
            )
            total_steps = num_update_steps_per_epoch * int(self.args.num_train_epochs)

        # 2. 計算當前進度 p (0.0 -> 1.0)
        current_step = self.state.global_step
        p = float(current_step) / max(total_steps, 1)

        # ============================================================
        # 🔑 關鍵修正 (Critical Fix)
        # ============================================================
        # 原本公式: alpha 會從 0 成長到 1.0
        # alpha = 2.0 / (1.0 + exp(-10.0 * p)) - 1.0
        
        # 修正後: alpha 會從 0 成長到 0.01
        # 說明: Speaker Loss 本身數值很大 (~5.0)，必須乘上小係數 (0.01)
        # 才能讓它與 Depression Loss (~0.7) 保持平衡。
        alpha = (2.0 / (1.0 + exp(-10.0 * p)) - 1.0) * 0.01
        
        # 將 alpha 傳入模型 (這會影響 GradientReversalLayer)
        model._alpha = alpha

        # ============================================================
        # 3. 標準訓練流程 (Forward + Backward)
        # ============================================================
        model.train()
        inputs = self._prepare_inputs(inputs)

        # 支援混合精度訓練 (FP16/BF16)
        is_amp = self.args.fp16 or self.args.bf16
        if is_amp:
            with torch.amp.autocast("cuda"):
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        # 梯度累積處理
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # 反向傳播 (Backward)
        if is_amp:
            if hasattr(self, "scaler") and self.scaler is not None:
                self.scaler.scale(loss).backward()
            elif hasattr(self, "accelerator"):
                self.accelerator.backward(loss)
            else:
                loss.backward()
        else:
            loss.backward()

        return loss.detach()


# ============================================================
#  資料處理
# ============================================================
def extract_speaker_id(filepath: str) -> str:
    return os.path.basename(filepath).split("_")[0]


def extract_egemaps(wav_path: str) -> np.ndarray:
    try:
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        signal = waveform.squeeze().numpy()
        feat   = SMILE.process_signal(signal, 16000).values.flatten().astype(np.float32)
        return np.nan_to_num(feat, nan=0.0)
    except Exception as e:
        print(f"⚠️ eGeMAPS 提取失敗: {wav_path} → {e}")
        return np.zeros(EGEMAPS_DIM, dtype=np.float32)


def load_audio_dataset(csv_path: str, speaker_to_idx: dict = None, is_train: bool = True):
    df = pd.read_csv(csv_path)
    print(f"📂 讀取 {csv_path}，共 {len(df)} 筆資料")

    if is_train and speaker_to_idx is None:
        # Scenario A：從 TRAIN 建立 Speaker Map
        all_speakers = sorted(set(extract_speaker_id(p) for p in df["path"].tolist()))
        speaker_to_idx = {spk: idx for idx, spk in enumerate(all_speakers)}
        print(f"🔍 偵測到 {len(speaker_to_idx)} 位 speaker（train）")

    records = []
    for _, row in df.iterrows():
        wav_path  = os.path.join(AUDIO_ROOT, row["path"])
        raw_label = str(row["label"]).strip().lower()
        if raw_label not in LABEL_MAP or not os.path.exists(wav_path):
            continue
        spk_str = extract_speaker_id(wav_path)
        records.append({
            "path":           wav_path,
            "label":          LABEL_MAP[raw_label],
            "speaker_labels": speaker_to_idx.get(spk_str, -1),
        })

    dataset = HFDataset.from_dict({
        "path":           [r["path"]           for r in records],
        "label":          [r["label"]          for r in records],
        "speaker_labels": [r["speaker_labels"] for r in records],
    })
    print(f"✅ 成功載入 {len(records)} 筆")
    return dataset, speaker_to_idx


def speech_file_to_array_fn(batch, processor):
    wav_path = batch["path"]
    speech, sr = torchaudio.load(wav_path)
    if speech.shape[0] > 1:
        speech = torch.mean(speech, dim=0, keepdim=True)
    speech = speech.squeeze().numpy()
    if sr != 16000:
        import librosa
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
    batch["speech"]       = speech
    batch["egemaps_feat"] = extract_egemaps(wav_path)
    return batch


def preprocess_function(batch, processor):
    result = processor(batch["speech"], sampling_rate=16000, return_tensors="np", padding=False)
    batch["input_values"] = result.input_values[0]
    batch["labels"]       = batch["label"]
    return batch


def split_train_valid(dataset: HFDataset, valid_ratio: float = 0.15, seed: int = 42):
    split = dataset.train_test_split(test_size=valid_ratio, seed=seed)
    return split["train"], split["test"]


# ============================================================
#  評估
# ============================================================
def full_evaluation(trainer, test_dataset, output_dir, run_i):
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    y_pred = np.argmax(preds, axis=1)
    y_true = predictions.label_ids
    if isinstance(y_true, tuple):
        y_true = y_true[0]

    results_path = os.path.join(output_dir, f"results_run{run_i}")
    os.makedirs(results_path, exist_ok=True)

    report    = classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    cm_df     = pd.DataFrame(confusion_matrix(y_true, y_pred), index=LABEL_NAMES, columns=LABEL_NAMES)
    report_df.to_csv(os.path.join(results_path, "clsf_report.csv"), sep="\t")
    cm_df.to_csv(    os.path.join(results_path, "conf_matrix.csv"), sep="\t")

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc     = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC - XLS-R+eGeMAPS+DANN FT Scenario A Run {run_i}")
    plt.legend(); plt.savefig(os.path.join(results_path, "roc_curve.png")); plt.close()

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="binary")
    print(f"\n🎯 Run {run_i}: Acc={acc:.4f} | F1={f1:.4f} | AUC={roc_auc:.4f}")
    return {"accuracy": acc, "f1": f1, "auc": roc_auc}


# ============================================================
#  主程式
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 XLS-R + eGeMAPS + DANN（Fine-tune Transformer）- Scenario A")
    print("   CNN：凍結 | Transformer：可訓練")
    print(f"   XLS-R：{MODEL_NAME}  eGeMAPS：{EGEMAPS_DIM} 維")
    print("=" * 60)

    set_seed(SEED)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

    print("\n📦 載入資料集（只執行一次）...")
    train_dataset_full, speaker_to_idx = load_audio_dataset(TRAIN_CSV, is_train=True)
    test_dataset_raw, _                = load_audio_dataset(TEST_CSV, speaker_to_idx=speaker_to_idx, is_train=False)
    num_speakers = len(speaker_to_idx)
    print(f"👥 共 {num_speakers} 位 speaker")

    print("\n🔊 預處理音訊 + 提取 eGeMAPS（只執行一次，耗時較長）...")
    train_dataset_full = train_dataset_full.map(speech_file_to_array_fn, fn_kwargs={"processor": processor})
    test_dataset_raw   = test_dataset_raw.map(  speech_file_to_array_fn, fn_kwargs={"processor": processor})
    train_dataset_full = train_dataset_full.map(preprocess_function,      fn_kwargs={"processor": processor})
    test_dataset       = test_dataset_raw.map(  preprocess_function,      fn_kwargs={"processor": processor})

    all_results = []

    for run_i in range(1, TOTAL_RUNS + 1):
        print(f"\n{'='*60}\n🎬 Run {run_i} / {TOTAL_RUNS}\n{'='*60}")
        set_seed(SEED + run_i - 1)

        train_dataset = train_dataset_full

        print(f"📊 Train: {len(train_dataset)} | Test(eval): {len(test_dataset)}")

        config = Wav2Vec2Config.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            num_speakers=num_speakers,
            final_dropout=0.1,
        )
        model = XLSR_eGeMaps_DANN_FT.from_pretrained(MODEL_NAME, config=config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🔥 可訓練參數: {trainable:,}")

        run_output_dir = os.path.join(OUTPUT_DIR, f"run_{run_i}")
        os.makedirs(run_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=run_output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            label_names=["labels"],
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            evaluation_strategy="steps",
            num_train_epochs=NUM_EPOCHS,
            fp16=FP16,
            gradient_checkpointing=True,
            save_steps=SAVE_STEPS,
            eval_steps=EVAL_STEPS,
            logging_steps=LOGGING_STEPS,
            learning_rate=LEARNING_RATE,
            save_total_limit=SAVE_TOTAL_LIMIT,
            seed=SEED + run_i - 1,
            data_seed=SEED + run_i - 1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",  # 👈 修正：指定以 F1 挑選最佳模型
            greater_is_better=True,      # 👈 修正：F1 越高越好
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = CTCTrainer(
            model=model,
            data_collator=DataCollatorWithEGeMAPSAndSpeaker(processor=processor),
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=processor,
        )

        print("⚔️ 開始訓練...")
        try:
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️ OOM，跳過此 run")
                torch.cuda.empty_cache()
                continue
            raise e

        best_path = os.path.join(run_output_dir, "best_model")
        trainer.save_model(best_path)
        processor.save_pretrained(best_path)
        print(f"💾 最佳模型儲存至: {best_path}")

        # 修正：存檔名為 A
        pth_path = os.path.join(OUTPUT_DIR, f"xlsr_egemaps_dann_finetune_A_shared_encoder_run_{run_i}.pth")
        torch.save(trainer.model.down_proj.state_dict(), pth_path)
        print(f"💾 down_proj 已儲存: {pth_path}")

        results = full_evaluation(trainer, test_dataset, OUTPUT_DIR, run_i)
        all_results.append(results)

        import gc; del model, trainer; torch.cuda.empty_cache(); gc.collect()

    if all_results:
        print(f"\n{'='*60}\n📈 跨 Run 統計\n{'='*60}")
        for metric in ["accuracy", "f1", "auc"]:
            vals = [r[metric] for r in all_results]
            print(f"  {metric.upper():10s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")
    print("\n🏁 XLS-R+eGeMAPS+DANN FT Scenario A 完成！")