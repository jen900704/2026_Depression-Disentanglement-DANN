"""
XLS-R + eGeMAPS（無 DANN，無 fine-tune，Scenario A）
=====================================================
架構說明：
  - XLS-R (wav2vec2-xls-r-300m) 主幹：完全凍結
  - mean pooling 取最後一層 hidden state → 1024 維
  - eGeMAPS (opensmile, eGeMAPSv02 Functionals)：88 維
  - 特徵拼接後 → down_proj(256) → dep_classifier (binary)
  - 無 DANN（移除 GRL 和 spk_classifier）
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
#  設定區
# ============================================================
TRAIN_CSV  = "./experiment_sisman_scientific/scenario_A_screening/train.csv"
TEST_CSV   = "./experiment_sisman_scientific/scenario_A_screening/test.csv"
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"

MODEL_NAME  = "facebook/wav2vec2-xls-r-300m"
OUTPUT_DIR  = "./output_xlsr_egemaps_A"
EGEMAPS_DIM = 88

SEED             = 103
TOTAL_RUNS       = 5
NUM_EPOCHS       = 10
LEARNING_RATE    = 1e-4
BATCH_SIZE       = 2
GRAD_ACCUM       = 8
EVAL_STEPS       = 10
SAVE_STEPS       = 10
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
#  ModelOutput
# ============================================================
@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss:          Optional[torch.FloatTensor]          = None
    logits:        torch.FloatTensor                    = None
    hidden_states: Optional[Tuple[torch.FloatTensor]]  = None
    attentions:    Optional[Tuple[torch.FloatTensor]]   = None


# ============================================================
#  模型定義（無 DANN）
# ============================================================
class XLSR_eGeMaps(Wav2Vec2PreTrainedModel):
    """
    XLS-R (frozen) + eGeMAPS concat，純分類，無 DANN
    - xlsr mean pooling: 1024 維
    - eGeMAPS functionals: 88 維
    - concat → 1112 維 → down_proj(256) → dep_classifier
    """
    def __init__(self, config, egemaps_dim: int = EGEMAPS_DIM):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)

        # 凍結 XLS-R 全部參數
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        combined_dim = config.hidden_size + egemaps_dim  # 1024 + 88 = 1112
        self.down_proj = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.dep_classifier = nn.Linear(256, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_values,
        attention_mask=None,
        egemaps_feat=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs   = self.wav2vec2(input_values, attention_mask=attention_mask, return_dict=True)
        xlsr_feat = torch.mean(outputs.last_hidden_state, dim=1)  # [B, 1024]

        if egemaps_feat is not None:
            egemaps_feat = egemaps_feat.to(xlsr_feat.dtype)
            combined = torch.cat([xlsr_feat, egemaps_feat], dim=-1)  # [B, 1112]
        else:
            zero_pad = torch.zeros(xlsr_feat.size(0), EGEMAPS_DIM,
                                   dtype=xlsr_feat.dtype, device=xlsr_feat.device)
            combined = torch.cat([xlsr_feat, zero_pad], dim=-1)

        shared     = self.down_proj(combined)        # [B, 256]
        dep_logits = self.dep_classifier(shared)     # [B, 2]

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(dep_logits.view(-1, self.config.num_labels), labels.view(-1))

        return SpeechClassifierOutput(loss=loss, logits=dep_logits)


# ============================================================
#  DataCollator（含 egemaps_feat，無 speaker_labels）
# ============================================================
@dataclass
class DataCollatorWithEGeMAPS:
    processor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.processor.pad(
            [{"input_values": f["input_values"]} for f in features],
            padding=self.padding, return_tensors="pt",
        )
        batch["labels"]       = torch.tensor([f["labels"]       for f in features], dtype=torch.long)
        batch["egemaps_feat"] = torch.tensor(
            np.stack([f["egemaps_feat"] for f in features]), dtype=torch.float32
        )
        return batch


# ============================================================
#  compute_metrics
# ============================================================
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    true_labels = p.label_ids[0] if isinstance(p.label_ids, tuple) else p.label_ids
    return {
        "accuracy": accuracy_score(true_labels, preds),
        "f1":       f1_score(true_labels, preds, average="macro"),
    }


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


def load_audio_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"📂 讀取 {csv_path}，共 {len(df)} 筆資料")

    records = []
    skipped = 0
    for _, row in df.iterrows():
        wav_path  = os.path.join(AUDIO_ROOT, row["path"])
        raw_label = str(row["label"]).strip().lower()
        if raw_label not in LABEL_MAP:
            skipped += 1; continue
        if not os.path.exists(wav_path):
            print(f"⚠️ 檔案不存在: {wav_path}"); skipped += 1; continue
        records.append({"path": wav_path, "label": LABEL_MAP[raw_label]})

    if skipped:
        print(f"⚠️ 跳過 {skipped} 筆")
    print(f"✅ 成功載入 {len(records)} 筆資料")

    return HFDataset.from_dict({
        "path":  [r["path"]  for r in records],
        "label": [r["label"] for r in records],
    })


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


# ============================================================
#  評估
# ============================================================
def full_evaluation(trainer, test_dataset, output_dir, run_i):
    predictions = trainer.predict(test_dataset)
    preds  = predictions.predictions
    if isinstance(preds, tuple): preds = preds[0]
    y_pred = np.argmax(preds, axis=1)
    y_true = predictions.label_ids
    if isinstance(y_true, tuple): y_true = y_true[0]

    results_path = os.path.join(output_dir, f"results_run{run_i}")
    os.makedirs(results_path, exist_ok=True)

    report    = classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    cm_df     = pd.DataFrame(confusion_matrix(y_true, y_pred), index=LABEL_NAMES, columns=LABEL_NAMES)
    report_df["MSE"]  = mean_squared_error(y_true, y_pred)
    report_df["RMSE"] = sqrt(report_df["MSE"].iloc[0])
    report_df.to_csv(os.path.join(results_path, "clsf_report.csv"), sep="\t")
    cm_df.to_csv(    os.path.join(results_path, "conf_matrix.csv"), sep="\t")

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc     = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC - XLS-R+eGeMAPS Scenario A Run {run_i}")
    plt.legend(); plt.savefig(os.path.join(results_path, "roc_curve.png")); plt.close()

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"\n🎯 Run {run_i}: Acc={acc:.4f} | F1={f1:.4f} | AUC={roc_auc:.4f}")
    return {"accuracy": acc, "f1": f1, "auc": roc_auc}


# ============================================================
#  主程式
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 XLS-R + eGeMAPS（無 DANN）— Scenario A")
    print(f"   XLS-R：全部凍結 | eGeMAPS：{EGEMAPS_DIM} 維")
    print("=" * 60)

    set_seed(SEED)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

    print("\n📦 載入資料集（只執行一次）...")
    train_dataset_full = load_audio_dataset(TRAIN_CSV)
    test_dataset_raw   = load_audio_dataset(TEST_CSV)

    print("\n🔊 預處理音訊 + 提取 eGeMAPS（只執行一次）...")
    train_dataset_full = train_dataset_full.map(speech_file_to_array_fn, fn_kwargs={"processor": processor})
    test_dataset_raw   = test_dataset_raw.map(  speech_file_to_array_fn, fn_kwargs={"processor": processor})
    train_dataset_full = train_dataset_full.map(preprocess_function,      fn_kwargs={"processor": processor})
    test_dataset       = test_dataset_raw.map(  preprocess_function,      fn_kwargs={"processor": processor})

    all_results = []
    for run_i in range(1, TOTAL_RUNS + 1):
        print(f"\n{'='*60}\n🎬 Run {run_i} / {TOTAL_RUNS}\n{'='*60}")
        run_seed = SEED + run_i - 1
        set_seed(run_seed)
        print(f"🎲 seed: {run_seed}")

        train_dataset = train_dataset_full
        eval_dataset  = test_dataset
        print(f"📊 Train: {len(train_dataset)} | Test(eval): {len(test_dataset)}")

        config = Wav2Vec2Config.from_pretrained(MODEL_NAME, num_labels=2, final_dropout=0.1)
        model  = XLSR_eGeMaps.from_pretrained(MODEL_NAME, config=config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🔥 可訓練參數: {trainable:,}")

        run_output_dir = os.path.join(OUTPUT_DIR, f"run_{run_i}")
        os.makedirs(run_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=run_output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            evaluation_strategy="steps",
            num_train_epochs=NUM_EPOCHS,
            fp16=FP16,
            save_steps=SAVE_STEPS,
            eval_steps=EVAL_STEPS,
            logging_steps=LOGGING_STEPS,
            learning_rate=LEARNING_RATE,
            save_total_limit=SAVE_TOTAL_LIMIT,
            label_names=["labels"],
            dataloader_drop_last=True,
            seed=run_seed,
            data_seed=run_seed,
            load_best_model_at_end=True,
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            data_collator=DataCollatorWithEGeMAPS(processor=processor),
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor,
        )

        print("⚔️ 開始訓練...")
        try:
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️ OOM，跳過此 run")
                torch.cuda.empty_cache(); continue
            raise e

        best_path = os.path.join(run_output_dir, "best_model")
        trainer.save_model(best_path)
        processor.save_pretrained(best_path)
        print(f"💾 最佳模型儲存至: {best_path}")

        results = full_evaluation(trainer, test_dataset, OUTPUT_DIR, run_i)
        results["run"] = run_i
        all_results.append(results)

        import gc; del model, trainer; torch.cuda.empty_cache(); gc.collect()

    if all_results:
        print(f"\n{'='*60}\n📈 Scenario A XLS-R+eGeMAPS — {TOTAL_RUNS} 次彙總\n{'='*60}")
        results_df = pd.DataFrame(all_results)
        for metric in ["accuracy", "f1", "auc"]:
            vals = results_df[metric].values
            print(f"  {metric.upper():10s}  mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")
        results_df.to_csv(os.path.join(OUTPUT_DIR, "summary_5runs.csv"), index=False)

    print("\n🏁 Scenario A 完成！")
