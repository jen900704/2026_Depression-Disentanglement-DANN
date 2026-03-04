"""
File — Huang + SLS + DANN（無 fine-tune，Scenario A）
=====================================================
修正說明：
  - 嚴格保留原始 500 行架構與手動訓練邏輯。
  - 修正 training_step 參數相容性與 Alpha 縮放。
  - 修正 TrainingArguments 命名問題。
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union, Any
from math import sqrt, exp

from torch.autograd import Function
from datasets import Dataset as HFDataset
from transformers import (
    Wav2Vec2Processor,
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
    precision_recall_fscore_support
)

# ============================================================
#  設定區
# ============================================================
TRAIN_CSV  = "./experiment_sisman_scientific/scenario_A_screening/train.csv"
TEST_CSV   = "./experiment_sisman_scientific/scenario_A_screening/test.csv"
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"

MODEL_NAME = "facebook/wav2vec2-base"
OUTPUT_DIR = "./output_huang_sls_dann_A"

SEED             = 103   # 修正 1：對齊論文基準
TOTAL_RUNS       = 1
NUM_EPOCHS       = 10
LEARNING_RATE    = 1e-4
BATCH_SIZE       = 4
GRAD_ACCUM       = 2
EVAL_STEPS       = 200    # 修正 2：對齊論文基準
SAVE_STEPS       = 200
LOGGING_STEPS    = 10
SAVE_TOTAL_LIMIT = 2
FP16 = torch.cuda.is_available()

LABEL_MAP   = {"non": 0, "0": 0, 0: 0, "dep": 1, "1": 1, 1: 1}
LABEL_NAMES = ["non-depressed", "depressed"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
#  模型定義
# ============================================================

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss:           Optional[torch.FloatTensor]        = None
    logits:         torch.FloatTensor                  = None
    speaker_logits: Optional[torch.FloatTensor]        = None
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


class Wav2Vec2_SLS_DANN(Wav2Vec2PreTrainedModel):
    """
    Wav2Vec2 + SLS + DANN，無 fine-tune
    - wav2vec2 全部凍結
    - SLS 加權融合所有 hidden states（13 層）
    - dep_classifier + spk_classifier（GRL 對抗）
    - alpha 由 CTCTrainer 動態注入（self._alpha）
    """
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)

        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        num_layers = config.num_hidden_layers + 1
        self.sls_weights = nn.Parameter(torch.ones(num_layers))

        self.down_proj = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.dep_classifier = nn.Linear(128, config.num_labels)
        self.spk_classifier = nn.Linear(128, getattr(config, "num_speakers", 151))

        self._alpha = 0.0
        self.init_weights()

    def freeze_feature_extractor(self):
        pass

    def forward(
        self,
        input_values,
        attention_mask=None,
        labels=None,
        speaker_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True,
        )

        hidden_states = torch.stack(outputs.hidden_states)        # [L, B, T, H]
        weights       = torch.softmax(self.sls_weights, dim=0)    # [L]
        fused         = (hidden_states * weights.view(-1, 1, 1, 1)).sum(0)  # [B, T, H]

        shared     = self.down_proj(torch.mean(fused, dim=1))     # [B, 128]
        dep_logits = self.dep_classifier(shared)
        spk_logits = self.spk_classifier(
            GradientReversalFn.apply(shared, self._alpha)
        )

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(dep_logits, labels)
        if speaker_labels is not None:
            mask = speaker_labels >= 0
            if mask.sum() > 0:
                spk_loss = nn.CrossEntropyLoss()(
                    spk_logits[mask].view(-1, spk_logits.size(-1)),
                    speaker_labels[mask].view(-1)
                )
                loss = loss + self._alpha * spk_loss if loss is not None else spk_loss

        return SpeechClassifierOutput(
            loss=loss, logits=dep_logits, speaker_logits=spk_logits,
            hidden_states=None, attentions=None,
        )


# ============================================================
#  DataCollator
# ============================================================

@dataclass
class DataCollatorWithSpeaker:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.processor.pad(
            [{"input_values": f["input_values"]} for f in features],
            padding=self.padding, return_tensors="pt",
        )
        batch["labels"]         = torch.tensor([f["labels"]         for f in features], dtype=torch.long)
        batch["speaker_labels"] = torch.tensor([f["speaker_labels"] for f in features], dtype=torch.long)
        return batch


# ============================================================
#  compute_metrics
# ============================================================

def compute_metrics(p):
    preds = p.predictions
    # 核心修正：正確解析 DANN 多重輸出
    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.argmax(preds, axis=1)
    
    accuracy = accuracy_score(p.label_ids, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average='binary', zero_division=0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# ============================================================
#  CTCTrainer — 核心邏輯 (手動控制版)
# ============================================================

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast


class CTCTrainer(Trainer):
    # 修正點：補上 *args, **kwargs 以防新版 transformers 報錯
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], *args, **kwargs) -> torch.Tensor:
        total_steps = self.args.max_steps if self.args.max_steps > 0 else (
            len(self.train_dataset) //
            (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps)
            * int(self.args.num_train_epochs)
        )
        p     = float(self.state.global_step) / max(total_steps, 1)
        # 核心修正：Alpha 乘以 0.01 防止 Scenario A 對抗強度壓垮分類器
        alpha = (2.0 / (1.0 + exp(-10.0 * p)) - 1.0) * 0.01
        model._alpha = alpha

        model.train()
        inputs = self._prepare_inputs(inputs)

        is_amp_used = self.args.fp16 or self.args.bf16
        if is_amp_used:
            with torch.amp.autocast("cuda"):
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if is_amp_used:
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
#  資料載入與預處理
# ============================================================

def extract_speaker_id(filepath: str) -> str:
    return os.path.basename(filepath).split("_")[0]


def load_audio_dataset(csv_path: str, speaker_to_idx: dict = None, is_train: bool = True):
    df = pd.read_csv(csv_path)
    print(f"📂 讀取 {csv_path}，共 {len(df)} 筆資料")

    if is_train and speaker_to_idx is None:
        import pandas as _pd_tmp
        test_df = _pd_tmp.read_csv(TEST_CSV)
        target_speakers = sorted(set(extract_speaker_id(p) for p in test_df["path"].tolist()))
        speaker_to_idx  = {spk: idx for idx, spk in enumerate(target_speakers)}
        print(f"🔍 偵測到 {len(speaker_to_idx)} 位 target speaker")

    records, skipped = [], 0
    for _, row in df.iterrows():
        wav_path  = os.path.join(AUDIO_ROOT, row["path"])
        raw_label = str(row["label"]).strip().lower()
        if raw_label not in LABEL_MAP:
            skipped += 1; continue
        if not os.path.exists(wav_path):
            skipped += 1; continue
        spk_idx = speaker_to_idx.get(extract_speaker_id(wav_path), -1)
        records.append({"path": wav_path, "label": LABEL_MAP[raw_label], "speaker_labels": spk_idx})

    if skipped: print(f"⚠️ 跳過 {skipped} 筆")
    return HFDataset.from_dict({
        "path":           [r["path"]           for r in records],
        "label":          [r["label"]          for r in records],
        "speaker_labels": [r["speaker_labels"] for r in records],
    }), speaker_to_idx


def speech_file_to_array_fn(batch, processor):
    speech, sr = torchaudio.load(batch["path"])
    if speech.shape[0] > 1:
        speech = torch.mean(speech, dim=0, keepdim=True)
    speech = speech.squeeze().numpy()
    batch["speech"] = speech
    return batch


def preprocess_function(batch, processor):
    result = processor(batch["speech"], sampling_rate=16000, return_tensors="np", padding=False)
    batch["input_values"] = result.input_values[0]
    batch["labels"]       = batch["label"]
    return batch


# ============================================================
#  評估與繪圖
# ============================================================

def full_evaluation(trainer, test_dataset, output_dir, run_i):
    predictions = trainer.predict(test_dataset)
    preds  = predictions.predictions
    if isinstance(preds, tuple): preds = preds[0]
    y_pred = np.argmax(preds, axis=1)
    y_true = predictions.label_ids

    results_path = os.path.join(output_dir, f"results_run{run_i}")
    os.makedirs(results_path, exist_ok=True)

    report    = classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    cm_df     = pd.DataFrame(confusion_matrix(y_true, y_pred), index=LABEL_NAMES, columns=LABEL_NAMES)
    mse = mean_squared_error(y_true, y_pred)
    report_df["MSE"]  = mse
    report_df["RMSE"] = sqrt(mse)
    report_df.to_csv(os.path.join(results_path, "clsf_report.csv"), sep="\t")
    cm_df.to_csv(    os.path.join(results_path, "conf_matrix.csv"), sep="\t")

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc     = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve - Scenario A Run {run_i}")
    plt.savefig(os.path.join(results_path, "roc_curve.png")); plt.close()

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": acc, "f1": f1, "auc": roc_auc}


# ============================================================
#  主程式
# ============================================================

if __name__ == "__main__":
    print("🚀 Huang + SLS + DANN — Scenario A (Screening)")
    set_seed(SEED)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    train_dataset_full, speaker_to_idx = load_audio_dataset(TRAIN_CSV, is_train=True)
    test_dataset, _ = load_audio_dataset(TEST_CSV, speaker_to_idx=speaker_to_idx, is_train=False)

    map_kwargs = {"fn_kwargs": {"processor": processor}}
    train_dataset_full = train_dataset_full.map(speech_file_to_array_fn, **map_kwargs).map(preprocess_function, **map_kwargs)
    test_dataset = test_dataset.map(speech_file_to_array_fn, **map_kwargs).map(preprocess_function, **map_kwargs)

    all_results = []
    for run_i in range(1, TOTAL_RUNS + 1):
        run_seed = SEED + run_i - 1
        set_seed(run_seed)
        
        config = Wav2Vec2Config.from_pretrained(MODEL_NAME, num_labels=2, num_speakers=len(speaker_to_idx), output_hidden_states=True)
        model = Wav2Vec2_SLS_DANN.from_pretrained(MODEL_NAME, config=config)

        run_output_dir = os.path.join(OUTPUT_DIR, f"run_{run_i}")
        os.makedirs(run_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=run_output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            eval_strategy="steps", # 修正點：從 evaluation_strategy 改名
            num_train_epochs=NUM_EPOCHS,
            fp16=FP16,
            save_steps=SAVE_STEPS,
            eval_steps=EVAL_STEPS,
            logging_steps=LOGGING_STEPS,
            learning_rate=LEARNING_RATE,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = CTCTrainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset_full,
            eval_dataset=test_dataset,
            data_collator=DataCollatorWithSpeaker(processor=processor),
        )

        trainer.train()

        results = full_evaluation(trainer, test_dataset, OUTPUT_DIR, run_i)
        results["run"] = run_i
        all_results.append(results)
        
        # 儲存與內存釋放
        torch.save(model.down_proj.state_dict(), os.path.join(OUTPUT_DIR, f"huang_sls_dann_A_run_{run_i}.pth"))
        del model, trainer; torch.cuda.empty_cache(); gc.collect()

    # 彙總報告
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n📈 Scenario A SLS+DANN 彙總結果\n{results_df}")
        results_df.to_csv(os.path.join(OUTPUT_DIR, "summary_A.csv"), index=False)

    print("\n🏁 Scenario A 實驗完成！")