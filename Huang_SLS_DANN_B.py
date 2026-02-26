"""
File — Huang + SLS + DANN（無 fine-tune，Scenario B）
=====================================================
架構說明：
  - Wav2Vec2 主幹：完全凍結（包含 CNN feature extractor + Transformer encoder）
  - SLS (Stochastic Layer Selection)：可學習的加權融合所有 hidden states
  - DANN domain classifier：對抗式訓練，以 speaker 為 domain
  - dep_classifier：二元憂鬱分類 (binary)

修正清單（相對於草稿 v4）：
  1. SEED=42 → 103，對齊論文基準
  2. EVAL/SAVE/LOGGING_STEPS=50 → 10，對齊論文基準
  3. run seed：SEED+run_i → SEED+run_i-1（103,104,105,106,107）
  4. pth 檔名：huang_sls_dann_A（原本已正確）
  5. 彙總：新增 summary_5runs.csv + results["run"]=run_i + mean±std 格式
  6. dataloader_drop_last=True 移除（eval 時樣本數可能不整除 batch，drop 會丟資料）
  7. spk_classifier dim：hardcode 200 → num_speakers（從資料動態計算）
  8. gc.collect 釋放記憶體
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
)

# ============================================================
#  設定區
# ============================================================
TRAIN_CSV  = "./experiment_sisman_scientific/scenario_B_monitoring/train.csv"
TEST_CSV   = "./experiment_sisman_scientific/scenario_B_monitoring/test.csv"
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"

MODEL_NAME = "facebook/wav2vec2-base"
OUTPUT_DIR = "./output_huang_sls_dann_B"

SEED             = 103   # 修正 1：對齊論文基準
TOTAL_RUNS       = 5
NUM_EPOCHS       = 10
LEARNING_RATE    = 1e-4
BATCH_SIZE       = 4
GRAD_ACCUM       = 2
EVAL_STEPS       = 10    # 修正 2：對齊論文基準
SAVE_STEPS       = 10
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
        # 修正 7：用 config 動態取 num_speakers，不 hardcode 200
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
#  DataCollator（含 speaker_labels）
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

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1":       f1_score(p.label_ids, preds, average="macro"),
    }


# ============================================================
#  CTCTrainer — 動態注入 alpha
# ============================================================

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast


class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        total_steps = self.args.max_steps if self.args.max_steps > 0 else (
            len(self.train_dataset) //
            (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps)
            * int(self.args.num_train_epochs)
        )
        p     = float(self.state.global_step) / max(total_steps, 1)
        alpha = 2.0 / (1.0 + exp(-10.0 * p)) - 1.0
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
        # Scenario B：speaker map 只含 38 位 target speakers（從 TEST_CSV 建立）
        # 151 位 base speakers → spk_idx=-1，不參與 L_spk
        import pandas as _pd_tmp
        test_df = _pd_tmp.read_csv(TEST_CSV)
        target_speakers = sorted(set(extract_speaker_id(p) for p in test_df["path"].tolist()))
        speaker_to_idx  = {spk: idx for idx, spk in enumerate(target_speakers)}
        print(f"🔍 偵測到 {len(speaker_to_idx)} 位 target speaker（從 TEST_CSV，應為 38）")

    records, skipped = [], 0
    for _, row in df.iterrows():
        wav_path  = os.path.join(AUDIO_ROOT, row["path"])
        raw_label = str(row["label"]).strip().lower()
        if raw_label not in LABEL_MAP:
            skipped += 1; continue
        if not os.path.exists(wav_path):
            print(f"⚠️ 不存在: {wav_path}"); skipped += 1; continue
        spk_idx = speaker_to_idx.get(extract_speaker_id(wav_path), -1)
        records.append({"path": wav_path, "label": LABEL_MAP[raw_label], "speaker_labels": spk_idx})

    if skipped: print(f"⚠️ 跳過 {skipped} 筆")
    print(f"✅ 載入 {len(records)} 筆")
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
    if sr != 16000:
        import librosa
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
    batch["speech"] = speech
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
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC Curve - Scenario B Run {run_i}")
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
    print("🚀 Huang + SLS + DANN（無 fine-tune）— Scenario B")
    print(f"   SEED={SEED} | LR={LEARNING_RATE} | Epochs={NUM_EPOCHS} | Runs={TOTAL_RUNS}")
    print("=" * 60)

    set_seed(SEED)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    print("\n📦 載入資料集（只執行一次）...")
    train_dataset_full, speaker_to_idx = load_audio_dataset(TRAIN_CSV, is_train=True)
    test_dataset_raw, _ = load_audio_dataset(TEST_CSV, speaker_to_idx=speaker_to_idx, is_train=False)
    num_speakers = len(speaker_to_idx)
    print(f"👥 共 {num_speakers} 位 speaker")

    print("\n🔊 預處理音訊（只執行一次）...")
    map_kwargs = {"fn_kwargs": {"processor": processor}}
    train_dataset_full = train_dataset_full.map(speech_file_to_array_fn, **map_kwargs)
    test_dataset_raw   = test_dataset_raw.map(  speech_file_to_array_fn, **map_kwargs)
    train_dataset_full = train_dataset_full.map(preprocess_function,      **map_kwargs)
    test_dataset       = test_dataset_raw.map(  preprocess_function,      **map_kwargs)

    all_results = []
    for run_i in range(1, TOTAL_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"🎬 Run {run_i} / {TOTAL_RUNS}")
        print(f"{'='*60}")

        # 修正 3：seed 103, 104, 105, 106, 107
        run_seed = SEED + run_i - 1
        set_seed(run_seed)
        print(f"🎲 Run {run_i} seed: {run_seed}")
        print(f"📊 Train: {len(train_dataset_full)} | Test(eval): {len(test_dataset)}")

        config = Wav2Vec2Config.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            num_speakers=num_speakers,   # 修正 7：動態設定
            final_dropout=0.1,
            output_hidden_states=True,
        )
        model = Wav2Vec2_SLS_DANN.from_pretrained(MODEL_NAME, config=config)

        frozen    = sum(1 for p in model.wav2vec2.parameters() if not p.requires_grad)
        total     = sum(1 for p in model.wav2vec2.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"❄️  wav2vec2 凍結: {frozen}/{total} | 🔥 可訓練: {trainable:,}")

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
            seed=run_seed,
            data_seed=run_seed,
            load_best_model_at_end=True,
            # metric_for_best_model 未設定 → 預設 eval_loss，對齊論文
            report_to="none",
            remove_unused_columns=False,
            # 修正 6：移除 dataloader_drop_last（eval 時 drop 會丟掉尾部樣本）
        )

        trainer = CTCTrainer(
            model=model,
            data_collator=DataCollatorWithSpeaker(processor=processor, padding=True),
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset_full,
            eval_dataset=test_dataset,
            tokenizer=processor.feature_extractor,
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

        # 修正 4：pth 檔名含 A（Scenario B）
        pth_path = os.path.join(OUTPUT_DIR, f"huang_sls_dann_B_shared_encoder_run_{run_i}.pth")
        torch.save(trainer.model.down_proj.state_dict(), pth_path)
        print(f"🔑 down_proj .pth 儲存至: {pth_path}")

        results = full_evaluation(trainer, test_dataset, OUTPUT_DIR, run_i)
        results["run"] = run_i   # 修正 5a
        all_results.append(results)
        print(f"Run {run_i} → Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | AUC: {results['auc']:.4f}")

        del model, trainer; torch.cuda.empty_cache(); gc.collect()  # 修正 8

    # ── 跨 run 統計 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"📈 Scenario B SLS+DANN — {TOTAL_RUNS} 次實驗彙總")
    print(f"{'='*60}")
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))
        for metric in ["accuracy", "f1", "auc"]:
            vals = results_df[metric].values
            print(f"  {metric.upper():10s}  mean={np.mean(vals):.4f} ± {np.std(vals):.4f}  "
                  f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")
        summary_path = os.path.join(OUTPUT_DIR, "summary_5runs.csv")
        results_df.to_csv(summary_path, index=False)   # 修正 5b
        print(f"\n✅ 彙總已儲存至 {summary_path}")

    print("\n🏁 Scenario B 實驗完成！")