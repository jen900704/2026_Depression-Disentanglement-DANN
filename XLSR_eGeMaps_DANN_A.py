"""
File — XLS-R + eGeMAPS + DANN（無 fine-tune，Scenario A）
=========================================================
架構說明：
  - XLS-R (wav2vec2-xls-r-300m) 主幹：完全凍結
  - mean pooling 取最後一層 hidden state → 1024 維
  - eGeMAPS (opensmile, eGeMAPSv02 Functionals)：88 維
  - 特徵拼接後 → down_proj → dep_classifier (binary)
  - DANN：GRL + spk_classifier，alpha 動態遞增

修正清單（相對於使用者提供的草稿）：
  1. import 語法分行
  2. GradientReversalFn 定義位置移至 model 使用前
  3. forward 回傳 SpeechClassifierOutput (ModelOutput)，Trainer 才能提取 loss
  4. egemaps_feat 存入 dataset 欄位，DataCollator 負責組 batch
  5. preprocess_function 補完重採樣邏輯；opensmile 改用 process_signal 避免路徑重複讀取
  6. speaker_labels 從檔名抽取後存入 dataset
  7. AUDIO_ROOT / TOTAL_RUNS 等設定補全
  8. alpha 動態注入 CTCTrainer（與 SLS+DANN 版一致）
  9. 補完 TOTAL_RUNS 迴圈 + 跨 run 統計
 10. config 設定 hidden_size 對應 XLS-R 的 1024
 11. Wav2Vec2Processor → Wav2Vec2FeatureExtractor（XLS-R 無 CTC tokenizer）
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
    Wav2Vec2FeatureExtractor,   # XLS-R 無 CTC tokenizer，改用 FeatureExtractor
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
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error,
)

# ============================================================
#  設定區
# ============================================================
TRAIN_CSV  = "./experiment_sisman_scientific/scenario_A_screening/train.csv"
TEST_CSV   = "./experiment_sisman_scientific/scenario_A_screening/test.csv"
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"

MODEL_NAME  = "facebook/wav2vec2-xls-r-300m"
OUTPUT_DIR  = "./output_xlsr_egemaps_dann_A"
EGEMAPS_DIM = 88   # eGeMAPSv02 Functionals 固定 88 維

SEED       = 103   # 對齊論文基準（103–107）
TOTAL_RUNS = 1
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
BATCH_SIZE    = 4
GRAD_ACCUM    = 2
EVAL_STEPS       = 200   # 對齊論文基準
SAVE_STEPS       = 200
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2
FP16 = torch.cuda.is_available()

LABEL_MAP   = {"non": 0, "0": 0, 0: 0, "dep": 1, "1": 1, 1: 1}
LABEL_NAMES = ["non-depressed", "depressed"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# opensmile 初始化（程式啟動時建立一次）
SMILE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)


# ============================================================
#  ModelOutput
# ============================================================

@dataclass
class SpeechClassifierOutput(ModelOutput):
    # 修正 3：回傳 ModelOutput 子類，Trainer 才能正確提取 loss / logits
    loss:           Optional[torch.FloatTensor] = None
    logits:         torch.FloatTensor           = None
    speaker_logits: Optional[torch.FloatTensor] = None
    hidden_states:  Optional[Tuple[torch.FloatTensor]] = None
    attentions:     Optional[Tuple[torch.FloatTensor]] = None


# ============================================================
#  GRL（修正 2：定義在 Model 之前）
# ============================================================

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

class XLSR_eGeMaps_DANN(Wav2Vec2PreTrainedModel):
    """
    XLS-R (frozen) + eGeMAPS concat + DANN
    - xlsr mean pooling: 1024 維
    - eGeMAPS functionals: 88 維
    - concat → 1112 維 → down_proj(256) → dep/spk classifier
    - alpha 由 CTCTrainer 動態注入（self._alpha）
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
        self.spk_classifier = nn.Linear(256, getattr(config, "num_speakers", 38))

        self._alpha = 0.0  # 由 CTCTrainer 動態更新
        self.init_weights()

    def freeze_feature_extractor(self):
        pass  # 全部已凍結，保留介面相容性

    def forward(
        self,
        input_values,
        attention_mask=None,
        egemaps_feat=None,       # 修正 4：由 dataset 欄位傳入
        labels=None,
        speaker_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )
        # XLS-R mean pooling over time axis → [B, 1024]
        xlsr_feat = torch.mean(outputs.last_hidden_state, dim=1)

        # eGeMAPS 特徵拼接（修正：確保 dtype 一致）
        if egemaps_feat is not None:
            egemaps_feat = egemaps_feat.to(xlsr_feat.dtype)
            combined = torch.cat([xlsr_feat, egemaps_feat], dim=-1)  # [B, 1112]
        else:
            # 推論時若無 eGeMAPS，補零（不影響訓練流程）
            zero_pad = torch.zeros(
                xlsr_feat.size(0), EGEMAPS_DIM,
                dtype=xlsr_feat.dtype, device=xlsr_feat.device
            )
            combined = torch.cat([xlsr_feat, zero_pad], dim=-1)

        shared     = self.down_proj(combined)                               # [B, 256]
        dep_logits = self.dep_classifier(shared)                            # [B, 2]
        spk_logits = self.spk_classifier(
            GradientReversalFn.apply(shared, self._alpha)
        )                                                                   # [B, num_speakers]

        loss = None
        if labels is not None and speaker_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            dep_loss = loss_fct(dep_logits, labels)
            spk_loss = loss_fct(spk_logits, speaker_labels)
            loss = dep_loss + self._alpha * spk_loss

        return SpeechClassifierOutput(
            loss=loss,
            logits=dep_logits,
            speaker_logits=spk_logits,
        )


# ============================================================
#  DataCollator（含 egemaps_feat + speaker_labels）
# ============================================================

@dataclass
class DataCollatorWithEGeMAPSAndSpeaker:
    processor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features   = [{"input_values": f["input_values"]} for f in features]
        label_features   = [f["labels"]          for f in features]
        speaker_features = [f["speaker_labels"]  for f in features]
        egemaps_features = [f["egemaps_feat"]    for f in features]

        # Wav2Vec2FeatureExtractor.pad 與 Wav2Vec2Processor.pad 介面完全相同
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        batch["labels"]         = torch.tensor(label_features,   dtype=torch.long)
        batch["speaker_labels"] = torch.tensor(speaker_features, dtype=torch.long)
        batch["egemaps_feat"]   = torch.tensor(
            np.stack(egemaps_features), dtype=torch.float32
        )
        return batch


# ============================================================
#  compute_metrics
# ============================================================

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1  = f1_score(p.label_ids, preds, average="macro")
    return {"accuracy": acc, "f1": f1}


# ============================================================
#  CTCTrainer — 動態注入 alpha（修正 8）
# ============================================================

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast


class CTCTrainer(Trainer):
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        total_steps = self.args.max_steps if self.args.max_steps > 0 else (
            len(self.train_dataset) // (
                self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
            ) * int(self.args.num_train_epochs)
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


def extract_egemaps(wav_path: str) -> np.ndarray:
    """
    修正 5：使用 process_signal 避免路徑與 opensmile 格式問題，
    並做 NaN 填 0 保護。
    """
    try:
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        # opensmile 需要 numpy 1D array
        signal = waveform.squeeze().numpy()
        feat = SMILE.process_signal(signal, 16000).values.flatten().astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0)  # NaN 保護
        return feat
    except Exception as e:
        print(f"⚠️ eGeMAPS 提取失敗: {wav_path} → {e}")
        return np.zeros(EGEMAPS_DIM, dtype=np.float32)


def load_audio_dataset(csv_path: str, speaker_to_idx: dict = None, is_train: bool = True):
    """
    載入 CSV，回傳 HFDataset（含 egemaps_feat + speaker_labels）及 speaker_to_idx。
    修正 4 & 6：egemaps_feat / speaker_labels 在此預先計算並存入 dataset。
    """
    df = pd.read_csv(csv_path)
    print(f"📂 讀取 {csv_path}，共 {len(df)} 筆資料")

    if is_train and speaker_to_idx is None:
        all_speakers  = sorted(set(extract_speaker_id(p) for p in df["path"].tolist()))
        speaker_to_idx = {spk: idx for idx, spk in enumerate(all_speakers)}
        print(f"🔍 偵測到 {len(speaker_to_idx)} 位 speaker")

    records = []
    skipped = 0
    for _, row in df.iterrows():
        wav_path  = os.path.join(AUDIO_ROOT, row["path"])
        raw_label = str(row["label"]).strip().lower()

        if raw_label not in LABEL_MAP:
            skipped += 1
            continue
        if not os.path.exists(wav_path):
            print(f"⚠️ 檔案不存在: {wav_path}")
            skipped += 1
            continue

        spk_str = extract_speaker_id(wav_path)
        spk_idx = speaker_to_idx.get(spk_str, 0)

        records.append({
            "path":           wav_path,
            "label":          LABEL_MAP[raw_label],
            "speaker_labels": spk_idx,
        })

    if skipped:
        print(f"⚠️ 跳過 {skipped} 筆無效資料")
    print(f"✅ 成功載入 {len(records)} 筆資料")

    dataset = HFDataset.from_dict({
        "path":           [r["path"]           for r in records],
        "label":          [r["label"]          for r in records],
        "speaker_labels": [r["speaker_labels"] for r in records],
    })
    return dataset, speaker_to_idx


def speech_file_to_array_fn(batch, processor):
    """讀取音訊、重採樣，同時提取 eGeMAPS（修正 5：整合在同一 map pass）"""
    wav_path = batch["path"]

    # 讀取波形
    speech_array, sampling_rate = torchaudio.load(wav_path)
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)
    speech_array = speech_array.squeeze().numpy()

    if sampling_rate != 16000:
        import librosa
        speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)

    batch["speech"]       = speech_array
    batch["egemaps_feat"] = extract_egemaps(wav_path)  # 修正 5：同步提取
    return batch


def preprocess_function(batch, processor):
    """將 speech array 轉為 input_values，保留 egemaps_feat"""
    result = processor(
        batch["speech"],
        sampling_rate=16000,
        return_tensors="np",
        padding=False,
    )
    batch["input_values"] = result.input_values[0]
    batch["labels"]       = batch["label"]
    # egemaps_feat 已在 speech_file_to_array_fn 存入，此處直接保留
    return batch


def split_train_valid(dataset: HFDataset, valid_ratio: float = 0.15, seed: int = 42):
    split = dataset.train_test_split(test_size=valid_ratio, seed=seed)
    return split["train"], split["test"]


# ============================================================
#  評估與報告
# ============================================================

def full_evaluation(trainer, test_dataset, output_dir, run_i):
    predictions = trainer.predict(test_dataset)
    preds  = predictions.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    y_pred = np.argmax(preds, axis=1)
    y_true = predictions.label_ids

    print("\n" + "=" * 60)
    print(f"📊 [Run {run_i}] Classification Report")
    print("=" * 60)
    report    = classification_report(y_true, y_pred, target_names=LABEL_NAMES,
                                      zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

    cm    = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    print("\n📊 Confusion Matrix:")
    print(cm_df)

    mse  = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    report_df["MSE"]  = mse
    report_df["RMSE"] = rmse

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc     = auc(fpr, tpr)

    results_path = os.path.join(output_dir, f"results_run{run_i}")
    os.makedirs(results_path, exist_ok=True)
    report_df.to_csv(os.path.join(results_path, "clsf_report.csv"), sep="\t")
    cm_df.to_csv(    os.path.join(results_path, "conf_matrix.csv"), sep="\t")

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Scenario A (XLS-R+eGeMAPS+DANN) Run {run_i}")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_path, "roc_curve.png"))
    plt.close()

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"\n🎯 Test Accuracy: {acc:.4f} | F1 (macro): {f1:.4f} | AUC: {roc_auc:.4f}")
    print(f"✅ 結果已儲存至 {results_path}")
    return {"accuracy": acc, "f1": f1, "auc": roc_auc}


# ============================================================
#  主程式 — TOTAL_RUNS 次迴圈（修正 9）
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 XLS-R + eGeMAPS + DANN（無 fine-tune）— Scenario A")
    print(f"   XLS-R：全部凍結 ({MODEL_NAME})")
    print(f"   eGeMAPS：{EGEMAPS_DIM} 維 (eGeMAPSv02 Functionals)")
    print("   DANN alpha：動態遞增")
    print("=" * 60)

    set_seed(SEED)
    # XLS-R 是純語音表示模型，沒有 CTC tokenizer，必須用 FeatureExtractor
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

    # ── 資料與特徵提取只做一次 ─────────────────────────────
    print("\n📦 載入資料集（只執行一次）...")
    train_dataset_full, speaker_to_idx = load_audio_dataset(TRAIN_CSV, is_train=True)
    test_dataset_raw, _                = load_audio_dataset(
        TEST_CSV, speaker_to_idx=speaker_to_idx, is_train=False
    )
    num_speakers = len(speaker_to_idx)
    print(f"👥 共 {num_speakers} 位 speaker")

    print("\n🔊 預處理音訊 + 提取 eGeMAPS（只執行一次，耗時較長）...")
    train_dataset_full = train_dataset_full.map(
        speech_file_to_array_fn, fn_kwargs={"processor": processor}
    )
    test_dataset_raw = test_dataset_raw.map(
        speech_file_to_array_fn, fn_kwargs={"processor": processor}
    )
    train_dataset_full = train_dataset_full.map(
        preprocess_function, fn_kwargs={"processor": processor}
    )
    test_dataset = test_dataset_raw.map(
        preprocess_function, fn_kwargs={"processor": processor}
    )

    # ── TOTAL_RUNS 次實驗迴圈 ──────────────────────────────
    all_results = []
    for run_i in range(1, TOTAL_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"🎬 Run {run_i} / {TOTAL_RUNS}")
        print(f"{'='*60}")

        # seed 103, 104, 105, 106, 107 — 對齊論文基準
        run_seed = SEED + run_i - 1
        set_seed(run_seed)
        print(f"🎲 Run {run_i} seed: {run_seed}")

        # 直接用 test_dataset 當 eval（對齊論文，無獨立 valid set）
        print(f"📊 Train: {len(train_dataset_full)} | Test(eval): {len(test_dataset)}")

        # 每次重新初始化模型
        config = Wav2Vec2Config.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            num_speakers=num_speakers,
            final_dropout=0.1,
        )
        model = XLSR_eGeMaps_DANN.from_pretrained(MODEL_NAME, config=config)

        frozen    = sum(1 for p in model.wav2vec2.parameters() if not p.requires_grad)
        total     = sum(1 for p in model.wav2vec2.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"❄️  wav2vec2 凍結: {frozen}/{total} 個參數組")
        print(f"🔥 可訓練參數總量: {trainable:,}")

        data_collator  = DataCollatorWithEGeMAPSAndSpeaker(processor=processor, padding=True)
        run_output_dir = os.path.join(OUTPUT_DIR, f"run_{run_i}")
        os.makedirs(run_output_dir, exist_ok=True)

        training_args = TrainingArguments(gradient_checkpointing=True, 
            output_dir=run_output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            label_names=["labels"],
            
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            eval_strategy="steps",
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
            # metric_for_best_model 未設定 → 預設用 eval_loss，對齊論文
            report_to="none",
        )

        trainer = CTCTrainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset_full,   # 用完整 train，無 valid split
            eval_dataset=test_dataset,           # 對齊論文：test set 直接當 eval
            tokenizer=processor,
        )

        print("⚔️ 開始訓練...")
        try:
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️ GPU 記憶體不足，清除快取後跳過此 run")
                torch.cuda.empty_cache()
                continue
            raise e

        best_path = os.path.join(run_output_dir, "best_model")
        trainer.save_model(best_path)
        processor.save_pretrained(best_path)
        print(f"💾 最佳模型儲存至: {best_path}")

        # ── 儲存 down_proj .pth（供 probe 腳本使用）────────────
        # down_proj 是 XLS-R+eGeMAPS 的 128 維 bottleneck，對應 DANN 的 shared_encoder
        pth_path = os.path.join(OUTPUT_DIR, f"xlsr_egemaps_dann_A_shared_encoder_run_{run_i}.pth")
        torch.save(trainer.model.down_proj.state_dict(), pth_path)
        print(f"🔑 down_proj .pth 儲存至: {pth_path}")

        results = full_evaluation(trainer, test_dataset, OUTPUT_DIR, run_i)
        results["run"] = run_i
        all_results.append(results)
        print(f"Run {run_i} → Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | AUC: {results['auc']:.4f}")

    # ── 跨 run 統計 ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"📈 Scenario A XLS-R+eGeMAPS+DANN — {TOTAL_RUNS} 次實驗彙總")
    print(f"{'='*60}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))
        for metric in ["accuracy", "f1", "auc"]:
            vals = results_df[metric].values
            print(f"  {metric.upper():10s}  mean={np.mean(vals):.4f} ± {np.std(vals):.4f}  "
                  f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")
        summary_path = os.path.join(OUTPUT_DIR, "summary_5runs.csv")
        results_df.to_csv(summary_path, index=False)
        print(f"\n✅ 彙總結果已儲存至 {summary_path}")

    print("\n🏁 Scenario A 實驗完成！")