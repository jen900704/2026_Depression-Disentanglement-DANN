"""
File — DANN + Fine-tune (Scenario B) [Linear Probing Family]
============================================================
架構說明：
  - Wav2Vec2 CNN：凍結
  - Wav2Vec2 Transformer：可訓練 (Fine-tune)
  - Pooling：Mean Pooling
  - DANN：Speaker Classifier + GRL
  
設定：
  - Scenario B (Screening)
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
    Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2PreTrainedModel, Wav2Vec2Model,
    Trainer, TrainingArguments, EvalPrediction, set_seed,
)
from transformers.file_utils import ModelOutput
from packaging import version
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc, mean_squared_error

# ================= 參數設定區 (Scenario B) =================
TRAIN_CSV  = "./experiment_sisman_scientific/scenario_B_monitoring/train.csv"
TEST_CSV   = "./experiment_sisman_scientific/scenario_B_monitoring/test.csv"
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"
MODEL_NAME = "facebook/wav2vec2-base"
OUTPUT_DIR = "./output_dann_finetune_B_linear"

SEED             = 103
TOTAL_RUNS       = 5
NUM_EPOCHS       = 10
LEARNING_RATE    = 1e-5
BATCH_SIZE       = 4
GRAD_ACCUM       = 2
EVAL_STEPS       = 10
SAVE_STEPS       = 10
LOGGING_STEPS    = 10
SAVE_TOTAL_LIMIT = 2
FP16             = torch.cuda.is_available()
LABEL_MAP        = {"non": 0, "0": 0, 0: 0, "dep": 1, "1": 1, 1: 1}
LABEL_NAMES      = ["non-depressed", "depressed"]

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    speaker_logits: Optional[torch.FloatTensor] = None

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class Wav2Vec2_DANN_FT(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        
        self.wav2vec2.feature_extractor._freeze_parameters()
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.spk_classifier = nn.Linear(config.hidden_size, getattr(config, "num_speakers", 151))
        self._alpha = 0.0
        self.init_weights()

    def forward(self, input_values, attention_mask=None, labels=None, speaker_labels=None, **kwargs):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask, return_dict=True)
        
        last_hidden_state = outputs.last_hidden_state
        if attention_mask is not None:
            input_lengths = attention_mask.sum(-1).float()
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            sum_hidden = (last_hidden_state * mask_expanded).sum(1)
            pooled_output = sum_hidden / input_lengths.unsqueeze(-1)
        else:
            pooled_output = last_hidden_state.mean(1)

        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        spk_logits = self.spk_classifier(GradientReversalFn.apply(pooled_output, self._alpha))
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        if speaker_labels is not None:
            mask = speaker_labels >= 0
            if mask.sum() > 0:
                spk_loss = nn.CrossEntropyLoss()(spk_logits[mask], speaker_labels[mask])
                loss = loss + self._alpha * spk_loss if loss is not None else spk_loss
                
        return SpeechClassifierOutput(loss=loss, logits=logits, speaker_logits=spk_logits)

@dataclass
class DataCollatorWithSpeaker:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.processor.pad([{"input_values": f["input_values"]} for f in features], padding=self.padding, return_tensors="pt")
        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        batch["speaker_labels"] = torch.tensor([f["speaker_labels"] for f in features], dtype=torch.long)
        return batch

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds), "f1": f1_score(p.label_ids, preds, average="macro")}

class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        total_steps = self.args.max_steps if self.args.max_steps > 0 else (len(self.train_dataset) // (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps) * int(self.args.num_train_epochs))
        p = float(self.state.global_step) / max(total_steps, 1)
        alpha = 2.0 / (1.0 + exp(-10.0 * p)) - 1.0
        model._alpha = alpha
        model.train()
        inputs = self._prepare_inputs(inputs)
        with torch.amp.autocast("cuda"):
            loss = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1: loss = loss / self.args.gradient_accumulation_steps
        self.scaler.scale(loss).backward()
        return loss.detach()

def extract_speaker_id(filepath: str) -> str: return os.path.basename(filepath).split("_")[0]

def load_audio_dataset(csv_path: str, speaker_to_idx: dict = None, is_train: bool = True):
    df = pd.read_csv(csv_path)
    if is_train and speaker_to_idx is None:
        all_speakers = sorted(set(extract_speaker_id(p) for p in df["path"].tolist()))
        speaker_to_idx = {spk: idx for idx, spk in enumerate(all_speakers)}

    records = []
    for _, row in df.iterrows():
        wav_path = os.path.join(AUDIO_ROOT, row["path"])
        if str(row["label"]).strip().lower() not in LABEL_MAP or not os.path.exists(wav_path): continue
        spk_idx = speaker_to_idx.get(extract_speaker_id(wav_path), -1)
        records.append({"path": wav_path, "label": LABEL_MAP[str(row["label"]).strip().lower()], "speaker_labels": spk_idx})
    return HFDataset.from_dict({"path": [r["path"] for r in records], "label": [r["label"] for r in records], "speaker_labels": [r["speaker_labels"] for r in records]}), speaker_to_idx

def speech_file_to_array_fn(batch, processor):
    speech, sr = torchaudio.load(batch["path"])
    if speech.shape[0] > 1: speech = torch.mean(speech, dim=0, keepdim=True)
    speech = speech.squeeze().numpy()
    if sr != 16000:
        import librosa
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
    batch["speech"] = speech
    return batch

def preprocess_function(batch, processor):
    result = processor(batch["speech"], sampling_rate=16000, return_tensors="np", padding=False)
    batch["input_values"] = result.input_values[0]
    batch["labels"] = batch["label"]
    return batch

def full_evaluation(trainer, test_dataset, output_dir, run_i):
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions, axis=1)
    y_true = predictions.label_ids
    results_path = os.path.join(output_dir, f"results_run{run_i}")
    os.makedirs(results_path, exist_ok=True)
    pd.DataFrame(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0, output_dict=True)).transpose().to_csv(os.path.join(results_path, "clsf_report.csv"), sep="\t")
    return {"accuracy": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred, average="macro"), "auc": 0.0}

if __name__ == "__main__":
    set_seed(SEED)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    train_dataset_full, speaker_to_idx = load_audio_dataset(TRAIN_CSV, is_train=True)
    test_dataset_raw, _ = load_audio_dataset(TEST_CSV, speaker_to_idx=speaker_to_idx, is_train=False)
    num_speakers = len(speaker_to_idx)
    
    map_kwargs = {"fn_kwargs": {"processor": processor}}
    train_dataset_full = train_dataset_full.map(speech_file_to_array_fn, **map_kwargs).map(preprocess_function, **map_kwargs)
    test_dataset = test_dataset_raw.map(speech_file_to_array_fn, **map_kwargs).map(preprocess_function, **map_kwargs)
    
    all_results = []
    for run_i in range(1, TOTAL_RUNS + 1):
        set_seed(SEED + run_i - 1)
        config = Wav2Vec2Config.from_pretrained(MODEL_NAME, num_labels=2, num_speakers=num_speakers, final_dropout=0.1)
        model = Wav2Vec2_DANN_FT.from_pretrained(MODEL_NAME, config=config)
        
        training_args = TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR, f"run_{run_i}"), 
            per_device_train_batch_size=BATCH_SIZE, 
            gradient_accumulation_steps=GRAD_ACCUM, 
            num_train_epochs=NUM_EPOCHS, 
            fp16=FP16, 
            save_steps=SAVE_STEPS, 
            eval_steps=EVAL_STEPS, 
            logging_steps=LOGGING_STEPS, 
            learning_rate=LEARNING_RATE, 
            save_total_limit=SAVE_TOTAL_LIMIT, 
            load_best_model_at_end=True, 
            report_to="none", 
            remove_unused_columns=False
        )
        
        trainer = CTCTrainer(
            model=model, 
            data_collator=DataCollatorWithSpeaker(processor=processor), 
            args=training_args, 
            compute_metrics=compute_metrics, 
            train_dataset=train_dataset_full, 
            eval_dataset=test_dataset, 
            tokenizer=processor.feature_extractor
        )
        
        try:
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️ OOM, Skipping Run")
                torch.cuda.empty_cache(); continue
            raise e
            
        trainer.save_model(os.path.join(OUTPUT_DIR, f"run_{run_i}", "best_model"))
        all_results.append(full_evaluation(trainer, test_dataset, OUTPUT_DIR, run_i))
        del model, trainer; torch.cuda.empty_cache(); gc.collect()

    if all_results:
        pd.DataFrame(all_results).to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)