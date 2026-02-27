import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchaudio
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.file_utils import ModelOutput

# ================= 🔧 1. 設定區 (Scenario A) =================
TRAIN_CSV = "./experiment_sisman_scientific/scenario_A_screening/train.csv"
TEST_CSV = "./experiment_sisman_scientific/scenario_A_screening/test.csv"
AUDIO_ROOT = "/export/fs05/hyeh10/depression/daic_5utt_full/merged_5"

MODEL_NAME = "facebook/wav2vec2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # 🔥 建議調小以防 OOM
EPOCHS = 30
LEARNING_RATE = 1e-3 

# ================= 🧠 2. Huang et al. (2024) 架構定義 (Frozen 強化版) =================

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

class Wav2Vec2ClassificationHead(nn.Module):
    """Huang et al. 定義的分類頭: Linear -> Tanh -> Linear"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class HuangForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = "mean" 
        
        # 1. 載入主幹網路
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # 🔥 修改點 A: 初始化時立即凍結 Wav2Vec2 所有參數
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
        # 2. 載入分類頭 (這部分維持可訓練)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.init_weights()

    def forward(self, input_values, attention_mask=None, labels=None):
        # 🔥 修改點 B: 使用 torch.no_grad() 包裹 Frozen 的主幹運算
        with torch.no_grad():
            outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
            hidden_states = outputs[0]
            # Mean Pooling
            hidden_states = torch.mean(hidden_states, dim=1)
        
        # 分類頭 (Classifier) 必須在 no_grad 之外，因為它需要更新權重
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SpeechClassifierOutput(loss=loss, logits=logits)

# ================= 📂 3. 資料載入 (Scenario A) =================

class ScenarioADataset(Dataset):
    def __init__(self, csv_path, processor):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        # 🔥 修正: 加入標籤對照表
        self.label_map = {
            'non': 0, '0': 0, 0: 0,
            'dep': 1, '1': 1, 1: 1
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = os.path.join(AUDIO_ROOT, row['path'])
        
        # 讀取音訊
        speech, sr = torchaudio.load(wav_path)
        if sr != 16000:
            speech = torchaudio.transforms.Resample(sr, 16000)(speech)
        
        # 🔥 加入長度截斷以防 OOM
        MAX_LEN = 16000 * 10 # 10秒
        if speech.shape[1] > MAX_LEN:
             speech = speech[:, :MAX_LEN]

        input_values = self.processor(speech.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values[0]
        
        # --- 修正後的 Label 處理邏輯 ---
        raw_label = str(row['label']).strip().lower()
        if raw_label in self.label_map:
            label_int = self.label_map[raw_label]
        else:
            label_int = 0
            
        label = torch.tensor(label_int, dtype=torch.long)
        
        return {"input_values": input_values, "labels": label}

def collate_fn(batch):
    input_values = [item['input_values'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Padding 到該 batch 最長長度
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    labels = torch.stack(labels)
    
    return {"input_values": input_values, "labels": labels}

# ================= 🚀 4. 執行實驗 =================

if __name__ == "__main__":
    print(f"🚀 啟動實驗: 複製 Huang et al. 架構在 Scenario A 執行 Linear Probing")
    
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    config = Wav2Vec2Config.from_pretrained(MODEL_NAME, num_labels=2, final_dropout=0.1)
    
    model = HuangForSpeechClassification.from_pretrained(MODEL_NAME, config=config).to(DEVICE)

    # 再次確認凍結狀態
    print("❄️ Wav2Vec2 Backbone 已凍結。只訓練 Classification Head。")

    train_ds = ScenarioADataset(TRAIN_CSV, processor)
    test_ds = ScenarioADataset(TEST_CSV, processor)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            inputs = batch['input_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 評估
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input_values'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                logits = model(inputs).logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f"✅ Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Test Acc: {acc:.4f} | Test F1: {f1:.4f}")
        
    print("\n🏁 實驗結束。")
    print(classification_report(all_labels, all_preds))