import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Function
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# ================= 設定區 =================
TRAIN_CSV_PATH = "./experiment_sisman_scientific/scenario_B_monitoring/train.csv"
TEST_CSV_PATH = "./experiment_sisman_scientific/scenario_B_monitoring/test.csv"
AUDIO_ROOT = "" 

MODEL_NAME = "facebook/wav2vec2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 20  # Baseline 跑 20 Epoch 應該就夠了

print(f"🖥️ 使用裝置: {DEVICE}")

# ================= 模型定義 (同 DANN，但我們等下會關掉 GRL) =================
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
    def forward(self, x, alpha=1.0):
        return GradientReversalFn.apply(x, alpha)

class DANN_Model(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_classes=2, num_speakers=38):
        super(DANN_Model, self).__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_speakers)
        )
        self.grl = GradientReversalLayer()

    def forward(self, x, alpha=0.0): # 預設 alpha=0 (無對抗)
        features = self.shared_encoder(x)
        class_output = self.class_classifier(features)
        reverse_features = self.grl(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        return class_output, domain_output

# ================= 資料處理 (含 Augmentation) =================
def extract_speaker_id(filepath):
    filename = os.path.basename(filepath)
    speaker_id = filename.split('_')[0]
    return speaker_id

def prepare_data(csv_path, processor, model, speaker_to_idx=None, is_train=True):
    df = pd.read_csv(csv_path)
    print(f"📂 正在處理 {csv_path} (共 {len(df)} 筆)...")
    if is_train: print("🔥 注意：正在對訓練資料應用 Random Pitch Shift (隨機變調)...")
    
    features_list = []
    labels_list = []
    speaker_indices_list = []
    
    label_map = {'dep': 1, '1': 1, 1: 1, 'non': 0, '0': 0, 0: 0}

    if is_train and speaker_to_idx is None:
        all_speakers = df['path'].apply(extract_speaker_id).unique()
        all_speakers = sorted(all_speakers)
        speaker_to_idx = {spk: idx for idx, spk in enumerate(all_speakers)}
    
    model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            wav_path = os.path.join(AUDIO_ROOT, row['path'])
            try:
                waveform, sample_rate = torchaudio.load(wav_path)
                if sample_rate != 16000:
                    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
                if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # 🔥🔥🔥【關鍵修改：Data Augmentation】🔥🔥🔥
                # 只有訓練集要做變調，測試集要保持原樣（這樣才公平）
                if is_train:
                    # 隨機決定變調多少 (-3 到 +3 半音)
                    n_steps = torch.randint(low=-3, high=4, size=(1,)).item()
                    if n_steps != 0:
                        effects = [['pitch', str(n_steps * 100)], ['rate', '16000']]
                        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, 16000, effects)

                inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu()
                
                # Label 處理
                raw_label = str(row['label']).strip().lower()
                if raw_label in label_map:
                    final_label = label_map[raw_label]
                else:
                    continue
                
                features_list.append(embeddings)
                labels_list.append(final_label)
                spk_str = extract_speaker_id(wav_path)
                speaker_indices_list.append(speaker_to_idx.get(spk_str, 0))
                
            except Exception as e:
                # print(f"Error: {e}") # 忽略雜訊
                continue

    X = torch.cat(features_list, dim=0)
    y = torch.tensor(labels_list, dtype=torch.long)
    s = torch.tensor(speaker_indices_list, dtype=torch.long)
    return X, y, s, speaker_to_idx

# ================= 主程式 =================
if __name__ == "__main__":
    print("🧠 載入 Wav2Vec2...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    w2v_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    
    # 1. 準備資料
    # Train: 會做 Random Pitch Shift
    X_train, y_train, s_train, speaker_map = prepare_data(TRAIN_CSV_PATH, processor, w2v_model, is_train=True)
    # Test: 不做 Augmentation (公平比較)
    X_test, y_test, s_test, _ = prepare_data(TEST_CSV_PATH, processor, w2v_model, speaker_to_idx=speaker_map, is_train=False)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train, s_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test, s_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. 初始化模型
    num_speakers = len(speaker_map)
    print(f"\n🏗️ 初始化 Augmentation Baseline 模型 (Class=2, Speakers={num_speakers})...")
    model = DANN_Model(num_speakers=num_speakers).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    
    print("\n⚔️ 開始 Augmentation Baseline 訓練...")
    print("⚠️ 注意：這裡 alpha=0，所以身分分類器 (Speaker Head) 只負責監測，不影響 Encoder！")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for inputs, labels, speakers in train_loader:
            inputs, labels, speakers = inputs.to(DEVICE), labels.to(DEVICE), speakers.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 🔥 關鍵：alpha=0，切斷 GRL 梯度
            class_out, domain_out = model(inputs, alpha=0.0)
            
            # Loss 只算 Class Loss (因為我們不是在做 DANN，只是在做普通訓練)
            # 但我們還是算 domain_loss 來讓 Speaker Head 學習 (作為 Probe)
            loss_class = criterion_class(class_out, labels)
            loss_domain = criterion_domain(domain_out, speakers) 
            
            # Backprop: 這裡我們「只對 Class Loss」做優化，讓 Speaker Head 自己玩
            # 這樣 Speaker Head 就會變成一個「誠實的評分員」，告訴我們還剩多少身分資訊
            loss = loss_class + loss_domain 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # 評估
        model.eval()
        correct_spk = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels, speakers in test_loader:
                inputs, labels, speakers = inputs.to(DEVICE), labels.to(DEVICE), speakers.to(DEVICE)
                c_out, d_out = model(inputs, alpha=0.0)
                
                _, preds = torch.max(c_out, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                _, spk_preds = torch.max(d_out, 1)
                correct_spk += (spk_preds == speakers).sum().item()
                total += labels.size(0)
                
        dep_acc = accuracy_score(all_labels, all_preds)
        spk_acc = correct_spk / total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.2f}")
        print(f"   👉 [Test] Dep Acc: {dep_acc:.4f} (預期會掉)")
        print(f"   👉 [Test] Spk Acc: {spk_acc:.4f} (預期還很高，例如 0.4~0.6)")
        print("-" * 50)