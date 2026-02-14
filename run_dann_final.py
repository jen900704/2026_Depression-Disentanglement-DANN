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
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 🔧 1. 設定區 (Config)
# ==========================================
TRAIN_CSV_PATH = "./experiment_sisman_scientific/scenario_B_monitoring/train.csv"
TEST_CSV_PATH = "./experiment_sisman_scientific/scenario_B_monitoring/test.csv"
AUDIO_ROOT = "" 

MODEL_NAME = "facebook/wav2vec2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 30
RUN_NAME = "Final_Defense" # 用於檔名

print(f"🖥️ 使用裝置: {DEVICE}")

# ==========================================
# 🧠 2. 模型定義 (DANN Architecture)
# ==========================================
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

    def forward(self, x, alpha=1.0):
        features = self.shared_encoder(x)
        class_output = self.class_classifier(features)
        reverse_features = self.grl(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        return class_output, domain_output

# ==========================================
# 📂 3. 資料處理工具
# ==========================================
def extract_speaker_id(filepath):
    filename = os.path.basename(filepath)
    speaker_id = filename.split('_')[0] 
    return speaker_id

def prepare_data(csv_path, processor, model, speaker_to_idx=None, is_train=True):
    df = pd.read_csv(csv_path)
    print(f"📂 正在處理 {csv_path} (共 {len(df)} 筆)...")
    
    features_list = []
    labels_list = []
    speaker_indices_list = []
    
    label_map = {'dep': 1, '1': 1, 1: 1, 'non': 0, '0': 0, 0: 0}

    if is_train and speaker_to_idx is None:
        all_speakers = df['path'].apply(extract_speaker_id).unique()
        all_speakers = sorted(all_speakers)
        speaker_to_idx = {spk: idx for idx, spk in enumerate(all_speakers)}
        print(f"🔍 [訓練集] Speaker Map: {list(speaker_to_idx.items())[:5]}...")
    
    model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
            wav_path = os.path.join(AUDIO_ROOT, row['path'])
            try:
                waveform, sample_rate = torchaudio.load(wav_path)
                if sample_rate != 16000:
                    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
                if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                raw_label = str(row['label']).strip().lower()
                if raw_label in label_map:
                    final_label = label_map[raw_label]
                else:
                    continue

                inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu()
                
                features_list.append(embeddings)
                labels_list.append(final_label)
                
                spk_str = extract_speaker_id(wav_path)
                speaker_indices_list.append(speaker_to_idx.get(spk_str, 0))
                
            except Exception as e:
                print(f"⚠️ Error: {wav_path} -> {e}")
                continue

    if len(features_list) == 0:
        raise ValueError("❌ 錯誤：沒有任何資料被成功讀取！")

    X = torch.cat(features_list, dim=0)
    y = torch.tensor(labels_list, dtype=torch.long)
    s = torch.tensor(speaker_indices_list, dtype=torch.long)
    return X, y, s, speaker_to_idx

# ==========================================
# 🚀 4. 主程式執行
# ==========================================
if __name__ == "__main__":
    # --- A. 準備特徵 (只跑一次) ---
    print("🧠 載入 Wav2Vec2 模型...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    w2v_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    
    print("\n📦 準備資料...")
    X_train, y_train, s_train, speaker_map = prepare_data(TRAIN_CSV_PATH, processor, w2v_model, is_train=True)
    X_test, y_test, s_test, _ = prepare_data(TEST_CSV_PATH, processor, w2v_model, speaker_to_idx=speaker_map, is_train=False)
    
    num_speakers = len(speaker_map)
    train_dataset = TensorDataset(X_train, y_train, s_train)
    test_dataset = TensorDataset(X_test, y_test, s_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- B. 訓練 DANN ---
    print(f"\n🏗️ 初始化 DANN 模型 (Class=2, Speakers={num_speakers})...")
    dann_model = DANN_Model(num_speakers=num_speakers).to(DEVICE)
    optimizer = optim.Adam(dann_model.parameters(), lr=0.001)
    
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    
    best_f1 = 0.0
    
    print("\n⚔️ 開始訓練...")
    for epoch in range(EPOCHS):
        dann_model.train()
        total_loss = 0
        
        p = float(epoch) / EPOCHS
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        
        for inputs, labels, speakers in train_loader:
            inputs, labels, speakers = inputs.to(DEVICE), labels.to(DEVICE), speakers.to(DEVICE)
            
            optimizer.zero_grad()
            class_out, domain_out = dann_model(inputs, alpha=alpha)
            loss = criterion_class(class_out, labels) + criterion_domain(domain_out, speakers)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 評估
        dann_model.eval()
        correct_spk = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels, speakers in test_loader:
                inputs, labels, speakers = inputs.to(DEVICE), labels.to(DEVICE), speakers.to(DEVICE)
                c_out, d_out = dann_model(inputs, alpha=0)
                
                _, preds = torch.max(c_out, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                _, spk_preds = torch.max(d_out, 1)
                correct_spk += (spk_preds == speakers).sum().item()
                total += labels.size(0)
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro') # 使用 Macro F1
        spk_acc = correct_spk / total
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(dann_model.state_dict(), "best_dann_model.pth") # 存最好的模型
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.2f} | Dep Acc: {acc:.4f} | F1: {f1:.4f} | Spk Acc: {spk_acc:.4f} 🔥 New Best!")
        else:
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.2f} | Dep Acc: {acc:.4f} | F1: {f1:.4f} | Spk Acc: {spk_acc:.4f}")

    # --- C. 雙重 t-SNE 繪圖 ---
    print("\n🎨 正在載入最佳模型並繪製 t-SNE 圖...")
    dann_model.load_state_dict(torch.load("best_dann_model.pth")) # 載入最佳權重
    dann_model.eval()
    
    feats = []
    spks = []
    lbls = []
    
    with torch.no_grad():
        for inputs, labels, speakers in test_loader:
            inputs = inputs.to(DEVICE)
            f = dann_model.shared_encoder(inputs).cpu().numpy()
            feats.append(f)
            spks.extend(speakers.cpu().numpy())
            lbls.extend(labels.cpu().numpy())
            
    feats = np.vstack(feats)
    spks = np.array(spks)
    lbls = np.array(lbls)
    
    # 執行一次 t-SNE
    print("⏳ 計算 t-SNE 中...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    feats_2d = tsne.fit_transform(feats)
    
    # 圖 1: Speaker 分佈 (證明去識別化)
    plt.figure(figsize=(10, 8))
    # 這裡為了展示效果，只選前 10 個人的資料來畫，不然顏色太多會看不清楚
    # 如果想畫全部，就把 mask 拿掉
    top_speakers = pd.Series(spks).value_counts().index[:10]
    mask = np.isin(spks, top_speakers)
    
    sns.scatterplot(x=feats_2d[mask, 0], y=feats_2d[mask, 1], hue=spks[mask], palette="tab10", legend="full", s=60, alpha=0.7)
    plt.title("DANN Features by Speaker (Should be Mixed)", fontsize=16)
    plt.legend(title="Speaker ID", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("tsne_speaker.png")
    print("✅ 已儲存: tsne_speaker.png (請檢查是否混在一起)")
    
    # 圖 2: Depression 分佈 (證明保留病理特徵)
    plt.figure(figsize=(10, 8))
    # 畫出所有點，按憂鬱症標籤著色
    sns.scatterplot(x=feats_2d[:, 0], y=feats_2d[:, 1], hue=lbls, palette={0: 'blue', 1: 'red'}, style=lbls, s=60, alpha=0.6)
    plt.title("DANN Features by Depression (Should be Separated)", fontsize=16)
    plt.legend(title="Depression", labels=["Non-Depressed", "Depressed"])
    plt.savefig("tsne_depression.png")
    print("✅ 已儲存: tsne_depression.png (請檢查紅藍是否分開)")