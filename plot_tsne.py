import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ================= 1. 核心配置 =================
TEST_CSV_PATH = "./experiment_sisman_scientific/scenario_B_monitoring/test.csv"
MODEL_WEIGHTS_PATH = "best_dann_model.pth" 
MODEL_NAME = "facebook/wav2vec2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DANN_Model(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_classes=2, num_speakers=189):
        super(DANN_Model, self).__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.class_classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, num_classes))
        self.domain_classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, num_speakers))

    def forward(self, x):
        features = self.shared_encoder(x)
        return self.class_classifier(features), self.domain_classifier(features)

def run_pro_visual():
    # 載入資料並修正標籤
    df = pd.read_csv(TEST_CSV_PATH)
    mapping = {0: "Healthy", 1: "Depressed", '0': "Healthy", '1': "Depressed", 'non': 'Healthy', 'dep': 'Depressed'}
    df['label_name'] = df['label'].map(mapping)
    
    print(f"📊 數據分佈: \n{df['label_name'].value_counts()}")

    # 載入模型
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    w2v = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    model = DANN_Model(num_speakers=189).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    w2v.eval(); model.eval()

    base_feats, dann_feats, speakers = [], [], []
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            waveform, _ = torchaudio.load(row['path'])
            inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True).to(DEVICE)
            w2v_out = w2v(**inputs).last_hidden_state.mean(dim=1)
            base_feats.append(w2v_out.cpu().numpy())
            dann_feats.append(model.shared_encoder(w2v_out).cpu().numpy())
            speakers.append(os.path.basename(row['path']).split('_')[0])

    # --- 高階 t-SNE 參數調整 ---
    # 增加 Perplexity 與 Iteration 讓分佈更均勻
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=3000, init='pca', learning_rate='auto')
    b2d = tsne.fit_transform(np.vstack(base_feats))
    d2d = tsne.fit_transform(np.vstack(dann_feats))

    # --- 專業繪圖設定 ---
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    palette_dep = {"Depressed": "#FF3B30", "Healthy": "#007AFF"} # 蘋果標準對比色
    top_10 = pd.Series(speakers).value_counts().index[:10]
    m = np.isin(speakers, top_10)

    # A: Baseline Speaker (使用更鮮豔的色板)
    sns.scatterplot(x=b2d[m,0], y=b2d[m,1], hue=np.array(speakers)[m], ax=axes[0,0], palette="bright", s=120, alpha=0.8, edgecolor='w')
    axes[0,0].set_title("(a) Baseline: Clear Biometric Clusters", fontsize=18, fontweight='bold')

    # B: DANN Speaker (應顯示徹底混合)
    sns.scatterplot(x=d2d[m,0], y=d2d[m,1], hue=np.array(speakers)[m], ax=axes[0,1], palette="bright", s=120, alpha=0.8, edgecolor='w')
    axes[0,1].set_title("(b) DANN: Anonymized Representations", fontsize=18, fontweight='bold')

    # C: Baseline Depression (混合狀態)
    sns.scatterplot(x=b2d[:,0], y=b2d[:,1], hue=df['label_name'], ax=axes[1,0], palette=palette_dep, 
                    style=df['label_name'], markers={"Depressed": "X", "Healthy": "o"}, s=150, alpha=0.4)
    axes[1,0].set_title("(c) Baseline: Entangled Pathological Features", fontsize=18, fontweight='bold')

    # D: DANN Depression (期望看到紅藍兩軍對壘)
    sns.scatterplot(x=d2d[:,0], y=d2d[:,1], hue=df['label_name'], ax=axes[1,1], palette=palette_dep, 
                    style=df['label_name'], markers={"Depressed": "X", "Healthy": "o"}, s=150, alpha=0.5)
    axes[1,1].set_title("(d) DANN: Disentangled Disease Cues", fontsize=18, fontweight='bold')

    for ax in axes.flat:
        ax.set_xlabel("t-SNE Dimension 1", fontsize=14)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=14)
        leg = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=12)
        leg.get_frame().set_linewidth(0.0)

    plt.suptitle("Feature Disentanglement Analysis via Domain-Adversarial Training", fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig("tsne_publication_quality.png", dpi=300)
    print("✅ 專業級圖片已存為 tsne_publication_quality.png")

if __name__ == "__main__":
    run_pro_visual()