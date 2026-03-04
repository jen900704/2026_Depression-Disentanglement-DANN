import os, torch, torch.nn as nn, numpy as np, pandas as pd, torchaudio, glob
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2Config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ============================================================
# 自動搜尋權重檔案 (關鍵修正)
# ============================================================
SEARCH_ROOT = "./output_xlsr_egemaps_ft_B"
print(f"🔍 正在 {SEARCH_ROOT} 裡面地毯式搜索權重...")

# 找所有的 .bin 檔
bins = glob.glob(os.path.join(SEARCH_ROOT, "**", "pytorch_model.bin"), recursive=True)
# 找所有的 .safetensors 檔
safes = glob.glob(os.path.join(SEARCH_ROOT, "**", "model.safetensors"), recursive=True)

all_files = bins + safes
if not all_files:
    print(f"❌ 完蛋了，在 {SEARCH_ROOT} 找不到任何模型檔案！請用 ls -R 檢查資料夾。")
    exit()

# 選擇路徑最短的（通常是 best_model），或者數字最大的 checkpoint
# 這裡我們優先選 best_model，沒有的話選最新的 checkpoint
target_ckpt = None
for f in all_files:
    if "best_model" in f:
        target_ckpt = f
        break
if not target_ckpt:
    # 沒 best_model，選 checkpoint 數字最大的
    target_ckpt = sorted(all_files, key=lambda x: len(x))[-1]

print(f"🎯 鎖定目標: {target_ckpt}")
CKPT_DIR = os.path.dirname(target_ckpt) # 取得資料夾路徑

CSV_B_TRAIN = "./experiment_sisman_scientific/scenario_B_monitoring/train.csv"
CSV_B_TEST  = "./experiment_sisman_scientific/scenario_B_monitoring/test.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# G3 模型 (強制讀取，避開 Config 檢查)
# ============================================================
class G3_Saviour_Model(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        # 1. 直接用官方 Config (不管本地有沒有 config.json)
        print("   🧠 載入官方 XLSR Config...")
        config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.wav2vec2 = Wav2Vec2Model(config).to(DEVICE)
        
        # 2. 載入權重
        print(f"   📥 載入權重: {weights_path}")
        if weights_path.endswith(".bin"):
            sd = torch.load(weights_path, map_location=DEVICE)
        else:
            from safetensors.torch import load_file
            sd = load_file(weights_path)
            
        # 載入骨幹
        self.wav2vec2.load_state_dict({k.replace("wav2vec2.", ""): v for k, v in sd.items() if "wav2vec2." in k}, strict=False)
        
        # 3. 初始化 DownProj
        self.down_proj = nn.Sequential(
            nn.Linear(1024 + 88, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        ).to(DEVICE)
        
        # 載入 DownProj
        self.down_proj.load_state_dict({k.replace("down_proj.", ""): v for k, v in sd.items() if "down_proj." in k}, strict=False)

    def get_embedding(self, input_values):
        out = self.wav2vec2(input_values).last_hidden_state.mean(1)
        pad = torch.zeros(out.size(0), 88, device=DEVICE, dtype=out.dtype)
        return self.down_proj(torch.cat([out, pad], dim=-1))

# ============================================================
# 執行 Probe
# ============================================================
def extract(csv_path, proc, model, filter_target=False):
    df = pd.read_csv(csv_path)
    test_df = pd.read_csv(CSV_B_TEST)
    target_spks = set(test_df['path'].apply(lambda x: os.path.basename(x).split('_')[0]))
    
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            spk = os.path.basename(row['path']).split('_')[0]
            if filter_target and spk not in target_spks: continue
            
            try:
                wav, sr = torchaudio.load(row['path'])
                if sr != 16000: wav = torchaudio.transforms.Resample(sr, 16000)(wav)
                inp = proc(wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
                emb = model.get_embedding(inp)
                feats.append(emb.squeeze().cpu().numpy()); labels.append(spk)
            except: continue
    return np.array(feats), np.array(labels)

if __name__ == "__main__":
    print(f"🚀 啟動 G3 B 救援任務")
    
    model = G3_Saviour_Model(target_ckpt)
    proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    
    print("📦 Extracting Train...")
    X_tr, y_tr = extract(CSV_B_TRAIN, proc, model, True)
    print("📦 Extracting Test...")
    X_te, y_te = extract(CSV_B_TEST,  proc, model, False)
    
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"\n✅✅✅ G3 Orig/FT B Final Score: {acc:.4f}")