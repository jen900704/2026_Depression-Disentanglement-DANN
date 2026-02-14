import os
import argparse
import yaml
import torch
import numpy as np
from datasets import load_from_disk
from transformers import Wav2Vec2Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

def run_linear_probing(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    features_path = os.path.join(config['output_dir'], 'features')
    train_dir = os.path.join(features_path, "train_dataset")
    eval_dir = os.path.join(features_path, "eval_dataset")

    try:
        train_dataset = load_from_disk(train_dir)
        eval_dataset = load_from_disk(eval_dir)
    except Exception as e:
        print(f"❌ 讀取資料集失敗: {e}")
        return

    # 1. 喚醒 Wav2Vec2 模型來當「特徵抽取器」
    print("🧠 正在載入 Wav2Vec2 模型提取深層特徵 (Embeddings)...")
    
    # 自動偵測是否有 GPU 可以加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 運算設備: {device}")
    
    model_name = config.get('processor_name_or_path', 'facebook/wav2vec2-base')
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval() # 鎖定模型，不更新權重

    # 2. 定義抽取函數
    def get_embeddings(dataset):
        embeddings = []
        labels = []
        for item in tqdm(dataset, desc="抽取特徵中"):
            # 取出原始波形，轉成 tensor 並丟到 GPU/CPU
            input_values = torch.tensor(item['input_values']).unsqueeze(0).to(device)
            if input_values.shape[1] == 0: continue
            
            with torch.no_grad(): # 省記憶體大法
                outputs = model(input_values)
                # 取出最後一層的特徵，並對時間軸做平均 (Global Average Pooling) -> 變成 768 維
                hidden_state = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            embeddings.append(hidden_state)
            labels.append(item['labels'])
        return np.array(embeddings), np.array(labels)

    # 3. 開始抽取
    print("\n⏳ 轉換【訓練集】 (這需要一點時間)...")
    X_train, y_train = get_embeddings(train_dataset)
    print("\n⏳ 轉換【測試集】...")
    X_test, y_test = get_embeddings(eval_dataset)

    print(f"\n✅ 成功獲得深層特徵！進入 Linear Probing 模型形狀: X_train={X_train.shape}")
    
    # 4. 真正公平的對決：Logistic Regression
    print("\n🚀 開始執行 Linear Probing...")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n" + "="*40)
    print("🎯 Linear Probing 測試集結果")
    print("="*40)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("混淆矩陣:\n", confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run_linear_probing(args.config)