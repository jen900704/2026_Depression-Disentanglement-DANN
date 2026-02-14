import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# ================= 設定區 =================
source_csv = "train_m5_absolute.csv"  # 替換成這個擁有 6545 筆的完整檔案
output_base = "./experiment_sisman_scientific"
# =========================================

print("🚀 [V6] 開始建構「科學隨機抽樣：嚴格數量匹配版 (Size-Matched)」對照實驗...")

# 1. 讀取與前置處理
if not os.path.exists(source_csv):
    print(f"❌ 找不到來源檔案: {source_csv}"); exit()

df = pd.read_csv(source_csv, sep=None, engine='python')

# ⭐⭐⭐ 照妖鏡 1：看原始檔案到底有多大 ⭐⭐⭐
print(f"🔍 [抓漏] 剛讀取 CSV 時的總筆數: {len(df)}") 

# 確保路徑與ID
path_col = next((c for c in df.columns if c.lower() in ['path', 'file_path', 'filename']), None)
if not path_col: print("❌ 找不到路徑欄位"); exit()
df['participant_id'] = df[path_col].apply(lambda x: os.path.basename(str(x)).split('_')[0])

# 確保 Label
label_col = next((c for c in df.columns if c.lower() in ['class_2', 'label', 'target']), None)
if not label_col: print("❌ 找不到 Label"); exit()

# ⭐⭐⭐ 照妖鏡 2：檢查有多少筆 Label 是空的 ⭐⭐⭐
print(f"🔍 [抓漏] 發現有 {df[label_col].isna().sum()} 筆資料沒有 Label (準備被刪除)!")

df = df.dropna(subset=[label_col])

# ⭐⭐⭐ 照妖鏡 3：刪除後的最終數量 ⭐⭐⭐
print(f"🔍 [抓漏] 刪除沒有 Label 的資料後，剩下筆數: {len(df)}")

# ==========================================
# 🧠 核心邏輯：雙盲隨機抽樣 + 數量匹配
# ==========================================

# 第一步：選出 20% 的人當作「測試對象」(Test Subjects)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_people_idx, test_people_idx = next(gss.split(df, groups=df['participant_id']))

df_background = df.iloc[train_people_idx].copy() # 背景路人 
df_test_subjects = df.iloc[test_people_idx].copy() # 測試對象 

print(f"\n🔒 鎖定測試對象: {df_test_subjects['participant_id'].nunique()} 人")

# 第二步：針對測試對象的切片，進行「隨機打散切分」 
split_leak_history, split_exam_target = train_test_split(
    df_test_subjects, 
    test_size=0.5, 
    random_state=42, 
    stratify=df_test_subjects['participant_id'] 
)

print(f"   📂 歷史資料 (Leak Source): {len(split_leak_history)} 筆切片")
print(f"   📂 當下資料 (Fixed Exam):  {len(split_exam_target)} 筆切片")

# 第三步：⭐ 數量匹配 (Size-Matched Control) 核心機制 ⭐
# 計算外洩歷史資料的切片數量
leak_size = len(split_leak_history)

# 從背景路人中，隨機抽出剛好等於 leak_size 數量的切片當作「替身」
# 剩下的當作「打底基礎」
df_base_train, df_filler_train = train_test_split(
    df_background,
    test_size=leak_size,
    random_state=42
)

print(f"   ⚖️  為了維持公平，從路人中抽出 {len(df_filler_train)} 筆切片作為「替身」")
print(f"   🧱 剩下的 {len(df_base_train)} 筆切片作為 A 和 B 共同的「打底基礎」")

# ==========================================
# 🧪 實驗 A: 初診篩檢 (Scenario: First Visit)
# ==========================================
print("\n[實驗 A] 初診篩檢 (Strict No Leakage, Size-Matched)...")
save_dir_A = os.path.join(output_base, "scenario_A_screening")
os.makedirs(save_dir_A, exist_ok=True)

# 訓練集 = 基礎路人 + 路人替身 (數量完美補齊)
train_A = pd.concat([df_base_train, df_filler_train])
test_A = split_exam_target.copy()

# 存檔
path_c, label_c = path_col, label_col
train_A.rename(columns={path_c: 'path', label_c: 'label'})[['path', 'label']].to_csv(os.path.join(save_dir_A, "train.csv"), index=False)
test_A.rename(columns={path_c: 'path', label_c: 'label'})[['path', 'label']].to_csv(os.path.join(save_dir_A, "test.csv"), index=False)

print(f"   👉 Train: {len(train_A)} 筆 | Test: {len(test_A)} 筆")
print(f"   👉 邏輯: 模型完全沒聽過這 {df_test_subjects['participant_id'].nunique()} 位病人的聲音。")


# ==========================================
# 🧪 實驗 B: 長期監測 (Scenario: Longitudinal Monitoring)
# ==========================================
print("\n[實驗 B] 長期監測 (With Historical Leakage)...")
save_dir_B = os.path.join(output_base, "scenario_B_monitoring")
os.makedirs(save_dir_B, exist_ok=True)

# 訓練集 = 基礎路人 + 測試對象的歷史資料 (Leakage)
train_B = pd.concat([df_base_train, split_leak_history])
test_B = split_exam_target.copy()

# 存檔
train_B.rename(columns={path_c: 'path', label_c: 'label'})[['path', 'label']].to_csv(os.path.join(save_dir_B, "train.csv"), index=False)
test_B.rename(columns={path_c: 'path', label_c: 'label'})[['path', 'label']].to_csv(os.path.join(save_dir_B, "test.csv"), index=False)

print(f"   👉 Train: {len(train_B)} 筆 | Test: {len(test_B)} 筆")
print(f"   👉 邏輯: 模型在訓練時聽過了這 {df_test_subjects['participant_id'].nunique()} 位病人『其他的』錄音檔。")
print(f"   ✨ 科學性: A 與 B 的 Train/Test 數量 100% 完全相等，唯一變因只有歷史資訊！")

print("\n✅ 頂級科學對照數據生成完畢！")
print(f"資料夾: {output_base}")