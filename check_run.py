import pandas as pd

# =====================================================================
# 🔍 終極版：12 個 SLURM JOBID 真實身分解密 (不會再搞混！)
# =====================================================================
jobs_data = [
    # ------------------ 【Group 2: Wav2Vec-SLS Models】 ------------------
    {"JOBID": "1527483", "NAME": "H_SLS_A",  "ST": "R",  "Partition": "gpu", 
     "論文表格對應": "Group 2 - Original Model (Frozen)", "說明": "無 DANN、無 Fine-tune (Scenario A)"},
    {"JOBID": "1527482", "NAME": "H_SLS_B",  "ST": "R",  "Partition": "gpu", 
     "論文表格對應": "Group 2 - Original Model (Frozen)", "說明": "無 DANN、無 Fine-tune (Scenario B)"},
    {"JOBID": "1527190", "NAME": "Huang_A",  "ST": "R",  "Partition": "gpu-a100", 
     "論文表格對應": "Group 2 - DANN Model (Frozen)",    "說明": "有 DANN、無 Fine-tune (Scenario A) [已跑 9小時]"},
    {"JOBID": "1527191", "NAME": "Huang_B",  "ST": "PD", "Partition": "gpu-a100", 
     "論文表格對應": "Group 2 - DANN Model (Frozen)",    "說明": "有 DANN、無 Fine-tune (Scenario B) [排隊中]"},
    {"JOBID": "1527195", "NAME": "Huang_FT", "ST": "PD", "Partition": "gpu-a100", 
     "論文表格對應": "Group 2 - Fine-Tuned Model",     "說明": "有 DANN、解凍 backbone 微調 [排隊中]"},

    # ------------------ 【Group 3: XLSR-eGeMAPS Models】 ------------------
    {"JOBID": "1527487", "NAME": "XLSR_A",   "ST": "R",  "Partition": "gpu", 
     "論文表格對應": "Group 3 - DANN Model (Frozen)",    "說明": "有 DANN、無 Fine-tune (Scenario A)"},
    {"JOBID": "1527488", "NAME": "XLSR_B",   "ST": "R",  "Partition": "gpu", 
     "論文表格對應": "Group 3 - DANN Model (Frozen)",    "說明": "有 DANN、無 Fine-tune (Scenario B)"},
    {"JOBID": "1527196", "NAME": "XLSR_FT_", "ST": "PD", "Partition": "gpu-a100", 
     "論文表格對應": "Group 3 - Fine-Tuned Model",     "說明": "有 DANN、解凍 backbone 微調 (A) [排隊中]"},
    {"JOBID": "1527197", "NAME": "XLSR_FT_", "ST": "PD", "Partition": "gpu-a100", 
     "論文表格對應": "Group 3 - Fine-Tuned Model",     "說明": "有 DANN、解凍 backbone 微調 (B) [排隊中]"},

    # ------------------ 【🗑️ 可以放心刪除的重複/卡住任務】 ------------------
    {"JOBID": "1527194", "NAME": "Huang_SL", "ST": "PD", "Partition": "gpu-a100", 
     "論文表格對應": "❌ 應刪除 (重複)", "說明": "這是 H_SLS 舊版，卡在 a100，正版已在 gpu 跑"},
    {"JOBID": "1527490", "NAME": "XLSR_B",   "ST": "PD", "Partition": "gpu-a100", 
     "論文表格對應": "❌ 應刪除 (重複)", "說明": "這是 XLSR_B 重複送出，正版 1527488 已在 gpu 跑"},
    {"JOBID": "1527491", "NAME": "XLSR_A",   "ST": "PD", "Partition": "gpu-a100", 
     "論文表格對應": "❌ 應刪除 (重複)", "說明": "這是 XLSR_A 重複送出，正版 1527487 已在 gpu 跑"},
]

df = pd.DataFrame(jobs_data)
print(df.to_string(index=False))