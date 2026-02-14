import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 📊 1. 寫入我們剛剛跑出來的真實數據
# ==========================================

cm_A = np.array([[125, 196], 
                 [123, 266]])
acc_A = 0.55
f1_A = 0.63

cm_B = np.array([[247, 74], 
                 [78, 311]])
acc_B = 0.79
f1_B = 0.80

labels = ['Non-Depressed (0)', 'Depressed (1)']

# ==========================================
# 🎨 2. 繪製圖表一：混淆矩陣對比 (Heatmap 統一藍色)
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

vmax = max(cm_A.max(), cm_B.max())

# 繪製 A
sns.heatmap(cm_A, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            vmax=vmax, annot_kws={"size": 16}, cbar=False)
axes[0].set_title('Scenario A: Strict Screening\n(True Baseline: 55% Acc)', fontsize=16, fontweight='bold', pad=15)
axes[0].set_xticklabels(labels, fontsize=12)
axes[0].set_yticklabels(labels, fontsize=12, rotation=0)
axes[0].set_xlabel('Predicted Label', fontsize=14)
axes[0].set_ylabel('True Label', fontsize=14)

# 繪製 B
sns.heatmap(cm_B, annot=True, fmt='d', cmap='Blues', ax=axes[1], 
            vmax=vmax, annot_kws={"size": 16})
axes[1].set_title('Scenario B: Longitudinal Monitoring\n(With Speaker Leakage: 79% Acc)', fontsize=16, fontweight='bold', pad=15)
axes[1].set_xticklabels(labels, fontsize=12)
axes[1].set_yticklabels(labels, fontsize=12, rotation=0)
axes[1].set_xlabel('Predicted Label', fontsize=14)
axes[1].set_ylabel('True Label', fontsize=14)

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 📈 3. 繪製圖表二：指標暴增長條圖 (極簡乾淨版)
# ==========================================
fig, ax = plt.subplots(figsize=(9, 6))

metrics = ['Overall Accuracy', 'Depression (Class 1) F1-Score']
A_scores = [acc_A * 100, f1_A * 100]
B_scores = [acc_B * 100, f1_B * 100]

x = np.arange(len(metrics))  
width = 0.35                 

rects1 = ax.bar(x - width/2, A_scores, width, label='Scenario A (No Leakage)', color='#4c72b0', edgecolor='black')
rects2 = ax.bar(x + width/2, B_scores, width, label='Scenario B (With Leakage)', color='#dd8452', edgecolor='black')

ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
ax.set_title('Impact of Speaker Identity Leakage on Performance', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=14, fontweight='bold')

# 因為沒有多餘文字了，把 Y 軸高度縮減回 100 讓柱子更飽滿
ax.set_ylim(0, 100) 
ax.legend(fontsize=12, loc='upper left')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# 箭頭與紅字已徹底刪除

plt.tight_layout()
plt.savefig('performance_jump_barchart.png', dpi=300, bbox_inches='tight')
print("✅ 成功儲存圖表 (極簡乾淨版)！")
plt.close()