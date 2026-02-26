"""
對齊檢查腳本 v2 — 只檢查 16 個訓練腳本
用法：python check_alignment_v2.py /path/to/training_scripts/
"""
import os, sys, ast, re

# ============================================================
# 只掃這 16 個檔案（精確比對）
# ============================================================
TARGET_FILES = {
    # Huang A/B (v2)
    "replicate_huang_scenario_a_v2.py",
    "replicate_huang_scenario_b_v2.py",
    # DANN static A/B
    "run_dann_a_v2.py",
    "run_dann_b_v2.py",
    # DANN finetune A/B
    "run_dann_finetune_a.py",
    "run_dann_finetune_b.py",
    # SLS no-finetune A/B
    "huang_sls_a.py",
    "huang_sls_b.py",
    # SLS finetune A/B  (有些叫 huang_sls_ft_A / huang_sls_dann_finetune_A)
    "huang_sls_ft_a.py",
    "huang_sls_ft_b.py",
    "huang_sls_dann_finetune_a.py",
    "huang_sls_dann_finetune_b.py",
    # XLSR no-finetune A/B
    "xlsr_egemaps_a.py",
    "xlsr_egemaps_b.py",
    # XLSR finetune A/B
    "xlsr_egemaps_dann_finetune_a.py",
    "xlsr_egemaps_dann_finetune_b.py",
    # SLS+DANN no-finetune（你已通過的版本，保留做 reference）
    "huang_sls_dann_a.py",
    "huang_sls_dann_b.py",
}


def get_training_args_block(src):
    """擷取 TrainingArguments(...) 的內容。"""
    start = src.find("TrainingArguments(")
    if start == -1:
        return ""
    depth, end = 0, start
    for i, ch in enumerate(src[start:], start):
        if ch == "(": depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i; break
    return src[start:end+1]


def check_file(filepath):
    with open(filepath) as f:
        src = f.read()
    fname  = os.path.basename(filepath).lower()
    issues, ok_list = [], []

    def chk(label, cond, fix=""):
        (ok_list if cond else issues).append(
            ("✅ " if cond else "❌ ") + label + (f"  →  {fix}" if not cond and fix else "")
        )

    args_block = get_training_args_block(src)

    # 1. SEED = 103
    m = re.search(r'^SEED\s*=\s*(\d+)', src, re.MULTILINE)
    chk("SEED=103", m and m.group(1) == "103", "SEED = 103")

    # 2. EVAL/SAVE/LOGGING_STEPS = 10
    for step in ["EVAL_STEPS", "SAVE_STEPS", "LOGGING_STEPS"]:
        m = re.search(rf'^{step}\s*=\s*(\d+)', src, re.MULTILINE)
        chk(f"{step}=10", m and m.group(1) == "10", f"{step} = 10")

    # 3. run seed = SEED + run_i - 1
    chk("run seed=103-107",
        "SEED + run_i - 1" in src,
        "run_seed = SEED + run_i - 1")

    # 4. eval_dataset = test_dataset（非 split valid）
    chk("eval_dataset=test_dataset",
        "eval_dataset=test_dataset" in src,
        "eval_dataset=test_dataset （不切 valid split）")

    # 5. 無 metric_for_best_model（TrainingArguments 裡）
    metric_line = re.search(r'metric_for_best_model\s*=', args_block)
    is_comment  = metric_line and args_block[max(0, metric_line.start()-2):metric_line.start()].strip().endswith("#")
    chk("無 metric_for_best_model",
        metric_line is None or is_comment,
        "移除 metric_for_best_model（預設 eval_loss）")

    # 6. pth 儲存（down_proj.state_dict）
    chk("pth 儲存 down_proj",
        "down_proj.state_dict()" in src,
        "torch.save(trainer.model.down_proj.state_dict(), pth_path)")

    # 7. pth 檔名含正確 A_ 或 B_
    if fname.endswith("_a.py"):
        match_a = bool(re.search(r'_A_shared_encoder|_A_encoder', src))
        match_b = bool(re.search(r'_B_shared_encoder|_B_encoder', src))
        chk("pth 檔名含 _A_（非_B_）",
            match_a and not match_b,
            "pth 路徑應含 _A_，不能含 _B_")
    elif fname.endswith("_b.py"):
        match_a = bool(re.search(r'_A_shared_encoder|_A_encoder', src))
        match_b = bool(re.search(r'_B_shared_encoder|_B_encoder', src))
        chk("pth 檔名含 _B_（非_A_）",
            match_b and not match_a,
            "pth 路徑應含 _B_，不能含 _A_")

    # 8. summary_5runs.csv
    chk("summary_5runs.csv",
        "summary_5runs.csv" in src,
        "results_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_5runs.csv'), index=False)")

    # 9. results["run"] = run_i
    chk('results["run"]=run_i',
        'results["run"] = run_i' in src or "results['run'] = run_i" in src,
        'results["run"] = run_i')

    # 10. gc.collect
    chk("gc.collect()",
        "gc.collect()" in src,
        "del model, trainer; torch.cuda.empty_cache(); gc.collect()")

    # 11. 無 dataloader_drop_last（TrainingArguments 裡）
    drop_line = re.search(r'dataloader_drop_last\s*=', args_block)
    is_comment = drop_line and args_block[max(0, drop_line.start()-2):drop_line.start()].strip().endswith("#")
    chk("無 dataloader_drop_last",
        drop_line is None or is_comment,
        "移除 dataloader_drop_last=True")

    # 12. Scenario 路徑一致
    if "_a.py" in fname:
        chk("CSV 路徑含 scenario_A",
            "scenario_A_screening" in src,
            "TRAIN/TEST_CSV 應含 scenario_A_screening")
    elif "_b.py" in fname:
        chk("CSV 路徑含 scenario_B",
            "scenario_B_monitoring" in src,
            "TRAIN/TEST_CSV 應含 scenario_B_monitoring")

    # 13. XLS-R backbone
    if "xlsr" in fname:
        chk("XLS-R backbone",
            "wav2vec2-xls-r-300m" in src or "xls-r" in src.lower(),
            'MODEL_NAME = "facebook/wav2vec2-xls-r-300m"')

    # 14. spk_classifier 非 hardcode 200
    if "spk_classifier" in src and "nn.Linear(128," in src:
        chk("spk_classifier 非 hardcode 200",
            "Linear(128, 200)" not in src,
            "nn.Linear(128, getattr(config, 'num_speakers', ...))")

    # 15. 語法
    try:
        ast.parse(src); ok_list.append("✅ 語法合法")
    except SyntaxError as e:
        issues.append(f"❌ 語法錯誤: {e}")

    return ok_list, issues


def main():
    search_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    all_py = [f for f in os.listdir(search_dir) if f.endswith(".py")]
    matched = [f for f in all_py if f.lower() in TARGET_FILES]
    missing = [t for t in TARGET_FILES if t not in [f.lower() for f in all_py]]

    print(f"\n{'='*65}")
    print(f"🔍 對齊檢查 v2 — 訓練腳本專用")
    print(f"   掃描目錄：{search_dir}")
    print(f"   找到 {len(matched)}/{len(TARGET_FILES)} 個目標檔案")
    print(f"{'='*65}\n")

    if missing:
        print(f"⚠️  以下 {len(missing)} 個目標檔案不存在：")
        for f in sorted(missing):
            print(f"  ❓ {f}")
        print()

    clean, dirty = [], []
    for fname in sorted(matched):
        fpath = os.path.join(search_dir, fname)
        ok_list, issues = check_file(fpath)
        if issues:
            dirty.append((fname, issues))
        else:
            clean.append(fname)

    if dirty:
        print(f"⚠️  以下 {len(dirty)} 個檔案有問題：\n")
        for fname, issues in dirty:
            print(f"  📄 {fname}")
            for iss in issues:
                print(f"      {iss}")
            print()

    if clean:
        print(f"✅ 以下 {len(clean)} 個檔案全部通過：")
        for f in clean:
            print(f"  ✅ {f}")

    print(f"\n{'='*65}")
    print(f"📊 {len(clean)}/{len(matched)} 通過  |  {len(dirty)} 有問題  |  {len(missing)} 不存在")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()