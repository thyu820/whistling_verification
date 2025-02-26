#!/usr/bin/env python
import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.logger import setup_logger
from src.models.gmm_ubm import GMMUBMModel
from src.models.ivector import IVectorModel
from src.evaluation.metrics import compute_eer, compute_dcf, compute_auc
from src.evaluation.det_curve import plot_det_curve

def main():
    parser = argparse.ArgumentParser(description="Evaluate speaker verification model")
    parser.add_argument("--exp-dir", type=str, required=True,
                        help="Experiment directory")
    parser.add_argument("--features", type=str,
                        help="Features file path (overrides config)")
    parser.add_argument("--model", type=str,
                        help="Model file path (overrides default location)")
    args = parser.parse_args()
    
    # 設置路徑
    exp_dir = args.exp_dir
    config_file = os.path.join(exp_dir, "config", "model_config.yaml")
    results_dir = os.path.join(exp_dir, "results")
    plots_dir = os.path.join(exp_dir, "plots")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 讀取配置
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except:
        print(f"Warning: Could not load config file {config_file}. Using defaults.")
        config = {}
    
    # 設置模型和特徵路徑
    model_config = config.get("model", {})
    model_type = model_config.get("type", "gmm_ubm")
    
    if args.model:
        model_file = args.model
    else:
        model_file = os.path.join(exp_dir, "model", f"{model_type}_model.pkl")
    
    if args.features:
        features_file = args.features
    else:
        features_file = config.get("features_file", "data/features/all_features.csv")
    
    # 評估配置
    eval_config = config.get("evaluation", {})
    p_target = eval_config.get("p_target", 0.01)
    c_miss = eval_config.get("c_miss", 10)
    c_fa = eval_config.get("c_fa", 1)
    
    # 設置日誌
    logger = setup_logger("evaluate", log_dir=os.path.join(exp_dir, "logs"))
    logger.info("Starting model evaluation")
    
    # 加載模型
    logger.info(f"Loading model from {model_file}")
    if model_type == "gmm_ubm":
        model = GMMUBMModel.load(model_file)
    elif model_type == "ivector":
        model = IVectorModel.load(model_file)
    else:
        logger.error(f"Unknown model type: {model_type}")
        return
    
    if model is None:
        logger.error("Failed to load model!")
        return
    
    # 加載特徵
    logger.info(f"Loading features from {features_file}")
    features_df = pd.read_csv(features_file)
    
    # 按說話人分組
    speaker_groups = features_df.groupby("speaker_id")
    speaker_ids = list(speaker_groups.groups.keys())
    n_speakers = len(speaker_ids)
    
    logger.info(f"Found {n_speakers} speakers")
    
    # 生成試驗對
    logger.info("Generating trial pairs")
    
    all_scores = []
    all_labels = []
    trial_pairs = []
    
    # 每個說話人作為目標說話人
    for i, target_id in enumerate(tqdm(speaker_ids, desc="Evaluating speakers")):
        target_group = speaker_groups.get_group(target_id)
        
        # 排除非數值列
        feature_cols = [col for col in target_group.columns if col not in ["speaker_id", "file"]]
        
        # 對每個樣本進行評估
        for _, row in target_group.iterrows():
            test_features = row[feature_cols].values.reshape(1, -1)
            test_file = row["file"]
            
            # 目標試驗（真正例）
            score = model.score(test_features, target_id)
            all_scores.append(score)
            all_labels.append(1)
            trial_pairs.append((test_file, target_id, True))
            
            # 非目標試驗（真負例）- 對每個其他說話人
            for non_target_id in speaker_ids:
                if non_target_id != target_id:
                    try:
                        score = model.score(test_features, non_target_id)
                        all_scores.append(score)
                        all_labels.append(0)
                        trial_pairs.append((test_file, non_target_id, False))
                    except Exception as e:
                        logger.warning(f"Error scoring {test_file} against {non_target_id}: {str(e)}")
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # 計算評估指標
    logger.info("Computing evaluation metrics")
    
    eer, eer_threshold = compute_eer(all_scores, all_labels)
    min_dcf, dcf_threshold = compute_dcf(all_scores, all_labels, p_target=p_target, c_miss=c_miss, c_fa=c_fa)
    auc_value = compute_auc(all_scores, all_labels)
    
    # 保存評估結果
    results = {
        "eer": eer,
        "min_dcf": min_dcf,
        "auc": auc_value,
        "eer_threshold": eer_threshold,
        "dcf_threshold": dcf_threshold,
        "n_trials": len(all_scores),
        "n_target_trials": sum(all_labels),
        "n_non_target_trials": len(all_labels) - sum(all_labels)
    }
    
    # 保存指標
    results_file = os.path.join(results_dir, "metrics.csv")
    pd.DataFrame([results]).to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    # 保存分數
    scores_df = pd.DataFrame({
        "test_file": [p[0] for p in trial_pairs],
        "claimed_speaker": [p[1] for p in trial_pairs],
        "target": [p[2] for p in trial_pairs],
        "score": all_scores
    })
    scores_file = os.path.join(results_dir, "scores.csv")
    scores_df.to_csv(scores_file, index=False)
    
    # 繪製DET曲線
    logger.info("Plotting DET curve")
    det_fig = plot_det_curve(
        all_scores, all_labels,
        title=f"{model_type.upper()} DET Curve - EER: {eer:.2%}, minDCF: {min_dcf:.4f}"
    )
    det_file = os.path.join(plots_dir, "det_curve.png")
    det_fig.savefig(det_file, dpi=300, bbox_inches='tight')
    
    # 繪製分數分佈
    logger.info("Plotting score distributions")
    plt.figure(figsize=(10, 6))
    
    # 目標和非目標分數
    target_scores = all_scores[all_labels == 1]
    non_target_scores = all_scores[all_labels == 0]
    
    plt.hist(non_target_scores, bins=50, alpha=0.5, label='Non-target', density=True)
    plt.hist(target_scores, bins=50, alpha=0.5, label='Target', density=True)
    
    # 添加閾值線
    plt.axvline(x=eer_threshold, color='r', linestyle='--', label=f'EER Threshold: {eer_threshold:.2f}')
    plt.axvline(x=dcf_threshold, color='g', linestyle='--', label=f'minDCF Threshold: {dcf_threshold:.2f}')
    
    plt.title(f"Score Distributions - EER: {eer:.2%}")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    dist_file = os.path.join(plots_dir, "score_distributions.png")
    plt.savefig(dist_file, dpi=300, bbox_inches='tight')
    
    # 顯示結果摘要
    logger.info("Evaluation completed!")
    logger.info(f"Equal Error Rate (EER): {eer:.2%}")
    logger.info(f"Minimum DCF: {min_dcf:.4f}")
    logger.info(f"AUC: {auc_value:.4f}")
    logger.info(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    main()
