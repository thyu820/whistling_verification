#!/usr/bin/env python
import os
import argparse
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.logger import setup_logger
from src.models.gmm_ubm import GMMUBMModel
from src.models.ivector import IVectorModel

def main():
    parser = argparse.ArgumentParser(description="Train speaker verification model")
    parser.add_argument("--config", type=str, default="config/model_config.yaml",
                        help="Configuration file path")
    parser.add_argument("--features", type=str, default="data/features/all_features.csv",
                        help="Features file path")
    parser.add_argument("--exp-dir", type=str, default="experiments/exp_001",
                        help="Experiment directory")
    args = parser.parse_args()
    
    # 讀取配置
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except:
        print(f"Warning: Could not load config file {args.config}. Using defaults.")
        config = {}
    
    # 設置路徑和輸出目錄
    features_file = args.features
    exp_dir = args.exp_dir
    model_dir = os.path.join(exp_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # 設置模型參數
    model_config = config.get("model", {})
    model_type = model_config.get("type", "gmm_ubm")
    
    # GMM-UBM 模型參數
    gmm_params = model_config.get("gmm_ubm", {})
    n_components = gmm_params.get("n_components", 128)
    covariance_type = gmm_params.get("covariance_type", "diag")
    n_iter = gmm_params.get("n_iter", 100)
    
    # i-vector 模型參數
    ivector_params = model_config.get("ivector", {})
    tv_dim = ivector_params.get("tv_dim", 400)
    ivector_n_iter = ivector_params.get("n_iter", 10)
    
    # 數據配置
    data_config = config.get("data", {})
    train_ratio = data_config.get("train_ratio", 0.7)
    random_seed = data_config.get("random_seed", 42)
    
    # 設置日誌
    logger = setup_logger("train_model", log_dir=os.path.join(exp_dir, "logs"))
    logger.info("Starting model training")
    
    # 保存配置
    os.makedirs(os.path.join(exp_dir, "config"), exist_ok=True)
    with open(os.path.join(exp_dir, "config", "model_config.yaml"), 'w') as f:
        yaml.dump(config, f)
    
    # 加載特徵
    logger.info(f"Loading features from {features_file}")
    features_df = pd.read_csv(features_file)
    
    # 按說話人分組
    speaker_groups = features_df.groupby("speaker_id")
    speaker_ids = list(speaker_groups.groups.keys())
    n_speakers = len(speaker_ids)
    
    logger.info(f"Found {n_speakers} speakers")
    
    # 準備特徵
    features_list = []
    train_indices = []
    test_indices = []
    
    # 為每個說話人準備訓練和測試數據
    for i, (speaker_id, group) in enumerate(speaker_groups):
        # 排除特徵列中的非數值列
        feature_cols = [col for col in group.columns if col not in ["speaker_id", "file"]]
        speaker_features = group[feature_cols].values
        
        # 隨機混洗
        np.random.seed(random_seed + i)
        indices = np.random.permutation(len(speaker_features))
        
        # 分割訓練和測試
        train_size = int(len(indices) * train_ratio)
        speaker_train_idx = indices[:train_size]
        speaker_test_idx = indices[train_size:]
        
        # 存儲特徵和索引
        features_list.append(speaker_features)
        train_indices.append(speaker_train_idx)
        test_indices.append(speaker_test_idx)
    
    # 訓練模型
    if model_type == "gmm_ubm":
        logger.info(f"Training GMM-UBM model with {n_components} components")
        
        # 初始化模型
        model = GMMUBMModel(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_seed
        )
        
        # 準備所有訓練數據
        all_train_features = np.vstack([
            features_list[i][train_indices[i]] for i in range(n_speakers)
        ])
        
        # 訓練UBM
        logger.info(f"Training UBM with {len(all_train_features)} samples")
        model.train_ubm(all_train_features)
        
        # 為每個說話人適應模型
        for i, speaker_id in enumerate(speaker_ids):
            logger.info(f"Adapting model for speaker {speaker_id}")
            speaker_train_features = features_list[i][train_indices[i]]
            model.adapt_speaker_model(speaker_id, speaker_train_features)
        
    elif model_type == "ivector":
        logger.info(f"Training i-vector model with dim={tv_dim}")
        
        # 初始化模型
        model = IVectorModel(
            n_components=n_components,
            tv_dim=tv_dim,
            n_iter=ivector_n_iter,
            random_state=random_seed
        )
        
        # 準備所有訓練數據
        all_train_features = np.vstack([
            features_list[i][train_indices[i]] for i in range(n_speakers)
        ])
        
        # 訓練UBM
        logger.info(f"Training UBM with {len(all_train_features)} samples")
        model.train_ubm(all_train_features)
        
        # 準備每個說話人的訓練特徵
        speaker_train_features = [
            features_list[i][train_indices[i]] for i in range(n_speakers)
        ]
        
        # 訓練總變異性矩陣並提取i-vectors
        logger.info("Training TV matrix and extracting i-vectors")
        model.train_tv_matrix(speaker_train_features, speaker_ids)
    
    else:
        logger.error(f"Unknown model type: {model_type}")
        return
    
    # 保存模型
    model_file = os.path.join(model_dir, f"{model_type}_model.pkl")
    logger.info(f"Saving model to {model_file}")
    model.save(model_file)
    
    logger.info("Model training completed!")

if __name__ == "__main__":
    main()
