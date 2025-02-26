#!/usr/bin/env python
import os
import argparse
import yaml
import glob
import pandas as pd
from tqdm import tqdm

from src.utils.logger import setup_logger
from src.features.mfcc import extract_mfcc_from_dir
from src.features.pitch import extract_pitch_from_dir
from src.features.spectral import extract_spectral_from_dir

def main():
    parser = argparse.ArgumentParser(description="Extract features from whistling audio files")
    parser.add_argument("--config", type=str, default="config/feature_config.yaml",
                        help="Configuration file path")
    parser.add_argument("--input-dir", type=str, help="Input directory (overrides config)")
    parser.add_argument("--output-dir", type=str, help="Output directory (overrides config)")
    args = parser.parse_args()
    
    # 讀取配置
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except:
        print(f"Warning: Could not load config file {args.config}. Using defaults.")
        config = {}
    
    # 設置路徑
    input_dir = args.input_dir if args.input_dir else config.get("input_dir", "data/processed")
    output_dir = args.output_dir if args.output_dir else config.get("output_dir", "data/features")
    
    # 設置參數
    features_config = config.get("features", {})
    
    mfcc_config = features_config.get("mfcc", {})
    extract_mfcc = mfcc_config.get("extract", True)
    n_mfcc = mfcc_config.get("n_mfcc", 20)
    delta = mfcc_config.get("delta", True)
    delta_delta = mfcc_config.get("delta_delta", True)
    cmvn = mfcc_config.get("cmvn", True)
    
    pitch_config = features_config.get("pitch", {})
    extract_pitch = pitch_config.get("extract", True)
    min_f0 = pitch_config.get("min_f0", 80)
    max_f0 = pitch_config.get("max_f0", 500)
    pitch_method = pitch_config.get("method", "pyin")
    
    spectral_config = features_config.get("spectral", {})
    extract_spectral = spectral_config.get("extract", True)
    extract_lpc = spectral_config.get("extract_lpc", True)
    extract_hnr = spectral_config.get("extract_hnr", True)
    
    # 設置日誌
    logger = setup_logger("feature_extraction", log_dir="logs")
    logger.info("Starting feature extraction")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 尋找所有說話人目錄
    speaker_dirs = [d for d in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(d)]
    
    logger.info(f"Found {len(speaker_dirs)} speaker directories")
    
    # 處理每個說話人
    all_features_dfs = []
    
    for speaker_dir in tqdm(speaker_dirs, desc="Processing speakers"):
        speaker_id = os.path.basename(speaker_dir)
        logger.info(f"Processing speaker {speaker_id}")
        
        speaker_output_dir = os.path.join(output_dir, speaker_id)
        os.makedirs(speaker_output_dir, exist_ok=True)
        
        speaker_features = {}
        
        # 提取MFCC特徵
        if extract_mfcc:
            logger.info(f"Extracting MFCC features for {speaker_id}")
            mfcc_file = os.path.join(speaker_output_dir, "mfcc.csv")
            mfcc_df = extract_mfcc_from_dir(
                speaker_dir, mfcc_file,
                n_mfcc=n_mfcc,
                delta=delta,
                delta_delta=delta_delta,
                cmvn=cmvn
            )
            speaker_features["mfcc"] = mfcc_df
        
        # 提取音高特徵
        if extract_pitch:
            logger.info(f"Extracting pitch features for {speaker_id}")
            pitch_file = os.path.join(speaker_output_dir, "pitch.csv")
            pitch_df = extract_pitch_from_dir(
                speaker_dir, pitch_file,
                fmin=min_f0,
                fmax=max_f0,
                method=pitch_method
            )
            speaker_features["pitch"] = pitch_df
        
        # 提取頻譜特徵
        if extract_spectral:
            logger.info(f"Extracting spectral features for {speaker_id}")
            spectral_file = os.path.join(speaker_output_dir, "spectral.csv")
            spectral_df = extract_spectral_from_dir(
                speaker_dir, spectral_file,
                extract_lpc_features=extract_lpc,
                extract_hnr_features=extract_hnr
            )
            speaker_features["spectral"] = spectral_df
        
        # 合併所有特徵
        if speaker_features:
            dfs = list(speaker_features.values())
            merged_df = dfs[0]
            
            for df in dfs[1:]:
                merged_df = pd.merge(merged_df, df, on="file")
            
            # 添加說話人ID
            merged_df.insert(0, "speaker_id", speaker_id)
            
            # 保存合併特徵
            merged_file = os.path.join(speaker_output_dir, "all_features.csv")
            merged_df.to_csv(merged_file, index=False)
            
            # 添加到所有特徵列表
            all_features_dfs.append(merged_df)
    
    # 合併所有說話人的特徵
    if all_features_dfs:
        all_features = pd.concat(all_features_dfs, ignore_index=True)
        all_features_file = os.path.join(output_dir, "all_features.csv")
        all_features.to_csv(all_features_file, index=False)
        logger.info(f"Saved combined features to {all_features_file}")
    
    logger.info("Feature extraction completed!")

if __name__ == "__main__":
    main()
