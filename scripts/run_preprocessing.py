#!/usr/bin/env python
import os
import argparse
import yaml
import glob
from tqdm import tqdm

from src.utils.logger import setup_logger
from src.preprocessing.segmentation import segment_whistling
from src.preprocessing.denoising import apply_bandpass_filter, spectral_subtraction
from src.preprocessing.normalization import normalize_amplitude, apply_preemphasis
from src.utils.audio_io import load_audio, save_audio

def main():
    parser = argparse.ArgumentParser(description="Preprocess whistling audio files")
    parser.add_argument("--config", type=str, default="config/preprocess_config.yaml",
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
    input_dir = args.input_dir if args.input_dir else config.get("input_dir", "data/raw")
    output_dir = args.output_dir if args.output_dir else config.get("output_dir", "data/processed")
    
    # 設置參數
    segmentation = config.get("segmentation", {})
    min_duration = segmentation.get("min_duration", 0.5)
    max_duration = segmentation.get("max_duration", 5.0)
    top_db = segmentation.get("top_db", 20)
    
    denoising = config.get("denoising", {})
    apply_bandpass = denoising.get("apply_bandpass", True)
    lowcut = denoising.get("lowcut", 500)
    highcut = denoising.get("highcut", 4000)
    apply_spectral_sub = denoising.get("apply_spectral_subtraction", False)
    
    normalization = config.get("normalization", {})
    normalize = normalization.get("normalize_amplitude", True)
    target_dBFS = normalization.get("target_dBFS", -20)
    apply_pre_emphasis = normalization.get("apply_pre_emphasis", True)
    pre_emphasis_coef = normalization.get("pre_emphasis_coef", 0.97)
    
    # 設置日誌
    logger = setup_logger("preprocessing", log_dir="logs")
    logger.info("Starting audio preprocessing")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 尋找所有音頻文件
    audio_files = []
    for ext in ['.wav', '.WAV']:
        audio_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # 處理每個音頻文件
    for audio_file in tqdm(audio_files, desc="Processing files"):
        logger.info(f"Processing {audio_file}")
        
        # 加載音頻
        y, sr = load_audio(audio_file)
        if y is None:
            logger.warning(f"Could not load {audio_file}, skipping")
            continue
        
        # 降噪處理
        if apply_bandpass:
            logger.debug(f"Applying bandpass filter: {lowcut}-{highcut} Hz")
            y = apply_bandpass_filter(y, sr, lowcut, highcut)
        
        if apply_spectral_sub:
            logger.debug("Applying spectral subtraction")
            y = spectral_subtraction(y, sr)
        
        # 振幅歸一化
        if normalize:
            logger.debug(f"Normalizing amplitude to {target_dBFS} dBFS")
            y = normalize_amplitude(y, target_dBFS)
        
        # 預加重
        if apply_pre_emphasis:
            logger.debug(f"Applying pre-emphasis with coefficient {pre_emphasis_coef}")
            y = apply_preemphasis(y, pre_emphasis_coef)
        
        # 切分口哨片段
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        speaker_output_dir = os.path.join(output_dir, base_name)
        
        logger.info(f"Segmenting into {speaker_output_dir}")
        segments = segment_whistling(
            audio_file, speaker_output_dir,
            min_duration=min_duration,
            max_duration=max_duration,
            top_db=top_db
        )
        
        logger.info(f"Created {len(segments)} segments")
    
    logger.info("Preprocessing completed!")

if __name__ == "__main__":
    main()

