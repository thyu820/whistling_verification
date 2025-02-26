import os
import numpy as np
import librosa
from ..utils.audio_io import load_audio, save_audio

def silence_removal(y, sr, top_db=30, min_silence_duration=0.1):
    """移除靜音部分
    
    Args:
        y (numpy.ndarray): 音頻數據
        sr (int): 採樣率
        top_db (float): 檢測靜音的閾值(dB)
        min_silence_duration (float): 最小靜音持續時間(秒)
        
    Returns:
        numpy.ndarray: 去除靜音後的音頻
    """
    # 檢測非靜音間隔
    intervals = librosa.effects.split(y, top_db=top_db)
    
    # 計算最小靜音樣本數
    min_silence_samples = int(min_silence_duration * sr)
    
    # 過濾太短的間隔
    filtered_intervals = []
    for interval in intervals:
        if interval[1] - interval[0] >= min_silence_samples:
            filtered_intervals.append(interval)
    
    # 如果沒有足夠長的非靜音間隔，返回原始音頻
    if not filtered_intervals:
        return y
    
    # 拼接非靜音部分
    y_filtered = np.concatenate([y[start:end] for start, end in filtered_intervals])
    
    return y_filtered

def segment_whistling(audio_path, output_dir, min_duration=0.5, max_duration=5.0, top_db=20):
    """切分口哨聲片段並保存
    
    Args:
        audio_path (str): 輸入音頻路徑
        output_dir (str): 輸出目錄
        min_duration (float): 最小片段持續時間(秒)
        max_duration (float): 最大片段持續時間(秒)
        top_db (float): 檢測靜音的閾值(dB)
        
    Returns:
        list: 保存的片段文件路徑列表
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 加載音頻
    y, sr = load_audio(audio_path)
    if y is None:
        return []
    
    # 檢測非靜音間隔
    intervals = librosa.effects.split(y, top_db=top_db)
    
    # 計算最小和最大樣本數
    min_samples = int(min_duration * sr)
    max_samples = int(max_duration * sr)
    
    # 文件名基礎部分
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    # 切分並保存片段
    output_files = []
    for i, (start, end) in enumerate(intervals):
        # 檢查持續時間
        duration = end - start
        if duration < min_samples:
            continue  # 太短，跳過
        
        # 如果超過最大持續時間，進一步切分
        if duration > max_samples:
            # 計算需要多少個片段
            n_segments = int(np.ceil(duration / max_samples))
            # 計算每個片段的樣本數
            segment_samples = duration // n_segments
            
            for j in range(n_segments):
                seg_start = start + j * segment_samples
                # 確保最後一個片段不超過終點
                seg_end = min(seg_start + segment_samples, end)
                
                # 保存片段
                segment = y[seg_start:seg_end]
                output_file = os.path.join(output_dir, f"{base_filename}_seg{i}_{j}.wav")
                save_audio(segment, sr, output_file)
                output_files.append(output_file)
        else:
            # 保存片段
            segment = y[start:end]
            output_file = os.path.join(output_dir, f"{base_filename}_seg{i}.wav")
            save_audio(segment, sr, output_file)
            output_files.append(output_file)
    
    return output_files

