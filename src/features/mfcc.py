import os
import numpy as np
import librosa
import pandas as pd

def extract_mfcc(y, sr, n_mfcc=20, n_fft=2048, hop_length=512, lifter=22, delta=True, delta_delta=True):
    """提取MFCC特徵
    
    Args:
        y (numpy.ndarray): 音頻數據
        sr (int): 採樣率
        n_mfcc (int): MFCC係數數量
        n_fft (int): FFT窗口大小
        hop_length (int): 幀移動長度
        lifter (int): 提升參數
        delta (bool): 是否計算一階差分
        delta_delta (bool): 是否計算二階差分
        
    Returns:
        numpy.ndarray: MFCC特徵
    """
    # 提取基本MFCC
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    
    # 應用提升濾波器增強高階係數
    if lifter > 0:
        mfccs = apply_lifter(mfccs, lifter)
    
    # 計算差分
    feature_list = [mfccs]
    if delta:
        # 計算一階差分
        mfcc_delta = librosa.feature.delta(mfccs)
        feature_list.append(mfcc_delta)
        
        if delta_delta:
            # 計算二階差分
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            feature_list.append(mfcc_delta2)
    
    # 拼接特徵
    features = np.vstack(feature_list)
    
    return features

def apply_lifter(mfccs, lifter=22):
    """應用提升濾波器增強高階MFCC
    
    Args:
        mfccs (numpy.ndarray): MFCC係數
        lifter (int): 提升參數
        
    Returns:
        numpy.ndarray: 提升後的MFCC
    """
    if lifter == 0:
        return mfccs
    
    ncoeff = mfccs.shape[0]
    lift = 1 + (lifter / 2) * np.sin(np.pi * np.arange(ncoeff) / lifter)
    
    return lift.reshape(-1, 1) * mfccs

def compute_cmvn(features, axis=1):
    """計算均值和方差歸一化
    
    Args:
        features (numpy.ndarray): 輸入特徵
        axis (int): 計算均值和方差的軸
        
    Returns:
        numpy.ndarray: 歸一化後的特徵
    """
    # 計算均值和標準差
    mean = np.mean(features, axis=axis, keepdims=True)
    std = np.std(features, axis=axis, keepdims=True)
    
    # 避免除以零
    std = np.maximum(std, 1e-10)
    
    # 歸一化
    normalized = (features - mean) / std
    
    return normalized

def extract_mfcc_from_dir(input_dir, output_file, n_mfcc=20, delta=True, delta_delta=True, cmvn=True):
    """從目錄中提取所有音頻的MFCC特徵並保存
    
    Args:
        input_dir (str): 輸入音頻目錄
        output_file (str): 輸出特徵文件
        n_mfcc (int): MFCC係數數量
        delta (bool): 是否計算一階差分
        delta_delta (bool): 是否計算二階差分
        cmvn (bool): 是否應用CMVN歸一化
        
    Returns:
        pandas.DataFrame: 特徵數據框
    """
    features_list = []
    file_list = []
    
    # 遍歷目錄中的所有音頻文件
    for filename in os.listdir(input_dir):
        if filename.endswith(('.wav', '.WAV')):
            file_path = os.path.join(input_dir, filename)
            
            # 加載音頻
            y, sr = librosa.load(file_path, sr=None)
            
            # 提取MFCC
            mfccs = extract_mfcc(y, sr, n_mfcc=n_mfcc, delta=delta, delta_delta=delta_delta)
            
            # 應用CMVN
            if cmvn:
                mfccs = compute_cmvn(mfccs)
            
            # 計算統計量
            mean_features = np.mean(mfccs, axis=1)
            std_features = np.std(mfccs, axis=1)
            
            # 合併統計量
            combined_features = np.concatenate([mean_features, std_features])
            
            # 添加到列表
            features_list.append(combined_features)
            file_list.append(filename)
    
    # 創建特徵名稱
    feature_dim = n_mfcc * (1 + delta + delta_delta) * 2
    feature_names = []
    for i in range(n_mfcc * (1 + delta + delta_delta)):
        feature_names.append(f"mfcc_{i}_mean")
    for i in range(n_mfcc * (1 + delta + delta_delta)):
        feature_names.append(f"mfcc_{i}_std")
    
    # 創建數據框
    df = pd.DataFrame(features_list, columns=feature_names)
    df.insert(0, "file", file_list)
    
    # 保存到文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    return df


# whistling_verification/src/features/pitch.py
import os
import numpy as np
import librosa
import pandas as pd

def extract_f0(y, sr, fmin=80, fmax=500, frame_length=2048, hop_length=512, method='pyin'):
    """提取基頻(F0)特徵
    
    Args:
        y (numpy.ndarray): 音頻數據
        sr (int): 採樣率
        fmin (int): 最小F0頻率
        fmax (int): 最大F0頻率
        frame_length (int): 幀長度
        hop_length (int): 幀移動長度
        method (str): 提取方法，'yin'或'pyin'
        
    Returns:
        tuple: (F0序列, 可靠性估計)
    """
    if method == 'yin':
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop_length)
        # YIN沒有可靠性估計，所以我們使用一個空的數組
        confidence = np.ones_like(f0)
    elif method == 'pyin':
        f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop_length)
        confidence = voiced_prob
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 處理NaN值
    f0 = np.nan_to_num(f0)
    
    return f0, confidence

def compute_pitch_statistics(f0, confidence, voiced_threshold=0.5):
    """計算F0統計特徵
    
    Args:
        f0 (numpy.ndarray): F0序列
        confidence (numpy.ndarray): 可靠性估計
        voiced_threshold (float): 有聲判定閾值
        
    Returns:
        dict: 統計特徵字典
    """
    # 篩選可靠的有聲幀
    voiced_frames = confidence >= voiced_threshold
    if np.sum(voiced_frames) == 0:
        return {
            "f0_mean": 0,
            "f0_std": 0,
            "f0_min": 0,
            "f0_max": 0,
            "f0_range": 0,
            "f0_median": 0,
            "voiced_ratio": 0,
        }
    
    voiced_f0 = f0[voiced_frames]
    
    # 計算基本統計量
    f0_mean = np.mean(voiced_f0)
    f0_std = np.std(voiced_f0)
    f0_min = np.min(voiced_f0)
    f0_max = np.max(voiced_f0)
    f0_range = f0_max - f0_min
    f0_median = np.median(voiced_f0)
    
    # 計算有聲幀比例
    voiced_ratio = np.sum(voiced_frames) / len(f0)
    
    return {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "f0_range": f0_range,
        "f0_median": f0_median,
        "voiced_ratio": voiced_ratio,
    }

def extract_pitch_from_dir(input_dir, output_file, fmin=80, fmax=500, method='pyin'):
    """從目錄中提取所有音頻的F0特徵並保存
    
    Args:
        input_dir (str): 輸入音頻目錄
        output_file (str): 輸出特徵文件
        fmin (int): 最小F0頻率
        fmax (int): 最大F0頻率
        method (str): 提取方法，'yin'或'pyin'
        
    Returns:
        pandas.DataFrame: 特徵數據框
    """
    features_list = []
    file_list = []
    
    # 遍歷目錄中的所有音頻文件
    for filename in os.listdir(input_dir):
        if filename.endswith(('.wav', '.WAV')):
            file_path = os.path.join(input_dir, filename)
            
            # 加載音頻
            y, sr = librosa.load(file_path, sr=None)
            
            # 提取F0
            f0, confidence = extract_f0(y, sr, fmin=fmin, fmax=fmax, method=method)
            
            # 計算統計量
            stats = compute_pitch_statistics(f0, confidence)
            
            # 添加到列表
            features_list.append(list(stats.values()))
            file_list.append(filename)
    
    # 創建數據框
    df = pd.DataFrame(features_list, columns=list(stats.keys()))
    df.insert(0, "file", file_list)
    
    # 保存到文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    return df
