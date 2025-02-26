import os
import numpy as np
import librosa
import pandas as pd
from scipy.signal import lfilter

def extract_spectral_features(y, sr, n_fft=2048, hop_length=512):
    """提取頻譜特徵
    
    Args:
        y (numpy.ndarray): 音頻數據
        sr (int): 採樣率
        n_fft (int): FFT窗口大小
        hop_length (int): 幀移動長度
        
    Returns:
        dict: 頻譜特徵字典
    """
    # 計算頻譜質心
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    
    # 計算頻譜帶寬
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    
    # 計算頻譜衰減
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    
    # 計算頻譜平坦度
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    
    # 計算各特徵的統計量
    features = {
        "cent_mean": np.mean(cent),
        "cent_std": np.std(cent),
        "bandwidth_mean": np.mean(bandwidth),
        "bandwidth_std": np.std(bandwidth),
        "rolloff_mean": np.mean(rolloff),
        "rolloff_std": np.std(rolloff),
        "flatness_mean": np.mean(flatness),
        "flatness_std": np.std(flatness),
    }
    
    return features

def extract_lpc(y, sr, order=12, frame_length=2048, hop_length=512):
    """提取線性預測編碼(LPC)係數
    
    Args:
        y (numpy.ndarray): 音頻數據
        sr (int): 採樣率
        order (int): LPC階數
        frame_length (int): 幀長度
        hop_length (int): 幀移動長度
        
    Returns:
        numpy.ndarray: LPC係數，形狀為(order+1, n_frames)
    """
    # 確保音頻長度至少為一個幀
    if len(y) < frame_length:
        return np.zeros((order+1, 1))
    
    # 計算幀數
    n_frames = 1 + (len(y) - frame_length) // hop_length
    
    # 初始化LPC係數數組
    lpc_coeffs = np.zeros((order+1, n_frames))
    
    # 對每一幀計算LPC係數
    for i in range(n_frames):
        # 提取幀
        start = i * hop_length
        end = start + frame_length
        frame = y[start:end]
        
        # 應用窗函數
        frame = frame * np.hamming(frame_length)
        
        # 計算自相關
        r = np.correlate(frame, frame, mode='full')
        r = r[frame_length-1:frame_length+order]
        
        # Levinson-Durbin遞歸
        a = np.ones(order+1)
        e = r[0]
        
        for k in range(1, order+1):
            if e == 0:
                break
                
            # 計算反射係數
            lambda_k = -np.sum(a[:k] * r[k:0:-1]) / e
            
            # 更新LPC係數
            a_new = a.copy()
            for j in range(1, k):
                a_new[j] += lambda_k * a[k-j]
            a_new[k] = lambda_k
            a = a_new
            
            # 更新預測誤差
            e *= (1 - lambda_k**2)
        
        # 保存係數
        lpc_coeffs[:, i] = a
    
    return lpc_coeffs

def compute_hnr(y, sr, f0, confidence, min_f0=80, max_f0=500, voiced_threshold=0.5):
    """計算諧噪比(HNR)
    
    Args:
        y (numpy.ndarray): 音頻數據
        sr (int): 採樣率
        f0 (numpy.ndarray): 基頻估計
        confidence (numpy.ndarray): 可靠性估計
        min_f0 (int): 最小F0頻率
        max_f0 (int): 最大F0頻率
        voiced_threshold (float): 有聲判定閾值
        
    Returns:
        float: 平均諧噪比(dB)
    """
    # 篩選可靠的有聲幀
    voiced_frames = confidence >= voiced_threshold
    if np.sum(voiced_frames) == 0:
        return 0.0
    
    # 計算諧噪比的簡化實現
    # 注意：完整實現非常複雜，這裡僅做示例
    # 更精確的實現可以使用Praat或其他專門工具
    
    # 計算有聲幀的平均基頻
    mean_f0 = np.mean(f0[voiced_frames])
    
    # 基於平均基頻設置分析窗口長度
    period = int(sr / mean_f0)
    frame_length = period * 4
    
    # 如果窗口長度太大，則截斷
    if frame_length > len(y):
        frame_length = len(y)
    
    # 提取中心幀
    center = len(y) // 2
    start = center - frame_length // 2
    if start < 0:
        start = 0
    end = start + frame_length
    if end > len(y):
        end = len(y)
    
    frame = y[start:end]
    
    # 應用窗函數
    frame = frame * np.hamming(len(frame))
    
    # 計算功率譜
    spectrum = np.abs(np.fft.rfft(frame))**2
    freqs = np.fft.rfftfreq(len(frame), 1/sr)
    
    # 估計諧波部分和噪聲部分
    harmonic_power = 0
    noise_power = 0
    
    # 簡化處理：假設低頻部分主要是諧波，高頻部分主要是噪聲
    harmonic_idx = freqs < 1000
    noise_idx = freqs >= 1000
    
    harmonic_power = np.sum(spectrum[harmonic_idx])
    noise_power = np.sum(spectrum[noise_idx])
    
    # 避免除以零
    if noise_power < 1e-10:
        noise_power = 1e-10
    
    # 計算HNR(dB)
    hnr = 10 * np.log10(harmonic_power / noise_power)
    
    return hnr

def extract_spectral_from_dir(input_dir, output_file, extract_lpc_features=True, extract_hnr_features=True):
    """從目錄中提取所有音頻的頻譜特徵並保存
    
    Args:
        input_dir (str): 輸入音頻目錄
        output_file (str): 輸出特徵文件
        extract_lpc_features (bool): 是否提取LPC特徵
        extract_hnr_features (bool): 是否提取HNR特徵
        
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
            
            # 提取基本頻譜特徵
            spectral_features = extract_spectral_features(y, sr)
            features = spectral_features.copy()
            
            # 提取LPC特徵
            if extract_lpc_features:
                lpc = extract_lpc(y, sr)
                lpc_mean = np.mean(lpc, axis=1)
                lpc_std = np.std(lpc, axis=1)
                
                for i in range(len(lpc_mean)):
                    features[f"lpc_{i}_mean"] = lpc_mean[i]
                for i in range(len(lpc_std)):
                    features[f"lpc_{i}_std"] = lpc_std[i]
            
            # 提取HNR特徵
            if extract_hnr_features:
                # 首先需要提取F0
                f0, confidence = librosa.pyin(y, fmin=80, fmax=500, sr=sr)
                hnr = compute_hnr(y, sr, f0, confidence)
                features["hnr"] = hnr
            
            # 添加到列表
            features_list.append(list(features.values()))
            file_list.append(filename)
    
    # 創建數據框
    df = pd.DataFrame(features_list, columns=list(features.keys()))
    df.insert(0, "file", file_list)
    
    # 保存到文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    return df
