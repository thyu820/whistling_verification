import numpy as np
from scipy import signal

def apply_bandpass_filter(y, sr, lowcut=500, highcut=4000, order=5):
    """應用帶通濾波器
    
    Args:
        y (numpy.ndarray): 音頻數據
        sr (int): 採樣率
        lowcut (float): 低切頻率(Hz)
        highcut (float): 高切頻率(Hz)
        order (int): 濾波器階數
        
    Returns:
        numpy.ndarray: 濾波後的音頻
    """
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = signal.butter(order, [low, high], btype='band')
    y_filtered = signal.filtfilt(b, a, y)
    
    return y_filtered

def spectral_subtraction(y, sr, frame_length=2048, hop_length=512, alpha=2):
    """頻譜減法降噪
    
    Args:
        y (numpy.ndarray): 音頻數據
        sr (int): 採樣率
        frame_length (int): 幀長度
        hop_length (int): 幀移動長度
        alpha (float): 噪聲估計因子
        
    Returns:
        numpy.ndarray: 降噪後的音頻
    """
    # 計算STFT
    D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    
    # 計算幅度
    magnitude = np.abs(D)
    
    # 估計噪聲頻譜（假設前幾幀是噪聲）
    noise_frames = 5
    noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # 頻譜減法
    magnitude_subtracted = magnitude - alpha * noise_spectrum
    magnitude_subtracted = np.maximum(magnitude_subtracted, 0)
    
    # 重建相位
    phase = np.angle(D)
    D_subtracted = magnitude_subtracted * np.exp(1j * phase)
    
    # 反STFT
    y_subtracted = librosa.istft(D_subtracted, hop_length=hop_length)
    
    # 確保長度與原始信號相同
    if len(y_subtracted) > len(y):
        y_subtracted = y_subtracted[:len(y)]
    elif len(y_subtracted) < len(y):
        y_subtracted = np.pad(y_subtracted, (0, len(y) - len(y_subtracted)))
    
    return y_subtracted
