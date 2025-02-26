import numpy as np

def normalize_amplitude(y, target_dBFS=-20):
    """將音頻歸一化到目標分貝全刻度值
    
    Args:
        y (numpy.ndarray): 音頻數據
        target_dBFS (float): 目標分貝全刻度值
        
    Returns:
        numpy.ndarray: 歸一化後的音頻
    """
    # 計算當前RMS
    rms = np.sqrt(np.mean(y**2))
    
    # 如果信號為零，則返回原始信號
    if rms < 1e-10:
        return y
    
    # 計算目標RMS
    target_rms = 10 ** (target_dBFS / 20)
    
    # 計算增益
    gain = target_rms / rms
    
    # 應用增益
    y_normalized = y * gain
    
    return y_normalized

def apply_preemphasis(y, coef=0.97):
    """應用預加重濾波器
    
    Args:
        y (numpy.ndarray): 音頻數據
        coef (float): 預加重係數
        
    Returns:
        numpy.ndarray: 預加重後的音頻
    """
    return np.append(y[0], y[1:] - coef * y[:-1])
