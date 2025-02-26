import os
import librosa
import soundfile as sf
import numpy as np

def load_audio(file_path, sr=None, mono=True):
    """加載音頻文件
    
    Args:
        file_path (str): 音頻文件路徑
        sr (int, optional): 目標採樣率，None表示保持原始採樣率
        mono (bool): 是否轉換為單聲道
        
    Returns:
        tuple: (音頻數據, 採樣率)
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=mono)
        return y, sr
    except Exception as e:
        print(f"Error loading audio file {file_path}: {str(e)}")
        return None, None

def save_audio(y, sr, file_path):
    """保存音頻文件
    
    Args:
        y (numpy.ndarray): 音頻數據
        sr (int): 採樣率
        file_path (str): 目標文件路徑
        
    Returns:
        bool: 是否成功保存
    """
    try:
        # 確保目錄存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        sf.write(file_path, y, sr)
        return True
    except Exception as e:
        print(f"Error saving audio file {file_path}: {str(e)}")
        return False
        
def split_audio(y, sr, segment_length=1.0, overlap=0.0):
    """將音頻切分為等長片段
    
    Args:
        y (numpy.ndarray): 音頻數據
        sr (int): 採樣率
        segment_length (float): 片段長度（秒）
        overlap (float): 重疊比例 (0.0-0.9)
        
    Returns:
        list: 音頻片段列表
    """
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError("Overlap must be in range [0.0, 0.9]")
        
    # 計算每個片段的長度（樣本數）
    segment_samples = int(segment_length * sr)
    # 計算步長（樣本數）
    hop_samples = int(segment_samples * (1 - overlap))
    
    # 計算切分點
    starts = np.arange(0, len(y) - segment_samples + 1, hop_samples)
    
    # 切分音頻
    segments = [y[start:start + segment_samples] for start in starts]
    
    return segments
