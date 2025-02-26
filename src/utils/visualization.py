import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

def plot_waveform(y, sr, title="Waveform", figsize=(10, 4)):
    """繪製音頻波形圖
    
    Args:
        y (numpy.ndarray): 音頻數據
        sr (int): 採樣率
        title (str): 圖表標題
        figsize (tuple): 圖表大小
        
    Returns:
        matplotlib.figure.Figure: 圖表對象
    """
    fig, ax = plt.subplots(figsize=figsize)
    time = np.arange(0, len(y)) / sr
    ax.plot(time, y)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig

def plot_spectrogram(y, sr, title="Spectrogram", figsize=(10, 6)):
    """繪製聲譜圖
    
    Args:
        y (numpy.ndarray): 音頻數據
        sr (int): 採樣率
        title (str): 圖表標題
        figsize (tuple): 圖表大小
        
    Returns:
        matplotlib.figure.Figure: 圖表對象
    """
    fig, ax = plt.subplots(figsize=figsize)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig

def plot_feature_distributions(features, labels, feature_names=None, title="Feature Distributions", figsize=(12, 8)):
    """繪製特徵分佈圖
    
    Args:
        features (numpy.ndarray): 特徵數據，形狀為 (n_samples, n_features)
        labels (numpy.ndarray): 標籤數據
        feature_names (list): 特徵名稱列表
        title (str): 圖表標題
        figsize (tuple): 圖表大小
        
    Returns:
        matplotlib.figure.Figure: 圖表對象
    """
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(features.shape[1])]
    
    n_features = min(5, features.shape[1])  # 最多顯示5個特徵
    
    fig, axes = plt.subplots(1, n_features, figsize=figsize)
    if n_features == 1:
        axes = [axes]
    
    unique_labels = np.unique(labels)
    
    for i in range(n_features):
        for label in unique_labels:
            mask = labels == label
            sns.kdeplot(features[mask, i], ax=axes[i], label=f"Speaker {label}")
        
        axes[i].set_title(feature_names[i])
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.legend()
    
    return fig

def compare_experiments(exp_dirs, metrics, output_dir="results", title="Experiment Comparison"):
    """比較多個實驗的結果
    
    Args:
        exp_dirs (list): 實驗目錄列表
        metrics (list): 要比較的指標列表
        output_dir (str): 輸出目錄
        title (str): 圖表標題
        
    Returns:
        pandas.DataFrame: 比較結果
    """
    results = []
    
    for exp_dir in exp_dirs:
        # 讀取實驗結果
        result_file = os.path.join(exp_dir, "results", "metrics.csv")
        if not os.path.exists(result_file):
            print(f"Warning: Results file not found for experiment {exp_dir}")
            continue
            
        exp_results = pd.read_csv(result_file)
        exp_name = os.path.basename(exp_dir)
        
        row = {"Experiment": exp_name}
        for metric in metrics:
            if metric in exp_results.columns:
                row[metric] = exp_results[metric].iloc[0]
            else:
                row[metric] = np.nan
                
        results.append(row)
    
    if not results:
        print("No results found for comparison")
        return None
        
    # 創建比較結果DataFrame
    results_df = pd.DataFrame(results)
    
    # 繪製比較圖表
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.set_index("Experiment")[metrics].plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "experiment_comparison.png"))
    
    # 保存比較結果
    results_df.to_csv(os.path.join(output_dir, "experiment_comparison.csv"), index=False)
    
    return results_df
