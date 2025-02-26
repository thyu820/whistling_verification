import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_curve

def plot_det_curve(scores, labels, title="DET Curve", figsize=(10, 8), save_path=None):
    """繪製檢測錯誤權衡(DET)曲線
    
    Args:
        scores (numpy.ndarray): 相似度分數
        labels (numpy.ndarray): 真實標籤(1=目標, 0=非目標)
        title (str): 圖表標題
        figsize (tuple): 圖表大小
        save_path (str): 可選的保存路徑
        
    Returns:
        matplotlib.figure.Figure: 圖表對象
    """
    # 計算ROC曲線點
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # 計算錯誤拒絕率(FRR)
    frr = 1 - tpr
    
    # 應用正態偏差變換
    def probit(x):
        return norm.ppf(x)
    
    # 處理極端值
    fpr_safe = np.maximum(fpr, 1e-8)
    fpr_safe = np.minimum(fpr_safe, 1 - 1e-8)
    
    frr_safe = np.maximum(frr, 1e-8)
    frr_safe = np.minimum(frr_safe, 1 - 1e-8)
    
    # 應用變換
    x_det = probit(fpr_safe)
    y_det = probit(frr_safe)
    
    # 繪製DET曲線
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x_det, y_det, 'b-', linewidth=2)
    
    # 計算並標記EER點
    eer_index = np.argmin(np.abs(fpr - frr))
    eer = (fpr[eer_index] + frr[eer_index]) / 2
    eer_x = probit(fpr_safe[eer_index])
    eer_y = probit(frr_safe[eer_index])
    
    ax.plot(eer_x, eer_y, 'ro', markersize=8)
    ax.annotate(f'EER = {eer:.2%}', 
                xy=(eer_x, eer_y), 
                xytext=(eer_x+0.1, eer_y+0.1),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    # 設置軸標籤
    ticks = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 0.999]
    tick_labels = ['0.1', '1', '5', '10', '20', '40', '60', '80', '90', '95', '99', '99.9']
    
    tick_positions = probit(np.array(ticks))
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    # 添加網格
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # 設置軸範圍
    ax.set_xlim([probit(0.001), probit(0.5)])
    ax.set_ylim([probit(0.001), probit(0.5)])
    
    # 設置標題和軸標籤
    ax.set_title(title)
    ax.set_xlabel('False Alarm Rate (%)')
    ax.set_ylabel('Miss Rate (%)')
    
    plt.tight_layout()
    
    # 保存圖表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

