import numpy as np
from sklearn.metrics import roc_curve, auc

def compute_eer(scores, labels):
    """計算等錯誤率(EER)
    
    Args:
        scores (numpy.ndarray): 相似度分數
        labels (numpy.ndarray): 真實標籤(1=目標, 0=非目標)
        
    Returns:
        tuple: (EER值, EER對應的閾值)
    """
    # 計算ROC曲線
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # 計算錯誤接受率(FAR)和錯誤拒絕率(FRR)
    fnr = 1 - tpr  # 假負率(FRR)
    
    # 找出FAR和FRR最接近的點
    eer_index = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    
    return eer, eer_threshold

def compute_dcf(scores, labels, p_target=0.01, c_miss=10, c_fa=1):
    """計算檢測代價函數(DCF)
    
    Args:
        scores (numpy.ndarray): 相似度分數
        labels (numpy.ndarray): 真實標籤(1=目標, 0=非目標)
        p_target (float): 目標說話人先驗概率
        c_miss (float): 漏檢懲罰
        c_fa (float): 誤檢懲罰
        
    Returns:
        tuple: (min DCF, 對應閾值)
    """
    # 計算ROC曲線點
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # 計算各個閾值下的DCF值
    p_non_target = 1 - p_target
    dcf = p_target * c_miss * (1 - tpr) + p_non_target * c_fa * fpr
    
    # 歸一化DCF
    dcf_norm = dcf / min(p_target * c_miss, p_non_target * c_fa)
    
    # 最小DCF
    min_dcf = np.min(dcf_norm)
    min_dcf_threshold = thresholds[np.argmin(dcf_norm)]
    
    return min_dcf, min_dcf_threshold

def compute_auc(scores, labels):
    """計算ROC曲線下面積(AUC)
    
    Args:
        scores (numpy.ndarray): 相似度分數
        labels (numpy.ndarray): 真實標籤(1=目標, 0=非目標)
        
    Returns:
        float: AUC值
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc(fpr, tpr)
