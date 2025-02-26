import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ..evaluation.metrics import compute_eer, compute_dcf, compute_auc

def cross_validate(model_class, features, speaker_ids, n_splits=5, random_state=None, **model_params):
    """執行交叉驗證評估
    
    Args:
        model_class: 模型類(如GMMUBMModel)
        features (list): 特徵列表，每個元素對應一個話者的所有特徵
        speaker_ids (list): 對應的說話人標識符列表
        n_splits (int): 交叉驗證折數
        random_state (int): 隨機種子
        **model_params: 模型參數
        
    Returns:
        pandas.DataFrame: 評估結果
    """
    # 初始化結果列表
    results = []
    
    # 初始化KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 預處理數據
    all_features = np.vstack(features)
    all_speakers = np.concatenate([[id] * len(feat) for id, feat in zip(speaker_ids, features)])
    
    # 執行交叉驗證
    for fold, (train_idx, test_idx) in enumerate(kf.split(range(len(speaker_ids)))):
        print(f"Processing fold {fold+1}/{n_splits}")
        
        # 切分訓練集和測試集
        train_speakers = [speaker_ids[i] for i in train_idx]
        test_speakers = [speaker_ids[i] for i in test_idx]
        
        # 提取訓練數據
        train_features_list = [features[i] for i in train_idx]
        train_features_all = np.vstack([features[i] for i in train_idx])
        
        # 初始化模型
        model = model_class(**model_params)
        
        # 訓練模型
        if hasattr(model, 'train_ubm'):
            model.train_ubm(train_features_all)
            
            # 適應說話人模型
            for i, speaker_id in enumerate(train_speakers):
                if hasattr(model, 'adapt_speaker_model'):
                    model.adapt_speaker_model(speaker_id, train_features_list[i])
                
            # 如果是i-vector模型，還需要訓練TV矩陣
            if hasattr(model, 'train_tv_matrix'):
                model.train_tv_matrix(train_features_list, train_speakers)
        else:
            # 簡單模型直接訓練
            model.fit(train_features_all)
        
        # 評估模型
        all_scores = []
        all_labels = []
        
        # 對每個測試說話人生成正負樣本對
        for i, test_speaker in enumerate(test_speakers):
            # 正樣本: 正確說話人
            test_features = features[test_idx[i]]
            
            # 每個特徵向量獨立評分
            for feat in test_features:
                # 展開為2D數組
                feat_2d = feat.reshape(1, -1)
                
                # 正樣本
                if hasattr(model, 'score'):
                    score = model.score(feat_2d, test_speaker)
                else:
                    # 簡單模型使用log_likelihood作為分數
                    score = model.score_samples(feat_2d)[0]
                    
                all_scores.append(score)
                all_labels.append(1)  # 正樣本標籤
                
                # 負樣本: 對每個其他測試說話人
                for j, other_speaker in enumerate(test_speakers):
                    if other_speaker != test_speaker:
                        if hasattr(model, 'score'):
                            score = model.score(feat_2d, other_speaker)
                        else:
                            # 簡單模型時不生成負樣本
                            continue
                            
                        all_scores.append(score)
                        all_labels.append(0)  # 負樣本標籤
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # 計算評估指標
        eer, eer_threshold = compute_eer(all_scores, all_labels)
        mindcf, dcf_threshold = compute_dcf(all_scores, all_labels)
        auc_value = compute_auc(all_scores, all_labels)
        
        # 添加到結果
        results.append({
            'fold': fold + 1,
            'eer': eer,
            'min_dcf': mindcf,
            'auc': auc_value,
            'eer_threshold': eer_threshold,
            'dcf_threshold': dcf_threshold
        })
    
    # 創建數據框
    results_df = pd.DataFrame(results)
    
    # 添加平均結果
    mean_results = {
        'fold': 'Average',
        'eer': results_df['eer'].mean(),
        'min_dcf': results_df['min_dcf'].mean(),
        'auc': results_df['auc'].mean(),
        'eer_threshold': results_df['eer_threshold'].mean(),
        'dcf_threshold': results_df['dcf_threshold'].mean()
    }
    results_df = results_df.append(mean_results, ignore_index=True)
    
    return results_df
