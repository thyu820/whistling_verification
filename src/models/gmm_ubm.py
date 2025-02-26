import os
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture

class GMMUBMModel:
    """GMM-UBM模型用於聲者驗證
    
    通用背景模型(UBM)作為所有說話人的通用表示，
    每個說話人模型通過適應UBM而成。
    """
    
    def __init__(self, n_components=128, covariance_type='diag', n_iter=100, random_state=None):
        """初始化GMM-UBM模型
        
        Args:
            n_components (int): GMM組件數量
            covariance_type (str): 協方差類型 ('full', 'tied', 'diag', 'spherical')
            n_iter (int): EM算法最大迭代次數
            random_state (int): 隨機種子
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.ubm = None
        self.speaker_models = {}
        
    def train_ubm(self, features):
        """訓練通用背景模型
        
        Args:
            features (numpy.ndarray): 所有說話人的特徵，形狀為(n_samples, n_features)
            
        Returns:
            self: 返回模型實例
        """
        self.ubm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.n_iter,
            random_state=self.random_state
        )
        self.ubm.fit(features)
        return self
    
    def adapt_speaker_model(self, speaker_id, speaker_features, relevance_factor=16.0):
        """基於UBM適應生成說話人模型
        
        使用最大後驗(MAP)適應方法，重點適應均值參數
        
        Args:
            speaker_id (str): 說話人標識符
            speaker_features (numpy.ndarray): 說話人特徵，形狀為(n_samples, n_features)
            relevance_factor (float): 控制原始UBM與新數據間的平衡
            
        Returns:
            GaussianMixture: 適應後的說話人模型
        """
        if self.ubm is None:
            raise ValueError("UBM must be trained before adapting speaker models")
        
        # 複製UBM參數
        speaker_model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=0  # 不需要迭代
        )
        
        # 初始化參數
        speaker_model.weights_ = self.ubm.weights_.copy()
        speaker_model.means_ = self.ubm.means_.copy()
        speaker_model.covariances_ = self.ubm.covariances_.copy()
        speaker_model.precisions_cholesky_ = self.ubm.precisions_cholesky_.copy()
        
        # 計算說話人數據的統計量
        # 1. 計算每個樣本對應於每個組件的後驗概率
        posteriors = speaker_model.predict_proba(speaker_features)
        
        # 2. 計算足夠統計量
        n_samples = speaker_features.shape[0]
        n_i = np.sum(posteriors, axis=0) + 1e-10  # 避免除以零
        
        # 3. 計算適應因子
        adaptation_factor = n_i / (n_i + relevance_factor)
        
        # 4. 適應均值
        ubm_means = speaker_model.means_
        adapted_means = np.zeros_like(ubm_means)
        
        for c in range(self.n_components):
            # 計算組件c的特徵加權均值
            adapted_means[c] = np.sum(posteriors[:, c:c+1] * speaker_features, axis=0) / n_i[c]
            
            # 融合UBM和新數據
            speaker_model.means_[c] = adaptation_factor[c] * adapted_means[c] + \
                                     (1 - adaptation_factor[c]) * ubm_means[c]
        
        # 保存適應後的模型
        self.speaker_models[speaker_id] = speaker_model
        
        return speaker_model
    
    def score(self, features, speaker_id):
        """計算特徵與說話人模型的相似度分數
        
        Args:
            features (numpy.ndarray): 輸入特徵，形狀為(n_samples, n_features)
            speaker_id (str): 說話人標識符
            
        Returns:
            float: 相似度分數 (對數似然比)
        """
        if speaker_id not in self.speaker_models:
            raise ValueError(f"No model found for speaker {speaker_id}")
        
        # 計算說話人模型對特徵的對數似然
        speaker_ll = self.speaker_models[speaker_id].score_samples(features)
        
        # 計算UBM對特徵的對數似然
        ubm_ll = self.ubm.score_samples(features)
        
        # 計算對數似然比的平均值
        llr = np.mean(speaker_ll - ubm_ll)
        
        return llr
    
    def verify(self, features, speaker_id, threshold):
        """驗證說話人身份
        
        Args:
            features (numpy.ndarray): 輸入特徵，形狀為(n_samples, n_features)
            speaker_id (str): 聲稱的說話人標識符
            threshold (float): 接受/拒絕的決策閾值
            
        Returns:
            tuple: (決策結果, 分數)，決策結果為布爾值
        """
        score = self.score(features, speaker_id)
        decision = score >= threshold
        
        return decision, score
    
    def save(self, file_path):
        """保存模型到文件
        
        Args:
            file_path (str): 目標文件路徑
            
        Returns:
            bool: 是否成功保存
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    @classmethod
    def load(cls, file_path):
        """從文件加載模型
        
        Args:
            file_path (str): 模型文件路徑
            
        Returns:
            GMMUBMModel: 加載的模型實例
        """
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
