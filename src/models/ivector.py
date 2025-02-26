import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

class IVectorModel:
    """i-vector模型用於聲者驗證
    
    實現了簡化版的i-vector提取，包括UBM訓練、統計量計算和總變異性矩陣訓練。
    """
    
    def __init__(self, n_components=128, tv_dim=400, n_iter=10, random_state=None):
        """初始化i-vector模型
        
        Args:
            n_components (int): UBM組件數量
            tv_dim (int): 總變異性子空間維度
            n_iter (int): 訓練迭代次數
            random_state (int): 隨機種子
        """
        self.n_components = n_components
        self.tv_dim = tv_dim
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.ubm = None  # 通用背景模型
        self.T = None    # 總變異性矩陣
        self.ivectors = {}  # 說話人i-vectors字典
        
    def train_ubm(self, features):
        """訓練通用背景模型
        
        Args:
            features (numpy.ndarray): 所有說話人的特徵，形狀為(n_samples, n_features)
            
        Returns:
            self: 返回模型實例
        """
        self.ubm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            max_iter=100,
            random_state=self.random_state
        )
        self.ubm.fit(features)
        return self
    
    def compute_statistics(self, features):
        """計算用於i-vector提取的統計量
        
        Args:
            features (numpy.ndarray): 輸入特徵，形狀為(n_samples, n_features)
            
        Returns:
            tuple: (N, F)，分別為零階和中心化一階統計量
        """
        # 計算後驗概率
        posteriors = self.ubm.predict_proba(features)
        
        # 計算零階統計量
        N = np.sum(posteriors, axis=0)  # 形狀為(n_components,)
        
        # 計算一階統計量
        F = np.zeros((self.n_components, features.shape[1]))
        for c in range(self.n_components):
            F[c] = np.sum(posteriors[:, c:c+1] * features, axis=0) - N[c] * self.ubm.means_[c]
        
        return N, F
    
    def train_tv_matrix(self, all_features, speaker_ids):
        """訓練總變異性矩陣
        
        這是一個簡化版實現，實際上應該使用期望最大化(EM)算法
        
        Args:
            all_features (list): 特徵列表，每個元素是一個說話人的所有特徵
            speaker_ids (list): 對應的說話人標識符列表
            
        Returns:
            self: 返回模型實例
        """
        # 確保UBM已訓練
        if self.ubm is None:
            raise ValueError("UBM must be trained before TV matrix training")
        
        # 計算所有說話人的統計量
        all_N = []
        all_F = []
        for features in all_features:
            N, F = self.compute_statistics(features)
            all_N.append(N)
            all_F.append(F.flatten())  # 展平為一維向量
        
        # 將統計量堆疊為矩陣
        all_F = np.vstack(all_F)  # 形狀為(n_utterances, n_components*n_features)
        
        # 使用PCA初始化總變異性矩陣
        # 這是一個簡化，完整實現應該使用EM算法
        pca = PCA(n_components=self.tv_dim, random_state=self.random_state)
        pca.fit(all_F)
        
        # 總變異性矩陣
        self.T = pca.components_.T  # 形狀為(n_components*n_features, tv_dim)
        
        # 提取每個說話人的i-vector
        for i, speaker_id in enumerate(speaker_ids):
            features = all_features[i]
            self.ivectors[speaker_id] = self.extract_ivector(features)
        
        return self
    
    def extract_ivector(self, features):
        """提取i-vector
        
        Args:
            features (numpy.ndarray): 輸入特徵，形狀為(n_samples, n_features)
            
        Returns:
            numpy.ndarray: 提取的i-vector，形狀為(tv_dim,)
        """
        if self.T is None:
            raise ValueError("TV matrix must be trained before i-vector extraction")
        
        # 計算統計量
        N, F = self.compute_statistics(features)
        
        # 計算後驗協方差
        feature_dim = features.shape[1]
        I = np.eye(self.tv_dim)
        
        # 計算精度矩陣（協方差的逆）
        precision = np.zeros((self.tv_dim, self.tv_dim))
        
        for c in range(self.n_components):
            # 簡化處理，假設UBM協方差是對角的
            Sigma_c_inv = 1.0 / self.ubm.covariances_[c]
            Sigma_c_inv = np.diag(Sigma_c_inv)
            
            # 提取對應組件的T子矩陣
            T_c = self.T[c*feature_dim:(c+1)*feature_dim, :]
            
            # 更新精度矩陣
            precision += N[c] * (T_c.T @ Sigma_c_inv @ T_c)
        
        # 添加單位矩陣（先驗）
        posterior_covar = np.linalg.inv(I + precision)
        
        # 計算i-vector
        T_Sigma_inv_F = np.zeros(self.tv_dim)
        
        for c in range(self.n_components):
            Sigma_c_inv = 1.0 / self.ubm.covariances_[c]
            Sigma_c_inv = np.diag(Sigma_c_inv)
            
            T_c = self.T[c*feature_dim:(c+1)*feature_dim, :]
            F_c = F[c]
            
            T_Sigma_inv_F += T_c.T @ Sigma_c_inv @ F_c
        
        ivector = posterior_covar @ T_Sigma_inv_F
        
        # 長度歸一化
        norm = np.sqrt(np.sum(ivector**2))
        if norm > 0:
            ivector = ivector / norm
        
        return ivector
    
    def score(self, features, speaker_id):
        """計算特徵與說話人i-vector的相似度分數
        
        使用餘弦相似度作為評分標準
        
        Args:
            features (numpy.ndarray): 輸入特徵，形狀為(n_samples, n_features)
            speaker_id (str): 說話人標識符
            
        Returns:
            float: 相似度分數
        """
        if speaker_id not in self.ivectors:
            raise ValueError(f"No i-vector found for speaker {speaker_id}")
        
        # 提取測試i-vector
        test_ivector = self.extract_ivector(features)
        
        # 獲取註冊的i-vector
        enroll_ivector = self.ivectors[speaker_id]
        
        # 計算餘弦相似度
        similarity = np.dot(test_ivector, enroll_ivector) / \
                    (np.linalg.norm(test_ivector) * np.linalg.norm(enroll_ivector))
        
        return similarity
    
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
            IVectorModel: 加載的模型實例
        """
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
