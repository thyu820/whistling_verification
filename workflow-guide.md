# 口哨聲聲者驗證系統使用指南

本指南將帶您完成口哨聲聲者驗證實驗的完整流程，從環境設置到結果評估。

## 1. 環境設置

首先，克隆儲存庫並設置環境：

```bash
# 克隆儲存庫（假設您已建立GitHub儲存庫）
git clone https://github.com/your-username/whistling_verification.git
cd whistling_verification

# 創建並激活虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
pip install -e .  # 以開發模式安裝此包
```

## 2. 數據收集與組織

在開始實驗之前，您需要收集口哨聲數據並按以下方式組織：

1. 將原始口哨聲錄音放入 `data/raw/` 目錄
2. 創建元數據文件 `data/metadata/speakers.csv`，包含說話人信息

原始音頻命名格式建議：`{speaker_id}_{session_id}_{recording_id}.wav`

元數據文件格式示例：
```csv
speaker_id,gender,age,recording_env
001,M,25,quiet
002,F,30,quiet
003,M,22,noisy
```

## 3. 音頻預處理

預處理包括靜音消除、降噪和切分：

```bash
# 使用默認配置
python scripts/run_preprocessing.py

# 或指定配置文件
python scripts/run_preprocessing.py --config config/preprocess_config.yaml

# 或覆蓋輸入/輸出目錄
python scripts/run_preprocessing.py --input-dir data/raw --output-dir data/processed
```

預處理後，切分的口哨片段將保存在 `data/processed/{speaker_id}/` 目錄中。

## 4. 特徵提取

從預處理後的音頻中提取聲學特徵：

```bash
# 使用默認配置
python scripts/extract_features.py

# 或指定配置文件
python scripts/extract_features.py --config config/feature_config.yaml
```

這將提取MFCC、音高和頻譜特徵，並保存到 `data/features/` 目錄中。

## 5. 創建實驗

使用腳本創建新實驗：

```bash
# 創建新實驗
python scripts/create_experiment.py --name "GMM-UBM基準實驗"

# 或基於現有實驗創建
python scripts/create_experiment.py --name "調整組件數實驗" --based-on exp_20250226_123456
```

這將創建一個包含配置文件和目錄結構的新實驗目錄。

## 6. 模型訓練

使用提取的特徵訓練聲者驗證模型：

```bash
# 訓練模型
python scripts/train_model.py --exp-dir experiments/exp_20250226_123456

# 或指定特徵文件
python scripts/train_model.py --exp-dir experiments/exp_20250226_123456 --features data/features/all_features.csv
```

模型將保存在實驗目錄的 `model/` 子目錄中。

## 7. 模型評估

評估模型性能並生成DET曲線：

```bash
python scripts/evaluate.py --exp-dir experiments/exp_20250226_123456
```

評估結果將保存在實驗目錄的 `results/` 和 `plots/` 子目錄中。

## 8. 比較多個實驗

在Jupyter筆記本中使用以下代碼比較多個實驗的結果：

```python
from src.utils.visualization import compare_experiments

# 比較多個實驗
compare_experiments(
    ["exp_20250226_123456", "exp_20250226_234567"],
    metrics=["eer", "min_dcf", "auc"],
    output_dir="results/comparison"
)
```

## 常見問題與解決方案

### 問題：預處理未檢測到口哨片段

**解決方案**：調整 `preprocess_config.yaml` 中的 `top_db` 參數。如果口哨聲較弱，嘗試將值從默認的20降低到10-15。

### 問題：特徵提取過程中出現錯誤

**解決方案**：檢查音頻格式，確保所有音頻文件都是WAV格式，採樣率一致。也可以嘗試在預處理階段添加重採樣步驟。

### 問題：GMM訓練不收斂

**解決方案**：嘗試減少 `n_components` 參數，或增加 `n_iter` 參數。確保訓練數據量足夠大。

### 問題：驗證性能差

**解決方案**：
1. 嘗試不同的特徵組合（調整 `feature_config.yaml`）
2. 調整模型參數（增加GMM組件數）
3. 提高音頻質量或收集更多訓練數據

## 自定義實驗

### 調整預處理參數

編輯 `config/preprocess_config.yaml` 修改以下參數：

- 頻率過濾範圍（`lowcut` 和 `highcut`）
- 振幅歸一化目標（`target_dBFS`）
- 靜音閾值（`top_db`）

### 調整特徵參數

編輯 `config/feature_config.yaml` 修改以下參數：

- MFCC係數數量（`n_mfcc`）
- 基頻範圍（`min_f0` 和 `max_f0`）
- 是否提取LPC和HNR特徵

### 嘗試不同模型

編輯 `config/model_config.yaml` 修改以下參數：

- 模型類型（`type`: "gmm_ubm" 或 "ivector"）
- GMM組件數（`n_components`）
- i-vector維度（`tv_dim`）

## 下一步建議

1. **優化特徵組合**：通過實驗確定最適合口哨聲的特徵組合
2. **改進預處理**：開發專門針對口哨聲的去噪算法
3. **擴展模型**：實現x-vector或深度學習模型
4. **構建演示系統**：創建實時口哨聲驗證演示
