# 口哨聲聲者驗證系統

這個儲存庫用於口哨聲生物特徵識別研究，包含完整的數據處理管線、特徵提取方法、模型訓練和評估工具。

## 安裝

```bash
# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
pip install -e .  # 以開發模式安裝此包
```

## 使用流程

1. 數據預處理: `python scripts/run_preprocessing.py`
2. 特徵提取: `python scripts/extract_features.py`
3. 模型訓練: `python scripts/train_model.py`
4. 模型評估: `python scripts/evaluate.py`

詳細使用說明請參閱 `docs/` 目錄中的文檔。
