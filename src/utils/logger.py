import logging
import os
from datetime import datetime

def setup_logger(name, log_dir="logs", level=logging.INFO):
    """設置並返回一個格式化的logger實例
    
    Args:
        name (str): Logger名稱
        log_dir (str): 日誌文件目錄
        level (int): 日誌等級
        
    Returns:
        logging.Logger: 配置好的logger實例
    """
    # 確保日誌目錄存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 創建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重複處理器
    if logger.handlers:
        return logger
    
    # 創建控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 創建文件處理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # 創建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加處理器到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
