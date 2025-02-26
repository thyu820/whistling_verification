#!/usr/bin/env python
import os
import argparse
import yaml
import shutil
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Create a new experiment")
    parser.add_argument("--name", type=str, required=True,
                        help="Experiment name")
    parser.add_argument("--based-on", type=str,
                        help="Base experiment to copy configuration from")
    parser.add_argument("--output-dir", type=str, default="experiments",
                        help="Experiments root directory")
    parser.add_argument("--config-template", type=str, default="config/exp_config.yaml",
                        help="Configuration template file")
    args = parser.parse_args()
    
    # 生成實驗ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"exp_{timestamp}"
    
    # 設置實驗目錄
    exp_dir = os.path.join(args.output_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 創建子目錄
    os.makedirs(os.path.join(exp_dir, "config"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "model"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    
    # 配置文件路徑
    config_file = os.path.join(exp_dir, "config", "exp_config.yaml")
    
    # 如果基於現有實驗
    if args.based_on:
        base_exp_dir = os.path.join(args.output_dir, args.based_on)
        base_config_file = os.path.join(base_exp_dir, "config", "exp_config.yaml")
        
        if os.path.exists(base_config_file):
            print(f"Copying configuration from {args.based_on}")
            shutil.copy(base_config_file, config_file)
        else:
            print(f"Warning: Base experiment config not found at {base_config_file}")
            create_default_config(args.config_template, config_file, args.name)
    else:
        create_default_config(args.config_template, config_file, args.name)
    
    # 創建實驗描述文件
    with open(os.path.join(exp_dir, "README.md"), 'w') as f:
        f.write(f"# Experiment: {args.name}\n\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Description\n\n")
        f.write(f"{args.name}\n\n")
        f.write("## Configuration\n\n")
        f.write("See `config/exp_config.yaml` for detailed configuration.\n\n")
        f.write("## Results\n\n")
        f.write("TBD\n")
    
    print(f"Created new experiment: {exp_id}")
    print(f"Directory: {exp_dir}")

def create_default_config(template_path, output_path, exp_name):
    """從模板創建配置文件，如果模板不存在則創建基本配置"""
    if os.path.exists(template_path):
        print(f"Using template from {template_path}")
        shutil.copy(template_path, output_path)
        
        # 更新實驗名稱
        with open(output_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            config = {}
        
        if "experiment" not in config:
            config["experiment"] = {}
        
        config["experiment"]["name"] = exp_name
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        print(f"Template not found at {template_path}, creating default configuration")
        
        # 創建默認配置
        config = {
            "experiment": {
                "name": exp_name,
                "description": "口哨聲聲者驗證實驗"
            },
            "data": {
                "input_dir": "data/processed",
                "features_dir": "data/features",
                "train_ratio": 0.7,
                "random_seed": 42
            },
            "features": {
                "mfcc": {
                    "extract": True,
                    "n_mfcc": 20,
                    "delta": True,
                    "delta_delta": True,
                    "cmvn": True
                },
                "pitch": {
                    "extract": True,
                    "min_f0": 80,
                    "max_f0": 500,
                    "method": "pyin"
                },
                "spectral": {
                    "extract": True,
                    "extract_lpc": True,
                    "extract_hnr": True
                }
            },
            "model": {
                "type": "gmm_ubm",
                "gmm_ubm": {
                    "n_components": 128,
                    "covariance_type": "diag",
                    "n_iter": 100
                },
                "ivector": {
                    "tv_dim": 400,
                    "n_iter": 10
                }
            },
            "evaluation": {
                "metrics": ["eer", "min_dcf", "auc"],
                "p_target": 0.01,
                "c_miss": 10,
                "c_fa": 1
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

if __name__ == "__main__":
    main()

