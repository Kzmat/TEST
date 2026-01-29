# scalefl/models/minigpt_med.py

import torch
import torch.nn as nn
from scalefl.core.model import BaseModel
from peft import LoraConfig, get_peft_model, TaskType
import sys
import os

# 动态导入MiniGPT-Med
def import_minigpt_med():
    """动态导入MiniGPT-Med模块"""
    minigpt_path = os.environ.get('MINIGPT_MED_PATH', './MiniGPT-Med')
    if minigpt_path not in sys.path:
        sys.path.append(minigpt_path)
    
    try:
        from models.minigpt4.models.minigpt4 import MiniGPT4
        from models.minigpt4.common.config import Config
        from models.minigpt4.common.registry import registry
        return MiniGPT4, Config, registry
    except ImportError as e:
        raise ImportError(f"请先克隆MiniGPT-Med到指定路径: {e}")

class MiniGPTMedFLModel(BaseModel):
    """ScaleFL适配的MiniGPT-Med模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 导入MiniGPT-Med
        MiniGPT4, Config, registry = import_minigpt_med()
        
        # 初始化原始模型
        self.base_model = self._load_base_model(config, MiniGPT4, Config, registry)
        
        # 应用LoRA
        self.model = self._apply_lora(self.base_model, config)
        
        # 设备信息
        self.device_info = self._get_device_info()
    
    def _load_base_model(self, config, MiniGPT4, Config, registry):
        """加载MiniGPT-Med基础模型"""
        # 使用MiniGPT-Med原有的加载逻辑
        model_config = config.get('model_config', {})
        
        # 创建模型
        model = MiniGPT4.from_config(model_config)
        
        # 加载预训练权重
        if config.get('pretrained_path'):
            checkpoint = torch.load(config['pretrained_path'], map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
        
        return model
    
    def _apply_lora(self, model, config):
        """应用自适应LoRA"""
        lora_config = self._get_adaptive_lora_config(config)
        return get_peft_model(model, lora_config)
    
    def _get_adaptive_lora_config(self, config):
        """根据设备资源获取LoRA配置"""
        memory_gb = self._get_available_memory()
        
        # 自适应配置
        if memory_gb >= 16:
            r, alpha, targets = 16, 32, ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif memory_gb >= 8:
            r, alpha, targets = 8, 16, ["q_proj", "v_proj", "k_proj", "o_proj"]
        else:
            r, alpha, targets = 4, 8, ["q_proj", "v_proj"]
        
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.1,
            target_modules=targets
        )
    
    def forward(self, batch):
        """前向传播"""
        return self.model(batch)
    
    def get_trainable_params(self):
        """获取可训练参数（仅LoRA）"""
        return {k: v for k, v in self.model.named_parameters() if v.requires_grad}
    
    def get_lora_state_dict(self):
        """获取LoRA参数"""
        return {k: v for k, v in self.model.state_dict().items() if 'lora_' in k}
    
    def load_lora_state_dict(self, state_dict):
        """加载LoRA参数"""
        self.model.load_state_dict(state_dict, strict=False)