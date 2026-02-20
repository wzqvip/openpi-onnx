#!/usr/bin/env python3
"""
修复rotary embeddings避免complex类型产生
在openpi/src中的modeling_gemma.py中应用此补丁
"""

import sys
sys.path.insert(0, '/home/taco/openpi/src')

# 替换GemmaRotaryEmbedding.forward方法
import torch
from torch import nn

class GemmaRotaryEmbeddingPatched(nn.Module):
    """修补的rotary embeddings - 避免复数操作"""
    
    def __init__(self, original_emb):
        super().__init__()
        self.inv_freq = original_emb.inv_freq
        self.attention_scaling = original_emb.attention_scaling
        self.max_seq_len_cached = original_emb.max_seq_len_cached
        self.original_max_seq_len = original_emb.original_max_seq_len
        self.config = original_emb.config
        self.rope_type = original_emb.rope_type
        self.rope_init_fn = original_emb.rope_init_fn
        self.original_inv_freq = original_emb.original_inv_freq
    
    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        避免复数操作的rotary embeddings实现
        
        原方式（产生complex128）:
            freqs = (inv_freq @ position_ids).T
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling
        
        新方式（使用实数运算）:
            freqs = (inv_freq @ position_ids).T
            cos = freqs.cos() * attention_scaling
            sin = freqs.sin() * attention_scaling
            cos = torch.cat((cos, cos), dim=-1)  # 在计算后连接
            sin = torch.cat((sin, sin), dim=-1)
        
        数学上等价，但避免了中间的复数操作
        """
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        ).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # 计算频率
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            
            # 直接计算cos和sin，避免创建复数emb
            cos = freqs.cos() * self.attention_scaling  # [batch, seq_len, dim//2]
            sin = freqs.sin() * self.attention_scaling  # [batch, seq_len, dim//2]
            
            # 在计算后拼接（而不是在emb上）
            cos = torch.cat((cos, cos), dim=-1)  # [batch, seq_len, dim]
            sin = torch.cat((sin, sin), dim=-1)  # [batch, seq_len, dim]
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def patch_rotary_embeddings():
    """
    在gemma_pytorch导入后立即应用补丁
    """
    from openpi.models_pytorch.transformers_replace.models.gemma import modeling_gemma
    
    # 保存原始类
    OriginalGemmaRotaryEmbedding = modeling_gemma.GemmaRotaryEmbedding
    
    # 创建补丁版本
    class GemmaRotaryEmbeddingFixed(OriginalGemmaRotaryEmbedding):
        """固定版本 - 避免complex类型"""
        
        @torch.no_grad()
        def forward(self, x, position_ids):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
                position_ids.shape[0], -1, 1
            ).to(x.device)
            position_ids_expanded = position_ids[:, None, :].float()

            device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
            
            with torch.autocast(device_type=device_type, enabled=False):  # Force float32
                # 关键修改: 计算freqs后分别计算cos/sin，而不是创建复数emb
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                
                # 直接计算cos和sin
                cos = freqs.cos() * self.attention_scaling
                sin = freqs.sin() * self.attention_scaling
                
                # 在计算后拼接（避免torch.cat((freqs, freqs))产生的类型问题）
                cos_full = torch.cat((cos, cos), dim=-1)
                sin_full = torch.cat((sin, sin), dim=-1)
            
            return cos_full.to(dtype=x.dtype), sin_full.to(dtype=x.dtype)
    
    # 替换全局的GemmaRotaryEmbedding
    modeling_gemma.GemmaRotaryEmbedding = GemmaRotaryEmbeddingFixed
    
    print("✅ 已应用rotary embeddings补丁")
    print("   修改: 避免复数操作，直接计算cos/sin后拼接")


if __name__ == "__main__":
    print("patchy_rotary.py - RoPE补丁脚本")
    print("在openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py中")
    print("应用以下修改:\n")
    
    print("""
# 当前实现（行号155-161）:
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

# 修改为:
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
""")
