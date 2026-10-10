import torch
import torch.nn as nn 
import torch.nn.functional as F


class EMA:
    """指数移动平均：维护 shadow weights，评估时切换到 EMA 权重"""
    
    def __init__(self, model, decay=0.999):
        self.decay = decay  # EMA 衰减系数 λ，越接近 1 越平滑
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}  # 保存 EMA 权重
        self.backup = {}   # 保存原始权重，评估后恢复
        
    @torch.no_grad()
    def update(self, model):
        # 每个 step 调一次：θ_ema ← λ·θ_ema + (1-λ)·θ
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
        
    @torch.no_grad()
    def apply_shadow(self, model):
        # 评估前：备份当前权重，载入 EMA 权重
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        
    @torch.no_grad()
    def restore(self, model):
        # 评估后：恢复原始权重
        model.load_state_dict(self.backup)
    
    def apply_decay(self, step, warmup=1000):
         # 训练早期用更小的 decay（warmup），防止早期噪声参数被过分平均
         self.decay = min(self.decay, (1 + step) / (10 + step))
         
         
if __name__ == "__main__":
    
    """完整训练循环：EMA与评估切换"""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    ema = EMA(model, decay=0.999)
    
    for step in range(200):
        ema.apply_decay(step)                    # warmup 期 decay 从小往大涨
        x = torch.randn(32, 16)
        y = torch.randint(0, 4, (32,))
        optim.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optim.step()
        ema.update(model)                        # ★ 关键：优化器 step 后更新 EMA
    
    # 评估： 切到 EMA 权重 → 推理 → 切回
    ema.apply_shadow(model)                     # ★ 关键：评估前切换到 EMA 权重
    with torch.no_grad():
        x_test = torch.randn(16, 16)
        pred_ema = model(x_test)                # 用 EMA 权重推理
    ema.restore(model)                          # ★ 关键：评估后恢复原始权重
    pred_online = model(x_test)                 # 用在线权重推理
    diff = (pred_ema - pred_online).abs().max().item()
    print(pred_ema.shape, diff)                 # torch.Size([16, 4])，diff 如 0.023（两者不同）