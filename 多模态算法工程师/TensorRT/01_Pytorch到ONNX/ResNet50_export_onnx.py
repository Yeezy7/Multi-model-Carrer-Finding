import os
import time
import onnx
import torch
import torchvision.models as models
import numpy as np

# 指定下载/缓存目录
cache_dir = "/media/user/94e6e64b-6449-4be5-ba9e-dce33db8caa4/Doucments/flj/Multi-model-Carrer-Finding/pre-trained-model"
os.makedirs(cache_dir, exist_ok=True)

# 让 torchvision / pytorch 使用这个目录
torch.hub.set_dir(cache_dir)

weights = models.ResNet50_Weights.DEFAULT
checkpoint_dir = os.path.join(cache_dir, "ResNet50")
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "resnet50-11ad3fa6.pth")


def load_resnet50_state_dict():
    for attempt in range(1, 4):
        if os.path.exists(checkpoint_path):
            print(f"发现旧缓存文件，先删除：{checkpoint_path}")
            os.remove(checkpoint_path)

        try:
            print(f"开始下载权重，尝试 {attempt}/3 ...")
            return torch.hub.load_state_dict_from_url(
                weights.url,
                model_dir=checkpoint_dir,
                progress=True,
                check_hash=False,
            )
        except Exception as e:
            print(f"第 {attempt} 次下载失败：{e}")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            if attempt < 3:
                time.sleep(2)
            else:
                raise


def export_fixed_batch(model, input):
    # 导出为 ONNX格式
    torch.onnx.export(
        model,
        input,
        "fixed_resnet50.onnx",
        export_params=True,          # 权重写入 ONNX
        opset_version=17,            # 建议 13 或 17
        do_constant_folding=True,    # 常量折叠优化
        input_names=["input"],
        output_names=["output"],
    )
    print("exported fixed_resnet50.onnx")

def export_dynamic_batch(model, input):
    # 导出为 ONNX格式
    torch.onnx.export(
        model,
        input,
        "dynamic_resnet50.onnx",
        export_params=True,          # 权重写入 ONNX
        opset_version=17,            # 建议 13 或 17
        do_constant_folding=True,    # 常量折叠优化
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"}, 
            "output": {0: "batch_size"}
        }, # 动态 batch
    )
    print("exported dynamic_resnet50.onnx")
    
    # 检查 ONNX 是否合法
    onnx_model = onnx.load("dynamic_resnet50.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX check passed")
        

def ort_check():
    """ONNX Runtime 检查"""
    import onnxruntime as ort
    ort_session = ort.InferenceSession(   # 检查是否能加载
        "dynamic_resnet50.onnx",
        providers=["CPUExecutionProvider"],
    )
    print("输入：")
    for i in ort_session.get_inputs():
        print(i.name, i.shape, i.type)

    print("输出：")
    for o in ort_session.get_outputs():
        print(o.name, o.shape, o.type)

    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    ort_out = ort_session.run(None, {"input": x})[0]
    print(ort_out.shape)
    

if __name__ == "__main__":
    # 先创建空模型，再显式下载权重到指定目录
    model = models.resnet50(weights=None)
    state_dict = load_resnet50_state_dict()
    model.load_state_dict(state_dict)

    # 推理模式
    model.eval()
    
    # 固定 batch=1，输入 3x224x224
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # # 导出 固定 batch onnx
    # export_fixed_batch(model, dummy_input)

    # # 导出 动态 batch onnx
    # export_dynamic_batch(model, dummy_input)
    
    # ONNX Runtime 检查
    ort_check()
