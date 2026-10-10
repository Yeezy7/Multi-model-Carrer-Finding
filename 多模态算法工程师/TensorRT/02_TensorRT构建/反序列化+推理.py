import numpy as np
import tensorrt as trt
import torch

logger = trt.Logger(trt.Logger.WARNING)  # 调 ERROR/VERBOSE 可看更多细节

def load_engine(path: str) -> trt.ICudaEngine:
    """加载 TensorRT engine"""
    runtime = trt.Runtime(logger)
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine is not None, "加载失败：engine 与当前 GPU/CUDA/TRT 版本不匹配"
    return engine

class TRTInfer:
    def __init__(self, engine_path: str):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        # 枚举 I/O tensor 名与输入输出分组（TRT 8.5+ 的 get_tensor_name 系列 API）
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if n not in self.inputs]
        
    def __call__(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """推理：输入 feed = {input_name: np.ndarray}"""
        for name, arr in feed.items():  # ① 动态 shape：按实际输入设置（须在 min/max 内）
            self.context.set_input_shape(name, arr.shape)
            
        # ② 按 engine 声明的 dtype/shape 分配 device 端 buffer，③ host → device 拷贝
        t_dtype = {trt.float32: torch.float32, trt.float16: torch.float16,
                   trt.int32: torch.int32, trt.int8: torch.int8}
        bufs = {}
        for name in self.names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = t_dtype[self.engine.get_tensor_dtype(name)]
            bufs[name] = torch.empty(size=shape, dtype=dtype, device="cuda")
        for name, arr in feed.items():
            bufs[name].copy_(torch.from_numpy(arr.astype(np.float32)))
        
        # ④ 执行（TRT 10：先逐 tensor 绑定 device 地址，再传 stream 句柄）
        stream = torch.cuda.current_stream().cuda_stream
        for n in self.names:
            self.context.set_tensor_address(n, int(bufs[n].data_ptr()))
        self.context.execute_async_v3(stream)
        
        return {n: bufs[n].cpu().numpy() for n in self.outputs}  # ⑤ device → host（同 stream 同步）

if __name__ == "__main__":
    engine_path = "/media/user/94e6e64b-6449-4be5-ba9e-dce33db8caa4/Doucments/flj/Multi-model-Carrer-Finding/多模态算法工程师/TensorRT/02_TensorRT构建/resnet50.trt"
    model = TRTInfer(engine_path)
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    y = model({"input": x})["output"]
    print(y.shape)                       # 预期 (1, 1000)，分类 logits
    