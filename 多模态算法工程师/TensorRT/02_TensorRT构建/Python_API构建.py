import tensorrt as trt


logger = trt.Logger(trt.Logger.WARNING)  # 调 ERROR/VERBOSE 可看更多细节

def build_engine(onnx_path: str, save_path: str,
                 min_shape=(1, 3, 224, 224), opt_shape=(8, 3, 224, 224), 
                 max_shape=(32, 3, 224, 224)) -> bytes:
    builder = trt.Builder(logger)
    
    # ① 模型解析：ONNX → TRT Network
    # TensorRT 11.x 的网络默认 Strongly Typed，不再使用旧版 EXPLICIT_BATCH/弱类型流程
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(parser.get_error(i))   # 常见：不支持的 ONNX 算子
        raise RuntimeError("ONNX 解析失败")

    config = builder.create_builder_config()
    
    # ② TensorRT 11.x：精度已经由 ONNX 图中的 Tensor 类型决定
    # FP16/INT8 应先用 ModelOpt AutoCast/Quantization 生成对应 ONNX
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # workspace 上限，2 GiB
    
    # 动态 shape：一个 profile = (min, opt, max) 三元组
    profile = builder.create_optimization_profile()
    profile.set_shape("input", min_shape, opt_shape, max_shape) 
    config.add_optimization_profile(profile)
    
    # ③④ 图优化 + tactic 选 kernel 在 build 内部完成，本步最耗时
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None, "TensorRT 构建失败（调高 logger 级别排查）"

    with open(save_path, "wb") as f:
        f.write(serialized)
    return serialized

if __name__ == "__main__":
    onnx_path = "/media/user/94e6e64b-6449-4be5-ba9e-dce33db8caa4/Doucments/flj/Multi-model-Carrer-Finding/多模态算法工程师/TensorRT" + "dynamic_resnet50.onnx"
    save_path = "/media/user/94e6e64b-6449-4be5-ba9e-dce33db8caa4/Doucments/flj/Multi-model-Carrer-Finding/多模态算法工程师/TensorRT/02_TensorRT构建" + "/resnet50.trt"
    
    build_engine(onnx_path, save_path)  # 预期：engine 生成、无报错
    
    print("engine 已保存")
    
