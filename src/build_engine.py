import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_path, engine_path):
    print("="*50)
    print(f"🔥 开始将 {onnx_path} 投入 TensorRT 熔炉...")
    print("="*50)
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  ⚡ 检测到 GPU 支持，已开启 FP16 极限加速！")
    else:
        print("  ⚠️ 当前显卡不支持快速 FP16，使用 FP32 编译。")

    print("  📜 正在解析 ONNX 结构图...")
    # 【核心修复】：使用 parse_from_file 而不是在内存中 parse
    # 这样 TensorRT 就能顺藤摸瓜，在同一目录下找到附属的 .data 权重文件！
    if not parser.parse_from_file(onnx_path):
        print(f"❌ ONNX 解析失败，报错如下：")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return

    print("  ⏳ 正在进行算子融合与引擎编译... (请耐心等待，这可能需要几分钟！)")
    engine_bytes = builder.build_serialized_network(network, config)
    
    if engine_bytes is None:
        print("❌ 引擎编译失败！")
        return

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"  ✅ 锻造成功！已生成极速引擎: {engine_path}\n")

if __name__ == "__main__":
    os.makedirs("onnx_models", exist_ok=True)
    build_engine("onnx_models/vision_encoder.onnx", "onnx_models/vision_encoder.engine")
    build_engine("onnx_models/noise_unet.onnx", "onnx_models/noise_unet.engine")