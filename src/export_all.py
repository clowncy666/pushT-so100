import torch
import os
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

def main():
    print("=========================================")
    print("📦 [大一统] 终极模型导出：视觉与动作双核")
    print("=========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DiffusionPolicy.from_pretrained("qian1dqs/so100-pusht")
    policy.eval()
    policy.to(device)

    # 强制获取操作系统的绝对路径
    out_dir = os.path.abspath("onnx_models")
    os.makedirs(out_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. 导出视觉大脑 (Vision Encoder)
    # ---------------------------------------------------------
    print("\n[1/2] 正在提取视觉大脑...")
    vision_encoder = policy.diffusion.rgb_encoder
    dummy_image = torch.randn(1, 3,224, 224).to(device) # PushT 默认 96x96
    vision_onnx_path = os.path.join(out_dir, "vision_encoder.onnx")
    
    torch.onnx.export(
        vision_encoder, dummy_image, vision_onnx_path,
        export_params=True, opset_version=18, do_constant_folding=True,
        input_names=['image'], output_names=['vision_feature']
    )
    print(f"  ✅ 视觉大脑就绪: {vision_onnx_path}")

    # ---------------------------------------------------------
    # 2. 导出去噪小脑 (U-Net)
    # ---------------------------------------------------------
    print("\n[2/2] 正在提取去噪小脑...")
    unet = policy.diffusion.unet
    dummy_noisy_action = torch.randn(1, 16, 2).to(device) 
    dummy_timestep = torch.tensor([10], dtype=torch.long).to(device) 
    dummy_global_cond = torch.randn(1, 266).to(device)
    unet_onnx_path = os.path.join(out_dir, "noise_unet.onnx")
    
    torch.onnx.export(
        unet, (dummy_noisy_action, dummy_timestep, dummy_global_cond), unet_onnx_path,
        export_params=True, opset_version=18, do_constant_folding=True,
        input_names=['sample', 'timestep', 'global_cond'], output_names=['noise_pred']
    )
    print(f"  ✅ 去噪小脑就绪: {unet_onnx_path}")

    print("\n🎉 全部 ONNX 文件已完美生成！可以开始 TensorRT 锻造了！")

if __name__ == "__main__":
    main()