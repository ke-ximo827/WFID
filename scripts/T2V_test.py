# """
# Train a super-resolution model.
# """

# import argparse

# import torch.nn.functional as F
# from core.wandb_logger import WandbLogger
# from guided_diffusion import dist_util, logger
# from guided_diffusion.image_datasets import load_data
# from guided_diffusion.resample import create_named_schedule_sampler
# from guided_diffusion.script_util import (
#     sr_model_and_diffusion_defaults,
#     sr_create_model_and_diffusion,
#     args_to_dict,
#     add_dict_to_argparser,
# )
# from guided_diffusion.train_util import TrainLoop
# from torch.utils.data import DataLoader
# from guided_diffusion.valdata import  ValData
# import os
# import torch.distributed as dist
# import clip
# from guided_diffusion.test_diff import diffusion_test
# def main(run):
#     args = create_argparser().parse_args()
#     os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"  # set to DETAIL for runtime logging.

#     dist_util.setup_dist()
#     if(dist.get_rank()==0):

#         logger.configure(dir='./experiments/log/')
#     if(dist.get_rank()==0):
#         logger.log("creating model...")
#     model, diffusion = sr_create_model_and_diffusion(
#         **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
#     )
#     model.to(dist_util.dev())
#     model_weights=args.weights

#     model.convert_to_fp16()
#     model.load_state_dict(
#         dist_util.load_state_dict(model_weights, map_location="cpu")
#     )
#     model.eval()

#     val_data = DataLoader(ValData(args.data_dir), batch_size=1, shuffle=False, num_workers=1)  #load_superres_dataval()
#     diffusion_test(val_data,model,diffusion, './results/', run , 'test', skip_timesteps=40, iter=0)


# def create_argparser():
#     defaults = dict(
#         data_dir='./data/test_process/TH/',
#         weights="./weights/model099997.pt",
#         use_fp16=False,

#     )
#     defaults.update(sr_model_and_diffusion_defaults())
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser
# if __name__ == "__main__":
#     run=WandbLogger()
#     main(run)


"""
T2V测试脚本 - 完全修复版
关键：use_fp16=False，覆盖默认配置
"""
import argparse
import torch.nn.functional as F
from core.wandb_logger import WandbLogger
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from torch.utils.data import DataLoader
from guided_diffusion.valdata import  ValData
import os
import torch.distributed as dist
import clip
from guided_diffusion.test_diff import diffusion_test


def main(run):
    args = create_argparser().parse_args()
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    dist_util.setup_dist()
    
    if dist.get_rank() == 0:
        logger.configure(dir='./experiments/log/')
        logger.log("creating model...")
    
    # 创建模型
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    
    # 加载权重
    model_weights = args.weights
    if dist.get_rank() == 0:
        logger.log(f"loading weights: {model_weights}")
        logger.log(f"use_fp16: {args.use_fp16}")  # ← 打印确认
    
    model.load_state_dict(
        dist_util.load_state_dict(model_weights, map_location="cpu"),
        strict=False
    )
    
    # ✓ 根据配置决定是否转FP16
    if args.use_fp16:
        if dist.get_rank() == 0:
            logger.log("converting to FP16...")
        model.convert_to_fp16()
    else:
        if dist.get_rank() == 0:
            logger.log("using FP32 (no conversion)")
    
    model.eval()
    
    # 加载测试数据
    if dist.get_rank() == 0:
        logger.log("loading test data...")
    
    val_data = DataLoader(
        ValData(args.data_dir), 
        batch_size=32,
        shuffle=False, 
        num_workers=1
    )
    
    if dist.get_rank() == 0:
        logger.log(f"starting test with {len(val_data)} samples...")
    
    # 测试
    diffusion_test(
        val_data,
        model,
        diffusion, 
        './results/', 
        run, 
        'test', 
        skip_timesteps=0,
        iter=0
    )
    
    if dist.get_rank() == 0:
        logger.log("test completed!")


def create_argparser():
    # ✓ 关键：先设置use_fp16=False，再update默认值
    defaults = dict(
        data_dir='./data/test_process/TH/',
        weights="./weights/model099999.pt",
        use_fp16=False,  # ← 覆盖script_util的默认True
        use_arcface=True,
        use_wavelet=True,
        use_fourier=True,
        arcface_weight_path="./weights_T/arcface_thermal_finetuned_best.pth",
    )
    
    # 后update，但我们的use_fp16已经设置了
    defaults.update(sr_model_and_diffusion_defaults())
    
    # 再次确保use_fp16=False（防止被覆盖）
    defaults['use_fp16'] = False  # ← 二次确保
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    run = WandbLogger()
    main(run)