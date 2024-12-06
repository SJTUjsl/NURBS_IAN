import os
import sys
import logging
import json
import torch
import torch.distributed as dist
from glob import glob
from time import time, strftime, localtime
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader
import monai
from monai.data import CacheDataset, PersistentDataset, ThreadDataLoader, decollate_batch, DistributedSampler
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    EnsureTyped,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    SelectItemsd,
    RandRotate90d,
)
from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image
from monai.networks.nets import UNet
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss
from monai.optimizers import WarmupCosineSchedule, LinearLR
from torch.utils.tensorboard import SummaryWriter
from fusion_unet import FusionUNetV3

# Setup DDP
def setup(rank, world_size):
    os.environ['LOCAL_RANK'] = f"{rank}"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method="tcp://localhost:12355?use_libuv=0")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    set_determinism()

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, cross_val_num, summary=True):
    setup(rank, world_size)
    # monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    ROI = (96, 96, 96)
    model_path = os.path.join(dir_name, f"no_guide.pth")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    model = FusionUNetV3().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Load model successfully!")
    model = DDP(model, device_ids=[rank])

    with open(splits_json, 'r') as f:
        files = json.load(f)

    
    train_images = [os.path.join(img_dir, f+'.nii.gz') for f in files["train"]]
    val_images = [os.path.join(img_dir, f+'.nii.gz') for f in files["val"]]
    train_segs = [os.path.join(seg_dir, f+'.nii.gz') for f in files["train"]]
    val_segs = [os.path.join(seg_dir, f+'.nii.gz') for f in files["val"]]
    # train_splines = [os.path.join(spline_dir, f+'.nii.gz') for f in files["train"]]
    # val_splines = [os.path.join(spline_dir, f+'.nii.gz') for f in files["val"]]

    # train_files = [{"img": img, "seg": seg, "spline": spline} for img, seg, spline in zip(train_images, train_segs, train_splines)]
    # val_files = [{"img": img, "seg": seg, "spline": spline} for img, seg, spline in zip(val_images,  val_segs, val_splines)]
    train_files = [{"img": img, "seg": seg} for img, seg in zip(train_images, train_segs)]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(val_images,  val_segs)]
    print(len(train_files), len(val_files))

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureTyped(keys=["img",  "seg"]),
            EnsureChannelFirstd(keys=["img",  "seg"]),
            # Spacingd(keys=["img", "seg"], pixdim=SPACING),
            # ScaleIntensityd(keys="img"),
            ScaleIntensityRanged(keys="img", a_min=-1000, a_max=1500, b_min=0.0, b_max=1.0, clip=True),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=ROI, pos=1, neg=1, num_samples=16
            ),
            SelectItemsd(keys=["img", "seg"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureTyped(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            # Spacingd(keys=["img", "seg"], pixdim=SPACING),
            ScaleIntensityRanged(keys="img", a_min=-1000, a_max=1500, b_min=0.0, b_max=1.0, clip=True),
            SelectItemsd(keys=["img", "seg"]),
        ]
    )

    # Create CacheDataset and DataLoader for training
    train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir='/media/ps/85D96EED9527D342/JiangShuanglin_workspace/cache')
    # train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=None)
    train_sampler = DistributedSampler(train_ds, even_divisible=True, shuffle=True)
    train_loader = ThreadDataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0, sampler=train_sampler)

    # Create CacheDataset and DataLoader for validation
    val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir='/media/ps/85D96EED9527D342/JiangShuanglin_workspace/cache')
    # val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=None)
    val_sampler = DistributedSampler(val_ds, even_divisible=True, shuffle=True)
    val_loader = ThreadDataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, sampler=val_sampler)
    
    # Initialize the model, loss, and optimizer
    
    loss_function = DiceCELoss(
        include_background=False,
        softmax=True,
        to_onehot_y=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    lr_scheduler = monai.optimizers.WarmupCosineSchedule(optimizer=optimizer, warmup_steps=0, t_total=200)

    # Metrics and post-transforms
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(softmax=True), AsDiscrete(threshold=0.5)])

    # Training loop
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    metric_values = list()
    writer = SummaryWriter() if summary and rank == 0 else None

    max_epoch =200
    for epoch in range(max_epoch):
        t = time()
        model.train()
        epoch_loss = 0
        step = 0
        train_sampler.set_epoch(epoch)
        for batch_data in train_loader:
            step += 1
            images, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            # inputs = torch.cat((images, splines), dim=1)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            epoch_len = len(train_ds) // (train_loader.batch_size * world_size)
            print(f"[{strftime('%Y-%m-%d %H:%M:%S',localtime(time()))}] epoch:{epoch+1}/{max_epoch}, iter:{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            if rank == 0 and summary:
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        lr_scheduler.step()
        print(lr_scheduler.get_lr())
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_images, val_labels= val_data["img"].to(device), val_data["seg"].to(device)
                    roi_size = ROI
                    sw_batch_size = 16
                    # inputs = torch.cat((val_images, val_splines), dim=1)
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model, overlap=0.5)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()

                if rank == 0:
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.module.state_dict(), model_path)
                        print(f"saved new best metric model {model_path}")
                    t = time() - t
                    print(
                        "current epoch: {}, current mean dice: {:.4f}, best mean dice: {:.4f}, at epoch {}, time cost: {:.0f} min {:.2f} s".format(
                            epoch + 1, metric, best_metric, best_metric_epoch, t//60, t%60
                        )
                    )
                    if summary:
                        metric_values.append(metric)
                        writer.add_scalar("val_mean_dice", metric, epoch + 1)

                    

        
        should_stop = False
        if rank == 0 and epoch - best_metric_epoch > 8:
            should_stop = True
            print("Early stop triggered on rank 0.")
        
        # No early stop
        should_stop = False

        # Broadcasting early stop decision to all processes
        should_stop_tensor = torch.tensor(should_stop, dtype=torch.bool).to(device)
        dist.broadcast(should_stop_tensor, src=0)  # broadcast signal from rank 0 to all rank
        should_stop = should_stop_tensor.item()

        if should_stop:
            print(f"Rank {rank} stopping training.")
            if rank == 0:
                print(f"Best metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
                if summary:
                    writer.close()
            cleanup()
            return

    if rank == 0:
        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        if summary:
            writer.close()

    cleanup()

# Global variables
# data_folder = r'/media/tpx/600C94CE0C94A118/ShuanglinJiang_workspace/ToothFairy_Dataset/Dataset'
# splits_json = r'D:\Jiangshuanglin\Dataset\splits.json'
# img_dir = r'D:\Jiangshuanglin\Dataset\Dataset112_ToothFairy2\imagesTr_final'
# seg_dir = r'D:\Jiangshuanglin\Dataset\Dataset112_ToothFairy2\ian'
# spline_dir = r'D:\Jiangshuanglin\Dataset\Dataset112_ToothFairy2\spline'

splits_json = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/splits.json'
img_dir = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/images'
seg_dir = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/labels/gt_alpha'
spline_dir = '/mnt/sda2/Jiangshuanglin/Dataset/ToothFairy1/ToothFairy_Dataset/splines_test'
abs_path = os.path.abspath(__file__) # absolute path of this file
dir_name = os.path.dirname(abs_path) # directory name of this file

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    for i in [1, ]:
        t0 = time()
        print(f"----------------\n Start group {i} training \n---------------")
        torch.multiprocessing.spawn(main, args=(world_size, i, True,), nprocs=world_size, join=True)
        t = time() - t0
        print(f"----------------\n Finish group {i} training, cost {t//60:.0f} min {t%60:.2f} s \n---------------")
