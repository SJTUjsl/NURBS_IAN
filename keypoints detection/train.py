import os
import random
import time
from datetime import datetime
from glob import glob
from tqdm import tqdm
import monai
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from monai.data import CacheDataset, DataLoader, ThreadDataLoader, decollate_batch, PersistentDataset
from monai.inferers import sliding_window_inference
from monai.metrics import RMSEMetric
from monai.networks.nets import UNet
from monai.transforms import (Activations, AsDiscrete, Compose, EnsureChannelFirstd,
                              EnsureTyped, LoadImaged, RandCropByPosNegLabeld, RandFlipd,
                              ScaleIntensityd, RandRotate90d)
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
def setup(rank, world_size):
    os.environ['LOCAL_RANK'] = f"{rank}"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29400'
    dist.init_process_group("gloo", rank=rank, world_size=world_size,init_method="tcp://localhost:29400?use_libuv=0")
    set_determinism()

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    roi = (224, 224)
    batch_size = 64

    img_folder = r'D:\Jiangshuanglin\Dataset\Dataset112_ToothFairy2\slices\imageSlices'
    seg_folder = r'D:\Jiangshuanglin\Dataset\Dataset112_ToothFairy2\slices\heatmap'
    images = sorted(glob(os.path.join(img_folder, "*.png")))
    segs = sorted(glob(os.path.join(seg_folder, "*.png")))

    files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
    random.shuffle(files)

    split = int(len(files) * 0.8)
    train_files = files[:split]
    val_files = files[split:]

    train_transforms = Compose([
        LoadImaged(keys=["img", "seg"], image_only=True),
        EnsureTyped(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img", "seg"]),
        # RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=0),
        # RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=1),
        # RandRotate90d(keys=["img", "seg"], prob=0.5, max_k=3),
        # RandCropByPosNegLabeld(keys=["img", "seg"], label_key="seg", spatial_size=roi, pos=1, neg=1, num_samples=4)
    ])

    val_transforms = Compose([
        LoadImaged(keys=["img", "seg"], image_only=True),
        EnsureTyped(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys=["img", "seg"])
    ])

    # train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=16, copy_cache=False)
    train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir='C:\JiangShuanglin_workspace\cache')
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_loader = ThreadDataLoader(train_ds, batch_size=batch_size, num_workers=0, sampler=train_sampler)

    # val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=16, copy_cache=False)
    val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir='C:\JiangShuanglin_workspace\cache')
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
    val_loader = ThreadDataLoader(val_ds, batch_size=batch_size, num_workers=0, sampler=val_sampler)

    post_trans = Compose([Activations(other=lambda x: F.relu(x)), AsDiscrete(threshold=0.98)])

    model = UNet(
        spatial_dims=2, 
        in_channels=1, 
        out_channels=1, 
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2), 
        num_res_units=2,
    ).to(device)
    model = DDP(model, device_ids=[rank])
    loss_function = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    rmse_metric = RMSEMetric(reduction="mean")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    weight_path = "toothfairy.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        print("load pre-trained weights")

    val_interval = 1
    best_metric = 1e5
    best_metric_epoch = -1
    num_epochs = 1000
    if rank == 0:
        writer = SummaryWriter(comment="toothfairy")
    epoch_len = len(train_ds) // (train_loader.batch_size * world_size)
    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0
        step = 0
        train_sampler.set_epoch(epoch)
        scheduler.step()
        loop = tqdm((train_loader), total=len(train_loader))
        for batch_data in loop:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_description(f'Epoch[{epoch + 1}/{num_epochs}]')
            loop.set_postfix(loss = loss.item())
            if rank == 0:
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        if rank == 0:
            writer.add_scalar("train_epoch_loss", epoch_loss, epoch+1)
        
        # validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_rmse = []
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                    roi_size = roi
                    sw_batch_size = batch_size
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    rmse_metric(y_pred=val_outputs, y=val_labels)
                    val_rmse.append(rmse_metric.aggregate().item())
                    rmse_metric.reset()

                metric = sum(val_rmse) / len(val_rmse)
                if metric < best_metric and rank == 0:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.module.state_dict(), weight_path)
                    # torch.save(model.state_dict(), weight_path)
                    print("saved new best metric model")

                total_time = time.time() - t0
                
                if rank == 0:
                    print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}, time cost {:.4f}s".format(
                        epoch + 1, metric, best_metric, best_metric_epoch, total_time
                    )
                )
                    writer.add_scalar("val_mean_dice", metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    if rank == 0:
        writer.close()

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    # main(0, 1)
