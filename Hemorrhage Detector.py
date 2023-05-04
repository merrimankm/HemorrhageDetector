#!/usr/bin/env python
# coding: utf-8

# In[64]:


import matplotlib.pyplot as plt
import torch
import numpy as np

from monai.config import print_config

from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    AddChanneld,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    Activations,
    AsDiscrete,
    EnsureChannelFirstd,
    Orientationd,
    CenterSpatialCropd,
    RandCropByLabelClassesd,
    NormalizeIntensityd,
    Rotate90d
)
from monai.utils import set_determinism, first

from monai.data import Dataset, DataLoader, NibabelReader
from monai.data import pad_list_data_collate, decollate_batch

from monai.networks.nets import UNet
from monai.metrics import DiceMetric, compute_fp_tp_probs
from monai.losses import DiceLoss

# from torch.utils.tensorboard import SummaryWriter
from monai.inferers import sliding_window_inference
from monai.visualize import plot_2d_or_3d_image

print_config()

# In[2]:


directory = '/media/14TB/aarlova_hem'
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

# In[3]:


set_determinism(seed=0)

# In[98]:


# Dataset
from pathlib import Path

dataset_dir_name = '/media/14TB/aarlova_hem/data/'
dataset_dir = Path(dataset_dir_name)

images = sorted(list(dataset_dir.rglob('SURG*_axial.nii.gz')))
masks = sorted(list(dataset_dir.rglob('SURG*_axial_mask.nii.gz')))

test_dataset_dir_name = '/media/14TB/aarlova_hem/test_noHem'
test_masks_dir_name = '/media/14TB/aarlova_hem/test_masks'
test_dataset_dir = Path(test_dataset_dir_name)
test_masks_dir = Path(test_masks_dir_name)

test_images = sorted(list(test_dataset_dir.rglob('SURG*_axial.nii.gz')))
test_masks = sorted(list(test_masks_dir.rglob('SURG*_axial_mask.nii.gz')))

print(f'There are {len(images)} images, and {len(masks)} masks')
print(f'There are {len(test_images)} test images')

train_images = images[20:100]
val_images1 = images[0:20]
val_images2 = images[100:]
val_images = val_images1 + val_images2

train_masks = masks[20:100]
val_masks1 = masks[0:20]
val_masks2 = masks[100:]
val_masks = val_masks1 + val_masks2

# create a Python dictionary
# with two columns, one for the image paths and one for the label paths

train_files = [{'img': img_name, 'mask': mask_name} for img_name, mask_name in zip(train_images, train_masks)]
val_files = [{'img': img_name, 'mask': mask_name} for img_name, mask_name in zip(val_images, val_masks)]

test_files = [{'img': img_name} for img_name in test_images]
# test_files = [{'img': img_name, 'mask': mask_name} for img_name, mask_name in zip(test_images, test_masks)]


# In[99]:


print('First image-mask pair:')
print(train_files[0])

print('\nLast val image-mask pair')
print(val_files[-1:])

print('\nLast TEST image-mask pair')
print(test_files[-1:])

# In[103]:


# define transforms for image and segmentation

# , reader=NibabelReader(channel_dim=-1)),
# Orientationd(keys=['img', 'mask'], axcodes="RSP"),

# orig_transforms is a sanity check transforms
# CenterSpatialCropd(keys=['img', 'mask'], roi_size=(1,120,120)),


orig_transforms = Compose(
    [
        LoadImaged(keys=['img', 'mask'], reader=NibabelReader(channel_dim=-1)),
        AddChanneld(keys=['img', 'mask']),
        EnsureChannelFirstd(keys=['img', 'mask']),
        ToTensord(keys=['img', 'mask'])
    ])

train_transforms = Compose(
    [
        LoadImaged(keys=['img', 'mask'], reader=NibabelReader(channel_dim=-1)),
        AddChanneld(keys=['img', 'mask']),
        EnsureChannelFirstd(keys=['img', 'mask']),

        # ScaleIntensityRanged(keys=['img'], a_min = -200, a_max = 200, b_min = 0, b_max = 1),
        NormalizeIntensityd(keys=['img'], nonzero=True),
        # RandCropByLabelClassesd(keys=['img','mask'], label_key='mask', spatial_size=[1,64,64]),#, ratios=[1,1], num_classes=2, num_samples=6),
        CenterSpatialCropd(keys=['img', 'mask'], roi_size=(1, 120, 120)),

        ToTensord(keys=['img', 'mask'])
    ])

val_transforms = Compose(
    [
        LoadImaged(keys=['img', 'mask'], reader=NibabelReader(channel_dim=-1)),
        AddChanneld(keys=['img', 'mask']),
        EnsureChannelFirstd(keys=['img', 'mask']),

        # ScaleIntensityRanged(keys=['img'], a_min = -200, a_max = 200, b_min = 0, b_max = 1),
        NormalizeIntensityd(keys=['img'], nonzero=True),

        CenterSpatialCropd(keys=['img', 'mask'], roi_size=(1, 120, 120)),
        ToTensord(keys=['img', 'mask'])
    ])

### test transforms with no GT mask
test_transforms = Compose(
    [
        LoadImaged(keys=['img'], reader=NibabelReader(channel_dim=-1)),
        AddChanneld(keys=['img']),
        EnsureChannelFirstd(keys=['img']),

        # ScaleIntensityRanged(keys=['img'], a_min = -200, a_max = 200, b_min = 0, b_max = 1),
        NormalizeIntensityd(keys=['img'], nonzero=True),

        CenterSpatialCropd(keys=['img'], roi_size=(1, 120, 120)),

        ToTensord(keys=['img'])
    ])

### test transforms with GT mask
# test_transforms = Compose(
# [
#     LoadImaged(keys=['img','mask'], reader=NibabelReader(channel_dim=-1)),
#     AddChanneld(keys=['img','mask']),
#     EnsureChannelFirstd(keys=['img', 'mask']),

#     # ScaleIntensityRanged(keys=['img'], a_min = -200, a_max = 200, b_min = 0, b_max = 1),
#     NormalizeIntensityd(keys=['img'], nonzero=True),

#     CenterSpatialCropd(keys=['img', 'mask'], roi_size=(1,120,120)),

#     ToTensord(keys = ['img','mask'])
# ])


# In[104]:


orig_ds = Dataset(data=train_files, transform=orig_transforms)
orig_loader = DataLoader(orig_ds, batch_size=1)  # , collate_fn=pad_list_data_collate)

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1)  # , collate_fn=pad_list_data_collate)

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1)  # , collate_fn=pad_list_data_collate)

test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1)

# In[105]:


# wrong_labels = []
# for batch_data in train_loader:
#     # print(batch_data['img'].shape, batch_data['mask'].shape)
#     inputs, labels = batch_data['img'].squeeze(dim=0), batch_data['mask'].squeeze(dim=0).cpu()
#     labels = labels.get_array()
#     print(np.unique(labels))
#     print(batch_data['img_meta_dict']['filename_or_obj'])


# In[106]:


# visualize one slice from one patient

orig_patient = first(orig_loader)
test_patient = first(train_loader)

print(orig_patient['img'].shape)
print(test_patient['img'].shape)
print(test_patient['mask'].shape)

plt.figure('test', (10, 5))

plt.subplot(1, 3, 1)
plt.title('Orig patient')
plt.imshow(orig_patient['img'][0, orig_patient['img'].shape[1] - 3, 0, :, :], cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Slice of a patient')
plt.imshow(test_patient['img'][0, test_patient['img'].shape[1] - 3, 0, :, :], cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Mask of a patient')
plt.imshow(test_patient['mask'][0, test_patient['mask'].shape[1] - 3, 0, :, :], cmap='summer')

plt.show()
plt.cla()
plt.close()

# In[73]:


plt.rcParams['figure.dpi'] = 140
plt.rcParams['figure.figsize'] = [10.0, 5.0]

num_slices = test_patient['img'].shape[1]
n_rows = int(np.ceil(num_slices / 3))
print(num_slices)

for i in range(num_slices):
    plt.subplot(n_rows, 3, i + 1)
    plt.title('Slice %d' % i)
    plt.imshow(test_patient['img'][0, i, 0, :, :], cmap='gray')

# In[33]:


plt.figure('one patient', (10, 5))

for i in range(num_slices):
    plt.subplot(n_rows, 3, i + 1)
    plt.title('Slice %d' % i)
    plt.imshow(test_patient['mask'][0, i, 0, :, :], cmap='summer')

# In[42]:


dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
froc = compute_fp_tp_probs
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
# create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,  # decide what it should be?
    channels=(16, 32, 64, 128),  # , 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

# In[43]:


# torch.cuda.is_available()
print(device)

# In[15]:


# start a typical PyTorch training
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()

epochs = 200

for epoch in range(epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        # print(batch_data['img'].shape, batch_data['mask'].shape)
        inputs, labels = batch_data['img'].squeeze(dim=0).to(device), batch_data['mask'].squeeze(dim=0).to(device)
        # print(inputs.shape, labels.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    print('finished calculating average epoch loss')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    #     if (epoch + 1) % val_interval == 0:
    model.eval()
    with torch.no_grad():

        val_images = None
        val_labels = None
        val_outputs = None
        for val_data in val_loader:
            val_images, val_labels = val_data['img'].squeeze(dim=0).to(device), val_data['mask'].squeeze(dim=0).to(
                device)
            roi_size = [96, 96]
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            # compute_fp_tp_probs(y_pred=val_outputs, y=val_labels)

        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()
        metric_values.append(metric)
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_metric_model_segmentation2d_dict.pth")
            print("saved new best metric model")
        print(
            "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                epoch + 1, metric, best_metric, best_metric_epoch
            )
        )

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

# In[107]:


from matplotlib import colors

post_trans = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.9)])  # CenterSpatialCrop(roi_size = (100,100,-1))

plt.rcParams['figure.dpi'] = 340
plt.rcParams['figure.figsize'] = [10.0, 5.0]

model.load_state_dict(torch.load("best_metric_model_segmentation2d_dict.pth"))

model.eval()

cmap = colors.ListedColormap(['grey', 'red'])
cmap1 = colors.ListedColormap(['grey', 'green'])

with torch.no_grad():
    for test_data in test_loader:
        print(test_data['img_meta_dict']['filename_or_obj'])

        # test_images, test_labels = test_data['img'].squeeze(dim=0).to(device), test_data['mask'].squeeze(dim=0).to(device)
        test_images = test_data['img'].squeeze(dim=0).to(device)

        num_slices = len(test_images)
        n_rows = int(np.ceil(num_slices / 3))

        roi_size = (96, 96)
        sw_batch_size = 4
        test_outputs = sliding_window_inference(test_images, roi_size, sw_batch_size, model)
        test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
        plt.subplot(2, 2, 1)
        plt.title('Predicted Mask')
        plt.imshow(np.rot90(test_images[2].cpu().permute(1, 2, 0), k=3), cmap='gray')
        plt.imshow(np.rot90(test_outputs[2].cpu().permute(1, 2, 0), k=3), cmap=cmap, alpha=0.3)
        plt.subplot(2, 2, 2)
        plt.title('Ground Truth Mask')
        plt.imshow(np.rot90(test_images[2].cpu().permute(1, 2, 0), k=3), cmap='gray')
        # plt.imshow(np.rot90(test_labels[2].cpu().permute(1, 2, 0), k=3), cmap = cmap1, alpha=0.3)
        plt.show()
        plt.cla()
        plt.clf()
        plt.close()

# In[ ]:




