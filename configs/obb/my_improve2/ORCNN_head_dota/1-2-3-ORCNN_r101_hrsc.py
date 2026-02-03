_base_ = './1-2-3-ORCNN_r50_hrsc.py'

# model
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
