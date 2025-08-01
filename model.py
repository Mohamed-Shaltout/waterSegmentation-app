import segmentation_models_pytorch as smp

def UNet(in_channels=7, out_channels=1):
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",         # Use None if you trained from scratch
        in_channels=in_channels,
        classes=out_channels,
    )
    return model