import segmentation_models_pytorch as smp


def get_model(count_class, backbone):
    return smp.FPN(encoder_name=backbone, encoder_weights="imagenet", in_channels=3, classes=count_class,)
