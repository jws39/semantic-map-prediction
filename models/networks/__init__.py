from models.networks.resnetUnet import ResNetUNet, ResNetUNetDAMLastLayerv2


def get_mp_network(options):

    print('--------------------------unet_dam_last_layer_v2--------------------------')
    return ResNetUNetDAMLastLayerv2(n_channel_in=options.n_object_classes, n_class_out=options.n_object_classes)

