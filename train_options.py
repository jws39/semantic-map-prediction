from pytorch_utils.base_options import BaseOptions
import argparse


class TrainOptions(BaseOptions):
    """ Parses command line arguments for training
    This overwrites options from BaseOptions
    """
    def __init__(self): # pylint: disable=super-init-not-called
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=3600000,
                         help='Total time to run in seconds')
        gen.add_argument('--resume', dest='resume', default=False,
                         action='store_true',
                         help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=4,
                         help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                         help='Number of processes used for data loading')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false',
                         help='Number of processes used for data loading')
        gen.set_defaults(pin_memory=True)

        in_out = self.parser.add_argument_group('io')
        in_out.add_argument('--log_dir', default='~/semantic_grid/logs', help='Directory to store logs')
        in_out.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        in_out.add_argument('--from_json', default=None,
                            help='Load options from json file instead of the command line')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=1000,
                           help='Total number of training epochs')
        train.add_argument('--batch_size', type=int, default=1, help='Batch size')
        train.add_argument('--test_batch_size', type=int, default=1, help='Batch size')
        train.add_argument('--test_nav_batch_size', type=int, default=1, help='Batch size during navigation test')
        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true',
                                   help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false',
                                   help='Don\'t shuffle training data')
        shuffle_test = train.add_mutually_exclusive_group()
        shuffle_test.add_argument('--shuffle_test', dest='shuffle_test', action='store_true',
                                  help='Shuffle testing data')
        shuffle_test.add_argument('--no_shuffle_test', dest='shuffle_test', action='store_false',
                                  help='Don\'t shuffle testing data')

        # Dataset related options
        train.add_argument('--data_type', dest='data_type', type=str, default='train',
                           choices=['train', 'val'],
                           help='Choose which dataset to run on, valid only with --use_store')

        # train.add_argument('--network_name', dest='network_name', type=str, default='sscnav',
        #                    choices=['sscnav', 'unet_dam_last_layer_v2'],
        #                    help='Choose which network to use, valid only with --use_store')
        #
        # train.add_argument('--view_type', dest='view_type', type=str, default='multi_view',
        #                    choices=['single_view', 'multi_view'],
        #                    help='Choose which dataset to use, valid only with --use_store')
        #
        # train.add_argument('--use_l2m', dest='use_l2m', action='store_true',
        #                     help='Using l2m original model')
        #
        #
        # train.add_argument('--with_am', dest='with_am', action='store_true',
        #                    help='Using Affinity Map loss')


        train.add_argument('--dataset_percentage', dest='dataset_percentage', type=float, default=1.0,
                            help='percentage of dataset to be used during training for ensemble learning')

        train.add_argument('--summary_steps', type=int, default=1000,
                           help='Summary saving frequency')
        train.add_argument('--image_summary_steps', type=int, default=5000,
                           help='Image summary saving frequency')
        train.add_argument('--checkpoint_steps', type=int, default=30000000,  # default=30000, save checkpoint in test_steps
                           help='Chekpoint saving frequency')

        train.add_argument('--test_steps', type=int, default=10000, help='Testing frequency')

        train.add_argument('--is_train', dest='is_train', action='store_true',
                            help='Define whether training or testing mode')

        train.add_argument('--config_train_file', type=str, dest='config_train_file',
                            default='configs/my_objectnav_mp3d_train.yaml',
                            help='path to habitat dataset train config file')

        self.parser.add_argument('--config_test_file', type=str, dest='config_test_file',
                                default='configs/my_objectnav_mp3d_test.yaml',
                                help='path to test config file -- to be used with our episodes')

        self.parser.add_argument('--config_val_file', type=str, dest='config_val_file',
                                default='configs/my_objectnav_mp3d_val.yaml',
                                help='path to habitat dataset val config file')

        self.parser.add_argument('--ensemble_dir', type=str, dest='ensemble_dir', default=None,
                                help='Path containing the experiments comprising the ensemble')

        self.parser.add_argument('--n_spatial_classes', type=int, default=3, dest='n_spatial_classes',
                                help='number of categories for spatial prediction')
        self.parser.add_argument('--n_object_classes', type=int, default=27, dest='n_object_classes',
                                choices=[18, 27], help='number of categories for object prediction')
        self.parser.add_argument('--grid_dim', type=int, default=384, dest='grid_dim',
                                    help='Semantic grid size (grid_dim, grid_dim)')
        self.parser.add_argument('--cell_size', type=float, default=0.1, dest="cell_size",
                                    help='Physical dimensions (meters) of each cell in the grid')
        self.parser.add_argument('--crop_size', type=int, default=64, dest='crop_size',
                                    help='Size of crop around the agent')

        self.parser.add_argument('--img_size', dest='img_size', type=int, default=256)
        self.parser.add_argument('--img_segm_size', dest='img_segm_size', type=int, default=128)


        train.add_argument('--map_loss_scale', type=float, default=1.0, dest='map_loss_scale')
        train.add_argument('--mse_loss_scale', type=float, default=1.0, dest='mse_loss_scale')
        train.add_argument('--img_segm_loss_scale', type=float, default=1.0, dest='img_segm_loss_scale')

        train.add_argument('--init_gaussian_weights', dest='init_gaussian_weights', action='store_true',
                            help='initializes the model weights from gaussian distribution')


        train.set_defaults(shuffle_train=True, shuffle_test=True)

        optim = self.parser.add_argument_group('Optim')
        optim.add_argument("--lr_decay", type=float,
                           default=0.99, help="Exponential decay rate")
        optim.add_argument("--wd", type=float,
                           default=0, help="Weight decay weight")

        self.parser.add_argument('--test_iters', type=int, default=20000)

        optimizer_options = self.parser.add_argument_group('Optimizer')
        optimizer_options.add_argument('--lr', type=float, default=0.0002)
        optimizer_options.add_argument('--beta1', type=float, default=0.5)

        model_options = self.parser.add_argument_group('Model')


        # model_options.add_argument('--with_img_segm', dest='with_img_segm', default=False, action='store_true',
        #                             help='uses the img segmentation pre-trained model during training or testing')


        # model_options.add_argument('--img_segm_model_dir', dest='img_segm_model_dir', default=None,
        #                             help='job path that contains the pre-trained img segmentation model')


        self.parser.add_argument('--sem_map_test', dest='sem_map_test', default=False, action='store_true')
        self.parser.add_argument('--stored_episodes_dir', type=str, dest='stored_episodes_dir', default='mp3d_objnav_episodes_tmp/')
        self.parser.add_argument('--ensemble_size', type=int, dest='ensemble_size', default=1)


