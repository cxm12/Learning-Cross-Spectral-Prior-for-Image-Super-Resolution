import argparse

scale = 4
parser = argparse.ArgumentParser(description='EDSR and MDSR')
model = 'HAT_cocrtin_NIRw'
parser.add_argument('--modelname', default='%s' % model, help='save model name')
parser.add_argument('--rgb_range', type=int, default=1,  # 255,  #
                    help='maximum value of RGB')

test = 'Set5'  # 'Urban100'  # 'Set14'  # 'BSD100'  #
parser.add_argument('--pre_train', type=str,
     default='./experiment/RGBNIR_s4/HAT_cocrtin_NIRw/model_best.pt',
       help='pre-trained model directory')
# './experiment/RGBNIR_s4/MMNet_cocrtin_NIRw_RGBNIR/model_MMNet_cocrtin_NIRw_RGBNIR/model_latest.pt'
# /mnt/home/user1/MCX/StereoSR/RGBNIRStereo/
# experiment/RGBNIR_s4/HAT_cocrtin_NIRw
parser.add_argument('--resume', type=int, default=0, help='resume from specific checkpoint')
parser.add_argument('--cpu', action='store_true', default=True, help='use cpu only')
parser.add_argument('--save', type=str, default='%s_s%d' % (test, scale), help='file name to save')
parser.add_argument('--load', type=str, default='', help='file name to load')



parser.add_argument('--model', default='EDSR', help='model structure')  # 'StereoSR',  #
parser.add_argument('--test_only', action='store_true', default=True, help='set this option to test the model')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train')
parser.add_argument('--scale', type=str, default='%d' % scale, help='super resolution scale')
# Data specifications
parser.add_argument('--dir_data', type=str, default='/mnt/home/user1/MCX/dataset/StereoData/rgbnir/rgbnir_stereo/data/',  #
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='/mnt/home/user1/MCX/dataset/StereoData/rgbnir/rgbnir_stereo/test100/',  #
                    help='demo image directory')
# parser.add_argument('--dir_data', type=str, default='F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/',  #
#                     help='dataset directory')
# parser.add_argument('--dir_demo', type=str, default='F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/test100/',  #
#                     help='demo image directory')
parser.add_argument('--data_train', type=str, default='RGBNIR',  # 'DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='test',  # 'DIV2K',
                    help='test dataset name')
if 'v2' in model or 'v3' in model:
    if scale == 4 or scale == 2:
        parser.add_argument('--n_resblocks', type=int, default=16, help='16')
    else:
        parser.add_argument('--n_resblocks', type=int, default=32, help='16')
    parser.add_argument('--n_feats', type=int, default=256, help='number of feature maps')
elif model == 'EDSR_cocrtin_RGBNIRw2branch2':
    parser.add_argument('--n_resblocks', type=int, default=16, help='16')
    parser.add_argument('--n_feats', type=int, default=128, help='number of feature maps')
else:
    parser.add_argument('--n_resblocks', type=int, default=8, help='16')
    parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=0.1, help='residual scaling')


parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--patch_size', type=int, default=160, help='output patch size')
parser.add_argument('--debug', action='store_true', help='Enables debug mode')
parser.add_argument('--gammacorrect', action='store_true', default=True, help='RGB image gamma correction ?')
parser.add_argument('--self_ensemble', action='store_true', default=False,
                    help='use self-ensemble method for test')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=0, help='number of threads for data loading')
parser.add_argument('--n_GPUs', type=int, default=0, help='number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--extend', type=str, default='.', help='pre-trained model directory')


parser.add_argument('--shift_mean', default=True, help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true', help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save_models', action='store_true', default=True,
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true', default=True,
                    help='save output results')
parser.add_argument('--save_gt', action='store_true', help='save low-resolution and high-resolution images together')

args = parser.parse_args()

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
