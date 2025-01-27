import math
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

_C.NUM_GPUS = 4

_C.PIPELINE = ['train']

_C.MODELDATA = None

_C.TASK = "tmodeling"

_C.LINEAR_PROBING = False

_C.RESUME = None


# -----------------------------------------------------------------------------
# TModeling options
# -----------------------------------------------------------------------------

_C.TM = CfgNode()

_C.TM.ENABLE_BF16 = False

_C.TM.SPATIAL_POOL_DIM = None

_C.TM.CHANNEL_POOL_DIM = None

_C.TM.TEMPORAL_ARCH = 'conv3d'

_C.TM.NUM_BLOCKS = 4

_C.TM.NUM_HEADS = 6

_C.TM.SKIP_TM = False

_C.TM.FROM_SCRATCH = False

_C.TM.TEMPORAL_CAT = False

_C.TM.LINEAR_PROJ = False

# -----------------------------------------------------------------------------
# SSL options
# -----------------------------------------------------------------------------

_C.SSL = CfgNode()

_C.SSL.TASK = "speed"

_C.SSL.WEIGHT_OF_LOSS = [1.0]

_C.SSL.MARGIN = 0.5

_C.SSL.METHOD = 'random'

_C.SSL.JITTER = 0.2

_C.SSL.RANGE = [1, 2, 4, 8]

_C.SSL.HEAD_L2_NORM = False

_C.SSL.NUM_MLP_LAYERS = 2

_C.SSL.MLP_DIM = 2048

_C.SSL.BN_MLP = True

_C.SSL.BN_SYNC_MLP = True

_C.SSL.STAT = ['loss_spd', 'acc_spd']

_C.SSL.METRIC = ['ce', 'acc@1']

_C.SSL.SMOOTHING = 0.0


# -----------------------------------------------------------------------------
# Contrastive Model (for MoCo, SimCLR, SwAV, BYOL)
# -----------------------------------------------------------------------------

_C.CONTRASTIVE = CfgNode()

# temperature used for contrastive losses
_C.CONTRASTIVE.T = 0.07

# output dimension for the loss
_C.CONTRASTIVE.DIM = 128

_C.CONTRASTIVE.HIDDEN_DIM = 4096

# number of training samples (for kNN bank)
_C.CONTRASTIVE.LENGTH = 239975

# the length of MoCo's and MemBanks' queues
_C.CONTRASTIVE.QUEUE_LEN = 65536

# momentum for momentum encoder updates
_C.CONTRASTIVE.MOMENTUM = 0.5

# wether to anneal momentum to value above with cosine schedule
_C.CONTRASTIVE.MOMENTUM_ANNEALING = False

# either memorybank, moco, simclr, byol, swav
_C.CONTRASTIVE.TYPE = "mem"

# wether to interpolate memorybank in time
_C.CONTRASTIVE.INTERP_MEMORY = False

# 1d or 2d (+temporal) memory
_C.CONTRASTIVE.MEM_TYPE = "1d"

# number of classes for online kNN evaluation
_C.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM = 400

# use an MLP projection with these num layers
_C.CONTRASTIVE.NUM_MLP_LAYERS = _C.SSL.NUM_MLP_LAYERS

# dimension of projection and predictor MLPs
_C.CONTRASTIVE.MLP_DIM = _C.SSL.MLP_DIM

# use BN in projection/prediction MLP
_C.CONTRASTIVE.BN_MLP = _C.SSL.BN_MLP

# use synchronized BN in projection/prediction MLP
_C.CONTRASTIVE.BN_SYNC_MLP = _C.SSL.BN_SYNC_MLP

# shuffle BN only locally vs. across machines
_C.CONTRASTIVE.LOCAL_SHUFFLE_BN = True

# Wether to fill multiple clips (or just the first) into queue
_C.CONTRASTIVE.MOCO_MULTI_VIEW_QUEUE = False

# if sampling multiple clips per vid they need to be at least min frames apart
_C.CONTRASTIVE.DELTA_CLIPS_MIN = -math.inf

# if sampling multiple clips per vid they can be max frames apart
_C.CONTRASTIVE.DELTA_CLIPS_MAX = math.inf

# if non empty, use predictors with depth specified
_C.CONTRASTIVE.PREDICTOR_DEPTHS = []

# Wether to sequentially process multiple clips (=lower mem usage) or batch them
_C.CONTRASTIVE.SEQUENTIAL = False

# Wether to perform SimCLR loss across machines (or only locally)
_C.CONTRASTIVE.SIMCLR_DIST_ON = True

# Length of queue used in SwAV
_C.CONTRASTIVE.SWAV_QEUE_LEN = 0

# Wether to run online kNN evaluation during training
_C.CONTRASTIVE.KNN_ON = True


# -----------------------------------------------------------------------------
# Training options
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()

_C.TRAIN.LOG_FREQ = 20

_C.TRAIN.SAVE_FREQ = 5

_C.TRAIN.AMP = False

_C.TRAIN.SEED = 0

_C.TRAIN.RESUME = None

_C.TRAIN.AUTO_RESUME = False

_C.TRAIN.SEQUENTIAL = False


# -----------------------------------------------------------------------------
# Inference options
# -----------------------------------------------------------------------------
_C.INFER = CfgNode()

_C.INFER.SAMPLE_METHOD = "uniform"

_C.INFER.NUM_CLIPS = 10

_C.INFER.NUM_CROPS = 3

_C.INFER.RES = 256

_C.INFER.NUM_FRAMES = 8

_C.INFER.STRIDE = 8

_C.INFER.JITTER = 0.0

_C.INFER.SEED = 0

_C.INFER.DATADIR = None

_C.INFER.LABELDIR = None

_C.INFER.SPLITFILE = None

_C.INFER.BATCHSIZE_PER_GPU = 1

_C.INFER.WORKERS = 8


# -----------------------------------------------------------------------------
# Validatin options
# -----------------------------------------------------------------------------
_C.VAL = CfgNode()

_C.VAL.TARGET_SIZE = 224

_C.VAL.MIN_AREA = 0.08

_C.VAL.RAND_CROP_RATIO = (3.0 / 4.0, 4.0 / 3.0)

_C.VAL.NUM_FRAME = 8

_C.VAL.STRIDE = 1

_C.VAL.JITTER = 0.2

_C.VAL.RANGE = [1, 2, 4, 8]

_C.VAL.DATADIR = None

_C.VAL.LABELDIR = None

_C.VAL.SPLITFILE = None

_C.VAL.BATCHSIZE_PER_GPU = 2

_C.VAL.WORKERS = 8


# -----------------------------------------------------------------------------
# Solver options
# -----------------------------------------------------------------------------

_C.SOLVER = CfgNode()

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 100

_C.SOLVER.START_EPOCH = 0

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate.
_C.SOLVER.BASE_LR = 1e-1

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-6

# Learning rate policy.
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 10.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.001

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = False

# The layer-wise decay of learning rate. Set to 1. to disable.
_C.SOLVER.LAYER_DECAY = 1.0

# LARS optimizer
_C.SOLVER.LARS_ON = False

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = False

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Adam's beta
_C.SOLVER.BETAS = (0.9, 0.999)


# -----------------------------------------------------------------------------
# Augmentation options
# -----------------------------------------------------------------------------

_C.AUG = CfgNode()

_C.AUG.TYPE = "simple"

# options for AUG.TYPE "simple"
_C.AUG.RESIZE = [320, 256]

_C.AUG.TARGET_SIZE = 224

# options for AUG.TYPE "simple"
_C.AUG.COLOR = [0.8, 0.4, 0.4, 0.4, 0.2]

_C.AUG.GRAYSCALE = 0.2

_C.AUG.MIN_AREA = 0.08

_C.AUG.MAX_AREA = 0.76

_C.AUG.RAND_CROP_RATIO = (3.0 / 4.0, 4.0 / 3.0)

_C.AUG.AA_TYPE = "rand-m7-n4-mstd0.5-inc1"

_C.AUG.INTERPOLATION = "bicubic"

_C.AUG.CAMERA_SHAKE = 0.0

_C.AUG.CAMERA_SHIFT = 0.0

_C.AUG.ZOOM = 0.0

_C.AUG.COLOR_BRI = 0.0

_C.AUG.COLOR_SAT = 0.0

_C.AUG.WHITE_BALANCE = 0.0

_C.AUG.COLOR_AREA_SCALE = (0.04, 0.16)

_C.AUG.SSL_BLUR_SIGMA_MIN = [0.0, 0.1]

_C.AUG.SSL_BLUR_SIGMA_MAX = [0.0, 2.0]

_C.AUG.RANDOM_FLIP = True

_C.AUG.INV_UNIFORM_SAMPLE = False

_C.AUG.TRAIN_JITTER_MOTION_SHIFT = False


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------

_C.DATA = CfgNode()

_C.DATA.DATASET = "ucf-101"

_C.DATA.DATADIR = None

_C.DATA.LABELDIR = None

_C.DATA.SPLITFILE = None

_C.DATA.NUM_CLIP = 4

_C.DATA.TRAIN_CROP_NUM_TEMPORAL = _C.DATA.NUM_CLIP

_C.DATA.TRAIN_CROP_NUM_SPATIAL = 1

_C.DATA.NUM_FRAMES = 8

_C.DATA.STRIDE = 1

_C.DATA.BATCHSIZE_PER_GPU = 8

_C.DATA.WORKERS = 8

_C.DATA.INPUT_CHANNEL_NUM = [3]

# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------

_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

#  If true, initialize the final conv layer of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_CONV = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------

_C.MODEL = CfgNode()

_C.MODEL.MODEL_NAME = "ResNet"

_C.MODEL.ARCH = "Slow"

_C.MODEL.NUM_CLASSES = 5

_C.MODEL.DROPOUT_RATE = 0.0

# If True, detach the final fc layer from the network, by doing so, only the
# final fc layer will be trained.
_C.MODEL.DETACH_FINAL_FC = False

# If True, AllReduce gradients are compressed to fp16
_C.MODEL.FP16_ALLREDUCE = False

# Activation checkpointing enabled or not to save GPU memory.
_C.MODEL.ACT_CHECKPOINT = False

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

_C.MODEL.SPATIAL_MODEL_PATH = ""

_C.MODEL.SPATIAL_MODEL_ARCH = ""

_C.MODEL.SPATIAL_EMBEDDING_DIM = 768

_C.MODEL.TEMPORAL_NUM_LAYERS = 6

_C.MODEL.TEMPORAL_NUM_HEADS = 8

_C.MODEL.TEMPORAL_EMBEDDING_DIM = 512

_C.MODEL.TEMPORAL_HIDDEN_DIM = 256

_C.MODEL.TEMPORAL_NUM_EMBEDDINGS = 8


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #

_C.BN = CfgNode()

# Parameter for NaiveSyncBatchNorm. Setting `GLOBAL_SYNC` to True synchronizes
# stats across all devices, across all machines; in this case, `NUM_SYNC_DEVICES`
# must be set to None.
# By default ONLY applies to NaiveSyncBatchNorm3d; consider also setting
# CONTRASTIVE.BN_SYNC_MLP if appropriate.
_C.BN.GLOBAL_SYNC = False

# Parameter for NaiveSyncBatchNorm, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized. `NUM_SYNC_DEVICES` cannot be larger than number of
# devices per machine; if global sync is desired, set `GLOBAL_SYNC`.
# By default ONLY applies to NaiveSyncBatchNorm3d; consider also setting
# CONTRASTIVE.BN_SYNC_MLP if appropriate.
_C.BN.NUM_SYNC_DEVICES = 4

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "sync_batchnorm"

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0


# ---------------------------------------------------------------------------- #
# Detection options.
# ---------------------------------------------------------------------------- #

_C.DETECTION = CfgNode()

# Whether enable video detection.
_C.DETECTION.ENABLE = False


# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------

_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"

# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
    ]


# -----------------------------------------------------------------------------
# MViT options
# -----------------------------------------------------------------------------
_C.MVIT = CfgNode()

# Options include `conv`, `max`.
_C.MVIT.MODE = "conv"

# If True, perform pool before projection in attention.
_C.MVIT.POOL_FIRST = False

# If True, use cls embed in the network, otherwise don't use cls_embed in transformer.
_C.MVIT.CLS_EMBED_ON = True

# Kernel size for patchtification.
_C.MVIT.PATCH_KERNEL = [3, 7, 7]

# Stride size for patchtification.
_C.MVIT.PATCH_STRIDE = [2, 4, 4]

# Padding size for patchtification.
_C.MVIT.PATCH_PADDING = [2, 4, 4]

# If True, use 2d patch, otherwise use 3d patch.
_C.MVIT.PATCH_2D = False

# Base embedding dimension for the transformer.
_C.MVIT.EMBED_DIM = 96

# Base num of heads for the transformer.
_C.MVIT.NUM_HEADS = 1

# Dimension reduction ratio for the MLP layers.
_C.MVIT.MLP_RATIO = 4.0

# If use, use bias term in attention fc layers.
_C.MVIT.QKV_BIAS = True

# Drop path rate for the tranfomer.
_C.MVIT.DROPPATH_RATE = 0.1

# The initial value of layer scale gamma. Set 0.0 to disable layer scale.
_C.MVIT.LAYER_SCALE_INIT_VALUE = 0.0

# Depth of the transformer.
_C.MVIT.DEPTH = 16

# Normalization layer for the transformer. Only layernorm is supported now.
_C.MVIT.NORM = "layernorm"

# Dimension multiplication at layer i. If 2.0 is used, then the next block will increase
# the dimension by 2 times. Format: [depth_i: mul_dim_ratio]
_C.MVIT.DIM_MUL = []

# Head number multiplication at layer i. If 2.0 is used, then the next block will
# increase the number of heads by 2 times. Format: [depth_i: head_mul_ratio]
_C.MVIT.HEAD_MUL = []

# Stride size for the Pool KV at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_KV_STRIDE = []

# Initial stride size for KV at layer 1. The stride size will be further reduced with
# the raio of MVIT.DIM_MUL. If will overwrite MVIT.POOL_KV_STRIDE if not None.
_C.MVIT.POOL_KV_STRIDE_ADAPTIVE = None

# Stride size for the Pool Q at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_Q_STRIDE = []

# If not None, overwrite the KV_KERNEL and Q_KERNEL size with POOL_KVQ_CONV_SIZ.
# Otherwise the kernel_size is [s + 1 if s > 1 else s for s in stride_size].
_C.MVIT.POOL_KVQ_KERNEL = None

# If True, perform no decay on positional embedding and cls embedding.
_C.MVIT.ZERO_DECAY_POS_CLS = True

# If True, use norm after stem.
_C.MVIT.NORM_STEM = False

# If True, perform separate positional embedding.
_C.MVIT.SEP_POS_EMBED = False

# Dropout rate for the MViT backbone.
_C.MVIT.DROPOUT_RATE = 0.0

# If True, use absolute positional embedding.
_C.MVIT.USE_ABS_POS = True

# If True, use relative positional embedding for spatial dimentions
_C.MVIT.REL_POS_SPATIAL = False

# If True, use relative positional embedding for temporal dimentions
_C.MVIT.REL_POS_TEMPORAL = False

# If True, init rel with zero
_C.MVIT.REL_POS_ZERO_INIT = False

# If True, using Residual Pooling connection
_C.MVIT.RESIDUAL_POOLING = False

# Dim mul in qkv linear layers of attention block instead of MLP
_C.MVIT.DIM_MUL_IN_ATT = False

# If True, using separate linear layers for Q, K, V in attention blocks.
_C.MVIT.SEPARATE_QKV = False

# The initialization scale factor for the head parameters.
_C.MVIT.HEAD_INIT_SCALE = 1.0

# Whether to use the mean pooling of all patch tokens as the output.
_C.MVIT.USE_MEAN_POOLING = False

# If True, use frozen sin cos positional embedding.
_C.MVIT.USE_FIXED_SINCOS_POS = False


# ---------------------------------------------------------------------------- #
# Multigrid training options
# See https://arxiv.org/abs/1912.00998 for details about multigrid training.
# ---------------------------------------------------------------------------- #

_C.MULTIGRID = CfgNode()

# Multigrid training allows us to train for more epochs with fewer iterations.
# This hyperparameter specifies how many times more epochs to train.
# The default setting in paper trains for 1.5x more epochs than baseline.
_C.MULTIGRID.EPOCH_FACTOR = 1.5

# Enable short cycles.
_C.MULTIGRID.SHORT_CYCLE = False

# Short cycle additional spatial dimensions relative to the default crop size.
_C.MULTIGRID.SHORT_CYCLE_FACTORS = [0.5, 0.707]
_C.MULTIGRID.LONG_CYCLE = False

# (Temporal, Spatial) dimensions relative to the default shape.
_C.MULTIGRID.LONG_CYCLE_FACTORS = [
    [0.25, 0.707],
    [0.5, 0.707],
    [0.5, 1],
    [1, 1],
]

# While a standard BN computes stats across all examples in a GPU,
# for multigrid training we fix the number of clips to compute BN stats on.
# See https://arxiv.org/abs/1912.00998 for details.
_C.MULTIGRID.BN_BASE_SIZE = 8

# Multigrid training epochs are not proportional to actual training time or
# computations, so _C.TRAIN.EVAL_PERIOD leads to too frequent or rare
# evaluation. We use a multigrid-specific rule to determine when to evaluate:
# This hyperparameter defines how many times to evaluate a model per long
# cycle shape.
_C.MULTIGRID.EVAL_FREQ = 3

# No need to specify; Set automatically and used as global variables.
_C.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = 0
_C.MULTIGRID.DEFAULT_B = 0
_C.MULTIGRID.DEFAULT_T = 0
_C.MULTIGRID.DEFAULT_S = 0

def assert_and_infer_cfg(cfg):
    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    return cfg

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()

