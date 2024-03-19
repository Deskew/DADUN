_base_=[
        "../_base_/real_data.py",
        "../_base_/davis.py",
        "../_base_/default_runtime.py"
        ]

cr = 10
resize_h,resize_w = 256,256

train_pipeline = [ 
    dict(type='RandomResize'),
    dict(type='RandomCrop',crop_h=resize_h,crop_w=resize_w,random_size=True),
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Flip', direction='diagonal',flip_ratio=0.5,),
    dict(type='Resize', resize_h=resize_h,resize_w=resize_w),
]

gene_meas = dict(type='GenerationGrayMeas')

train_data = dict(
    type="DavisData",
    data_root="/home/lbs/dataset/SCI/DAVIS/JPEGImages/480p",
    mask_path="test_datasets/mask/real_mask.mat",#
    mask_shape=(resize_h,resize_w,cr),
    pipeline=train_pipeline,
    gene_meas = gene_meas,
)

real_data = dict(
    data_root="test_datasets/real_data/cr10",
    cr=cr
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)

model = dict(
    type='RNN_ViT_SCI',# RNN_ViT_SCI', #'
    color_channels=1,
    num_filters=64, #128+10, 64+10, 64+5
    num_iterations=5
)

eval=dict(
    flag=False,
    interval=1
)

checkpoints="work_dirs/dun_rnnvit_real_cr10/checkpoints/epoch_61.pth" #None #