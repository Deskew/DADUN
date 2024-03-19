_base_=[
        "../_base_/davis_bayer.py",
        "../_base_/matlab_bayer.py",
        "../_base_/default_runtime.py"
        ]
test_data = dict(
    data_root="test_datasets/middle_scale",
    mask_path="test_datasets/mask/mid_color_mask.mat",
    rot_flip_flag=True
)
resize_h,resize_w = 256,256#64,64#160,160
train_pipeline = [ 
    dict(type='RandomResize'),
    dict(type='RandomCrop',crop_h=resize_h,crop_w=resize_w,random_size=True),
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Flip', direction='diagonal',flip_ratio=0.5,),
    dict(type='Resize', resize_h=resize_h,resize_w=resize_w),
]
train_data = dict(
    mask_path = None,
    mask_shape = (resize_h,resize_w,8),
    pipeline = train_pipeline
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

model = dict(
    type='RNN_ViT_SCI',
    color_channels=3,
    num_filters=64, 
    num_iterations=5 #5,8
)
 
eval=dict(
    flag=False,#True,
    interval=1
)

checkpoints=None#"checkpoints/dun_rnnvit/epoch_199.pth" #