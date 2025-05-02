batch_size = 8
learning_rate = 1e-3
num_epochs = 100

image_dir = "./datasets/images"
label_dir = "./datasets/labels"
geojson_dir = "./datasets/geojson"
outputs = "./outputs"

model_save_path = "./model_save_path"

max_thin_iters = 1

distance_transform_weight = 32

# basic | thin | distance_transform | thin_distance_transforms
# model_id='basic'

# bce | distance_transform | bce_distance_transform
loss_fn = 'bce'