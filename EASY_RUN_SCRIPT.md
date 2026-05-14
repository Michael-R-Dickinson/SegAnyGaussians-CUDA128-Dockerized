Train the 3D Gaussian Splat Scene:
- `--images` param specifies which directory inside the scene to source images from. So `--images images_8` tell it to source images from `data/360_extra_scenes/flowers/images_8`. In this case `images_8` has 8x downsampled images so we want to use this as the originals are 5k resolution.
- `--model_path` param specifies the output directory. If not specified it gives it a random name inside `output/`

```
 python train_scene.py -s data/360_extra_scenes/flowers --images images_8 --iterations 10000 --model_path output/flowers_10k_iterations
```

Extract SAM Masks from the images:
- Downsamples the original 5k images before running SAM
```
python extract_segment_everything_masks.py --image_root data/360_extra_scenes/flowers --sam_checkpoint_path sam_vit_h_4b8939.pth --downsample 8
```

Get scale values from the SAM Masks (this is described in the paper but basically just takes the SAM masks and figures out how big they are in 3d space):
```
python get_scale.py --image_root data/360_extra_scenes/flowers --model_path output/flowers_10k_iterations
```

Train contrastive features (getting the actual affinity embeddings):
```
python train_contrastive_feature.py -m output/flowers_10k_iterations --iterations 10000 --num_sampled_rays 1000
```

Run the GUI!
```
python saga_gui.py --model_path output/flowers_10k_iterations --scene_iteration 10000 --feature_iteration 10000
```

