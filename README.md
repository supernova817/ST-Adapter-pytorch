# pytorch usage for ST-Adapter
![image](https://user-images.githubusercontent.com/58262251/226808005-4b6b1339-3a90-4a12-91c8-481b556cea09.png)


## Usage:
after model build
```python
model = VitModel(~~)

# ver1
for name, param in model.named_parameters():
    param.requires_grad = False
    if 'dw3dcnn' in name:
        param.requires_grad = True
    elif 'downscale' in name:
        param.requires_grad = True
    elif 'upscale' in name:
        param.requires_grad = True
        
# ver2
for name, param in model.named_parameters():
    param.requires_grad = False
    if 'st_dapter' in name:
        param.requires_grad = True

pred = model(x)
~~

```


## Citation:
```
@article{pan2022st,
  title={ST-Adapter: Parameter-Efficient Image-to-Video Transfer Learning for Action Recognition},
  author={Pan, Junting and Lin, Ziyi and Zhu, Xiatian and Shao, Jing and Li, Hongsheng},
  journal={arXiv preprint arXiv:2206.13559},
  year={2022}
}
```
