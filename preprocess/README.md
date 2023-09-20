## Extract features using ULIP-2

- First, we follow the instructions in [vil3dref](https://github.com/cshizhe/vil3dref/tree/main/preprocess/scannetv2) to preprocess the ScanNet data.
- Second, we utilize [process_pcd.py](preprocess/process_pcd.py) to extract the point cloud of each object from all the scenes and save them in the "pcd_by_instance" directory.
- Third, we download the pretrained ULIP-2 checkpoint file named "pointbert_ULIP-2.pt" and then employ [extract_3dfeat.py](preprocess/extract_3dfeat.py) to extract features for all the instances (objects) in the "pcd_by_instance" directory. You can use the following script:
```shell
python extract_3dfeat.py --test_ckpt_addr pointbert_ULIP-2.pt \
                                            --pcds_dir /path/to/pcd_by_instance \
                                            --output_feat_dir /path/to/pcd_feats_ulip2
```
The result, "scannet_ulip2_feats.pt," is a file that contains the combined features of all the objects. Please note that "extract_3dfeat.py" is based on the ULIP model, so you'll need to refer to the [ULIP](https://github.com/salesforce/ULIP) repository to ensure it runs correctly.