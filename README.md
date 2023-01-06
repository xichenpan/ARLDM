# Synthesizing Coherent Story with Auto-Regressive Latent Diffusion Models

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synthesizing-coherent-story-with-auto/story-visualization-on-pororo)](https://paperswithcode.com/sota/story-visualization-on-pororo?p=synthesizing-coherent-story-with-auto) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synthesizing-coherent-story-with-auto/story-continuation-on-pororosv)](https://paperswithcode.com/sota/story-continuation-on-pororosv?p=synthesizing-coherent-story-with-auto) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synthesizing-coherent-story-with-auto/story-continuation-on-flintstonessv)](https://paperswithcode.com/sota/story-continuation-on-flintstonessv?p=synthesizing-coherent-story-with-auto) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synthesizing-coherent-story-with-auto/story-continuation-on-vist)](https://paperswithcode.com/sota/story-continuation-on-vist?p=synthesizing-coherent-story-with-auto)

This version is immigrated from a internal implementation of Alibaba Group, feel free to open an issue to address any problem!

## Environment
```shell
conda create -n arldm python=3.8
conda activate arldm
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
git clone https://github.com/Flash-321/ARLDM.git
cd ARLDM
pip install -r requirements.txt
```
## Data Preparation
* Download the PororoSV dataset [here](https://drive.google.com/file/d/11Io1_BufAayJ1BpdxxV2uJUvCcirbrNc/view?usp=sharing).
* Download the FlintstonesSV dataset [here](https://drive.google.com/file/d/1kG4esNwabJQPWqadSDaugrlF4dRaV33_/view?usp=sharing).
* Download the VIST-SIS url links [here](https://visionandlanguage.net/VIST/json_files/story-in-sequence/SIS-with-labels.tar.gz)
* Download the VIST-DII url links [here](https://visionandlanguage.net/VIST/json_files/description-in-isolation/DII-with-labels.tar.gz)
* Download the VIST images running
```shell
python data_script/vist_img_download.py
--json_dir /path/to/dii_json_files
--img_dir /path/to/save_images
--num_process 32
```
* To accelerate I/O, using the following scrips to convert your downloaded data to HDF5
```shell
python data_script/pororo_hdf5.py
--data_dir /path/to/pororo_data
--save_path /path/to/save_hdf5_file

python data_script/flintstones_hdf5.py
--data_dir /path/to/flintstones_data
--save_path /path/to/save_hdf5_file

python data_script/vist_hdf5.py
--sis_json_dir /path/to/sis_json_files
--dii_json_dir /path/to/dii_json_files
--img_dir /path/to/vist_images
--save_path /path/to/save_hdf5_file
 ```

## Training
Specify your directory and device configuration in `config.yaml` and run
```shell
python main.py
```
## Sample
Specify your directory and device configuration in `config.yaml` and run
```shell
python main.py
```
## Citation
If you find this code useful for your research, please cite our paper:
```bibtex
@article{pan2022synthesizing,
  title={Synthesizing Coherent Story with Auto-Regressive Latent Diffusion Models},
  author={Pan, Xichen and Qin, Pengda and Li, Yuhong and Xue, Hui and Chen, Wenhu},
  journal={arXiv preprint arXiv:2211.10950},
  year={2022}
}
```
