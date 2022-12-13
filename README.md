# Synthesizing Coherent Story with Auto-Regressive Latent Diffusion Models
Our code is implemented based on an internal toolkit of Alibaba Group. We provide the core code regarding algorithms here. We plan to pack it using Pytorch Lightning and release the full training and inference code soon within 1-2 weeks.

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
python data_script/vist_img_download.py --json_path /path/to/dii_json_files --img_path /path/to/save/images --num_process 32
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