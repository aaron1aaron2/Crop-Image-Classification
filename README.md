# Crop-Image-Classification (AIcup 2022)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15kOuZZaUoDG33LCQCy-qHHHROab0ViiZ?usp=sharing)

# Model used - CoAtNet
![](doc/fig/image/Coatnet.png)
# Reproduce
## 📁 Folder schema 
```
Crop-Image-Classification
    |-- data 

    |-- doc: 相關文件

    |-- model: 模型

    |-- notebook: 分析視覺化

    |-- output: 輸出 log 和 model

    |-- scripts: shell or batch 腳本，包含批次跑實驗、訓練範例

    |- requirements.txt: python 依賴套件
    |- data_helper.py: 將輸入資料(data/input)依造參數設定檔(configs)轉換成訓練資料(data/train_data)
    |- train.py: 主要訓練程式碼
```
## 🖥️ Environment settings 
### `pytorch`
```shell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
- 如使用不同環境請到 [pytorch 官網](https://pytorch.org/) 選擇對應版的指令。

### `other packages`
```shell
pip3 install -r requirements.txt
```

## 🙋 Quick start 
這部分使用小樣本的資料，結果僅供參考
### `Step1: Clone 程式碼`
```shell
git clone https://github.com/aaron1aaron2/Crop-Image-Classification.git
cd Crop-Image-Classification
```

### `Step2: 資料準備`
```shell
source scripts/download_file.sh "1ew3d6llpvj7ev1CssUbxiZ3FcANFRaOW&" "sample100_L160(test)" "data/sample100_L160(test)"
```

也可以直接到 [Google drive](https://drive.google.com/uc?id=1ew3d6llpvj7ev1CssUbxiZ3FcANFRaOW&confirm=t) 下載，並解壓縮到 _data/sample100_L160(test)_ 底下。

### `Step3: 產生訓練資料`
```shell
python scripts/image_process.py
```

### `Step4: 開始訓練`
```shell
python train.py
```

# Citation
```bibtex
@article{dai2021coatnet,
  title={CoAtNet: Marrying Convolution and Attention for All Data Sizes},
  author={Dai, Zihang and Liu, Hanxiao and Le, Quoc V and Tan, Mingxing},
  journal={arXiv preprint arXiv:2106.04803},
  year={2021}
}
```

# Credits

Code adapted from [CoAtNet](https://github.com/chinhsuanwu/coatnet-pytorch)
