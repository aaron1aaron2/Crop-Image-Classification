# Crop-Image-Classification (AIcup 2022)

# Reproduce
## 📁 Folder schema 
```
Crop-Image-Classification
    |-- data 
        |-- input: 整理過的資料，包含目標土地 & 時價登入
        |-- train_data: 訓練用資料

    |-- doc: 相關文件

    |-- model: 模型

    |-- notebook: 分析視覺化

    |-- output: 輸出 log 和 model

    |-- scripts: shell or batch 腳本，包含批次跑實驗、訓練範例

    |-- tests: 測試檔

    |- requirements.txt: python 依賴套件
    |- data_helper.py: 將輸入資料(data/input)依造參數設定檔(configs)轉換成訓練資料(data/train_data)
    |- train.py: 主要訓練程式碼
```
## 🖥️ Environment settings 
### `pytorch`
```shell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
- 本專案是在 window 11、cuda(11.6)、pytorch(1.13.0)測試。
- 如使用不同環境請到 [pytorch 官網](https://pytorch.org/) 選擇對應版的指令。

### `other packages`
```shell
pip3 install -r requirements.txt
```
## 🙋 Quick start 
### `Step1: 資料準備`


### `Step2: 產生訓練資料`


### `Step3: 開始訓練`