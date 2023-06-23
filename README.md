# Bodypart Latent From Video (BLFV)

BLFV 是一個以 Python 編寫的專案，主要的功能是標記影片中的特定身體部位並生成對應的 latent vectors。

## 特性

- 生成特定身體部位的影像標記
- 生成與特定身體部位相關的 latent vectors
- 利用生成的 latent vectors 重建影片

## 安裝

1. 複製此專案至本地端

    ```
    git clone https://github.com/RaisoLiu/BLFV-GUI.git
    ```

2. 安裝所需的依賴套件

    ```
    pip install -r requirements.txt
    ```

## 使用方法

1. 執行 `main.py` 並提供影片檔案的名稱，例如：

    ```
    python main.py
    ```

2. 使用工具列開啟影片

3. 使用工具列載入模型 

4. 使用 'PCA1d' 與 'PCA3D' 將影片上的 patchs 分群

## Note

在 thershold 中可以使用鍵盤的上下鍵調整數值。
在影片的瀏覽中，可以使用 左右鍵 切換前後幀，或是拉進度條。
