# 在 convolutional base 卷積基底上增加密集層分類器
from keras import models  # 匯入 Keras 的模型模組，用於建立神經網路模型
from keras import layers  # 匯入 Keras 的層模組，用來建立不同類型的神經網路層
from keras.applications import VGG16  # 匯入 Keras 的 VGG16 預訓練模型
import os  # 匯入 os 模組，用於操作系統路徑
import numpy as np  # 匯入 numpy 模組，用來進行數值計算和處理
from keras.preprocessing.image import ImageDataGenerator  # 匯入影像數據生成器，用於資料增強
from keras import optimizers  # 匯入優化器模組，用於模型訓練的優化
from keras.optimizers import Adam  # 匯入 Adam 優化器，常用於深度學習模型的訓練

# 問題：TensorFlow 預設會一次性佔滿所有 GPU 記憶體，這可能導致記憶體不足。
# 解決方法：在載入模型之前，啟用 GPU 動態記憶體分配，讓 TensorFlow 只使用所需的記憶體。
import tensorflow as tf  # 匯入 TensorFlow 深度學習框架

# 檢查是否有可用的 GPU 設備
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # 啟用 GPU 記憶體動態分配，避免一次佔滿所有記憶體
    except RuntimeError as e:
        print(e)  # 如果記憶體分配設置失敗，輸出錯誤訊息

# 載入 VGG16 預訓練模型，作為卷積基底 (convolutional base)
conv_base = VGG16(weights='imagenet',  # 使用在 ImageNet 上預訓練的權重
                  include_top=False,  # 不包含原始 VGG16 的全連接分類層，僅保留卷積層
                  input_shape=(128, 128, 3))  # 定義輸入圖片大小為 128x128 像素，3 色通道 (RGB)

# 建立一個新的序列模型 (Sequential)
model = models.Sequential()
model.add(conv_base)  # 將 VGG16 卷積基底加入模型中
model.add(layers.Flatten())  # 將多維的特徵圖攤平成一維向量，以便接入全連接層
model.add(layers.Dense(64, activation='relu'))  # 加入一層全連接層，64 個神經元，使用 ReLU 激活函數
model.add(layers.Dense(1, activation='sigmoid'))  # 最後加入一個神經元，使用 sigmoid 激活函數，適用於二元分類
model.summary()  # 輸出模型的結構摘要，檢視各層的參數

# 凍結卷積基底神經網路，防止其權重在訓練過程中被更新
print('This is the number of trainable weights before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False  # 將卷積基底設為不可訓練，這樣只會訓練新增的全連接層

print('This is the number of trainable weights after freezing the conv base:', len(model.trainable_weights))

# 定義資料夾路徑
base_dir = 'cats_and_dogs_small'  # 基本資料夾路徑
train_dir = os.path.join(base_dir, 'train')  # 訓練資料夾路徑
validation_dir = os.path.join(base_dir, 'validation')  # 驗證資料夾路徑
test_dir = os.path.join(base_dir, 'test')  # 測試資料夾路徑

# 以凍結的卷積基底進行完整模型訓練
# 定義訓練數據的增強方式，這有助於減少過擬合 (overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 將像素值縮放到 0~1 之間
    rotation_range=40,  # 隨機旋轉圖像範圍 (0~40 度)
    width_shift_range=0.2,  # 隨機水平平移圖像 (最大 20% 寬度)
    height_shift_range=0.2,  # 隨機垂直平移圖像 (最大 20% 高度)
    shear_range=0.2,  # 隨機錯切變換圖像
    zoom_range=0.2,  # 隨機縮放圖像
    horizontal_flip=True,  # 隨機水平翻轉圖像
    fill_mode='nearest'  # 填充新像素的策略
)

# 驗證數據不需要進行資料增強，僅需正規化
test_datagen = ImageDataGenerator(rescale=1./255)

# 生成訓練數據集
train_generator = train_datagen.flow_from_directory(
    train_dir,  # 目標目錄路徑
    target_size=(128, 128),  # 調整所有圖像大小為 128x128
    batch_size=10,  # 每批次生成 10 張圖像
    class_mode='binary'  # 因為使用二元交叉熵損失函數，所以需要二元標籤 (0 或 1)
)

# 生成驗證數據集
validation_generator = test_datagen.flow_from_directory(
    validation_dir,  # 驗證數據的目錄路徑
    target_size=(128, 128),  # 調整圖像大小
    batch_size=10,  # 每批次生成 10 張圖像
    class_mode='binary'  # 二元分類標籤
)

# 編譯模型，設定損失函數、優化器和評估指標
model.compile(
    loss='binary_crossentropy',  # 使用二元交叉熵作為損失函數，適用於二元分類
    optimizer=Adam(learning_rate=1e-4),  # 使用 Adam 優化器，並設定學習率為 0.0001
    metrics=['acc']  # 評估模型表現的指標為準確率 (accuracy)
)

# 訓練模型
history = model.fit(
    train_generator,  # 訓練數據生成器
    steps_per_epoch=100,  # 每個 epoch 執行 100 步 (即使用 1000 張訓練圖片)
    epochs=30,  # 訓練 30 個 epochs
    validation_data=validation_generator,  # 驗證數據生成器
    validation_steps=50  # 每個 epoch 執行 50 步驗證 (即使用 500 張驗證圖片)
)

# 繪製訓練和驗證過程的結果
import matplotlib.pyplot as plt  # 匯入 matplotlib 用於繪圖

# 從訓練歷史中提取準確率和損失值
acc = history.history['acc']  # 訓練準確率
val_acc = history.history['val_acc']  # 驗證準確率
loss = history.history['loss']  # 訓練損失
val_loss = history.history['val_loss']  # 驗證損失

epochs = range(1, len(acc) + 1)  # 定義 epoch 數據範圍

# 繪製訓練和驗證準確率
plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')  # 'bo' 表示藍色圓點
plt.plot(epochs, val_acc, 'b', label='Validation acc')  # 'b' 表示藍色實線
plt.title('Training and validation accuracy')  # 標題
plt.legend()  # 顯示圖例
plt.savefig('Train_and_Validation_accuracy(HAVING_data_augmentation).png')  # 儲存圖表

# 繪製訓練和驗證損失
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')  # 訓練損失藍色圓點
plt.plot(epochs, val_loss, 'b', label='Validation loss')  # 驗證損失藍色實線
plt.title('Training and validation loss')  # 標題
plt.legend()  # 顯示圖例
plt.savefig('Train_and_Validation_loss(HAVING_data_augmentation).png')  # 儲存圖表
