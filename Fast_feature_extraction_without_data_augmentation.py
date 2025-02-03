from keras.applications import VGG16  # 從 Keras 的預訓練模型中匯入 VGG16，這是一個已經在 ImageNet 上訓練過的深度學習模型，可用於影像分類任務。
import os  # 匯入 os 模組，方便操作檔案和資料夾路徑。
import numpy as np  # 匯入 numpy 模組，用來進行數值運算，尤其是矩陣和陣列操作。
from keras.preprocessing.image import ImageDataGenerator  # 匯入影像資料生成器，用來自動讀取圖片並進行預處理（如正規化等）。
from keras.optimizers import Adam  # 匯入 Adam 優化器，這是一種常用於訓練深度學習模型的優化算法。

# 問題：TensorFlow 預設會一次性佔滿所有 GPU 記憶體，這可能導致記憶體不足。
# 解決方法：在載入模型之前，啟用 GPU 動態記憶體分配，讓 TensorFlow 只使用所需的記憶體。
import tensorflow as tf  # 匯入 TensorFlow 模組，Keras 是基於 TensorFlow 構建的高階 API。

gpus = tf.config.experimental.list_physical_devices('GPU')  # 列出系統中可用的 GPU 裝置。
if gpus:  # 如果有偵測到 GPU。
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # 啟用記憶體增長模式，讓 TensorFlow 只按需分配 GPU 記憶體。
    except RuntimeError as e:
        print(e)  # 如果設定記憶體增長時發生錯誤，印出錯誤訊息。

# 實作 VGG16 卷積基底（convolutional base），不包含頂層（即全連接層）。
conv_base = VGG16(weights='imagenet',  # 使用在 ImageNet 上預訓練的權重。
                  include_top=False,  # 不包含頂層（即不包含全連接層，只使用卷積層來萃取特徵）。
                  input_shape=(128, 128, 3))  # 設定輸入影像的尺寸為 128x128，3 代表 RGB 三個通道。
conv_base.summary()  # 顯示模型的結構摘要，讓我們了解模型的層次結構和參數數量。

# 設定資料夾路徑，這些資料夾中分別存放訓練、驗證和測試用的圖片。
base_dir = 'cats_and_dogs_small'  # 資料集的主資料夾名稱。
train_dir = os.path.join(base_dir, 'train')  # 訓練集資料夾路徑。
validation_dir = os.path.join(base_dir, 'validation')  # 驗證集資料夾路徑。
test_dir = os.path.join(base_dir, 'test')  # 測試集資料夾路徑。

# 使用預先訓練的卷積基底（conv_base）來萃取特徵。
datagen = ImageDataGenerator(rescale=1./255)  # 將影像的像素值正規化到 [0, 1] 之間，這樣有助於加速模型收斂。
batch_size = 10  # 設定每批次處理的圖片數量。

def extract_features(directory, sample_count):  # 定義一個函數來萃取指定資料夾中的影像特徵。
    features = np.zeros(shape=(sample_count, 4, 4, 512))  # 建立一個空的陣列來存放特徵，形狀根據 VGG16 輸出的特徵圖大小（4x4x512）。
    labels = np.zeros(shape=(sample_count))  # 建立一個空的陣列來存放標籤（0 或 1，代表貓或狗）。
    generator = datagen.flow_from_directory(directory,  # 從指定資料夾讀取圖片。
                                            target_size=(128, 128),  # 調整圖片大小為 128x128。
                                            batch_size=batch_size,  # 每次讀取 batch_size 張圖片。
                                            class_mode='binary')  # 使用二元分類模式，因為我們只有兩個類別（貓和狗）。
    i = 0  # 初始化計數器。
    for inputs_batch, labels_batch in generator:  # 從資料生成器中取得一批圖片和對應的標籤。
        features_batch = conv_base.predict(inputs_batch)  # 使用 VGG16 卷積基底萃取圖片特徵。
        features[i * batch_size : (i + 1) * batch_size] = features_batch  # 將特徵儲存到預先建立的陣列中。
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch  # 將標籤儲存到陣列中。
        i += 1  # 更新計數器。
        print(i, end=' ')  # 印出進度，方便追蹤萃取進度。
        if i * batch_size >= sample_count:  # 如果已經處理完指定數量的樣本，跳出迴圈。
            break
    return features, labels  # 回傳特徵和標籤。

# 分別萃取訓練集、驗證集和測試集的特徵。
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

# 將特徵展平成一維向量，以便輸入到全連接層中進行分類。
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))  # 將 4x4x512 展平成 8192 維向量。
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

# 定義並訓練密集連接的分類器。
from keras import models  # 匯入 Keras 模型模組。
from keras import layers  # 匯入 Keras 層模組。
from keras import optimizers  # 匯入 Keras 優化器模組。

model = models.Sequential()  # 建立一個線性堆疊的模型（Sequential 模型）。
model.add(layers.Dense(64, activation='relu', input_dim=4 * 4 * 512))  # 新增一個全連接層（Dense），有 64 個神經元，使用 ReLU 激活函數。
model.add(layers.Dropout(0.5))  # 新增 Dropout 層，隨機丟棄 50% 的神經元，防止過擬合。
model.add(layers.Dense(1, activation='sigmoid'))  # 新增一個輸出層，有 1 個神經元，使用 Sigmoid 激活函數進行二元分類。

# 編譯模型，指定優化器、損失函數和評估指標。
model.compile(optimizer=Adam(learning_rate=1e-4),  # 使用 Adam 優化器，學習率設為 0.0001。
              loss='binary_crossentropy',  # 使用二元交叉熵作為損失函數，適合二元分類問題。
              metrics=['acc'])  # 使用準確率作為評估指標。

# 訓練模型，並將訓練過程中的準確率和損失記錄下來。
history = model.fit(train_features,  # 輸入訓練特徵。
                    train_labels,  # 輸入訓練標籤。
                    epochs=30,  # 訓練 30 個世代。
                    batch_size=20,  # 每批次使用 20 筆資料進行訓練。
                    validation_data=(validation_features, validation_labels))  # 驗證資料。

# 繪製訓練和驗證的準確率與損失變化圖。
import matplotlib.pyplot as plt  # 匯入繪圖模組。

acc = history.history['acc']  # 取得訓練準確率。
val_acc = history.history['val_acc']  # 取得驗證準確率。
loss = history.history['loss']  # 取得訓練損失。
val_loss = history.history['val_loss']  # 取得驗證損失。

epochs = range(1, len(acc) + 1)  # 設定 X 軸為世代數。

plt.figure()  # 建立新圖表。
plt.plot(epochs, acc, 'bo', label='Training acc')  # 繪製訓練準確率，'bo' 代表藍色圓點。
plt.plot(epochs, val_acc, 'b', label='Validation acc')  # 繪製驗證準確率，'b' 代表藍色實線。
plt.title('Training and validation accuracy')  # 設定圖表標題。
plt.legend()  # 顯示圖例。
plt.savefig('Train_and_Validation_accuracy(WITHOUT_data_augmentation).png')  # 將圖表儲存為 PNG 圖檔。

plt.figure()  # 建立新圖表。
plt.plot(epochs, loss, 'bo', label='Training loss')  # 繪製訓練損失。
plt.plot(epochs, val_loss, 'b', label='Validation loss')  # 繪製驗證損失。
plt.title('Training and validation loss')  # 設定圖表標題。
plt.legend()  # 顯示圖例。
plt.savefig('Train_and_Validation_loss(WITHOUT_data_augmentation).png')  # 將圖表儲存為 PNG 圖檔。
