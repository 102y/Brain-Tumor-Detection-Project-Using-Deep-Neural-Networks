import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import cv2

warnings.filterwarnings('ignore')

# تحميل البيانات
tumor = []
no_tumor = []

for file in glob.iglob(r"C:\\Users\\NITRO\\Desktop\\AI Proj\\Brain Tumor Detection\\archive\\yes\\*"):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    tumor.append((img, 1))

for file in glob.iglob(r"C:\\Users\\NITRO\\Desktop\\AI Proj\\Brain Tumor Detection\\archive\\no\\*"):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    no_tumor.append((img, 0))

data = tumor + no_tumor
x = np.array([i[0] for i in data])
y = np.array([i[1] for i in data])

# خلط البيانات
x, y = shuffle(x, y, random_state=101)

# تقسيم البيانات
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.15, random_state=42, stratify=y)
x_train = x_train / 255.0
x_test = x_test / 255.0

# توسيع البيانات
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# بناء النموذج
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_uniform', input_shape=(150, 150, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.1),
    
    keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_uniform', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_uniform', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.2),
    
    keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_uniform', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(512, (3, 3), kernel_initializer='he_uniform', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.3),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# إعداد callback لإيقاف التدريب
class NewCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.995:
            self.model.stop_training = True
            
stop_epoch = NewCallback()

# تدريب النموذج
model_hist = model.fit(train_datagen.flow(x_train, y_train, batch_size=32), 
                        steps_per_epoch=len(x_train) // 32, 
                        epochs=12, 
                        callbacks=[stop_epoch])

# تقييم النموذج
model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
y_pred = np.round(y_pred, 0)

# تقرير التصنيف
print(classification_report(y_test, y_pred))

# مصفوفة الارتباك
def create_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.show()
    return cm

cm = create_confusion_matrix(y_test, y_pred)

# حفظ النموذج
model.save('BrainTumor.h5')

# دالة لتحليل صورة خارجية وتحديد مكان الورم
def predict_and_locate_tumor(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (150, 150))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    
    if prediction >= 0.5:
        print("Tahmin sonucu: Görüntüde tümör var")
        label = "tümör var"
        
        # تقسيم الصورة إلى مربعات صغيرة وتحليل كل مربع لتحديد مكان الورم
        step_size = 50  # حجم المربعات
        found = False
        for y in range(0, img.shape[0] - step_size, step_size):
            for x in range(0, img.shape[1] - step_size, step_size):
                # أخذ جزء صغير من الصورة
                sub_image = img[y:y+step_size, x:x+step_size]
                sub_image_resized = cv2.resize(sub_image, (150, 150)) / 255.0
                sub_image_resized = np.expand_dims(sub_image_resized, axis=0)
                
                # توقع على الجزء الصغير
                sub_pred = model.predict(sub_image_resized)[0][0]
                
                # إذا كان التنبؤ يشير إلى وجود ورم، نرسم مربعًا
                if sub_pred >= 0.5:
                    cv2.rectangle(img_rgb, (x, y), (x + step_size, y + step_size), (255, 0, 0), 2)
                    found = True
                    break
            if found:
                break
    else:
        print("Tahmin sonucu: Görüntü normal ve tümör içermiyor")
        label = "tümör yok"
    
    # عرض الصورة مع النتيجة
    plt.imshow(img_rgb)
    plt.title(f"Sonuç Result : {label}")
    plt.axis('on')
    plt.show()

# استدعاء الدالة لاختبار صورة خارجية وتحديد مكان الورم
predict_and_locate_tumor(r'C:\\Users\\NITRO\\Desktop\\AI Proj\\Brain Tumor Detection\\archive\\yes\\y0.jpg')
