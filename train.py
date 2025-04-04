import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 # Не preprocess_input, так как Rescaling в модели
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --- 1. Конфигурация ---
DATASET_PATH = 'dataset-resized'  # Путь к папке dataset-resized
IMAGE_SIZE = (224, 224) # Размер изображений для MobileNetV2
BATCH_SIZE = 32         # Количество изображений в одном батче
# N_CLASSES = 6 # Определим автоматически из данных
EPOCHS = 25             # Максимальное количество эпох обучения (можно увеличить)
MODEL_SAVE_PATH = 'best_garbage_classifier.keras' # Имя файла для сохранения лучшей модели

# --- 2. Загрузка и подготовка данных ---
print("Загрузка данных...")
# Загрузка данных с разделением на обучающую (80%) и валидационную (20%) выборки
train_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123, # для воспроизводимости разделения
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical' # Метки в формате one-hot encoding ([0,0,1,0,0,0])
)

val_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Получение имен и количества классов
class_names = train_ds.class_names
N_CLASSES = len(class_names)
print(f"Найдены классы ({N_CLASSES} шт.):", class_names)

# Слои аугментации данных
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")

# Применение аугментации и оптимизация датасетов
def prepare(ds, augment=False):
    # Сначала масштабируем в [0, 1] (для аугментации)
    rescale_0_1 = Rescaling(1./255)
    ds = ds.map(lambda x, y: (rescale_0_1(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Затем применяем аугментацию (только для train)
    if augment:
         ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    # Оптимизация производительности
    return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = prepare(train_ds, augment=True)
val_ds = prepare(val_ds) # Валидационный без аугментации

# --- 3. Создание модели (Transfer Learning) ---
print("Создание модели...")
# Загрузка базовой модели MobileNetV2
base_model = MobileNetV2(input_shape=IMAGE_SIZE + (3,),
                         include_top=False,
                         weights='imagenet')

# Замораживаем веса базовой модели
base_model.trainable = False

# Создаем "голову" модели
inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
# ВАЖНО: Добавляем масштабирование в [-1, 1] как первый слой,
# так как MobileNetV2 ожидает именно этот диапазон.
# Данные из генератора приходят в [0, 1] после нашей функции prepare.
x = Rescaling(1./127.5, offset=-1)(inputs) # Масштаб [0,1] -> [-1,1]
# Подаем на вход базовой модели
x = base_model(x, training=False) # training=False важно для замороженных слоев и BatchNorm
# Слой усреднения признаков
x = GlobalAveragePooling2D()(x)
# Слой Dropout
x = Dropout(0.3)(x)
# Выходной полносвязный слой
outputs = Dense(N_CLASSES, activation='softmax')(x)

# Собираем полную модель
model = Model(inputs, outputs)

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 4. Обучение модели ---
# Колбэки
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH,
                             monitor='val_accuracy',
                             save_best_only=True,
                             verbose=1)

early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=5, # Остановить после 5 эпох без улучшения
                               restore_best_weights=True, # Вернуть лучшие веса
                               verbose=1)

print("\n--- Начало обучения ---")
start_time = time.time()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[checkpoint, early_stopping]
)

end_time = time.time()
print(f"\nОбучение завершено за {(end_time - start_time):.2f} секунд.")
print(f"Лучшая модель сохранена в файл: {MODEL_SAVE_PATH}")

# --- 5. (Опционально) Визуализация обучения ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
# Определяем фактическое количество эпох (если сработал EarlyStopping)
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Точность на обучении')
plt.plot(epochs_range, val_acc, label='Точность на валидации')
plt.legend(loc='lower right')
plt.title('Точность обучения и валидации')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Потери на обучении')
plt.plot(epochs_range, val_loss, label='Потери на валидации')
plt.legend(loc='upper right')
plt.title('Потери обучения и валидации')
plt.suptitle("Результаты обучения") # Общий заголовок
plt.savefig("training_history.png") # Сохраняем график в файл
print("График истории обучения сохранен в training_history.png")
# plt.show() # Раскомментируйте, если хотите показать график сразу