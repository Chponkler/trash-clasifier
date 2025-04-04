import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- Конфигурация модели ---
MODEL_SAVE_PATH = 'best_garbage_classifier.keras'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# --- Загрузка модели ---
try:
    model = load_model(MODEL_SAVE_PATH)
    print("Модель успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    exit()


# --- Функция предсказания ---
def predict_user_image(image_path):
    try:
        img = image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_batch)
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = np.max(predictions[0]) * 100
        return predicted_class, confidence
    except Exception as e:
        print(f"Ошибка: {e}")
        return None, None


# --- GUI Интерфейс ---
class GarbageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Классификатор мусора")
        self.root.geometry("500x400")

        # Основной фрейм
        self.main_frame = tk.Frame(root, padx=20, pady=20)
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        # Заголовок
        self.label = tk.Label(
            self.main_frame,
            text="Выберите изображение для классификации",
            font=('Helvetica', 14)
        )
        self.label.pack(pady=10)

        # Кнопка выбора файла
        self.select_button = tk.Button(
            self.main_frame,
            text="Выбрать файл",
            command=self.load_image,
            bg="#4CAF50",
            fg="white",
            font=('Helvetica', 12),
            padx=10,
            pady=5
        )
        self.select_button.pack(pady=20)

        # Превью изображения
        self.image_label = tk.Label(self.main_frame)
        self.image_label.pack(pady=10)

        # Результаты классификации
        self.result_label = tk.Label(
            self.main_frame,
            text="",
            font=('Helvetica', 12),
            wraplength=400
        )
        self.result_label.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png")]
        )

        if file_path:
            try:
                # Показываем превью изображения
                img = Image.open(file_path)
                img.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo)
                self.image_label.image = photo

                # Классификация
                predicted_class, confidence = predict_user_image(file_path)

                if predicted_class and confidence:
                    result_text = (
                        f"Результат классификации:\n"
                        f"Тип мусора: {predicted_class}\n"
                        f"Точность: {confidence:.2f}%"
                    )
                    self.result_label.config(text=result_text)
                else:
                    self.result_label.config(text="Ошибка при обработке изображения")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")


# --- Запуск приложения ---
if __name__ == "__main__":
    root = tk.Tk()
    app = GarbageClassifierApp(root)
    root.mainloop()