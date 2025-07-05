import cv2
import dlib
import pyautogui as pyms
import os
import csv
import shutil
import uuid
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models, callbacks
import tkinter as tk
from tkinter import messagebox
import sys
from tensorflow.keras.models import load_model

# Genel ayarlar
CSV_FILE = "eye_dataset.csv"
SAVE_DIR = "./eye_images"
GRID_ROWS = 5
GRID_COLS = 5
SAMPLES_PER_POINT = 40
LANDMARK_PATH = "shape_predictor_68_face_landmarks.dat"
pyms.FAILSAFE = False
SCREEN_W, SCREEN_H = pyms.size()

MODEL_PATH_X = "eye_tracker_model_x.keras"
MODEL_PATH_Y = "eye_tracker_model_y.keras"


# Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LANDMARK_PATH)


# --- Yardımcı Fonksiyonlar ---
def setup_directories():
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)

    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "x", "y"])


def get_grid_points():
    return [
        (
            j * (SCREEN_W // GRID_COLS) + (SCREEN_W // (GRID_COLS * 2)),
            i * (SCREEN_H // GRID_ROWS) + (SCREEN_H // (GRID_ROWS * 2))
        )
        for i in range(GRID_ROWS)
        for j in range(GRID_COLS)
    ]


def safe_crop(frame, min_x, min_y, max_x, max_y):
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(frame.shape[1], max_x)
    max_y = min(frame.shape[0], max_y)
    return frame[min_y:max_y, min_x:max_x]


def eye_aspect_ratio(eye_points):
    # tuple listesini numpy array'e çeviriyoruz
    eye_points = np.array(eye_points)

    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear


# --- 1. VERİ TOPLAMA ---
def capture_eye_data(cap, target_x, target_y, window_name):
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)

        for _ in range(SAMPLES_PER_POINT):
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                try:
                    landmarks = predictor(gray, face)
                    right_eye = [(landmarks.part(i).x, landmarks.part(i).y)
                                 for i in range(36, 42)]

                    min_x = min(p[0] for p in right_eye) - 10
                    min_y = min(p[1] for p in right_eye) - 10
                    max_x = max(p[0] for p in right_eye) + 10
                    max_y = max(p[1] for p in right_eye) + 10

                    eye_roi = safe_crop(frame, min_x, min_y, max_x, max_y)
                    if eye_roi.size == 0:
                        continue

                    eye_img = cv2.resize(eye_roi, (100, 50))
                    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)

                    img_name = f"eye_{target_x}_{target_y}_{
                        uuid.uuid4().hex[:8]}.jpg"
                    img_path = os.path.join(SAVE_DIR, img_name)
                    cv2.imwrite(img_path, eye_img)
                    writer.writerow([img_name, target_x, target_y])
                except:
                    continue

            # Tam ekran görüntüleme
            display_frame = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            display_frame[:] = (30, 30, 30)  # Koyu gri arka plan
            current_x, current_y = pyms.position()
            cv2.circle(display_frame, (current_x, current_y),
                       10, (0, 255, 0), -1)
            cv2.putText(display_frame, "VERI TOPLANIYOR...", (SCREEN_W//2-200, SCREEN_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(window_name, display_frame)

            if cv2.waitKey(1) == 27:
                break


def veri_topla():
    print("Veri toplama başladı...")
    setup_directories()
    cap = cv2.VideoCapture(0)

    # Tam ekran penceresi oluştur
    window_name = "Eye Data Collection"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        for target_x, target_y in get_grid_points():
            pyms.moveTo(target_x, target_y, duration=0.5)
            capture_eye_data(cap, target_x, target_y, window_name)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("Veri toplama tamamlandı.")

# --- 2. MODEL EĞİTİM ---


def load_and_preprocess(image_path, label):
    img = tf.io.read_file(SAVE_DIR + "/" + image_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (100, 50))

    x_norm = label[0] / SCREEN_W
    y_norm = label[1] / SCREEN_H

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)

    return img, tf.stack([x_norm, y_norm])


def create_datasets():
    df = pd.read_csv(CSV_FILE)
    df.dropna(inplace=True)
    labels = df[["x", "y"]].values.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((df["image_path"], labels))\
        .shuffle(1000).map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    train_size = int(0.8 * len(df))
    return (
        dataset.take(train_size).batch(32).prefetch(tf.data.AUTOTUNE),
        dataset.skip(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
    )


def build_model():
    model = models.Sequential([
        layers.Input(shape=(100, 50, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='linear')
    ])

    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        # metrics=['mae']
    )
    return model


def model_egit():
    print("Model eğitimi başladı (X ve Y ayrı)...")
    df = pd.read_csv(CSV_FILE)
    df.dropna(inplace=True)

    def load_for_x(image_path, label):
        img = tf.io.read_file(SAVE_DIR + "/" + image_path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (100, 50))
        img = tf.expand_dims(img, axis=-1)  # NHWC için
        x_norm = label[0] / SCREEN_W
        return img, x_norm

    def load_for_y(image_path, label):
        img = tf.io.read_file(SAVE_DIR + "/" + image_path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (100, 50))
        img = tf.expand_dims(img, axis=-1)  # NHWC için
        y_norm = label[1] / SCREEN_H
        return img, y_norm

    def make_model():
        model = models.Sequential([
            layers.Input(shape=(100, 50, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    # Dataset hazırlığı
    paths = df["image_path"].values
    labels = df[["x", "y"]].values.astype(np.float32)

    # X için
    ds_x = tf.data.Dataset.from_tensor_slices(
        (paths, labels)).map(load_for_x).shuffle(1000)
    train_x = ds_x.take(int(len(df) * 0.8)).batch(32)
    val_x = ds_x.skip(int(len(df) * 0.8)).batch(32)

    model_x = make_model()
    model_x.fit(train_x, validation_data=val_x, epochs=60, callbacks=[
        callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ])
    model_x.save(MODEL_PATH_X)
    print("X modeli kaydedildi.")

    # Y için
    ds_y = tf.data.Dataset.from_tensor_slices(
        (paths, labels)).map(load_for_y).shuffle(1000)
    train_y = ds_y.take(int(len(df) * 0.8)).batch(32)
    val_y = ds_y.skip(int(len(df) * 0.8)).batch(32)

    model_y = make_model()
    model_y.fit(train_y, validation_data=val_y, epochs=60, callbacks=[
        callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ])
    model_y.save(MODEL_PATH_Y)
    print("Y modeli kaydedildi.")


# YENİ AYARLAR
BLINK_THRESHOLD = 0.5  # Modelin kapalı göz için eşik değeri
EYE_IMG_SIZE = (64, 64)  # Modelin beklediği giriş boyutu

# Yeni model yükleme
BLINK_MODEL = load_model('model1.h5')  # Modelin yolunu kontrol edin


def preprocess_eye(eye_roi):
    """Göz görüntüsünü model için hazırlama"""
    eye_img = cv2.resize(eye_roi, EYE_IMG_SIZE)
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    eye_img = np.expand_dims(eye_img, axis=-1)  # Kanal boyutu ekle
    eye_img = np.expand_dims(eye_img, axis=0)   # Batch boyutu ekle
    return eye_img / 255.0  # Normalizasyon


def model_tahmin_et():
    print("🎯 Göz takibi başlıyor... ESC ile çıkabilirsin.")
    model_x = tf.keras.models.load_model(MODEL_PATH_X)
    model_y = tf.keras.models.load_model(MODEL_PATH_Y)
    cap = cv2.VideoCapture(0)
    

    prev_x, prev_y = pyms.position()
    alpha = 0.2

    # Yeni eklenen tıklama bayrakları
    left_click_flag = False
    right_click_flag = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            # Sağ göz bölgesi (36-41)
            right_eye_points = [
                (landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            # Sol göz bölgesi (42-47)
            left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y)
                               for i in range(42, 48)]

            # Sağ göz işleme
            r_min_x = max(0, min(p[0] for p in right_eye_points) - 10)
            r_min_y = max(0, min(p[1] for p in right_eye_points) - 10)
            r_max_x = min(frame.shape[1], max(p[0]
                          for p in right_eye_points) + 10)
            r_max_y = min(frame.shape[0], max(p[1]
                          for p in right_eye_points) + 10)
            right_eye_roi = frame[r_min_y:r_max_y, r_min_x:r_max_x]

            # Sol göz işleme
            l_min_x = max(0, min(p[0] for p in left_eye_points) - 10)
            l_min_y = max(0, min(p[1] for p in left_eye_points) - 10)
            l_max_x = min(frame.shape[1], max(p[0]
                          for p in left_eye_points) + 10)
            l_max_y = min(frame.shape[0], max(p[1]
                          for p in left_eye_points) + 10)
            left_eye_roi = frame[l_min_y:l_max_y, l_min_x:l_max_x]

            # Sağ göz tahmini
            if right_eye_roi.size > 0:
                right_eye_input = preprocess_eye(right_eye_roi)
                right_eye_state = BLINK_MODEL.predict(
                    right_eye_input, verbose=0)[0][0]

                if right_eye_state < BLINK_THRESHOLD and not right_click_flag:
                    pyms.click(button='right')
                    right_click_flag = True
                    print("Sağ tık yapıldı!")
                elif right_eye_state >= BLINK_THRESHOLD:
                    right_click_flag = False

            # Sol göz tahmini
            if left_eye_roi.size > 0:
                left_eye_input = preprocess_eye(left_eye_roi)
                left_eye_state = BLINK_MODEL.predict(
                    left_eye_input, verbose=0)[0][0]

                if left_eye_state < BLINK_THRESHOLD and not left_click_flag:
                    pyms.click(button='left')
                    left_click_flag = True
                    print("Sol tık yapıldı!")
                elif left_eye_state >= BLINK_THRESHOLD:
                    left_click_flag = False

            # Göz bölgesi ve tahmin kısmı (önceki kodun aynısı)
            r_min_x = max(0, min(p[0] for p in right_eye_points) - 10)
            r_min_y = max(0, min(p[1] for p in right_eye_points) - 10)
            r_max_x = min(frame.shape[1], max(p[0]
                          for p in right_eye_points) + 10)
            r_max_y = min(frame.shape[0], max(p[1]
                          for p in right_eye_points) + 10)

            eye_roi = frame[r_min_y:r_max_y, r_min_x:r_max_x]
            if eye_roi.size == 0:
                continue

            eye_img = cv2.resize(eye_roi, (100, 50))
            eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            eye_img = np.expand_dims(eye_img, axis=(
                0, -1)).astype(np.float32) / 255.0

            pred_x = model_x.predict(eye_img, verbose=0)[0][0]
            pred_y = model_y.predict(eye_img, verbose=0)[0][0]

            target_x = int(pred_x * SCREEN_W)
            target_y = int(pred_y * SCREEN_H)

            smooth_x = int(prev_x * (1 - alpha) + target_x * alpha)
            smooth_y = int(prev_y * (1 - alpha) + target_y * alpha)

            pyms.moveTo(smooth_x, smooth_y, duration=0.05)
            prev_x, prev_y = smooth_x, smooth_y
            break

        cv2.imshow("Eye Tracker", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def show_start_dialog():
    root = tk.Tk()
    root.title("Başlangıç")
    root.geometry("500x150")

    def on_continue():
        root.destroy()
        #veri_topla()
        #model_egit()

        # Yeni bilgilendirme penceresi
        root_msg = tk.Tk()
        root_msg.withdraw()

        response = messagebox.askyesno(
            title="Bilgilendirme",
            message="Veri toplama ve model eğitimi tamamlandı!\nTakip işlemini başlatmak için 'Evet', çıkmak için 'Hayır'ı tıklayın."
        )

        if response:
            model_tahmin_et()
        else:
            root_msg.destroy()
            
            sys.exit()

        root_msg.destroy()

    def on_exit():
        root.destroy()
        print("Program sonlandırılıyor...")
        sys.exit()

    label = tk.Label(root, text="Az sonra bir pencere açılacak veri toplama işlemi o pencerede gerçekleşecek.\nLütfen devam etmek için 'Devam' tuşuna, aksi takdirde 'Çıkış' tuşuna basın.")
    label.pack(pady=10)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    btn_continue = tk.Button(btn_frame, text="Devam",
                             command=on_continue, width=10)
    btn_continue.pack(side=tk.LEFT, padx=20)

    btn_exit = tk.Button(btn_frame, text="Çıkış", command=on_exit, width=10)
    btn_exit.pack(side=tk.RIGHT, padx=20)

    root.mainloop()


# --- ANA AKIŞ ---
if __name__ == "__main__":
    show_start_dialog()

