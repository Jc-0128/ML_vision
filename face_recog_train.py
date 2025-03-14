import cv2
import numpy as np
import os
from bing_image_downloader import downloader

# download image
downloader.download("Draymond Green Headshot", limit=10, output_dir="")
downloader.download("Kevin Durant Headshot", limit=10, output_dir="")
os.rename("Draymond Green Headshot", "Green")
os.rename("Kevin Durant Headshot", "Durant")

# Augmentation (optional)
def augment_image(image_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 1. Enhanced Contrast
    alpha = 1.5  
    beta = 0 
    enhanced_contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 2. Noise
    noisy_image = np.copy(img)
    salt_vs_pepper = 0.5
    amount = 0.05

    num_salt = np.ceil(amount * img.size * salt_vs_pepper)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_image[tuple(coords)] = 255

    num_pepper = np.ceil(amount * img.size * (1.0 - salt_vs_pepper))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_image[tuple(coords)] = 0

    # 3. Gaussian Blur
    blurred_image = cv2.GaussianBlur(img, (31, 31), 0)

    # 4. Sharpen
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(img, -1, kernel)

    # Save augmented images
    cv2.imwrite(f"{output_path}_contrast.jpg", enhanced_contrast)
    cv2.imwrite(f"{output_path}_noisy.jpg", noisy_image)
    cv2.imwrite(f"{output_path}_blurred.jpg", blurred_image)
    cv2.imwrite(f"{output_path}_sharpened.jpg", sharpened_image)

    print(image_path, "augmented")

for name in ["Durant", "Green"]:
    for file in os.listdir(f"{name}"):
        img = cv2.imread(f"{name}/{file}")
        print(f"{name}/{file} augmenting...")
        augment_image(f"{name}/{file}", f"{name}/{file}")

# Train model
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # 載入人臉追蹤模型
recog = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型方法
faces = []   # 儲存人臉位置大小的串列
ids = []     # 記錄該人臉 id 的串列

for i, name in enumerate(["Green", "Durant"], 1):
    for file in os.listdir(f"{name}"):
        img = cv2.imread(f"{name}/{file}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_np = np.array(gray, "uint8")
        face = detector.detectMultiScale(gray)
        for (x, y, w, h) in face:
            faces.append(img_np[y:y+h, x:x+w])
            ids.append(i)

print('training...')                              # 提示開始訓練
recog.train(faces,np.array(ids))                  # 開始訓練
recog.save('face.yml')                            # 訓練完成儲存為 face.yml
print('ok!')

