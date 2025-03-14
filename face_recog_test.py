import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()  # 啟用訓練人臉模型方法
recognizer.read("face.yml")  # 讀取人臉模型檔
cascade_path = "haarcascade_frontalface_default.xml"  # 載入人臉追蹤模型
face_cascade = cv2.CascadeClassifier(cascade_path)  # 啟用人臉追蹤

# 建立姓名和 id 的對照表
name = {"1": "Green", "2": "Durant"}

cap = cv2.VideoCapture("interview.mp4")  # 開啟攝影機
cv2.namedWindow("interview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("interview", 1080, 1920)
if not cap.isOpened():
    print("Cannot load video")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉換成黑白
    faces = face_cascade.detectMultiScale(gray)  # 追蹤人臉 ( 目的在於標記出外框 )

    # 依序判斷每張臉屬於哪個 id
    for x, y, w, h in faces:
        # 取出 id 號碼以及信心指數 confidence
        idnum, confidence = recognizer.predict(gray[y : y + h, x : x + w]) 
        if confidence > 80:
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 標記人臉外框
        text = f"{name[str(idnum)]} {confidence:.2f}"
        # 在人臉外框旁加上名字
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA,)

    cv2.imshow("interview", img)
    if cv2.waitKey(5) == ord("q"):
        break  # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()
