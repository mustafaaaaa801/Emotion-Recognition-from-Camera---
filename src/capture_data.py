import cv2
import os
import mediapipe as mp

CLASSES = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust", "contempt", "boredom"]
BASE_DIR = "data"
TRAIN_COUNT = 20
VAL_COUNT = 5
IMG_SIZE = (224, 224)

# MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

def make_dirs():
    for split in ["train", "val"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(BASE_DIR, split, cls), exist_ok=True)

def capture_images():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # تكبير الشاشة
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    for cls in CLASSES:
        print(f"\n=== التحضير لجمع صور فئة: {cls} ===")
        print("SPACE = التقاط الصورة، N = الانتقال للفئة التالية، ESC = خروج")
        train_idx, val_idx = 0, 0

        while train_idx < TRAIN_COUNT or val_idx < VAL_COUNT:
            ret, frame = cap.read()
            if not ret:
                continue

            # كشف الوجه
            results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes = []
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = x1 + int(bbox.width * w)
                    y2 = y1 + int(bbox.height * h)
                    boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"Class: {cls} Train:{train_idx}/{TRAIN_COUNT} Val:{val_idx}/{VAL_COUNT}"
            cv2.putText(frame, text, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Capture Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                break
            elif key == 32:
                if len(boxes) == 0:
                    continue
                x1, y1, x2, y2 = boxes[0]  # استخدام أول وجه
                face_crop = cv2.resize(frame[y1:y2, x1:x2], IMG_SIZE)
                if train_idx < TRAIN_COUNT:
                    path = os.path.join(BASE_DIR, "train", cls, f"{cls}_{train_idx}.jpg")
                    train_idx += 1
                elif val_idx < VAL_COUNT:
                    path = os.path.join(BASE_DIR, "val", cls, f"{cls}_{val_idx}.jpg")
                    val_idx += 1
                else:
                    continue
                cv2.imwrite(path, face_crop)
                print(f"تم حفظ الصورة: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("تم جمع جميع الصور بنجاح!")

if __name__ == "__main__":
    make_dirs()
    capture_images()
