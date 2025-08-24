import cv2
import torch
from torchvision import transforms
from models import get_model
from data_pipeline import FaceExtractor

CLASSES = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust", "contempt", "boredom"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model("resnet18", num_classes=9, pretrained=False)
model.load_state_dict(torch.load("outputs/best.pth", map_location=device))
model.eval()
model.to(device)

face_ext = FaceExtractor(min_detection_confidence=0.5)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    boxes = face_ext.extract(frame)
    for (x1, y1, x2, y2) in boxes:
        face = frame[y1:y2, x1:x2]
        face_t = transform(face).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(face_t)
            pred = CLASSES[torch.argmax(output, dim=1).item()]
        cv2.putText(frame, pred, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
