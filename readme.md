- GPU إن وُجد يسرّع التدريب


## تثبيت
```
pip install -r requirements.txt
```


## بنية المشروع (مبسطة)
```
emotion_camera_project/
├─ data/ # ضع هنا الصور المصنفة: train/ val/ test/ each-class/
├─ src/
│ ├─ data_pipeline.py
│ ├─ models.py
│ ├─ train.py
│ ├─ infer_camera.py
│ └─ utils.py
├─ configs/config.yaml
├─ requirements.txt
└─ README.md
```


## أوامر تشغيل سريعة
- تدريب:
`python src/train.py --config configs/config.yaml`
- استدلال لايف من الكاميرا (بعد التدريب):
`python src/infer_camera.py --model outputs/best.pth --config configs/config.yaml`


## ملاحظات
- تأكد من تنظيم البيانات في `data/train/<class>/` و `data/val/<class>/`.
- الكود يعتمد على مكتبات شائعة: PyTorch, OpenCV, MediaPipe.
"""