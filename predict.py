from ultralytics import YOLO
import cv2 as cv

model = YOLO("yolov8n.pt")

for r in model.predict(
        source=r"C:\Users\ASUS\Desktop\yolo8_Stop\test",
        stream=True,
        conf=0.55,
        save=True,
        project=r"C:\Users\ASUS\Desktop\yolo8_Stop\runs",
        name="stop_test",
        exist_ok=True):
    frame = r.plot()
    cv.imshow("YOLOv8 - STOP", frame)

    # Go next result with any button
    key = cv.waitKey(0) & 0xFF
    if key in (27, ord('q')):
        break

cv.destroyAllWindows()
