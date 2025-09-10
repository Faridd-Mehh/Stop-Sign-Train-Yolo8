import multiprocessing as mp
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # transfer learning (hızlı yakınsar)
    results = model.train(
        data=r"C:\Users\ASUS\Desktop\yolo8_Stop\data.yaml",
        epochs=50,
        patience=7,
        imgsz=320,
        batch=-1,
        workers=4,
        device=0,
        cache=True,        
        freeze=10,          
        mosaic=0.0,         
        deterministic=False,
    )
    print(results)

if __name__ == "__main__":
    mp.freeze_support()
    main()
