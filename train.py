from ultralytics import YOLO


def main():
    model = YOLO("yolov8x.yaml")
    results = model.train(data="data.yaml", epochs=350, imgsz=640)
    success = model.export(format="onnx")


if __name__ == "__main__":
    main()
