from ultralytics import YOLO
import cv2

def main():
    model = YOLO("runs/detect/train5/weights/best.pt")

    cap = cv2.VideoCapture('predict_footage/IMG_5105.mov')

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Video', 100, 100)
    cv2.resizeWindow('Video', 800, 600)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source = frame, classes=[5])

        for result in results:
            if len(result.boxes) == 1:
                #put logic for blinking light here
                cpu_bound = result.boxes.cpu()
                coords = cpu_bound.numpy()
                x1, y1, x2, y2 = coords.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1), int(x2), int(y2)), (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

