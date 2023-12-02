import cv2
from PIL import Image
import numpy as np

def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        return faces[0]  # Return the first detected face
    else:
        return None

def main():
    cap = cv2.VideoCapture(0)  # 0 indicates the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_coords = detect_face(frame)

        if face_coords is not None:
            x, y, w, h = face_coords
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display name and image
            name = "You are a (Dog)"
            image_path = "Dog.jpg"
            img = Image.open(image_path)
            img = img.resize((w, h), Image.LANCZOS)
            frame[y:y+h, x:x+w] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Your Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
