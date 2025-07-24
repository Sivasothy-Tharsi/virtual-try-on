import cv2
import numpy as np
import os

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load glasses images from 'glasses' folder
GLASSES_FOLDER = "glasses"
glass_images = [cv2.imread(os.path.join(GLASSES_FOLDER, f), cv2.IMREAD_UNCHANGED) 
                for f in sorted(os.listdir(GLASSES_FOLDER)) if f.endswith('.png')]

current_glass_index = 0
eye_history = []
MAX_HISTORY = 5

def detect_eyes_filtered(roi_gray):
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
    filtered_eyes = []
    h = roi_gray.shape[0]

    for (ex, ey, ew, eh) in eyes:
        aspect_ratio = ew / float(eh)
        # Filter eyes to upper half of face ROI, reasonable aspect ratio and size
        if ey + eh/2 < h * 0.5 and 0.5 < aspect_ratio < 2.0 and ew > 15 and eh > 15:
            filtered_eyes.append((ex, ey, ew, eh))

    filtered_eyes = sorted(filtered_eyes, key=lambda e: e[0])  # left to right
    if len(filtered_eyes) >= 2:
        return filtered_eyes[:2]  # return best two eyes
    return None

def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    for i in range(h):
        for j in range(w):
            if 0 <= y+i < background.shape[0] and 0 <= x+j < background.shape[1]:
                alpha = overlay[i,j,3] / 255.0
                if alpha > 0:
                    background[y+i, x+j] = alpha*overlay[i,j,:3] + (1-alpha)*background[y+i, x+j]
    return background

def overlay_glasses(frame, eyes, glasses_img):
    if len(eyes) >= 2:
        # Sort eyes left to right
        eyes = sorted(eyes, key=lambda e: e[0])
        x1, y1, w1, h1 = eyes[0]
        x2, y2, w2, h2 = eyes[1]

        center1 = (x1 + w1 // 2, y1 + h1 // 2)
        center2 = (x2 + w2 // 2, y2 + h2 // 2)

        # Smoothing over frames
        eye_history.append((center1, center2))
        if len(eye_history) > MAX_HISTORY:
            eye_history.pop(0)

        avg1 = tuple(map(lambda x: int(sum(x) / len(x)), zip(*[e[0] for e in eye_history])))
        avg2 = tuple(map(lambda x: int(sum(x) / len(x)), zip(*[e[1] for e in eye_history])))

        # Calculate glasses size & position
        eye_width = abs(avg2[0] - avg1[0])
        overlay_width = int(eye_width * 2.2)
        overlay_height = int(overlay_width * glasses_img.shape[0] / glasses_img.shape[1])

        x = int(avg1[0] - overlay_width * 0.25)
        y = int(min(avg1[1], avg2[1]) - overlay_height * 0.5)

        # Resize glasses image
        resized_glasses = cv2.resize(glasses_img, (overlay_width, overlay_height), interpolation=cv2.INTER_AREA)

        # Overlay glasses
        overlay_transparent(frame, resized_glasses, x, y)

        # Draw eye centers for debugging
        cv2.circle(frame, avg1, 5, (0, 255, 0), -1)
        cv2.circle(frame, avg2, 5, (0, 255, 0), -1)

cap = cv2.VideoCapture(0)

print("Press 'n' to switch glasses, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_flipped = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        cv2.imshow("Virtual Try-On", frame_flipped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        # Use improved eye detection
        eyes = detect_eyes_filtered(roi_gray)
        if eyes is None:
            # No good eyes found, show frame as is
            continue

        # Convert eye coords to full frame
        eyes_global = [(ex + x, ey + y, ew, eh) for (ex, ey, ew, eh) in eyes]

        # Overlay glasses on frame
        overlay_glasses(frame_flipped, eyes_global, glass_images[current_glass_index])
        break  # Process only first face

    cv2.putText(frame_flipped, f'Glasses {current_glass_index + 1}/{len(glass_images)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Virtual Try-On", frame_flipped)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        current_glass_index = (current_glass_index + 1) % len(glass_images)
        print(f"Switched to Glasses {current_glass_index + 1}")

cap.release()
cv2.destroyAllWindows()

