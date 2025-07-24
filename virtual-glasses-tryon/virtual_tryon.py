# import cv2
# import numpy as np
# import os

# # Load Haar cascades for face and eye detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# # Load all glasses images from the 'glasses' folder
# glass_images = []
# glass_folder = "glasses"
# for file in sorted(os.listdir(glass_folder)):
#     if file.endswith(".png"):
#         img = cv2.imread(os.path.join(glass_folder, file), cv2.IMREAD_UNCHANGED)
#         glass_images.append(img)

# current_glass_index = 0

# # For stabilizing detection
# eye_history = []
# MAX_HISTORY = 5

# # Function to overlay glasses
# def overlay_glasses(frame, eyes, glasses_img):
#     if len(eyes) >= 2:
#         # Sort eyes left to right
#         eyes = sorted(eyes, key=lambda e: e[0])
#         x1, y1, w1, h1 = eyes[0]
#         x2, y2, w2, h2 = eyes[1]

#         # Eye centers
#         center1 = (x1 + w1 // 2, y1 + h1 // 2)
#         center2 = (x2 + w2 // 2, y2 + h2 // 2)

#         # Store in history
#         eye_history.append((center1, center2))
#         if len(eye_history) > MAX_HISTORY:
#             eye_history.pop(0)

#         # Averaging for smoothing
#         avg1 = tuple(map(lambda x: int(sum(x) / len(x)), zip(*[e[0] for e in eye_history])))
#         avg2 = tuple(map(lambda x: int(sum(x) / len(x)), zip(*[e[1] for e in eye_history])))

#         # Calculate overlay position and size
#         eye_width = abs(avg2[0] - avg1[0])
#         overlay_width = int(eye_width * 2.2)
#         overlay_height = int(overlay_width * glasses_img.shape[0] / glasses_img.shape[1])

#         x = int(avg1[0] - overlay_width * 0.25)
#         y = int(min(avg1[1], avg2[1]) - overlay_height * 0.5)

#         # Resize glasses
#         resized_glasses = cv2.resize(glasses_img, (overlay_width, overlay_height), interpolation=cv2.INTER_AREA)

#         # Overlay on frame
#         overlay_image(frame, resized_glasses, x, y)

#         # Draw eye centers for debugging
#         cv2.circle(frame, avg1, 3, (0, 255, 0), -1)
#         cv2.circle(frame, avg2, 3, (0, 255, 0), -1)

# # Alpha blending for overlay
# def overlay_image(background, overlay, x, y):
#     h, w = overlay.shape[:2]
#     for i in range(h):
#         for j in range(w):
#             if 0 <= y+i < background.shape[0] and 0 <= x+j < background.shape[1]:
#                 alpha = overlay[i, j, 3] / 255.0
#                 if alpha > 0:
#                     background[y+i, x+j] = (1 - alpha) * background[y+i, x+j] + alpha * overlay[i, j, :3]

# # Main loop
# cap = cv2.VideoCapture(0)

# print("Press 'n' to switch glasses, 'q' to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_flipped = cv2.flip(frame, 1)
#     gray = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2GRAY)

#     # Detect face
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (fx, fy, fw, fh) in faces:
#         roi_gray = gray[fy:fy+fh, fx:fx+fw]
#         roi_color = frame_flipped[fy:fy+fh, fx:fx+fw]

#         # Detect eyes
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         eyes_global = [(fx + ex, fy + ey, ew, eh) for (ex, ey, ew, eh) in eyes]

#         overlay_glasses(frame_flipped, eyes_global, glass_images[current_glass_index])
#         break  # Only use first face

#     cv2.imshow("Virtual Try-On", frame_flipped)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('n'):
#         current_glass_index = (current_glass_index + 1) % len(glass_images)
#         print(f"Switched to Glass {current_glass_index + 1}")
#     elif key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import os

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load glasses
GLASSES_FOLDER = "glasses"
glass_images = [cv2.imread(os.path.join(GLASSES_FOLDER, f), cv2.IMREAD_UNCHANGED) 
                for f in sorted(os.listdir(GLASSES_FOLDER)) if f.endswith('.png')]

current_glass_index = 0
eye_history = []
MAX_HISTORY = 5

# 3D model points of facial landmarks (approximate)
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Camera internals (assume no distortion)
def get_camera_matrix(frame_size):
    focal_length = frame_size[1]
    center = (frame_size[1] / 2, frame_size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    return camera_matrix

def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    for i in range(h):
        for j in range(w):
            if 0 <= y+i < background.shape[0] and 0 <= x+j < background.shape[1]:
                alpha = overlay[i,j,3] / 255.0
                if alpha > 0:
                    background[y+i, x+j] = alpha*overlay[i,j,:3] + (1-alpha)*background[y+i, x+j]
    return background

cap = cv2.VideoCapture(0)

print("Press 'n' to switch glasses, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        cv2.imshow("Virtual Try-On with Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) < 2:
            # If less than 2 eyes detected, show frame and continue
            cv2.imshow("Virtual Try-On with Pose", frame)
            break

        eyes = sorted(eyes, key=lambda e: e[0])  # Sort by x pos
        eye_1 = eyes[0]
        eye_2 = eyes[1]

        # Approximate 2D image points for head pose (nose tip and mouth are estimated approx)
        # Here we estimate nose and mouth positions based on face rectangle
        image_points = np.array([
            (x + w//2, y + int(h*0.6)),         # Nose tip (approx center lower face)
            (x + w//2, y + h),                  # Chin (bottom center)
            (x + eye_1[0] + eye_1[2]//2, y + eye_1[1] + eye_1[3]//2),  # Left eye center
            (x + eye_2[0] + eye_2[2]//2, y + eye_2[1] + eye_2[3]//2),  # Right eye center
            (x + int(w*0.3), y + int(h*0.8)),  # Left mouth corner approx
            (x + int(w*0.7), y + int(h*0.8))   # Right mouth corner approx
        ], dtype="double")

        size = frame.shape
        camera_matrix = get_camera_matrix(size)
        dist_coeffs = np.zeros((4,1))  # Assume no lens distortion

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # Project a 3D point (0,0,1000) in front of nose to visualize direction
        (nose_end_point2D, _) = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(frame, p1, p2, (255,0,0), 2)

        # Calculate glasses size & rotation based on eyes & pose
        eye_center1 = (int(image_points[2][0]), int(image_points[2][1]))
        eye_center2 = (int(image_points[3][0]), int(image_points[3][1]))

        dx = eye_center2[0] - eye_center1[0]
        dy = eye_center2[1] - eye_center1[1]
        eye_distance = np.sqrt(dx*dx + dy*dy)

        angle = np.degrees(np.arctan2(dy, dx))

        glasses_img = glass_images[current_glass_index]

        # Resize glasses according to eye distance
        scale = eye_distance / glasses_img.shape[1] * 2.2
        new_w = int(glasses_img.shape[1] * scale)
        new_h = int(glasses_img.shape[0] * scale)

        resized_glasses = cv2.resize(glasses_img, (new_w, new_h))

        # Rotate glasses to match head tilt
        M = cv2.getRotationMatrix2D((new_w//2, new_h//2), angle, 1.0)
        rotated_glasses = cv2.warpAffine(resized_glasses, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

        # Position glasses centered between eyes with slight vertical offset
        x_offset = int((eye_center1[0] + eye_center2[0]) / 2 - new_w / 2)
        y_offset = int((eye_center1[1] + eye_center2[1]) / 2 - new_h / 2.5)

        # Overlay
        overlay_transparent(frame, rotated_glasses, x_offset, y_offset)

        # Draw eye centers for debugging
        cv2.circle(frame, eye_center1, 5, (0,255,0), -1)
        cv2.circle(frame, eye_center2, 5, (0,255,0), -1)

        break  # Only use first detected face

    cv2.putText(frame, f'Glasses {current_glass_index+1}/{len(glass_images)}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Virtual Try-On with Head Pose", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('n'):
        current_glass_index = (current_glass_index + 1) % len(glass_images)
        print(f"Switched to Glasses {current_glass_index+1}")

cap.release()
cv2.destroyAllWindows()
