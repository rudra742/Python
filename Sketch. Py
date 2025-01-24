import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create layered canvas system
paint_layer = None
prev_x, prev_y = 0, 0
brush_size = 15
alpha_value = 0.7  # Global transparency
color_index = 0

# Deep colors (BGR format)
colors = [
    (0, 0, 255),    # Deep Blue
    (0, 255, 0),    # Emerald Green
    (255, 0, 0),    # Crimson Red
    (0, 255, 255)   # Gold
]

def overlay_transparent(background, overlay):
    # Split overlay into color and alpha channels
    overlay_img = overlay[:, :, :3]
    alpha = overlay[:, :, 3:] / 255.0
    
    # Convert to float for correct blending
    background = background.astype(float)
    overlay_img = overlay_img.astype(float)
    
    # Perform alpha blending
    blended = (overlay_img * alpha) + (background * (1.0 - alpha))
    return blended.astype(np.uint8)

def is_pointing_gesture(landmarks):
    # Check if index finger is extended and other fingers are closed
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]
    
    # Index finger extended (tip y < PIP y)
    index_extended = index_tip.y < index_pip.y
    
    # Other fingers closed (tip y > PIP y)
    middle_closed = middle_tip.y > middle_pip.y
    ring_closed = ring_tip.y > ring_pip.y
    pinky_closed = pinky_tip.y > pinky_pip.y
    
    return index_extended and middle_closed and ring_closed and pinky_closed

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Initialize paint layer
    if paint_layer is None:
        paint_layer = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Process hands
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    drawing_enabled = False
    current_color = colors[color_index]

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Get hand landmarks
        landmarks = hand_landmarks.landmark
        
        # Check gestures
        if is_pointing_gesture(landmarks):
            drawing_enabled = True
            # Get index finger tip coordinates
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            # Control brush size with hand height
            brush_size = int(35 * (1 - index_tip.y))
            brush_size = max(10, min(brush_size, 50))

            # Change color with middle finger gesture
            middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            if middle_tip.y < landmarks[mp_hands.HandLandmark.WRIST].y:
                color_index = (color_index + 1) % len(colors)

            # Draw on paint layer
            if drawing_enabled and (prev_x != 0 or prev_y != 0):
                temp_paint = np.zeros((h, w, 4), dtype=np.uint8)
                cv2.line(temp_paint, (prev_x, prev_y), (x, y), 
                        (*current_color, 255),
                        brush_size, lineType=cv2.LINE_AA)
                
                # Merge with existing paint layer
                paint_layer = cv2.addWeighted(paint_layer, 1.0, temp_paint, 1.0, 0)

            prev_x, prev_y = x, y
        else:
            # Reset previous coordinates when not pointing
            prev_x, prev_y = 0, 0

    # Apply global transparency
    transparent_paint = paint_layer.copy()
    transparent_paint[:, :, 3] = np.uint8(transparent_paint[:, :, 3] * alpha_value)

    # Combine layers
    combined = overlay_transparent(frame, transparent_paint)

    # Add UI elements
    cv2.putText(combined, f"Brush Size: {brush_size}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Color:", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.circle(combined, (150, 60), 20, current_color, -1)
    cv2.putText(combined, f"Opacity: {int(alpha_value*100)}%", (10, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Gesture: " + ("👆 Drawing" if drawing_enabled else "✊ Stopped"), 
               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, 
               (255, 255, 255), 2)

    cv2.imshow("Gesture-Controlled Paint", combined)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        paint_layer = np.zeros((h, w, 4), dtype=np.uint8)
    elif key == ord('+'):
        alpha_value = min(alpha_value + 0.1, 1.0)
    elif key == ord('-'):
        alpha_value = max(alpha_value - 0.1, 0.3)

cap.release()
cv2.destroyAllWindows()
