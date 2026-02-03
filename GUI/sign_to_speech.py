import cv2
import mediapipe as mp
import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()

def speak(text):
    """Speak out the detected gesture."""
    print(f"Detected: {text}")
    engine.say(text)
    engine.runAndWait()

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)  # Added tracking confidence for stability

# Start Video Capture
cap = cv2.VideoCapture(0)

# Margins to avoid flickering due to small movements
finger_margin = 0.015  # Slightly smaller margin for more sensitivity
thumb_margin = 0.03    # Adjusted thumb margin for better curl detection

gesture_detected = {}  # Dictionary to track gesture per hand (left/right) to prevent repeated speaking

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with Mediapipe
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get handedness (Left or Right)
            hand_label = results.multi_handedness[idx].classification[0].label
            hand_key = hand_label.lower()  # Use 'left' or 'right' as key

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark coordinates (normalized 0-1, but we use relative y for up/down)
            lm = hand_landmarks.landmark

            # Key points
            thumb_tip = lm[4].y
            thumb_mcp = lm[2].y
            index_tip = lm[8].y
            index_mcp = lm[5].y
            middle_tip = lm[12].y
            middle_mcp = lm[9].y
            ring_tip = lm[16].y
            ring_mcp = lm[13].y
            pinky_tip = lm[20].y
            pinky_mcp = lm[17].y

            # Debug: Print y-coordinates to console for troubleshooting (remove after testing)
            print(f"{hand_label} Hand - Thumb: tip={thumb_tip:.3f}, mcp={thumb_mcp:.3f} (diff={thumb_tip - thumb_mcp:.3f}) | "
                  f"Index: tip={index_tip:.3f}, mcp={index_mcp:.3f} (diff={index_tip - index_mcp:.3f}) | "
                  f"Middle: tip={middle_tip:.3f}, mcp={middle_mcp:.3f} (diff={middle_tip - middle_mcp:.3f}) | "
                  f"Ring: tip={ring_tip:.3f}, mcp={ring_mcp:.3f} (diff={ring_tip - ring_mcp:.3f}) | "
                  f"Pinky: tip={pinky_tip:.3f}, mcp={pinky_mcp:.3f} (diff={pinky_tip - pinky_mcp:.3f})")

            # Finger states (up if tip y < mcp y - margin, down if tip y > mcp y + margin)
            # Note: Negative diff means up (tip higher in image, lower y-value), positive diff means down
            thumb_up = (thumb_tip - thumb_mcp) < -thumb_margin
            thumb_down = (thumb_tip - thumb_mcp) > thumb_margin
            index_up = (index_tip - index_mcp) < -finger_margin
            index_down = (index_tip - index_mcp) > finger_margin
            middle_up = (middle_tip - middle_mcp) < -finger_margin
            middle_down = (middle_tip - middle_mcp) > finger_margin
            ring_up = (ring_tip - ring_mcp) < -finger_margin
            ring_down = (ring_tip - ring_mcp) > finger_margin
            pinky_up = (pinky_tip - pinky_mcp) < -finger_margin
            pinky_down = (pinky_tip - pinky_mcp) > finger_margin

            # Check gestures with further relaxed and prioritized logic
            current_gesture = ""

            # Peace ✌️: index & middle up, ring & pinky down, thumb down or neutral (not up)
            # Relaxed: Even if thumb is slightly up, allow if other conditions are strong
            if (index_up and middle_up and 
                ring_down and pinky_down and
                not thumb_up):  # Thumb not extended up (allows down or neutral/sideways curl)
                current_gesture = "Peace ✌️"

            # Hello 👋: all fingers up (thumb, index, middle, ring, pinky all up)
            # This remains strict to match your working "Hello" detection
            elif (thumb_up and index_up and middle_up and ring_up and pinky_up):
                current_gesture = "Hello 👋"

            # Debug: Display finger states on frame for visualization (remove after testing)
            # Shows diff signs: ↑ (negative diff, up), ↓ (positive diff, down), - (neutral)
            thumb_state = '↑' if thumb_up else '↓' if thumb_down else '-'
            index_state = '↑' if index_up else '↓' if index_down else '-'
            middle_state = '↑' if middle_up else '↓' if middle_down else '-'
            ring_state = '↑' if ring_up else '↓' if ring_down else '-'
            pinky_state = '↑' if pinky_up else '↓' if pinky_down else '-'
            debug_text = f"{hand_label}: T{thumb_state}, I{index_state}, M{middle_state}, R{ring_state}, P{pinky_state}"
            cv2.putText(frame, debug_text, (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

            # Prepare full message with hand label
            if current_gesture:
                full_gesture = f"{hand_label} hand: {current_gesture}"
                # Display on frame (position based on hand for better visibility)
                y_offset = 50 if hand_label == "Left" else 100
                cv2.putText(frame, full_gesture, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Speak only if gesture changed for this hand
                if hand_key not in gesture_detected or gesture_detected[hand_key] != current_gesture:
                    gesture_detected[hand_key] = current_gesture
                    speak(full_gesture)
            else:
                # Clear gesture if no detection (to reset for next change)
                if hand_key in gesture_detected:
                    del gesture_detected[hand_key]

    cv2.imshow("Sign to Speech", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
