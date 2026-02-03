import cv2
import mediapipe as mp
import pyttsx3
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time

# Initialize TTS engine
engine = pyttsx3.init()

def speak(text):
    """Speak out the detected gesture."""
    print(f"Detected: {text}")
    engine.say(text)
    engine.runAndWait()

class GestureRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign to Speech - Gesture Recognition")
        self.root.geometry("800x600")
        
        # Initialize variables
        self.cap = None
        self.hands = None
        self.is_running = False
        self.gesture_thread = None
        self.current_frame = None
        self.gesture_detected = {}  # Dictionary to track gesture per hand
        
        # GUI Elements
        self.setup_gui()
        
        # Initialize MediaPipe
        self.init_mediapipe()
    
    def setup_gui(self):
        # Video Display Frame
        video_frame = ttk.Frame(self.root)
        video_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(video_frame, text="Click Start to begin gesture recognition")
        self.video_label.pack(expand=True)
        
        # Control Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)
        
        # Gesture Display
        self.gesture_label = ttk.Label(self.root, text="No gesture detected", font=("Arial", 12, "bold"))
        self.gesture_label.pack(pady=10)
        
        # Debug/Status Text (optional, can be expanded to a scrollable text area)
        self.status_text = tk.Text(self.root, height=5, width=80, state=tk.DISABLED)
        self.status_text.pack(pady=10, padx=10, fill=tk.X)
    
    def init_mediapipe(self):
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp_hands
        self.mp_drawing = mp_drawing
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        
        # Margins
        self.finger_margin = 0.015
        self.thumb_margin = 0.03
    
    def start_detection(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
        
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.gesture_label.config(text="Starting detection...")
        
        # Start thread for video processing
        self.gesture_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.gesture_thread.start()
    
    def stop_detection(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.gesture_label.config(text="Detection stopped")
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def update_status(self, text):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, text + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def video_loop(self):
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with Mediapipe
            results = self.hands.process(rgb)
            
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get handedness
                    hand_label = results.multi_handedness[idx].classification[0].label
                    hand_key = hand_label.lower()
                    
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get landmark coordinates
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
                    
                    # Finger states
                    thumb_up = (thumb_tip - thumb_mcp) < -self.thumb_margin
                    thumb_down = (thumb_tip - thumb_mcp) > self.thumb_margin
                    index_up = (index_tip - index_mcp) < -self.finger_margin
                    index_down = (index_tip - index_mcp) > self.finger_margin
                    middle_up = (middle_tip - middle_mcp) < -self.finger_margin
                    middle_down = (middle_tip - middle_mcp) > self.finger_margin
                    ring_up = (ring_tip - ring_mcp) < -self.finger_margin
                    ring_down = (ring_tip - ring_mcp) > self.finger_margin
                    pinky_up = (pinky_tip - pinky_mcp) < -self.finger_margin
                    pinky_down = (pinky_tip - pinky_mcp) > self.finger_margin
                    
                    # Check gestures
                    current_gesture = ""
                    
                    # Peace ✌️
                    if (index_up and middle_up and 
                        ring_down and pinky_down and
                        not thumb_up):
                        current_gesture = "Peace ✌️"
                    
                    # Hello 👋
                    elif (thumb_up and index_up and middle_up and ring_up and pinky_up):
                        current_gesture = "Hello 👋"
                    
                    # Debug states (optional, printed to status)
                    thumb_state = '↑' if thumb_up else '↓' if thumb_down else '-'
                    index_state = '↑' if index_up else '↓' if index_down else '-'
                    middle_state = '↑' if middle_up else '↓' if middle_down else '-'
                    ring_state = '↑' if ring_up else '↓' if ring_down else '-'
                    pinky_state = '↑' if pinky_up else '↓' if pinky_down else '-'
                    debug_text = f"{hand_label}: T{thumb_state}, I{index_state}, M{middle_state}, R{ring_state}, P{pinky_state}"
                    self.update_status(debug_text)
                    
                    # Display gesture on frame
                    if current_gesture:
                        full_gesture = f"{hand_label} hand: {current_gesture}"
                        y_offset = 50 if hand_label == "Left" else 100
                        cv2.putText(frame, full_gesture, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Speak if changed
                        if hand_key not in self.gesture_detected or self.gesture_detected[hand_key] != current_gesture:
                            self.gesture_detected[hand_key] = current_gesture
                            speak(full_gesture)
                            # Update GUI label in main thread
                            self.root.after(0, lambda: self.gesture_label.config(text=full_gesture))
                    else:
                        if hand_key in self.gesture_detected:
                            del self.gesture_detected[hand_key]
                
                # Update GUI with current frame (convert to PhotoImage)
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.root.after(0, self.update_video_display)
            
            time.sleep(0.033)  # ~30 FPS
        
        # Cleanup on stop
        self.root.after(0, lambda: self.gesture_label.config(text="Detection stopped"))
    
    def update_video_display(self):
        if self.current_frame is not None:
            # Resize frame for display (fit to window)
            height, width = self.current_frame.shape[:2]
            max_width = 640
            max_height = 480
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                self.current_frame = cv2.resize(self.current_frame, (new_width, new_height))
            
            img = Image.fromarray(self.current_frame)
            photo = ImageTk.PhotoImage(img)
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Keep a reference
    
    def on_closing(self):
        self.stop_detection()
        if self.hands:
            self.hands.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureRecognizerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
