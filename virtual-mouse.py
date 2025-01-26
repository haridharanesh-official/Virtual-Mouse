import cv2
import mediapipe as mp
import pyautogui
import threading
import time
import platform

# Mediapipe and PyAutoGUI initialization
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# Smooth movement variables
previous_x, previous_y = 0, 0
smoothing_factor = 0.2
running = True

# OS Compatibility adjustments
is_raspberry_pi = platform.system() == "Linux" and "arm" in platform.machine()

# Background task for controlling the virtual mouse
def virtual_mouse_task():
    global previous_x, previous_y, running

    while running:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            for hand in hands:
                drawing_utils.draw_landmarks(frame, hand)
                landmarks = hand.landmark

                index_x, index_y, thumb_x, thumb_y = None, None, None, None

                for id, landmark in enumerate(landmarks):
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)

                    if id == 8:  # Index finger tip
                        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                        index_x = screen_width / frame_width * x
                        index_y = screen_height / frame_height * y
                    
                    if id == 4:  # Thumb tip
                        cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
                        thumb_x = screen_width / frame_width * x
                        thumb_y = screen_height / frame_height * y

                # Cursor movement with smoothing
                if index_x is not None and index_y is not None:
                    current_x = previous_x + (index_x - previous_x) * smoothing_factor
                    current_y = previous_y + (index_y - previous_y) * smoothing_factor
                    pyautogui.moveTo(current_x, current_y)
                    previous_x, previous_y = current_x, current_y

                # Gestures for click and scroll
                if index_y and thumb_y and abs(index_y - thumb_y) < 20:
                    pyautogui.click()
                    time.sleep(0.5)  # Prevent rapid clicks

                if index_y and thumb_y and abs(index_y - thumb_y) < 100:
                    pyautogui.scroll(-100)  # Scroll gesture

        if not is_raspberry_pi:
            cv2.imshow('Virtual Mouse', frame)

        # Exit mechanism
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

# Run the virtual mouse as a background task
background_thread = threading.Thread(target=virtual_mouse_task)
background_thread.start()

# Main program loop (can be used for other tasks)
try:
    while running:
        print("Main program running. Press 'q' to quit.")
        time.sleep(1)  # Simulate other tasks here

except KeyboardInterrupt:
    print("Program interrupted.")

# Cleanup
finally:
    running = False
    background_thread.join()
    cap.release()
    if not is_raspberry_pi:
        cv2.destroyAllWindows()
