import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import mediapipe as mp
import numpy as np
import time

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
font = cv2.FONT_HERSHEY_SIMPLEX

destinations = ["Chennai", "Bangalore", "Delhi", "Mumbai", "Hyderabad"]
selected_dest = 0
ticket_class = 0  
adult_tickets = 1  
child_tickets = 0
stage = 0 
current_gesture_frames = 0
last_gesture_change_time = 0
current_finger_count = 0
gesture_hold_time = 3  
def count_fingers(hand_landmarks):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]

    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    hand_orientation = "right" if hand_landmarks.landmark[5].x < hand_landmarks.landmark[17].x else "left"
    if hand_orientation == "right":
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.insert(0, 1)
        else:
            fingers.insert(0, 0)
    else:
        if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.insert(0, 1)
        else:
            fingers.insert(0, 0)

    return sum(fingers), fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    current_time = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            total_fingers, fingers = count_fingers(hand_landmarks)
            
            if total_fingers != current_finger_count:
                current_finger_count = total_fingers
                last_gesture_change_time = current_time
                current_gesture_frames = 0
            
            current_gesture_frames += 1
            gesture_duration = current_time - last_gesture_change_time
            
            if stage in [1, 2, 3, 4]: 
                progress_width = int((gesture_duration / gesture_hold_time) * 200)
                cv2.rectangle(frame, (w//2 - 100, 50), (w//2 - 100 + progress_width, 70), (0, 255, 0), -1)
                cv2.rectangle(frame, (w//2 - 100, 50), (w//2 + 100, 70), (255, 255, 255), 2)
                cv2.putText(frame, f"Hold for {gesture_hold_time}s", (w//2 - 90, 65), font, 0.5, (0, 0, 0), 1)

            if stage == 0:
                if fingers == [1, 0, 0, 0, 0] and gesture_duration > 1.0:
                    stage += 1
                    last_gesture_change_time = current_time
            elif stage == 1:
                selected_dest = min(total_fingers, len(destinations)-1)
                if gesture_duration > gesture_hold_time:
                    stage += 1
                    last_gesture_change_time = current_time
            elif stage == 2:
                adult_tickets = min(max(total_fingers, 1), 5)  
                if gesture_duration > gesture_hold_time:
                    stage += 1
                    last_gesture_change_time = current_time
            elif stage == 3:
                child_tickets = min(total_fingers, 5) 
                if gesture_duration > gesture_hold_time:
                    stage += 1
                    last_gesture_change_time = current_time
            elif stage == 4:
                ticket_class = 1 if total_fingers >= 3 else 0
                if gesture_duration > gesture_hold_time:
                    stage += 1
                    last_gesture_change_time = current_time
            elif stage == 5:
                if fingers == [1, 0, 0, 0, 0] and gesture_duration > 1.0:
                    stage += 1
                    last_gesture_change_time = current_time

    if stage == 0:
        cv2.putText(frame, "Welcome to Gesture Ticket Booking!", (50, 150), font, 1, (255, 0, 0), 2)
        cv2.putText(frame, "Show THUMBS UP to Begin", (150, 200), font, 0.8, (255, 255, 255), 2)
    elif stage == 1:
        cv2.putText(frame, "Select Destination", (10, 30), font, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "Show number of fingers (0-4):", (10, 70), font, 0.6, (255, 255, 255), 1)
        for i, dest in enumerate(destinations):
            color = (0, 255, 0) if i == selected_dest else (255, 255, 255)
            cv2.putText(frame, f"{i}: {dest}", (10, 110 + i*30), font, 0.7, color, 1)
    elif stage == 2:
        cv2.putText(frame, "Number of Adult Tickets", (10, 30), font, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "Show number of fingers (1-5):", (10, 70), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Selected: {adult_tickets} adult(s)", (10, 110), font, 0.8, (0, 255, 0), 2)
    elif stage == 3:
        cv2.putText(frame, "Number of Child Tickets", (10, 30), font, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "Show number of fingers (0-5):", (10, 70), font, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Selected: {child_tickets} child(ren)", (10, 110), font, 0.8, (0, 255, 0), 2)
    elif stage == 4:
        cls = "1st Class" if ticket_class == 1 else "2nd Class"
        cv2.putText(frame, "Select Ticket Class", (10, 30), font, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "1-2 fingers: 2nd Class | 3+ fingers: 1st Class", (10, 70), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Selected: {cls}", (10, 110), font, 0.8, (0, 255, 0), 2)
    elif stage == 5:
        cv2.putText(frame, "Confirm Your Booking", (10, 30), font, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Destination: {destinations[selected_dest]}", (50, 80), font, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Adults: {adult_tickets}, Children: {child_tickets}", (50, 120), font, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Class: {'1st' if ticket_class else '2nd'}", (50, 160), font, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Show THUMBS UP to Confirm Booking", (50, 220), font, 0.8, (0, 255, 255), 2)
    elif stage > 5:
        cv2.putText(frame, "Booking Confirmed!", (100, 150), font, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, "Thank you for your booking!", (80, 200), font, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Press ESC to exit", (150, 450), font, 0.8, (255, 255, 255), 2)

    progress_colors = [(100, 100, 100)] * 6
    for i in range(min(stage + 1, 6)):
        progress_colors[i] = (0, 200, 0) if i < stage else (0, 255, 255)
    
    for i, color in enumerate(progress_colors):
        cv2.rectangle(frame, (50 + i*100, h - 50), (130 + i*100, h - 30), color, -1)
        cv2.putText(frame, str(i+1), (85 + i*100, h - 35), font, 0.5, (0, 0, 0), 1)

    cv2.putText(frame, f"Fingers: {current_finger_count}", (w - 150, 30), font, 0.7, (255, 255, 0), 2)
    
    cv2.imshow("Train Ticket Booking - Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()