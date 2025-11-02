import cv2
import numpy as np
import time
import HandTracking as ht
import pyautogui

### Variables Declaration
pTime = 0               # Used to calculate frame rate
width = 640             # Width of Camera
height = 480            # Height of Camera
frameR = 100            # Frame Rate
smoothening = 7         # Smoothening Factor
prev_x, prev_y = 0, 0   # Previous coordinates
curr_x, curr_y = 0, 0   # Current coordinates
is_paused = False       # Pause flag
scroll_mode_active = False
scroll_baseline_x, scroll_baseline_y = 0, 0

cap = cv2.VideoCapture(0)   # Getting video feed from the webcam
cap.set(3, width)           # Adjusting size
cap.set(4, height)

detector = ht.handDetector(maxHands=1)                  # Detecting one hand at max
screen_width, screen_height = pyautogui.size()      # Getting the screen size
while True:
    if not is_paused:
        success, img = cap.read()
        img = detector.findHands(img)                       # Finding the hand
        lmlist, bbox = detector.findPosition(img)           # Getting position of hand

        if len(lmlist)!=0:
            x1, y1 = lmlist[8][1:]
            x2, y2 = lmlist[12][1:]

            fingers = detector.fingersUp()      # Checking if fingers are upwards
            cv2.rectangle(img, (frameR, frameR), (width - frameR, height - frameR), (255, 0, 255), 2)   # Creating boundary box

            # --- New Dynamic Scrolling Logic ---
            if detector.allFingersUp():
                if not scroll_mode_active:
                    # First frame of scroll gesture: set baseline and activate mode
                    scroll_mode_active = True
                    scroll_baseline_x, scroll_baseline_y = lmlist[9][1:] # Use palm base as baseline
                    cv2.circle(img, (scroll_baseline_x, scroll_baseline_y), 15, (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "SCROLL LOCK ON", (150, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                else:
                    # Continue scrolling based on baseline
                    x_palm, y_palm = lmlist[9][1:]
                    cv2.circle(img, (scroll_baseline_x, scroll_baseline_y), 15, (0, 0, 255), cv2.FILLED) # Keep showing baseline
                    
                    dx = x_palm - scroll_baseline_x
                    dy = y_palm - scroll_baseline_y
                    
                    dead_zone = 30 # Prevent jitter
                    # These constants are lowered to work with the new power-based calculation.
                    # You can tweak them to adjust sensitivity.
                    scroll_speed_v = 0.005
                    scroll_speed_h = 0.005

                    # Vertical Scrolling
                    if abs(dy) > dead_zone:
                        # Calculate distance past the dead zone
                        effective_dy = dy - dead_zone * np.sign(dy)
                        # Apply a power function (squared) for non-linear, accelerating speed
                        scroll_amount = -int(np.sign(effective_dy) * (abs(effective_dy) ** 2) * scroll_speed_v)
                        if scroll_amount != 0:
                            pyautogui.scroll(scroll_amount)

                    # Horizontal Scrolling
                    if abs(dx) > dead_zone:
                        effective_dx = dx - dead_zone * np.sign(dx)
                        scroll_amount = int(np.sign(effective_dx) * (abs(effective_dx) ** 2) * scroll_speed_h)
                        if scroll_amount != 0:
                            pyautogui.hscroll(scroll_amount)
            else:
                # If not all fingers are up, deactivate scroll mode and handle other gestures
                if scroll_mode_active:
                    scroll_mode_active = False
                
                if fingers[1] == 1 and fingers[2] == 0:     # If fore finger is up and middle finger is down
                    x3 = np.interp(x1, (frameR, width - frameR), (0, screen_width))
                    y3 = np.interp(y1, (frameR, height - frameR), (0, screen_height))

                    curr_x = prev_x + (x3 - prev_x)/smoothening
                    curr_y = prev_y + (y3 - prev_y) / smoothening

                    pyautogui.moveTo(screen_width - curr_x, curr_y)    # Moving the cursor
                    cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
                    prev_x, prev_y = curr_x, curr_y

                elif fingers[1] == 1 and fingers[2] == 1:     # If fore finger & middle finger both are up
                    length, img, lineInfo = detector.findDistance(8, 12, img)

                    if length < 40:     # If both fingers are really close to each other
                        cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                        pyautogui.click()    # Perform Click

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    else:
        cv2.putText(img, "Paused", (width//2 - 50, height//2), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 32:  # Spacebar to toggle pause
        is_paused = not is_paused
    elif key == 27:  # Esc to exit
        break
