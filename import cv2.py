import cv2

# Load the video
cap = cv2.VideoCapture('gta6.mp4')  # Replace with 0 for webcam

# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame (optional, for performance)
    frame = cv2.resize(frame, (640, 480))

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Find contours in the mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue  # Ignore small movements

        # Get bounding box
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show result
    cv2.imshow('Motion Tracking', frame)
    cv2.imshow('Foreground Mask', fgmask)

    if cv2.waitKey(30) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
