import cv2
import os
import mediapipe as mp

def main():
    # Initialize MediaPipe Hand module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Create output directory
    output_dir = 'dataset'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe Hand module
        results = hands.process(rgb_frame)

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the annotated frame
        cv2.imshow('Sign Language Dataset Generator', frame)

        # Prompt user for label and save frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            label = input("Enter label for this gesture: ")
            save_frame(frame, label, output_dir)
        elif key == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def save_frame(frame, label, output_dir):
    # Create directory for the label if it doesn't exist
    label_dir = os.path.join(output_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Generate unique filename
    filename = os.path.join(label_dir, f"{label}_{len(os.listdir(label_dir))}.jpg")

    # Save frame to disk
    cv2.imwrite(filename, frame)
    print(f"Frame saved: {filename}")

if __name__ == "__main__":
    main()
