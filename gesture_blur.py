import mediapipe as mp

print(mp)
print(mp.__file__)
print(dir(mp))

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    finger_count = 0

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            tips = [4, 8, 12, 16, 20]

            if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                finger_count += 1

            for tip in tips[1:]:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                    finger_count += 1

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    if finger_count == 2:
        frame = cv2.GaussianBlur(frame, (55, 55), 0)

    cv2.putText(
        frame,
        f"Jari: {finger_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Gesture Blur Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()