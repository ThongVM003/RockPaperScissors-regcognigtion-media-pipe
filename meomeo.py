import mediapipe as mp
import cv2
import onnxruntime as rt
from tools.preprocess import *

sess = rt.InferenceSession(
    "./results/trained_models/meomeo.onnx", providers=["CPUExecutionProvider"]
)

# get input and output name
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# get labels
with open("dataset/labels/labels.csv") as f:
    gestures = [i.strip() for i in f]

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
path = "./dataset/raw_data/scissors.mp4"
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame_fliped = cv2.flip(frame, 1)

    if ret:
        results = hands.process(cv2.cvtColor(frame_fliped, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            height, width, _ = frame_fliped.shape
            for hand, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                box_rect = calc_bounding_rect(frame_fliped, hand)
                landmark_list = calc_landmark_list(frame_fliped, hand)
                processed_landmark_list = pre_process_landmark(landmark_list)
                cv2.rectangle(
                    frame_fliped,
                    (box_rect[0], box_rect[1]),
                    (box_rect[2], box_rect[3]),
                    (0, 0, 0),
                    1,
                )

                processed_landmark_list = np.expand_dims(
                    np.array(processed_landmark_list), axis=0
                )

                dae = sess.run(None, {input_name: processed_landmark_list})[0]
                sign = gestures[np.squeeze(np.argmax(dae))]
                frame_fliped = draw_info_text(
                    image=frame_fliped,
                    brect=box_rect,
                    handedness=handedness,
                    hand_sign_text=sign,
                )

    else:
        break
    cv2.imshow("RPS", frame_fliped)
    key = cv2.waitKey(10)
    if key == 27:
        break
cv2.destroyAllWindows()
