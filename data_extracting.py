import cv2
from tools.preprocess import *
import mediapipe as mp
from multiprocessing import Process
import os

gestures_path = [i for i in map(lambda x: f"./dataset/raw_data/{x}", os.listdir("dataset/raw_data"))]
gestures_Class = [(i, j[19:-4]) for i, j in enumerate(gestures_path)]
with open("dataset/labels/labels.csv", "w") as f:
    f.write("\n".join([i[1] for i in gestures_Class]))


def extract_data(num_frame, ges_class, ges_path):
    print(ges_class)
    cap = cv2.VideoCapture(ges_path)

    while cap.isOpened() and num_frame != 0:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        ratio = (9, 16)
        wid = 100
        # frame = cv2.resize(frame, (100*))
        if ret:
            mp_hands = mp.solutions.hands
            with mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5,
            ) as hands:

                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                num_frame -= 1
                height, width, _ = frame.shape
                for hand in results.multi_hand_landmarks:
                    # box_rect = calc_bounding_rect(frame, hand)
                    landmark_list = calc_landmark_list(frame, hand)
                    processed_landmark_list = pre_process_landmark(landmark_list)

                    # print("meomeo")

                    with open(f'./dataset/processed_data/{ges_class[1]}.csv', 'a') as f:
                        f.write(','.join(map(str, processed_landmark_list)) + f",{ges_class[0]}" + "\n")
        else:
            break
    print(f"{ges_class} finished")


if __name__ == "__main__":
    num_frame = 5000
    processes = [Process(target=extract_data, args=(num_frame, i, j,)) for i, j in zip(gestures_Class, gestures_path)]

    for i in gestures_Class:
        with open(f'dataset/processed_data/{i[1]}.csv', 'w') as f:
            f.write("".join([f'x{i},y{i},' for i in range(21)]) + "class" + "\n")
            pass

    for process in processes:
        process.start()
