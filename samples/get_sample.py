import os

import cv2


def get_demo_sample_image(video_file_loc, dest_file_loc):
    cap = cv2.VideoCapture(video_file_loc)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file: {video_file_loc}")
        return None

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count == 50:
            success = cv2.imwrite(dest_file_loc, frame)
            break

        frame_count += 1

    cap.release()


if __name__ == "__main__":
    video_file_loc = (
        "/ext-data/datasets/training_lib_KTH/person01_handclapping_d1_uncomp.avi"
    )

    query_img = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "person01_handclapping_d1_uncomp_sample.png",
    )

    get_demo_sample_image(video_file_loc, query_img)
