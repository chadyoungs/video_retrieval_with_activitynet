import os

import cv2


def get_demo_sample_image(video_file_loc, time_loc, dest_file_loc):
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

        if frame_count == time_loc * int(cap.get(cv2.CAP_PROP_FPS)):
            success = cv2.imwrite(dest_file_loc, frame)
            break

        frame_count += 1

    cap.release()


if __name__ == "__main__":
    video_file_loc = (
        "/mnt/sdc/activitynet_caption/v1-3/test/v_0_1BQPWzRiw.mp4"
    )

    query_img = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "v_0_1BQPWzRiw_sample.png",
    )
    
    time_loc = 50
    get_demo_sample_image(video_file_loc, time_loc, query_img)
