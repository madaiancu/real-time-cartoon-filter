import cv2
import numpy as np
import time
from datetime import datetime


def nothing(x):
    pass


def get_timestamp_filename(prefix="capture", extension="png"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def stack_images(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return np.hstack((img1, img2))


def cartoon_filter(frame, alpha, beta, bilateral_d, sigma_color, sigma_space, edge_block, edge_c):
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    color = cv2.bilateralFilter(adjusted, bilateral_d, sigma_color, sigma_space)
    color = cv2.GaussianBlur(color, (3, 3), 0)

    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    if edge_block % 2 == 0:
        edge_block += 1
    if edge_block < 3:
        edge_block = 3

    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        edge_block,
        edge_c
    )

    result = cv2.bitwise_and(color, color, mask=edges)
    return result


def sketch_filter(frame, alpha, beta):
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blur = 255 - blurred

    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def bw_filter(frame, alpha, beta):
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def comic_filter(frame, alpha, beta):
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        9,
        5
    )

    color = cv2.bilateralFilter(adjusted, 9, 200, 200)
    result = cv2.bitwise_and(color, color, mask=edges)

    return result


def draw_overlay(image, mode, fps, face_count, recording):
    cv2.putText(
        image,
        "Real-time Cartoon Filter + Face Detection",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        image,
        f"Mode: {mode}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        image,
        f"FPS: {int(fps)}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"Faces detected: {face_count}",
        (10, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    rec_text = "REC: ON" if recording else "REC: OFF"
    rec_color = (0, 0, 255) if recording else (180, 180, 180)
    cv2.putText(
        image,
        rec_text,
        (10, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        rec_color,
        2
    )

    cv2.putText(
        image,
        "1-cartoon  2-sketch  3-bw  4-comic",
        (10, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        image,
        "S-save  R-record  V-split  ESC-exit",
        (10, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Nu s-a putut deschide camera.")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        print("Nu s-a putut încărca detectorul Haar Cascade.")
        cap.release()
        return

    cv2.namedWindow("Controls")

    cv2.createTrackbar("Brightness", "Controls", 50, 100, nothing)
    cv2.createTrackbar("Contrast x100", "Controls", 130, 300, nothing)
    cv2.createTrackbar("Bilateral D", "Controls", 9, 25, nothing)
    cv2.createTrackbar("Sigma Color", "Controls", 120, 255, nothing)
    cv2.createTrackbar("Sigma Space", "Controls", 120, 255, nothing)
    cv2.createTrackbar("Edge Block", "Controls", 9, 31, nothing)
    cv2.createTrackbar("Edge C", "Controls", 5, 20, nothing)

    mode = "cartoon"
    show_split = True
    previous_time = time.time()

    recording = False
    video_writer = None
    current_video_filename = None

    print("Taste disponibile:")
    print("1 - cartoon")
    print("2 - sketch")
    print("3 - black & white")
    print("4 - comic")
    print("S - save screenshot")
    print("R - start/stop recording")
    print("V - split screen on/off")
    print("ESC - exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nu s-a putut citi cadrul din cameră.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        brightness = cv2.getTrackbarPos("Brightness", "Controls")
        contrast_x100 = cv2.getTrackbarPos("Contrast x100", "Controls")
        bilateral_d = cv2.getTrackbarPos("Bilateral D", "Controls")
        sigma_color = cv2.getTrackbarPos("Sigma Color", "Controls")
        sigma_space = cv2.getTrackbarPos("Sigma Space", "Controls")
        edge_block = cv2.getTrackbarPos("Edge Block", "Controls")
        edge_c = cv2.getTrackbarPos("Edge C", "Controls")

        alpha = max(0.1, contrast_x100 / 100.0)
        beta = brightness - 50

        if mode == "cartoon":
            output = cartoon_filter(
                frame,
                alpha,
                beta,
                max(1, bilateral_d),
                max(1, sigma_color),
                max(1, sigma_space),
                max(3, edge_block),
                edge_c
            )
        elif mode == "sketch":
            output = sketch_filter(frame, alpha, beta)
        elif mode == "bw":
            output = bw_filter(frame, alpha, beta)
        else:
            output = comic_filter(frame, alpha, beta)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)

        current_time = time.time()
        fps = 1.0 / (current_time - previous_time) if current_time != previous_time else 0
        previous_time = current_time

        draw_overlay(output, mode, fps, len(faces), recording)

        if show_split:
            display = stack_images(frame, output)
        else:
            display = output

      
        if recording and video_writer is None:
            current_video_filename = get_timestamp_filename("screen_recording", "avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            height, width = display.shape[:2]
            video_writer = cv2.VideoWriter(current_video_filename, fourcc, 20.0, (width, height))

            if video_writer.isOpened():
                print(f"Inregistrare pornita: {current_video_filename}")
            else:
                print("Nu s-a putut porni inregistrarea video.")
                video_writer = None
                recording = False

        if recording and video_writer is not None:
            video_writer.write(display)

        cv2.imshow(" Cartoon Project", display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == ord('1'):
            mode = "cartoon"

        elif key == ord('2'):
            mode = "sketch"

        elif key == ord('3'):
            mode = "bw"

        elif key == ord('4'):
            mode = "comic"

        elif key == ord('v'):
            show_split = not show_split

           
            if recording and video_writer is not None:
                video_writer.release()
                video_writer = None
                recording = False
                print("Inregistrarea a fost oprita deoarece s-a schimbat modul split/full.")

        elif key == ord('s'):
            filename = get_timestamp_filename("filter", "png")
            cv2.imwrite(filename, display)
            print(f"Imagine salvata: {filename}")

        elif key == ord('r'):
            recording = not recording

            if not recording and video_writer is not None:
                video_writer.release()
                video_writer = None
                print(f"Inregistrare oprita: {current_video_filename}")

    cap.release()

    if video_writer is not None:
        video_writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()