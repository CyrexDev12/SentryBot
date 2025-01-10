from picamera2 import Picamera2
import cv2
import os

def main():
    print("Initializing Picamera2...")
    picam2 = Picamera2()

    # Configure preview: 640x480 at default framerate.
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    print("Camera started with picamera2.")

    # ----------------------------
    # Use local Haar cascade file
    # ----------------------------
    # We assume you downloaded `haarcascade_frontalface_default.xml` to the same folder as main.py.
    cascade_path = os.path.join(
        os.path.dirname(__file__),  # directory containing this script
        "haarcascade_frontalface_default.xml"
    )
    print(f"INFO: Using Haar Cascade from: {cascade_path}")

    # Load the Haar cascade
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"ERROR: Could not load Haar cascade from {cascade_path}")
        picam2.stop()
        return
    else:
        print("INFO: Haar Cascade loaded successfully.")

    frame_count = 0

    while True:
        # Capture a frame as a NumPy array
        frame = picam2.capture_array()
        if frame is None:
            print("WARNING: No frame received. Exiting...")
            break

        frame_count += 1
        print(f"DEBUG: Processing frame {frame_count}")

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("DEBUG: Converted frame to grayscale.")

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        print(f"DEBUG: Detected {len(faces)} face(s) in frame {frame_count}.")

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(f"DEBUG: Drew rectangle around face at (x={x}, y={y}, w={w}, h={h}).")

        # Display the resulting frame
        cv2.imshow("Face Detection (picamera2)", frame)
        print("INFO: Displayed frame.")

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("INFO: 'q' pressed, exiting...")
            break

    picam2.stop()
    cv2.destroyAllWindows()
    print("INFO: Stopped picamera2 and destroyed all windows.")

if __name__ == "__main__":
    main()
