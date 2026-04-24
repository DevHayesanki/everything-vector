# Vector's Vision: OpenCV with Vector

## The Idea

Vector has a camera exposed to the Python SDK, accessible with `robot.camera.init_camera_feed()`. Paired with OpenCV, this opens up a whole world of computer vision tricks:

**Things he CAN do:**
- Play Rock Paper Scissors
- Interact with and follow objects other than the cube (laser pointer, anyone?)
- Guard your snacks
- Detect faces and react to them
- Track colors or shapes in real time

**Things he CAN'T do:**
- Use OpenCV in wire-pod
- Take over the world
- Understand why you keep pointing the laser at him

---

## Prerequisites

Make sure you have the following installed:

```bash
pip install anki-vector opencv-python Pillow numpy
```

You'll also need:
- A configured Vector robot (run `python -m anki_vector.configure` if you haven't)
- Python 3.6+

---

## Getting the Camera Feed

Before doing anything with OpenCV, you need to pull frames from Vector's camera. Here's the basic setup:

```python
import anki_vector
import time

with anki_vector.Robot(enable_camera_feed=True) as robot:
    time.sleep(1)  # Give the feed a moment to warm up

    image = robot.camera.latest_image  # Returns a PIL Image
    image.raw_image.show()
```

To use it with OpenCV, you need to convert the PIL image to a NumPy array (which is what OpenCV works with):

```python
import anki_vector
import cv2
import numpy as np
import time

with anki_vector.Robot(enable_camera_feed=True) as robot:
    time.sleep(1)

    pil_image = robot.camera.latest_image.raw_image

    # Convert PIL → NumPy → BGR (OpenCV format)
    frame = np.array(pil_image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Vector's View", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

---

## Live Feed Loop

For a real live feed, you loop continuously and grab the latest frame each iteration:

```python
import anki_vector
import cv2
import numpy as np
import time

with anki_vector.Robot(enable_camera_feed=True) as robot:
    time.sleep(1)  # Warm up

    while True:
        latest = robot.camera.latest_image
        if latest is None:
            continue

        # Convert to OpenCV format
        frame = np.array(latest.raw_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("Vector's View", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
```

> **Tip:** Vector's camera runs at a low resolution (640×360). Don't expect HD — but it's more than enough for object detection and color tracking.

---

## Adding Objects with OpenCV

### 1. Drawing Bounding Boxes

You can draw rectangles on the frame to highlight detected regions. This is the foundation of almost all object detection overlays:

```python
import cv2
import numpy as np

# Draw a rectangle on the frame
# cv2.rectangle(image, top_left, bottom_right, color_BGR, thickness)
cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)

# Add a label above the box
cv2.putText(frame, "Object", (50, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
```

---

### 2. Color Detection (Track a Colored Object)

A classic OpenCV trick — detect a specific color using HSV (Hue, Saturation, Value) color space and draw a box around it:

```python
import anki_vector
import cv2
import numpy as np
import time

def detect_color(frame, lower_hsv, upper_hsv):
    """Returns a bounding box around the largest detected color blob, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Clean up noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:  # Ignore tiny blobs
            x, y, w, h = cv2.boundingRect(largest)
            return (x, y, w, h)
    return None


# HSV range for red (laser pointer / red object)
LOWER_RED = np.array([0, 120, 70])
UPPER_RED = np.array([10, 255, 255])

with anki_vector.Robot(enable_camera_feed=True) as robot:
    time.sleep(1)

    while True:
        latest = robot.camera.latest_image
        if latest is None:
            continue

        frame = np.array(latest.raw_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        bbox = detect_color(frame, LOWER_RED, UPPER_RED)

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Red Object", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Vector's View", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
```

> **HSV Color Cheat Sheet:**
> | Color  | Lower HSV       | Upper HSV        |
> |--------|-----------------|------------------|
> | Red    | `[0, 120, 70]`  | `[10, 255, 255]` |
> | Green  | `[36, 100, 100]`| `[86, 255, 255]` |
> | Blue   | `[94, 80, 2]`   | `[126, 255, 255]`|
> | Yellow | `[20, 100, 100]`| `[30, 255, 255]` |

---

### 3. Face Detection

OpenCV comes with a pre-trained Haar Cascade classifier for faces. You can use it to detect faces in Vector's view and draw boxes around them:

```python
import anki_vector
import cv2
import numpy as np
import time

# Load the pre-trained face detector (comes bundled with OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

with anki_vector.Robot(enable_camera_feed=True) as robot:
    time.sleep(1)

    while True:
        latest = robot.camera.latest_image
        if latest is None:
            continue

        frame = np.array(latest.raw_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show face count on screen
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Vector's View", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
```

---

### 4. Making Vector React to What He Sees

Detection is fun, but making Vector *respond* is where things get interesting. Here's an example where Vector says something when he spots a face:

```python
import anki_vector
import cv2
import numpy as np
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

with anki_vector.Robot(enable_camera_feed=True) as robot:
    time.sleep(1)
    already_reacted = False

    while True:
        latest = robot.camera.latest_image
        if latest is None:
            continue

        frame = np.array(latest.raw_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0 and not already_reacted:
            print("Face detected! Telling Vector...")
            robot.behavior.say_text("I see you!")
            already_reacted = True
        elif len(faces) == 0:
            already_reacted = False  # Reset so he can react again

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Vector's View", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
```

> **Note:** `robot.behavior.say_text()` is blocking, so calling it every frame will freeze the feed. The `already_reacted` flag prevents that.

---

## Putting It All Together

Here's a template you can build from — it runs the live feed and is ready for you to drop any detection logic into:

```python
import anki_vector
import cv2
import numpy as np
import time


def process_frame(frame):
    """Add your detection logic here. Return the annotated frame."""

    # Example: draw a crosshair in the center
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)

    return frame


with anki_vector.Robot(enable_camera_feed=True) as robot:
    print("Connecting to Vector...")
    time.sleep(1)
    print("Feed started. Press 'q' to quit.")

    while True:
        latest = robot.camera.latest_image
        if latest is None:
            continue

        frame = np.array(latest.raw_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = process_frame(frame)

        cv2.imshow("Vector's View", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Feed closed.")
```

---

## Example Programs

A full Rock Paper Scissors game is included as `rps.py`. Run it with:

```bash
python rps.py
```

Vector will watch your hand via his camera, count down, read your gesture, pick his own move, and announce the result out loud. It uses [MediaPipe](https://mediapipe.dev/) for hand tracking, so install that too:

```bash
pip install mediapipe
```

---

## Ideas to Try Next

- **Laser pointer follower** — use red color detection and make Vector's head track the dot
- **Snack guardian** — detect a specific object and trigger an animation when it moves
- **Optical flow** — detect motion in the scene using `cv2.calcOpticalFlowFarneback()`
- **QR code reader** — use `cv2.QRCodeDetector()` to let Vector read codes

---

*Built with the [Anki Vector Python SDK](https://github.com/anki/vector-python-sdk) and [OpenCV](https://opencv.org/).*
