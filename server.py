from datetime import datetime

from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import numpy as np
import torch

app = Flask(__name__,
            static_url_path="",
            static_folder="web/static",
            template_folder="web/templates")
app.config["SECRET_KEY"] = "firefighter"
socketio = SocketIO(app)

fire_model = torch.hub.load("yolov5", "custom", source="local", path="models/fire_best.pt")
fire_model.conf = 0.4
fire_model.iou = 0.2

WAIT_FOR_SECONDS = 1
DETECT_AREA = 500
STOP_AT_AREA = 4_000
EXTINGUISH_AT_AREA = 3_000


def get_rect_horizontal_section(frame_width, x, w):
    # Calculate center coordinates of the object and frame
    object_center_x = x + w // 2
    frame_center_x = frame_width // 2

    # Define thresholds for each section based on the frame width
    section_width = frame_width / 3
    left_threshold = section_width
    right_threshold = 3 * section_width

    if object_center_x < left_threshold:
        return "left"
    elif object_center_x > right_threshold:
        return "right"
    else:
        return "middle"


def get_rect_area(w, h):
    return w * h


def get_bounding_box(xmin, ymin, xmax, ymax):
    width = xmax - xmin
    height = ymax - ymin
    return int(xmin.iloc[0]), int(ymin.iloc[0]), int(width.iloc[0]), int(height.iloc[0])


def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around
    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit


def action_forward():
    print("Forward")
    socketio.emit("serial", "1")


def action_left():
    print("Left")
    socketio.emit("serial", "3")


def action_right():
    print("Right")
    socketio.emit("serial", "4")


def action_extinguish():
    print("Extinguish")
    socketio.emit("serial", "9")


def action_stop():
    print("Stop")
    socketio.emit("serial", "0")


FIRE_BOX_COLOR = (0, 0, 255)

VEST_BASE_COLOR = [2, 255, 174]  # BGR
lower_vest_color, upper_vest_color = get_limits(VEST_BASE_COLOR)

HOUSE_BASE_COLOR = [0, 255, 255]  # BGR
lower_house_color, upper_house_color = get_limits(HOUSE_BASE_COLOR)

FIRE = "fire"
VEST = "vest"
HOUSE = "house"
MOVE_SECONDS = 3

processing = False
action_start_time: datetime = None
current_target = None


def detect(frame, detection_type: str):
    width = frame.shape[0]

    if detection_type == FIRE:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fires = fire_model(frame, size=640)
        for xyxy in fires.pandas().xyxy:
            fire_df = xyxy[xyxy["name"] == "fire"]
            if fire_df.empty:
                continue
            x, y, w, h = get_bounding_box(fire_df["xmin"], fire_df["ymin"], fire_df["xmax"], fire_df["ymax"])
            section = get_rect_horizontal_section(width, x, w)
            area = w * h
            if area >= DETECT_AREA:
                return {
                    "area": w * h,
                    "x": x, "y": y,
                    "w": w, "h": h,
                    "section": section,
                    "color": FIRE_BOX_COLOR
                }
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if detection_type == VEST:
            vest_mask = cv2.inRange(hsv, lower_vest_color, upper_vest_color)
            vest_contours = cv2.findContours(vest_mask.copy(), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)[-2]
            vest_contours = sorted(vest_contours, key=cv2.contourArea, reverse=True)
            if vest_contours:
                vest_detect = vest_contours[0]
                x, y, w, h = cv2.boundingRect(vest_detect)
                area = cv2.contourArea(vest_detect)
                section = get_rect_horizontal_section(width, x, w)
                if area >= DETECT_AREA:
                    return {
                        "area": area,
                        "x": x, "y": y,
                        "w": w, "h": h,
                        "section": section,
                        "color": VEST_BASE_COLOR
                    }
        elif detection_type == HOUSE:
            house_mask = cv2.inRange(hsv, lower_house_color, upper_house_color)
            house_contours = cv2.findContours(house_mask.copy(), cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)[-2]
            house_contours = sorted(house_contours, key=cv2.contourArea, reverse=True)
            if house_contours:
                house_detect = house_contours[0]
                x, y, w, h = cv2.boundingRect(house_detect)
                area = cv2.contourArea(house_detect)
                section = get_rect_horizontal_section(width, x, w)
                if area >= DETECT_AREA:
                    return {
                        "area": area,
                        "x": x, "y": y,
                        "w": w, "h": h,
                        "section": section,
                        "color": HOUSE_BASE_COLOR
                    }

    return None


def appropriate_action(target: str, detection_data: dict):
    socketio.emit("announcer", f"Actioning {target}")

    area = detection_data["area"]
    socketio.emit("announcer", f"Target area: {area}")
    section = detection_data["section"]
    socketio.emit("announcer", f"Target section: {section}")

    print(detection_data)

    if section == "middle":
        if target == FIRE and area >= EXTINGUISH_AT_AREA:
            socketio.emit("announcer", "Triggering fire extinguisher...")
            action_extinguish()
        else:
            socketio.emit("announcer", "Moving closer...")
            action_forward()
    elif section == "left":
        socketio.emit("announcer", "Going left...")
        action_left()
    elif section == "right":
        socketio.emit("announcer", "Going right...")
        action_right()


def process_frame(frame):
    global action_start_time
    global current_target

    frame = cv2.resize(frame, (640, 480))
    width, height, _ = frame.shape
    section_width = int(width / 3)
    cv2.line(frame, (section_width, 0), (section_width, height), (0, 255, 0), 2)
    cv2.line(frame, (3 * section_width, 0), (3 * section_width, height), (0, 255, 0), 2)

    def draw_rectangle(frame, detection_data):
        x = detection_data["x"]
        y = detection_data["y"]
        w = detection_data["w"]
        h = detection_data["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), detection_data["color"], 2)

    if action_start_time:
        elapsed = datetime.now() - action_start_time
        if elapsed.seconds < MOVE_SECONDS:
            socketio.emit("announcer", "Movement lock, skipping frame...")
            return frame

    if current_target:
        socketio.emit("announcer", f"Looking for target {current_target}...")
        detected = detect(frame, current_target)
        if detected:
            socketio.emit("announcer", "Target found.")
            draw_rectangle(frame, detected)
            appropriate_action(current_target, detected)
            return frame
        else:
            socketio.emit("announcer", "No target found. Looking for next priority")

    socketio.emit("announcer", "Looking for fire...")
    found_fire = detect(frame, FIRE)
    if found_fire:
        socketio.emit("announcer", "Fire found")
        draw_rectangle(frame, found_fire)

        action_start_time = datetime.now()
        appropriate_action(FIRE, found_fire)
        current_target = FIRE

        return frame

    socketio.emit("announcer", "Looking for vest...")
    found_vest = detect(frame, VEST)
    if found_vest:
        socketio.emit("announcer", "Vest found")
        draw_rectangle(frame, found_vest)

        action_start_time = datetime.now()
        appropriate_action(VEST, found_vest)

        current_target = VEST
        return frame

    socketio.emit("announcer", "Looking for house...")
    found_house = detect(frame, HOUSE)
    if found_house:
        socketio.emit("announcer", "House found")
        draw_rectangle(frame, found_house)

        action_start_time = datetime.now()
        appropriate_action(HOUSE, found_house)

        current_target = HOUSE
        return frame

    socketio.emit("announcer", "Did not find anything, stopping all motors and resetting state.")
    action_stop()
    current_target = None
    action_start_time = None
    return frame


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("raw_frame")
def handle_new_frame(raw_frame):
    global processing

    print("Received new frame")

    if processing:
        return

    image_np = np.frombuffer(bytes(raw_frame), dtype=np.uint8)
    try:
        print("Processing frame")
        processing = True
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if frame is not None:
            result_frame = process_frame(frame)
            _, buffer = cv2.imencode(".jpg", result_frame)
            socketio.emit("frame", buffer.tobytes())
    except:
        socketio.emit("announcer", "Cant process frame")
    finally:
        processing = False


def house_tester_img():
    from pathlib import Path
    image_folder = Path.home() / "Downloads" / "test images"

    for image in image_folder.glob("*.jpg"):
        frame = cv2.imread(str(image.absolute()), cv2.IMREAD_UNCHANGED)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))

        house_detect = detect(frame, HOUSE)
        if house_detect:
            x = house_detect["x"]
            y = house_detect["y"]
            w = house_detect["w"]
            h = house_detect["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), HOUSE_BASE_COLOR, 2)

        cv2.imshow(image.name, frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def vest_tester_img():
    from pathlib import Path
    image_folder = Path.home() / "Downloads" / "Vest"

    for image in image_folder.glob("*.jpg"):
        frame = cv2.imread(str(image.absolute()), cv2.IMREAD_UNCHANGED)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (640, 480))

        vest_detect = detect(frame, FIRE)
        if vest_detect:
            x = vest_detect["x"]
            y = vest_detect["y"]
            w = vest_detect["w"]
            h = vest_detect["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), VEST_BASE_COLOR, 2)

        cv2.imshow(image.name, frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def house_tester_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        house_detect = detect(frame, HOUSE)
        if house_detect:
            x = house_detect["x"]
            y = house_detect["y"]
            w = house_detect["w"]
            h = house_detect["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), HOUSE_BASE_COLOR, 2)

        cv2.imshow("orig", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def fire_tester():
    cap = cv2.VideoCapture("videos/input.mp4")
    while True:
        ret, frame = cap.read()
        house_detect = detect(frame, FIRE)
        if house_detect:
            x = house_detect["x"]
            y = house_detect["y"]
            w = house_detect["w"]
            h = house_detect["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), FIRE_BOX_COLOR, 2)

        cv2.imshow("orig", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8001, debug=True)
