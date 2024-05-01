from datetime import datetime
import time

from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import numpy as np
from PIL import Image
import io
import torch

app = Flask(__name__,
            static_url_path="",
            static_folder="web/static",
            template_folder="web/templates")
app.config["SECRET_KEY"] = "firefighter"
socketio = SocketIO(app)


FIRE_BOX_COLOR = (0, 0, 255)
VEST_BOX_COLOR = (137, 87, 163)
HOUSE_BOX_COLOR = (56, 57, 83)
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


lower_purple = np.array([120, 50, 100], dtype=np.uint8)
upper_purple = np.array([150, 255, 255], dtype=np.uint8)

lower_maroon = np.array([170, 151, 140], dtype=np.uint8)
upper_maroon = np.array([83, 57, 56], dtype=np.uint8)

fire_cascade = cv2.CascadeClassifier("fire_mushiq.xml")

target = None
target_current_section = None
target_first_in_section: datetime = None
running = False

fire_model = torch.hub.load("yolov5", "custom", source="local", path="models/fire_best.pt")
fire_model.conf = 0.5
fire_model.iou = 0.2


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
    print("Down")
    socketio.emit("serial", "9")
    # socketio.emit("serial", "0")
    # socketio.emit("serial", "7")
    # socketio.emit("serial", "6")
    # socketio.emit("serial", "0")
    
def action_stop():
    print("Stop")
    socketio.emit("serial", "0")


def process_frame(frame):
    global target
    global target_current_section
    global target_first_in_section
    global running

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 480))
    width, height, _ = frame.shape

    # Calculate section width (all sections are equal)
    section_width = int(width / 3)

    # Draw section separators
    cv2.line(frame, (section_width, 0), (section_width, height),
             (0, 255, 0), 2)  # Green line for left section boundary
    cv2.line(frame, (3 * section_width, 0), (3 * section_width, height),
             (0, 255, 0), 2)  # Red line for right section boundary

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    vest_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    vest_cnts = cv2.findContours(vest_mask.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)[-2]
    vest_cnts = sorted(vest_cnts, key=cv2.contourArea, reverse=True)

    valid_follow_data = None
    if vest_cnts:
        largest_cnt = vest_cnts[0]
        x, y, w, h = cv2.boundingRect(largest_cnt)
        area = cv2.contourArea(largest_cnt)
        section = get_rect_horizontal_section(width, x, w)

        if area >= DETECT_AREA:
            cv2.rectangle(frame, (x, y), (x + w, y + h), VEST_BOX_COLOR, 2)
            data = {
                "area": area,
                "x": x, "y": y,
                "w": w, "h": h,
                "section": section,
            }
            socketio.emit("vest", data)
            valid_follow_data = data
    else:
        socketio.emit("vest", "No vest in sight")

    fires = fire_model(frame, size=640)
    valid_fire_data = None
    for xyxy in fires.pandas().xyxy:
        fire_df = xyxy[xyxy["name"] == "fire"]
        if fire_df.empty:
            continue

        x, y, w, h = get_bounding_box(fire_df["xmin"], fire_df["ymin"], fire_df["xmax"], fire_df["ymax"])
        cv2.rectangle(frame, (x, y), (x+w, y+h), FIRE_BOX_COLOR, 2)
        section = get_rect_horizontal_section(width, x, w)
        area = w * h
        if area >= DETECT_AREA:
            data = {
                "area": area,
                "x": x, "y": y,
                "w": w, "h": h,
                "section": section,
            }
            valid_fire_data = data

    if valid_fire_data or valid_follow_data:
        if valid_fire_data:
            new_target = "fire"
            data_use = valid_fire_data
        else:
            new_target = "vest"
            data_use = valid_follow_data

        section = data_use["section"]
        area = data_use["area"]
        
        print("Current Target: ", new_target)
        print("Section: ", section)
        print("Area: ", area)
        
        if running and area < STOP_AT_AREA:
            action_stop()
            running = False
            
            target = None
            target_current_section = None
            target_first_in_section = None
        else:
            if not target:
                target = new_target
                target_first_in_section = datetime.now()
                target_current_section = section
            else:
                if target == new_target:
                    if target_current_section == section:
                        if section =="middle" and area >= STOP_AT_AREA:
                            action_stop()
                        else:
                            now = datetime.now()
                            elapsed = now - target_first_in_section
                            if elapsed.seconds >= WAIT_FOR_SECONDS:
                                if section == "middle":
                                    if new_target == "fire" and area >= EXTINGUISH_AT_AREA:
                                        action_extinguish()
                                    else:
                                        action_forward()
                                        running = True
                                elif section == "left":
                                    action_left()
                                    running = True
                                elif section == "right":
                                    action_right()
                                    running = True
                    else:
                        target_current_section = section
                        target_first_in_section = datetime.now()
                else:
                    target = new_target
                    target_current_section = section
                    target_first_in_section = datetime.now()
    else:
        if target_first_in_section:
            now = datetime.now()
            elapsed = now - target_first_in_section
            if elapsed.seconds > 1:
                action_stop()
                running = False
        
        target = None
        target_current_section = None
        target_first_in_section = None
        
        

    ret, buffer = cv2.imencode(".jpg", frame)

    valid_fire_data = None
    valid_follow_data = None

    socketio.emit("frame", buffer.tobytes())


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("raw_frame")
def handle_new_frame(raw_frame):
    image = Image.open(io.BytesIO(bytes(raw_frame)))
    image_np = np.array(image)
    process_frame(image_np)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8001, debug=True)


# def house_detect(hsv):
#     house_mask = cv2.inRange(hsv, lower_maroon, upper_maroon)
#     house_cnts = cv2.findContours(house_mask.copy(), cv2.RETR_EXTERNAL,
#                                   cv2.CHAIN_APPROX_SIMPLE)[-2]
#     house_cnts = sorted(house_cnts, key=cv2.contourArea, reverse=True)
#     valid_house_data = None
#     if house_cnts:
#         largest_cnt = house_cnts[0]
#         x, y, w, h = cv2.boundingRect(largest_cnt)
#         area = cv2.contourArea(largest_cnt)
#         section = get_rect_horizontal_section(width, x, w)

#         if area >= min_vest_area:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), HOUSE_COLOR, 2)
#             data = {
#                 "area": area,
#                 "x": x, "y": y,
#                 "w": w, "h": h,
#                 "section": section,
#             }
#             socketio.emit("house", data)
#             print("House: {}".format(data))
#             valid_house_data = data
#     else:
#         socketio.emit("House", "No house in sight")

    # fires = fire_cascade.detectMultiScale(frame, 1.2, 5)
    # if fires is not None:
    #     fires = sorted(fires, key=lambda x: x[2] * x[3], reverse=True)

    # valid_fire_data = None
    # if fires:
    #     x, y, w, h = fires[0]
    #     x = int(x)
    #     y = int(y)
    #     w = int(w)
    #     h = int(h)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), FIRE_COLOR, 2)
    #     area = w * h
    #     section = get_rect_horizontal_section(width, x, w)
    #     data = {
    #         "area": area,
    #         "x": x, "y": y,
    #         "w": w, "h": h,
    #         "section": section,
    #     }
    #     socketio.emit("fire", data)
    #     print("Fire: {}".format(data))
    #     valid_fire_data = data
    # else:
    #     socketio.emit("fire", "No fire in sight")
