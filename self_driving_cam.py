import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO(r"C:\Users\Lenovo\OneDrive\Desktop\AIROBOCAR\yolov8n.pt")


ROBOT_WIDTH_CM = 18
ROBOT_LENGTH_CM = 20
ROBOT_HEIGHT_CM = 25
MIN_WHEEL_TRACK_CM = 6  

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def detect_lane_lines(frame):
    
    
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    
    roi_vertices = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.6)),
        (0, int(height * 0.6))
    ]], np.int32)
    masked_edges = region_of_interest(edges, roi_vertices)

    
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=150)

    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = 0 if (x2 - x1) == 0 else (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:  
                continue
            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])

    def average_line(lines):
        if len(lines) == 0:
            return None
        x_coords = []
        y_coords = []
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        poly = np.polyfit(y_coords, x_coords, 1)  
        slope = poly[0]
        intercept = poly[1]
        y1 = height
        y2 = int(height * 0.6)
        x1 = int(slope * y1 + intercept)
        x2 = int(slope * y2 + intercept)
        return (x1, y1, x2, y2)

    left_avg = average_line(left_lines)
    right_avg = average_line(right_lines)

    lane_img = frame.copy()
    if left_avg is not None:
        cv2.line(lane_img, (left_avg[0], left_avg[1]), (left_avg[2], left_avg[3]), (0, 255, 0), 5)
    if right_avg is not None:
        cv2.line(lane_img, (right_avg[0], right_avg[1]), (right_avg[2], right_avg[3]), (0, 255, 0), 5)

   
    if left_avg is not None and right_avg is not None:
        lane_center_x = (left_avg[0] + right_avg[0]) // 2
    elif left_avg is not None:
        lane_center_x = left_avg[0] + 200
    elif right_avg is not None:
        lane_center_x = right_avg[0] - 200
    else:
        lane_center_x = width // 2

    return lane_img, lane_center_x

def decide_steering(lane_center_x, frame_width, obstacle_detected, obstacle_position=None):
    """
    Advanced steering logic using proportional control and obstacle-aware biasing.
    Returns a string direction and steering angle (for future motor integration).
    """
    center_offset = lane_center_x - frame_width // 2
    max_steering_angle = 30  # degrees (you can map this to motor PWM later)
    
    # Proportional control: angle is proportional to how far off-center we are
    Kp = 0.1  # steering gain (adjust based on your robot's behavior)
    steering_angle = Kp * center_offset

    direction = ""
    if obstacle_detected:
        # Add avoidance bias if obstacle is close to center
        if obstacle_position == "CENTER":
            direction = "AVOID CENTER OBSTACLE - TURN LEFT"
            steering_angle -= 15  # steer more to the left
        elif obstacle_position == "LEFT":
            direction = "AVOID LEFT OBSTACLE - TURN RIGHT"
            steering_angle += 10
        elif obstacle_position == "RIGHT":
            direction = "AVOID RIGHT OBSTACLE - TURN LEFT"
            steering_angle -= 10
    else:
        if abs(center_offset) < 20:
            direction = "GO STRAIGHT"
        elif center_offset > 0:
            direction = "STEER RIGHT"
        else:
            direction = "STEER LEFT"

    # Clamp steering angle
    steering_angle = max(min(steering_angle, max_steering_angle), -max_steering_angle)

    return direction, steering_angle


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        lane_frame, lane_center_x = detect_lane_lines(frame)
        results = model(frame)[0]

        obstacle_detected = False
        obstacle_position = None
        frame_center_x = frame.shape[1] // 2
        danger_zone_y = int(frame.shape[0] * 0.75)

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            color = (0, 0, 255)
            cv2.rectangle(lane_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(lane_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2
            if box_center_y > danger_zone_y:
                obstacle_detected = True
                if abs(box_center_x - frame_center_x) < 80:
                    obstacle_position = "CENTER"
                elif box_center_x < frame_center_x:
                    obstacle_position = "LEFT"
                else:
                    obstacle_position = "RIGHT"

        # === Enhanced Steering Logic ===
        direction, steering_angle = decide_steering(
            lane_center_x, frame.shape[1], obstacle_detected, obstacle_position
        )

        cv2.putText(lane_frame, f"Direction: {direction}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.putText(lane_frame, f"Steering Angle: {steering_angle:.1f} deg", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(lane_frame, f"Robot Size (WxLxH cm): {ROBOT_WIDTH_CM}x{ROBOT_LENGTH_CM}x{ROBOT_HEIGHT_CM}", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(lane_frame, "Wheel Track >= 6cm Front & Rear", (10, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("AI Autonomous Driving Robot", lane_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
