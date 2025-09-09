import gym
import numpy as np
import cv2
from gym_duckietown.envs import DuckietownEnv
from wrapper import PerceptionModule  # Assumed available

class PIDController:
    def __init__(self, kp, ki=0, kd=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0

    def update(self, error, dt=1):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def ideal_lane(y: float, W: int, H: int) -> float:
    # Define start and end points as percentages
    x1, y1 = W * 0.05, H * 0.95  # Bottom-leftish
    x2, y2 = W * 0.40, H * 0.35  # Horizon cutoff

    slope = (x2 - x1) / (y2 - y1)
    intercept = x1 - slope * y1

    return slope * y + intercept

class LaneFollower:
    def __init__(self, kp=1.5, ki=0, kd=0.3):
        self.pid = PIDController(kp, ki, kd)
        self.last_error = 0
        self.prev_steering = 0.0
        self.prev_forward_speed = 0.0

    def lane_following_control(self, seg_mask):
        H, W = seg_mask.shape
        scan_lines = np.linspace(int(H * 0.4), int(H * 1.0) - 1, 20).astype(int).tolist()

        y_vertical_line = int(H * 0.5) 
        weighted_weights = np.linspace(1.0, 0.05, 20)
        weighted_weights /= weighted_weights.sum()  # normalize to sum 1
        weighted_weights = weighted_weights.tolist()

        # Collect mid lane points
        mid_lane_xs = []
        for y in scan_lines:
            line = seg_mask[y, :]
            mid_lane_x = np.where(line == 3)[0]
            if len(mid_lane_x) > 0:
                mid_lane_xs.append(np.mean(mid_lane_x))
            else:
                mid_lane_xs.append(None)

        # No fallback policy, ignore right lane

        valid_positions = [(x, w) for x, w in zip(mid_lane_xs, weighted_weights) if x is not None]
        if valid_positions:
            positions, weights = zip(*valid_positions)
            x_mid_weighted = np.average(positions, weights=weights)
        else:
            # No valid mid lane points, hold previous action
            return [self.prev_forward_speed, self.prev_steering], self.last_error, {
                "info": "No valid mid lane points, holding previous action",
                "fallback": False,
                "mid_lane_xs": mid_lane_xs,
                "right_lane_xs": None,
                "scan_lines": scan_lines,
                "right_danger": False,
                "left_danger": False,
                "x_vertical_line": int(ideal_lane(y_vertical_line, W, H)),
                "x_target": None,
                "steering": self.prev_steering,
                "forward_speed": self.prev_forward_speed,
            }

        # Danger detection zone
        danger_zone_y = int(H * 0.85)
        danger_zone_x_start_right = int(W * 0.75)
        right_danger_line = seg_mask[danger_zone_y, danger_zone_x_start_right:]
        right_danger_detected = np.any(right_danger_line == 2)

        danger_zone_x_end_left = int(W * 0.25)
        left_danger_line = seg_mask[danger_zone_y, :danger_zone_x_end_left]
        left_danger_detected = np.any(left_danger_line == 1)

        # Vertical mid lane check (optional turning offset)
        x_vertical_line = int(ideal_lane(y_vertical_line, W, H))
        lane_positions_y = np.where(seg_mask[:, x_vertical_line] == 3)[0]
        if len(lane_positions_y) > 0:
            avg_y = np.mean(lane_positions_y)
            vertical_factor = 0.5 if avg_y < H * 0.5 else 1.0
        else:
            avg_y = None
            vertical_factor = 1.0

        shifted_center = W * 0.25
        lane_std = np.std([x for x in mid_lane_xs if x is not None])
        offset = 50 * vertical_factor if lane_std <= 30 else 0

        x_target = x_mid_weighted + offset
        error = (shifted_center - x_target) / (W / 2)

        # Reaction logic
        if right_danger_detected:
            steering = +1.0  # steer hard LEFT to avoid right danger
            forward_speed = 0.1
        elif left_danger_detected:
            steering = -1.0  # steer hard RIGHT to avoid left danger
            forward_speed = 0.1
        else:
            steering = np.clip(self.pid.update(error), -1.0, 1.0)
            if abs(error) > 0.6:
                forward_speed = 0.0
            elif abs(error) > 0.3:
                forward_speed = 0.1
            else:
                forward_speed = 0.44

        self.last_error = error
        self.prev_steering = steering
        self.prev_forward_speed = forward_speed

        debug_info = {
            "scan_lines": scan_lines,
            "mid_lane_xs": mid_lane_xs,
            "right_lane_xs": None,
            "fallback": False,
            "right_danger": right_danger_detected,
            "left_danger": left_danger_detected,
            "x_vertical_line": x_vertical_line,
            "avg_y": avg_y,
            "offset": offset,
            "x_target": int(x_target),
            "steering": steering,
            "forward_speed": forward_speed,
            "info": "Normal driving",
        }

        return [forward_speed, steering], error, debug_info









def colorize_segmentation(seg_mask):
    colors = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
    }
    h, w = seg_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        color_mask[seg_mask == class_id] = color
    return color_mask


def draw_ideal_lane_line(image, W, H, color=(0, 255, 0), thickness=2):
    # Sample points along the vertical axis in the region of interest
    ys = np.linspace(int(H * 0.35), int(H * 0.95), 100).astype(int)

    # Compute x for each y using the ideal_lane function
    xs = [int(ideal_lane(y, W, H)) for y in ys]

    # Pair points as line segments and draw on image
    for i in range(len(ys) - 1):
        pt1 = (xs[i], ys[i])
        pt2 = (xs[i + 1], ys[i + 1])
        cv2.line(image, pt1, pt2, color, thickness)

    return image

def main():
    env = gym.make("Duckietown-udem1-v0", map_name="udem1.yaml", camera_width=640, camera_height=640)
    obs = env.reset()

    model = PerceptionModule("slow_color.pt", "fine_tune_myset.pth")
    follower = LaneFollower()

    done = False
    while not done:
        bboxes, seg_mask, _ = model.predict(obs)
        seg_vis = colorize_segmentation(seg_mask)
        seg_vis_resized = cv2.resize(seg_vis, (640, 640), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Perception View", seg_vis_resized)

        action, last_error, debug_info = follower.lane_following_control(seg_mask)

        obs, reward, done, info = env.step(action)

        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        H, W, _ = frame.shape

        if debug_info["right_danger"]:
            cv2.putText(frame, "AVOIDING RIGHT LANE!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.rectangle(frame, (int(W * 0.75), 0), (W, H), (0, 0, 255), 2)

        # Draw horizontal scan lines
        for y in debug_info["scan_lines"]:
            cv2.line(frame, (0, y), (W, y), (255, 255, 0), 1)

        # Draw ideal lane line instead of vertical line
        frame = draw_ideal_lane_line(frame, W, H, color=(0, 255, 255), thickness=2)

        # Draw mid lane points on each scan line
        for i, x in enumerate(debug_info["mid_lane_xs"]):
            if x is not None:
                y = debug_info["scan_lines"][i]
                cv2.circle(frame, (int(x), y), 5, (0, 0, 255), -1)

        # Draw weighted x_target on the middle scan line
        middle_y = debug_info["scan_lines"][1]
        cv2.circle(frame, (debug_info["x_target"], middle_y), 7, (0, 255, 0), -1)

        cv2.putText(frame, f"Err: {last_error:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Duckietown Test", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
