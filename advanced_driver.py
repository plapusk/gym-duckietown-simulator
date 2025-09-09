import gym
import numpy as np
import cv2
from gym_duckietown.envs import DuckietownEnv
from wrapper import PerceptionModule  # Assumed available
from wrapper import PerceptionModuleSCNN
from typing import List, Tuple, Dict, Any

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

def get_lane_objects(
    seg_mask: np.ndarray,
    lane_class: int,
    min_size: int = 50
) -> List[Dict[str, Any]]:
    binary_mask = (seg_mask == lane_class).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary_mask)

    lane_objects = []
    for label in range(1, num_labels):
        coords = np.column_stack(np.where(labels == label))
        size = len(coords)
        if size < min_size:
            continue
        centroid = coords.mean(axis=0)
        lane_objects.append({
            'label': label,
            'coords': coords,
            'centroid': centroid,
            'size': size
        })
    return lane_objects

def ideal_lane(y: float, W: int, H: int) -> float:
    # Define start and end points as percentages
    x1, y1 = W * 0.05, H * 0.95  # Bottom-leftish
    x2, y2 = W * 0.40, H * 0.35  # Horizon cutoff

    slope = (x2 - x1) / (y2 - y1)
    intercept = x1 - slope * y1

    return slope * y + intercept

def ideal_lane(y: float, W: int, H: int, symmetric: bool = False) -> float:
    if not symmetric:
        # Default: left-to-center
        x1, y1 = W * 0.05, H * 0.95
        x2, y2 = W * 0.40, H * 0.35
    else:
        # Symmetrical: right-to-center
        x1, y1 = W * 0.95, H * 0.95
        x2, y2 = W * 0.60, H * 0.35

    slope = (x2 - x1) / (y2 - y1)
    intercept = x1 - slope * y1

    return slope * y + intercept

class LaneFollower:
    def __init__(self, kp=1.5, ki=0, kd=0.3):
        self.pid = PIDController(kp, ki, kd)
        self.last_error = 0
        self.prev_steering = 0.0
        self.prev_forward_speed = 0.0

    def calculate_steering_from_lane_points(self, x_points, x_ideal, width, weights=None):
        valid_points = [(x, w if weights else 1.0) for x, w in zip(x_points, weights or []) if x is not None]
        if not valid_points:
            return self.prev_steering, self.last_error, None

        positions, weights = zip(*valid_points)
        x_target = np.average(positions, weights=weights)
        error = (x_ideal - x_target) / (width / 2)
        steering = np.clip(self.pid.update(error), -1.0, 1.0)
        return steering, error, int(x_target)

    def lane_following_control(self, seg_mask):
        H, W = seg_mask.shape
        scan_lines = np.linspace(int(H * 0.4), int(H * 1.0) - 1, 20).astype(int).tolist()
        weighted_weights = np.linspace(1.0, 0.05, 20)
        weighted_weights /= weighted_weights.sum()
        weighted_weights = weighted_weights.tolist()

        lane_objects = get_lane_objects(seg_mask, lane_class=2, min_size=50)
        lane_objects = classify_lane_objects_by_boundary(lane_objects, seg_mask.shape)

        largest_left = max((obj for obj in lane_objects if obj['position'] == 'left'), key=lambda x: x['size'], default=None)
        largest_right = max((obj for obj in lane_objects if obj['position'] == 'right'), key=lambda x: x['size'], default=None)

        left_mask = np.zeros_like(seg_mask, dtype=np.uint8)
        right_mask = np.zeros_like(seg_mask, dtype=np.uint8)

        if largest_left:
            coords = largest_left['coords']
            left_mask[coords[:, 0], coords[:, 1]] = 1

        if largest_right:
            coords = largest_right['coords']
            right_mask[coords[:, 0], coords[:, 1]] = 1

        mid_lane_xs, left_lane_xs, right_lane_xs = [], [], []
        for y in scan_lines:
            mid = np.where(seg_mask[y, :] == 3)[0]
            mid_lane_xs.append(np.mean(mid) if len(mid) > 0 else None)

            left = np.where(left_mask[y, :] == 1)[0] if largest_left else []
            left_lane_xs.append(max(left) if len(left) > 0 else None)

            right = np.where(right_mask[y, :] == 1)[0] if largest_right else []
            right_lane_xs.append(min(right) if len(right) > 0 else None)

        bottom_10_mid = mid_lane_xs[-10:]
        bottom_10_right = right_lane_xs[-10:]
        bottom_10_left = left_lane_xs[-10:]

        mid_valid_count = sum(x is not None for x in bottom_10_mid)
        right_valid_count = sum(x is not None for x in bottom_10_right)
        left_valid_count = sum(x is not None for x in bottom_10_left)

        # First fallback policy:
        fallback = (mid_valid_count < 2) and (right_valid_count > mid_valid_count)

        x_ideal = int(ideal_lane(int(H * 0.5), W, H))

        if not fallback:
            steering, error, x_target = self.calculate_steering_from_lane_points(
                mid_lane_xs, x_ideal=W * 0.25, width=W, weights=weighted_weights
            )
        else:
            x_ideal_flipped = W * 0.75
            steering, error, x_target = self.calculate_steering_from_lane_points(
            right_lane_xs, x_ideal=x_ideal_flipped, width=W, weights=weighted_weights
        )

        # Second fallback policy:
        # If both mid-lane and right-lane points are sparse (<2)
        # and left lane exists with valid points,
        # then steer gently to the right.
        if (mid_valid_count < 2) and (right_valid_count < 2) and (left_valid_count > 0):
            # Gentle right turn, e.g. steering around 0.3 (tweak as needed)
            steering = 0.3
            error = None
            x_target = None
            fallback = True  # Optionally mark fallback active here

        # Continue with danger zone detection and speed control (unchanged)
        danger_zone_y = int(H * 0.85)
        right_danger = np.any(seg_mask[danger_zone_y, int(W * 0.75):] == 2)
        left_danger = np.any(seg_mask[danger_zone_y, :int(W * 0.25)] == 1)

        if right_danger:
            forward_speed = 0.1
            steering = +1.0
        elif left_danger:
            forward_speed = 0.1
            steering = -1.0
        else:
            if error is not None:
                if abs(error) > 0.6:
                    forward_speed = 0.0
                elif abs(error) > 0.3:
                    forward_speed = 0.1
                else:
                    forward_speed = 0.44
            else:
                # If no error available (second fallback), keep slow speed
                forward_speed = 0.1

        self.last_error = error if error is not None else self.last_error
        self.prev_steering = steering
        self.prev_forward_speed = forward_speed

        debug_info = {
            "scan_lines": scan_lines,
            "mid_lane_xs": mid_lane_xs,
            "left_lane_xs": left_lane_xs,
            "right_lane_xs": right_lane_xs,
            "fallback": fallback,
            "right_danger": right_danger,
            "left_danger": left_danger,
            "x_vertical_line": x_ideal,
            "x_target": x_target,
            "steering": steering,
            "forward_speed": forward_speed,
            "info": "Fallback - left lane fallback" if (mid_valid_count < 2 and right_valid_count < 2 and left_valid_count > 0)
                    else ("Fallback - right lane fallback" if fallback else "Normal driving"),
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


def draw_ideal_lane_line(image, W, H, color=(0, 255, 0), thickness=2, symmetric=False):
    # Sample points along the vertical axis in the region of interest
    ys = np.linspace(int(H * 0.35), int(H * 0.95), 100).astype(int)

    # Compute x for each y using the ideal_lane function
    xs = [int(ideal_lane(y, W, H, symmetric=symmetric)) for y in ys]

    # Pair points as line segments and draw on image
    for i in range(len(ys) - 1):
        pt1 = (xs[i], ys[i])
        pt2 = (xs[i + 1], ys[i + 1])
        cv2.line(image, pt1, pt2, color, thickness)

    return image




def visualize_lane_masks(image: np.ndarray, lane_objects: List[Dict[str, Any]], alpha=0.5, color=(0, 255, 0)) -> np.ndarray:
    overlay = image.copy()

    for obj in lane_objects:
        # Create a blank mask same size as image
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # coords are (y,x) pairs
        coords = obj['coords']
        mask[coords[:,0], coords[:,1]] = 255  # Mark object pixels

        # Create colored mask for this object
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[:, :] = color

        # Blend colored mask on the overlay using the binary mask
        overlay = np.where(mask[:, :, None] == 255,
                           cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0),
                           overlay)

    return overlay

def visualize_lane_classification(frame, lane_objects, alpha=0.6):
    overlay = frame.copy()
    for obj in lane_objects:
        coords = obj['coords']
        position = obj.get('position', 'none')
        color = (0, 255, 0)  # default green
        
        if position == 'left':
            color = (255, 0, 0)    # blue for left
        elif position == 'right':
            color = (0, 0, 255)    # red for right
        elif position == 'lower':
            color = (0, 255, 255)  # yellow for lower
        
        # Fill the mask area with color
        ys, xs = coords[:, 0], coords[:, 1]
        overlay[ys, xs] = color
        
        # Put text at centroid
        centroid = obj['centroid'].astype(int)
        cv2.putText(overlay, position, (centroid[1], centroid[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Blend overlay with original frame
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def classify_lane_objects_by_boundary(
    lane_objects: List[Dict[str, Any]],
    img_shape: tuple,
    margin: int = 3
) -> List[Dict[str, Any]]:
    H, W = img_shape[:2]

    for obj in lane_objects:
        coords = obj['coords']
        ys = coords[:, 0]
        xs = coords[:, 1]

        touches_bottom = np.any(ys >= H - margin)
        touches_left = np.any(xs <= margin)
        touches_right = np.any(xs >= W - margin)

        if touches_left and touches_right:
            # Check if the object stays fully above 50% of the screen
            touches_below_half = np.any(ys >= H * 0.5)
            if not touches_below_half:
                obj['position'] = 'upper'
                continue

        if touches_bottom:
            # Fit a line x = m*y + b to check leaning direction
            if len(xs) > 1:
                A = np.vstack([ys, np.ones(len(ys))]).T
                m, b = np.linalg.lstsq(A, xs, rcond=None)[0]
                if m < -0.1:
                    obj['position'] = 'left'
                elif m > 0.1:
                    obj['position'] = 'right'
                else:
                    obj['position'] = 'right' if touches_right else 'left'
            else:
                obj['position'] = 'unknown'

        elif touches_left and not touches_right:
            obj['position'] = 'left'
        elif touches_right and not touches_left:
            obj['position'] = 'right'
        elif touches_left and touches_right:
            # Compare contact depth on both sides
            left_ys = ys[xs <= margin]
            right_ys = ys[xs >= W - margin]
            left_max = np.max(left_ys) if len(left_ys) > 0 else -1
            right_max = np.max(right_ys) if len(right_ys) > 0 else -1

            obj['position'] = 'right' if right_max > left_max else 'left'
        else:
            obj['position'] = 'unknown'

    return lane_objects

def main():
    env = gym.make("Duckietown-udem1-v0", map_name="maps/udem1.yaml", camera_width=640, camera_height=640)
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

        # Draw ideal lane lines on the frame
        draw_ideal_lane_line(frame, W, H, color=(0, 255, 0), thickness=2)                     # Normal
        draw_ideal_lane_line(frame, W, H, color=(0, 0, 255), thickness=2, symmetric=True)     # Symmetric fallback

        if debug_info["right_danger"]:
            cv2.putText(frame, "AVOIDING RIGHT LANE!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.rectangle(frame, (int(W * 0.75), 0), (W, H), (0, 0, 255), 2)

        # Draw horizontal scan lines
        for y in debug_info["scan_lines"]:
            cv2.line(frame, (0, y), (W, y), (255, 255, 0), 1)

        # Draw mid lane points (red)
        for i, x in enumerate(debug_info["mid_lane_xs"]):
            if x is not None:
                y = debug_info["scan_lines"][i]
                cv2.circle(frame, (int(x), y), 5, (0, 0, 255), -1)

        # Draw left lane points (blue)
        for i, x in enumerate(debug_info.get("left_lane_xs", [])):
            if x is not None:
                y = debug_info["scan_lines"][i]
                cv2.circle(frame, (int(x), y), 5, (255, 0, 0), -1)

        # Draw right lane points (green)
        for i, x in enumerate(debug_info.get("right_lane_xs", [])):
            if x is not None:
                y = debug_info["scan_lines"][i]
                cv2.circle(frame, (int(x), y), 5, (0, 255, 0), -1)

        # Draw x_target on middle scan line
        middle_index = len(debug_info["scan_lines"]) // 2
        middle_y = debug_info["scan_lines"][middle_index]
        cv2.circle(frame, (debug_info["x_target"], middle_y), 7, (0, 255, 0), -1)

        # Overlay classified lane object masks
        lane_objects = get_lane_objects(seg_mask, 2, min_size=50)
        lane_objects = classify_lane_objects_by_boundary(lane_objects, frame.shape)
        frame_with_lane_masks = visualize_lane_classification(frame, lane_objects, alpha=0.6)
        cv2.imshow("Right Lane Objects Mask Overlay", frame_with_lane_masks)

        # Show error
        cv2.putText(frame, f"Err: {last_error:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Duckietown Test", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
