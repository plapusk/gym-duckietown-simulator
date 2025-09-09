import gym
import numpy as np
import cv2
from gym_duckietown.envs import DuckietownEnv
from wrapper import PerceptionModule  # Assumed available
from typing import List, Tuple, Dict, Any, Optional
from scipy.interpolate import interp1d

class PIDController:
    def __init__(self, kp: float, ki: float = 0, kd: float = 0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error: float, dt: float = 1.0) -> float:
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


class LaneFollower:
    def __init__(self, kp: float = 1.5, ki: float = 0, kd: float = 0.3):
        self.pid = PIDController(kp, ki, kd)
        self.last_error = 0.0
        self.prev_steering = 0.0
        self.prev_forward_speed = 0.0


    def generate_weights(self, scan_lines: List[int], H: int) -> List[float]:
        normalized_positions = 1.0 - (np.array(scan_lines) / H)
        weights = np.exp(4 * normalized_positions)
        weights /= weights.sum()
        return weights.tolist()

    def get_extended_lane_curve(self, poly_func: np.poly1d, scan_lines: List[int], H: int, extend_pixels: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        y_min, y_max = min(scan_lines), max(scan_lines)
        y_extended = np.linspace(y_min - extend_pixels, y_max + extend_pixels, 100)
        x_extended = poly_func(y_extended)
        return x_extended, y_extended

    def extract_lane_positions(self, seg_mask: np.ndarray, scan_lines: List[int], lane_id: int) -> List[Any]:
        return [
            np.mean(np.where(seg_mask[y, :] == lane_id)[0]) if len(np.where(seg_mask[y, :] == lane_id)[0]) > 0 else None
            for y in scan_lines
        ]

    def interpolate_missing_points(self, lane_xs: List[Any], scan_lines: List[int], max_interp_points=3) -> List[float]:
        valid_mask = np.array([x is not None for x in lane_xs], dtype=bool)
        if valid_mask.sum() < 2:
            return lane_xs  # Not enough points to interpolate

        ys_valid = np.array([y for y, valid in zip(scan_lines, valid_mask) if valid])
        xs_valid = np.array([x for x in lane_xs if x is not None])

        interp_func = interp1d(ys_valid, xs_valid, kind='linear', fill_value="extrapolate")

        interpolated = lane_xs.copy()

        # Identify missing gaps as start-end indices of consecutive None
        n = len(lane_xs)
        i = 0
        while i < n:
            if lane_xs[i] is None:
                start = i
                while i < n and lane_xs[i] is None:
                    i += 1
                end = i - 1  # end of gap

                gap_size = end - start + 1

                # Interpolate points close to left edge of gap
                for idx in range(start, min(start + max_interp_points, end + 1)):
                    y = scan_lines[idx]
                    interpolated[idx] = float(interp_func(y))

                # Interpolate points close to right edge of gap
                for idx in range(max(end - max_interp_points + 1, start), end + 1):
                    y = scan_lines[idx]
                    interpolated[idx] = float(interp_func(y))

                # The points in the middle (if any) remain None to avoid large error
            else:
                i += 1

        return interpolated


    def adjust_weights_for_missing_points(self, base_weights: np.ndarray, lane_xs: List[Any]) -> np.ndarray:
        valid_mask = np.array([x is not None for x in lane_xs], dtype=float)
        adjusted_weights = base_weights.copy()
        missing_indices = np.where(valid_mask == 0)[0]
        for miss_idx in missing_indices:
            adjusted_weights[miss_idx + 1 :] *= 1.2
        adjusted_weights *= valid_mask
        if adjusted_weights.sum() > 0:
            adjusted_weights /= adjusted_weights.sum()
        else:
            adjusted_weights = base_weights
        return adjusted_weights

    def compute_polyfit_error_and_curve(self, lane_xs: List[float], scan_lines: List[int], W: int, H: int):
        poly_coeffs = np.polyfit(scan_lines, lane_xs, deg=2)
        poly_func = np.poly1d(poly_coeffs)
        predicted_xs = poly_func(np.array(scan_lines))
        ideal_xs = np.array([ideal_lane(y, W, H) for y in scan_lines])

        bottom_y = H * 0.95
        predicted_x_bottom = poly_func(bottom_y)
        ideal_x_bottom = ideal_lane(bottom_y, W, H)
        bottom_error = (ideal_x_bottom - predicted_x_bottom) / (W / 2)

        errors = ideal_xs - predicted_xs
        base_weights = self.generate_weights(scan_lines, H)
        position_error = np.average(errors, weights=base_weights) / (W / 2)

        look_ahead_y = H * 0.90
        predicted_slope = np.polyder(poly_func)(look_ahead_y)
        x_start, x_end = int(W * 0.05), int(W * 0.40)
        y_start, y_end = int(H * 0.95), int(H * 0.40)
        ideal_slope = (x_end - x_start) / (y_end - y_start + 1e-5)

        v1 = np.array([1.0, predicted_slope])
        v2 = np.array([1.0, ideal_slope])
        angle_error_rad = np.arctan2(v1[1]*v2[0] - v1[0]*v2[1], v1 @ v2)
        slope_error = angle_error_rad / np.pi

        combined_error = bottom_error if abs(bottom_error) > 0.3 else 0.7 * position_error + 0.3 * slope_error
        x_extended, y_extended = self.get_extended_lane_curve(poly_func, scan_lines, H)
        return combined_error, poly_coeffs, x_extended, y_extended, bottom_error, predicted_slope, ideal_slope

    def compute_fallback_error(self, lane_xs: List[float], W: int):
        weights = np.ones(len(lane_xs)) / len(lane_xs)
        x_mid_weighted = np.average(lane_xs, weights=weights)
        shifted_center = W * 0.9
        combined_error = (shifted_center - x_mid_weighted) / (W / 2)
        return combined_error

    

    def detect_danger_zones(self, seg_mask: np.ndarray, H: int, W: int) -> Tuple[bool, bool]:
        danger_zone_y = int(H * 0.85)
        right_start = int(W * 0.75)
        left_end = int(W * 0.25)

        right_danger = np.any(seg_mask[danger_zone_y, right_start:] == 2)
        left_danger = np.any(seg_mask[danger_zone_y, :left_end] == 1)

        return right_danger, left_danger

    def lane_following_control(self, seg_mask: np.ndarray) -> Tuple[List[float], float, Dict[str, Any]]:
        H, W = seg_mask.shape
        scan_lines = np.linspace(H // 3, H - 1, 20).astype(int).tolist()
        base_weights = np.array(self.generate_weights(scan_lines, H))

        # Extract lane positions
        mid_lane_xs = self.extract_lane_positions(seg_mask, scan_lines, lane_id=3)
        mid_lane_xs = self.interpolate_missing_points(mid_lane_xs, scan_lines)

        right_lane_xs = self.extract_lane_positions(seg_mask, scan_lines, lane_id=2)

        num_valid_mid_lane = sum(x is not None for x in mid_lane_xs)
        num_valid_right_lane = sum(x is not None for x in right_lane_xs)
        fallback_to_right_lane = (num_valid_mid_lane < 2) and (num_valid_right_lane >= 2)

        if fallback_to_right_lane:
            lane_xs = right_lane_xs
            weights = base_weights
        else:
            lane_xs = mid_lane_xs
            weights = self.adjust_weights_for_missing_points(base_weights, lane_xs)

        valid_positions = [(x, w, y) for x, w, y in zip(lane_xs, weights, scan_lines) if x is not None]
        if len(valid_positions) < 2:
            # Not enough data points, hold previous control
            return [self.prev_forward_speed, self.prev_steering], self.last_error, {
                "info": "No valid lane points, holding previous action",
                "fallback": fallback_to_right_lane,
                "mid_lane_xs": mid_lane_xs,
                "right_lane_xs": right_lane_xs,
                "scan_lines": scan_lines,
                "right_danger": False,
                "left_danger": False,
                "x_vertical_line": int(W * 0.3),
                "x_target": None,
                "steering": self.prev_steering,
                "forward_speed": self.prev_forward_speed,
            }

        positions, weights, ys = zip(*valid_positions)

        if not fallback_to_right_lane:
            (combined_error, poly_coeffs, x_extended, y_extended, bottom_error,
             predicted_slope, ideal_slope) = self.compute_polyfit_error_and_curve(list(positions), list(ys), W, H)
        else:
            combined_error = self.compute_fallback_error(list(positions), W)
            poly_coeffs = None
            x_extended, y_extended = None, None
            bottom_error = None
            predicted_slope = None
            ideal_slope = None


        right_danger_detected, left_danger_detected = self.detect_danger_zones(seg_mask, H, W)

        # Decision logic for steering and speed
        if right_danger_detected:
            steering = +1.0
            forward_speed = 0.1
        elif left_danger_detected:
            steering = -1.0
            forward_speed = 0.1
        else:
            base_steering = self.pid.update(combined_error)
            steering = np.clip(base_steering, -1.0, 1.0)

            if not fallback_to_right_lane:
                if bottom_error is not None and abs(bottom_error) > 0.5:
                    forward_speed = 0.0  # stop if bottom error too big (very off)
                else:
                    base_speed = 0.44

                    if predicted_slope is not None and ideal_slope is not None:
                        slope_diff = predicted_slope - ideal_slope

                        # If slope_diff < 0 (lane curves left) → higher speed, else slower
                        slope_speed_factor = 1.1 if slope_diff < 0 else 0.7

                        # Slow down more if bottom error is moderately high (between 0.2 and 0.5)
                        if bottom_error is not None:
                            if abs(bottom_error) > 0.2:
                                bottom_speed_factor = max(0.0, 1.0 - (abs(bottom_error) - 0.2) * 2.5)  # linearly decrease speed from 1 to 0 as error goes from 0.2 to 0.6
                            else:
                                bottom_speed_factor = 1.0
                        else:
                            bottom_speed_factor = 1.0

                        forward_speed = base_speed * slope_speed_factor * bottom_speed_factor
                        forward_speed = np.clip(forward_speed, 0.0, 1.0)

                    else:
                        # If slope info missing, just depend on bottom error
                        if bottom_error is not None and abs(bottom_error) > 0.2:
                            forward_speed = 0.22
                        else:
                            forward_speed = base_speed

            else:
                if abs(combined_error) > 0.6:
                    forward_speed = 0.1
                elif abs(combined_error) > 0.3:
                    forward_speed = 0.22
                else:
                    forward_speed = 0.35

        # Save last outputs
        self.last_error = combined_error
        self.prev_steering = steering
        self.prev_forward_speed = forward_speed

        debug_info = {
            "fallback": fallback_to_right_lane,
            "mid_lane_xs": mid_lane_xs,
            "right_lane_xs": right_lane_xs,
            "scan_lines": scan_lines,
            "right_danger": right_danger_detected,
            "left_danger": left_danger_detected,
            "poly_coeffs": poly_coeffs.tolist() if poly_coeffs is not None else None,
            "combined_error": combined_error,
            "steering": steering,
            "forward_speed": forward_speed,
            "info": "Polynomial predictive lane fitting" if not fallback_to_right_lane else "Fallback to right lane weighted avg",
            "extended_lane_curve": (x_extended, y_extended)
        }

        return [forward_speed, steering], combined_error, debug_info





    

def draw_ideal_lane(frame, W, H, color=(0, 255, 0), thickness=2):
    # Define start and end points in percentages of image size
    y_start = int(H * 0.95)   # Bottom of the screen
    y_end = int(H * 0.35)     # Horizon cutoff

    x_start = int(W * 0.05)   # 5% from left
    x_end = int(W * 0.40)     # 40% from left

    # Draw the line from bottom-leftish toward center
    pt1 = (x_start, y_start)
    pt2 = (x_end, y_end)
    cv2.line(frame, pt1, pt2, color, thickness)

    return frame




def colorize_segmentation(seg_mask: np.ndarray) -> np.ndarray:
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


def visualize_side_lane_components(
    seg_mask: np.ndarray, target_class: int = 2, min_size: int = 50
) -> np.ndarray:
    side_lane_mask = (seg_mask == target_class).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_mask = cv2.morphologyEx(side_lane_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels = cv2.connectedComponents(closed_mask)
    h, w = seg_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed=target_class + 42)

    for label in range(1, num_labels):
        component_size = np.sum(labels == label)
        if component_size < min_size:
            continue
        color = tuple(rng.integers(50, 255, size=3).tolist())
        vis[labels == label] = color

    return vis


def main() -> None:
    env: DuckietownEnv = gym.make("Duckietown-udem1-v0", map_name="maps/udem1.yaml", camera_width=640, camera_height=640)
    obs = env.reset()

    model = PerceptionModule("slow_color.pt", "fine_tune_myset.pth")
    follower = LaneFollower()

    done = False
    while not done:
        bboxes, seg_mask, _ = model.predict(obs)

        # Segmentation visualizations
        side_lane_view = visualize_side_lane_components(seg_mask, target_class=2)
        seg_vis = colorize_segmentation(seg_mask)
        seg_vis_resized = cv2.resize(seg_vis, (640, 640), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Perception View", seg_vis_resized)
        cv2.imshow("Side Lane Components", side_lane_view)

        # Control logic
        action, last_error, debug_info = follower.lane_following_control(seg_mask)

        # Step environment
        obs, reward, done, info = env.step(action)

        # Convert observation to BGR and get size
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        H, W, _ = frame.shape

        # Draw ideal lane
        frame = draw_ideal_lane(frame, W, H)

        # Draw debug overlays
        if debug_info["right_danger"]:
            cv2.putText(frame, "AVOIDING RIGHT LANE!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.rectangle(frame, (int(W * 0.75), 0), (W, H), (0, 0, 255), 2)

        for y in debug_info["scan_lines"]:
            cv2.line(frame, (0, y), (W, y), (255, 255, 0), 1)

        for i, x in enumerate(debug_info["mid_lane_xs"]):
            if x is not None:
                y = debug_info["scan_lines"][i]
                cv2.circle(frame, (int(x), y), 5, (0, 0, 255), -1)

        # Display last error
        cv2.putText(frame, f"Err: {last_error:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show final output frame
        cv2.imshow("Duckietown Debug View", frame)

        # Logs
        print(f"Last Error: {last_error:.3f}")
        print(f"Mid Lane Xs: {[f'{x:.1f}' if x is not None else 'None' for x in debug_info['mid_lane_xs']]}")
        print(f"Right Lane Xs: {[f'{x:.1f}' if x is not None else 'None' for x in debug_info['right_lane_xs']]}")
        print(f"Right Danger: {debug_info['right_danger']}, Left Danger: {debug_info['left_danger']}")
        print("---")

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    env.close()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
