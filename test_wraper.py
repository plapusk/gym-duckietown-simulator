import gym
import cv2
import numpy as np
from wrapper import PerceptionModule

def visualize_result_live(frame_bgr, bboxes, seg_mask, duration):
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

    color_map = np.array([
        [0, 0, 0],       # fundal
        [0, 255, 255],   # clasa 1
        [0, 255, 0],     # clasa 2
        [255, 0, 255]    # clasa 3
    ])
    seg_overlay = color_map[seg_mask]
    seg_overlay = cv2.addWeighted(frame_bgr, 0.6, seg_overlay.astype(np.uint8), 0.4, 0)

    cv2.putText(seg_overlay, f"Eval time: {duration:.3f} sec", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return seg_overlay

def main():
    env = gym.make("Duckietown-udem1-v0")
    obs = env.reset()

    model = PerceptionModule("slow_color.pt", "fine_tune_myset.pth")

    window_name = "Duckietown Perception"
    cv2.namedWindow(window_name)

    done = False
    while not done:
        # obs vine RGB numpy array (height, width, 3)
        bboxes, seg_mask, duration = model.predict(obs)

        # Convertim la BGR pentru OpenCV
        frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        vis_img = visualize_result_live(frame_bgr, bboxes, seg_mask, duration)

        cv2.imshow(window_name, vis_img)

        # Pasare random in simulator (sau 0,0 pt pauza)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
