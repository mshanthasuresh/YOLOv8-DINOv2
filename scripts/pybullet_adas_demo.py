from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pybullet as bullet
import pybullet_data
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def build_simulation(native_gui: bool) -> tuple[int, int]:
    client = bullet.connect(bullet.GUI if native_gui else bullet.DIRECT)
    bullet.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    bullet.setGravity(0, 0, -9.81, physicsClientId=client)
    bullet.loadURDF("plane.urdf", physicsClientId=client)
    vehicle = bullet.loadURDF(
        "racecar/racecar.urdf", basePosition=[0, 0, 0.2], useFixedBase=False,
        physicsClientId=client,
    )
    for position in ([0, 12, 0.5], [-2, 20, 0.5], [2, 28, 0.5]):
        bullet.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=bullet.createCollisionShape(
                bullet.GEOM_BOX, halfExtents=[0.8, 0.8, 0.5], physicsClientId=client
            ),
            baseVisualShapeIndex=bullet.createVisualShape(
                bullet.GEOM_BOX, halfExtents=[0.8, 0.8, 0.5],
                rgbaColor=[0.8, 0.1, 0.1, 1], physicsClientId=client
            ),
            basePosition=position,
            physicsClientId=client,
        )
    return client, vehicle


def apply_vehicle_control(vehicle: int, throttle: float, steering: float) -> None:
    position, orientation = bullet.getBasePositionAndOrientation(vehicle)
    rotation = np.asarray(bullet.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    forward = rotation[:, 0]
    lateral = rotation[:, 1]
    bullet.applyExternalForce(
        vehicle, -1, (forward * (180 * throttle)).tolist(), position,
        bullet.WORLD_FRAME,
    )
    bullet.applyExternalTorque(
        vehicle, -1, (lateral * (35 * steering)).tolist(), bullet.WORLD_FRAME
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 in a dynamic PyBullet ADAS replay.")
    parser.add_argument("--input", default=str(ROOT / "data" / "coco8" / "images" / "val" / "000000000036.jpg"))
    parser.add_argument("--weights", default=str(ROOT / "runs" / "baseline_yolov8" / "weights" / "best.pt"))
    parser.add_argument("--output", default=str(ROOT / "runs" / "pybullet_adas" / "adas_replay.mp4"))
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--device", default="0")
    parser.add_argument("--gui", action="store_true", help="Show the live OpenCV ADAS window.")
    parser.add_argument(
        "--native-pybullet-gui",
        action="store_true",
        help="Use PyBullet's native renderer instead of the WSLg-compatible OpenCV window.",
    )
    args = parser.parse_args()

    input_frame = cv2.imread(args.input)
    if input_frame is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")
    model = YOLO(args.weights if Path(args.weights).exists() else "yolov8n.pt")
    client, vehicle = build_simulation(args.native_pybullet_gui)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = input_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height)
    )
    telemetry_path = output_path.with_suffix(".csv")

    with telemetry_path.open("w", newline="", encoding="utf-8") as telemetry_file:
        telemetry = csv.writer(telemetry_file)
        telemetry.writerow(["step", "x", "y", "speed", "target_class", "target_confidence", "steering"])
        display_enabled = args.gui
        for step in range(args.steps):
            result = model.predict(
                input_frame, device=args.device, imgsz=640, conf=0.25, verbose=False
            )[0]
            annotated = result.plot()
            target_class, target_confidence, steering = "none", 0.0, 0.0
            if result.boxes is not None and len(result.boxes):
                confidences = result.boxes.conf.cpu().numpy()
                best_index = int(np.argmax(confidences))
                box = result.boxes.xyxy[best_index].cpu().numpy()
                center_x = float((box[0] + box[2]) / 2)
                steering = float(np.clip((center_x - width / 2) / (width / 2), -1, 1))
                target_class = str(model.names[int(result.boxes.cls[best_index].item())])
                target_confidence = float(confidences[best_index])
            apply_vehicle_control(vehicle, throttle=0.8, steering=-steering)
            bullet.stepSimulation(physicsClientId=client)
            position, _ = bullet.getBasePositionAndOrientation(vehicle)
            velocity, _ = bullet.getBaseVelocity(vehicle)
            speed = float(np.linalg.norm(velocity))
            telemetry.writerow([step, position[0], position[1], speed, target_class, target_confidence, steering])
            cv2.putText(
                annotated, f"PyBullet ADAS | x={position[0]:.2f} y={position[1]:.2f} speed={speed:.2f}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
            )
            cv2.putText(
                annotated, f"steering={steering:+.2f} target={target_class}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
            )
            writer.write(annotated)
            if display_enabled:
                try:
                    cv2.imshow("YOLOv8 PyBullet ADAS", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    print("OpenCV GUI backend unavailable; continuing with video output.")
                    display_enabled = False
                time.sleep(1 / 60)
    writer.release()
    if display_enabled:
        cv2.destroyAllWindows()
    bullet.disconnect(client)
    print(f"Saved ADAS replay: {output_path}")
    print(f"Saved vehicle telemetry: {telemetry_path}")


if __name__ == "__main__":
    main()