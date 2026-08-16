from collections import defaultdict, deque
import numpy as np
import supervision as sv

import config


class MultiObjectTracker:
    """
    ByteTrack-based multi-object tracker for vehicles and pedestrians.
    Maintains persistent tracking IDs and trajectory histories.
    """

    def __init__(self, history_len: int = config.TRAJECTORY_HISTORY):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.tracker = sv.ByteTrack()
        self.history_len = history_len

        # Map track_id -> deque of (center_x, center_y)
        self.trajectories = defaultdict(lambda: deque(maxlen=self.history_len))

        # Map track_id -> total frames tracked
        self.frame_counts = defaultdict(int)

        # Set of active unique track IDs seen in the current session
        self.all_seen_tracks = set()
        self.all_seen_vehicles = set()
        self.all_seen_persons = set()

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the tracker with current frame detections and records trajectories.
        Returns tracked detections with tracker_id populated.
        """
        if len(detections) == 0:
            return detections

        # Perform tracking update
        tracked_detections = self.tracker.update_with_detections(detections)

        # Update trajectory histories for tracked objects
        for i in range(len(tracked_detections)):
            track_id = tracked_detections.tracker_id[i]
            if track_id is None:
                continue

            track_id = int(track_id)
            self.all_seen_tracks.add(track_id)
            self.frame_counts[track_id] += 1

            cls_id = int(tracked_detections.class_id[i])
            if cls_id in config.VEHICLE_CLASS_IDS:
                self.all_seen_vehicles.add(track_id)
            elif cls_id == config.CLASS_PERSON:
                self.all_seen_persons.add(track_id)

            box = tracked_detections.xyxy[i]
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            self.trajectories[track_id].append((cx, cy))

        return tracked_detections

    def get_trajectory(self, track_id: int):
        """Returns trajectory list of (x, y) coordinates for a given track."""
        return list(self.trajectories[track_id])

    def get_last_position(self, track_id: int):
        """Returns the most recent (x, y) center for a given track."""
        hist = self.trajectories[track_id]
        return hist[-1] if len(hist) > 0 else None

    def get_stats(self) -> dict:
        """Returns aggregate tracking counts."""
        return {
            "total_tracks": len(self.all_seen_tracks),
            "total_vehicles": len(self.all_seen_vehicles),
            "total_persons": len(self.all_seen_persons)
        }
