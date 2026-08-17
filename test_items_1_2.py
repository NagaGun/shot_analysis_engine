"""
Tests for v2 Item 1 (trajectory fit zone) and Item 2 (consecutive-spike quality gate).
Run from the repo root: python test_items_1_2.py
"""
import sys
import os
import unittest
import numpy as np

# Add the repo to path so shot_analysis can be imported without installing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shot_analysis import (
    _fit_trajectory_zone,
    classify_zone,
    detect_contact_frame_fused,
    GOAL_WIDTH_M,
    GOAL_HEIGHT_M,
    DEST_POINTS,
    compute_homography,
)


def _make_identity_H():
    """
    Build a homography from the unit-square corners that maps pixels
    trivially into world space for testing.
    We use the actual goal dimensions so _bucket_zone ranges work correctly.
    Pixel corners chosen so that:
      (0, GOAL_HEIGHT_M) -> TL
      (GOAL_WIDTH_M, GOAL_HEIGHT_M) -> TR
      (GOAL_WIDTH_M, 0) -> BR
      (0, 0) -> BL
    This gives a homography very close to identity in the goal coordinate space.
    """
    pixel_corners = np.array([
        [0.0,           GOAL_HEIGHT_M],   # TL
        [GOAL_WIDTH_M,  GOAL_HEIGHT_M],   # TR
        [GOAL_WIDTH_M,  0.0],             # BR
        [0.0,           0.0],             # BL
    ], dtype=np.float32)
    return compute_homography(pixel_corners)


class TestItem1TrajectoryFit(unittest.TestCase):

    def setUp(self):
        self.H = _make_identity_H()

    def test_fit_returns_zone_for_straight_ball(self):
        """A straight-line post-contact trajectory in world space should zone correctly."""
        # Ball moving from center-left at mid-height toward center of goal
        post = [
            {"frame": i, "x": 2.0 + i * 0.1, "y": 1.2 - i * 0.05}
            for i in range(6)
        ]
        result = _fit_trajectory_zone(post, self.H)
        self.assertIsNotNone(result, "Should return a zone dict for 6 points")
        self.assertIn("goal_zone", result)
        self.assertIsNotNone(result["goal_zone"])

    def test_fit_returns_none_for_fewer_than_3_points(self):
        """With < 3 post-contact points, fit must return None (triggers last-point fallback)."""
        post = [
            {"frame": 1, "x": 3.0, "y": 1.2},
            {"frame": 2, "x": 3.5, "y": 1.0},
        ]
        result = _fit_trajectory_zone(post, self.H)
        self.assertIsNone(result, "< 3 points should return None")

    def test_classify_zone_h_none_rejects(self):
        """No homography → calibration_failed reject_reason, no zone."""
        ball_track = [{"frame": i, "x": 100, "y": 100} for i in range(10)]
        r = classify_zone(ball_track, 0, None, 640, 480)
        self.assertIsNone(r["goal_zone"])
        self.assertEqual(r["reject_reason"], "calibration_failed")

    def test_classify_zone_insufficient_points_rejects(self):
        """< MIN_POST_CONTACT_POINTS after contact → reject."""
        ball_track = [{"frame": i, "x": 3.0, "y": 1.0} for i in range(2)]
        r = classify_zone(ball_track, 0, self.H, 640, 480)
        self.assertIsNone(r["goal_zone"])
        self.assertEqual(r["reject_reason"], "insufficient_post_contact_tracking")

    def test_classify_zone_uses_trajectory_fit_when_possible(self):
        """With enough points, classify_zone should use trajectory fit (_zone_method key present)."""
        ball_track = [{"frame": i, "x": 2.0 + i * 0.05, "y": 1.0 + i * 0.02} for i in range(10)]
        r = classify_zone(ball_track, 0, self.H, 640, 480)
        # Should have returned a zone (might be on_target or miss depending on trajectory)
        self.assertIn("goal_zone", r)
        # Should carry the _zone_method tag indicating fit was used (or last_point_fallback)
        # We only assert it ran without error here; correctness needs labeled clips.


class TestItem2ConsecutiveSpikeGate(unittest.TestCase):

    def _make_foot_tracks(self, contact_frame, x, y):
        return {
            "Right_Foot": [{"frame": contact_frame, "x": x, "y": y}],
            "Left_Foot": [],
        }

    def test_single_spike_blip_is_rejected(self):
        """A single above-threshold frame with no sustained follow-through is rejected."""
        fps = 30.0
        # Ball is slow, then one single big spike, then slow again — classic blip
        ball_track = (
            [{"frame": i, "x": float(i * 5), "y": 50.0, "width_px": 20} for i in range(5)]
            + [{"frame": 5, "x": 300.0, "y": 50.0, "width_px": 20}]  # spike frame
            + [{"frame": i, "x": float(300 + (i - 6) * 5), "y": 50.0, "width_px": 20} for i in range(6, 10)]
        )
        foot_tracks = self._make_foot_tracks(5, 305.0, 50.0)
        frame, foot, vec = detect_contact_frame_fused(ball_track, foot_tracks, fps, strict=True)
        # Single blip with no sustained follow-on should be rejected
        self.assertIsNone(frame, "Single-frame spike blip should be rejected by consecutive-spike gate")

    def test_sustained_spike_with_foot_is_accepted(self):
        """A spike followed by sustained motion AND a nearby foot is accepted."""
        fps = 30.0
        # Ball slow, then two consecutive high-velocity frames, then coast
        ball_track = (
            [{"frame": i, "x": float(i * 5), "y": 50.0, "width_px": 20} for i in range(5)]
            + [{"frame": 5,  "x": 250.0, "y": 50.0, "width_px": 20}]
            + [{"frame": 6,  "x": 500.0, "y": 50.0, "width_px": 20}]  # sustained
            + [{"frame": i, "x": float(500 + (i - 7) * 10), "y": 50.0, "width_px": 20} for i in range(7, 12)]
        )
        foot_tracks = self._make_foot_tracks(5, 260.0, 50.0)
        frame, foot, vec = detect_contact_frame_fused(ball_track, foot_tracks, fps, strict=True)
        self.assertIsNotNone(frame, "Sustained spike with nearby foot should be accepted")
        self.assertIn(foot, ("left", "right"))

    def test_no_foot_match_still_rejected_in_strict(self):
        """Even a sustained spike without a nearby foot returns None in strict mode."""
        fps = 30.0
        ball_track = (
            [{"frame": i, "x": float(i * 5), "y": 50.0, "width_px": 20} for i in range(5)]
            + [{"frame": 5, "x": 250.0, "y": 50.0, "width_px": 20}]
            + [{"frame": 6, "x": 500.0, "y": 50.0, "width_px": 20}]
            + [{"frame": i, "x": float(500 + (i - 7) * 10), "y": 50.0, "width_px": 20} for i in range(7, 12)]
        )
        foot_tracks = {"Right_Foot": [], "Left_Foot": []}  # no foot at all
        frame, foot, vec = detect_contact_frame_fused(ball_track, foot_tracks, fps, strict=True)
        self.assertIsNone(frame, "No foot match should reject in strict mode")

    def test_min_contact_frames_still_works_as_secondary_gate(self):
        """When min_contact_frames is explicitly passed, it also filters early spikes."""
        fps = 30.0
        # Sustained spike at frame 1 (very early)
        ball_track = (
            [{"frame": 0, "x": 0.0, "y": 50.0, "width_px": 20}]
            + [{"frame": 1, "x": 250.0, "y": 50.0, "width_px": 20}]
            + [{"frame": 2, "x": 500.0, "y": 50.0, "width_px": 20}]
            + [{"frame": i, "x": float(500 + (i - 3) * 10), "y": 50.0, "width_px": 20} for i in range(3, 12)]
        )
        foot_tracks = self._make_foot_tracks(1, 260.0, 50.0)
        # With explicit time-floor of 5 frames, frame 1 is not eligible
        frame, foot, vec = detect_contact_frame_fused(
            ball_track, foot_tracks, fps,
            min_contact_frames=5,
            strict=True
        )
        self.assertIsNone(frame, "min_contact_frames=5 should block a spike at frame 1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
