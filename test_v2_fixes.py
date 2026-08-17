import os
import sys
import unittest

from shot_analysis import (
    determine_acceptance,
    generate_coaching_note,
)

class TestV2Fixes(unittest.TestCase):

    def test_item3_multi_reject_reasons(self):
        # When multiple checks fail, all reasons must be captured in order
        accepted, reasons = determine_acceptance(
            contact_found=False,
            foot_known=False,
            calibration_ok=False,
            zone_reject_reason="outside_goal_plane"
        )
        self.assertFalse(accepted)
        self.assertEqual(
            reasons,
            ["no_contact_detected", "foot_not_identified", "calibration_failed", "outside_goal_plane"]
        )

    def test_item3_single_reject_reason(self):
        accepted, reasons = determine_acceptance(
            contact_found=True,
            foot_known=True,
            calibration_ok=False,
            zone_reject_reason=None
        )
        self.assertFalse(accepted)
        self.assertEqual(reasons, ["calibration_failed"])

    def test_item3_accepted(self):
        accepted, reasons = determine_acceptance(
            contact_found=True,
            foot_known=True,
            calibration_ok=True,
            zone_reject_reason=None
        )
        self.assertTrue(accepted)
        self.assertEqual(reasons, [])

    def test_item5_env_var_prioritization(self):
        # Simulate environment variable injected via Modal secret
        os.environ["GEMINI_API_KEY"] = "your-dummy-key-for-test"
        note = generate_coaching_note({"foot": "right", "confidence": 0.8, "accepted": True})
        # Should gracefully drop to local fallback without error when key is dummy
        self.assertTrue(isinstance(note, str))
        self.assertTrue(len(note) > 0)

if __name__ == "__main__":
    unittest.main()
