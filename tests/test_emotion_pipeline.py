import unittest

from backend.services.emotion_pipeline import (
    build_multimodal_result,
    compute_acoustic_features,
    normalize_emotion_label,
)


class EmotionPipelineTests(unittest.TestCase):
    def test_normalize_emotion_label_maps_known_variants(self):
        self.assertEqual(normalize_emotion_label("angry"), "angry")
        self.assertEqual(normalize_emotion_label("happy"), "happy")
        self.assertEqual(normalize_emotion_label("neutral"), "neutral")
        self.assertEqual(normalize_emotion_label("fearful"), "fear")
        self.assertEqual(normalize_emotion_label("sadness"), "sad")
        self.assertEqual(normalize_emotion_label("frustrated"), "angry")

    def test_multimodal_fusion_keeps_angry_when_voice_is_strong(self):
        voice = {"label": "angry", "score": 0.82}
        text = {"label": "neutral", "score": 0.48}
        result = build_multimodal_result(voice, text)
        self.assertEqual(result["final_label"], "angry")
        self.assertGreater(result["confidence"], 0.55)

    def test_acoustic_features_capture_energy_and_variation(self):
        audio = [0.0, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9]
        features = compute_acoustic_features(audio)
        self.assertIn("intensity", features)
        self.assertIn("pitch_variation", features)
        self.assertGreater(features["intensity"], 0)
        self.assertGreaterEqual(features["pitch_variation"], 0)


if __name__ == "__main__":
    unittest.main()
