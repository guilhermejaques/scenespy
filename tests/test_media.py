import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scenespy.media import probe_media


class MediaProbeTests(unittest.TestCase):
    def test_mkv_prefers_video_tag_duration_over_longer_container_audio(self):
        payload = {
            "streams": [{
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "24/1",
                "r_frame_rate": "24/1",
                "start_time": "0.083",
                "tags": {"DURATION": "00:00:06.083000000"},
            }],
            "format": {"duration": "6.120"},
        }
        completed = SimpleNamespace(stdout=json.dumps(payload))

        with patch("scenespy.media.subprocess.run", return_value=completed):
            media = probe_media("movie.mkv")

        self.assertEqual(media.duration, 6.0)
        self.assertEqual(media.total_frames, 144)


if __name__ == "__main__":
    unittest.main()
