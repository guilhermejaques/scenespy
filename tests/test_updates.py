import unittest

from scenespy.qt_app import _trusted_update_url, update_asset_suffix, version_tuple


class UpdateSupportTests(unittest.TestCase):
    def test_semantic_versions_compare_numerically(self):
        self.assertGreater(version_tuple("v0.10.0"), version_tuple("0.9.9"))
        self.assertEqual(version_tuple("1.2"), (1, 2, 0))

    def test_release_asset_matches_platform_and_architecture(self):
        self.assertEqual(
            update_asset_suffix("win32", "AMD64"),
            "-windows-x64-setup.exe",
        )
        self.assertEqual(
            update_asset_suffix("linux", "x86_64"),
            "-linux-amd64.deb",
        )
        self.assertEqual(
            update_asset_suffix("darwin", "arm64"),
            "-macos-arm64.pkg",
        )
        self.assertEqual(
            update_asset_suffix("darwin", "x86_64"),
            "-macos-x64.pkg",
        )

    def test_update_downloads_only_use_trusted_https_hosts(self):
        self.assertTrue(_trusted_update_url(
            "https://github.com/guilhermejaques/scenespy/releases/download/v1/file.exe"))
        self.assertTrue(_trusted_update_url(
            "https://release-assets.githubusercontent.com/file"))
        self.assertFalse(_trusted_update_url(
            "http://github.com/guilhermejaques/scenespy/file.exe"))
        self.assertFalse(_trusted_update_url("https://example.com/file.exe"))


if __name__ == "__main__":
    unittest.main()
