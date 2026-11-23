import unittest

from scripts.install_runtime import (CPU_INDEX_URL, CUDA_INDEX_URL,
                                     torch_index_url)


class RuntimeInstallerTests(unittest.TestCase):
    def test_macos_uses_pypi_for_official_pytorch_wheels(self):
        self.assertIsNone(torch_index_url("cpu", "darwin"))

    def test_linux_and_windows_use_hardware_specific_indexes(self):
        self.assertEqual(torch_index_url("cpu", "linux"), CPU_INDEX_URL)
        self.assertEqual(torch_index_url("cpu", "win32"), CPU_INDEX_URL)
        self.assertEqual(torch_index_url("cuda", "linux"), CUDA_INDEX_URL)
        self.assertEqual(torch_index_url("cuda", "win32"), CUDA_INDEX_URL)


if __name__ == "__main__":
    unittest.main()
