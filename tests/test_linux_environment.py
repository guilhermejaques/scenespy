import os
import sys
import unittest
from unittest.mock import mock_open, patch

from scenespy.shared import (configure_linux_virtual_machine_graphics,
                             external_process_env)


class LinuxEnvironmentTests(unittest.TestCase):
    def test_frozen_linux_child_restores_original_library_path(self):
        source = {
            "LD_LIBRARY_PATH": "/tmp/_MEI123",
            "LD_LIBRARY_PATH_ORIG": "/usr/local/lib",
            "KEEP": "yes",
        }
        with patch.object(sys, "platform", "linux"), \
                patch.object(sys, "frozen", True, create=True):
            env = external_process_env(source)
        self.assertEqual(env["LD_LIBRARY_PATH"], "/usr/local/lib")
        self.assertEqual(env["KEEP"], "yes")

    def test_frozen_linux_child_removes_injected_library_path_without_original(self):
        with patch.object(sys, "platform", "linux"), \
                patch.object(sys, "frozen", True, create=True):
            env = external_process_env({"LD_LIBRARY_PATH": "/tmp/_MEI123"})
        self.assertNotIn("LD_LIBRARY_PATH", env)

    def test_virtualbox_linux_uses_non_glx_software_rendering(self):
        with patch.object(sys, "platform", "linux"), \
                patch("builtins.open", mock_open(read_data="Oracle Corporation VirtualBox")), \
                patch.dict(os.environ, {}, clear=True):
            configure_linux_virtual_machine_graphics()
            self.assertEqual(os.environ["QT_XCB_GL_INTEGRATION"], "none")
            self.assertEqual(os.environ["QT_OPENGL"], "software")

    def test_physical_linux_does_not_force_graphics_backend(self):
        with patch.object(sys, "platform", "linux"), \
                patch("builtins.open", mock_open(read_data="System Product Name")), \
                patch.dict(os.environ, {}, clear=True):
            configure_linux_virtual_machine_graphics()
            self.assertNotIn("QT_XCB_GL_INTEGRATION", os.environ)
            self.assertNotIn("QT_OPENGL", os.environ)


if __name__ == "__main__":
    unittest.main()
