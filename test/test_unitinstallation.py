# test_installation.py
import subprocess
import unittest


class TestInstallation(unittest.TestCase):
    def test_package_import_after_installation(self):
        # Run the installation command as a subprocess
        result = subprocess.run(
            ["pip", "install", "housing"], capture_output=True, text=True
        )

        # Check if the installation process was successful (return code 0)
        self.assertEqual(
            result.returncode,
            0,
            f"Package installation failed: {result.stderr}",
        )

        # Attempt to import the package after installation
        try:
            import housing
        except ImportError:
            self.fail("housing import failed after installation")


if __name__ == "__main__":
    unittest.main()
