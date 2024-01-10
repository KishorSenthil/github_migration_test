# # import unittest


# # def test_import():
# #     import package


# # if __name__ == "__main__":
# #     test_import()

# # test_installation.py

# # import subprocess


# # def test_installation():
# #     # Run the installation command as a subprocess
# #     result = subprocess.run(
# #         ["python", "pyproject.toml", "install"], capture_output=True, text=True
# #     )

# #     # Check if the installation process was successful (return code 0)
# #     assert result.returncode == 0, f"Installation failed: {result.stderr}"

# #     # You can also check for specific files, configurations, or other indicators of a successful installation
# #     # For example, check if a package/module is importable after installation
# #     try:
# #         import your_module
# #     except ImportError:
# #         raise AssertionError("Module not found after installation")

# import unittest

# import package


# class TestImport(unittest.TestCase):
#     def test_import(self):
#         # Your assertion here, for example:

#         self.assertTrue(hasattr(package, "pandas"), "Package import failed")


# if __name__ == "__main__":
#     unittest.main()


def test_pkg_installation():
    try:
        import housing

    except Exception as e:
        assert False, f"Error: {e}. package is not installed properly."
