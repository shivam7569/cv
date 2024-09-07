import unittest
import cv
from cv.src.gpu_devices import GPU_Support

class TestModuleImport(unittest.TestCase):
    
    def test_import_module(self):
        try:
            # Attempt to import the module
            import cv.backbones.AlexNet as AlexNet
        except ImportError:
            self.fail("Failed to import the module 'your_module_name'")
        
    def test_import_specific_function(self):
        try:
            # Attempt to import a specific function from the module
            from cv.utils.os_utils import check_dir
        except ImportError:
            self.fail("Failed to import 'your_function_name' from 'your_module_name'")

if __name__ == '__main__':
    unittest.main()