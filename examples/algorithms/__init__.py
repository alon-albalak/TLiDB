import os
import sys
package_directory = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_FOLDER = os.path.join(package_directory, "..")
sys.path.append(EXAMPLES_FOLDER)

from .initializer import initialize_algorithm