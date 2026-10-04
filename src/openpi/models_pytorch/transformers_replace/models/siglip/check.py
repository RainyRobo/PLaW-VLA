import packaging.version
import transformers


def check_whether_transformers_replace_is_installed_correctly():
    try:
        return packaging.version.parse(transformers.__version__).major == 5
    except packaging.version.InvalidVersion:
        return False
