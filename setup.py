from setuptools import find_packages, setup

PACKAGE_NAME = "chemprop_pretrained"
PACKAGE_DIRS = ["chemprop_pretrained"]
VERSION = "0.0.0"
RUNTIME_DEPS = None


def main():
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        install_requires=RUNTIME_DEPS,
        packages=find_packages(include=PACKAGE_DIRS),
    )

if __name__ == "__main__":
    main()
