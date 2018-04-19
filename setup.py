import setuptools
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as file_handle:
    long_description = file_handle.read()

setuptools.setup(
    name="pymotutils",
    version="0.1.0",
    description="Multiple object tracking utilities",
    long_description=long_description,
    url="https://github.com/nwojke/pymotutils",
    author="Nicolai Wojke",
    author_email="nwojke@uni-koblenz.de",
    license="GPL3",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
    ],
    keywords="computer_vision, tracking",
    packages=setuptools.find_packages(exclude=["examples"]),
    install_requires=[
        "numpy",
        "opencv-python>=3.0",
        "six",
        "scipy",
        "scikit_learn",
    ],
    python_requires=">=2.7,>=3.0"
)
