import os.path
import setuptools

repository_dir = os.path.dirname(__file__)

with open(os.path.join(repository_dir, "requirements.txt")) as fh:
    requirements = [line for line in fh.readlines()]

setuptools.setup(
    name="ancer",
    version=1.0,
    author="Motasem Alfarra",
    author_email="motasem.alfarra@kaust.edu.sa",
    python_requires=">=3.7",
    description="Anisotropic sample-wise randomized smoothing package",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7"
    ],
    install_requires=requirements,
    include_package_data=True,
)
