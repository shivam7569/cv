from setuptools import setup, find_packages

setup(
    name="cv",
    version="0.1.0",
    author="Shivam Chaudhary",
    author_email="shivam.iitmandi@gmail.com",
    description="Model implementations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/shivam7569/cv",
    packages=find_packages(exclude=["tests", "workshop"]),
    install_requires=[
        "requests",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
