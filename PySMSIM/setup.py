import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PySMSIM", # Replace with your own username
    version="0.0.1",
    author="Jamie Manson",
    author_email="jamie.manson377@gmail.com",
    description="Python implementation of the super modified simplex algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jmanson377/PySMSIM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)