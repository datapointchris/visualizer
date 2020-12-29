import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="visualizer",
    version="0.1.0",
    author="Chris Birch",
    author_email="chrisbirch@live.com",
    description="Visualizer for data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/datapointchris/visualizer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy",
                      "pandas",
                      "wordcloud",
                      "matplotlib",
                      "seaborn",
                      "pillow",
                      "sklearn"],
    python_requires='>=3.6',
)
