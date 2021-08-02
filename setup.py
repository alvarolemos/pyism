import pathlib
from setuptools import setup, find_packages


README = (pathlib.Path(__file__).parent / 'README.md').read_text()


setup(
    name='pyism',
    version='0.0.2',
    description='A scikit-multiflow API for Tensorflow-based models implementing Incremental Sequence Models (ISM)',
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url="https://github.com/alvarolemos/pyism",
    author='Álvaro Lemos',
    author_email='alvarolemos@ufmg.br',
    install_requires=[
        'numpy==1.19.2',
        'tensorflow==2.5.0',
        'scikit-multiflow==0.5.3'
    ],
)
