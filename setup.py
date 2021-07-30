from setuptools import setup, find_packages


setup(
    name='pyism',
    version='0.0.2',
    description='A scikit-multiflow API for Tensorflow-based models implementing Incremental Sequence Models (ISM)',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    author='Álvaro Lemos',
    author_email='alvarolemos@ufmg.br',
    install_requires=['tensorflow', 'scikit-multiflow'],
)
