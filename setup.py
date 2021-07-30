from setuptools import setup, find_packages


setup(
    name='ism',
    version='0.0.1',
    description='A scikit-multiflow API for Tensorflow-based models implementing Incremental Sequence Models (ISM)',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    author='Álvaro Lemos',
    author_email='alvarolemos@ufmg.br',
    licence='',
    install_requires=[
        'numpy==1.19.5',
        'tensorflow==2.5.0',
        'scikit-multiflow==0.5.3',
    ],
)
