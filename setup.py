from distutils.core import setup

INSTALL_REQUIRES = ['tensorflow-gpu==2.0.0b1', 'scipy', 'pillow',
                    'tqdm', 'matplotlib', 'loguru', 'pandas', 'pep8',
                    'autopep8', 'numpy']

setup(
    name='kss',
    version='0.1.0',
    packages=['kss'],
    install_requires=INSTALL_REQUIRES
)
