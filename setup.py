from distutils.core import setup

INSTALL_REQUIRES = ['tensorflow-gpu==2.0.0b']

setup(
    name='earthbeaver',
    version='0.1.0',
    packages=['beaver'],
    install_requires=INSTALL_REQUIRES
)
