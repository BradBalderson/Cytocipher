from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cytocipher',
    version='0.1.3',
    description='Cluster significance analysis in scRNA-seq',
    url='https://github.com/BradBalderson/Cytocipher',
    author='Brad Balderson',
    author_email='brad.balderson@uqconnect.edu.au',
    license='GNU GENERAL PUBLIC LICENSE V3',
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=['cytocipher'],
    install_requires=['scanpy==1.9.1',
                      'numba==0.56.0',
                      ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
    ],
)
