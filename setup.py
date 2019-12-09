"""Setup file for mitoutils."""

from setuptools import setup

requires = [
    'numba',
    'numpy',
    'matplotlib',
    'xarray',
    'xmitgcm',
    'xgcm',
    'scipy',
]

setup(name='MITgcmtools',
      version='0.1.0',
      description='Toolbox built around the MITgcm',
      url='https://github.com/aulemahal/MITgcmtools-py',
      author='PascalB',
      author_email='pascal.bourgault@gmail.com',
      packages=['MITgcmtools'],
      entry_points={'console_scripts': ['gendata=MITgcmtools.gendata:main',
                                        'diagnose=MITgcmtools.quantities:diagnose'],
                    },
      install_requires=requires,
      )
