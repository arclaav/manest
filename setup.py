from setuptools import setup, find_packages

setup(
    name='manest',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'natumpy',
        'numpy',
        'pyyaml',
        'rich',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'manest=manest.cli:main',
        ],
    },
    description='modeling',
    author='Eternals',
)
