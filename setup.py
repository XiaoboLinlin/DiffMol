from setuptools import setup, find_packages

setup(
    name='DiffMol',  # Replace with your own package name
    version='0.1.0',  # Replace with your own version
    author='Your Name',  # Replace with your name
    author_email='your.email@example.com',  # Replace with your email
    description='A short description of the project',  # Replace with a short description
    long_description=open('README.md').read(),  # Include a detailed description from the README.md
    long_description_content_type='text/markdown',  # This is important to ensure your README.md is rendered correctly on PyPI
    url='https://github.com/yourusername/DiffMol',  # Replace with the URL of your project
    packages=find_packages(),  # Automatically find and include all packages
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace with your chosen license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Replace with the Python version requirements
    install_requires=[
        # List your project's dependencies here.
        # Examples:
        # 'numpy>=1.18.1',
        # 'pandas>=1.0.3',
    ],
    extras_require={
        'dev': [
            'pytest>=3.7',
            # Include any additional package that is required for development
        ],
    },
    entry_points={
        'console_scripts': [
            'diffmol=DiffMol.main:main',  # This makes the package executable through the command line as `diffmol`
        ],
    },
    include_package_data=True,  # This is required to include non-code files specified in MANIFEST.in
)
