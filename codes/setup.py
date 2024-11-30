from setuptools import setup, find_packages

setup(
    name='pfss',
    version='1.0.0',
    packages=find_packages(),
    description='PFSS Solver',
    long_description='Python Tools to Solve the PFSS model by the syntopic magnetic map',
    author='Li Yihua, Chen Guoyin',
    author_email='gychen@smail.nju.edu.cn',
    install_requires=[
        'numpy>=1.24.0', 
        'matplotlib>=3.9.2',  
        'scipy>=1.8.1',   
        'sunpy>=5.1.3', 
        'torch>=2.3.0', 
        'astropy>=6.1.0', 
        'vtk>=9.3.1',
        'pyevtk>=1.6.0'
    ],
)
