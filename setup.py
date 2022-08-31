from setuptools import setup, find_packages

setup( 
    name="mayoclinic",
    version="0.1dev",
    packages=find_packages(include=["mayoclinic"]),
    install_requires=[
       'rasterio',
       'timm'
    ]
)
