from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Island divertor optimization framework'
LONG_DESCRIPTION = 'Island divertor generation and optimization for stellarators'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="island_divertor_opt", 
        version=VERSION,
        author="Avigdor Veksler",
        author_email="av3249@columbia.edu",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'divertor'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Research",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
