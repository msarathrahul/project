from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path : str) -> List:
    """
    input : File path to requirements.txt
    return : A list of required modules/packages
    """
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
    
    requirements_lst = [req.replace('\n','') for req in requirements]

    if HYPHEN_E_DOT in requirements_lst:
        requirements_lst.remove(HYPHEN_E_DOT)

    return requirements_lst

setup(
    name = 'project',
    version = '0.0.1',
    author = 'msarathrahul',
    author_email = 'msarathrahul@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)