import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="utils_project",
    version="0.0.1",
    author="Kristina Kaliagina",
    author_email="Kristina_Kaliagina@epam.com",
    description="Project_2month",
    url="https://github.com/kkalyagina/epam.git",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=required
)
