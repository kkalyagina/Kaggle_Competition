import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="project_2month",
    version="0.0.1",
    author="Kristina Kaliagina",
    author_email="Kristina_Kaliagina@epam.com",
    description="EPAM project",
    url="https://github.com/kkalyagina/epam.git",
    packages=['project_2month'],
    python_requires='>=3.6',
    install_requires=required
)
