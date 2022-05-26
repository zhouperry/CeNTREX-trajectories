import setuptools

setuptools.setup(
    name="CeNTREX_trajectories",
    author="Olivier Grasdijk",
    author_email="jgrasdijk@anl.gov",
    description="TlF trajectory calculations used in the CeNTREX experiment",
    url="https://github.com/",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy", "joblib"],
    data_files=[
        (
            "CeNTREX_trajectories/saved_data",
            ["CeNTREX_trajectories/saved_data/stark_poly.pkl",],
        ),
    ],
    python_requires=">=3.8",
    version="0.1",
)
