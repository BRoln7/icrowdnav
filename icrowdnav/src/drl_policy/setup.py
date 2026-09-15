from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=[
        "bev_perception",
        "bev_perception.layers",
        "bev_perception.models",
        "bev_perception.utils",
        "socbev_gym",
        "socbev_gym.envs",
        "mdp",
    ],
    package_dir={"": "."},
)

setup(**setup_args)
