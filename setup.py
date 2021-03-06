from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension

extra_compile_args = []
# extra_compile_args = ['-g', '-O0', '-DDEBUG']

ext_modules = [
    CppExtension('kplex_pool.kplex_cpu', ['cpu/kplex.cpp'], extra_compile_args=extra_compile_args),
    CppExtension('kplex_pool.pool_edges_cpu', ['cpu/pool_edges.cpp'], extra_compile_args=extra_compile_args),
    CppExtension('kplex_pool.cc_cpu', ['cpu/cc.cpp'], extra_compile_args=extra_compile_args),
    CppExtension('kplex_pool.simplify_cpu', [
                     'cpu/simplify.cpp',
                     'cpu/disjoint_sets.cpp'
                 ], extra_compile_args=extra_compile_args),
]
cmdclass = {'build_ext': BuildExtension}

__version__ = '0.0.1'
url = 'https://github.com/flandolfi/kplex-pool'

install_requires = [
    'numpy',
    'torch', 
    'torch_cluster', 
    'torch_sparse', 
    'torch_geometric',
    'tqdm'
]
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='kplex_pool',
    version=__version__,
    description='K-plex pooling layer for Graph Neural Network',
    author='Francesco Landolfi',
    author_email='fran.landolfi@gmail.com',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=['pytorch', 'cluster', 'geometric-deep-learning', 'graph'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)