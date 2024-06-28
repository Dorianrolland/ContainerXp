#!/usr/bin/env python

from __future__ import print_function
import logging
from distutils.version import StrictVersion

import hpccm
import hpccm.config
from arguments import arguments, doc, footer
from hpccm.building_blocks import *
from hpccm.primitives import *

def sdk():
    args, distro = arguments()
    tdoc = doc(args, 'sdk')

    # Set the image tag based on the specified version (default to 10.0)
    mkl = 'no'
    if args.cuda != 'no':
        # If CUDA is enabled, use the NVIDIA CUDA base image
        image = 'nvidia/cuda:{}-devel-{}{}'.format(args.cuda, args.system, args.system_version)
        if args.oneapi != 'no':
            oneapi = 'no'
            logging.warning("CUDA and OneAPI are not compatible. Ignoring OneAPI for now.")  # Avertissement si OneAPI est aussi spécifié
    elif args.oneapi != 'no':
        # If OneAPI is enabled, use the Intel OneAPI base image
        image = 'intel/oneapi-hpckit:devel-{}{}'.format(args.system, args.system_version)
        if args.target_arch != 'x86_64':
            logging.error('OneAPI is only valid for amd64 processors')  # Erreur si OneAPI est utilisé sur une architecture non-x86_64
    else:
        # Otherwise, use the base system image
        image = '{}:{}'.format(args.system, args.system_version if args.system_version is not None else 'latest')

    Stage0 = hpccm.Stage()
    Stage0 += comment(tdoc, reformat=False)  # Add documentation comment
    Stage0.name = 'sdk'
    Stage0.baseimage(image, _distro=distro)  # Set the base image and Linux distribution
    Stage0 += comment('SDK stage', reformat=False)  # Add another comment

    Stage0 += label(metadata={'maintainer': 'bigdft-developers@lists.launchpad.net'})  # Add a maintainer label
    Stage0 += environment(variables={'DEBIAN_FRONTEND': 'noninteractive'})  # Set environment variable to avoid interactive prompts
    Stage0 += raw(docker='SHELL ["/bin/bash", "-c"]')  # Use bash as the default shell
    Stage0 += shell(commands=['useradd -ms /bin/bash lsim'])  # Create a user named 'lsim'

    # BigDFT packages
    # system independent ones
    ospackages = ['autoconf', 'automake', 'bison', 'bzip2', 'chrpath', 'cmake', 'cpio', 'curl', 'doxygen', 'ethtool', 'flex',
                  'gdb', 'gettext', 'git', 'gnome-common', 'graphviz', 'intltool', 'kmod', 'libtool', 'lsof', 'net-tools', 'ninja-build',
                  'patch', 'pciutils', 'perl', 'pkg-config', 'rsync', 'strace', 'swig', 'tcl', 'tk', 'valgrind', 'vim', 'wget']

    # Additional packages for Debian/Ubuntu
    apt_packages = ospackages + [
        'autotools-dev', 'libpcre3-dev', 'libltdl-dev', 'lsb-release', 'libz-dev', 'zlib1g-dev', 'libzmq3-dev', 'libmount-dev',
        'iproute2', 'libnl-route-3-200', 'libnuma1', 'linux-headers-generic', 'gtk-doc-tools', 'libxml2-dev', 'libglu1-mesa-dev',
        'libnetcdf-dev', 'libgirepository1.0-dev', 'dpatch', 'libgtk-3-dev', 'libmount-dev', 'locales', 'ssh', 'libyaml-dev'
    ]

    # Additional packages for CentOS/RHEL
    yum_packages = ospackages + ['pcre-devel', 'libtool-ltdl-devel', 'redhat-lsb', 'glibc-devel', 'zlib-devel', 'zeromq-devel', 'libmount-dev',
        'iproute', 'libnl3-devel', 'numactl-libs', 'kernel-headers', 'gtk-doc', 'libxml2-devel', 'mesa-libGLU-devel', 'netcdf-devel',
        'gobject-introspection-devel', 'gtk3-devel', 'libmount-devel', 'openssh', 'libarchive', 'libyaml-devel'
    ]

    # Install Boost from packages except for OneAPI or Intel Python builds
    if args.target_arch != "x86_64" or not (args.python == 'intel' or args.oneapi != 'no'):
        apt_packages += ['libboost-dev', 'libboost-python-dev']
        yum_packages += ['boost-devel', 'boost-python3-devel']

    if args.cuda != 'no':
        # If CUDA is enabled, add CUDA-related packages and environment variables
        apt_packages += ['ocl-icd-libopencl1']
        yum_packages += ['ocl-icd']
        Stage0 += environment(variables={
            'LD_LIBRARY_PATH': '/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}',
            'LIBRARY_PATH': '/usr/local/cuda/lib64:${LIBRARY_PATH}',
            'NVIDIA_VISIBLE_DEVICES': 'all',
            'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility'
        })
        Stage0 += shell(commands=[
            'mkdir -p /etc/OpenCL/vendors',
            'echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd',
            'cp /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/lib/libcuda.so.1',
            'cp /usr/local/cuda/lib64/stubs/libnvidia-ml.so /usr/local/lib/libnvidia-ml.so.1'
        ])

    Stage0 += packages(apt=apt_packages, yum=yum_packages, powertools=True, epel=True, release_stream=True)

    # ... (Installation de Miniconda et des paquets conda - corrigé) ...
    
    if args.target_arch == 'x86_64':
        # Télécharger Miniconda sur l'hôte (à faire manuellement avant de lancer le workflow)
        Stage0 += copy(src='Miniconda3-latest-Linux-x86_64.sh', dest='/var/tmp/')

        # Installer Miniconda
        Stage0 += shell(commands=[
            'bash /var/tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local/anaconda',
            '/usr/local/anaconda/bin/conda init',
            'rm -rf /var/tmp/Miniconda3-latest-Linux-x86_64.sh'
        ])

        conda_path = '/usr/local/anaconda/'
        commands = [
            '. {}/etc/profile.d/conda.sh'.format(conda_path),  # Activer conda
            'conda config --set channel_priority strict', # Prioriser les canaux spécifiés
            'conda config --add channels conda-forge --add channels intel --add channels nvidia',
            'conda install -y ' + ' '.join(conda_packages), # Installer les paquets individuellement
            'conda clean -afy'  # Nettoyer le cache conda
        ]

        # ... (Commandes de configuration supplémentaires - inchangées) ...
    else:
        # ... (Installation de Python pour les autres architectures - inchangé) ...

    # ... (Installation de Boost, Jupyter, etc. - inchangé) ...

    return Stage0

if __name__ == '__main__':
    print(footer(sdk()))
