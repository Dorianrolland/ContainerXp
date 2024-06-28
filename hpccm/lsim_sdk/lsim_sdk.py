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

  tdoc=doc(args, 'sdk')

  #######
  ## SDK stage
  #######
  # Set the image tag based on the specified version (default to 10.0)
  mkl = 'no'
  if args.cuda != 'no':
    image = 'nvidia/cuda:{}-devel-{}{}'.format(args.cuda, args.system, args.system_version)
    if args.oneapi != 'no':
      oneapi = 'no'
      logging.warning('For now we can\'t mix CUDA SDK with OneAPI base image. MKL can still be installed later. Ignoring OneAPI at this step')  
  elif args.oneapi != 'no':
    image = 'intel/oneapi-hpckit:devel-{}{}'.format(args.system, args.system_version)
    if args.target_arch != 'x86_64':
      logging.error('OneAPI is only valid for amd64 processors')
  else:
    image = '{}:{}'.format(args.system, args.system_version if args.system_version is not None else 'latest')

  Stage0 = hpccm.Stage()

  Stage0 += comment(tdoc, reformat=False)
  Stage0.name = 'sdk'
  Stage0.baseimage(image,_distro=distro)
  Stage0 += comment('SDK stage', reformat=False)

  Stage0 += label(metadata={'maintainer': 'bigdft-developers@lists.launchpad.net'})
  Stage0 += environment(variables={'DEBIAN_FRONTEND': 'noninteractive'})
  #SHELL ['/bin/bash', '-c']
  Stage0 += raw(docker='SHELL ["/bin/bash", "-c"]')
  Stage0 += shell(commands=['useradd -ms /bin/bash lsim'])

  #BigDFT packages
  #system independent ones
  ospackages=['autoconf', 'automake', 'bison', 'bzip2', 'chrpath', 'cmake', 'cpio', 'curl', 'doxygen', 'ethtool', 'flex',
              'gdb', 'gettext', 'git', 'gnome-common', 'graphviz', 'intltool', 'kmod', 'libtool', 'lsof', 'net-tools', 'ninja-build',
              'patch', 'pciutils', 'perl', 'pkg-config', 'rsync', 'strace', 'swig', 'tcl', 'tk', 'valgrind', 'vim', 'wget']

  apt_packages=ospackages+[
          'autotools-dev', 'libpcre3-dev', 'libltdl-dev', 'lsb-release', 'libz-dev', 'zlib1g-dev', 'libzmq3-dev', 'libmount-dev',
          'iproute2', 'libnl-route-3-200', 'libnuma1', 'linux-headers-generic', 'gtk-doc-tools', 'libxml2-dev', 'libglu1-mesa-dev',
          'libnetcdf-dev', 'libgirepository1.0-dev', 'dpatch', 'libgtk-3-dev', 'libmount-dev', 'locales', 'ssh', 'libyaml-dev']
  yum_packages=ospackages+['pcre-devel', 'libtool-ltdl-devel', 'redhat-lsb', 'glibc-devel', 'zlib-devel', 'zeromq-devel', 'libmount-devel', 
          'iproute', 'libnl3-devel', 'numactl-libs', 'kernel-headers', 'gtk-doc', 'libxml2-devel', 'mesa-libGLU-devel', 'netcdf-devel',
          'gobject-introspection-devel',  'gtk3-devel', 'libmount-devel', 'openssh', 'libarchive', 'libyaml-devel']
  #boost from packages except for oneapi or intel python builds.
  if args.target_arch != "x86_64" or not (args.python == 'intel' or args.oneapi != 'no'):
    apt_packages+=['libboost-dev', 'libboost-python-dev']
    yum_packages+=['boost-devel', 'boost-python3-devel']
  
  if args.cuda != 'no':
    apt_packages += ['ocl-icd-libopencl1']
    yum_packages += ['ocl-icd']
    Stage0 += environment(variables={ 'LD_LIBRARY_PATH': '/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}',
                                      'LIBRARY_PATH': '/usr/local/cuda/lib64:${LIBRARY_PATH}',
                                      'NVIDIA_VISIBLE_DEVICES': 'all',
                                      'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility'})
    Stage0 += shell(commands=['mkdir -p /etc/OpenCL/vendors',
                              'echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd',
                              'cp /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/lib/libcuda.so.1',
                              'cp /usr/local/cuda/lib64/stubs/libnvidia-ml.so /usr/local/lib/libnvidia-ml.so.1'])

  Stage0 += packages(apt=apt_packages, yum=yum_packages, powertools=True, epel=True, release_stream=True)
  Stage0 += comment('wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwoooooooooooooooooooooooooooooooooooooowwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww', reformat=False)
  
  #Install boost with the provided python

  if (args.jupyter == 'yes'):
    Stage0 += raw(docker='EXPOSE 8888')
    Stage0 += raw(docker='CMD jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.token=bigdft --no-browser', 
                  singularity='%runscript\n jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.token=bigdft --no-browser')

  #workaround for an issue in Ubuntu 20 on docker
  if args.system == 'ubuntu' and args.system_version >= StrictVersion('20.04') and args.target_arch == "x86_64":
    Stage0 += environment(variables={'LD_PRELOAD': '/usr/lib/x86_64-linux-gnu/libtinfo.so.6'})

  if args.system == 'ubuntu':
    Stage0 += environment(variables={ "LANG": "en_US.UTF-8",
                                      "LANGUAGE": "en_US.UTF-8",
                                      "LC_ALL": "en_US.UTF-8"})
  else:
    Stage0 += environment(variables={ "LANG": "C.UTF-8",
                                      "LANGUAGE": "C.UTF-8",
                                      "LC_ALL": "C.UTF-8",
                                      "PKG_CONFIG_PATH": "/usr/lib64:/usr/share/lib64"})   

  if args.oneapi != 'no':
    Stage0 += shell(commands=['if [ -e /root/.oneapi_env_vars ]; then cp /root/.oneapi_env_vars /opt/intel/.oneapi_env_vars; chmod +x /opt/intel/.oneapi_env_vars; fi'])
    Stage0 += raw(docker='ENTRYPOINT ["bash", "-c", "source /opt/intel/.oneapi_env_vars && \\\"$@\\\"", "bash"]', 
                  singularity="%runscript\n bash -c 'source /opt/intel/.oneapi_env_vars && \\\"$@\\\"' bash")

  return Stage0

if __name__ == '__main__':
  print(footer(sdk()))
