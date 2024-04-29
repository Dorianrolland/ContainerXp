doc="""
LSim bigdft build + runtime 

Contents:
  Ubuntu {}""".format(USERARG.get('ubuntu', '16.04'))+"""
  CUDA {}""".format(USERARG.get('cuda', '10.0'))+"""
  MKL
  GNU compilers (upstream)
  Python 3 (intel)
  jupyter notebook and jupyter lab
  v_sim-dev in the optional target

  This recipe was generated with command line :
$ hpccm.py --recipe hpccm_lsim-mpi.py --userarg cuda={}""".format(USERARG.get('cuda', '10.0'))+""" ubuntu={}""".format(USERARG.get('ubuntu', '16.04'))+""" mpi={}""".format(USERARG.get('mpi', 'ompi'))

from hpccm.templates.git import git
from distutils.version import LooseVersion, StrictVersion

#######
## Build bigdft
#######
# Make sure your base image is appropriate for CentOS if changing from Ubuntu
image = 'dorianalp38/sdk:latest'  # Assumed to be suitable for your needs, confirm if it's CentOS-based.

cuda_version = USERARG.get('cuda', '10.0')
if cuda_version == "8.0":
    ubuntu_version = "16.04"
else:
    ubuntu_version = USERARG.get('ubuntu', '16.04')

if ubuntu_version == "18.04" or ubuntu_version == "18.04-rc":
   distro = 'ubuntu18'
elif ubuntu_version == "20.04":
   distro = 'ubuntu20'
else:
   distro = 'ubuntu'

bigdft_version = USERARG.get('bigdft', 'devel')

Stage0 += comment(doc, reformat=False)
Stage0.name = 'bigdft_build'
Stage0.baseimage(image, _distro='centos7')  # Assurez-vous que l'image de base est correcte pour CentOS.

target_arch = USERARG.get('target_arch', 'x86_64')
import hpccm.config
hpccm.config.set_cpu_architecture(target_arch)
hpccm.config.set_cpu_target(target_arch)

Stage0 += raw(docker='USER root')
Stage0 += workdir(directory='/opt/')
Stage0 += shell(commands=['rm -rf /opt/bigdft'])
Stage0 += shell(commands=['git clone --branch ' + bigdft_version + ' https://gitlab.com/l_sim/bigdft-suite.git ./bigdft'])
Stage0 += shell(commands=['chown -R lsim:lsim /opt/bigdft', 'chmod -R 777 /opt/bigdft', 'mkdir /usr/local/bigdft', 'chmod -R 777 /usr/local/bigdft'])
# Assurez-vous que les paquets Python et les locales sont installés
Stage0 += packages(yum=['python3', 'python3-pip', 'glibc-all-langpacks'])

use_mkl = USERARG.get('mkl', 'yes') if target_arch == "x86_64" else "no"
if use_mkl == "yes":
    Stage0 += workdir(directory='/usr/local/anaconda/lib')
    Stage0 += raw(docker='SHELL ["/bin/bash", "-c"]')
    Stage0 += shell(commands=[
        "export GLOBIGNORE=libmkl_gf_lp64.*:libmkl_gnu_thread.*:libmkl_core.*:libmkl_avx2.*:libmkl_avx.*:libmkl_avx512.*:libmkl_def.*:libmkl_rt.*:libmkl_intel_thread.*:libmkl_intel_lp64.*",
        "rm -rf libmkl*",
        "unset GLOBIGNORE"
    ])
    Stage0 += raw(docker='SHELL ["/bin/sh", "-c"]')

Stage0 += workdir(directory='/opt/bigdft/build')
Stage0 += shell(commands=['chmod -R 777 /opt/bigdft/build'])

Stage0 += shell(commands=['mkdir /docker', 'chmod -R 777 /docker'])

Stage0 += raw(docker='USER lsim')
Stage0 += environment(variables={"LD_LIBRARY_PATH": "/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"})
Stage0 += environment(variables={"LIBRARY_PATH": "/usr/local/cuda/lib64:${LIBRARY_PATH}"})
Stage0 += environment(variables={"PYTHON": "python"})

Stage0 += shell(commands=[git().clone_step(repository='https://github.com/BigDFT-group/ContainerXP.git', directory='/docker')])
Stage0 += copy(src="./rcfiles/container.rc", dest="/tmp/container.rc")


mpi = USERARG.get('mpi', 'ompi')

cuda_version = USERARG.get('cuda', '10').split(".", 1)[0]
cuda_gencodes = [
    "-arch=sm_50 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_37,code=sm_37",
    "-gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52",
    "-gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61",
    "-gencode=arch=compute_70,code=sm_70"
]
if cuda_version == "10":
    cuda_gencodes.append("-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75")
elif cuda_version == "11":
    cuda_gencodes.append("-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80")

Stage0 += environment(variables={"CUDA_GENCODES": ' '.join(cuda_gencodes)})

Stage0 += workdir(directory='/opt/bigdft/build/')
Stage0 += shell(commands=['sed -i "s/__shfl_down(/__shfl_down_sync(0xFFFFFFFF,/g" ../psolver/src/cufft.cu'])

if use_mkl == "yes":
    Stage0 += environment(variables={
        "MKLROOT": "/usr/local/anaconda/",
        "LD_LIBRARY_PATH": "/usr/local/mpi/lib:/usr/local/mpi/lib64:/usr/local/anaconda/lib/:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}",
        "LIBRARY_PATH": "/usr/local/mpi/lib:/usr/local/mpi/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/local/anaconda/lib/:${LIBRARY_PATH}",
        "CPATH": "/usr/local/anaconda/include/:${CPATH}",
        "PKG_CONFIG_PATH": "/usr/local/anaconda/lib/pkgconfig:${PKG_CONFIG_PATH}"
    })

if "arm" in target_arch:
    Stage0 += environment(variables={
        "LD_LIBRARY_PATH": "/opt/arm/armpl-20.3.0_Generic-AArch64_Ubuntu-16.04_gcc_aarch64-linux/lib:${LD_LIBRARY_PATH}",
        "LIBRARY_PATH": "/opt/arm/armpl-20.3.0_Generic-AArch64_Ubuntu-16.04_gcc_aarch64-linux/lib:${LIBRARY_PATH}",
        "ARMPL": "/opt/arm/armpl-20.3.0_Generic-AArch64_Ubuntu-16.04_gcc_aarch64-linux"
    })
    arches = ["-march=armv8-a"]
    folders = ["arm"]
else:
    arches = [None, "-march=core-avx2", "-march=skylake-avx512"]
    folders = ["native", "haswell", "haswell/avx512_1"]

for i in range(len(arches)):
  directory = '/opt/bigdft/build/' + folders[i]
  Stage0 += raw(docker='USER root')
  Stage0 += yum(ospackages=['gsl-devel'])  # Changement pour utiliser yum
  Stage0 += workdir(directory=directory)
  Stage0 += shell(commands=['chown -R lsim:lsim .', 'chmod -R 777 .'])
  Stage0 += raw(docker='USER lsim')

  if arches[i] is not None:
    Stage0 += environment(variables={"BIGDFT_OPTFLAGS": arches[i]})

Stage0 += workdir(directory='/home/lsim')



#######
## Runtime image
#######

cuda_version = USERARG.get('cuda', '10.0')
repo = "nvidia/cuda"
if "arm" in target_arch:
    repo += "-arm64"

target_arch = USERARG.get('target_arch', 'x86_64')
if "arm" in target_arch:
    # Utilisez une image appropriée pour ARM sous CentOS
    image = 'nvidia/cuda:12.4.0-runtime-centos7'  # Exemple, ajustez selon la disponibilité
else:
    # Utilisez une image appropriée pour x86_64 sous CentOS
    image = 'nvidia/cuda:12.4.0-runtime-centos7'  # Exemple, ajustez selon la disponibilité

# Configuration de Stage1 avec la nouvelle image
Stage1.baseimage(image)
Stage1.name = 'runtime'
Stage1 += comment("Runtime stage", reformat=False)

import hpccm.config
hpccm.config.set_cpu_architecture(target_arch)

# Configuration des bibliothèques CUDA pour le runtime
if cuda_version >= StrictVersion('11.0'):
    cuvers = '-' + cuda_version[:-2].replace('.', '-')
    libs = ['libcublas' + cuvers, 'libcufft' + cuvers, 'cuda-cudart' + cuvers, 'cuda-nvtx' + cuvers]
else:
    cuvers = '-' + cuda_version.replace('.', '-')
    if cuda_version >= StrictVersion('10.1'):
        cublas = "libcublas10"
    else:
        cublas = 'cuda-cublas' + cuvers
    libs = [cublas, 'cuda-cufft' + cuvers, 'cuda-cudart' + cuvers, 'cuda-nvtx' + cuvers]

Stage1 += yum(ospackages=libs)

if "arm" not in target_arch:
    Stage1 += copy(_from="bigdft_build", src="/usr/local/anaconda", dest="/usr/local/anaconda")
    Stage1 += environment(variables={"LD_LIBRARY_PATH": "/usr/local/anaconda/lib/:${LD_LIBRARY_PATH}"})
    Stage1 += environment(variables={"LIBRARY_PATH": "/usr/local/anaconda:${LIBRARY_PATH}"})
    Stage1 += environment(variables={"PATH": "/usr/local/anaconda/bin/:${PATH}"})


## Compiler runtime (use upstream)
Stage1 += gnu().runtime()
tc = gnu().toolchain
tc.CUDA_HOME = '/usr/local/cuda'
Stage1 += environment(variables={'DEBIAN_FRONTEND': 'noninteractive'})

if "arm" in target_arch:
    # Utilisez les bibliothèques système Python sous CentOS
    ospack = [
      'python3', 'python3-Cython', 'python3-flake8', 'python3-ipykernel',
      'python3-ipython', 'python3-pip', 'jupyter-notebook', 'python3-matplotlib',
      'python3-six', 'python3-sphinx', 'python3-sphinx-bootstrap-theme',
      'python3-scipy', 'python3-numpy', 'python3-sphinx-rtd-theme', 'watchdog'
    ]
    Stage1 += yum(ospackages=ospack)

    # Créez des liens symboliques pour python et pip
    Stage1 += shell(commands=[
      'ln -s /usr/bin/python3 /usr/local/bin/python',
      'ln -s /usr/bin/pip3 /usr/local/bin/pip'
    ])

# Utilisez yum pour installer les paquets nécessaires
Stage1 += yum(ospackages=['ocl-icd-libopencl1', 'openbabel', 'flex', 'blas', 'lapack', 'pcre', 'openssh-clients', 'xorg-x11-server-Xorg', 'gsl'])

if mpi == "ompi":
    ## normal OFED
    Stage1 += ofed().runtime(_from='bigdft_build')
    mpi_version = USERARG.get('mpi_version', '3.0.0')
    mpi_lib = openmpi(infiniband=False, version=mpi_version, prefix="/usr/local/mpi")
    Stage1 += mpi_lib.runtime(_from='bigdft_build')
    Stage1 += environment(variables={"OMPI_MCA_btl_vader_single_copy_mechanism": "none",
                                     "OMPI_MCA_rmaps_base_mapping_policy": "slot",
                                     "OMPI_MCA_hwloc_base_binding_policy": "none",
                                     "OMPI_MCA_btl_openib_cuda_async_recv": "false",
                                     "OMPI_MCA_mpi_leave_pinned": "true",
                                     "OMPI_MCA_opal_warn_on_missing_libcuda": "false",
                                     "OMPI_MCA_rmaps_base_oversubscribe": "true"})
elif mpi in ["mvapich2", "mvapich"]:
    ## Mellanox OFED
    Stage1 += mlnx_ofed(version='5.0-2.1.8.0', oslabel='centos7').runtime(_from='bigdft_build')
    mpi_version = USERARG.get('mpi_version', '2.3')
    Stage1 += yum(ospackages=['libpciaccess', 'numactl', 'libgfortran'])
    Stage1 += copy(_from="bigdft_build", src="/usr/local/mpi", dest="/usr/local/mpi")
    Stage1 += environment(variables={"MV2_USE_GPUDIRECT_GDRCOPY": "0",
                                     "MV2_SMP_USE_CMA": "0",
                                     "MV2_ENABLE_AFFINITY": "0",
                                     "MV2_CPU_BINDING_POLICY": "scatter",
                                     "MV2_CPU_BINDING_LEVEL": "socket"})

if "arm" in target_arch:
  Stage1 += yum(ospackages=[
    'python3', 'python3-Cython', 'python3-flake8', 'python3-ipykernel',
    'python3-ipython', 'python3-pip', 'jupyter-notebook', 'python3-matplotlib',
    'python3-six', 'python3-sphinx', 'python3-sphinx-bootstrap-theme',
    'python3-scipy', 'python3-numpy', 'python3-sphinx-rtd-theme', 'watchdog'
  ])

  Stage1 += shell(commands=[
    'ln -s /usr/bin/python3 /usr/local/bin/python',
    'ln -s /usr/bin/pip3 /usr/local/bin/pip'
  ])

if mpi == 'impi':
  # Assurez-vous d'avoir une stratégie d'installation valide pour Intel MPI sous CentOS.
  mpi_lib = intel_mpi(eula=True)  # Ajustez selon la disponibilité du package ou installez manuellement
  Stage1 += mpi_lib.runtime(_from='bigdft_build')

Stage1 += shell(commands=['mkdir -p /etc/OpenCL/vendors',
                          'echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd'])

Stage1 += environment(variables={
  'NVIDIA_VISIBLE_DEVICES': 'all',
  'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility'
})

Stage1 += copy(_from="bigdft_build", src="/usr/local/cuda/lib64/stubs/libcuda.so", dest="/usr/local/lib/libcuda.so.1")
Stage1 += copy(_from="bigdft_build", src="/usr/local/cuda/lib64/stubs/libnvidia-ml.so", dest="/usr/local/lib/libnvidia-ml.so.1")

Stage1 += copy(_from="bigdft_build", src="/usr/local/bigdft", dest="/usr/local/bigdft")
Stage1 += copy(_from="bigdft_build", src="/docker", dest="/docker")
Stage1 += shell(commands=['chmod -R 777 /docker'])

Stage1 += environment(variables={"XDG_CACHE_HOME": "/root/.cache/"})
Stage1 += shell(commands=['MPLBACKEND=Agg python -c "import matplotlib.pyplot"'])

Stage1 += raw(docker='EXPOSE 8888')
Stage1 += raw(docker='CMD jupyter-notebook --ip=0.0.0.0 --allow-root --NotebookApp.token=bigdft --no-browser', singularity='%runscript\n jupyter-notebook --ip=0.0.0.0 --allow-root --NotebookApp.token=bigdft --no-browser')

if "arm" in target_arch:
  Stage1 += copy(_from="bigdft_build", src="/opt/arm/armpl-20.3.0_Generic-AArch64_Ubuntu-16.04_gcc_aarch64-linux", dest="/opt/arm/armpl-20.3.0_Generic-AArch64_Ubuntu-16.04_gcc_aarch64-linux")
  Stage1 += copy(_from="bigdft_build", src="/opt/arm/armpl-20.3.0_ThunderX2CN99_Ubuntu-16.04_gcc_aarch64-linux", dest="/opt/arm/armpl-20.3.0_ThunderX2CN99_Ubuntu-16.04_gcc_aarch64-linux")
  Stage1 += environment(variables={"LD_LIBRARY_PATH": "/opt/arm/armpl-20.3.0_Generic-AArch64_Ubuntu-16.04_gcc_aarch64-linux/lib:${LD_LIBRARY_PATH}"})

Stage1 += shell(commands=["rm -rf $(find / | perl -ne 'print if /[^[:ascii:]]/')"])

Stage1 += shell(commands=[
  'echo "/usr/local/bigdft/lib" > /etc/ld.so.conf.d/bigdft.conf',
  'echo "/usr/local/anaconda/lib" >> /etc/ld.so.conf.d/conda.conf',
  'ldconfig'
])

# Ensure that environment variables are set correctly for all architectures
if "arm" not in target_arch:
    Stage0 += environment(variables={
        "LD_LIBRARY_PATH": "/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}",
        "LIBRARY_PATH": "/usr/local/cuda/lib64:${LIBRARY_PATH}",
        "PYTHON": "python"
    })

# Adding environment variables universally for all architectures
# En supposant que cuda_gencodes est une liste de chaînes de caractères
cuda_gencodes_str = ' '.join(cuda_gencodes)
Stage0 += environment(variables={"CUDA_GENCODES": '"' + cuda_gencodes_str + '"'})

Stage0 += shell(commands=['sed -i "s/__shfl_down(/__shfl_down_sync(0xFFFFFFFF,/g" ../psolver/src/cufft.cu'])

# Configure user and environment for Stage1
Stage1 += shell(commands=['useradd -ms /bin/bash bigdft'])
Stage1 += raw(docker='USER bigdft')

if use_mkl == "yes":
    Stage1 += environment(variables={"MKLROOT": "/usr/local/anaconda"})

# Set up comprehensive environment paths for Stage1
Stage1 += environment(variables={
    "LD_LIBRARY_PATH": "/usr/local/anaconda/lib:${LD_LIBRARY_PATH}",
    "LIBRARY_PATH": "/usr/local/anaconda/lib:${LIBRARY_PATH}",
    "CPATH": "/usr/local/anaconda/include:${CPATH}",
    "PKG_CONFIG_PATH": "/usr/local/anaconda/lib/pkgconfig:${PKG_CONFIG_PATH}",
    "PATH": "/usr/local/anaconda/bin:${PATH}",
    "PATH": "/usr/local/mpi/bin:/usr/local/bigdft/bin:${PATH}",
    "LD_LIBRARY_PATH": "/usr/local/mpi/lib:/usr/local/mpi/lib64:/usr/local/bigdft/lib:${LD_LIBRARY_PATH}",
    "PYTHONPATH": "/usr/local/bigdft/lib/python3.6/site-packages:/usr/local/bigdft/lib/python3.7/site-packages:/usr/local/bigdft/lib/python3.8/site-packages:${PYTHONPATH}",
    "PKG_CONFIG_PATH": "/usr/local/bigdft/lib/pkgconfig:${PKG_CONFIG_PATH}",
    "CHESS_ROOT": "/usr/local/bigdft/bin",
    "BIGDFT_ROOT": "/usr/local/bigdft/bin",
    "GI_TYPELIB_PATH": "/usr/local/bigdft/lib/girepository-1.0:${GI_TYPELIB_PATH}",
    "XDG_CACHE_HOME": "/home/bigdft/.cache/"
})

Stage1 += shell(commands=['MPLBACKEND=Agg python -c "import matplotlib.pyplot"'])
