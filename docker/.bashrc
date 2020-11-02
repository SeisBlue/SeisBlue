# Add SeisNN into Python PATH
export PYTHONPATH=$PYTHONPATH:/SeisNN

# Add Cuda related file
export PATH=/usr/local/cuda-10.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:/usr/local/cuda-10.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda-10.1/lib64/stubs:$LIBRARY_PATH
export CUPIT_LIB_PATH=/usr/local/cuda-10.1/extras/CUPTI/lib64:$CUPIT_LIB_PATH
export LD_LIBRARY_PATH=$CUPIT_LIB_PATH:$LD_LIBRARY_PATH