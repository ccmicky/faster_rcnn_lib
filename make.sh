CUDA_PATH=/usr/local/cuda/
CXXFLAGS=''
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

if [[ "$OSTYPE" =~ ^darwin ]]; then
	CXXFLAGS+='-undefined dynamic_lookup'
fi

cd roi_pooling_layer


g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc -D_GLIBCXX_USE_CXX11_ABI=0 \
                -I $TF_INC -I $TF_INC/external/nsync/public/ -L $TF_LIB -ltensorflow_framework \
                -fPIC $CXXFLAGS

cd ..

#cd feature_extrapolating_layer

#nvcc -std=c++11 -c -o feature_extrapolating_op.cu.o feature_extrapolating_op_gpu.cu.cc \
#	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

#g++ -std=c++11 -shared -o feature_extrapolating.so feature_extrapolating_op.cc \
#	feature_extrapolating_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64
#cd ..
