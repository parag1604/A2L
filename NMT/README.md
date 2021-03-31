# A2L NMT codes



## Environment Setup

### Anaconda environment setup command:

    conda create -n <A2L_NMT> tensorflow-gpu=2 matplotlib scikit-learn sacrebleu tensorflow-datasets -c conda-forge -c anaconda

### Subword models
1. download Europarl dataset (English to spanish) and save it at 'global_data/Dataset/Europarl'
2. go to 'dev' folder and execute:

        python generate_subword.py

### Word token models (subword prerequisite)
go to 'dev' folder and execute:

    python preproc.py

### Iso-siamese baseline
1. download glove embeddings
2. go to 'dev' folder and execute:

        python generate_glove.py <path/to/glove_300d_embeddings>




## Directory structure

The codes have been grouped based on the AL strategies
Within each AL strategy folder, the various baselines are present.

So, to execute, for example, A2L training over Least Confidence strategy:
  go to 'LC/A2L'





## Code Execution

### To start training
Go to appropriate folder and execute:

    python master.py





## Complete list of pacakges and versions (in case of inconsistencies)

    # Name                    Version                   Build  Channel
    _libgcc_mutex             0.1                        main
    _tflow_select             2.1.0                       gpu
    absl-py                   0.9.0                    py37_0
    astor                     0.8.1                    py37_0
    astunparse                1.6.3                      py_0
    attrs                     20.1.0                     py_0
    blas                      1.0                         mkl
    blinker                   1.4                      py37_0
    brotlipy                  0.7.0           py37h7b6447c_1000
    c-ares                    1.15.0            h7b6447c_1001
    ca-certificates           2020.7.22                     0
    cachetools                4.1.1                      py_0
    certifi                   2020.6.20                py37_0
    cffi                      1.14.2           py37he30daa8_0
    chardet                   3.0.4                 py37_1003
    click                     7.1.2                      py_0
    cryptography              3.0              py37h1ba5d50_0
    cudatoolkit               10.0.130                      0
    cudnn                     7.6.5                cuda10.0_0
    cupti                     10.0.130                      0
    cycler                    0.10.0                   py37_0
    dbus                      1.13.16              hb2f20db_0
    dill                      0.3.2                      py_0
    expat                     2.2.9                he6710b0_2
    fontconfig                2.13.0               h9420a91_0
    freetype                  2.10.2               h5ab3b9f_0
    future                    0.18.2                   py37_1
    gast                      0.2.2                    py37_0
    glib                      2.65.0               h3eb4bd4_0
    google-auth               1.20.1                     py_0
    google-auth-oauthlib      0.4.1                      py_2
    google-pasta              0.2.0                      py_0
    googleapis-common-protos  1.51.0                   py37_2
    grpcio                    1.31.0           py37hf8bcb03_0
    gst-plugins-base          1.14.0               hbbd80ab_1
    gstreamer                 1.14.0               hb31296c_0
    h5py                      2.10.0           py37hd6299e0_1
    hdf5                      1.10.6               hb1b8bf9_0
    icu                       58.2                 he6710b0_3
    idna                      2.10                       py_0
    importlib-metadata        1.7.0                    py37_0
    intel-openmp              2020.2                      254
    joblib                    0.16.0                     py_0
    jpeg                      9b                   h024ee3a_2
    keras-applications        1.0.8                      py_1
    keras-preprocessing       1.1.0                      py_1
    kiwisolver                1.2.0            py37hfd86e86_0
    lcms2                     2.11                 h396b838_0
    ld_impl_linux-64          2.33.1               h53a641e_7
    libedit                   3.1.20191231         h14c3975_1
    libffi                    3.3                  he6710b0_2
    libgcc-ng                 9.1.0                hdf63c60_0
    libgfortran-ng            7.3.0                hdf63c60_0
    libpng                    1.6.37               hbc83047_0
    libprotobuf               3.12.4               hd408876_0
    libstdcxx-ng              9.1.0                hdf63c60_0
    libtiff                   4.1.0                h2733197_1
    libuuid                   1.0.3                h1bed415_2
    libxcb                    1.14                 h7b6447c_0
    libxml2                   2.9.10               he19cac6_1
    lz4-c                     1.9.2                he6710b0_1
    markdown                  3.2.2                    py37_0
    matplotlib                3.3.1                         0
    matplotlib-base           3.3.1            py37h817c723_0
    mkl                       2020.2                      256
    mkl-service               2.3.0            py37he904b0f_0
    mkl_fft                   1.1.0            py37h23d657b_0
    mkl_random                1.1.1            py37h0573a6f_0
    ncurses                   6.2                  he6710b0_1
    nltk                      3.5                        py_0
    numpy                     1.19.1           py37hbc911f0_0
    numpy-base                1.19.1           py37hfa32c7d_0
    oauthlib                  3.1.0                      py_0
    olefile                   0.46                       py_0
    openssl                   1.1.1h               h7b6447c_0
    opt_einsum                3.1.0                      py_0
    pandas                    1.1.1            py37he6710b0_0
    pcre                      8.44                 he6710b0_0
    pillow                    7.2.0            py37hb39fc2d_0
    pip                       20.2.2                   py37_0
    portalocker               2.0.0                    pypi_0    pypi
    promise                   2.3                      py37_0
    protobuf                  3.12.4           py37he6710b0_0
    psutil                    5.7.2            py37h7b6447c_0
    pyasn1                    0.4.8                      py_0
    pyasn1-modules            0.2.7                      py_0
    pycparser                 2.20                       py_2
    pyjwt                     1.7.1                    py37_0
    pyopenssl                 19.1.0                     py_1
    pyparsing                 2.4.7                      py_0
    pyqt                      5.9.2            py37h05f1152_2
    pysocks                   1.7.1                    py37_1
    python                    3.7.7                hcff3b4d_5
    python-dateutil           2.8.1                      py_0
    pytz                      2020.1                     py_0
    qt                        5.9.7                h5867ecd_1
    readline                  8.0                  h7b6447c_0
    regex                     2020.7.14        py37h7b6447c_0
    requests                  2.24.0                     py_0
    requests-oauthlib         1.3.0                      py_0
    rsa                       4.6                        py_0
    sacrebleu                 1.4.13                   pypi_0    pypi
    scikit-learn              0.23.2           py37h0573a6f_0
    scipy                     1.5.2            py37h0b6359f_0
    setuptools                49.6.0                   py37_0
    sip                       4.19.8           py37hf484d3e_0
    six                       1.15.0                     py_0
    sqlite                    3.33.0               h62c20be_0
    tensorboard               2.2.1              pyh532a8cf_0
    tensorboard-plugin-wit    1.6.0                      py_0
    tensorflow                2.0.0           gpu_py37h768510d_0
    tensorflow-base           2.0.0           gpu_py37h0ec5d1f_0
    tensorflow-datasets       1.2.0                    py37_0
    tensorflow-estimator      2.0.0              pyh2649769_0
    tensorflow-gpu            2.0.0                h0d30ee6_0
    tensorflow-metadata       0.14.0             pyhe6710b0_1
    termcolor                 1.1.0                    py37_1
    threadpoolctl             2.1.0              pyh5ca1d4c_0
    tk                        8.6.10               hbc83047_0
    tornado                   6.0.4            py37h7b6447c_1
    tqdm                      4.48.2                     py_0
    urllib3                   1.25.10                    py_0
    werkzeug                  0.16.1                     py_0
    wheel                     0.35.1                     py_0
    wrapt                     1.12.1           py37h7b6447c_1
    xz                        5.2.5                h7b6447c_0
    zipp                      3.1.0                      py_0
    zlib                      1.2.11               h7b6447c_3
    zstd                      1.4.5                h9ceee32_0
