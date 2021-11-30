#!/bin/bash

cd ../..
dset=${1}
cfg=${2}
gpus=${3}
method=${4}

exp_name=${dset}_${cfg}

out_dir=/data/home/xxx/neurocomputing/exps/ckpt/${exp_name}
if [ -d ${out_dir} ]
then
  rm -rf ${out_dir}
fi
mkdir -p ${out_dir}

if [ ${dset} == Office-31 ]; then
  cfgs=(a2d a2w d2a d2w w2a w2d)
elif [ ${dset} == Office-home ]; then
  cfgs=(a2c a2p a2r c2a c2p c2r p2a p2c p2r r2a r2c r2p)
elif [ ${dset} == VisDA-2017 ]; then
  cfgs=(t2v)
elif [ ${dset} == ImageCLEF ]; then
  cfgs=(c2i cip i2c i2p p2c p2i)
fi

for cfg in ${cfgs[*]}; do
  exp_name=${dset}_${cfg}
  out_dir=/data/home/xxx/neurocomputing/exps/ckpt/${exp_name}
  CUDA_VISIBLE_DEVICES=${gpus} python ./experiments/tools/train.py --cfg ./experiments/config/${dset}/${method}/${cfg}.yaml --method ${method} --exp_name ${exp_name} 2>&1 | tee ${out_dir}/log.txt
done
