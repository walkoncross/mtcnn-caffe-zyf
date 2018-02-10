#/usr/bin/evn bash
splits=1
num_gpu=1
# declare -x GLOG_minloglevel=2

if [ $# -gt 0 ]; then
    splits=$1
fi

if [ $# -gt 1 ]; then
    num_gpu=$2
fi

echo 'splits: ' $splits
echo 'num_gpu: ' $num_gpu

if [ $splits -lt 2 ]; then
    nohup python batch_mtcnn_align_crop_112x112_for_idcard1m.py \
        --image-list=/disk2/data/FACE/face-idcard-1M/face-idcard-1M-image-list.txt \
        --image-root-dir=/disk2/data/FACE/face-idcard-1M/ori \
        --rect-root-dir=/disk2/data/FACE/face-idcard-1M/result \
        --mtcnn-model-dir=../../model \
        --save-dir=/disk2/data/FACE/face-idcard-1M/face-idcard-1M-mtcnn-aligned-112x112 \
        --gpu-id=0 \
        --nsplits=1 \
        --split-id=0 > ./process-log.txt &
else

    # run #splits_per_gpu splits on each gpu
    let last_gpu_id=num_gpu-1
    let splits_per_gpu=splits/num_gpu

    for gpu_id in `seq 0 $last_gpu_id`; do

        let start=splits_per_gpu*gpu_id
        let end=start+splits_per_gpu-1

        echo 'start split id: ' $start
        echo 'end split id: ' $end

        for i in `seq $start $end`; do
            echo 'loop-'$i
            nohup python batch_mtcnn_align_crop_112x112_for_idcard1m.py \
                --image-list=/disk2/data/FACE/face-idcard-1M/face-idcard-1M-image-list.txt \
                --image-root-dir=/disk2/data/FACE/face-idcard-1M/ori \
                --rect-root-dir=/disk2/data/FACE/face-idcard-1M/result \
                --mtcnn-model-dir=../../model \
                --save-dir=/disk2/data/FACE/face-idcard-1M/face-idcard-1M-mtcnn-aligned-112x112 \
                --gpu-id=$gpu_id \
                --nsplits=$splits \
                --split-id=$i   > '/workspace/process-log-'$splits'-'$i'.txt' &
            echo 'pause 1 seconds to wait for initialization'
            sleep 1
        done
    done
fi