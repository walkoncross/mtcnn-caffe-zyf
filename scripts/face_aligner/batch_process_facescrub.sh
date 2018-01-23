#/usr/bin/evn bash
declare -x GLOG_minloglevel=2
splits=32


#let cnt=splits-1
start=0

let end=start+31

for i in `seq $start $end`;
do
        echo 'loop-'$i
        nohup ./mtcnn_align_crop_96x112_for_facescrub.py $splits $i > '/workspace/process-log-'$splits'-'$i'.txt' &
        echo 'pause 1 seconds to wait for initialzation'
        sleep 1
done
