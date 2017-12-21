splits=96

#let cnt=splits-1
start=32

let end=start+31

for i in `seq $start $end`;
do
	echo 'loop-'$i
	nohup ./mtcnn_align_crop_96x112_for_vggface2_train.py $splits $i > '/workspace/process-log-'$splits'-'$i'.txt' &
done

