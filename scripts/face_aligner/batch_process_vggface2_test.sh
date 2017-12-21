splits=32

let cnt=splits-1
for i in `seq 0 $cnt`;
do
	echo 'loop-'$i
	nohup ./mtcnn_align_crop_96x112_for_vggface2_test.py $splits $i > '/workspace/vgg2-test-process-log-'$splits'-'$i'.txt' &
done

