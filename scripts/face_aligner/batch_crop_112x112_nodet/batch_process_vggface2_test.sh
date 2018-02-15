splits=32
save_dir=/disk2/data/FACE/vggface2/vggface2_test_aligned_112x112

mkdir $save_dir

let cnt=splits-1
for i in `seq 0 $cnt`;
do
	echo 'loop-'$i
	nohup python ./batch_crop_112x112_nodet.py \
		--image-list=/disk2/data/FACE/vggface2/meta/test_list.txt \
		--image-root-dir=/disk2/data/FACE/vggface2/test/ \
		--rect-root-dir=/disk2/data/FACE/vggface2/vggface2_test_aligned/face_rects \
		--save-dir=$save_dir \
		--nsplits=$splits \
		--split-id=$i \
		> '/workspace/process-log-'$splits'-'$i'-vggface2-test.txt' &
done

