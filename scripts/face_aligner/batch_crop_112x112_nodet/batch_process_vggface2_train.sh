splits=128

let cnt=splits-1
for i in `seq 0 $cnt`;
do
	echo 'loop-'$i
	nohup python ./batch_crop_112x112_nodet.py \
		--image-list=/disk2/data/FACE/vggface2/meta/train_list.txt \
		--image-root-dir=/disk2/data/FACE/vggface2/train/ \
		--rect-root-dir=/disk2/data/FACE/vggface2/vggface2_train_aligned/face_rects \
		--save-dir=/disk2/data/FACE/vggface2/vggface2_train_aligned_112x112 \
		--nsplits=$splits \
		--split-id=$i \
		> '/workspace/process-log-'$splits'-'$i'-vggface2-train.txt' &
done

