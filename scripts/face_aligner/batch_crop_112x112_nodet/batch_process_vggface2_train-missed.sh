splits=32
save_dir=/disk2/data/FACE/vggface2/vggface2_train_aligned_112x112

mkdir $save_dir

# let cnt=splits-1
# for i in `seq 0 $cnt`;
list=(4 5 8 10 11 12 14 15)
for i in ${list[@]}
do
	echo 'loop-'$i
	nohup python ./batch_crop_112x112_nodet.py \
		--image-list=/disk2/data/FACE/vggface2/meta/train_list.txt \
		--image-root-dir=/disk2/data/FACE/vggface2/train/ \
		--rect-root-dir=/disk2/data/FACE/vggface2/vggface2_train_aligned/face_rects \
		--save-dir=$save_dir \
		--nsplits=$splits \
		--split-id=$i \
		> '/workspace/process-log-'$splits'-'$i'-vggface2-train.txt' &
done

