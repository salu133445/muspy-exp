for data in music21jsb nmd essen lmd wikifonia nes jsb maestro hymnal hymnal_tune music21
do
  CUDA_VISIBLE_DEVICES=1 python main.py --cuda --n_jobs 8 --model GRU --data $data
done
