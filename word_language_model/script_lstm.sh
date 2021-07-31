for data in lmd wikifonia nes jsb maestro hymnal hymnal_tune music21 music21jsb nmd essen
do
  CUDA_VISIBLE_DEVICES=1 python main.py --cuda --n_jobs 8 --data $data
done
