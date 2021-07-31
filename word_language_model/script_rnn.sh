for data in jsb maestro hymnal hymnal_tune music21 music21jsb nmd essen lmd wikifonia nes
do
  CUDA_VISIBLE_DEVICES=1 python main.py --cuda --n_jobs 8 --model RNN_TANH --data $data
done
