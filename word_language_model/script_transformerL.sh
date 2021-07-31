for data in hymnal_tune music21 music21jsb nmd essen lmd wikifonia nes jsb maestro hymnal
do
  CUDA_VISIBLE_DEVICES=1 python main.py --cuda --n_jobs 16 --model TransformerL --data $data
done
