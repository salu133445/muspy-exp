for model in LSTM RNN_TANH GRU Transformer
do
  CUDA_VISIBLE_DEVICES=3 python combined_vs_combined.py --cuda --trials 1000 --n_jobs 8 -m $model
done
