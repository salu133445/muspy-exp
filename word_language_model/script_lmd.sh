for model in LSTM RNN_TANH GRU Transformer #TransformerL
do
  CUDA_VISIBLE_DEVICES=3 python main.py --cuda --n_jobs 8 --data lmd --model $model
done
