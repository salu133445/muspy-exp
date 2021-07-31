for model in LSTM RNN_TANH GRU Transformer # TransformerL
do
  CUDA_VISIBLE_DEVICES=3 python combined_dataset.py --cuda --n_jobs 8 --model $model
done
