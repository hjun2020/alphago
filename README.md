# Alpahgo applied to OMOK #
* The goal of the project is to apply alphago system to a simpler board game called OMOK.
* Currently, most codes written are from Google resesarch's alpha zero implementation, https://github.com/tensorflow/minigo
* However, their codes are written using Tensorflow 1. 
* I converted it to Tensorflow 2.
* The training part and OMOK implementation has not been finished yet.
* However, you can try selfplay part at this point:
- python3 selfplay.py \           
  --load_file=outputs/models/$MODEL_NAME \
  --num_readouts 10 \               
  --verbose 3 \                                       
  --selfplay_dir=outputs/data/selfplay \
  --holdout_dir=outputs/data/holdout \

* Note that current selfplay doesn't utilize any trained neural net so that it is just random movements.
* The selfplay generates move histories of players and this will be the dataset of future training(To be done) 