# Neural Machine Translation TF

Click on the badge to open code notebook on Google colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SuXGz17r9JrQVcE55E2PvWT8Ek3YGDAz)
This is the repo associated with the Neural Machine Translation with TensorFlow blog on Paperspace. Checkpoint out the tutorial on Paperspace to learn more.

Training command:
python train.py --epochs=EPOCHS --batch_size=BATCH_SIZE --log_every=LOG_EVERY --buffer_size=BUFFER_SIZE

eg. python train.py --epochs=10 --batch_size=16 --log_every=50 --buffer_size=100000

Translate command:
python translate.py --sentence=SENTENCE --max_length=MAX_LENGTH

eg. python translate.py --sentence="quand tu vas au marche ?" --max_length=10
*****remember to enclose input sentence in quotation marks.*****
