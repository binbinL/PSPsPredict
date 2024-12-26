# PSPsPredict

## Introduction
PSPsPredict is a tool for predicting the probability of liquid-liquid phase separation of proteins.
## info

Considering the size of the video memory, the data generation script needs to be executed in advance maybe.

Execute embedding.py script, generate proteins embedding.

```python
python embedding.py -i .../data/xxx.csv -o .../emddata/ --model .../prot_t5_xl_half_uniref50-enc
```

- -i: A path to a csv-formatted file containing name and protein sequence.
- -o: A path for saving the created embeddings as NumPy npz file.
- -- model: A path to a directory holding the checkpoint for a pre-trained model

Execute Predict.py script, Predict the result.

```python
python Predict.py -i data/demo.csv -src .../emddata/ -o .../data/demo_res.csv
```

- -i: A path to a csv-formatted file containing name and protein sequence.
- -src: A directory contain ProtT5 embeddings.
- -o: A path to a csv-formatted file containing name and result.
