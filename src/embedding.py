from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))

def get_T5_model(model_dir, transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"):
    # print("Loading: {}".format(transformer_link))
    if model_dir is not None:
        print("##########################")
        print("Loading cached model from: {}".format(model_dir))
        print("##########################")
        model = T5EncoderModel.from_pretrained(model_dir)
        tokenizer = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False,legacy=False)
    else:
        print("##########################")
        print(f"model_dir is empty. Downloading model and tokenizer from: {transformer_link}")
        print("##########################")
        model = T5EncoderModel.from_pretrained(transformer_link)
        tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False, legacy=False)
    # only cast to full-precision if no GPU is available
    if device==torch.device("cpu"):
        print("Casting model to full precision for running on CPU ...")
        model.to(torch.float32)

    model = model.to(device)
    model = model.eval()
    
    return model, tokenizer

def extract_fea(seqs):
    '''
    extract hiddendim 
    '''
    sequence_examples=[]
    sequence_examples.append(seqs)
    # sequence_examples = ["PRTEINO", "SEQWENCE"]
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding_rpr = model(input_ids=input_ids, attention_mask=attention_mask)
    emb_0 = embedding_rpr.last_hidden_state[0, :] 
    
    return emb_0.detach().cpu().numpy()

def emdding(src,tgt):
    '''
    Preparing a dataframe(src) containing name and seq columns and result will be save in tgt_dir
    Considering the lack of viability, the protein is executed one by one
    Input:
        src = '/home/.../data.csv'
        tgt_dir = '/home/.../'
    Output:
        save embedding results in tgt_dir
    '''

    data = pd.read_csv(src)
    namelist=list(data['name'])
    seqlist=list(data['seq'])
    for i in range(len(data)):
        out = extract_fea(seqlist[i])
        dir=str(tgt)+namelist[i]+'.npy'
        np.save(dir,out)

def create_arg_parser():

    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument( '-i', '--input', required=True, type=str,
                    help='A path to a csv-formatted file containing name and protein sequence.')

    # Optional positional argument
    parser.add_argument( '-o', '--output', required=True, type=str, 
                    help='A path for saving the created embeddings as NumPy npz file.')

    # Required positional argument
    parser.add_argument('--model', required=False, type=str,
                    default=None,
                    help='A path to a directory holding the checkpoint for a pre-trained model' )

    return parser

def main():
    global model
    global tokenizer

    parser = create_arg_parser()
    args = parser.parse_args()

    dataframe   = Path( args.input )
    output   = Path( args.output)
    model_dir  = Path( args.model ) if args.model is not None else None

    model, tokenizer = get_T5_model(model_dir)

    emdding(dataframe,output)

if __name__ == '__main__':
    main()