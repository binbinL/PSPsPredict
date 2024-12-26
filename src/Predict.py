import torch # type: ignore
from tqdm import tqdm # type: ignore
import pandas as pd # type: ignore
from torch import nn # type: ignore
from torch.utils.data import DataLoader,Dataset # type: ignore
from MyNet import MyModel # type: ignore
from config import dropout,d_model,n_class,vocab_size,nlayers,nhead,dim_feedforward,d_embedding,kmers # type: ignore
import numpy as np
import argparse

nt_int={'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 21}

def seq2int(nt_sequence,target_length=1024):
    int_sequence=[]
    for nt in nt_sequence:
        nt=nt.upper()
        if nt in nt_int:
            int_sequence.append(nt_int[nt])
    int_sequence=np.asarray(int_sequence,dtype='int32')
    if len(int_sequence) > target_length:
        int_sequence = int_sequence[:target_length]
    if len(int_sequence) < target_length:
        int_sequence=np.pad(int_sequence,(0,target_length-len(int_sequence)),constant_values=0)
    return int_sequence

class MyDataset(Dataset):
    def __init__(self, file, src_root):
        self.embedding, self.seq ,self.name= self.read_file(file,src_root)
    def loadembedding(self,name,src_root):
        fea = np.load( src_root + name + ".npy")
        fea = fea[1:-1]
        mean_fea = np.mean(fea, axis=0)
        return mean_fea

    def read_file(self,file_path,src_root):
        name = []
        embedding = []
        seqs = []
        df_all = pd.read_csv(file_path)
        for i in range(len(df_all)):
            name.append(df_all['name'][i])
            embedding.append(self.loadembedding(df_all['name'][i],src_root))
            seqs.append(seq2int(df_all['seq'][i]))
        return embedding,seqs,name

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, index):
        name =self.name[index]
        embedding=self.embedding[index]
        seq = self.seq[index]
        return embedding, seq, name
    
def evaluate(model, loader):
    model.eval()
    preds_list = []
    emds = []
    names_list = []
    with torch.no_grad():
        for data, seq, name in tqdm(loader): 
            data = data.to(device)
            seq = seq.to(device)
            outputs,emd = model(data,seq)

            probabilities = nn.functional.softmax(outputs, dim=1)
            preds = probabilities[:, 1].detach().cpu().numpy()
            preds_list.extend(preds)
            names_list.extend(name)
            emds.extend(emd.detach().cpu().numpy())
    
    return preds_list,names_list


def predict(src,src_root,tgt):
    Data = MyDataset(src,src_root)
    pDataLoader = DataLoader(Data, batch_size=32)
    df = pd.DataFrame()
    preds_list,names_list = evaluate(model, pDataLoader)
    df['name'] = names_list
    df['preds'] = preds_list
    df.to_csv(tgt, index=False)
    print('Done!')



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = MyModel(d_embedding,d_model,dropout,n_class,vocab_size,nlayers,nhead,dim_feedforward,kmers)
state_dict = torch.load('/data/lwb/WorkSpace/PSPs_Predict/model/best-0.901.pth')
model.load_state_dict(state_dict)
model = model.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_file',
        type=str,
        required=True,
        help="A path to a csv-formatted file containing name and protein sequence." 
    )
    parser.add_argument(
        '-src', '--ProtT5_directory',
        type=str,
        required=True,
        help="ProtT5 directory"
    )
    parser.add_argument(
        '-o', '--outfile',
        type=str,
        required=True,
        help="A path to a csv-formatted file containing name and result."
    )
    args = parser.parse_args()
    predict(args.input_file, args.ProtT5_directory, args.outfile)

if __name__ == '__main__':
    # src = '/data/lwb/WorkSpace/PSPs/Net/data/test.csv'
    # src_root = '...' # .npz files folder
    # tgt = '/data/lwb/WorkSpace/PSPs/Net/data/test_res.csv'
    main()

