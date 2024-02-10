import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
import glob

def main(args):
    # coords = torch.load(args.input_dir)
    # coords = coords.numpy()
    # seq = 'A' * coords.shape[0]
    preprocess(args.input_dir, args.output_dir)


def preprocess(input_dir, output_dir, verbose=True):
    """
    Convert coordinate files to pdb files.
    """

    # create output directory
    pdbs_dir = os.path.join(output_dir, 'pdbs')
    if not os.path.exists(pdbs_dir):
        os.mkdir(pdbs_dir)
    
    
    # detect file type:
    file_type = detect_file_type(input_dir)
    # process
    for filepath in tqdm(
        glob.glob(os.path.join(input_dir, f'*.{file_type}')),
        desc='Preprocessing', disable=not verbose
    ):
        
        domain_name = filepath.split('/')[-1].split('.')[0]
        pdb_filepath = os.path.join(pdbs_dir, f'{domain_name}.pdb')
        if file_type == 'pt':
            # coords = torch.load(filepath)[0].detach().numpy()
            coords = torch.load(filepath).numpy()
        if file_type == 'npy':
            coords = np.loadtxt(filepath, delimiter=',')
        coords = np.around(coords, decimals=2)
        if np.isnan(coords).any():
            print(f'Error: {domain_name}')
        else:
            seq = 'A' * coords.shape[0]
            save_as_pdb(seq, coords, pdb_filepath)

    return pdbs_dir

def detect_file_type(dir):
    file_list = os.listdir(dir)

    for file in file_list:
        if file.endswith('.pt'):
            file_type = 'pt'
            break
        if file.endswith('.npy'):
            file_type = 'npy'
    return file_type

restype_1to3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP',
    'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY',
    'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
    'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER',
    'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}

def save_as_pdb(seq, coords, filename, ca_only=True):
    
    def pad_left(string, length):
        assert len(string) <= length
        return ' ' * (length - len(string)) + string
    
    def pad_right(string, length):
        assert len(string) <= length
        return string + ' ' * (length - len(string))
    
    atom_list = ['N', 'CA', 'C', 'O']
    with open(filename, 'w') as file:
        for i in range(coords.shape[0]):
            atom = 'CA' if ca_only else pad_right(atom_list[i%4], 2)
            atom_idx = i + 1
            residue_idx = i + 1 if ca_only else i // 4 + 1
            residue_name = restype_1to3[seq.upper()[residue_idx-1]]
            line = 'ATOM  ' + pad_left(str(atom_idx), 5) + '  ' + pad_right(atom, 3) + ' ' + \
                residue_name + ' ' + 'A' + pad_left(str(residue_idx), 4) + ' ' + '   ' + \
                pad_left(str(coords[i][0]), 8) + pad_left(str(coords[i][1]), 8) + pad_left(str(coords[i][2]), 8) + \
                '     ' + '      ' + '   ' + '  ' + pad_left(atom[0], 2)
            file.write(line + '\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Input directory', required=True)
    parser.add_argument('--output_dir', type=str, help='Output directory', required=True)
    args = parser.parse_args()
    main(args)