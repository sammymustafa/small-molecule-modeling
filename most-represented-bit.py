from setup import * 

def most_represented_bit(arr):
    return np.argmax(arr)

# Change num_item to any value < count for that bit
num_item = 1
bi = ao.GetBitInfoMap()
idx = most_represented_bit(arr)
Chem.Draw.DrawMorganBit(mol, idx, bi, whichExample=num_item)