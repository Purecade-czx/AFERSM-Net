import torch
if __name__ == '__main__':
    a = torch.tensor([ [ [1,2,3,2,1],
                         [2,5,6,3,6] ,
                        [2,1,5,9,8],
                         [4,6,8,1,1] ] ],dtype=torch.float)
    print('the a is:',a.shape)
    a = torch.reshape(a,(-1,5))
    print(a)
    print('the a is:',a.shape)
