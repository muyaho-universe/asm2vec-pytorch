import torch
import torch.nn as nn

bce, sigmoid, softmax = nn.BCELoss(), nn.Sigmoid(), nn.Softmax(dim=1)

class ASM2VEC(nn.Module):
    def __init__(self, vocab_size, function_size, embedding_size):
        super(ASM2VEC, self).__init__()
        self.embeddings   = nn.Embedding(vocab_size, embedding_size, _weight=torch.zeros(vocab_size, embedding_size))
        self.embeddings_f = nn.Embedding(function_size, 2 * embedding_size, _weight=(torch.rand(function_size, 2 * embedding_size)-0.5)/embedding_size/2)
        self.embeddings_r = nn.Embedding(vocab_size, 2 * embedding_size, _weight=(torch.rand(vocab_size, 2 * embedding_size)-0.5)/embedding_size/2)

    def update(self, function_size_new, vocab_size_new):
        device = self.embeddings.weight.device
        vocab_size, function_size, embedding_size = self.embeddings.num_embeddings, self.embeddings_f.num_embeddings, self.embeddings.embedding_dim
        if vocab_size_new != vocab_size:
            weight = torch.cat([self.embeddings.weight, torch.zeros(vocab_size_new - vocab_size, embedding_size).to(device)])
            self.embeddings = nn.Embedding(vocab_size_new, embedding_size, _weight=weight)
            weight_r = torch.cat([self.embeddings_r.weight, ((torch.rand(vocab_size_new - vocab_size, 2 * embedding_size)-0.5)/embedding_size/2).to(device)])
            self.embeddings_r = nn.Embedding(vocab_size_new, 2 * embedding_size, _weight=weight_r)
        self.embeddings_f = nn.Embedding(function_size_new, 2 * embedding_size, _weight=((torch.rand(function_size_new, 2 * embedding_size)-0.5)/embedding_size/2).to(device))

    def v(self, inp):
        e  = self.embeddings(inp[:,1:])
        v_f = self.embeddings_f(inp[:,0]) # theta function
        v_prev = torch.cat([e[:,0], (e[:,1] + e[:,2]) / 2], dim=1)  # in_j-1 / e[:,0] is opcode, e[:,1] is operand1, e[:,2] is operand2 (equation (3))
        v_next = torch.cat([e[:,3], (e[:,4] + e[:,5]) / 2], dim=1)  # in_j+1
        v = ((v_f + v_prev + v_next) / 3).unsqueeze(2) # equation (4)
        return v

    # Softmax loss
    # def forward(self, inp, pos, neg):
    #     device, batch_size = inp.device, inp.shape[0]
    #     v = self.v(inp)
    #     # negative sampling loss
    #     pred = torch.bmm(self.embeddings_r(torch.cat([pos, neg], dim=1)), v).squeeze()
    #     label = torch.cat([torch.ones(batch_size, 3), torch.zeros(batch_size, neg.shape[1])], dim=1).to(device)
    #     print(f"Shape of pred: {pred.shape}")
    #     print(f"Shape of label: {label.shape}") 
    #     return bce(sigmoid(pred), label)
    def forward(self, inp, pos, neg):
        device, batch_size = inp.device, inp.shape[0]
        v = self.v(inp)

        # negative sampling loss
        pred = torch.bmm(self.embeddings_r(torch.cat([pos, neg], dim=1)), v).squeeze()

        # Ensure label and pred have the same shape
        label = torch.cat([torch.ones(batch_size, 3), torch.zeros(batch_size, neg.shape[1])], dim=1).to(device)

        # Print shapes for debugging
        # print(f"Shape of pred: {pred.shape}")
        # print(f"Shape of label: {label.shape}")

        # If pred and label have different shapes, adjust them
        if len(pred.shape) == 1 and len(label.shape) == 2:
            # If pred is [28] and label is [1, 28], adjust pred
            pred = pred.unsqueeze(0)
            print(f"Adjusted Shape of pred: {pred.shape}")
        
        elif len(label.shape) == 1 and len(pred.shape) == 2:
            # If label is [28] and pred is [1, 28], adjust label
            label = label.unsqueeze(0)
            print(f"Adjusted Shape of label: {label.shape}")

        # Final check and matching
        if pred.shape != label.shape:
            raise ValueError(f"Final Shape mismatch: pred {pred.shape}, label {label.shape}")

        return bce(sigmoid(pred), label)


    def predict(self, inp, pos):
        device, batch_size = inp.device, inp.shape[0]
        v = self.v(inp)
        probs = torch.bmm(self.embeddings_r(torch.arange(self.embeddings_r.num_embeddings).repeat(batch_size, 1).to(device)), v).squeeze(dim=2)
        return softmax(probs)
