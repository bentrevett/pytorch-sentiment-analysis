import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNClassification(nn.Module):
    """
    Standard bidirectional LSTM
    """
    def __init__(self, input_dim, output_dim, translation=False, attention=False, embedding_dim=256, hidden_dim=256, rnn_type='LSTM', n_layers=2, bidirectional=True, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        #set these as attributes as we need in the forward method
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.translation = translation
        self.attention = attention

        #translation idea taken from: https://github.com/Smerity/keras_snli
        if translation:
            self.translation_fc = TimeDistributed(nn.Linear(embedding_dim, embedding_dim))
        
        #self-attention idea taken from: https://github.com/cyb70289/kaggle/blob/master/toxic-comment/models/rnn_att.py
        if attention:
            self.query_weights = nn.Parameter(torch.zeros(hidden_dim*(2 if bidirectional else 1)))
            if torch.cuda.is_available():
                self.query_weigts = self.query_weights.cuda()

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout)
        else:
            raise ValueError(f'rnn_type must be LSTM, GRU or RNN! Got {rnn_type}')

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0)
            elif 'weight_hh' in name:
                nn.init.orthogonal(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform(param)

        #linear input size is num_directions * hidden_dim
        fc_inp = 2 * hidden_dim if bidirectional else hidden_dim 
        self.fc = nn.Linear(fc_inp, output_dim)

        for name, param in self.fc.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal(param)

        #layer normalization
        self.ln = LayerNorm(hidden_dim)

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        #x = [bsz, seq. len]
        #print(x.shape)

        x = self.embedding(x)

        #x = [bsz, seq. len, emb. dim.]
        #print(x.shape)

        if self.translation:
            x = self.translation_fc(x)

        #reshape as need to be [seq. len, bsz, emb. dim] for the rnn
        x = self.do(x.permute(1, 0, 2))

        #x = [seq. len, bsz, emb. dim]
        #print(x.shape)

        if self.rnn_type == 'LSTM':
            o, (h, _) = self.rnn(x)
        else:
            o, h = self.rnn(x)

        #   h is [num dir * num layer, bsz, hid. dim]
        #   the first dim of h goes [layer1 forward, layer1 backward, layer2 forward, layer2 backward]
        #   top layer forward == h[-2,:,:], top layer backward == h[-1,:,:]
        #   so to get the final forward+backward hidden, use h[-2:,:,:]

        #   o is [seq len, bsz, hid_dim * num dir]
        #   last dim of o is cat(forward, backward, so o[-1,:,:hid dim] is the final forward hidden state, equal to h[-2,:,:]
        #   to get the final backward state, you need to get the first element of the seq. len and the last hid dim of the final dimension
        #   i.e. o[0, :, hid dim:], which equals h[-1,:,:]

        #assert torch.equal(o[0, :, self.hidden_dim:], h[-1, :, :])
        #assert torch.equal(o[-1, :, :self.hidden_dim], h[-2, :, :])

        #h = [n_layers*n_directions, bsz, hid. dim.]
        #print(h.shape)

        if self.attention:
            o = o.permute(1, 0, 2) #o = [bsz, seq. len., hid. dim.]
            attention_weights = (o * self.query_weights).sum(dim=-1) #a_w = [bsz, seq. len], q_w = [hid. dim.]
            attention_weights = F.softmax(attention_weights, dim=1) #a_w = [bsz, seq. len]
            out = o * attention_weights[...,None] #out = [bsz, seq. len, seq. len], #a_w[...,None] = [bsz, seq. len, 1]
            #...notation means attention_weights[:], see: https://stackoverflow.com/questions/772124/what-does-the-python-ellipsis-object-do
            # None means add a new axis, see: https://stackoverflow.com/questions/1408311/numpy-array-slice-using-none
            out = out.sum(dim=1) #[bsz, seq. len]
            output = self.fc(out)
        else:
            if self.bidirectional:
                #h[-2,:,:] = [bsz, hid dim.]
                #h[-1,:,:] = [bsz, hid dim.]
                #h = [bsz, hid dim. * 2]
                h = self.do(torch.cat((h[-2,:,:], h[-1,:,:]), dim=1))
            else:
                #h = [bsz, hid dim.]
                h = self.do(h[-1,:,:])
                
            output = self.fc(h)

        #output = [bsz, output dim.]

        return output

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

"""class RnnAtt(nn.Module):
    ''' RNN with attention '''

    def __init__(self, input_dim, output_dim, embedding_dim=256, hidden_dim=256, rnn_type='LSTM', n_layers=2, bidirectional=True, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        else:
            raise ValueError(f'rnn_type must be LSTM, GRU or RNN! Got {rnn_type}')

        self.query_weights = nn.Parameter(torch.zeros(hidden_dim*(self.bidir+1)))
        if torch.cuda.is_available():
                self.query_weights = self.query_weights.cuda()
        self.fc = nn.Linear(self.hidden_dim*(2 if bidirectional else 1), output_dim)
  
    def forward(self, x):
        
        #x = [bsz, seq. len]

        x = self.embedding(x)

        #x = [bsz, seq. len, emb. dim]

        o, _ = self.rnn(x)

        # atten_weights: batch * sequence
        atten_weights = (o * self.query_weights).sum(dim=-1)
        atten_weights = F.softmax(atten_weights, dim=1)
        o = o * atten_weights[..., None]
        # out: batch * hidden_dim*(1+bidir)
        
        o = o.sum(dim=1)
        o = o.view(out.size(0), -1)
        output = self.fc(o)
        # out: batch * n_classes
        return output"""

"""
STOLEN FROM ALLENNLP
"""
class TimeDistributed(torch.nn.Module):
    """
    Given an input shaped like ``(batch_size, time_steps, [rest])`` and a ``Module`` that takes
    inputs like ``(batch_size, [rest])``, ``TimeDistributed`` reshapes the input to be
    ``(batch_size * time_steps, [rest])``, applies the contained ``Module``, then reshapes it back.
    Note that while the above gives shapes with ``batch_size`` first, this ``Module`` also works if
    ``batch_size`` is second - we always just combine the first two dimensions, then split them.
    """
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self._module = module

    def forward(self, *inputs):  # pylint: disable=arguments-differ
        reshaped_inputs = []
        for input_tensor in inputs:
            input_size = input_tensor.size()
            if len(input_size) <= 2:
                raise RuntimeError("No dimension to distribute: " + str(input_size))

            # Squash batch_size and time_steps into a single axis; result has shape
            # (batch_size * time_steps, input_size).
            squashed_shape = [-1] + [x for x in input_size[2:]]
            reshaped_inputs.append(input_tensor.contiguous().view(*squashed_shape))

        reshaped_outputs = self._module(*reshaped_inputs)

        # Now get the output back into the right shape.
        # (batch_size, time_steps, [hidden_size])
        new_shape = [input_size[0], input_size[1]] + [x for x in reshaped_outputs.size()[1:]]
        outputs = reshaped_outputs.contiguous().view(*new_shape)

        return outputs