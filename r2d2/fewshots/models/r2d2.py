import torch
from torch import nn
from torch.autograd import Variable
from torch import transpose as t
from torch import inverse as inv
from torch import mm
# gesv is deprecated / not usable, so we have to guess the right solver to use instead
#from torch import gesv
import numpy as np

from fewshots.labels_r2d2 import make_float_label, make_long_label
from fewshots.models.adjust import AdjustLayer, LambdaLayer
from fewshots.data.queries import shuffle_queries_multi


def t_(x):
    return t(x, 0, 1)


class RRNet(nn.Module):
    def __init__(self, encoder, debug, out_dim, learn_lambda, init_lambda, init_adj_scale, lambda_base, adj_base,
                 n_augment, linsys):
        super(RRNet, self).__init__()
        self.encoder = encoder
        self.debug = debug
        self.lambda_rr = LambdaLayer(learn_lambda, init_lambda, lambda_base)
        self.L = nn.CrossEntropyLoss()
        self.adjust = AdjustLayer(init_scale=init_adj_scale, base=adj_base)
        self.output_dim = out_dim
        self.n_augment = n_augment
        self.linsys = linsys

    def loss(self, sample):
        # I don't really see a way to fix this other than to read the paper so I'm gonna do that
        # We're finding the loss between y_hat (eq 6) and y_outer
        # We reorganize our parameters into a matrix W? But I guess we're finding that with the Woodbury formula
            # W = \(Z) = X^T * (X*X^T + \I)^-1 * Y
            # This is the ridge regression formula
        # We have some labels for dimension sizes: o, e, p, and m. What are these?
            # m -> x in R^m, size of input dims
            # o -> y in R^o, size of output dims
                # Are these supposed to be labels or features (i.e. for unsupervised learning)?
            # e, p -> episode-specific predictor f(phi(x); w[epsilon]) : R^e x R^p -> R^o
                # Maps input embeddings to outputs
                # Parameterized by parameter set w[epsilon] in R^p, specific to episode epsilon
        # So we're working in a single few-shot episode? And we do expect inputs = data, outputs = labels?
        # Still don't really get what's wrong but I understand it better, stare at math some more i guess
        # Also re-read the meta-learning section, it gives context to understand what we're doing w/ R2

        # Okay looking at the code itself, here's what I still don't get
        # Why are we setting n_way, n_shot, and n_query from xs.size and xq.size?
            # Are we supposed to be receiving like, a pre-packaged few-shot episode?
            # And maybe pytorch used to return that but it doesn't now?
            # Need to code trace a bit to figure that out
            # Oh wait episodic batch sampler is supposed to do that!
        # What is xq.size(1) even supposed to be? Why would the output be 2-dimensional?
        # I guess output_dim is just that random fucking number they get somehow, i guess the output of layer 3 and 4 of conv
        # self.n_augment = 1
        # so i guess what we're supposed to have is like, n_way = n_classes, n_shot = 

        # Wait, okay, so xs is (n_way*n_shot*n_aug, xs.size()[2:])
            # wtf is xs.size()[2:]?
        # and xq is (n_way*n_query, xq.size()[2:])  
        # and we concatenate them on dim 0, which is xs.size(0) = xq.size(0) = n_way
        # so is xs the augmented training set/episode and xq is the holdout set?
        # as in like s for n_shot, q for n_query?
        
        # ok then y is our label set. y_inner is labels for xs, y_outer is labels for xq
        # what is make_float_label?
        # idk why y_outer_binary is here but its the same
        # torch.eye takes n and makes an n*n identity matrix, so that's how we get I

        # what is shuffle_queries_multi doing? i guess dw about that bc it doesn't break our code (yet)

        # ok now we need z to calculate w
        # do a forward call w/ encoder on x, store in z
        # zs = z[0:len(xs)], zq = z[len(xs):len(z)]
        # w uses fucking self.output_dim again lmao how do i get this number???

        # In this paper, n_ways = n_class, n_shots = samples per class, n_query = number of query images per class
        # Right now I have n_ways = 16, n_shots = channels = 23, n_query = n_ways = 16
        # How do I like, not have that?

        # OKAY THIS FINALLY ACTUALLY RUNS
        # now the problem is that it doesn't work LMAO

        # maybe I can just,
        #xs, xq = Variable(sample[0]), Variable(sample[1])
        xs, xq = Variable(sample['xs']), Variable(sample['xq'])
        assert (xs.size(0) == xq.size(0))
        n_way, n_shot, n_query = xs.size(0), xs.size(1), xq.size(1) #xq.size(1) --> xq.size(0)
        #print(xs.shape, xq.shape)
        #print(n_way, n_shot, n_query, self.n_augment)
        #print(self.output_dim)
        if n_way * n_shot * self.n_augment > self.output_dim + 1:
            rr_type = 'standard'
            I = Variable(torch.eye(self.output_dim + 1).cuda())
        else:
            rr_type = 'woodbury'
            I = Variable(torch.eye(n_way * n_shot * self.n_augment).cuda())

        y_inner = make_float_label(n_way, n_shot * self.n_augment) / np.sqrt(n_way * n_shot * self.n_augment)
        y_outer_binary = make_float_label(n_way, n_query)
        y_outer = make_long_label(n_way, n_query)

        x = torch.cat([xs.view(n_way * n_shot * self.n_augment, *xs.size()[2:]),
                       xq.view(n_way * n_query, *xq.size()[2:])], 0)

        x, y_outer_binary, y_outer = shuffle_queries_multi(x, n_way, n_shot, n_query, self.n_augment, y_outer_binary,
                                                           y_outer)

        z = self.encoder.forward(x)
        zs = z[:n_way * n_shot * self.n_augment]
        zq = z[n_way * n_shot * self.n_augment:]
        # add a column of ones for the bias
        ones = Variable(torch.unsqueeze(torch.ones(zs.size(0)).cuda(), 1))
        if rr_type == 'woodbury':
            wb = self.rr_woodbury(torch.cat((zs, ones), 1), n_way, n_shot, I, y_inner, self.linsys)

        else:
            wb = self.rr_standard(torch.cat((zs, ones), 1), n_way, n_shot, I, y_inner, self.linsys)

        # it didn't like when we set dimension=, start=, length=
        # so i just made it (dimension, start, length)
        w = wb.narrow(0, 0, self.output_dim)
        b = wb.narrow(0, self.output_dim, 1)
        out = mm(zq, w) + b
        y_hat = self.adjust(out)
        # print("%.3f  %.3f  %.3f" % (w.mean()*1e5, b.mean()*1e5, y_hat.max()))

        _, ind_prediction = torch.max(y_hat, 1)
        _, ind_gt = torch.max(y_outer_binary, 1)

        #print(y_hat, y_outer)
        loss_val = self.L(y_hat, y_outer)
        acc_val = torch.eq(ind_prediction, ind_gt).float().mean()
        # print('Loss: %.3f Acc: %.3f' % (loss_val.data[0], acc_val.data[0]))
        # data[0] --> item()
        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

    def rr_standard(self, x, n_way, n_shot, I, yrr_binary, linsys):
        x /= np.sqrt(n_way * n_shot * self.n_augment)

        if not linsys:
            w = mm(mm(inv(mm(t(x, 0, 1), x) + self.lambda_rr(I)), t(x, 0, 1)), yrr_binary)
        else:
            A = mm(t_(x), x) + self.lambda_rr(I)
            v = mm(t_(x), yrr_binary)
            #w, _ = gesv(v, A)
            # Hopefully this replacement works right
            w = torch.linalg.solve(A, v)

        return w

    def rr_woodbury(self, x, n_way, n_shot, I, yrr_binary, linsys):
        x /= np.sqrt(n_way * n_shot * self.n_augment)

        if not linsys:
            w = mm(mm(t(x, 0, 1), inv(mm(x, t(x, 0, 1)) + self.lambda_rr(I))), yrr_binary)
        else:
            A = mm(x, t_(x)) + self.lambda_rr(I)
            v = yrr_binary
            #w_, _ = gesv(v, A)
            w_ = torch.linalg.solve(A, v)
            w = mm(t_(x), w_)

        return w
