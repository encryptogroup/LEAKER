"""MADE and ResMADE."""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kmeans_pytorch import kmeans

from . import common
from .import distributions
from . import utils


class MaskedLinear(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 condition_on_ordering=False):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

        self.condition_ordering_linear = None
        if condition_on_ordering:
            self.condition_ordering_linear = nn.Linear(in_features,
                                                       out_features,
                                                       bias=False)

        self.masked_weight = None

    def set_mask(self, mask):
        """Accepts a mask of shape [in_features, out_features]."""
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def set_cached_mask(self, mask):
        self.mask.data.copy_(mask)

    def get_cached_mask(self):
        return self.mask.clone().detach()

    def forward(self, input):
        if self.masked_weight is None:
            return F.linear(input, self.mask * self.weight, self.bias)
        else:
            # ~17% speedup for Prog Sampling.
            out = F.linear(input, self.masked_weight, self.bias)

        if self.condition_ordering_linear is None:
            return out
        return out + F.linear(torch.ones_like(input),
                              self.mask * self.condition_ordering_linear.weight)


class MaskedResidualBlock(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 activation,
                 condition_on_ordering=False,
                 resmade_drop_prob=0.):
        assert in_features == out_features, [in_features, out_features]
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            MaskedLinear(in_features,
                         out_features,
                         bias=True,
                         condition_on_ordering=condition_on_ordering))
        self.layers.append(
            MaskedLinear(in_features,
                         out_features,
                         bias=True,
                         condition_on_ordering=condition_on_ordering))
        self.dropout = nn.Dropout(p=resmade_drop_prob)
        self.activation = activation

    def set_mask(self, mask):
        self.layers[0].set_mask(mask)
        self.layers[1].set_mask(mask)

    def set_cached_mask(self, mask):
        # They have the same mask.
        self.layers[0].mask.copy_(mask)
        self.layers[1].mask.copy_(mask)

    def get_cached_mask(self):
        return self.layers[0].mask.clone().detach()

    def forward(self, input):
        out = input
        out = self.activation(out)
        out = self.layers[0](out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.layers[1](out)
        return input + out


class MADE(nn.Module):

    def __init__(
            self,
            nin,
            hidden_sizes,
            nout,
            num_masks=1,
            natural_ordering=True,
            input_bins=None,
            activation=nn.ReLU,
            do_direct_io_connections=False,
            input_encoding=None,
            output_encoding='one_hot',
            embed_size=32,
            input_no_emb_if_leq=True,
            embs_tied=True,
            residual_connections=False,
            factor_table=None,
            seed=11123,
            fixed_ordering=None,
            # Wildcard skipping.
            dropout_p=0,
            fixed_dropout_p=False,
            learnable_unk=None,
            grouped_dropout=False,
            per_row_dropout=False,
            # DMoL.
            num_dmol=0,
            scale_input=False,
            dmol_col_indexes=[],
            # Join support.
            num_joined_tables=0,
            table_dropout=False,
            table_num_columns=None,
            table_column_types=None,
            table_indexes=None,
            table_primary_index=None,
            resmade_drop_prob=0.,
            column_model=[],
            id2column=[],
            n_mm=[],
            sample_num=1,
            init_data=None,
            tol=1.0
    ):
        """MADE and ResMADE.

        Args:
          nin: integer; number of input variables.  Each input variable
            represents a column.
          hidden_sizes: a list of integers; number of units in hidden layers.
          nout: integer; number of outputs, the sum of all input variables'
            domain sizes.
          num_masks: number of orderings + connectivity masks to cycle through.
          natural_ordering: force natural ordering of dimensions, don't use
            random permutations.
          input_bins: classes each input var can take on, e.g., [5, 2] means
            input x1 has values in {0, ..., 4} and x2 in {0, 1}.  In other
            words, the domain sizes.
          activation: the activation to use.
          do_direct_io_connections: whether to add a connection from inputs to
            output layer.  Helpful for information flow.
          input_encoding: input encoding mode, see EncodeInput().
          output_encoding: output logits decoding mode, either 'embed' or
            'one_hot'.  See logits_for_col().
          embed_size: int, embedding dim.
          input_no_emb_if_leq: optimization, whether to turn off embedding for
            variables that have a domain size less than embed_size.  If so,
            those variables would have no learnable embeddings and instead are
            encoded as one hot vecs.
          residual_connections: use ResMADE?  Could lead to faster learning.
            Recommended to be set for any non-trivial datasets.
          seed: seed for generating random connectivity masks.
          fixed_ordering: variable ordering to use.  If specified, order[i]
            maps natural index i -> position in ordering.  E.g., if order[0] =
            2, variable 0 is placed at position 2.
          dropout_p, learnable_unk: if True, turn on column masking during
            training time, which enables the wildcard skipping (variable
            skipping) optimization during inference.  Recommended to be set for
            any non-trivial datasets.
          grouped_dropout: bool, whether to mask factorized subvars for an
            original var together or independently.
          per_row_dropout: bool, whether to make masking decisions per tuple or
            per batch.
          num_dmol, scale_input, dmol_col_indexes: (experimental) use
            discretized mixture of logistics as outputs for certain columns.
          num_joined_tables: int, number of joined tables.
          table_dropout: bool, whether to use a table-aware dropout scheme
            (make decisions on each table, then drop all columns or none from
            each).
          table_num_columns: list of int; number of columns from each table i.
          table_column_types: list of int; variable i's column type.
          table_indexes: list of int; variable i is from which table?
          table_primary_index: int; used as an optimization where we never mask
            out this table.
          resmade_drop_prob: float, normal dropout probability inside ResMADE.
        """
        super().__init__()
        self.nin = nin

        if num_masks > 1:
            # Double the weights, so need to reduce the size to be fair.
            hidden_sizes = [int(h // 2**0.5) for h in hidden_sizes]
            print('Auto reducing MO hidden sizes to', hidden_sizes, num_masks)
        # None: feed inputs as-is, no encoding applied.  Each column thus
        #     occupies 1 slot in the input layer.  For testing only.
        assert input_encoding in [None, 'one_hot', 'embed']
        self.input_encoding = input_encoding
        assert output_encoding in ['one_hot', 'embed']
        if num_dmol > 0:
            assert output_encoding == 'embed'
        self.embed_size = self.emb_dim = embed_size
        self.output_encoding = output_encoding
        self.activation = activation
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        self.input_bins = input_bins
        #print('INPUT_BINS: ', input_bins)
        self.input_no_emb_if_leq = input_no_emb_if_leq
        self.do_direct_io_connections = do_direct_io_connections
        self.embs_tied = embs_tied
        self.factor_table = factor_table

        self.residual_connections = residual_connections

        self.num_masks = num_masks
        self.learnable_unk = learnable_unk
        self.dropout_p = dropout_p
        self.fixed_dropout_p = fixed_dropout_p
        self.grouped_dropout = grouped_dropout
        self.per_row_dropout = per_row_dropout
        if self.per_row_dropout:
            assert self.dropout_p

        self.resmade_drop_prob = resmade_drop_prob

        self.fixed_ordering = fixed_ordering
        if fixed_ordering is not None:
            assert num_masks == 1
            print('** Fixed ordering {} supplied, ignoring natural_ordering'.
                  format(fixed_ordering))

        self.column_model = column_model
        self.n_mm = n_mm
        self.sample_num = sample_num
        self.gmm_dict = {}
        self.umm_dict = {}
        self.id2column = id2column 
        self.column2id = []
        gmm_num = 0
        gmm_comp_num = []
        for i in range(self.nin):
            ids = []
            for j in range(len(self.id2column)):
                if self.id2column[j] == i:
                    ids.append(j)
            self.column2id.append(np.array(ids, dtype=np.int))
            if self.column_model[i] == 1:
                self.gmm_dict[i] = gmm_num
                gmm_num += 1
                gmm_comp_num.append(self.n_mm[i])
        self.tol = tol
        self.phi = nn.ParameterList([nn.Parameter(torch.randn(gmm_comp_num[i])) for i in range(gmm_num)])
        self.mu = nn.ParameterList([nn.Parameter(torch.randn(gmm_comp_num[i], 1)) for i in range(gmm_num)])
        self.cov_sqrt = nn.ParameterList([nn.Parameter(torch.randn(gmm_comp_num[i], 1, 1)) for i in range(gmm_num)])

        self.weight = nn.ParameterList()
        self.min_v = None
        self.max_v = None 
        self.gap = nn.ParameterList()
        
        # Join support.  Flags below are used for training time only.
        self.num_joined_tables = num_joined_tables
        self.table_dropout = table_dropout
        self.table_num_columns = table_num_columns
        self.table_column_types = table_column_types
        self.table_indexes = table_indexes
        self.table_primary_index = table_primary_index

        # Discretized MoL.
        self.num_dmol = num_dmol
        self.scale_input = scale_input
        self.dmol_col_indexes = dmol_col_indexes

        assert self.input_bins is not None
        self.input_bins_encoded = [
            self._get_input_encoded_dist_size(self.input_bins[i], i)
            for i in range(len(self.input_bins))
        ]
        self.input_bins_encoded_cumsum = np.cumsum(self.input_bins_encoded)
        encoded_bins = [
            self._get_output_encoded_dist_size(self.input_bins[i], i)
            for i in range(len(self.input_bins))
        ]
        hs = [nin] + hidden_sizes + [sum(encoded_bins)]
        print('encoded_bins (output)', encoded_bins)
        print('encoded_bins (input)', self.input_bins_encoded)

        self.kOnes = None

        self.net = []
        for i, (h0, h1) in enumerate(zip(hs, hs[1:])):
            if residual_connections:
                if i == 0 or i == len(hs) - 2:
                    if i == len(hs) - 2:
                        self.net.append(activation())
                    # Input / Output layer.
                    self.net.extend([
                        MaskedLinear(h0,
                                     h1,
                                     condition_on_ordering=self.num_masks > 1)
                    ])
                else:
                    # Middle residual blocks must have same dims.
                    assert h0 == h1, (h0, h1, hs)
                    self.net.extend([
                        MaskedResidualBlock(
                            h0,
                            h1,
                            activation=activation(inplace=False),
                            condition_on_ordering=self.num_masks > 1,
                            resmade_drop_prob=self.resmade_drop_prob)
                    ])
            else:
                self.net.extend([
                    MaskedLinear(h0,
                                 h1,
                                 condition_on_ordering=self.num_masks > 1),
                    activation(inplace=True),
                ])
        if not residual_connections:
            self.net.pop()
        self.net = nn.Sequential(*self.net)

        if self.input_encoding is not None:
            # Input layer should be changed.
            assert self.input_bins is not None
            input_size = 0
            for i, dist_size in enumerate(self.input_bins):
                input_size += self._get_input_encoded_dist_size(dist_size, i)
            new_layer0 = MaskedLinear(input_size,
                                      self.net[0].out_features,
                                      condition_on_ordering=self.num_masks > 1)
            self.net[0] = new_layer0

        if self.output_encoding == 'embed':
            assert self.input_encoding == 'embed'
        if self.input_encoding == 'embed':
            self.embeddings = nn.ModuleList()
            if not self.embs_tied:
                self.embeddings_out = nn.ModuleList()
            for i, dist_size in enumerate(self.input_bins):
                if dist_size <= self.embed_size and self.input_no_emb_if_leq:
                    embed = embed2 = None
                else:
                    embed = nn.Embedding(dist_size, self.embed_size)
                    embed2 = nn.Embedding(
                        dist_size,
                        self.embed_size) if not self.embs_tied else None
                self.embeddings.append(embed)
                if not self.embs_tied:
                    self.embeddings_out.append(embed2)

        # Learnable [MASK] representation.
        if self.dropout_p:
            self.unk_embeddings = nn.ParameterList()
            for i, dist_size in enumerate(self.input_bins):
                self.unk_embeddings.append(
                    nn.Parameter(torch.zeros(1, self.input_bins_encoded[i])))

        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = seed if seed is not None else 11123

        self.direct_io_layer = None
        self.logit_indices = np.cumsum(encoded_bins)
        self.m = {}
        self.cached_masks = {}

        self.update_masks()
        self.orderings = [self.m[-1]]

        # Optimization: cache some values needed in EncodeInput().
        self.bin_as_onehot_shifts = None


    def compute_mm_params(self, data):
        #print('DATA:', data)
        gmm_index = 0
        umm_index = 0
        min_v = []
        max_v = []
        for i in range(self.nin):
            if self.column_model[i] == 0:
                continue
            elif self.column_model[i] == 1:
                #column_id_list = self.get_column_id_list[i]
                column_id_list = self.column2id[i]
                # GMM or UMM
                self.gmm_dict[i] = gmm_index
                z = data[:, column_id_list]
                #print('Z: ', z )
                #print(torch.isnan(z).any())
                clusters, centers = kmeans(z.unsqueeze(1), self.n_mm[i], tol=self.tol, device=z.device)
                #gamma = torch.rand(z.shape[0], self.n_mm[i])
                #gamma = nn.functional.softmax(gamma, dim=1)
                clusters = clusters.to(device=z.device)
                centers = centers.to(device=z.device)
                #print(clusters)
                #print(centers)
                N = z.size(0)
                gamma = torch.zeros([N, self.n_mm[i]], device=z.device).scatter_(1, clusters.unsqueeze(1), 1)
                # K
                sum_gamma = torch.sum(gamma, dim=0)
                phi = (sum_gamma / N)
                self.phi[gmm_index] = nn.Parameter(phi)

                # N x K
                #z_p = gamma * z.unsqueeze(1)

                # K
                #mu = torch.rand(self.n_mm[i]) * (torch.max(z) - torch.min(z)) + torch.min(z)
                #mu = torch.sum(z_p, dim=0) / sum_gamma
                # K x D
                mu = centers.squeeze(1)
                #mu = centers
                self.mu[gmm_index] = nn.Parameter(mu)
                #print('GAMMA: ', gamma)
                #print('MU: ', mu)
                # N x K x D
                z_mu = z.unsqueeze(1) - mu.unsqueeze(0)

                # N x K x D x D
                z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
                # K x D X D
                cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
                #print(cov)

                #var = torch.sum(gamma * ((z.unsqueeze(1) - mu) ** 2), dim=0) / sum_gamma
                cov_l = []
                for j in range(self.n_mm[i]):
                    cov_a = torch.cholesky(torch.inverse(cov[j]))
                    cov_l.append(cov_a.unsqueeze(0))
                cov_l = torch.cat(cov_l, dim=0)
                self.cov_sqrt[gmm_index] = nn.Parameter(cov_l)
                #self.cov_sqrt.append(nn.Parameter(torch.sqrt(cov)))
                #print(phi)
                #print(mu)
                #print(cov_l)
                gmm_index += 1
            elif self.column_model[i] == 2:
                # UMM
                self.umm_dict[i] = umm_index
                umm_index += 1
                z = data[:, i]
                weight = torch.ones(self.n_mm[i]) / self.n_mm[i]
                self.weight.append(nn.Parameter(weight))

                min_v.append(torch.min(z))
                max_v.append(torch.max(z))
                gap = torch.rand(self.n_mm[i])
                self.gap.append(nn.Parameter(gap))
                '''
                gap_value, gap_index = torch.sort(torch.rand(self.n_mm[i] - 1))
                for i in range(self.n_mm[i] - 1):
                    gap = min_v + gap_value[i] * (max_v - min_v)
                    lb.append(gap)
                    ub.append(gap)
                ub.append(max_v)
                print(lb)
                lb = torch.tensor(lb)
                ub = torch.tensor(ub)
                mu = (ub + lb) / 2
                self.mu.append(nn.Parameter(mu))
                var = (ub - lb) ** 2 / 12
                self.std_sqrt.append(nn.Parameter(torch.sqrt(torch.sqrt(var))))
                '''               
        self.min_v = torch.tensor(min_v)
        self.max_v = torch.tensor(max_v)
        #self.register_buffer("min_v", torch.tensor(min_v))
        #self.register_buffer("max_v", torch.tensor(max_v))
        #print(min_v)
        #print(max_v)

    def get_gmm_parameter(self, column_id):
        index = self.gmm_dict[column_id]
        phi = self.phi[index]
        phi = nn.functional.softmax(phi, dim=0)
        mu = self.mu[index]
        cov_sqrt = self.cov_sqrt[index]
        cov = []
        for i in range(self.n_gmm[index]):
            cov_k = cov_sqrt[i]
            cov_i = torch.inverse(cov_k.mm(cov_k.t()))
            cov.append(cov_i.unsqueeze(0))
        cov = torch.cat(cov, dim=0)
        if phi.is_cuda:
            phi = phi.cpu().detach().numpy()
            mu = mu.cpu().detach().numpy()
            cov = cov.cpu().detach().numpy()
        else:
            phi = phi.detach().numpy()
            mu = mu.detach().numpy()
            cov = cov.detach().numpy()
        return phi, mu, cov
    
    def get_umm_parameter(self, column_id):
        index = self.umm_dict[column_id]
        weight = self.weight[index]
        weight = nn.functional.softmax(weight, dim=0)
        min_v = self.min_v[index]
        max_v = self.max_v[index]
        gap = self.gap[index]
        gap = nn.functional.softmax(gap, dim=0)
        gap_cum = torch.cumsum(gap, dim=0)
        gap_value = (max_v - min_v) * gap_cum[:-1] + min_v
        lb = torch.cat([min_v.unsqueeze(0), gap_value], dim=0)
        ub = torch.cat([gap_value, max_v.unsqueeze(0)], dim=0)
        if weight.is_cuda:
            weight = weight.cpu().detach().numpy()
            lb = lb.cpu().detach().numpy()
            ub = ub.cpu().detach().numpy()
        else:
            weight = weight.detach().numpy()
            lb = lb.detach().numpy()
            ub = ub.detach().numpy()
        return weight, lb, ub

    def get_parameter(self, column_id):
        if self.column_model[column_id] == 1:
            return self.get_gmm_parameter(column_id)
        elif self.column_model[column_id] == 2:
            return self.get_umm_parameter(column_id)
 

    def UseDMoL(self, natural_idx):
        """Returns True if we want to use DMoL for this column."""
        if self.num_dmol <= 0:
            return False
        return natural_idx in self.dmol_col_indexes

    def _build_or_update_direct_io(self):
        assert self.nout > self.nin and self.input_bins is not None
        direct_nin = self.net[0].in_features
        direct_nout = self.net[-1].out_features
        if self.direct_io_layer is None:
            self.direct_io_layer = MaskedLinear(
                direct_nin,
                direct_nout,
                condition_on_ordering=self.num_masks > 1)
        mask = np.zeros((direct_nout, direct_nin), dtype=np.uint8)

        # Inverse: ord_idx -> natural idx.
        inv_ordering = utils.InvertOrder(self.m[-1])

        for ord_i in range(self.nin):
            nat_i = inv_ordering[ord_i]
            # x_(nat_i) in the input occupies range [inp_l, inp_r).
            inp_l = 0 if nat_i == 0 else self.input_bins_encoded_cumsum[nat_i -
                                                                        1]
            inp_r = self.input_bins_encoded_cumsum[nat_i]
            assert inp_l < inp_r

            for ord_j in range(ord_i + 1, self.nin):
                nat_j = inv_ordering[ord_j]
                # Output x_(nat_j) should connect to input x_(nat_i); it
                # occupies range [out_l, out_r) in the output.
                out_l = 0 if nat_j == 0 else self.logit_indices[nat_j - 1]
                out_r = self.logit_indices[nat_j]
                assert out_l < out_r
                mask[out_l:out_r, inp_l:inp_r] = 1
        mask = mask.T
        self.direct_io_layer.set_mask(mask)

    def _get_input_encoded_dist_size(self, dist_size, i):
        del i  # Unused.
        # TODO: Allow for different encodings for different cols.
        if self.input_encoding == 'embed':
            if self.input_no_emb_if_leq:
                dist_size = min(dist_size, self.embed_size)
            else:
                dist_size = self.embed_size
        elif self.input_encoding == 'one_hot':
            pass
        elif self.input_encoding is None:
            return 1
        else:
            assert False, self.input_encoding
        return dist_size

    def _get_output_encoded_dist_size(self, dist_size, i):
        # TODO: allow different encodings for different cols.
        if self.output_encoding == 'embed':
            if self.input_no_emb_if_leq:
                dist_size = min(dist_size, self.embed_size)
            else:
                dist_size = self.embed_size
        elif self.UseDMoL(i):
            dist_size = self.num_dmol * 3
        elif self.output_encoding == 'one_hot':
            pass
        return dist_size

    def update_masks(self, invoke_order=None):
        """Update m() for all layers and change masks correspondingly.

        This implements multi-order training support.

        No-op if "self.num_masks" is 1.
        """
        if self.m and self.num_masks == 1:
            return
        L = len(self.hidden_sizes)

        layers = [
            l for l in self.net if isinstance(l, MaskedLinear) or
            isinstance(l, MaskedResidualBlock)
        ]

        ### Precedence of several params determining ordering:
        #
        # invoke_order
        # orderings
        # fixed_ordering
        # natural_ordering
        #
        # from high precedence to low.

        # For multi-order models, we associate RNG seeds with orderings as
        # follows:
        #   orderings = [ o0, o1, o2, ... ]
        #   seeds = [ 0, 1, 2, ... ]
        # This must be consistent across training & inference.
        print('ORDER: ', invoke_order, hasattr(self, 'ordering'))

        if invoke_order is not None:
            # Inference path.
            found = False
            for i in range(len(self.orderings)):
                if np.array_equal(self.orderings[i], invoke_order):
                    found = True
                    break
            assert found, 'specified={}, avail={}'.format(
                invoke_order, self.orderings)

            if self.seed == (i + 1) % self.num_masks and np.array_equal(
                    self.m[-1], invoke_order):
                # During querying, after a multi-order model is configured to
                # take a specific ordering, it can be used to do multiple
                # forward passes per query.
                return

            self.seed = i
            self.m[-1] = np.asarray(invoke_order)

            if self.seed in self.cached_masks:
                masks, direct_io_mask = self.cached_masks[self.seed]
                assert len(layers) == len(masks), (len(layers), len(masks))
                for l, m in zip(layers, masks):
                    l.set_cached_mask(m)

                if self.do_direct_io_connections:
                    assert direct_io_mask is not None
                    self.direct_io_layer.set_cached_mask(direct_io_mask)

                self.seed = (self.seed + 1) % self.num_masks
                return  # Early return

            rng = np.random.RandomState(self.seed)
            print('creating rng with seed', self.seed)
            curr_seed = self.seed
            self.seed = (self.seed + 1) % self.num_masks

        elif hasattr(self, 'orderings'):
            # Training path: cycle through the special orderings.
            assert 0 <= self.seed and self.seed < len(self.orderings)
            self.m[-1] = self.orderings[self.seed]

            if self.seed in self.cached_masks:
                masks, direct_io_mask = self.cached_masks[self.seed]
                assert len(layers) == len(masks), (len(layers), len(masks))
                for l, m in zip(layers, masks):
                    l.set_cached_mask(m)

                if self.do_direct_io_connections:
                    assert direct_io_mask is not None
                    self.direct_io_layer.set_cached_mask(direct_io_mask)

                self.seed = (self.seed + 1) % self.num_masks
                return  # Early return

            rng = np.random.RandomState(self.seed)
            print('creating rng with seed', self.seed)
            print('constructing masks with seed', self.seed, 'self.m[-1]',
                  self.m[-1])
            curr_seed = self.seed
            self.seed = (self.seed + 1) % self.num_masks

        else:
            # Train-time initial construction: either single-order, or
            # .orderings has not been assigned yet.
            rng = np.random.RandomState(self.seed)
            print('creating rng with seed', self.seed)
            self.seed = (self.seed + 1) % self.num_masks
            self.m[-1] = np.arange(
                self.nin) if self.natural_ordering else rng.permutation(
                    self.nin)
            print(self.nin)
            print(self.m[-1])
            print(self.fixed_ordering)
            if self.fixed_ordering is not None:
                self.m[-1] = np.asarray(self.fixed_ordering)

        print('self.seed', self.seed, 'self.m[-1]', self.m[-1])

        if self.nin > 1:
            for l in range(L):
                if self.residual_connections:
                    assert len(np.unique(
                        self.hidden_sizes)) == 1, self.hidden_sizes
                    # Sequential assignment for ResMade: https://arxiv.org/pdf/1904.05626.pdf
                    if l > 0:
                        self.m[l] = self.m[0]
                    else:
                        self.m[l] = np.array([
                            (k - 1) % (self.nin - 1)
                            for k in range(self.hidden_sizes[l])
                        ])
                        if self.num_masks > 1:
                            self.m[l] = rng.permutation(self.m[l])
                else:
                    # Samples from [0, ncols - 1).
                    self.m[l] = rng.randint(self.m[l - 1].min(),
                                            self.nin - 1,
                                            size=self.hidden_sizes[l])
        else:
            # This should result in first layer's masks == 0.
            # So output units are disconnected to any inputs.
            for l in range(L):
                self.m[l] = np.asarray([-1] * self.hidden_sizes[l])

        print(self.m)
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        if self.nout > self.nin:
            # Last layer's mask needs to be changed.

            if self.input_bins is None:
                k = int(self.nout / self.nin)
                # Replicate the mask across the other outputs
                # so [x1, x2, ..., xn], ..., [x1, x2, ..., xn].
                masks[-1] = np.concatenate([masks[-1]] * k, axis=1)
            else:
                # [x1, ..., x1], ..., [xn, ..., xn] where the i-th list has
                # input_bins[i - 1] many elements (multiplicity, # of classes).
                mask = np.asarray([])
                for k in range(masks[-1].shape[0]):
                    tmp_mask = []
                    for idx, x in enumerate(zip(masks[-1][k], self.input_bins)):
                        mval, nbins = x[0], self._get_output_encoded_dist_size(
                            x[1], idx)
                        tmp_mask.extend([mval] * nbins)
                    tmp_mask = np.asarray(tmp_mask)
                    if k == 0:
                        mask = tmp_mask
                    else:
                        mask = np.vstack([mask, tmp_mask])
                masks[-1] = mask

        if self.input_encoding is not None:
            # Input layer's mask should be changed.

            assert self.input_bins is not None
            # [nin, hidden].
            mask0 = masks[0]
            print('MASK0: ', mask0)
            new_mask0 = []
            for i, dist_size in enumerate(self.input_bins):
                dist_size = self._get_input_encoded_dist_size(dist_size, i)
                # [dist size, hidden]
                new_mask0.append(
                    np.concatenate([mask0[i].reshape(1, -1)] * dist_size,
                                   axis=0))
            # [sum(dist size), hidden]
            new_mask0 = np.vstack(new_mask0)
            masks[0] = new_mask0

        assert len(layers) == len(masks), (len(layers), len(masks))
        for l, m in zip(layers, masks):
            l.set_mask(m)

        dio_mask = None
        if self.do_direct_io_connections:
            self._build_or_update_direct_io()
            dio_mask = self.direct_io_layer.get_cached_mask()

        # Cache.
        if hasattr(self, 'orderings'):
            print('caching masks for seed', curr_seed)
            masks = [l.get_cached_mask() for l in layers]
            assert curr_seed not in self.cached_masks
            self.cached_masks[curr_seed] = (masks, dio_mask)

    def name(self):
        n = 'made'
        if self.residual_connections:
            n += '-resmade'
        n += '-hidden' + '_'.join(str(h) for h in self.hidden_sizes)
        n += '-emb' + str(self.embed_size)
        if self.num_masks > 1:
            n += '-{}masks'.format(self.num_masks)
        if not self.natural_ordering:
            n += '-nonNatural'
        n += ('-no' if not self.do_direct_io_connections else '-') + 'directIo'
        n += '-{}In{}Out'.format(self.input_encoding, self.output_encoding)
        n += '-embsTied' if self.embs_tied else '-embsNotTied'
        if self.input_no_emb_if_leq:
            n += '-inputNoEmbIfLeq'
        if self.num_dmol > 0:
            n += '-DMoL{}'.format(self.num_dmol)
            if self.scale_input:
                n += '-scale'
        if self.dropout_p:
            n += '-dropout'
            if self.learnable_unk:
                n += '-learnableUnk'
            if self.fixed_dropout_p:
                n += '-fixedDropout{:.2f}'.format(self.dropout_p)
        if self.factor_table:
            n += '-factorized'
            if self.dropout_p and self.grouped_dropout:
                n += '-groupedDropout'
            n += '-{}wsb'.format(str(self.factor_table.word_size_bits))
        return n

    def Embed(self, data, natural_col=None, out=None):
        if data is None:
            if out is None:
                return self.unk_embeddings[natural_col]
            out.copy_(self.unk_embeddings[natural_col])
            return out

        bs = data.size()[0]
        y_embed = [None] * len(self.input_bins)
        data = data.long()

        if natural_col is not None:
            # Fast path only for inference.  One col.

            coli_dom_size = self.input_bins[natural_col]
            # Embed?
            if coli_dom_size > self.embed_size or not self.input_no_emb_if_leq:
                res = self.embeddings[natural_col](data.view(-1,))
                if out is not None:
                    out.copy_(res)
                    return out
                return res
            else:
                if out is None:
                    out = torch.zeros(bs, coli_dom_size, device=data.device)
                out.scatter_(1, data, 1)
                return out
        else:
            if self.table_dropout:
                # TODO: potential improvement: don't drop all foreign.
                assert self.learnable_unk

                if self.per_row_dropout:
                    # NOTE: torch.rand* funcs on GPU are ~4% slower than
                    # generating them on CPU via np.random.
                    num_dropped_tables = np.random.randint(
                        1, self.num_joined_tables, (bs, 1)).astype(np.float32,
                                                                   copy=False)
                    table_dropped = np.random.rand(
                        bs, self.num_joined_tables) <= (num_dropped_tables /
                                                        self.num_joined_tables)
                    if self.table_primary_index is not None:
                        table_dropped[:, self.table_primary_index] = False
                    normal_drop_rands = np.random.rand(bs, len(self.input_bins))
                    table_dropped = table_dropped.astype(np.float32, copy=False)
                else:
                    # 1 means drop that table.
                    num_dropped_tables = np.random.randint(
                        1, self.num_joined_tables)
                    table_dropped = np.random.rand(
                        self.num_joined_tables
                    ) <= num_dropped_tables / self.num_joined_tables
                    if self.table_primary_index is not None:
                        table_dropped[self.table_primary_index] = False
                    table_dropped = table_dropped.astype(np.float32, copy=False)

            if self.kOnes is None or self.kOnes.shape[0] != bs:
                with torch.no_grad():
                    self.kOnes = torch.ones(bs, 1, device=data.device)

            for i, coli_dom_size in enumerate(self.input_bins):
                # Wildcard column? use -1 as special token.
                # Inference pass only (see estimators.py).

                # Embed?
                if coli_dom_size > self.embed_size or not self.input_no_emb_if_leq:
                    col_i_embs = self.embeddings[i](
                        data[:, i])  # size is (bs, emb_size)
                    if not self.dropout_p:
                        y_embed[i] = col_i_embs
                        continue
                    elif self.grouped_dropout and self.factor_table and self.factor_table.columns[
                            i].factor_id not in [None, 0]:
                        pass  # Use previous column's batch mask
                    elif not self.table_dropout:
                        # Normal column dropout.

                        dropped_repr = torch.ones(
                            bs, self.embed_size,
                            device=data.device) / coli_dom_size
                        if self.learnable_unk:
                            dropped_repr = self.unk_embeddings[i]
                        # During training, non-dropped 1's are scaled by
                        # 1/(1-p), so we clamp back to 1.
                        def dropout_p():
                            if self.fixed_dropout_p:
                                return self.dropout_p
                            return 1. - np.random.randint(
                                1, self.nin + 1) * 1. / self.nin

                        batch_mask = torch.clamp(
                            torch.dropout(torch.ones(bs, 1, device=data.device),
                                          p=dropout_p(),
                                          train=self.training), 0, 1)
                    else:
                        # Table dropout.  Logic:
                        #  First, draw the tables to be dropped.
                        #  If a table T is dropped:
                        #    Drop its content columns & indicator only.
                        #    Don't drop its fanout.
                        #  Otherwise:
                        #    Uniformly wraw # content columns to drop.
                        #    Don't drop its indicator.
                        #    Drop its fanout.
                        table_index = self.table_indexes[i]
                        dropped_repr = self.unk_embeddings[i]
                        if self.per_row_dropout:
                            # table_dropped[table_index]: shaped [BS, 1]
                            #   elem 0 : True
                            #   elem 1 : True
                            #   elem 2 : False, etc.
                            is_content = float(self.table_column_types[i] ==
                                               common.TYPE_NORMAL_ATTR)
                            is_fanout = float(self.table_column_types[i] ==
                                              common.TYPE_FANOUT)
                            use_unk = table_dropped[:, table_index]
                            if is_fanout:
                                # Column i is a fanout column.
                                # Drop iff table not dropped.
                                batch_mask = torch.tensor(
                                    use_unk).float().unsqueeze(1).to(
                                        data.device)
                            else:
                                # Handle batch elements where this table is not
                                # dropped.
                                normal_drop_prob = np.random.randint(
                                    0, self.table_num_columns[table_index] + 1,
                                    (bs,)
                                ) * 1. / self.table_num_columns[table_index]

                                normal_drop = normal_drop_rands[:,
                                                                i] <= normal_drop_prob
                                # Make sure we drop content only.
                                normal_drop = normal_drop * is_content

                                not_dropped_pos = (use_unk == 0.0)
                                use_unk[not_dropped_pos] = normal_drop[
                                    not_dropped_pos]

                                # Shaped [bs, 1].
                                batch_mask = torch.as_tensor(
                                    1.0 - use_unk).unsqueeze(1).to(data.device)
                        else:
                            # Make decisions for entire batch.
                            if table_dropped[table_index]:
                                # Drop all its normal attributes + indicator.
                                # Don't drop fanout.
                                batch_mask = torch.clamp(
                                    torch.dropout(self.kOnes,
                                                  p=1.0 -
                                                  (self.table_column_types[i] ==
                                                   common.TYPE_FANOUT),
                                                  train=self.training), 0, 1)
                            else:
                                # Drop each normal attribute with drawn
                                # propability.
                                # Don't drop indicator.
                                # Drop fanout.
                                drop_p = 0.0
                                if self.table_column_types[
                                        i] == common.TYPE_NORMAL_ATTR:
                                    # Possible to drop all columns of this
                                    # table (it participates in join but no
                                    # attributes are filtered).
                                    drop_p = np.random.randint(
                                        0, self.table_num_columns[table_index] +
                                        1) / self.table_num_columns[table_index]
                                elif self.table_column_types[
                                        i] == common.TYPE_FANOUT:
                                    drop_p = 1.0
                                batch_mask = torch.clamp(
                                    torch.dropout(self.kOnes,
                                                  p=drop_p,
                                                  train=self.training), 0, 1)

                    # Use the column embeddings where batch_mask is 1, use
                    # unk_embs where batch_mask is 0.
                    y_embed[i] = (batch_mask * col_i_embs +
                                  (1. - batch_mask) * dropped_repr)

                else:
                    if self.learnable_unk:
                        dropped_repr = self.unk_embeddings[i]
                    else:
                        y_multihot = torch.ones(bs,
                                                coli_dom_size,
                                                device=data.device)
                        dropped_repr = y_multihot / coli_dom_size

                    y_onehot = torch.zeros(bs,
                                           coli_dom_size,
                                           device=data.device)
                    y_onehot.scatter_(1, data[:, i].view(-1, 1), 1)
                    if self.dropout_p:
                        if self.grouped_dropout and self.factor_table and self.factor_table.columns[
                                i].factor_id not in [None, 0]:
                            pass  # use prev col's batch mask
                        else:
                            # During training, non-dropped 1's are scaled by
                            # 1/(1-p), so we clamp back to 1.
                            def dropout_p():
                                if self.fixed_dropout_p:
                                    return self.dropout_p
                                return 1. - np.random.randint(
                                    1, self.nin + 1) * 1. / self.nin

                            batch_mask = torch.clamp(
                                torch.dropout(torch.ones(bs,
                                                         1,
                                                         device=data.device),
                                              p=dropout_p(),
                                              train=self.training), 0, 1)
                        y_embed[i] = (batch_mask * y_onehot +
                                      (1. - batch_mask) * dropped_repr)
                    else:
                        y_embed[i] = y_onehot
            return torch.cat(y_embed, 1)

    def ToOneHot(self, data):
        assert not self.dropout_p, 'not implemented'
        bs = data.size()[0]
        y_onehots = []
        data = data.long()
        for i, coli_dom_size in enumerate(self.input_bins):
            if coli_dom_size <= 2:
                y_onehots.append(data[:, i].view(-1, 1).float())
            else:
                y_onehot = torch.zeros(bs, coli_dom_size, device=data.device)
                y_onehot.scatter_(1, data[:, i].view(-1, 1), 1)
                y_onehots.append(y_onehot)

        # [bs, sum(dist size)]
        return torch.cat(y_onehots, 1)

    def ToOneHot_i(self, data, index):
        bs = data.size()[0]
        y_onehots = []
        data = data.long()
        coli_dom_size = self.input_bins[index]
        if coli_dom_size <= 2:
            y_onehots.append(data.view(-1, 1).float())
        else:
            y_onehot = torch.zeros(bs, coli_dom_size, device=data.device)
            y_onehot.scatter_(1, data.view(-1, 1), 1)
            y_onehots.append(y_onehot)
        return torch.cat(y_onehots, 1)

    def EncodeInput(self, data, natural_col=None, out=None):
        """"Encodes token IDs.

        Warning: this could take up a significant portion of a forward pass.

        Args:
          data: torch.Long, shaped [N, nin] or [N, 1].
          natural_col: if specified, 'data' has shape [N, 1] corresponding to
              col-'natural-col'.  Otherwise 'data' corresponds to all cols.
          out: if specified, assign results into this Tensor storage.
        """
        if self.input_encoding == 'embed':
            return self.Embed(data, natural_col=natural_col, out=out)
        elif self.input_encoding is None:
            return data
        elif self.input_encoding == 'one_hot':
            return self.ToOneHot(data)
        else:
            assert False, self.input_encoding

    def forward(self, x):
        """Calculates unnormalized logits.

        If self.input_bins is not specified, the output units are ordered as:
            [x1, x2, ..., xn], ..., [x1, x2, ..., xn].
        So they can be reshaped as thus and passed to a cross entropy loss:
            out.view(-1, model.nout // model.nin, model.nin)

        Otherwise, they are ordered as:
            [x1, ..., x1], ..., [xn, ..., xn]
        And they can't be reshaped directly.

        Args:
          x: [bs, ncols].
        """
        da_input = []
        mm_loss = torch.zeros(1, device=x.device)
        for index, m in enumerate(self.column_model):
            id_column_list = self.column2id[index]
            if m == 0:
                #tmp = self.ToOneHot_i(x[:, id_column_list[0]], index).repeat(self.sample_num, 1)
                da_input.append(x[:, id_column_list])
                #da_input.append(tmp)
            elif m == 1:
                loss, region_onehot = self.gmm_loss(x[:, id_column_list], index)
                mm_loss += loss
                region = region_onehot.nonzero()[:,1]
                #print('REGION: ', region.shape)
                da_input.append(region.unsqueeze(1).to(dtype=torch.float))
                #da_input.append(region_onehot)
            elif m == 2:
                loss, region_onehot = self.umm_loss(x[:, id_column_list[0]], index)
                mm_loss += loss
                region = region_onehot.nonzero()[:,1]
                da_input.append(region.unsqueeze(1).to(dtype=torch.float))
                #da_input.append(region_onehot)
        da_input = torch.cat(da_input, dim=1)
        #print('DA_INPUT: ', da_input.shape)
        da_input_onehot = self.EncodeInput(da_input)
        #return da_input, self.forward_with_encoded_input(da_input)
        if self.direct_io_layer is not None:
            residual = self.direct_io_layer(da_input_onehot)
            return da_input, self.net(da_input_onehot) + residual, mm_loss
        return da_input, self.net(da_input_onehot), mm_loss

    def forward_with_encoded_input(self, x):

        if self.direct_io_layer is not None:
            residual = self.direct_io_layer(x)
            return self.net(x) + residual

        return self.net(x)

    def do_forward(self, x, ordering):
        """Performs forward pass, invoking a specified ordering."""
        self.update_masks(invoke_order=ordering)
        if self.direct_io_layer is not None:
            residual = self.direct_io_layer(x)
            return self.net(x) + residual
        return self.net(x)

    def logits_for_col(self, idx, logits, out=None):
        """Returns the logits (vector) corresponding to log p(x_i | x_(<i)).

        Args:
          idx: int, in natural (table) ordering.
          logits: [batch size, hidden] where hidden can either be sum(dom
            sizes), or emb_dims.

        Returns:
          logits_for_col: [batch size, domain size for column idx].
        """
        assert self.input_bins is not None

        if idx == 0:
            logits_for_var = logits[:, :self.logit_indices[0]]
        else:
            logits_for_var = logits[:, self.logit_indices[idx - 1]:self.
                                    logit_indices[idx]]
        if self.output_encoding != 'embed' or self.UseDMoL(idx):
            return logits_for_var

        if self.embs_tied:
            embed = self.embeddings[idx]
        else:
            embed = self.embeddings_out[idx]

        if embed is None:
            # Can be None for small-domain columns.
            return logits_for_var

        # Otherwise, dot with embedding matrix to get the true logits.
        # [bs, emb] * [emb, dom size for idx]
        t = embed.weight.t()
        # Inference path will pass in output buffer, which shaves off a bit of
        # latency.
        return torch.matmul(logits_for_var, t, out=out)

    def nll(self, logits, data, label_smoothing=0):
        """Calculates -log p(data), given logits (the conditionals).

        Args:
          logits: [batch size, hidden] where hidden can either be sum(dom
            sizes), or emb_dims.
          data: [batch size, nin].

        Returns:
          nll: [batch size].
        """
        if data.dtype != torch.long:
            data = data.long()
        nll = torch.zeros(logits.size()[0], device=logits.device)
        for i in range(self.nin):
            logits_i = self.logits_for_col(i, logits)
            if not self.UseDMoL(i):

                if label_smoothing == 0:
                    loss = F.cross_entropy(logits_i,
                                           data[:, i],
                                           reduction='none')
                else:
                    log_probs_i = logits_i.log_softmax(-1)
                    with torch.no_grad():
                        true_dist = torch.zeros_like(log_probs_i)
                        true_dist.fill_(label_smoothing /
                                        (self.input_bins[i] - 1))
                        true_dist.scatter_(1, data[:, i].unsqueeze(1),
                                           1.0 - label_smoothing)
                    loss = (-true_dist * log_probs_i).sum(-1)
            else:
                loss = distributions.dmol_loss(logits_i,
                                               data[:, i],
                                               num_classes=self.input_bins[i],
                                               num_mixtures=self.num_dmol,
                                               scale_input=self.scale_input)
            assert loss.size() == nll.size()
            nll += loss
        return nll

    def gmm_loss(self, z, z_index):
        eps = 1e-20 
        para_index = self.gmm_dict[z_index]
        # K
        phi = nn.functional.softmax(self.phi[para_index], dim=0)
        
        # K x D
        mu = self.mu[para_index]
        #std = self.std_sqrt[para_index]**2
        # K x D x D 
        cov_sqrt = self.cov_sqrt[para_index]
        K, D, _ = cov_sqrt.size()
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for i in range(K):
            cov_k = cov_sqrt[i] + torch.eye(D, device=z.device) * eps
            cov_inverse.append(cov_k.mm(cov_k.t()).unsqueeze(0))
            det_cov.append(torch.det(cov_k.mm(cov_k.t())).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = torch.cat(det_cov)
        # N x K
        #z_sq = (z.unsqueeze(1) - self.mu[para_index])**2
        #e_pow = torch.exp(-z_sq / (2 * (std**2)))
        #prob = phi * (1 / (std * math.sqrt(2 * np.pi))) * e_pow
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        exp_term = torch.exp(exp_term_tmp)
        prob = phi.unsqueeze(0) * exp_term * ((2*np.pi)**(-D / 2)) * (torch.sqrt(det_cov)).unsqueeze(0)
        #print(prob)
        # N
        loss = -torch.log(torch.sum(prob, dim=1) + eps)
        # N
        #region = torch.argmax(prob, dim=1).float()
        prob_sample = prob.repeat(self.sample_num, 1)
        #print('PROB_SAMPLE: ', prob_sample.shape)
        #region_onehot = F.gumbel_softmax(prob_sample, tau=1, hard=True)
        #sample_index = torch.multinomial(prob_sample + eps, num_samples=1)
        sample_index = prob_sample.max(1, keepdim=True)[1]
        #print('SAMPLE_INDEX: ', sample_index.shape)
        y_hard = torch.zeros_like(prob_sample).scatter_(1, sample_index, 1.0)
        #print('Y_HARD: ', y_hard.shape)
        #region_onehot = (y_hard - prob_sample.detach() + prob_sample).to(dtype=torch.long)
        #print('REGION_ONEHOT: ', region_onehot.shape)
        #region = torch.multinomial(prob + eps, num_samples=1, replacement=True).squeeze(1).float()
        
        #return loss.mean(), region_onehot
        return loss.mean(), y_hard

    def umm_loss(self, z, z_index):
        torch.nan_num(z) 
        eps = 1e-6
        N = z.shape[0]
        para_index = self.umm_dict[z_index]
        # K
        '''
        phi = nn.functional.softmax(self.phi[para_index], dim=0)
        mu = self.mu[para_index]
        std = self.std_sqrt[para_index]**2
        ub = mu + math.sqrt(3) * std
        lb = mu - math.sqrt(3) * std
        '''
        weight = nn.functional.softmax(self.weight[para_index], dim=0)
        min_v = self.min_v[para_index]
        max_v = self.max_v[para_index]
        gap = self.gap[para_index]
        gap = nn.functional.softmax(gap, dim=0)
        gap_cum = torch.cumsum(gap, dim=0)
        gap_value = (max_v - min_v) * gap_cum[:-1] + min_v
        lb = torch.cat([min_v.unsqueeze(0), gap_value], dim=0)
        ub = torch.cat([gap_value, max_v.unsqueeze(0)], dim=0)
        gap = ub - lb
        
        # N x K
        is_lb = z.unsqueeze(1) - lb + eps
        is_ub = ub - z.unsqueeze(1)
        is_lb_ub = torch.max(torch.sign(is_lb * is_ub), torch.zeros(1, device=z.device))
        #gap = ub - lb
        #gap_sum = torch.sum(gap, dim=0)
        pdf =(1 / gap) / (torch.sum(1 / gap, dim=0))
        #print(pdf)
        #is_lb_ub = torch.sign(is_lb * is_ub)
        #min_gap = torch.min(torch.min(ub - lb), torch.tensor(eps, device=z.device))

        #gap = torch.max((ub - lb), torch.tensor(1e-3, device=z.device))
        #print(gap)
        #prob = phi * (1 / gap) * is_lb_ub
        prob = weight * pdf * is_lb_ub
        #prob = phi * is_lb_ub
        #print(prob)
        #print(prob)
        # N
        #range_prob = torch.sum(is_lb_ub, dim=0)
        #range_prob = range_prob / torch.sum(range_prob, dim=0)
        loss = -torch.log(torch.sum(prob, dim=1) + eps)
        #print(loss)
        #oss = 1 / (torch.sum(prob, dim=1) + eps)
        #print(loss)
        #loss = F.mse_loss(range_prob, phi, reduction='sum')
        #region = torch.argmax(prob, dim=1).float()
        prob_sample = prob.repeat(self.sample_num, 1)
        region_onehot = F.gumbel_softmax(prob_sample, tau=1, hard=True, dim=1)
        return loss.mean(), region_onehot

    def gaussian_sampling(self, num=10000):
        self.gmm_samples = []
        for i in range(len(self.phi)):
            mu = self.mu[i]
            cov_sqrt = self.cov_sqrt[i]
            K = mu.size(0)
            cov = []
            for j in range(K):
                cov_j = cov_sqrt[j]
                cov_j = torch.inverse(cov_j.mm(cov_j.t()))
                cov.append(cov_j.unsqueeze(0))
            cov = torch.cat(cov, dim=0)
            mvn = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov)
            # N x K x D
            samples = mvn.sample_n(num)
            self.gmm_samples.append(samples)
 

    def gaussian_prob(self, column_id, lb, ub):
        eps = 1e-10
        gmm_id = self.gmm_dict[column_id]
        # N x K x D
        samples = self.gmm_samples[gmm_id]      
        lb = torch.tensor(lb, dtype=torch.float, device=samples.device)
        ub = torch.tensor(ub, dtype=torch.float, device=samples.device)
        # whether each value in each dimension in each model in each sample is between lb and ub
        # 1 for in, 0 for not in
        is_in = torch.sign((ub - samples) * (samples - lb) - eps)
        # N x K
        # whether values in all dimension in each model in each sample is between lb and ub
        # 1 for in, 0 for not in
        is_in = torch.sum(is_in, dim=-1) - (samples.size(-1) - 1)
        is_in = torch.max(is_in, torch.tensor(0.0, device=is_in.device))
        # K
        # probability of [lb, ub] for each model
        prob = torch.mean(is_in, dim=0)
        return prob
 
    def sample(self, num=1, device=None):
        assert self.natural_ordering
        with torch.no_grad():
            sampled = torch.zeros((num, self.nin), device=device)
            indices = np.cumsum(self.input_bins)
            for i in range(self.nin):
                logits = self.forward(sampled)
                s = torch.multinomial(
                    torch.softmax(self.logits_for_i(i, logits), -1), 1)
                sampled[:, i] = s.view(-1,)
        return sampled


if __name__ == '__main__':
    # Checks for the autoregressive property.
    rng = np.random.RandomState(14)
    # (nin, hiddens, nout, input_bins, direct_io)
    configs_with_input_bins = [
        (2, [10], 2 + 5, [2, 5], False),
        (2, [10, 30], 2 + 5, [2, 5], False),
        (3, [6], 2 + 2 + 2, [2, 2, 2], False),
        (3, [4, 4], 2 + 1 + 2, [2, 1, 2], False),
        (4, [16, 8, 16], 2 + 3 + 1 + 2, [2, 3, 1, 2], False),
        (2, [10], 2 + 5, [2, 5], True),
        (2, [10, 30], 2 + 5, [2, 5], True),
        (3, [6], 2 + 2 + 2, [2, 2, 2], True),
        (3, [4, 4], 2 + 1 + 2, [2, 1, 2], True),
        (4, [16, 8, 16], 2 + 3 + 1 + 2, [2, 3, 1, 2], True),
    ]
    for nin, hiddens, nout, input_bins, direct_io in configs_with_input_bins:
        print(nin, hiddens, nout, input_bins, direct_io, '...', end='')
        model = MADE(nin,
                     hiddens,
                     nout,
                     input_encoding=None,
                     output_encoding='one_hot',
                     input_bins=input_bins,
                     natural_ordering=True,
                     do_direct_io_connections=direct_io)
        model.eval()
        print(model)
        for k in range(nout):
            inp = torch.tensor(rng.rand(1, nin).astype(np.float32),
                               requires_grad=True)
            loss = model(inp)
            l = loss[0, k]
            l.backward()
            depends = (inp.grad[0].numpy() != 0).astype(np.uint8)

            depends_ix = np.where(depends)[0].astype(np.int32)
            var_idx = np.argmax(k < np.cumsum(input_bins))
            prev_idxs = np.arange(var_idx).astype(np.int32)

            # Asserts that k depends only on < var_idx.
            print('depends', depends_ix, 'prev_idxs', prev_idxs)
            assert len(torch.nonzero(inp.grad[0, var_idx:])) == 0
        print('ok')
    print('[MADE] Passes autoregressive-ness check!')
