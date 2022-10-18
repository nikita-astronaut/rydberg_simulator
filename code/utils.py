import numpy as np
from numba import jit
import os
import sys
from copy import deepcopy

class Sampler(object):
    def __init__(self, config):
        self.config = config

        self.state = np.random.randint(0, 2, size=(self.config.N_sites, self.config.M)) * 2 - 1
        #self.state = np.tile(self.state, (1, self.config.M))

        self.E = self.energy(self.state)

        self.refresh_rnd()

    def refresh_rnd(self):
        self.buff_size = 1000000
        self.numbers = np.random.uniform(0, 1, size=self.buff_size)
        self.ctr = 0

    def return_rnd(self):
        self.ctr += 1
        if self.ctr == self.buff_size:
            self.refresh_rnd()

        return self.numbers[self.ctr]


    def energy(self, conf):
        return (1. / self.config.M) * np.trace(conf.T @ self.config.interaction @ conf) - \
               (self.config.delta / self.config.M) * np.sum(conf) - \
               self.config.Jt * np.sum(conf * np.roll(conf, shift=-1, axis=-1))

    def energy_print(self, conf):
        return (1. / self.config.M) * np.trace(conf.T @ self.config.interaction @ conf) - \
               (self.config.delta / self.config.M) * np.sum(conf) + \
               self.config.omega / 4 / self.config.M * np.sum(conf * np.roll(conf, shift=-1, axis=-1)) - self.config.omega / 4.

    # https://iopscience.iop.org/article/10.1088/1742-6596/320/1/012054/pdf
    def move_hexagonal_tube(self, hexagon):
        proposal = self.state.copy()
        #t_slice = np.random.randint(0, self.config.M)
        proposal[hexagon, :] *= -1


        dE = self.energy(proposal) - self.E
        #print('hex move, dE = ', dE)
        #print('dE_delta = ', - (self.config.delta / self.config.M) * np.sum(proposal) + (self.config.delta / self.config.M) * np.sum(self.state))
        #print('dE_J_pert = ', (1. / self.config.M) * np.diag(proposal.T @ self.config.interaction @ proposal) - (1. / self.config.M) * np.diag(self.state.T @ self.config.interaction @ self.state))

        #print('dE_Jt = ', self.config.Jt * np.sum(proposal * np.roll(proposal, shift=-1, axis=-1)) - self.config.Jt * np.sum(self.state * np.roll(self.state, shift=-1, axis=-1)))
        if self.return_rnd() < np.exp(-dE * self.config.beta):
            return proposal, True
        else:
            return self.state, False

    # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.101.210602
    def move_local_clusters(self, ridx):
        proposal = self.state.copy()
        #print('PROBA OF CLUSTER BREAK', np.exp(-2 * self.config.beta * self.config.Jt), (1. - np.exp(-2 * self.config.beta * self.config.Jt)) ** self.config.M)

        boundaries = np.random.uniform(0, 1, size=(self.config.M,)) < 1. - np.exp(-2 * self.config.beta * self.config.Jt)
        #print(np.sum(boundaries))
        boundaries *= (self.state[ridx] == np.roll(self.state[ridx], shift=-1)).astype(np.bool)
        #print(np.sum(boundaries))

        clusters = get_clusters(boundaries)
        #print(clusters)

        #print(proposal[ridx])
        #print(clusters)
        #exit(-1)

        fields = np.array([get_local_field(self.state, self.config.interaction_list_numba, cluster, ridx) for cluster in clusters])
        repr_fields = self.state[ridx, np.array([cluster[0] for cluster in clusters])]  # check that all in cluster are the same conf
        #print(boundaries)
        #print(reprs)
        lens = np.array([len(cluster) for cluster in clusters])
        

        #print(np.sum(lens), self.config.M)
        #assert np.sum(lens) == self.config.M
        ## DEBUG ##
        #for cluster in clusters:
        #    assert len(np.unique(self.state[ridx, np.array(cluster)])) == 1
        ## DEBUG ##

        ## TODO: account for the z-field
        flips = np.random.uniform(0, 1, size=(len(clusters),)) < 1. / (1 + np.exp(-2 * repr_fields * self.config.beta * fields + \
                                                                                   2 * repr_fields * self.config.beta * self.config.delta * lens))
        #print('exponentials:,', -2 * repr_fields * self.config.beta * fields + 2 * repr_fields * self.config.beta * self.config.delta * lens)
        #print(flips)
        #print('accept probas', 1. / (1 + np.exp(-2 * self.state[ridx, reprs] * self.config.beta * fields + 2 * self.state[ridx, reprs] * self.config.beta * self.config.delta * lens)))
        #print('flips:', flips)
        flipped = 0
        for cluster, flip in zip(clusters, flips):
            if flip:
                proposal[ridx, np.array(cluster)] *= -1
                flipped += len(cluster)
        return proposal, flipped / self.config.M


    def update_state(self, proposal, accepted):
        if accepted:
            self.state = proposal.copy()
            self.E = self.energy(self.state)

        
from copy import deepcopy


@jit(nopython=True)
def get_clusters(config):
    arrays : List[List[int]] = []
    current_array : List[int] = [0]

    for idx in range(len(config)):
        if config[idx] == 1:
            current_array.append(idx)
        else:
            current_array.append(idx)
            if len(current_array) > 1:
                arrays.append(current_array[1:])
                current_array = [0]
                

    if len(current_array) > 1:
        arrays.append(current_array[1:])

    #
    #if 0 in arrays[0] and len(config) - 1 in arrays[-1]:
    if config[-1] == 1 and len(arrays) > 1:
        arrays[0] = arrays[0] + arrays[-1]

        return arrays[:-1]
    return arrays


@jit(nopython=True)
def get_local_field(config, interaction_list, cluster, ridx):
    field = 0
    for c in cluster:
        for term in interaction_list[ridx]:
            field += term[1] * config[term[0], c]

    return field


def import_config(filename: str):
    import importlib

    module_name, extension = os.path.splitext(os.path.basename(filename))
    module_dir = os.path.dirname(filename)
    if extension != ".py":
        raise ValueError(
            "Could not import the module from {!r}: not a Python source file.".format(
                filename
            )
        )
    if not os.path.exists(filename):
        raise ValueError(
            "Could not import the module from {!r}: no such file or directory".format(
                filename
            )
        )
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    sys.path.pop(0)
    return module