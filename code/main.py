import sys
import config as cv_module
import numpy as np
import utils
import os

config_file = utils.import_config(sys.argv[1])
config_import = config_file.opt_parameters()

opt_config = cv_module.opt_parameters()
opt_config.__dict__ = config_import.__dict__.copy()

sampler = utils.Sampler(opt_config)
correlator = np.zeros((opt_config.N_sites, opt_config.N_sites))
magnetisation = 0
energy = 0
accepted_hex = 0
accepted_hex_single = 0
accepted_tube = 0
accepted_tube_bond = 0
accepted_bond = 0

history_len = 10000
states_history = np.zeros((history_len, opt_config.N_sites, opt_config.M))

nobs = 0

for n_iter in range(1000000):
    if n_iter < history_len:
        states_history[n_iter] = sampler.state
    else:
        states_history[:-1] = states_history[1:]; states_history[-1] = sampler.state


        A = np.sum(states_history[-1] ** 2)
        integral = np.einsum('kit,it->k', states_history, states_history[-1])
        #print(integral[-100:] / A)
        integral = np.sum(integral)

        tau_0 = integral / A

        print('autocorrelation estimate:', tau_0)

        print('FM-like:', np.einsum('ia,ib->ab', sampler.state, sampler.state).sum() / opt_config.M ** 2 / opt_config.N_sites)


        correlator += np.einsum('it,jt->ij', sampler.state, sampler.state) / opt_config.M
        print('CORRELATOR:', correlator[0, np.array([8, 16, 24, 12, 18, 25])] / (n_iter - history_len))

    if n_iter > 10000:
        correlator += sampler.state @ sampler.state.T / opt_config.M
        magnetisation += np.sum(sampler.state) / opt_config.M
        energy += sampler.energy_print(sampler.state)
        nobs += 1

        #print('MEGNETIZATION:', magnetisation / nobs)
        #print('CORRELATOR:', correlator[0, np.array([8, 16, 24, 12, 18, 25])] / nobs)
        #print('ENERGY:', energy / nobs)
        print('ENERGY CURRENT {:.6f} {:.6f}'.format(sampler.energy_print(sampler.state), energy / nobs))

    if n_iter % 5 == 1:
        hexagon = opt_config.hexagons[np.random.randint(0, len(opt_config.hexagons))]

        proposal, accepted = sampler.move_hexagonal_tube(hexagon)
        sampler.update_state(proposal, accepted)
        #if accepted:
        #    print('!!!!!!!!!!!!!!!!!!!!!!!!!! HEXAGON FLIP ACCEPTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        accepted_hex += accepted


        #print('hex acceptance:', accepted_hex / (n_iter + 1) / 0.3333)
        continue

    if n_iter % 5 == 2:
        hexagon = opt_config.hexagons[np.random.randint(0, len(opt_config.hexagons))]

        proposal, accepted = sampler.move_hexagon(hexagon, np.random.randint(0, opt_config.M))
        sampler.update_state(proposal, accepted)
        #if accepted:
        #    print('!!!!!!!!!!!!!!!!!!!!!!!!!! HEXAGON FLIP ACCEPTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        accepted_hex_single += accepted

        #print('hex acceptance single:', accepted_hex_single / (n_iter + 1) / 0.3333)
        continue


    if n_iter % 5 == 3:
        bond = opt_config.edges[0][np.random.randint(0, len(opt_config.edges[0]))]

        proposal, accepted = sampler.move_bond_tube(np.array(bond))
        sampler.update_state(proposal, accepted)
        #if accepted:
        #    print('!!!!!!!!!!!!!!!!!!!!!!!!!! BOND TUBE FLIP ACCEPTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        accepted_tube_bond += accepted

        #print('hex acceptance single:', accepted_hex_single / (n_iter + 1) / 0.3333)
        continue

    if n_iter % 5 == 4:
        bond = opt_config.edges[0][np.random.randint(0, len(opt_config.edges[0]))]

        proposal, accepted = sampler.move_bond(np.array(bond), np.random.randint(0, opt_config.M))
        sampler.update_state(proposal, accepted)
        #if accepted:
        #    print('!!!!!!!!!!!!!!!!!!!!!!!!!! BOND FLIP ACCEPTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        accepted_bond += accepted

        #print('hex acceptance single:', accepted_hex_single / (n_iter + 1) / 0.3333)
        continue


    ridx = np.random.randint(0, opt_config.N_sites)
    proposal, accepted = sampler.move_local_clusters(ridx)
    #if accepted:
    #    print(print('!!!!!!!!!!!!!!!!!!!!!!!!!! TUBE FLIP ACCEPTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'))
    sampler.update_state(proposal, accepted)

    accepted_tube += accepted

    #print('tube acceptance:', accepted_tube / (n_iter + 1) / 0.3333)

    #print('ENERGY:', sampler.energy(sampler.state))
    #print('ENERGY_PRINT', sampler.energy_print(sampler.state))
    

    print('acc bond', accepted_bond / 0.2 / (n_iter + 1))
    print('acc bond tube', accepted_tube_bond / 0.2 / (n_iter + 1))
    print('acc hex', accepted_hex_single / 0.2 / (n_iter + 1))
    print('acc hex tube', accepted_hex / 0.2 / (n_iter + 1))
    print('acc cluster', accepted_tube / 0.2 / (n_iter + 1))

    #os.system('cls' if os.name == 'nt' else 'clear')