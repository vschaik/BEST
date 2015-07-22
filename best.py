# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 00:23:13 2015

@author: Andre
"""

from __future__ import division
import pymc as pymc
import numpy as np
from bestplot import plot_posterior, plot_data_and_prediction
import matplotlib.pyplot as plt

def best(y1, y2, name1 = 'Group1', name2 = 'Group2', 
         pum = 1000.0,
         ROPE = ([-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]),
         ctd = ('mean', 'mean', 'mode', 'mode', 'mode', 'mean', 'mode', 'mode'),
         iter = 110000, burn = 10000, 
         printstats = False
         ):
    y = np.concatenate((y1, y2))        # combine both groups
    mu_m = np.mean(y)                   # mean of distribution of group means
    mu_p = 1/(pum*np.std(y))**2         # precision of distribution of group means
    sigma_low = np.std(y)/pum           # lower bound is 1000 times less than std of data
    sigma_high = np.std(y)*pum          # upper bound is 1000 times more than std of data

    group1_mean = pymc.Normal('group1_mean', mu_m, mu_p)
    group2_mean = pymc.Normal('group2_mean', mu_m, mu_p)
    group1_std = pymc.Uniform('group1_std', sigma_low, sigma_high)
    group2_std = pymc.Uniform('group2_std', sigma_low, sigma_high)
    nu_minus_one = pymc.Exponential('nu_minus_one', 1/29)

    # Normality nu = nu_minus_one + 1 
    @pymc.deterministic(plot = False)
    def nu(n = nu_minus_one):  
        out = n+1
        return out
    
    # Scale lam1 = 1/std**2
    @pymc.deterministic(plot=False)
    def lam1(s = group1_std):
        out = 1/s**2
        return out
    
    # Scale lam2 = 1/std**2    
    @pymc.deterministic(plot=False)
    def lam2(s = group2_std):
        out = 1/s**2
        return out

    group1 = pymc.NoncentralT('group1', group1_mean, lam1, nu, value = y1, observed = True)
    group2 = pymc.NoncentralT('group2', group2_mean, lam2, nu, value = y2, observed = True)

    model = pymc.Model({group1, group2, 
                        group1_mean, group2_mean, 
                        group1_std, group2_std, 
                        nu_minus_one})
    
    # Define the MCMC object on the model
    M = pymc.MCMC(model)

    M.sample(iter = iter, burn = burn, thin = 1)   
    
    n_bins = 30
    
    # Get the posterior distributions of the means and their difference
    posterior_mean1 = M.trace('group1_mean')[:]
    posterior_mean2 = M.trace('group2_mean')[:]
    diff_means = posterior_mean1 - posterior_mean2
    
    # Calculate common bin edges for both posteriors
    posterior_means = np.concatenate((posterior_mean1, posterior_mean2))
    bin_edges_means = np.linspace( np.min(posterior_means), np.max(posterior_means), n_bins )

    # Get the posteriaor distributions of the std and their difference
    posterior_std1 = M.trace('group1_std')[:]
    posterior_std2 = M.trace('group2_std')[:]
    diff_stds = posterior_std1 - posterior_std2
    
    # Calculate common bin edges for both posteriors
    posterior_stds = np.concatenate((posterior_std1, posterior_std2))
    bin_edges_stds = np.linspace(np.min(posterior_stds), np.max(posterior_stds), n_bins)
    
    # Calculate effect size
    effect_size = diff_means/np.sqrt((posterior_std1**2 + posterior_std2**2)/2)
    
    # Get the distribution of the normality parameter
    post_nu_minus_one = M.trace('nu_minus_one')[:]
    # and convert to log10 scale
    lognu = np.log10(post_nu_minus_one + 1)
    
    # Create a figure to hold 5 rows and 2 columns
    f = plt.figure(figsize = (9,12), facecolor = 'white')
    
    # Left column, top two: plot the distribution of the means
    ax1 = f.add_subplot(5, 2, 1, axisbg = 'none')
    plot_posterior( posterior_mean1, 
                    ax = ax1, 
                    bins = bin_edges_means, 
                    title = name1 + ' Mean', 
                    label = r'$\mu_1$',
                    ctd = ctd[0],
                    printstats = printstats
                  )
    
    ax3 = f.add_subplot(5, 2, 3, axisbg = 'none')
    plot_posterior( posterior_mean2, 
                    ax = ax3, 
                    bins = bin_edges_means,
                    title = name2 + ' Mean', 
                    label = r'$\mu_2$', 
                    ctd = ctd[1], 
                    printstats = printstats
                  )
    
    # Left column, next two: plot the distribution of the stds
    ax5 = f.add_subplot(5, 2, 5, axisbg = 'none')
    plot_posterior( posterior_std1, 
                    ax = ax5, 
                    bins = bin_edges_stds,
                    title = name1 + ' Std. Dev.', 
                    label = r'$\sigma_1$', 
                    ctd = ctd[2], 
                    printstats = printstats
                  )
    
    ax7 = f.add_subplot(5, 2, 7, axisbg = 'none')
    plot_posterior( posterior_std2, 
                    ax = ax7, 
                    bins = bin_edges_stds,
                    title = name2 + ' Std. Dev.',
                    label = r'$\sigma_2$', 
                    ctd = ctd[3], 
                    printstats = printstats
                  )
    
    # Left column, bottom row: Plot log10(nu)
    ax9 = f.add_subplot(5, 2, 9, axisbg = 'none')
    plot_posterior( lognu, 
                    ax = ax9, 
                    bins = n_bins,
                    title = 'Normality',
                    label = r'$\mathrm{log10}(\nu)$', 
                    ctd = ctd[4], 
                    printstats = printstats
                  )
    
    #Right column, top two: plot histogram of data and 50 of the t-distribution fits from the MCMC chain
    orig_vals = np.concatenate( (M.get_node('group1').value, M.get_node('group1').value) )
    bin_edges = np.linspace( np.min(orig_vals), np.max(orig_vals), n_bins )
    ax2 = f.add_subplot(5, 2, 2, axisbg = 'none')
    plot_data_and_prediction( M.get_node('group1').value, 
                              posterior_mean1, 
                              posterior_std1,
                              post_nu_minus_one, 
                              ax2, 
                              bins = bin_edges, 
                              group = 1, 
                              name = name1, 
                            )
    
    ax4 = f.add_subplot(5, 2, 4, axisbg = 'none', sharex = ax2, sharey = ax2)
    plot_data_and_prediction( M.get_node('group2').value, 
                              posterior_mean2, 
                              posterior_std2,
                              post_nu_minus_one, 
                              ax4, 
                              bins = bin_edges, 
                              group = 2, 
                              name = name2, 
                            )
    
    # Right column, third panel: plot the distribution of the differences of the means.
    ax6 = f.add_subplot(5, 2, 6, axisbg = 'none')
    plot_posterior( diff_means, 
                    ax = ax6, 
                    bins = n_bins,
                    title = 'Difference of Means',
                    label = r'$\mu_1 - \mu_2$',
                    ctd = ctd[5], 
                    draw_zero = True,
                    compVal = 0.0,
                    ROPE = ROPE[0], 
                    printstats = printstats
                  )
    
    # Right column, fourth panel: plot the distribution of the differences of the stds.
    ax8 = f.add_subplot(5, 2, 8, axisbg = 'none')
    plot_posterior( diff_stds, 
                    ax = ax8, 
                    bins = n_bins,
                    title = 'Difference of Std. Dev.s',
                    label = r'$\sigma_1 - \sigma_2$',
                    ctd = ctd[6], 
                    draw_zero = True,
                    compVal = 0.0,
                    ROPE = ROPE[1], 
                    printstats = printstats
                  )
    
    # Right column, bottom panel: plot the effect size
    ax10 = f.add_subplot(5, 2, 10, axisbg = 'none')
    plot_posterior( effect_size, 
                    ax = ax10, 
                    bins = n_bins,
                    title = 'Effect Size',
                    label = r'$(\mu_1 - \mu_2)/\sqrt{(\sigma_1^2 + \sigma_2^2)/2}$',
                    ctd = ctd[7], 
                    draw_zero = True,
                    compVal = 0.0,
                    ROPE = ROPE[2], 
                    printstats = printstats
                  )
    
    f.subplots_adjust(hspace = 0.5, top = 0.92, bottom = 0.09,
                      left = 0.09, right = 0.95, wspace = 0.25)
                      
    return f


def best_paired(y1, y2, name1 = 'Group1', name2 = 'Group2', 
         pum = 1000.0,
         ROPE = [-0.1, 0.1],
         ctd = ('mode', 'mode', 'mean'),
         iter = 110000, burn = 10000, 
         printstats = False
         ):
    y = y1 - y2                         # get distribution of differences
    mu_m = np.mean(y)                   # mean of distribution of group means
    mu_p = 1/(pum*np.std(y))**2         # precision of distribution of group meanssigma_low = np.std(y)/1000          # lower bound is 1000 times less than std of data
    sigma_low = np.std(y)/pum          # lower bound is 1000 times less than std of data
    sigma_high = np.std(y)*pum          # upper bound is 1000 times more than std of data
    
    group_mean = pymc.Normal('group_mean', mu_m, mu_p)
    group_std = pymc.Uniform('group_std', sigma_low, sigma_high)
    nu_minus_one = pymc.Exponential('nu_minus_one', 1/29)
    
    # Normality nu = nu_minus_one + 1 
    @pymc.deterministic(plot = False)
    def nu(n = nu_minus_one):  
        out = n+1
        return out
    
    # Scale lam = 1/std**2
    @pymc.deterministic(plot=False)
    def lam(s = group_std):
        out = 1/s**2
        return out
    
    group = pymc.NoncentralT('group', group_mean, lam, nu, value = y, observed = True)
    
    model = pymc.Model({group, group_mean, group_std, nu_minus_one})
    
    # Define the MCMC object on the model
    M = pymc.MCMC(model)
    
    M.sample(iter = 110000, burn = 10000, thin = 1) 
    
    # Create a figure to hold 2 rows and 2 columns
    f = plt.figure(figsize = (9,5), facecolor = 'white')

    n_bins = 30
    bins = np.linspace(min(y1.min(), y2.min()), max(y1.max(), y2.max()), n_bins)    

    # Left column: plot the original distributions of the data
    ax1 = f.add_subplot(2, 2, 1, axisbg = 'none')
    plot_posterior( y1, 
                    ax = ax1, 
                    bins = bins, 
                    title = r'$\,%s$'%name1,
                    label = '',
                    ctd = ctd[0], 
                    printstats = printstats
                  )
    
    ax3 = f.add_subplot(2, 2, 3, axisbg = 'none')
    plot_posterior( y2, 
                    ax = ax3, 
                    bins = bins,
                    title = r'$\,%s$'%name2, 
                    label = '',
                    ctd = ctd[1], 
                    printstats = printstats
                  )
    
    # Right column, top panel: plot histogram of data differences 
    # and 50 of the t-distribution fits from the MCMC chain
    orig_vals = M.get_node('group').value
    bin_edges = np.linspace( np.min(orig_vals), np.max(orig_vals), n_bins )
    ax2 = f.add_subplot(2, 2, 2, axisbg = 'none')
    plot_data_and_prediction( M.get_node('group').value, 
                              M.trace('group_mean')[:], 
                              M.trace('group_std')[:],
                              M.trace('nu_minus_one')[:], 
                              ax2, 
                              bins = bin_edges, 
                              group = '', 
                              name = r'$\,%s - %s$'%(name1, name2), 
                              colour = 'blue'
                            )
    
    # Right column, bottom panel: plot the posterior distribution for the mean difference
    ax4 = f.add_subplot(2, 2, 4, axisbg = 'none')
    plot_posterior( M.trace('group_mean')[:], 
                    ax = ax4, 
                    bins = n_bins,
                    title = 'Difference of Means',
                    label = r'$\mu_{%s-%s}$'%(name1, name2),
                    ctd = ctd[2], 
                    draw_zero = True,
                    compVal = 0.0,
                    ROPE = ROPE, 
                    printstats = printstats
                  )
    
    f.subplots_adjust(hspace = 0.5, top = 0.92, bottom = 0.09,
                      left = 0.09, right = 0.95, wspace = 0.25)
                      
    return f
