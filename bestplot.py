"""Plotting routines for displaying results of BEST test.

This module produces subplots similar to those in

Kruschke, J. (2012) Bayesian estimation supersedes the t-test
    Journal of Experimental Psychology: General.
"""
from __future__ import division
import numpy as np
from scipy.stats import gaussian_kde

from matplotlib.transforms import blended_transform_factory
import matplotlib.lines as mpllines
import matplotlib.ticker as mticker
from pymc.distributions import noncentral_t_like

def hdi(sample_vec, cred_mass = 0.95):
    assert len(sample_vec), 'need points to find HDI'
    sorted_pts = np.sort(sample_vec)

    ci_idx_inc = int(np.floor(cred_mass * len(sorted_pts)))
    n_cis = len(sorted_pts) - ci_idx_inc
    ci_width = sorted_pts[ci_idx_inc:] - sorted_pts[:n_cis]

    min_idx = np.argmin(ci_width)
    hdi_min = sorted_pts[min_idx]
    hdi_max = sorted_pts[min_idx + ci_idx_inc]

    return hdi_min, hdi_max

def calculate_sample_statistics(sample_vec):
    hdi_min, hdi_max = hdi(sample_vec)
    # calculate mean
    mean_val = np.mean(sample_vec)    
    # calculate median
    median_val = np.median(sample_vec)
    # calculate mode (use kernel density estimate)
    kernel = gaussian_kde(sample_vec)
    bw = kernel.covariance_factor()
    cut = 3 * bw
    xlow = np.min(sample_vec) - cut * bw
    xhigh = np.max(sample_vec) + cut * bw
    n = 512
    x = np.linspace(xlow, xhigh, n)
    vals = kernel.evaluate(x)
    max_idx = np.argmax(vals)
    mode_val = x[max_idx]

    return {'hdi_min':hdi_min,
            'hdi_max':hdi_max,
            'mean':mean_val,
            'median':median_val,
            'mode':mode_val,
            }


def plot_posterior( sample_vec, 
                    ax, 
                    bins = None, 
                    title = None,
                    label = '', 
                    ctd = 'ctd', 
                    draw_zero = False,
                    compVal = [], 
                    ROPE = [], 
                    colour = True, 
                    pf = '3g', 
                    printstats = True
                  ):

    if colour:   
        # for colour plots use:
        light_blue  = '#89d1ea'
        dark_green  = '#105810'
        dark_red    = '#881010'
    else:
        # for greyscale plots use:
        light_blue  = '#909090'
        dark_green  = '#101010'
        dark_red    = '#101010'

    stats = calculate_sample_statistics(sample_vec)

    if printstats:
        print title  
        print stats
        print ''
    
    hdi_min = stats['hdi_min']
    hdi_max = stats['hdi_max']

    if bins is not None:
        kwargs = {'bins':bins}
    else:
        kwargs = {}
    ax.hist(sample_vec, rwidth=0.8,
            facecolor=light_blue, edgecolor='none', **kwargs)

    if title is not None:
        ax.set_title(title, size=12, weight='bold')

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    
    ctd_string = ctd + ' = %.' + pf + ', %.' + pf + ', %.' + pf  
    ctd_data = (stats['mean'], stats['median'], stats['mode'])
    pos = 0.5
    t = ax.transAxes
    if ctd == 'mean':
        ctd_data = (stats['mean'])
        pos = stats['mean']
        if np.abs(pos) < 10:
            pf = '2f'
        ctd_string = ctd + ' = %.' + pf 
        t = trans
    if ctd == 'median':
        ctd_data = (stats['median'])
        pos = stats['median'] 
        if np.abs(pos) < 10:
            pf = '2f'
        ctd_string = ctd + ' = %.' + pf 
        t = trans
    if ctd == 'mode':
        ctd_data = (stats['mode'])
        pos = stats['mode'] 
        if np.abs(pos) < 10:
            pf = '2f'
        ctd_string = ctd + ' = %.' + pf 
        t = trans
    if ctd == 'none':
        ctd_string = '' 

    #draw central tendencies
    ax.text( pos, 1.0, ctd_string % ctd_data,
             transform=t,
             horizontalalignment='center',
             verticalalignment='top',
             )
             
    # draw zero line
    if draw_zero:
        ax.axvline(0,linestyle=':')

    # plot HDI line
    hdi_line, = ax.plot([hdi_min, hdi_max], [0,0], lw = 5.0, color = 'k')
    hdi_line.set_clip_on(False)

    # plot HDI minimum value
    if np.abs(hdi_min) < 10:
        hdi_string = '%.2f'
    else:
        hdi_string = '%.3g'
    ax.text( hdi_min, 0.04, hdi_string % hdi_min,
             transform = trans,
             horizontalalignment = 'center',
             verticalalignment = 'bottom',
           )

    # plot HDI maximum value
    if np.abs(hdi_max) < 10:
        hdi_string = '%.2f'
    else:
        hdi_string = '%.3g'
    ax.text( hdi_max, 0.04, hdi_string % hdi_max,
             transform = trans,
             horizontalalignment = 'center',
             verticalalignment = 'bottom',
           )

    # plot '95% HDI'
    ax.text( (hdi_min + hdi_max)/2, 0.20, '95% HDI',
             transform = trans,
             horizontalalignment = 'center',
             verticalalignment = 'bottom',
           )

    # plot comparative values
    if compVal!=[]:
        pcgtCompVal = round(100 * sum(sample_vec > compVal) / sample_vec.size, 1)
        pcltCompVal = 100 - pcgtCompVal
        ax.text( compVal, 0.62, '%g%% < %g < %g%%'%(pcltCompVal,compVal,pcgtCompVal),
                 transform = trans,
                 horizontalalignment = 'center',
                 verticalalignment = 'bottom',
#                 size = 9, 
#                 weight = 'bold', 
                 color = dark_green
               )
                 
    # Display the ROPE.
    if ROPE!=[]:
        ropeCol = dark_red
        pcInROPE = sum((sample_vec > ROPE[0]) & (sample_vec < ROPE[1])) / sample_vec.size
        ax.text( np.mean(ROPE), 0.42, '%g%% in ROPE' % (round(100*pcInROPE)),
                 transform = trans,
                 horizontalalignment = 'center',
                 verticalalignment = 'bottom',
#                 size = 9, 
#                 weight = 'bold',
                 color = ropeCol 
               )
        ax.axvline(ROPE[0], linestyle = '--', color = ropeCol)
        ax.axvline(ROPE[1], linestyle = '--', color = ropeCol)

        
    # make it pretty
    ax.spines['bottom'].set_position(('outward', 2))
    for loc in ['left', 'top', 'right']:
        ax.spines[loc].set_color('none')        # only draw bottom axis
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks([])                      # don't draw y-axis ticks
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins = 4))
    for line in ax.get_xticklines():
        line.set_marker(mpllines.TICKDOWN)
    if title=='Effect Size':
        ax.set_xlabel(label, size = 12, verticalalignment = 'top')
    elif title=='Normality':
        ax.set_xlabel(label, size = 15, verticalalignment = 'top')   
    else:
        ax.set_xlabel(label, size = 15, verticalalignment = 'center')


def plot_data_and_prediction( data,
                              means,
                              stds,
                              numos,
                              ax, 
                              bins = None,
                              n_curves = 50, 
                              group = 'x', 
                              name = '', 
                              colour = True,
                              plot_y = False
                            ):

    if colour:   
        # for colour plots use:
        light_blue  = '#89d1ea'
        red         = '#FF0000'
    else:
        # for greyscale plots use:
        light_blue  = '#909090'
        red         = '#101010'

    # plot histogram of data
    ax.hist(data, bins = bins, rwidth = 0.7,
            facecolor = red, edgecolor = 'none', normed = True)

    if bins is not None:
        if hasattr(bins,'__len__'):
            xmin = bins[0]
            xmax = bins[-1]
        else:
            xmin = np.min(data)
            xmax = np.max(data)

    n_samps = len(means)
    idxs = map(int, np.round(np.random.uniform(size = n_curves)*n_samps))

    x = np.linspace(xmin, xmax, 100)
    if plot_y:
        ax.set_xlabel('y', verticalalignment = 'center')
    ax.set_ylabel('p(y)')

    for i in idxs:
        m = means[i]
        s = stds[i]
        lam = 1/s**2
        numo = numos[i]
        nu = numo+1

        v = np.exp([noncentral_t_like(xi, m, lam, nu) for xi in x])
        ax.plot(x, v, color = light_blue, zorder = -10)

    ax.text(0.99,0.95,'$\mathrm{N}_{%s}= %d$' % (group, len(data),),
            transform = ax.transAxes,
            horizontalalignment = 'right',
            verticalalignment = 'top'
            )
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins = 4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins = 4))
    ax.set_title(name + ' data \nwith Predicted Posteriors', 
                 size = 12, weight = 'bold')

def plot_data( data, 
               ax, 
               bins = None,
               group = 'x', 
               name = '', 
               colour = True,
               plot_y = False
             ):

    if colour:   
        # for colour plots use:
        light_blue  = '#89d1ea'
        red         = '#FF0000'
    else:
        # for greyscale plots use:
        light_blue  = '#909090'
        red         = '#101010'

    # plot histogram of data
    ax.hist(data, bins = bins, rwidth = 0.7,
            facecolor = light_blue, edgecolor = 'none', normed = False)

    ax.text(0.99,0.95,'$\mathrm{N}_{%s}= %d$' % (group, len(data),),
            transform = ax.transAxes,
            horizontalalignment = 'right',
            verticalalignment = 'top'
            )
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins = 4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins = 4))
    ax.set_title(name + ' data ', size = 12, weight = 'bold')
