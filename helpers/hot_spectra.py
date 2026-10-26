"""Higher-temperature cached I/Q/Pi comparisons, including the plateau region."""
import numpy as np
import matplotlib.pyplot as plt
from .spectra import load_case
from plot_style import apply_mnras_style, get_three_column_size

def plot_hot_case(temperature):
    models,audits=load_case(float(temperature),1.)
    apply_mnras_style()
    fig,axs=plt.subplots(2,3,figsize=get_three_column_size(2.6),layout='constrained')
    xmax=1000.
    for row,surface in enumerate(['up','down']):
        pi_bounds=[];q_bounds=[];i_peak=0.
        for j,window in enumerate(['th015','th045','th075']):
            for name,d,style,mc in models:
                m=(d['boundary']==surface)&(d['theta_label']==window)
                e=d['E_center_keV'][m];i=d['I_N_per_keV_sr'][m];q=d['Q_N_per_keV_sr'][m]
                pi=d['Pi_signed'][m];ef=e*e
                visible=(e>=.004)&(e<=xmax)
                # Ratios in a vanishing deterministic tail are not informative.
                reliable=visible&(ef*i>1e-12*np.max(ef*i))
                if mc:
                    supported=visible&d['plot_supported'][m]
                    reliable=supported&(d['count'][m]>=100)&np.isfinite(d['Pi_signed_se'][m])
                    errs=[ef*d['I_N_se_per_keV_sr'][m],ef*d['Q_N_se_per_keV_sr'][m],100*d['Pi_signed_se'][m]]
                    xerr=np.array([e-d['E_left_keV'][m],d['E_right_keV'][m]-e])
                else:supported=visible
                for col,values in enumerate([ef*i,ef*q,100*pi]):
                    use=reliable if col==2 else supported
                    if col==0:use=use&(values>0)
                    if mc:
                        axs[row,col].errorbar(e[use],values[use],xerr=xerr[:,use],yerr=errs[col][use],
                            fmt='o',mfc='none',ms=2.5,color=f'C{j}',alpha=.4,lw=.55)
                    else:
                        axs[row,col].plot(e,np.where(use,values,np.nan),style,color=f'C{j}',lw=1.2)
                    if not mc:
                        if col==0:i_peak=max(i_peak,float(values[use].max()))
                        if col==1:q_bounds.extend(values[use])
                        if col==2:pi_bounds.extend(values[use])
        axs[row,0].set_yscale('log');axs[row,0].set_ylim(i_peak*1e-9,i_peak*2)
        for col,bounds in [(1,q_bounds),(2,pi_bounds)]:
            lo,hi=min(bounds),max(bounds);pad=max((hi-lo)*.12,1e-12)
            axs[row,col].set_ylim(lo-pad,hi+pad);axs[row,col].axhline(0,color='.65',lw=.6)
        for ax in axs[row]:ax.set_xscale('log');ax.set_xlim(.004,xmax);ax.tick_params(labelsize=10)
        axs[row,0].set_ylabel(('Upper' if row==0 else 'Lower')+' surface')
    axs[0,0].set_title(r'$E^2 I_N$');axs[0,1].set_title(r'$E^2 Q_N$');axs[0,2].set_title(r'$\Pi=Q/I$ (%)')
    for ax in axs[-1]:ax.set_xlabel('Energy (keV)',fontsize=11)
    from matplotlib.lines import Line2D
    axs[0,0].legend(handles=[Line2D([],[],color=f'C{j}',label=t) for j,t in enumerate(['0–30°','30–60°','60–90°'])],fontsize=8,loc='lower left')
    axs[1,0].legend(handles=[Line2D([],[],color='.2',ls='-',label='Recursion + Jones'),
                            Line2D([],[],color='.2',ls='--',label='compPSc'),
                            Line2D([],[],color='.4',ls='',marker='o',mfc='none',label='MCRT')],fontsize=8,loc='lower left')
    fig.suptitle(rf'$kT_e={temperature:g}$ keV, $\tau=1$; 5 eV blackbody, Lambert injection',fontsize=12)
    plt.show();plt.close(fig)
    print(f'{temperature:g} keV: {audits["MCRT"]["total_histories"]:,} MCRT histories; '
          'fixed normalization; no new simulations or figure exports.')
