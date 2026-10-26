"""Cached thermal-angular diagnostic: no transport, fitting or figure exports."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from .spectra import load_case, _read_csv
from plot_style import apply_mnras_style, get_three_column_size

DATA=Path(__file__).resolve().parents[1]/'data/thermal_angular'

def load_thermal(tau):
    manifest=json.loads((DATA/'manifest.json').read_text())
    for name,digest in manifest['files_sha256'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest,name
    return _read_csv(DATA/f'tau{tau:g}.csv',False)

def plot_comparison(tau):
    """One I/Q figure per surface, all three escape windows; display only."""
    apply_mnras_style()
    baseline,_=load_case(20.,float(tau));thermal=load_thermal(tau)
    datasets=[(baseline[0][0],baseline[0][1],'--','C0',False),
              ('Thermal angular + Jones',thermal,'-','C3',False),
              ('compPSc',baseline[1][1],':','C2',False),
              ('MCRT',baseline[2][1],None,'0.45',True)]
    for side in ['up','down']:
        fig,axs=plt.subplots(2,3,figsize=get_three_column_size(2.4),sharex=True,layout='constrained')
        for col,window in enumerate(['th015','th045','th075']):
            for row,st in enumerate('IQ'):
                ax=axs[row,col];bounds=[]
                for label,d,style,color,is_mc in datasets:
                    m=(d['boundary']==side)&(d['theta_label']==window)
                    e=d['E_center_keV'][m];y=e**2*d[f'{st}_N_per_keV_sr'][m]
                    visible=(e>=.004)&(e<=100)
                    if is_mc:
                        use=visible&(d['count'][m]>=50)
                        err=e**2*d[f'{st}_N_se_per_keV_sr'][m]
                        xe=np.array([e-d['E_left_keV'][m],d['E_right_keV'][m]-e])
                        ax.errorbar(e[use],y[use],xerr=xe[:,use],yerr=err[use],fmt='.',
                                    color=color,ms=2,lw=.5,alpha=.65,label=label)
                    else:
                        ax.plot(e,y,ls=style,color=color,lw=1.2,label=label)
                        bounds.extend(y[visible])
                ax.set_xscale('log');ax.set_xlim(.004,100)
                if st=='I':
                    ax.set_yscale('log');ax.set_ylim(max(bounds)*1e-7,max(bounds)*2)
                else:
                    span=max(bounds)-min(bounds)
                    ax.set_ylim(min(bounds)-.2*span,max(bounds)+.2*span)
                    ax.axhline(0,color='.65',lw=.6)
                if col==0:ax.set_ylabel(r'$E^2 '+st+r'_N$ (keV sr$^{-1}$ per photon)',fontsize=10)
                if row==0:ax.set_title(['0–30°','30–60°','60–90°'][col],fontsize=11)
                if row==1:ax.set_xlabel('Energy (keV)',fontsize=11)
                ax.tick_params(labelsize=10)
        axs[0,0].legend(fontsize=7.5,loc='lower left',handlelength=1.2)
        fig.suptitle(rf'$kT_e=20$ keV, $\tau={tau:g}$; '+('upper' if side=='up' else 'lower')+' surface',fontsize=12)
        plt.show();plt.close(fig)

def metric_rows():
    """Reintegrate all models into identical MCRT bins, with no amplitude fit."""
    rows=[]
    for tau in [1,10]:
        base,_=load_case(20.,float(tau));mc=base[2][1]
        models=[('Cold Thomson + Jones',base[0][1]),('Thermal angular + Jones',load_thermal(tau)),('compPSc',base[1][1])]
        for side in ['up','down']:
            for st in 'IQ':
                for band,(emin,emax) in [('0.005–100 keV',(.005,100)),('0.1–40 keV',(.1,40))]:
                    sums={label:[0.,0] for label,_ in models}
                    for window in ['th015','th045','th075']:
                        m=(mc['boundary']==side)&(mc['theta_label']==window)
                        e=mc['E_center_keV'][m];lo=mc['E_left_keV'][m];hi=mc['E_right_keV'][m]
                        y=mc[f'{st}_N_per_keV_sr'][m];se=mc[f'{st}_N_se_per_keV_sr'][m]
                        intensity=e**2*mc['I_N_per_keV_sr'][m]
                        use=(mc['count'][m]>=50)&np.isfinite(se)&(se>0)&(e>=emin)&(e<=emax)&(intensity>1e-8*intensity.max())
                        for label,d in models:
                            k=(d['boundary']==side)&(d['theta_label']==window)
                            overlap=np.maximum(0,np.minimum(hi[:,None],d['E_right_keV'][k])-np.maximum(lo[:,None],d['E_left_keV'][k]))
                            np.testing.assert_allclose(overlap.sum(axis=1),hi-lo,rtol=1e-9,atol=1e-13)
                            prediction=overlap@d[f'{st}_N_per_keV_sr'][k]/(hi-lo)
                            sums[label][0]+=float(np.sum(((prediction[use]-y[use])/se[use])**2))
                            sums[label][1]+=int(use.sum())
                    assert len(set(v[1] for v in sums.values()))==1
                    rows.append([tau,side,st,band,next(iter(sums.values()))[1],*[f'{v[0]/v[1]:.3f}' for v in sums.values()]])
    return rows
