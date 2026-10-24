"""Render the factorized routing diagram from the final method definitions."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


def render(output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'mathtext.fontset': 'stix', 'svg.fonttype': 'none',
                         'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(12.6, 6.7))
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    ax.set(xlim=(0, 12.6), ylim=(0, 6.7))
    ax.axis('off')
    fit_checks=[]
    ink, supply, congestion = '#24282c', '#355b88', '#277c6c'
    def box(x, y, w, h, title, body, color, size=16):
        ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=color,lw=1.3))
        heading=ax.text(x+w/2,y+h-.22,title,ha='center',va='top',color=color,
                        weight='bold',fontsize=16)
        content=ax.text(x+w/2,y+(h-.38)/2,body,ha='center',va='center',
                        color=ink,fontsize=size,linespacing=1.2)
        fit_checks.extend([(heading,x,y,w,h),(content,x,y,w,h)])
    def arrow(a,b,color=ink):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,
                                    lw=1.15,color=color))
    for x, label in [(0.1,'a  Forecast-time inputs'),
                     (4.25,'b  State-conditioned correction'),
                     (8.5,'c  Selective final forecast')]:
        ax.text(x,6.46,label,weight='bold',fontsize=16,color=ink)
    box(.12,4.4,3.5,1.54,'Supply-price pathway',
        'Non-congestion inputs $M_t$\nSupply-state probabilities\nLow-price expert',supply)
    box(.12,2.43,3.5,1.54,'Congestion pathway',
        'Matched residual forecasts\n'+r'$\hat{\lambda}^{\rm blind}_t,\ \hat{\lambda}^{\rm full}_t$'
        +'\nRouter inputs $[M_t,C_t]$',congestion)
    box(.12,.45,3.5,1.54,'Adapted Chronos-2',
        'Price history and covariates\n'+r'$\hat q^{\rm car}_{t,\tau}$'
        +'\nShared quantile trajectory',ink)
    box(4.2,4.4,3.65,1.54,'Supply-price candidate',
        'Low / negative-price response\n'+r'$\Delta^{S,\rm cand}_t$'
        +'\nEq. (7)',supply)
    box(4.2,2.43,3.65,1.54,'Congestion reference',
        'Weighted residual increment\n'+r'$\hat\lambda^C_t,\quad\Delta^C_t=\hat\lambda^C_t-\hat q^{\rm car}_{t,0.50}$'
        +'\nEqs. (8)-(9)',congestion,16)
    box(4.2,.45,3.65,1.54,'Policy calibration',
        'Chronological holdouts\nand validation records\nSupport and risk screens',ink)
    box(8.5,4.4,3.93,1.54,'Selective acceptance',
        'State evidence and proxy mask\nAcceptance indicator $m_t$\nEqs. (10)-(12)',supply)
    box(8.5,2.43,3.93,1.54,'Final point forecast',
        'Acceptance / rejection fallback\n'+r'$\hat y_t=\max\{\hat\lambda^C_t+\Delta^S_t,-200\}$'
        +'\nEq. (15)',ink,16)
    box(8.5,.45,3.93,1.54,'Shift and market floor',
        r'$\delta_t=\hat y_t-\hat q^{\rm car}_{t,0.50}$'+'\n'
        +r'$\hat q_{t,\tau}=\max\{\hat q^{\rm car}_{t,\tau}+\delta_t,-200\}$'
        +'\n'+r'$\tau\in\{0.05,0.50,0.95\}$',ink,16)
    arrow((3.62,5.17),(4.2,5.17),supply)
    arrow((3.62,3.2),(4.2,3.2),congestion)
    arrow((7.85,5.17),(8.5,5.17),supply)
    arrow((7.85,3.2),(8.5,3.2),congestion)
    arrow((10.46,4.4),(10.46,3.97),supply)
    arrow((10.46,2.43),(10.46,1.99))
    arrow((6.03,3.97),(6.03,4.4),congestion)
    arrow((3.62,1.22),(4.2,1.22))
    ax.plot([3.62,3.88,3.88],[1.22,1.22,3.2],color=ink,lw=1.05)
    arrow((3.88,3.2),(4.2,3.2))
    ax.plot([1.87,1.87,10.46],[.45,.30,.30],color=ink,lw=1.05)
    arrow((10.46,.30),(10.46,.45))
    ax.plot([7.85,8.15,8.15],[1.22,1.22,5.17],color=ink,lw=1.05)
    arrow((8.15,5.17),(8.5,5.17))
    ax.text(.12,.04,'Price quantities are in CNY/MWh. Rejected corrections use the nonnegative reference in Eq. (15).',
            fontsize=12,color=ink)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for text,x,y,w,h in fit_checks:
        bounds=text.get_window_extent(renderer)
        left,bottom=ax.transData.transform((x+.04,y+.04))
        right,top=ax.transData.transform((x+w-.04,y+h-.04))
        assert bounds.x0>=left and bounds.x1<=right and bounds.y0>=bottom and bounds.y1<=top,text.get_text()
    for ext in ['png','pdf','svg']:
        fig.savefig(output/f'Fig2_factorized_routing.{ext}',dpi=600,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/final_revision_figures'))
    render(parser.parse_args().output)
