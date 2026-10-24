"""Draw the rolling-origin protocol specified in manuscript Table 5(a)."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch


def render(output):
    output.mkdir(parents=True, exist_ok=True)
    colors={'Training':'#4D899A','Validation':'#C7983D','Test':'#AD4F50'}
    with plt.rc_context({'font.family':'DejaVu Serif','font.size':10,
                         'pdf.fonttype':42,'svg.fonttype':'none',
                         'svg.hashsalt':'CCR-HQFNet-rolling-protocol'}):
        fig,ax=plt.subplots(figsize=(9.2,3.8))
        for row,shift in enumerate([0,7,14]):
            y=2-row
            for label,start,width,text in [('Training',-97+shift,90,'90 training days'),
                                          ('Validation',-7+shift,7,'7 d'),
                                          ('Test',shift,7,'7 d')]:
                ax.add_patch(Rectangle((start,y-.25),width,.5,facecolor=colors[label],
                                       edgecolor='white',linewidth=1))
                ax.text(start+width/2,y,text,ha='center',va='center',color='white',fontsize=9)
            ax.text(-100,y,f'Fold {row+1}',ha='right',va='center')
            ax.plot([shift,shift],[y-.36,y+.36],color='#424242',lw=.8)
        for y,start in [(1.48,0),(.48,7)]:
            ax.annotate('',xy=(start+7,y),xytext=(start,y),
                        arrowprops={'arrowstyle':'<->','color':'#424242','lw':.8})
            ax.text(start+3.5,y+.065,'7-day stride',ha='center',fontsize=8)
        ax.set_xlim(-111,24);ax.set_ylim(-.65,2.95)
        ax.set_xticks([-97,-69,-41,-13,0,21])
        ax.set_xticklabels(['D - 97','D - 69','D - 41','D - 13','D','D + 21'],fontsize=8)
        ax.set_yticks([])
        ax.set_xlabel('Delivery day relative to the first test origin D',labelpad=10)
        for key in ['top','right','left']:ax.spines[key].set_visible(False)
        ax.spines['bottom'].set_position(('data',-.42))
        handles=[Patch(facecolor=color,label=label) for label,color in colors.items()]
        ax.legend(handles=handles,ncol=3,loc='upper left',bbox_to_anchor=(.07,1.015),frameon=False)
        fig.subplots_adjust(left=.04,right=.99,bottom=.22,top=.94)
        for ext in ['png','pdf','svg']:
            metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
            fig.savefig(output/f'Fig6_rolling_protocol.{ext}',dpi=600,facecolor='white',metadata=metadata)
        svg=output/'Fig6_rolling_protocol.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'outputs/final_revision_figures')
    render(parser.parse_args().output)
