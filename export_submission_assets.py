"""Export reference manuscript figures and final table transcriptions.

This exports the deposited figure assets; it does not recalculate plots.
Use the numerical and plotting entry points in OUTPUT_INDEX.md for recalculation.
"""
import argparse
from pathlib import Path
import shutil

FIGURES={'1':'image1.png','3':'image3.png','4':'image4.png','5':'image5.png',
         '7':'image13.png','8':'image14.png','9':'image9.png',
         '10':'image10.png','B1':'image11.png','C1':'image15.png'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/submission_assets'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    archive=args.archive.resolve()
    output=args.output.resolve()
    if output.is_relative_to(archive) or archive.is_relative_to(output):
        raise ValueError('Output must be separate from the frozen archive')
    figure=output/'figures'
    figure.mkdir(parents=True,exist_ok=True)
    for label,name in FIGURES.items():
        shutil.copy2(archive/'manuscript_reference'/name,figure/f'Fig{label}.png')
    shutil.copy2(root/'outputs/final_revision_figures/Fig2_factorized_routing.png',figure/'Fig2.png')
    shutil.copy2(root/'outputs/final_revision_figures/Fig6_rolling_protocol.png',figure/'Fig6.png')
    shutil.copytree(root/'outputs/final_manuscript_tables',output/'tables',dirs_exist_ok=True)
    print('Exported all 12 deposited figure assets and final table transcriptions.')

if __name__=='__main__':main()
