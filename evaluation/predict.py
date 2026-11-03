"""Load the unchanged prediction adapters for the five benchmark methods."""
from pathlib import Path
from pathscm.model import ResponseGenerator
from baselines.adapters.mean import MeanShiftGenerator
from baselines.adapters.scgen import ScGenGenerator
from baselines.adapters.crl import CRLGenerator
ROOT=Path(__file__).resolve().parents[1]

def generator_for(run,data,device,controls):
    if run.startswith('npz:'):return MeanShiftGenerator(str(ROOT/run[4:]),data,device,controls)
    if run.startswith('scgen:'):return ScGenGenerator(str(ROOT/run[6:]),data,device,controls)
    if (ROOT/'checkpoints/pathscm'/run/'response.pth').exists():
        return ResponseGenerator(ROOT/'checkpoints/pathscm'/run,device,data.node_mask,controls)
    if (ROOT/'checkpoints/crl'/run/'best_model.pt').exists():return CRLGenerator(run,data,device,controls)
    raise FileNotFoundError(f'Missing model: {run}')
