"""Summaries must never re-aggregate stale summary CSVs."""
import tempfile,unittest
from pathlib import Path
import pandas as pd
from evaluation.summarize import summarize
class TestAggregation(unittest.TestCase):
    def test_stale_summaries_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for seed in range(3):
                pd.DataFrame([dict(dataset='k562',method='PathSCM',seed=seed,perturbation='A',pearson_delta=seed/10,mse=.2,mmd=.3)]).to_csv(p/f'k562_PathSCM_seed{seed}.csv',index=False)
            first=summarize(p);second=summarize(p)
            pd.testing.assert_frame_equal(first,second)
            self.assertAlmostEqual(first.iloc[0].pearson_delta_mean,.1)
            self.assertAlmostEqual(first.iloc[0].pearson_delta_std,.1)
