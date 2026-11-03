"""Numerical and leakage guards for unseen perturbation transfer."""
import numpy as np
import unittest
import tempfile
import json
from pathlib import Path
import torch
from pathscm.transfer import transfer_effects, UnseenResponseGenerator


def example():
    support = torch.tensor([[0,0,0], [1,0,0], [1,1,0], [0,0,1], [1,0,0]])
    fitted = torch.tensor([False,True,True,False,False])
    targets = torch.tensor([False,False,False,True,True])
    a = torch.tensor([[0.,0.,0.], [2,0,0], [4,6,0], [float('nan')]*3, [float('nan')]*3])
    c = torch.tensor([[0.], [3.], [9.], [float('nan')], [float('nan')]])
    return support, fitted, targets, a, c


class TestUnseenTransfer(unittest.TestCase):
    def test_split_is_by_gene_name_after_filtering_and_reordering(self):
        from data.dataset import gene_split_masks
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'split.json'
            path.write_text(json.dumps(dict(seen=['A','B'],heldout_tune=['C'],heldout_test=['D','E'])))
            seen,tune,test=gene_split_masks(['control','E','A','C'],path)
            self.assertEqual(seen.tolist(),[False,False,True,False])
            self.assertEqual(tune.tolist(),[False,False,False,True])
            self.assertEqual(test.tolist(),[False,True,False,False])
            with self.assertRaisesRegex(ValueError,'absent'):
                gene_split_masks(['control','unknown'],path)

    def test_known_jaccard_weights_both_branches_and_zero_overlap(self):
        aa, cc, W, covered = transfer_effects(*example())
        torch.testing.assert_close(W, torch.tensor([[0.,0.],[2/3,1/3]],dtype=torch.float64))
        torch.testing.assert_close(aa,torch.tensor([[0.,0.,0.],[8/3,2.,0.]]))
        torch.testing.assert_close(cc,torch.tensor([[0.],[5.]]))
        assert covered.tolist() == [False,True]
        # The user's direct weighted-a rule can put input outside target support.
        assert aa[1,1] == 2


    def test_heldout_parameters_never_influence_prediction(self):
        args=example();before=transfer_effects(*args)
        args[-2][args[2]]=1e9;args[-1][args[2]]=-1e9
        after=transfer_effects(*args)
        for x,y in zip(before,after):torch.testing.assert_close(x,y)


    def test_seen_checkpoint_cannot_be_relabelled_unseen(self):
        from pathlib import Path
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        tmp_path = Path(temp.name)
        torch.save({'scm_state':{'conditions':['control','A','B'],'held_out_conditions':[]}},tmp_path/'response.pth')
        with self.assertRaisesRegex(ValueError, 'genuinely gene-held-out'):
            UnseenResponseGenerator(tmp_path,'cpu',None,None,np.array([False,True,False]),np.array([False,False,True]))


    def test_transfer_commutes_with_linear_scm_solve(self):
        S,fitted,targets,a,c=example();aa,_,W,_=transfer_effects(S,fitted,targets,a,c)
        A=torch.tensor([[0.,.1,0.],[.2,0.,.1],[0.,.1,0.]])
        B=torch.eye(3)-A
        actual=torch.linalg.solve(B,aa.T).T
        expected=W.float()@torch.linalg.solve(B,a[fitted].T).T
        torch.testing.assert_close(actual,expected)


    def test_overlap_between_source_and_target_is_rejected(self):
        S,fitted,targets,a,c=example();targets[1]=True
        with self.assertRaisesRegex(ValueError, 'disjoint'):transfer_effects(S,fitted,targets,a,c)


if __name__ == '__main__':
    unittest.main()
