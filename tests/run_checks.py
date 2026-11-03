"""Run both numerical function tests and unittest cases, without pytest."""
import runpy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
ns=runpy.run_path(str(ROOT/'tests/test_model.py'));tests=[f for n,f in ns.items() if n.startswith('test_') and callable(f)]
for f in tests:f()
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
print(f'{len(tests)} function tests; {r.testsRun} unittest tests')
if not r.wasSuccessful():sys.exit(1)
