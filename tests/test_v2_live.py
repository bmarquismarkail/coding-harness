"""Opt-in, unassisted v0.2 coding runs against the configured llama servers."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from coding_agent import Harness
from coding_agent.sandbox import SandboxRunner
from coding_agent.types import Profile
from coding_agent.workspace import copy_source
from test_v2 import fixture, request
from test_v2_acceptance import cpp_fixture


@unittest.skipUnless(os.environ.get("RUN_V2_LIVE")=="1","set RUN_V2_LIVE=1 for unassisted v0.2 model acceptance")
class LiveCodingAcceptance(unittest.TestCase):
    def run_case(self,language):
        evidence_root=Path(os.environ.get("HARNESS_LIVE_EVIDENCE_DIR",str(Path.home()/".local/state/coding-agent-acceptance")))
        evidence_root.mkdir(parents=True,exist_ok=True)
        folder=Path(tempfile.mkdtemp(prefix=language+"-",dir=evidence_root))
        source=folder/"source"
        if language=="python":
            fixture(source); req=request(source)
        else:
            req=cpp_fixture(source)
        req=req.model_copy(update={"profile":"live-acceptance"})
        harness=Harness(folder/"runtime",config_path=Path(__file__).resolve().parents[1]/"agent-config.json",
                        profiles={"live-acceptance":{"max_actions":200,"max_active_seconds":600}})
        run_id=harness.submit(req)
        (folder/"acceptance.json").write_text(json.dumps({"run_id":run_id,"language":language,"state_dir":str(harness.store.root),"status":"running"},indent=2))
        print(f"\nLive {language} evidence: {folder}",flush=True)
        status=harness.wait(run_id,timeout=620)
        result=harness.result(run_id)
        independent=None
        if result and result.outcome=="verified":
            grade=folder/"independent"; copy_source(harness.store.run_dir(run_id)/"workspace",grade)
            runner=SandboxRunner(Profile(name="independent"))
            if language=="python":
                script='''from latency import mean_latency
for values in ([1],[1,2],[-5,5],[0,0],[0.5,1.5],list(range(100))):
    assert mean_latency(x for x in values)==sum(values)/len(values)
    assert mean_latency(values)==sum(values)/len(values)
for empty in ([],iter([]),(x for x in [])):
    try: mean_latency(empty)
    except ValueError: pass
    else: raise AssertionError("empty input must fail")
print("independent generator checks passed")
'''
                independent=runner.run(grade,[sys.executable,"-c",script])
            else:
                (grade/"independent.cpp").write_text('#include "level.hpp"\nint main(){for(int x=-1000;x<1000;++x){int want=x<0?0:x>100?100:x;if(clamp_level(x)!=want)return 1;}return clamp_level(2147483647)!=100;}\n')
                build=runner.run(grade,["g++","-std=c++17","-I.","independent.cpp","-o","build-independent"])
                independent=runner.run(grade,["./build-independent"]) if build["exit_code"]==0 else build
        with harness.store.db() as db:
            roles={json.loads(r["data"])["role"] for r in db.execute("SELECT data FROM events WHERE run_id=? AND kind='model_completed'",(run_id,))}
        passed=bool(result and result.outcome=="verified" and independent and independent["exit_code"]==0 and roles=={"main","subagent"})
        report={"run_id":run_id,"language":language,"state_dir":str(harness.store.root),"status":"passed" if passed else "failed",
                "execution":status.model_dump(),"result":result.model_dump() if result else None,"independent":independent,"roles":sorted(roles)}
        (folder/"acceptance.json").write_text(json.dumps(report,indent=2))
        self.assertTrue(passed,f"Unassisted live acceptance failed; retained evidence: {folder}; status={status.state}, error={status.error}, gaps={result.gaps if result else None}")

    def test_python_repair(self):
        self.run_case("python")

    def test_cpp_repair(self):
        self.run_case("cpp")


if __name__=="__main__":
    unittest.main()
