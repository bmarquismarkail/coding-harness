"""Offline qualification reports. This module never invokes a model provider.

Usage: python -m coding_agent.qualification measurements.json
Each protocol opportunity has a list of boolean attempt-validity observations.
Each defect case supplies expected IDs, detected IDs, and unsupported critical count.
Promotion observations are booleans produced by the provider-free evidence validator.
"""
import argparse
import json
import math
from pathlib import Path
from typing import Literal
from pydantic import Field, StrictBool, model_validator
from .contracts import Contract


class DefectCase(Contract):
    id: str
    language: str
    expected: list[str] = Field(default_factory=list)
    detected: list[str] = Field(default_factory=list)
    unsupported_critical: int = Field(default=0, ge=0)


class Measurements(Contract):
    purpose: Literal['smoke', 'comparison', 'release'] = 'smoke'
    configuration: dict = Field(default_factory=dict)
    protocol: list[list[StrictBool]] = Field(default_factory=list)
    cases: list[DefectCase] = Field(default_factory=list)
    promotions: list[StrictBool] = Field(default_factory=list)

    @model_validator(mode='after')
    def unique_cases(self):
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError('duplicate benchmark case')
        if any(len(attempts) > 3 for attempts in self.protocol):
            raise ValueError('protocol samples exceed the bounded recovery limit')
        return self


def rate(successes, total):
    if total == 0:
        return {'successes': successes, 'total': total, 'observed': None, 'wilson_95': None}
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return {'successes': successes, 'total': total, 'observed': p,
            'wilson_95': [max(0, center-radius), min(1, center+radius)]}


def report(measurements):
    data = Measurements.model_validate(measurements)
    first = sum(bool(a and a[0]) for a in data.protocol)
    recovered = sum(any(a) for a in data.protocol)
    expected = sum(len(set(c.expected)) for c in data.cases)
    matched = sum(len(set(c.expected) & set(c.detected)) for c in data.cases)
    clean = [c for c in data.cases if not c.expected]
    unsupported = sum(c.unsupported_critical for c in data.cases)
    minimum = 100 if data.purpose == 'smoke' else 1000
    metrics = {'first_response': rate(first, len(data.protocol)),
               'after_recovery': rate(recovered, len(data.protocol)),
               'high_critical_recall': rate(matched, expected),
               'clean_false_positive': rate(sum(bool(c.detected) for c in clean), len(clean)),
               'valid_promotions': rate(sum(data.promotions), len(data.promotions))}
    gates = {'protocol_sample': len(data.protocol) >= minimum,
             'first_response_observed_99': bool(data.protocol) and first / len(data.protocol) >= .99,
             'recall_observed_90': expected > 0 and matched / expected >= .9,
             'clean_cases_present': bool(clean),
             'zero_unsupported_critical': unsupported == 0,
             'all_promotions_valid': bool(data.promotions) and all(data.promotions)}
    return {'version': 1, 'purpose': data.purpose, 'configuration': data.configuration,
            'metrics': metrics, 'observed_gates': gates, 'observed_gates_pass': all(gates.values()),
            'statistical_probability_99_established': bool(data.protocol) and metrics['first_response']['wilson_95'][0] >= .99,
            'cases': [c.model_dump() for c in data.cases],
            'failed_protocol_opportunities': [i for i, a in enumerate(data.protocol) if not a or not a[0]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurements', type=Path)
    args = parser.parse_args()
    print(json.dumps(report(json.loads(args.measurements.read_text())), indent=2))


if __name__ == '__main__':
    main()


def prepare_fixtures(root, *, protocol_opportunities=100):
    """Materialize fixed clean/defective tasks with separately retained gold observations."""
    from .types import CheckSpec, TaskRequest
    if protocol_opportunities < 100:
        raise ValueError("prepare at least 100 protocol opportunities")
    root = Path(root)
    requests, gold = [], []
    for language in ('python', 'c', 'cpp'):
        for faulty in (True, False):
            identity = language + ('-defect' if faulty else '-clean')
            folder = root / identity
            folder.mkdir(parents=True, exist_ok=False)
            (folder / 'tests').mkdir()
            value = 99 if faulty else 100
            if language == 'python':
                (folder/'clamp.py').write_text(f'def clamp(v):\n    return {value} if v > 100 else max(0, v)\n')
                (folder/'tests/probe.py').write_text('from clamp import clamp\nassert clamp(101) == 100\nassert clamp(-1) == 0\n')
                checks = [CheckSpec(id='acceptance', argv=['python3', '-c',
                    'exec(open("tests/probe.py").read())'])]
                path = 'clamp.py'
            else:
                path = 'clamp.h'
                (folder/path).write_text(f'static inline int clamp(int v) {{return v > 100 ? {value} : (v < 0 ? 0 : v);}}\n')
                extension = 'c' if language == 'c' else 'cpp'
                (folder/f'tests/probe.{extension}').write_text('#include "clamp.h"\nint main(void){return clamp(101)!=100 || clamp(-1)!=0;}\n')
                compiler, standard = ('gcc', '-std=c11') if language == 'c' else ('g++', '-std=c++17')
                checks = [CheckSpec(id='build', kind='build', argv=[compiler, standard, '-I.', f'tests/probe.{extension}', '-o', 'build-probe']),
                          CheckSpec(id='acceptance', argv=['./build-probe'], requires=['build'])]
            request = TaskRequest(root=str(folder.resolve()), task='Review clamp against its contract: values are clamped to [0,100].',
                                  mode='review', focus_paths=[path], checks=checks)
            requests.append({'id': identity, 'request': request.model_dump()})
            gold.append({'id': identity, 'language': language, 'expected': [identity] if faulty else [],
                         'path': path, 'mechanism': 'upper clamp returns 99 for values above 100' if faulty else None})
    actions = [
        {"type": "tool", "tool": "list_files", "args": {}},
        {"type": "tool", "tool": "read_file", "args": {"path": "clamp.py"}},
        {"type": "finish", "summary": "Protocol exercise complete", "findings": [], "gaps": []},
    ]
    opportunities = [{"id": index, "root": str((root/'python-clean').resolve()),
                      "instruction": "Return the requested action using the harness JSON action envelope.",
                      "requested_action": actions[index % len(actions)]}
                     for index in range(protocol_opportunities)]
    (root/'protocol.json').write_text(json.dumps(opportunities, indent=2)+'\n')
    (root/'requests.json').write_text(json.dumps(requests, indent=2)+'\n')
    (root/'gold.json').write_text(json.dumps(gold, indent=2)+'\n')
    return requests, gold
