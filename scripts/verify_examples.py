#!/usr/bin/env python3
"""Run the README regression matrix on Linux or Windows (Python 3.6+)."""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', default='build', help='Directory containing train/inference, including Release on MSVC')
    parser.add_argument('--output-dir', default='build/assert_matrix')
    options = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    binaries = Path(options.build_dir).resolve()
    output = Path(options.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = []
    trained = 0
    for line in (root / 'README.md').read_text(encoding='utf-8').splitlines():
        if not line.startswith('| '):
            continue
        match = re.search(r'`(build/(?:train|inference) [^`]+)`', line)
        if not match:
            continue
        args = shlex.split(match.group(1))
        phase = Path(args[0]).name
        name = Path(args[1]).stem
        args[0] = str(binaries / (phase + ('.exe' if os.name == 'nt' else '')))
        for i, argument in enumerate(args[1:], 1):
            if argument.startswith('build/assert_matrix/'):
                args[i] = str(output / Path(argument).name)
        print('{} {}'.format(phase, name), flush=True)
        log_path = output / '{}_{}.log'.format(phase, name)
        with log_path.open('w', encoding='utf-8') as log:
            result = subprocess.run(args, cwd=str(root), stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Command failed ({}); see {}'.format(result.returncode, log_path))
        if phase == 'train':
            trained += 1
            for option in ('--output', '--pyramid-output'):
                artifact = Path(args[args.index(option) + 1])
                if not artifact.is_file() or not artifact.stat().st_size:
                    raise RuntimeError('Missing training artifact: {}'.format(artifact))
            continue
        image_path = Path(args[args.index('--output') + 1])
        if not image_path.is_file() or not image_path.stat().st_size:
            raise RuntimeError('Missing result image: {}'.format(image_path))
        data = json.loads(image_path.with_suffix('.json').read_text(encoding='utf-8'))
        actual = dict(collections.Counter(str(r['template_id']) for r in data['results']))
        cells = [cell.strip() for cell in line.split('|')]
        expected_count = int(cells[-3])
        expected = dict((key, int(value)) for key, value in re.findall(r'(\d+):(\d+)', cells[-2]))
        passed = len(data['results']) == expected_count and actual == expected
        report.append(dict(case=name, count=len(data['results']), expected_count=expected_count,
                           distribution=actual, expected_distribution=expected, passed=passed))
        # Preserve completed checks even if a later command fails.
        (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('  {} {}'.format('PASS' if passed else 'MISMATCH', actual), flush=True)
    passed = sum(row['passed'] for row in report)
    print('Trained {}; passed {}/{} inference cases'.format(trained, passed, len(report)))
    return 0 if trained == 11 and len(report) == 26 and passed == 26 else 1


if __name__ == '__main__':
    sys.exit(main())
