#!/usr/bin/env python3
"""Optional, manual cleanup of unused SDKs on a disposable public hosted runner.

Default only reports measured sizes. Never target home/workspace, model files,
credentials, package databases, active Python/runtime/toolcache or shared mounts.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess

ALLOWLIST = ('/usr/local/lib/android', '/usr/share/dotnet', '/opt/ghc')


def host_gate(env):
    if (env.get('GITHUB_ACTIONS') != 'true' or
            env.get('RUNNER_ENVIRONMENT') != 'github-hosted' or
            env.get('RUNNER_OS') != 'Linux' or
            env.get('GITHUB_REPOSITORY') != 'MusicJapanLLC/music-japan-cloud-inference' or
            env.get('GITHUB_EVENT_NAME') != 'workflow_dispatch'):
        raise ValueError('HOSTED_RUNNER_ONLY')
    workspace = env.get('GITHUB_WORKSPACE', '')
    if not workspace.startswith('/home/runner/work/') or '..' in PurePosixPath(workspace).parts:
        raise ValueError('WORKSPACE_IDENTITY')
    return workspace


def target_gate(target, workspace, mounts):
    # Provider paths are POSIX paths even when policy tests run on Windows.
    path = PurePosixPath(target)
    if str(path) not in ALLOWLIST or not path.is_absolute():
        raise ValueError('TARGET_NOT_ALLOWLISTED')
    work = PurePosixPath(workspace)
    if path == work or path in work.parents or work in path.parents:
        raise ValueError('TARGET_WORKSPACE_OVERLAP')
    for value in mounts:
        mounted = PurePosixPath(value)
        if mounted == path or path in mounted.parents:
            raise ValueError('TARGET_HAS_MOUNT')
    path = Path(target)
    resolved = path.resolve()
    if resolved != path or path.is_symlink():
        raise ValueError('TARGET_SYMLINK_OR_ALIAS')
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('TARGET_SYMLINK_ANCESTOR')
    return resolved


def mountpoints():
    # Kernel-escaped mount names: unescape spaces/tabs/backslashes before comparison.
    points = []
    for line in Path('/proc/self/mountinfo').read_text('utf8').splitlines():
        part = line.split()[4]
        for encoded, decoded in [('\\040', ' '), ('\\011', '\t'), ('\\012', '\n'), ('\\134', '\\')]:
            part = part.replace(encoded, decoded)
        points.append(part)
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    workspace = host_gate(os.environ)
    before = shutil.disk_usage(workspace).free
    entries = []
    for allowed in ALLOWLIST:
        target = target_gate(allowed, workspace, mountpoints())
        if not target.exists():
            entries.append({'target': allowed, 'status': 'ABSENT'})
            continue
        if not target.is_dir():
            raise ValueError('TARGET_NOT_DIRECTORY')
        size = int(subprocess.check_output(['du', '-sx', '-B1', '--', str(target)], text=True, timeout=60).split()[0])
        row = {'target': str(target), 'observed_bytes': size, 'status': 'DRY_RUN'}
        if args.execute:
            # Recheck the absolute target and mount boundaries immediately before mutation.
            target_gate(str(target), workspace, mountpoints())
            subprocess.run(['sudo', '-n', 'rm', '-rf', '--one-file-system', '--', str(target)],
                           check=True, timeout=120)
            row['status'] = 'REMOVED'
        entries.append(row)
    after = shutil.disk_usage(workspace).free
    print(json.dumps({'schema': 'mj.disposable-runner-sdk-cleanup/1.0', 'executed': args.execute,
                      'free_bytes_before': before, 'free_bytes_after': after,
                      'recovered_bytes': after-before, 'entries': entries}, indent=2))


if __name__ == '__main__':
    main()
