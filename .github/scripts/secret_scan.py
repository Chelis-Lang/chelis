"""Run a pinned, redacted secret scan with validated Git ranges."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import urllib.request

VERSION='8.30.1'
ASSET='gitleaks_8.30.1_linux_x64.tar.gz'
SHA256='551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb'
URL=f'https://github.com/gitleaks/gitleaks/releases/download/v{VERSION}/{ASSET}'


def git(repo,*args,check=True):
    result=subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True)
    if check and result.returncode:raise ValueError('Git history required for secret scanning is unavailable')
    return result


def valid_sha(value):
    if not isinstance(value,str) or not re.fullmatch(r'(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})',value):
        raise ValueError('Secret scanning requires a valid commit SHA')
    return value


def select_range(event_name,event,repo):
    if git(repo,'rev-parse','--is-shallow-repository').stdout.strip()!='false':
        raise ValueError('Secret scanning requires full Git history')
    if event_name=='workflow_dispatch':return '--all'
    if event_name=='push':
        before=valid_sha(event['before']);after=valid_sha(event['after'])
        if set(after)=={'0'}:
            raise ValueError('A deleted ref has no candidate history to scan')
        git(repo,'cat-file','-e',after+'^{commit}')
        if set(before)=={'0'} or git(repo,'cat-file','-e',before+'^{commit}',check=False).returncode:
            # A new ref or rewritten history has no usable push boundary. Scan
            # the candidate since its fork from the trusted default branch,
            # without including unrelated refs or old main history.
            # Tags and a rewritten default branch have no branch fork to use.
            if not event.get('ref','').startswith('refs/heads/') or event['ref']=='refs/heads/main':
                return after
            main=valid_sha(git(repo,'rev-parse','--verify','refs/remotes/origin/main^{commit}').stdout.strip())
            ancestor=git(repo,'merge-base',main,after,check=False)
            if ancestor.returncode:
                return after
            return valid_sha(ancestor.stdout.strip())+'..'+after
        return before+'..'+after
    if event_name=='pull_request':
        pr=event['pull_request'];base=valid_sha(pr['base']['sha']);head=valid_sha(pr['head']['sha'])
        git(repo,'cat-file','-e',base+'^{commit}');git(repo,'cat-file','-e',head+'^{commit}')
        ancestor=git(repo,'merge-base',base,head,check=False)
        if ancestor.returncode:return '--all'
        return valid_sha(ancestor.stdout.strip())+'..'+head
    raise ValueError('Unsupported secret scanning event')


def prepare_source(repo,destination):
    """Isolate Git history from checkout-controlled scanner configuration."""
    candidate=valid_sha(git(repo,'rev-parse','HEAD^{commit}').stdout.strip())
    git(repo,'-c','core.hooksPath=/dev/null','clone','--mirror','--shared','--',
        str(repo.resolve()),str(destination.resolve()))
    # Explicitly preserve an unadvertised, detached Actions checkout in --all.
    git(destination,'update-ref','refs/secret-scan/checked-out',candidate)
    return destination


def extract_verified(data,digest,destination):
    if hashlib.sha256(data).hexdigest()!=digest:
        raise ValueError('Secret scanner download failed checksum verification')
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        members=[x for x in archive.getmembers() if x.name in ('gitleaks','./gitleaks') and x.isfile()]
        if len(members)!=1:raise ValueError('Secret scanner archive has an unexpected layout')
        source=archive.extractfile(members[0])
        if source is None:raise ValueError('Secret scanner binary is absent')
        path=destination/'gitleaks';path.write_bytes(source.read());path.chmod(0o700)
    return path


def download_binary(destination):
    with urllib.request.urlopen(URL,timeout=120) as response:data=response.read()
    return extract_verified(data,SHA256,destination)


def command(binary,repo,selected,config,ignore):
    return [str(binary),'git',str(repo),'--config',str(config),'--gitleaks-ignore-path',str(ignore),
            '--ignore-gitleaks-allow','--max-decode-depth','8','--max-archive-depth','8',
            '--redact=100','--no-banner','--no-color','--log-opts='+selected+' --full-history -m']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event-name',default=os.environ.get('GITHUB_EVENT_NAME'))
    parser.add_argument('--event-file',type=Path,default=os.environ.get('GITHUB_EVENT_PATH'))
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--gitleaks',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--install',type=Path,help='Install only the checksum-verified scanner in this directory')
    args=parser.parse_args()
    try:
        if args.install is not None:
            download_binary(args.install)
            return 0
        event=json.loads(args.event_file.read_text()) if args.event_file else {}
        selected=select_range(args.event_name,event,args.repo)
        # Validate the entire range before calling Gitleaks: invalid ranges can exit 0.
        git(args.repo,'rev-list',*selected.split())
        with tempfile.TemporaryDirectory(prefix='secret-scan-') as directory:
            root=Path(directory)
            source=prepare_source(args.repo,root/'source.git')
            git(source,'rev-list',*selected.split())
            binary=args.gitleaks
            if binary is None:
                binary=download_binary(root/'bin')
            config=args.config
            if config is None:
                config=root/'gitleaks.toml';config.write_text('title = "Repository secret scan"\n[extend]\nuseDefault = true\n')
            ignore=root/'empty.gitleaksignore';ignore.write_text('')
            env={k:v for k,v in os.environ.items() if not k.startswith('GITLEAKS')}
            return subprocess.run(command(binary,source,selected,config,ignore),env=env).returncode
    except (ValueError,KeyError,OSError,json.JSONDecodeError,tarfile.TarError) as error:
        # Diagnostics contain no event payloads, download bodies or credentials.
        print('Secret scan could not complete: '+type(error).__name__)
        return 2


if __name__=='__main__':raise SystemExit(main())
