"""Execute exact reviewed exception boundaries with the pinned real scanner."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib
import unittest

DRIVER=Path(__file__).with_name('secret_scan.py')
CONFIG=Path(__file__).parent.parent/'security'/'gitleaks.toml'


def literal(pattern):
    if not pattern.startswith('^') or not pattern.endswith('$'):
        raise ValueError('An exception must bound the entire path or detector match')
    return re.sub(r'\\(.)',r'\1',pattern[1:-1],flags=re.DOTALL)


class ExactExceptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary=Path(os.environ['SECRET_SCAN_TEST_BINARY']).resolve()
        if not cls.binary.is_file():raise RuntimeError('The verified scanner binary is required')

    def scan(self,root,path,content,config=True):
        repo=root/'repo';repo.mkdir()
        def git(*args):
            return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.DEVNULL,text=True).strip()
        git('init','-b','main');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
        (repo/'initial').write_text('initial');git('add','.');git('commit','-m','initial');before=git('rev-parse','HEAD')
        target=repo/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content)
        git('add','.');git('commit','-m','candidate');after=git('rev-parse','HEAD')
        event=root/'event.json';event.write_text(json.dumps({'before':before,'after':after}))
        args=[os.sys.executable,str(DRIVER),'--repo',str(repo),'--event-name','push','--event-file',str(event),'--gitleaks',str(self.binary)]
        if config:args+=['--config',str(CONFIG)]
        return subprocess.run(args,capture_output=True,text=True)

    def test_reviewed_literals_and_unreviewed_paths(self):
        rows=tomllib.loads(CONFIG.read_text())['allowlists']
        for number,row in enumerate(rows):
            path=literal(row['paths'][0]);text=literal(row['regexes'][0])+'\n'
            for kind,target,configured,expected in [('default',path,False,1),('reviewed',path,True,0),('nested','nested/'+path,True,1),('suffix',path+'.extra',True,1)]:
                with self.subTest(exception=number,case=kind),tempfile.TemporaryDirectory() as directory:
                    result=self.scan(Path(directory),target,text,configured)
                    self.assertEqual(result.returncode,expected,'Exception boundary violated: '+kind)

    def test_changed_cache_values_and_new_credentials_are_detected(self):
        rows=tomllib.loads(CONFIG.read_text())['allowlists']
        token='ghp_'+hashlib.sha256(b'non-live exception boundary fixture').hexdigest()[:36]
        for number,row in enumerate(rows):
            path=literal(row['paths'][0]);text=literal(row['regexes'][0])
            variants=[text+'\nTOKEN='+token+'\n']
            if path in ('.github/workflows/ci.yml', '.github/workflows/ci-cache-warm.yml'):
                variants.append(text+hashlib.sha256(b'unreviewed cache suffix').hexdigest()+'\n')
            for content in variants:
                with self.subTest(exception=number),tempfile.TemporaryDirectory() as directory:
                    result=self.scan(Path(directory),path,content)
                    self.assertEqual(result.returncode,1,'Unreviewed content was suppressed')
                    self.assertNotIn(token,result.stdout+result.stderr)

    def test_current_smt_cache_identity_is_reviewed_at_both_exact_paths(self):
        for path in ['.github/workflows/ci.yml', '.github/workflows/ci-cache-warm.yml']:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as directory:
                result=self.scan(Path(directory),path,'shared-key: '+'smt-'+'glibc231'+'\n')
                self.assertEqual(result.returncode,0,'Public cache identity was treated as a credential')


if __name__=='__main__':unittest.main()
