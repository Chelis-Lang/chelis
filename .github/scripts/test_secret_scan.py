"""Secret-scanning range and download contract tests."""
import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('secret_scan',Path(__file__).with_name('secret_scan.py'))
scan=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scan)


class SecretScanTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.repo=Path(self.temp.name)
        self.git('init','-b','main');self.git('config','user.email','fixture@example.invalid');self.git('config','user.name','Fixture')
        (self.repo/'fixture').write_text('initial');self.git('add','fixture');self.git('commit','-m','initial')
        self.before=self.git('rev-parse','HEAD').strip()
        (self.repo/'fixture').write_text('updated');self.git('add','fixture');self.git('commit','-m','update')
        self.after=self.git('rev-parse','HEAD').strip()

    def tearDown(self):self.temp.cleanup()

    def git(self,*args):
        return subprocess.check_output(['git','-C',str(self.repo),*args],stderr=subprocess.DEVNULL,text=True)

    def test_each_push_keeps_its_scan(self):
        workflow=(Path(__file__).parent.parent/'workflows'/'secret-scan.yml').read_text()
        # GitHub concurrency can cancel running or replace pending push scans.
        # Every push must retain the scan of its own introduced commit range.
        self.assertNotRegex(workflow,r'(?m)^[ \t]*concurrency:')
        self.assertNotIn('cancel-in-progress:',workflow)

    def test_isolated_bare_source_preserves_refs_and_detached_head(self):
        self.git('tag','retained-tag',self.before)
        self.git('checkout','--detach',self.after)
        (self.repo/'detached').write_text('detached candidate')
        self.git('add','detached');self.git('commit','-m','detached candidate')
        candidate=self.git('rev-parse','HEAD').strip()
        (self.repo/'.gitleaksignore').write_text('checkout-controlled suppression')
        with tempfile.TemporaryDirectory() as directory:
            source=scan.prepare_source(self.repo,Path(directory)/'source.git')
            self.assertFalse((source/'.gitleaksignore').exists())
            self.assertEqual(scan.git(source,'rev-parse','--is-bare-repository').stdout.strip(),'true')
            self.assertEqual(scan.git(source,'rev-parse','refs/tags/retained-tag').stdout.strip(),self.before)
            self.assertIn(candidate,scan.git(source,'rev-list','--all').stdout.splitlines())
        self.assertEqual((self.repo/'.gitleaksignore').read_text(),'checkout-controlled suppression')

    def test_push_scans_introduced_commits(self):
        selected=scan.select_range('push',{'before':self.before,'after':self.after},self.repo)
        self.assertEqual(selected,self.before+'..'+self.after)
        self.assertEqual(self.git('rev-list','--count',selected).strip(),'1')

    def test_zero_before_scans_full_history(self):
        self.assertEqual(scan.select_range('push',{'before':'0'*40,'after':self.after},self.repo),'--all')

    def test_invalid_sha_is_rejected(self):
        for value in ['--all','not-a-sha','$(anything)']:
            with self.assertRaises(ValueError):scan.select_range('push',{'before':value,'after':self.after},self.repo)

    def test_missing_before_scans_full_history(self):
        self.assertEqual(scan.select_range('push',{'before':'a'*40,'after':self.after},self.repo),'--all')

    def test_missing_after_rejects(self):
        with self.assertRaises(ValueError):scan.select_range('push',{'before':self.before,'after':'a'*40},self.repo)

    def test_pull_request_uses_merge_base(self):
        event={'pull_request':{'base':{'sha':self.before},'head':{'sha':self.after}}}
        self.assertEqual(scan.select_range('pull_request',event,self.repo),self.before+'..'+self.after)

    def test_unknown_event_is_rejected(self):
        with self.assertRaises(ValueError):scan.select_range('unexpected',{},self.repo)
        self.assertEqual(scan.select_range('workflow_dispatch',{},self.repo),'--all')

    def test_binary_download_digest_and_member_are_checked(self):
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w:gz') as archive:
            info=tarfile.TarInfo('gitleaks');info.size=7;archive.addfile(info,io.BytesIO(b'fixture'))
        data=stream.getvalue()
        with self.assertRaises(ValueError):scan.extract_verified(data,'0'*64,self.repo/'tools')
        result=scan.extract_verified(data,hashlib.sha256(data).hexdigest(),self.repo/'tools')
        self.assertEqual(result.read_bytes(),b'fixture')
        self.assertTrue(result.stat().st_mode&0o100)

    def test_command_ignores_inline_and_file_suppressions_and_redacts(self):
        args=scan.command(Path('/bin/gitleaks'),self.repo,'--all',Path('/config'),Path('/empty-ignore'))
        self.assertIn('--ignore-gitleaks-allow',args)
        self.assertIn('--redact=100',args)
        self.assertIn('--log-opts=--all --full-history -m',args)
        self.assertIn('--gitleaks-ignore-path',args)


if __name__=='__main__':unittest.main()
