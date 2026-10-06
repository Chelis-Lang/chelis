"""Secret-scanning range and download contract tests."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import tomllib
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


class AxisReceiptAllowlistTests(unittest.TestCase):
    """Public source digests may be exempted only at their exact evidence seam."""

    def setUp(self):
        self.root = Path(__file__).resolve().parents[2]
        config = tomllib.loads((self.root / '.github/security/gitleaks.toml').read_text())
        blocks = [block for block in config['allowlists']
                  if block['description'].startswith('Reviewed axis campaign source hashes;')]
        self.assertEqual(len(blocks), 1)
        self.block = blocks[0]
        self.assertEqual(self.block['condition'], 'AND')
        self.assertEqual(self.block['regexTarget'], 'match')

    def accepts(self, path, match):
        return (any(re.search(pattern, path) for pattern in self.block['paths'])
                and any(re.search(pattern, match) for pattern in self.block['regexes']))

    def test_allowlisted_values_are_exact_frozen_manifest_source_digests(self):
        evidence = self.root / 'docs/investigations/axis_oracle_comparison_results/receipts.tar.gz'
        with tarfile.open(evidence) as archive:
            manifest = json.load(archive.extractfile('raw/manifest.json'))
        candidates = []
        for path, digest in manifest['frozen_inputs'].items():
            for start in range(len(path)):
                match = path[start:] + '\": \"' + digest + '\"'
                if self.accepts(str(evidence.relative_to(self.root)) + '!raw/manifest.json', match):
                    # The independent Git-blob audit was run at the freeze. Bind
                    # exemptions to that immutable manifest, not future source.
                    self.assertRegex(digest, r'^[0-9a-f]{64}$')
                    candidates.append(match)
        self.assertEqual(len(candidates), 41)
        self.assertEqual(len(self.block['regexes']), 41)
        for pattern in self.block['regexes']:
            self.assertTrue(pattern.startswith('^') and pattern.endswith('$'))
            self.assertEqual(sum(bool(re.fullmatch(pattern, match)) for match in candidates), 1)

    def test_other_paths_and_changed_values_are_not_exempt(self):
        prefix = 'docs/investigations/axis_oracle_comparison_results/receipts.tar.gz!raw/'
        # Decode exact literal regex spellings back into the reviewed scanner matches.
        for pattern in self.block['regexes']:
            match = re.sub(r'\\(.)', r'\1', pattern[1:-1])
            self.assertTrue(self.accepts(prefix + 'manifest.json', match))
            self.assertTrue(self.accepts(prefix + 'calibration-provenance.json', match))
            self.assertFalse(self.accepts(prefix + 'credentials.json', match))
            self.assertFalse(self.accepts('manifest.json', match))
            self.assertFalse(self.accepts(prefix + 'manifest.json.bak', match))
            digest = re.search(r'[0-9a-f]{64}', match).group()
            replacement = ('0' if digest[0] != '0' else '1') + digest[1:]
            self.assertFalse(self.accepts(prefix + 'manifest.json', match.replace(digest, replacement)))
            self.assertFalse(self.accepts(prefix + 'manifest.json', 'api_key: \"synthetic-control-123456789\"'))


if __name__=='__main__':unittest.main()
