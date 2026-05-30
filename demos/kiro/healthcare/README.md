# Healthcare Drug Dosing: Kiro-shaped Spec Verification Demo

Demonstrates verifying EARS requirements (Kiro format) against implementation
with formal SMT proofs.

## Run

```sh
chelis verify-spec demos/kiro/healthcare/ --format human
```

## Expected Output

```
Verifying: demos/kiro/healthcare/

  ✓ prop_H1_001  [SMT  <50ms]
  ✓ prop_H1_002  [SMT  <50ms]
  ✓ prop_H1_003  [SMT  <50ms]
  ✓ prop_H1_004  [SMT  <50ms]
  ✓ prop_H1_005  [SMT  <50ms]

Summary: 5/5 verified
  Proved by SMT: 5
  Validated by fuzz: 0
  Failed: 0
```

## Via MCP

```json
{"jsonrpc":"2.0","method":"tools/call","params":{"name":"chelis_verify_spec","arguments":{"spec_directory":"demos/kiro/healthcare/"}},"id":1}
```
