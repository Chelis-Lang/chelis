The numeric surface census now resolves stdlib imports, aliases and recursive
generic types through compiler declaration APIs, rejecting unresolved types
instead of silently dropping numeric capacity. Existing registered stdlib
identities are preserved.
