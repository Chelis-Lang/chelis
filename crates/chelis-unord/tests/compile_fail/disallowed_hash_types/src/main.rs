#![allow(dead_code, unused_imports, unused_variables)]

use std::collections as collections_alias;
use std::collections::HashMap as RenamedMap;
use std::collections::hash_map;
use std::collections::*;

type ForbiddenAlias = std::collections::HashMap<String, usize>;

fn fully_qualified() -> std::collections::HashMap<String, usize> {
    std::collections::HashMap::new()
}

fn module_qualified() -> hash_map::HashMap<String, usize> {
    hash_map::HashMap::new()
}

fn renamed_import() -> RenamedMap<String, usize> {
    RenamedMap::new()
}

fn glob_import() -> HashSet<String> {
    HashSet::new()
}

fn aliased_receiver() {
    let _ = collections_alias::HashMap::<String, usize>::new();
}

fn alias_definition() -> ForbiddenAlias {
    ForbiddenAlias::new()
}

fn main() {}
