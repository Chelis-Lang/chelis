use crate::*;
use std::collections::{BTreeMap, BTreeSet};

// Every value is length framed, including domain tags. No serialized host-sized
// integers, physical paths, unit indices or hash-table iteration enter a digest.
struct Canonical(ContentHasher);
impl Canonical {
    fn new(domain: &str) -> Self {
        let mut value = Self(ContentHasher::new());
        value.bytes(b"chelis.runtime.identity.v1");
        value.text(domain);
        value
    }
    fn bytes(&mut self, value: &[u8]) {
        self.0.update(&(value.len() as u64).to_le_bytes());
        self.0.update(value);
    }
    fn text(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }
    fn number(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }
    fn digest(&mut self, value: ContentDigest) {
        self.bytes(value.as_bytes());
    }
    fn optional_digest(&mut self, value: Option<ContentDigest>) {
        self.number(u64::from(value.is_some()));
        if let Some(value) = value {
            self.digest(value);
        }
    }
    fn strings(&mut self, values: &[String], ordered: bool) {
        let mut values: Vec<_> = values.iter().map(String::as_str).collect();
        if !ordered {
            values.sort_unstable();
            values.dedup();
        }
        self.number(values.len() as u64);
        for value in values {
            self.text(value);
        }
    }
    fn digests(&mut self, mut values: Vec<ContentDigest>) {
        values.sort_unstable();
        self.number(values.len() as u64);
        for value in values {
            self.digest(value);
        }
    }
    fn finish(self) -> ContentDigest {
        self.0.finish()
    }
}

fn fact(value: &str, field: &str) -> Result<(), InputError> {
    if value.trim().is_empty() || value.contains('\0') {
        return Err(InputError::InvalidObservation {
            field: field.into(),
        });
    }
    Ok(())
}

fn validate_unit(
    unit: &CompileUnit,
    required: &BTreeMap<&str, InputClass>,
) -> Result<(), InputError> {
    for (value, field) in [
        (&unit.package.name, "package.name"),
        (&unit.package.version, "package.version"),
        (&unit.package.source, "package.source"),
        (&unit.target_name, "target_name"),
        (&unit.target.triple, "target.triple"),
        (&unit.target.cpu, "target.cpu"),
        (&unit.target.llvm_triple, "target.llvm_triple"),
        (&unit.target.data_layout, "target.data_layout"),
        (&unit.configuration.opt_level, "configuration.opt_level"),
        (&unit.configuration.debuginfo, "configuration.debuginfo"),
        (&unit.configuration.panic, "configuration.panic"),
        (&unit.compiler.verbose_version, "compiler.verbose_version"),
    ] {
        fact(value, field)?;
    }
    // Cargo path package IDs use `path+file:///...`, not just `path:/...`.
    let source = unit.package.source.as_str();
    let location = source.strip_prefix("path:").unwrap_or(source);
    let bytes = location.as_bytes();
    let windows_absolute = bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'/' | b'\\');
    if location.starts_with('/')
        || location.starts_with("\\\\")
        || source.contains("file:")
        || windows_absolute
    {
        return Err(InputError::InvalidObservation {
            field: "package.source must be a logical source identity".into(),
        });
    }
    if (source.starts_with("registry+") || source.starts_with("sparse+"))
        && unit.package.checksum.is_none()
    {
        return Err(InputError::InvalidObservation {
            field: "registry package checksum".into(),
        });
    }
    for values in [
        &unit.features,
        &unit.target.features,
        &unit.configuration.cfg,
        &unit.configuration.rustflags,
    ] {
        for value in values {
            fact(value, "feature/cfg/rustflag")?;
        }
    }
    if unit.inputs.is_empty() {
        return Err(InputError::InvalidObservation {
            field: "unit.inputs".into(),
        });
    }
    let mut inputs = BTreeSet::new();
    for path in &unit.inputs {
        crate::input::validate_path(path)?;
        if !inputs.insert(path) {
            return Err(InputError::DuplicateInput { path: path.clone() });
        }
        if !required.contains_key(path.as_str()) {
            return Err(InputError::MissingInput { path: path.clone() });
        }
    }
    let mut tools = BTreeSet::new();
    for tool in &unit.compiler.tools {
        fact(&tool.name, "tool.name")?;
        fact(&tool.identity, "tool.identity")?;
        if !tools.insert(&tool.name) {
            return Err(InputError::InvalidObservation {
                field: "duplicate tool name".into(),
            });
        }
    }
    for (values, field) in [
        (&unit.build_environment, "build_environment"),
        (&unit.compiler_environment, "compiler_environment"),
    ] {
        let mut environment = BTreeSet::new();
        for variable in values {
            fact(&variable.name, "environment.name")?;
            if variable.name.contains('=')
                || !environment.insert(&variable.name)
                || variable
                    .value
                    .as_ref()
                    .is_some_and(|value| value.contains('\0'))
            {
                return Err(InputError::InvalidObservation {
                    field: field.into(),
                });
            }
        }
    }
    let mut edges = BTreeSet::new();
    for dependency in &unit.dependencies {
        fact(&dependency.name, "dependency.name")?;
        if !edges.insert(&dependency.name) {
            return Err(InputError::InvalidObservation {
                field: "duplicate dependency binding".into(),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_recipe(recipe: &RuntimeRecipe) -> Result<Vec<usize>, InputError> {
    if recipe.schema_version != 1 {
        return Err(InputError::Unsupported {
            field: "recipe schema".into(),
            value: recipe.schema_version.to_string(),
        });
    }
    if recipe.public_abi != 2 {
        return Err(InputError::Unsupported {
            field: "public ABI".into(),
            value: recipe.public_abi.to_string(),
        });
    }
    let mut required = BTreeMap::new();
    for input in &recipe.required_inputs {
        crate::input::validate_path(&input.logical_path)?;
        if required
            .insert(input.logical_path.as_str(), input.class)
            .is_some()
        {
            return Err(InputError::DuplicateInput {
                path: input.logical_path.clone(),
            });
        }
    }
    if !required.values().any(|class| *class == InputClass::Header) {
        return Err(InputError::InvalidObservation {
            field: "public header closure".into(),
        });
    }
    // Iterative postorder bounds stack memory for caller-provided deep DAGs.
    let mut state = vec![0u8; recipe.units.len()];
    let mut pending = vec![(recipe.runtime_unit, false)];
    let mut order = Vec::new();
    while let Some((index, exiting)) = pending.pop() {
        let unit = recipe
            .units
            .get(index)
            .ok_or(InputError::MissingUnit { unit: index })?;
        if exiting {
            state[index] = 2;
            order.push(index);
            continue;
        }
        match state[index] {
            2 => continue,
            1 => return Err(InputError::Cycle { unit: index }),
            _ => {}
        }
        validate_unit(unit, &required)?;
        state[index] = 1;
        pending.push((index, true));
        for dependency in &unit.dependencies {
            pending.push((dependency.unit, false));
        }
        if let Some(build_script) = unit.build_script {
            let script = recipe
                .units
                .get(build_script)
                .ok_or(InputError::MissingUnit { unit: build_script })?;
            if script.kind != UnitKind::BuildScript {
                return Err(InputError::InvalidObservation {
                    field: "build_script edge must name a build-script unit".into(),
                });
            }
            pending.push((build_script, false));
        }
    }
    if recipe.units[recipe.runtime_unit].kind != UnitKind::Library {
        return Err(InputError::InvalidObservation {
            field: "runtime unit must be a library".into(),
        });
    }
    Ok(order)
}

fn package(out: &mut Canonical, unit: &CompileUnit) {
    out.text(&unit.package.name);
    out.text(&unit.package.version);
    out.text(&unit.package.source);
    out.optional_digest(unit.package.checksum);
    out.text(&unit.target_name);
    out.text(match unit.kind {
        UnitKind::Library => "library",
        UnitKind::ProcMacro => "proc_macro",
        UnitKind::BuildScript => "build_script",
    });
}
fn target(out: &mut Canonical, value: &TargetObservation) {
    out.text(&value.triple);
    out.text(&value.cpu);
    out.strings(&value.features, false);
    out.optional_digest(value.specification);
    out.text(&value.llvm_triple);
    out.text(&value.data_layout);
}
fn configuration(out: &mut Canonical, value: &CompileConfiguration) {
    out.text(&value.opt_level);
    out.text(&value.debuginfo);
    out.number(u64::from(value.debug_assertions));
    out.text(&value.panic);
    out.strings(&value.rustflags, true);
    out.strings(&value.cfg, false);
}
fn compiler(out: &mut Canonical, value: &CompilerObservation) {
    out.text(&value.verbose_version);
    let mut tools: Vec<_> = value.tools.iter().collect();
    tools.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    out.number(tools.len() as u64);
    for tool in tools {
        out.text(&tool.name);
        out.text(&tool.identity);
    }
}
fn environment(out: &mut Canonical, values: &[EnvironmentObservation]) {
    let mut values: Vec<_> = values.iter().collect();
    values.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    out.number(values.len() as u64);
    for value in values {
        out.text(&value.name);
        out.number(u64::from(value.value.is_some()));
        if let Some(value) = &value.value {
            out.text(value);
        }
    }
}
fn class_name(class: InputClass) -> &'static str {
    match class {
        InputClass::Source => "source",
        InputClass::Build => "build",
        InputClass::Header => "header",
        InputClass::Toolchain => "toolchain",
    }
}

pub fn derive_descriptor(
    recipe: &RuntimeRecipe,
    captured: &[CapturedInput],
) -> Result<Descriptor, InputError> {
    let order = validate_recipe(recipe)?;
    let required: BTreeMap<_, _> = recipe
        .required_inputs
        .iter()
        .map(|input| (input.logical_path.as_str(), input.class))
        .collect();
    let mut inputs = BTreeMap::new();
    for input in captured {
        crate::input::validate_path(&input.logical_path)?;
        if inputs
            .insert(input.logical_path.as_str(), input.digest)
            .is_some()
        {
            return Err(InputError::DuplicateInput {
                path: input.logical_path.clone(),
            });
        }
        if !required.contains_key(input.logical_path.as_str()) {
            return Err(InputError::UnexpectedInput {
                path: input.logical_path.clone(),
            });
        }
    }
    for &path in required.keys() {
        if !inputs.contains_key(path) {
            return Err(InputError::MissingInput { path: path.into() });
        }
    }
    let mut source = Canonical::new("source");
    let mut interface = Canonical::new("interface");
    let mut target_dimension = Canonical::new("target");
    let mut features = Canonical::new("features");
    let mut compile = Canonical::new("compile");
    let mut toolchain = Canonical::new("toolchain");
    interface.number(recipe.public_abi.into());
    compile.text(match recipe.cargo_profile {
        ProfileClass::Debug => "debug",
        ProfileClass::Release => "release",
    });
    // Source is the complete closure; dedicated dimensions additionally isolate
    // header/tool changes for diagnostics without weakening the recipe binding.
    for (&path, &class) in &required {
        source.text(path);
        source.text(class_name(class));
        source.digest(inputs[path]);
        if class == InputClass::Header {
            interface.text(path);
            interface.digest(inputs[path]);
        }
        if class == InputClass::Toolchain {
            toolchain.text(path);
            toolchain.digest(inputs[path]);
        }
    }
    let mut nodes = BTreeMap::new();
    let mut source_units = Vec::new();
    let mut target_units = Vec::new();
    let mut feature_units = Vec::new();
    let mut compile_units = Vec::new();
    let mut compiler_units = Vec::new();
    for &index in &order {
        let unit = &recipe.units[index];
        let mut node = Canonical::new("unit");
        package(&mut node, unit);
        target(&mut node, &unit.target);
        node.strings(&unit.features, false);
        configuration(&mut node, &unit.configuration);
        compiler(&mut node, &unit.compiler);
        environment(&mut node, &unit.build_environment);
        environment(&mut node, &unit.compiler_environment);
        node.strings(&unit.inputs, false);
        let mut dependencies: Vec<_> = unit.dependencies.iter().collect();
        dependencies.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        node.number(dependencies.len() as u64);
        for dependency in dependencies {
            node.text(&dependency.name);
            node.digest(nodes[&dependency.unit]);
        }
        node.optional_digest(unit.build_script.map(|index| nodes[&index]));
        nodes.insert(index, node.finish());
        let mut value = Canonical::new("source-unit");
        package(&mut value, unit);
        value.strings(&unit.inputs, false);
        source_units.push(value.finish());
        let mut value = Canonical::new("target-unit");
        package(&mut value, unit);
        target(&mut value, &unit.target);
        target_units.push(value.finish());
        let mut value = Canonical::new("features-unit");
        package(&mut value, unit);
        value.strings(&unit.features, false);
        feature_units.push(value.finish());
        let mut value = Canonical::new("compile-unit");
        package(&mut value, unit);
        configuration(&mut value, &unit.configuration);
        environment(&mut value, &unit.build_environment);
        environment(&mut value, &unit.compiler_environment);
        compile_units.push(value.finish());
        let mut value = Canonical::new("toolchain-unit");
        package(&mut value, unit);
        compiler(&mut value, &unit.compiler);
        compiler_units.push(value.finish());
    }
    source.digests(source_units);
    target_dimension.digests(target_units);
    features.digests(feature_units);
    compile.digests(compile_units);
    toolchain.digests(compiler_units);
    let source = source.finish();
    let interface = interface.finish();
    let target = target_dimension.finish();
    let features = features.finish();
    let compile = compile.finish();
    let toolchain = toolchain.finish();
    let mut complete = Canonical::new("recipe");
    complete.number(recipe.schema_version.into());
    complete.number(recipe.public_abi.into());
    complete.digest(nodes[&recipe.runtime_unit]);
    complete.digests(nodes.into_values().collect());
    for digest in [source, interface, target, features, compile, toolchain] {
        complete.digest(digest);
    }
    Ok(Descriptor {
        schema_version: 1,
        recipe: complete.finish(),
        source,
        interface,
        target,
        features,
        compile,
        toolchain,
    })
}

pub fn compare(expected: &Descriptor, actual: &Descriptor) -> Result<(), IdentityMismatch> {
    let mut dimensions = Vec::new();
    if expected.schema_version != 1 || actual.schema_version != 1 {
        dimensions.push(IdentityDimension::Schema);
    }
    for (dimension, expected, actual) in [
        (IdentityDimension::Recipe, expected.recipe, actual.recipe),
        (IdentityDimension::Source, expected.source, actual.source),
        (
            IdentityDimension::Interface,
            expected.interface,
            actual.interface,
        ),
        (IdentityDimension::Target, expected.target, actual.target),
        (
            IdentityDimension::Features,
            expected.features,
            actual.features,
        ),
        (IdentityDimension::Compile, expected.compile, actual.compile),
        (
            IdentityDimension::Toolchain,
            expected.toolchain,
            actual.toolchain,
        ),
    ] {
        if expected != actual {
            dimensions.push(dimension);
        }
    }
    if dimensions.is_empty() {
        Ok(())
    } else {
        Err(IdentityMismatch { dimensions })
    }
}
