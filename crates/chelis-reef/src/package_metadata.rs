use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;
use std::str::FromStr;
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;
use url::Url;

pub const METADATA_FILE_MAX_BYTES: usize = 4 * 1024 * 1024;
pub const METADATA_TOTAL_MAX_BYTES: usize = 2 * METADATA_FILE_MAX_BYTES;
const DESCRIPTION_MAX_BYTES: usize = 512;
const SPDX_LICENSE_MAX_BYTES: usize = 1024;
const PACKAGE_URL_MAX_BYTES: usize = 2048;
const PORTABLE_PATH_MAX_BYTES: usize = 1024;
const PORTABLE_SEGMENT_MAX_BYTES: usize = 255;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackageMetadataError {
    Description {
        rule: &'static str,
    },
    License {
        value: String,
        message: String,
    },
    LicenseConflict,
    Url {
        value: String,
        rule: &'static str,
    },
    Path {
        path: String,
        segment: Option<String>,
        rule: &'static str,
    },
    ArchiveCollision {
        declared: String,
        existing: String,
    },
    SymbolicLink {
        path: String,
        segment: String,
    },
    Open {
        path: String,
        segment: String,
        message: String,
    },
    NotRegular {
        path: String,
    },
    TooLarge {
        path: String,
        limit: usize,
        attempted: usize,
    },
    Unstable {
        path: String,
    },
    UnsupportedPlatform,
}

impl fmt::Display for PackageMetadataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Description { rule } => write!(formatter, "package description {rule}"),
            Self::License { value, message } => {
                write!(
                    formatter,
                    "license expression `{value}` is not valid SPDX: {message}"
                )
            }
            Self::LicenseConflict => formatter
                .write_str("package.license and package.license-file are exclusive alternatives"),
            Self::Url { value, rule } => write!(formatter, "package URL `{value}` {rule}"),
            Self::Path {
                path,
                segment,
                rule,
            } => {
                write!(formatter, "portable package path `{path}`")?;
                if let Some(segment) = segment {
                    write!(formatter, " segment `{segment}`")?;
                }
                write!(formatter, " {rule}")
            }
            Self::ArchiveCollision { declared, existing } => write!(
                formatter,
                "declared metadata path `{declared}` has a portable spelling collision with archive member `{existing}`"
            ),
            Self::SymbolicLink { path, segment } => write!(
                formatter,
                "declared file `{path}` rejects symbolic-link segment `{segment}`"
            ),
            Self::Open {
                path,
                segment,
                message,
            } => write!(
                formatter,
                "declared file `{path}` failed secure open at segment `{segment}`: {message}"
            ),
            Self::NotRegular { path } => {
                write!(formatter, "declared file `{path}` is not one regular file")
            }
            Self::TooLarge {
                path,
                limit,
                attempted,
            } => write!(
                formatter,
                "declared file `{path}` exceeds the {limit}-byte limit (read at least {attempted} bytes)"
            ),
            Self::Unstable { path } => write!(
                formatter,
                "declared file `{path}` is unstable because two reads from one handle differ"
            ),
            Self::UnsupportedPlatform => formatter.write_str(
                "declared package files require supported Unix no-follow relative-open APIs",
            ),
        }
    }
}

impl std::error::Error for PackageMetadataError {}

fn is_unicode_format_or_separator(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::Format
            | GeneralCategory::LineSeparator
            | GeneralCategory::ParagraphSeparator
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PackageDescription(String);

impl PackageDescription {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PackageDescription {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for PackageDescription {
    type Err = PackageMetadataError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        if value.is_empty() {
            return Err(PackageMetadataError::Description {
                rule: "must not be empty after trim",
            });
        }
        if value.len() > DESCRIPTION_MAX_BYTES {
            return Err(PackageMetadataError::Description {
                rule: "must contain at most 512 UTF-8 bytes after trim",
            });
        }
        if value.chars().any(|character| {
            character.is_control()
                || matches!(character, '\u{2028}' | '\u{2029}')
                || is_unicode_format_or_separator(character)
        }) {
            return Err(PackageMetadataError::Description {
                rule: "must not contain a line break or control character",
            });
        }
        Ok(Self(value.to_string()))
    }
}

impl<'de> Deserialize<'de> for PackageDescription {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpdxLicense(String);

impl SpdxLicense {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SpdxLicense {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for SpdxLicense {
    type Err = PackageMetadataError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() > SPDX_LICENSE_MAX_BYTES {
            return Err(PackageMetadataError::License {
                value: "<overlong>".to_string(),
                message: "the expression must contain at most 1024 UTF-8 bytes".to_string(),
            });
        }
        if value.is_empty() {
            return Err(PackageMetadataError::License {
                value: value.to_string(),
                message: "the expression is empty".to_string(),
            });
        }
        spdx::Expression::parse(value).map_err(|error| PackageMetadataError::License {
            value: value.to_string(),
            message: error.to_string(),
        })?;
        Ok(Self(value.to_string()))
    }
}

impl<'de> Deserialize<'de> for SpdxLicense {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PackageUrl(String);

impl PackageUrl {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PackageUrl {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl FromStr for PackageUrl {
    type Err = PackageMetadataError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() > PACKAGE_URL_MAX_BYTES {
            return Err(PackageMetadataError::Url {
                value: "<overlong>".to_string(),
                rule: "must contain at most 2048 UTF-8 bytes",
            });
        }
        let parsed = Url::parse(value).map_err(|_| PackageMetadataError::Url {
            value: value.to_string(),
            rule: "must be an absolute HTTPS URL with a host",
        })?;
        let explicit_https = value
            .get(.."https://".len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("https://"));
        if parsed.scheme() != "https"
            || parsed.host_str().is_none()
            || !explicit_https
            || value["https://".len()..].starts_with('/')
        {
            return Err(PackageMetadataError::Url {
                value: value.to_string(),
                rule: "must be an absolute HTTPS URL with a host",
            });
        }
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(PackageMetadataError::Url {
                value: value.to_string(),
                rule: "must not contain a username or password",
            });
        }
        let canonical = parsed.to_string();
        if canonical.len() > PACKAGE_URL_MAX_BYTES {
            return Err(PackageMetadataError::Url {
                value: "<overlong after URL normalization>".to_string(),
                rule: "must contain at most 2048 UTF-8 bytes after normalization",
            });
        }
        Ok(Self(canonical))
    }
}

impl<'de> Deserialize<'de> for PackageUrl {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PortablePackagePath {
    portable: String,
    segments: Vec<String>,
}

pub(crate) fn archive_collision_key(value: &str) -> String {
    value
        .nfc()
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

pub(crate) fn ensure_archive_spellings_do_not_collide(
    existing: &str,
    declared: &str,
) -> Result<(), PackageMetadataError> {
    if existing != declared && archive_collision_key(existing) == archive_collision_key(declared) {
        return Err(PackageMetadataError::ArchiveCollision {
            declared: declared.to_string(),
            existing: existing.to_string(),
        });
    }
    Ok(())
}

#[cfg(feature = "test-hooks")]
#[doc(hidden)]
pub fn check_archive_spelling_collision_for_test(
    existing: &str,
    declared: &str,
) -> Result<(), PackageMetadataError> {
    ensure_archive_spellings_do_not_collide(existing, declared)
}

impl PortablePackagePath {
    pub fn as_str(&self) -> &str {
        &self.portable
    }

    pub fn segments(&self) -> &[String] {
        &self.segments
    }
}

impl fmt::Display for PortablePackagePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.portable.fmt(formatter)
    }
}

impl FromStr for PortablePackagePath {
    type Err = PackageMetadataError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let path_error = |segment: Option<&str>, rule| PackageMetadataError::Path {
            path: value.to_string(),
            segment: segment.map(str::to_string),
            rule,
        };
        if value.is_empty() {
            return Err(path_error(None, "must not be empty"));
        }
        if value.len() > PORTABLE_PATH_MAX_BYTES {
            return Err(path_error(None, "must contain at most 1024 UTF-8 bytes"));
        }
        if value.contains('\\') {
            return Err(path_error(None, "must use `/` as its only separator"));
        }
        if value
            .chars()
            .any(|character| character.is_control() || is_unicode_format_or_separator(character))
        {
            return Err(path_error(
                None,
                "must not contain a control, format, or line-separator character",
            ));
        }
        if value.nfc().collect::<String>() != value {
            return Err(path_error(None, "must use Unicode NFC"));
        }
        if value.eq_ignore_ascii_case("reef.toml") || value.eq_ignore_ascii_case("reef.lock") {
            return Err(path_error(None, "must not name a reserved Reef document"));
        }

        let mut segments = Vec::new();
        for segment in value.split('/') {
            if segment.is_empty() {
                return Err(path_error(Some(segment), "must not be empty"));
            }
            if segment == "." || segment == ".." {
                return Err(path_error(Some(segment), "must not be `.` or `..`"));
            }
            if segment.len() > PORTABLE_SEGMENT_MAX_BYTES {
                return Err(path_error(
                    Some(segment),
                    "must contain at most 255 UTF-8 bytes",
                ));
            }
            if segment
                .chars()
                .any(|character| matches!(character, '<' | '>' | ':' | '"' | '|' | '?' | '*'))
            {
                return Err(path_error(
                    Some(segment),
                    "must not contain a platform-reserved character",
                ));
            }
            if segment.ends_with(' ') || segment.ends_with('.') {
                return Err(path_error(
                    Some(segment),
                    "must not end in a space or period",
                ));
            }
            let device_base = segment
                .split('.')
                .next()
                .unwrap_or(segment)
                .trim_end_matches([' ', '.'])
                .to_ascii_lowercase();
            if matches!(device_base.as_str(), "con" | "prn" | "aux" | "nul")
                || device_base.strip_prefix("com").is_some_and(|suffix| {
                    matches!(suffix, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
                })
                || device_base.strip_prefix("lpt").is_some_and(|suffix| {
                    matches!(suffix, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
                })
            {
                return Err(path_error(
                    Some(segment),
                    "must not use a Windows device name",
                ));
            }
            segments.push(segment.to_string());
        }
        Ok(Self {
            portable: value.to_string(),
            segments,
        })
    }
}

impl Serialize for PortablePackagePath {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.portable)
    }
}

impl<'de> Deserialize<'de> for PortablePackagePath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct PackageMetadata {
    description: Option<PackageDescription>,
    license: Option<SpdxLicense>,
    #[serde(rename = "license-file")]
    license_file: Option<PortablePackagePath>,
    repository: Option<PackageUrl>,
    documentation: Option<PackageUrl>,
    homepage: Option<PackageUrl>,
    readme: Option<PortablePackagePath>,
}

impl PackageMetadata {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        description: Option<PackageDescription>,
        license: Option<SpdxLicense>,
        license_file: Option<PortablePackagePath>,
        repository: Option<PackageUrl>,
        documentation: Option<PackageUrl>,
        homepage: Option<PackageUrl>,
        readme: Option<PortablePackagePath>,
    ) -> Result<Self, PackageMetadataError> {
        if license.is_some() && license_file.is_some() {
            return Err(PackageMetadataError::LicenseConflict);
        }
        Ok(Self {
            description,
            license,
            license_file,
            repository,
            documentation,
            homepage,
            readme,
        })
    }

    pub fn description(&self) -> Option<&PackageDescription> {
        self.description.as_ref()
    }

    pub fn license(&self) -> Option<&SpdxLicense> {
        self.license.as_ref()
    }

    pub fn license_file(&self) -> Option<&PortablePackagePath> {
        self.license_file.as_ref()
    }

    pub fn repository(&self) -> Option<&PackageUrl> {
        self.repository.as_ref()
    }

    pub fn documentation(&self) -> Option<&PackageUrl> {
        self.documentation.as_ref()
    }

    pub fn homepage(&self) -> Option<&PackageUrl> {
        self.homepage.as_ref()
    }

    pub fn readme(&self) -> Option<&PortablePackagePath> {
        self.readme.as_ref()
    }
}

#[derive(Deserialize)]
struct PackageMetadataWire {
    #[serde(default)]
    description: Option<PackageDescription>,
    #[serde(default)]
    license: Option<SpdxLicense>,
    #[serde(default, rename = "license-file")]
    license_file: Option<PortablePackagePath>,
    #[serde(default)]
    repository: Option<PackageUrl>,
    #[serde(default)]
    documentation: Option<PackageUrl>,
    #[serde(default)]
    homepage: Option<PackageUrl>,
    #[serde(default)]
    readme: Option<PortablePackagePath>,
}

impl<'de> Deserialize<'de> for PackageMetadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PackageMetadataWire::deserialize(deserializer)?;
        Self::new(
            wire.description,
            wire.license,
            wire.license_file,
            wire.repository,
            wire.documentation,
            wire.homepage,
            wire.readme,
        )
        .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredFileSnapshot {
    path: PortablePackagePath,
    bytes: Vec<u8>,
}

impl DeclaredFileSnapshot {
    pub fn path(&self) -> &PortablePackagePath {
        &self.path
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn into_parts(self) -> (PortablePackagePath, Vec<u8>) {
        (self.path, self.bytes)
    }
}

fn read_bounded(
    file: &mut File,
    path: &PortablePackagePath,
) -> Result<Vec<u8>, PackageMetadataError> {
    file.rewind().map_err(|error| PackageMetadataError::Open {
        path: path.to_string(),
        segment: path.segments.last().cloned().unwrap_or_default(),
        message: format!("seek to start: {error}"),
    })?;
    let mut bytes = Vec::new();
    file.take((METADATA_FILE_MAX_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| PackageMetadataError::Open {
            path: path.to_string(),
            segment: path.segments.last().cloned().unwrap_or_default(),
            message: format!("read: {error}"),
        })?;
    if bytes.len() > METADATA_FILE_MAX_BYTES {
        return Err(PackageMetadataError::TooLarge {
            path: path.to_string(),
            limit: METADATA_FILE_MAX_BYTES,
            attempted: bytes.len(),
        });
    }
    Ok(bytes)
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn secure_open_error(
    path: &PortablePackagePath,
    segment: &str,
    error: rustix::io::Errno,
) -> PackageMetadataError {
    if error == rustix::io::Errno::LOOP {
        PackageMetadataError::SymbolicLink {
            path: path.to_string(),
            segment: segment.to_string(),
        }
    } else {
        PackageMetadataError::Open {
            path: path.to_string(),
            segment: segment.to_string(),
            message: error.to_string(),
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn open_declared_file_with_hook<F>(
    root: &Path,
    path: &PortablePackagePath,
    mut before_segment_open: F,
) -> Result<File, PackageMetadataError>
where
    F: FnMut(usize),
{
    use rustix::fs::{Mode, OFlags, fcntl_getfl, fcntl_setfl, open, openat};

    let directory_flags =
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::DIRECTORY | OFlags::NONBLOCK;
    let mut directory = open(root, directory_flags, Mode::empty())
        .map_err(|error| secure_open_error(path, "<package-root>", error))?;
    for (index, segment) in path.segments[..path.segments.len() - 1].iter().enumerate() {
        before_segment_open(index);
        directory = openat(&directory, segment.as_str(), directory_flags, Mode::empty())
            .map_err(|error| secure_open_error(path, segment, error))?;
    }
    let final_segment = path.segments.last().expect("portable path is nonempty");
    before_segment_open(path.segments.len() - 1);
    let file = openat(
        &directory,
        final_segment.as_str(),
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(|error| secure_open_error(path, final_segment, error))?;
    let file = File::from(file);
    let metadata = file
        .metadata()
        .map_err(|error| PackageMetadataError::Open {
            path: path.to_string(),
            segment: final_segment.clone(),
            message: format!("inspect open handle: {error}"),
        })?;
    if !metadata.file_type().is_file() {
        return Err(PackageMetadataError::NotRegular {
            path: path.to_string(),
        });
    }
    let mut status = fcntl_getfl(&file).map_err(|error| PackageMetadataError::Open {
        path: path.to_string(),
        segment: final_segment.clone(),
        message: format!("read file status flags: {error}"),
    })?;
    status.remove(OFlags::NONBLOCK);
    fcntl_setfl(&file, status).map_err(|error| PackageMetadataError::Open {
        path: path.to_string(),
        segment: final_segment.clone(),
        message: format!("clear nonblocking flag: {error}"),
    })?;
    Ok(file)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn open_declared_file_with_hook<F>(
    _root: &Path,
    _path: &PortablePackagePath,
    _before_segment_open: F,
) -> Result<File, PackageMetadataError>
where
    F: FnMut(usize),
{
    Err(PackageMetadataError::UnsupportedPlatform)
}

pub fn snapshot_declared_metadata_file(
    root: &Path,
    path: &PortablePackagePath,
) -> Result<DeclaredFileSnapshot, PackageMetadataError> {
    let mut file = open_declared_file_with_hook(root, path, |_| {})?;
    let first = read_bounded(&mut file, path)?;
    let second = read_bounded(&mut file, path)?;
    if first != second {
        return Err(PackageMetadataError::Unstable {
            path: path.to_string(),
        });
    }
    Ok(DeclaredFileSnapshot {
        path: path.clone(),
        bytes: first,
    })
}

#[cfg(feature = "test-hooks")]
#[doc(hidden)]
pub fn snapshot_declared_metadata_file_with_walk_hook<F>(
    root: &Path,
    path: &PortablePackagePath,
    before_segment_open: F,
) -> Result<DeclaredFileSnapshot, PackageMetadataError>
where
    F: FnMut(usize),
{
    let mut file = open_declared_file_with_hook(root, path, before_segment_open)?;
    let first = read_bounded(&mut file, path)?;
    let second = read_bounded(&mut file, path)?;
    if first != second {
        return Err(PackageMetadataError::Unstable {
            path: path.to_string(),
        });
    }
    Ok(DeclaredFileSnapshot {
        path: path.clone(),
        bytes: first,
    })
}

#[cfg(feature = "test-hooks")]
#[doc(hidden)]
pub fn snapshot_declared_metadata_file_with_hook<F>(
    root: &Path,
    path: &PortablePackagePath,
    after_first_read: F,
) -> Result<DeclaredFileSnapshot, PackageMetadataError>
where
    F: FnOnce(),
{
    let mut file = open_declared_file_with_hook(root, path, |_| {})?;
    let first = read_bounded(&mut file, path)?;
    after_first_read();
    let second = read_bounded(&mut file, path)?;
    if first != second {
        return Err(PackageMetadataError::Unstable {
            path: path.to_string(),
        });
    }
    Ok(DeclaredFileSnapshot {
        path: path.clone(),
        bytes: first,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct InvalidMetadataWire {
        description: Option<PackageDescription>,
        license: Option<SpdxLicense>,
        license_file: Option<PortablePackagePath>,
        repository: Option<PackageUrl>,
        documentation: Option<PackageUrl>,
        homepage: Option<PackageUrl>,
        readme: Option<PortablePackagePath>,
    }

    #[test]
    fn bincode_revalidates_metadata_and_portable_paths() {
        let path = PortablePackagePath::from_str("docs/README.md").unwrap();
        let encoded = bincode::serialize(&path).unwrap();
        let decoded: PortablePackagePath = bincode::deserialize(&encoded).unwrap();
        assert_eq!(decoded, path);

        let invalid = InvalidMetadataWire {
            description: None,
            license: Some(SpdxLicense::from_str("MIT").unwrap()),
            license_file: Some(PortablePackagePath::from_str("LICENSE").unwrap()),
            repository: None,
            documentation: None,
            homepage: None,
            readme: None,
        };
        let encoded = bincode::serialize(&invalid).unwrap();
        let error = bincode::deserialize::<PackageMetadata>(&encoded).unwrap_err();
        assert!(error.to_string().contains("exclusive alternatives"));
    }
}
