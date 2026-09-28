//! Admission for the stored execution carriers in spec/10 §3.2.

use chelis_types::{ScalarValue, TensorStorage};
use schemars::{JsonSchema, r#gen::SchemaGenerator, schema::Schema};
use serde::{Deserialize, Serialize};

/// Numeric subset of the sealed scalar carrier. Execution booleans have their
/// own variant, so a second boolean spelling cannot be constructed here.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(try_from = "ScalarValue", into = "ScalarValue")]
pub struct NumericScalar(ScalarValue);

impl NumericScalar {
    pub fn get(self) -> ScalarValue {
        self.0
    }
}

impl TryFrom<ScalarValue> for NumericScalar {
    type Error = String;

    fn try_from(value: ScalarValue) -> Result<Self, Self::Error> {
        if value.prim().is_float() || value.prim().is_integer() {
            Ok(Self(value))
        } else {
            Err("numeric execution scalar requires a float or integer dtype; use the bool execution variant for booleans".to_string())
        }
    }
}

impl From<NumericScalar> for ScalarValue {
    fn from(value: NumericScalar) -> Self {
        value.0
    }
}

impl JsonSchema for NumericScalar {
    fn schema_name() -> String {
        "NumericScalar".to_string()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        let mut schema = ScalarValue::json_schema(generator).into_object();
        let variants = schema
            .subschemas
            .as_mut()
            .expect("scalar enum schema")
            .one_of
            .as_mut()
            .expect("tagged scalar alternatives");
        variants.retain(|variant| {
            let Schema::Object(variant) = variant else {
                unreachable!("scalar variant schema")
            };
            let dtype = &variant.object.as_ref().expect("scalar object").properties["dtype"];
            let Schema::Object(dtype) = dtype else {
                unreachable!("scalar dtype schema")
            };
            dtype
                .enum_values
                .as_ref()
                .expect("scalar dtype discriminant")
                != &[serde_json::json!("bool")]
        });
        schema.into()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorWire {
    shape: Vec<i64>,
    /// The execution storage grammar: a key tensor's storage object is
    /// admitted here and nowhere in a graph.
    #[serde(with = "chelis_types::dtype_semantics::execution_storage")]
    data: TensorStorage,
}

impl super::TensorValue {
    /// Check the exact shape/count relationship before a consumer clones or
    /// accesses the payload. Zero extents make the product zero independent of
    /// their position; intermediate machine overflow cannot reject an empty tensor.
    pub fn validate(&self) -> Result<(), String> {
        i32::try_from(self.shape.len()).map_err(|_| "tensor rank exceeds int32")?;
        if self.shape.iter().any(|extent| *extent < 0) {
            return Err("tensor extent must be a nonnegative int64".to_string());
        }
        let count = if self.shape.contains(&0) {
            0
        } else {
            self.shape.iter().try_fold(1_usize, |count, extent| {
                let extent =
                    usize::try_from(*extent).map_err(|_| "tensor extent exceeds host capacity")?;
                count
                    .checked_mul(extent)
                    .ok_or("tensor element count exceeds host capacity")
            })?
        };
        if count != self.data.len() {
            return Err(format!(
                "tensor shape requires {count} elements, payload contains {}",
                self.data.len()
            ));
        }
        let width = self
            .data
            .prim()
            .runtime_dtype()
            .map_err(|error| error.to_string())?
            .byte_width();
        let bytes = count
            .checked_mul(width)
            .ok_or("tensor byte count exceeds host capacity")?;
        isize::try_from(bytes).map_err(|_| "tensor byte count exceeds host allocation capacity")?;
        Ok(())
    }

    pub(crate) fn host_shape(&self) -> Result<Vec<usize>, String> {
        self.validate()?;
        self.shape
            .iter()
            .map(|extent| {
                usize::try_from(*extent)
                    .map_err(|_| "tensor extent exceeds host capacity".to_string())
            })
            .collect()
    }
}

impl Serialize for super::TensorValue {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.validate().map_err(serde::ser::Error::custom)?;
        TensorWire {
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for super::TensorValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = TensorWire::deserialize(deserializer)?;
        let value = Self {
            shape: wire.shape,
            data: wire.data,
        };
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

pub(super) fn shape_schema(generator: &mut SchemaGenerator) -> Schema {
    let mut schema = Vec::<i64>::json_schema(generator).into_object();
    let array = schema.array.as_mut().expect("shape array schema");
    array.max_items = Some(i32::MAX as u32);
    let mut extent = i64::json_schema(generator).into_object();
    extent.number().minimum = Some(0.0);
    array.items = Some(Schema::Object(extent).into());
    schema.into()
}
