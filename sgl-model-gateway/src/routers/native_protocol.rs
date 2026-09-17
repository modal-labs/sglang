//! Native HTTP requests are forwarded without an OpenAI conversion.
//! The engine owns protocol validation, tools, reasoning and multimodal semantics.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocols::common::GenerationRequest;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(transparent)]
pub struct NativeRequest(pub Value);

impl GenerationRequest for NativeRequest {
    fn is_stream(&self) -> bool {
        self.0
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    fn get_model(&self) -> Option<&str> {
        self.0.get("model").and_then(Value::as_str)
    }

    fn extract_text_for_routing(&self) -> String {
        // This is only a routing key, never a replacement generation payload.
        // Include system/tools/media identity so distinct prefixes do not alias.
        let mut prefix = self.0.clone();
        if let Some(object) = prefix.as_object_mut() {
            object.retain(|key, _| {
                matches!(
                    key.as_str(),
                    "system" | "tools" | "messages" | "input" | "instructions"
                )
            });
        }
        prefix.to_string()
    }
}

/// Prefix-preserving token encoding reuses the routing tree without detokenizing.
/// One fixed-width codeword per token; namespaces prevent mixing text/token keys.
pub fn generation_routing_key(body: &Value) -> Option<String> {
    use std::fmt::Write;
    if let Some(ids) = body.get("input_ids").and_then(Value::as_array) {
        let mut key = String::with_capacity(ids.len().saturating_mul(8).saturating_add(16));
        key.push_str("\0input_ids\0");
        // Token placeholders alone do not identify multimodal cache entries.
        for field in ["image_data", "video_data", "audio_data", "extra_key", "input_embeds",
                      "positional_embed_overrides"] {
            if let Some(value) = body.get(field).filter(|v| !v.is_null()) {
                let bytes = serde_json::to_vec(value).ok()?;
                write!(key, "{field}:{};", blake3::hash(&bytes).to_hex()).ok()?;
            }
        }

        for id in ids {
            let id = u32::try_from(id.as_u64()?).ok()?;
            write!(key, "{id:08x}").ok()?;
        }
        Some(key)
    } else {
        body.get("text")
            .and_then(Value::as_str)
            .map(|text| format!("\0text\0{text}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn token_keys_preserve_prefix_without_numeric_aliasing() {
        let a = generation_routing_key(&serde_json::json!({"input_ids":[1,23]})).unwrap();
        let b = generation_routing_key(&serde_json::json!({"input_ids":[12,3]})).unwrap();
        let c = generation_routing_key(&serde_json::json!({"input_ids":[1,23,4]})).unwrap();
        assert_ne!(a, b);
        assert!(c.starts_with(&a));
        assert!(generation_routing_key(&serde_json::json!({"input_ids":[[1],[2]]})).is_none());
    }
}
