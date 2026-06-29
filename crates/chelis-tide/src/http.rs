use std::net::SocketAddr;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    routing::post,
};
use serde_json::Value;
use tokio::net::TcpListener;

use crate::compiler;
use crate::schema::{
    AddFunctionRequest, ApiEnvelope, BatchRequestEnvelope, BatchResultEnvelope, CheckRequest,
    CompileRequest, DecompileRequest, DesugarRequest, Diagnostic, EvalRequest, GradRequest,
    LowerRequest, ParseRequest, ReplaceFunctionBodyRequest, ValidateRequest,
};

#[derive(Debug, Clone, Default)]
pub struct AppState;

pub fn router() -> Router {
    Router::new()
        .route("/parse", post(parse))
        .route("/desugar", post(desugar))
        .route("/check", post(check))
        .route("/lower", post(lower))
        .route("/compile", post(compile))
        .route("/eval", post(eval))
        .route("/grad", post(grad))
        .route("/validate", post(validate))
        .route("/decompile", post(decompile))
        .route("/replace_function_body", post(replace_function_body))
        .route("/add_function", post(add_function))
        .route("/batch", post(batch))
        .with_state(AppState)
}

pub async fn serve(addr: SocketAddr) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, router()).await
}

pub fn serve_blocking(addr: SocketAddr) -> std::io::Result<()> {
    tokio::runtime::Runtime::new()?.block_on(serve(addr))
}

async fn parse(
    State(_state): State<AppState>,
    Json(request): Json<ParseRequest>,
) -> Json<ApiEnvelope<crate::schema::ParseResult>> {
    Json(compiler::result_envelope(compiler::parse(request)))
}

async fn desugar(
    State(_state): State<AppState>,
    Json(request): Json<DesugarRequest>,
) -> Json<ApiEnvelope<crate::schema::DesugarResult>> {
    Json(compiler::result_envelope(compiler::desugar(request)))
}

async fn check(
    State(_state): State<AppState>,
    Json(request): Json<CheckRequest>,
) -> Json<ApiEnvelope<crate::schema::CheckResult>> {
    Json(compiler::result_envelope(compiler::check(request)))
}

async fn lower(
    State(_state): State<AppState>,
    Json(request): Json<LowerRequest>,
) -> Json<ApiEnvelope<crate::schema::LowerResult>> {
    Json(compiler::result_envelope(compiler::lower(request)))
}

async fn compile(
    State(_state): State<AppState>,
    Json(request): Json<CompileRequest>,
) -> Json<ApiEnvelope<crate::schema::CompileResult>> {
    Json(compiler::result_envelope(compiler::compile(request)))
}

async fn eval(
    State(_state): State<AppState>,
    Json(request): Json<EvalRequest>,
) -> Json<ApiEnvelope<crate::schema::EvalResult>> {
    Json(compiler::result_envelope(compiler::eval(request)))
}

async fn grad(
    State(_state): State<AppState>,
    Json(request): Json<GradRequest>,
) -> Json<ApiEnvelope<crate::schema::GradResult>> {
    Json(compiler::result_envelope(compiler::grad(request)))
}

async fn validate(
    State(_state): State<AppState>,
    Json(request): Json<ValidateRequest>,
) -> Json<ApiEnvelope<crate::schema::ValidateResult>> {
    Json(compiler::result_envelope(compiler::validate(request)))
}

async fn decompile(
    State(_state): State<AppState>,
    Json(request): Json<DecompileRequest>,
) -> Json<ApiEnvelope<crate::schema::DecompileResult>> {
    Json(compiler::result_envelope(compiler::decompile(request)))
}

async fn replace_function_body(
    State(_state): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Json<ApiEnvelope<crate::schema::ReplaceFunctionBodyResult>> {
    Json(authoring_request_envelope::<
        ReplaceFunctionBodyRequest,
        crate::schema::ReplaceFunctionBodyResult,
        _,
    >(body, compiler::replace_function_body))
}

async fn add_function(
    State(_state): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Json<ApiEnvelope<crate::schema::AddFunctionResult>> {
    Json(authoring_request_envelope::<
        AddFunctionRequest,
        crate::schema::AddFunctionResult,
        _,
    >(body, compiler::add_function))
}

fn authoring_request_envelope<T, R, F>(
    body: Result<Json<Value>, JsonRejection>,
    f: F,
) -> ApiEnvelope<R>
where
    T: serde::de::DeserializeOwned,
    F: FnOnce(T) -> Result<R, compiler::CompilerError>,
{
    let value = match body {
        Ok(Json(value)) => value,
        Err(err) => return invalid_http_request(err.to_string()),
    };
    match serde_json::from_value::<T>(value) {
        Ok(request) => compiler::result_envelope(f(request)),
        Err(err) => invalid_http_request(err.to_string()),
    }
}

fn invalid_http_request<T>(message: String) -> ApiEnvelope<T> {
    ApiEnvelope::failure(
        "http",
        vec![Diagnostic {
            kind: "invalid_request".to_string(),
            message,
            severity: 1.0,
            expected: None,
            got: None,
            suggestions: Vec::new(),
            span: None,
            deep_path: None,
        }],
    )
}

async fn batch(
    State(_state): State<AppState>,
    Json(request): Json<BatchRequestEnvelope>,
) -> Json<BatchResultEnvelope> {
    Json(compiler::batch(request.requests))
}
