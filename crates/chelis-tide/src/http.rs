use std::net::SocketAddr;

use axum::{Json, Router, extract::State, routing::post};
use tokio::net::TcpListener;

use crate::compiler;
use crate::schema::{
    ApiEnvelope, BatchRequestEnvelope, BatchResultEnvelope, CheckRequest, CompileRequest,
    DecompileRequest, DesugarRequest, EvalRequest, GradRequest, LowerRequest, ParseRequest,
    ValidateRequest,
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

async fn batch(
    State(_state): State<AppState>,
    Json(request): Json<BatchRequestEnvelope>,
) -> Json<BatchResultEnvelope> {
    Json(compiler::batch(request.requests))
}
