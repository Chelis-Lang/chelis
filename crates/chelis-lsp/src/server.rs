use chelis_unord::UnordMap;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::{
    CompletionOptions, CompletionResponse, DidChangeTextDocumentParams, DidCloseTextDocumentParams,
    DidOpenTextDocumentParams, ExecuteCommandOptions, ExecuteCommandParams, Hover,
    HoverProviderCapability, InitializeParams, InitializeResult, InitializedParams, Location,
    MessageType, OneOf, ServerCapabilities, TextDocumentContentChangeEvent,
    TextDocumentSyncCapability, TextDocumentSyncKind, TextDocumentSyncOptions, Url,
};
use tower_lsp::{Client, LanguageServer, LspService, Server};

use crate::analysis::{
    DocumentState, analyze_document, completion_items, deep_view, definition_location,
    document_status, hover_markdown,
};

#[derive(Debug, Default)]
struct WorkspaceState {
    workspace_root: Option<PathBuf>,
    documents: UnordMap<Url, DocumentState>,
}

pub struct Backend {
    client: Client,
    state: Arc<RwLock<WorkspaceState>>,
}

const SHOW_DEEP_COMMAND: &str = "chelis.server.showDeep";
const STATUS_COMMAND: &str = "chelis.server.status";

#[derive(Debug, Clone, Deserialize)]
struct DeepViewArgs {
    uri: Url,
    #[serde(default)]
    selection: Option<tower_lsp::lsp_types::Range>,
}

#[derive(Debug, Clone, Deserialize)]
struct StatusArgs {
    uri: Url,
}

#[derive(Debug, Clone, Serialize)]
struct DeepViewResponse {
    uri: String,
    language_id: String,
    text: String,
}

impl Backend {
    fn new(client: Client) -> Self {
        Self {
            client,
            state: Arc::new(RwLock::new(WorkspaceState::default())),
        }
    }

    async fn update_document(&self, uri: Url, text: String) {
        let analysis = analyze_document(&uri, &text);
        let diagnostics = analysis.diagnostics.clone();
        {
            let mut state = self.state.write().await;
            state.documents.insert(
                uri.clone(),
                DocumentState {
                    uri: uri.clone(),
                    text,
                    analysis,
                },
            );
        }
        self.client
            .publish_diagnostics(uri, diagnostics, None)
            .await;
    }

    fn workspace_root(params: &InitializeParams) -> Option<PathBuf> {
        params
            .workspace_folders
            .as_ref()
            .and_then(|folders| folders.first())
            .and_then(|folder| folder.uri.to_file_path().ok())
            .or_else(|| {
                params
                    .root_uri
                    .as_ref()
                    .and_then(|uri| uri.to_file_path().ok())
            })
    }

    fn first_change_text(changes: &[TextDocumentContentChangeEvent]) -> Option<String> {
        changes.last().map(|change| change.text.clone())
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, params: InitializeParams) -> Result<InitializeResult> {
        {
            let mut state = self.state.write().await;
            state.workspace_root = Self::workspace_root(&params);
        }

        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Options(
                    TextDocumentSyncOptions {
                        open_close: Some(true),
                        change: Some(TextDocumentSyncKind::FULL),
                        ..TextDocumentSyncOptions::default()
                    },
                )),
                completion_provider: Some(CompletionOptions::default()),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                execute_command_provider: Some(ExecuteCommandOptions {
                    commands: vec![SHOW_DEEP_COMMAND.to_string(), STATUS_COMMAND.to_string()],
                    ..ExecuteCommandOptions::default()
                }),
                ..ServerCapabilities::default()
            },
            ..InitializeResult::default()
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "Chelis LSP initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.update_document(params.text_document.uri, params.text_document.text)
            .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(text) = Self::first_change_text(&params.content_changes) {
            self.update_document(params.text_document.uri, text).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        {
            let mut state = self.state.write().await;
            state.documents.remove(&params.text_document.uri);
        }
        self.client
            .publish_diagnostics(params.text_document.uri, Vec::new(), None)
            .await;
    }

    async fn completion(
        &self,
        params: tower_lsp::lsp_types::CompletionParams,
    ) -> Result<Option<CompletionResponse>> {
        let state = self.state.read().await;
        let Some(document) = state
            .documents
            .get(&params.text_document_position.text_document.uri)
        else {
            return Ok(None);
        };
        let items = completion_items(document, params.text_document_position.position);
        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn hover(&self, params: tower_lsp::lsp_types::HoverParams) -> Result<Option<Hover>> {
        let state = self.state.read().await;
        let Some(document) = state
            .documents
            .get(&params.text_document_position_params.text_document.uri)
        else {
            return Ok(None);
        };
        Ok(hover_markdown(
            document,
            params.text_document_position_params.position,
        ))
    }

    async fn goto_definition(
        &self,
        params: tower_lsp::lsp_types::GotoDefinitionParams,
    ) -> Result<Option<tower_lsp::lsp_types::GotoDefinitionResponse>> {
        let state = self.state.read().await;
        let Some(document) = state
            .documents
            .get(&params.text_document_position_params.text_document.uri)
        else {
            return Ok(None);
        };
        let location = definition_location(
            document,
            params.text_document_position_params.position,
            state.workspace_root.as_deref(),
        );
        Ok(location.map(|location: Location| {
            tower_lsp::lsp_types::GotoDefinitionResponse::Scalar(location)
        }))
    }

    async fn execute_command(&self, params: ExecuteCommandParams) -> Result<Option<Value>> {
        match params.command.as_str() {
            SHOW_DEEP_COMMAND => {
                let Some(arg) = params.arguments.first() else {
                    return Ok(None);
                };
                let args: DeepViewArgs =
                    serde_json::from_value(arg.clone()).map_err(invalid_params)?;
                let state = self.state.read().await;
                let Some(document) = state.documents.get(&args.uri) else {
                    return Ok(None);
                };
                let text = deep_view(document, args.selection).map_err(invalid_params)?;
                Ok(Some(
                    serde_json::to_value(DeepViewResponse {
                        uri: format!("chelis-deep:{}", args.uri.path()),
                        language_id: "chelis-deep".to_string(),
                        text,
                    })
                    .map_err(invalid_params)?,
                ))
            }
            STATUS_COMMAND => {
                let Some(arg) = params.arguments.first() else {
                    return Ok(None);
                };
                let args: StatusArgs =
                    serde_json::from_value(arg.clone()).map_err(invalid_params)?;
                let state = self.state.read().await;
                let Some(document) = state.documents.get(&args.uri) else {
                    return Ok(None);
                };
                Ok(Some(document_status(document)))
            }
            other => {
                self.client
                    .log_message(MessageType::WARNING, format!("Unknown command `{other}`"))
                    .await;
                Ok(None)
            }
        }
    }
}

fn invalid_params(err: impl ToString) -> tower_lsp::jsonrpc::Error {
    tower_lsp::jsonrpc::Error::invalid_params(err.to_string())
}

pub async fn serve_stdio() -> io::Result<()> {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let (service, socket) = LspService::new(Backend::new);
    Server::new(stdin, stdout, socket).serve(service).await;
    Ok(())
}

pub fn serve_stdio_blocking() -> io::Result<()> {
    tokio::runtime::Runtime::new()?.block_on(serve_stdio())
}
