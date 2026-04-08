const fs = require("fs");
const path = require("path");
const vscode = require("vscode");
const { LanguageClient, TransportKind } = require("vscode-languageclient/node");

const deepDocuments = new Map();
let client;
let statusBarItem;
let outputChannel;
let statusTimer;
const SERVER_SHOW_DEEP = "chelis.server.showDeep";
const SERVER_STATUS = "chelis.server.status";

class DeepDocumentProvider {
  constructor() {
    this.onDidChangeEmitter = new vscode.EventEmitter();
    this.onDidChange = this.onDidChangeEmitter.event;
  }

  provideTextDocumentContent(uri) {
    return deepDocuments.get(uri.toString()) || "";
  }
}

function activate(context) {
  outputChannel = vscode.window.createOutputChannel("Chelis");
  context.subscriptions.push(outputChannel);

  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 10);
  statusBarItem.name = "Chelis Fitness";
  statusBarItem.text = "Chelis: starting";
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  const provider = new DeepDocumentProvider();
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider("chelis-deep", provider),
  );

  const command = vscode.commands.registerCommand("chelis.showDeep", async () => {
    if (!client) {
      vscode.window.showErrorMessage("Chelis LSP is not running.");
      return;
    }
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      return;
    }

    try {
      const response = await client.sendRequest("workspace/executeCommand", {
        command: SERVER_SHOW_DEEP,
        arguments: [
          {
            uri: editor.document.uri.toString(),
            selection: editor.selection.isEmpty ? null : toLspRange(editor.selection),
          },
        ],
      });

      if (!response || typeof response.text !== "string" || typeof response.uri !== "string") {
        vscode.window.showWarningMessage("Chelis LSP did not return a Deep view.");
        return;
      }

      const uri = vscode.Uri.parse(response.uri);
      deepDocuments.set(uri.toString(), response.text);
      provider.onDidChangeEmitter?.fire(uri);
      const document = await vscode.workspace.openTextDocument(uri);
      await vscode.languages.setTextDocumentLanguage(document, response.language_id || "chelis-deep");
      await vscode.window.showTextDocument(document, {
        preview: false,
        viewColumn: vscode.ViewColumn.Beside,
      });
    } catch (error) {
      vscode.window.showErrorMessage(`Chelis Deep view failed: ${error}`);
    }
  });
  context.subscriptions.push(command);

  const launch = resolveServerLaunch();
  outputChannel.appendLine(
    `Starting Chelis LSP: ${launch.command} ${launch.args.join(" ")}${launch.options?.cwd ? ` (cwd=${launch.options.cwd})` : ""}`,
  );
  const serverOptions = {
    ...launch,
    transport: TransportKind.stdio,
  };

  const clientOptions = {
    outputChannel,
    traceOutputChannel: outputChannel,
    documentSelector: [
      { scheme: "file", language: "chelis" },
      { scheme: "file", language: "chelis-deep" },
    ],
  };

  client = new LanguageClient("chelis", "Chelis Tide", serverOptions, clientOptions);
  context.subscriptions.push({
    dispose: () => {
      if (client) {
        return client.stop();
      }
      return undefined;
    },
  });

  client.start().then(
    () => {
      outputChannel.appendLine("Chelis LSP client started.");
      if (typeof client.onDidChangeState === "function") {
        context.subscriptions.push(
          client.onDidChangeState((event) => {
            outputChannel.appendLine(
              `Chelis LSP state: ${String(event.oldState)} -> ${String(event.newState)}`,
            );
            queueRefreshStatus();
          }),
        );
      }
      context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor(() => {
          queueRefreshStatus();
        }),
      );
      context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument((event) => {
          const active = vscode.window.activeTextEditor;
          if (active && event.document.uri.toString() === active.document.uri.toString()) {
            queueRefreshStatus();
          }
        }),
      );
      context.subscriptions.push(
        vscode.languages.onDidChangeDiagnostics(() => {
          queueRefreshStatus();
        }),
      );
      queueRefreshStatus();
    },
    (error) => {
      outputChannel.appendLine(`Chelis LSP startup failed: ${error}`);
      statusBarItem.text = "Chelis: unavailable";
      statusBarItem.tooltip = `Chelis LSP startup failed: ${error}`;
      vscode.window.showErrorMessage(`Chelis LSP startup failed: ${error}`);
    },
  );
}

async function refreshStatus() {
  if (!client || typeof client.isRunning === "function" && !client.isRunning()) {
    statusBarItem.text = "Chelis: unavailable";
    statusBarItem.tooltip = "Chelis LSP is not running.";
    return;
  }
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document.languageId !== "chelis") {
    statusBarItem.text = "Chelis";
    statusBarItem.tooltip = "Open a Chelis file to see fitness.";
    statusBarItem.command = undefined;
    return;
  }

  try {
    const status = await client.sendRequest("workspace/executeCommand", {
      command: SERVER_STATUS,
      arguments: [{ uri: editor.document.uri.toString() }],
    });
    if (!status || typeof status.score !== "number") {
      statusBarItem.text = "Chelis: unavailable";
      statusBarItem.tooltip = "Fitness score unavailable for the current document.";
      return;
    }
    statusBarItem.text = `Chelis ${status.score.toFixed(2)}`;
    statusBarItem.tooltip = status.stale
      ? "Chelis fitness score is stale."
      : "Chelis fitness score from the latest successful check.";
    statusBarItem.command = "chelis.showDeep";
  } catch (error) {
    statusBarItem.text = "Chelis: unavailable";
    statusBarItem.tooltip = `Chelis status request failed: ${error}`;
    statusBarItem.command = undefined;
  }
}

function deactivate() {
  if (statusTimer) {
    clearTimeout(statusTimer);
    statusTimer = undefined;
  }
  if (!client) {
    return undefined;
  }
  return client.stop();
}

function queueRefreshStatus() {
  if (statusTimer) {
    clearTimeout(statusTimer);
  }
  statusTimer = setTimeout(() => {
    statusTimer = undefined;
    refreshStatus();
  }, 150);
}

function resolveServerLaunch() {
  const workspaceFolders = vscode.workspace.workspaceFolders || [];
  for (const folder of workspaceFolders) {
    const root = folder.uri.fsPath;
    const localBinary = path.join(root, "target", "debug", process.platform === "win32" ? "chelis.exe" : "chelis");
    if (fs.existsSync(localBinary)) {
      return {
        command: localBinary,
        args: ["tide", "lsp"],
        options: { cwd: root },
      };
    }

    const cargoToml = path.join(root, "Cargo.toml");
    if (fs.existsSync(cargoToml)) {
      return {
        command: "cargo",
        args: ["run", "--quiet", "-p", "chelis-cli", "--", "tide", "lsp"],
        options: { cwd: root },
      };
    }
  }

  return {
    command: "chelis",
    args: ["tide", "lsp"],
  };
}

function toLspRange(selection) {
  return {
    start: {
      line: selection.start.line,
      character: selection.start.character,
    },
    end: {
      line: selection.end.line,
      character: selection.end.character,
    },
  };
}

module.exports = {
  activate,
  deactivate,
};
