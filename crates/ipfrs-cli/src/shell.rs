//! Interactive shell (REPL) for IPFRS
//!
//! Provides a read-eval-print loop for interactive IPFRS operations

use anyhow::{Context, Result};
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Editor, Helper};
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::{debug, info};

use crate::output::{error, print_header, success};

/// Calculate Levenshtein distance between two strings
#[allow(clippy::needless_range_loop)]
fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();
    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    for (i, c1) in s1.chars().enumerate() {
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            matrix[i + 1][j + 1] = std::cmp::min(
                std::cmp::min(matrix[i][j + 1] + 1, matrix[i + 1][j] + 1),
                matrix[i][j] + cost,
            );
        }
    }

    matrix[len1][len2]
}

/// Command completer for tab completion
#[derive(Debug, Clone)]
struct CommandCompleter {
    commands: Vec<String>,
}

impl CommandCompleter {
    fn new() -> Self {
        Self {
            commands: vec![
                // General commands
                "help".to_string(),
                "?".to_string(),
                "h".to_string(),
                "exit".to_string(),
                "quit".to_string(),
                "q".to_string(),
                "bye".to_string(),
                "clear".to_string(),
                "cls".to_string(),
                "clean".to_string(),
                "version".to_string(),
                "pwd".to_string(),
                "info".to_string(),
                // File operations
                "add".to_string(),
                "get".to_string(),
                "cat".to_string(),
                "ls".to_string(),
                // Network commands
                "id".to_string(),
                "peers".to_string(),
                "peer".to_string(),
                "connect".to_string(),
                "disconnect".to_string(),
                // Statistics
                "stats".to_string(),
                "stat".to_string(),
                // Advanced commands
                "semantic".to_string(),
                "search".to_string(),
                "find".to_string(),
                "logic".to_string(),
                "infer".to_string(),
                "query".to_string(),
                "tensor".to_string(),
                "model".to_string(),
                "gradient".to_string(),
                // Pin management
                "pin".to_string(),
                "unpin".to_string(),
                // Alias management
                "alias".to_string(),
                "unalias".to_string(),
                // Common aliases
                "ll".to_string(),
                "list".to_string(),
                "show".to_string(),
                "view".to_string(),
                "download".to_string(),
                "upload".to_string(),
                "put".to_string(),
                "connections".to_string(),
                "nodes".to_string(),
                "whoami".to_string(),
                "status".to_string(),
                "statistics".to_string(),
                "logout".to_string(),
                "leave".to_string(),
            ],
        }
    }
}

impl Completer for CommandCompleter {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let start = line[..pos]
            .rfind(char::is_whitespace)
            .map(|i| i + 1)
            .unwrap_or(0);

        let prefix = &line[start..pos];

        let matches: Vec<Pair> = self
            .commands
            .iter()
            .filter(|cmd| cmd.starts_with(prefix))
            .map(|cmd| Pair {
                display: cmd.clone(),
                replacement: cmd.clone(),
            })
            .collect();

        Ok((start, matches))
    }
}

impl Hinter for CommandCompleter {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>) -> Option<String> {
        if pos < line.len() {
            return None;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return None;
        }

        // Provide hints based on incomplete commands
        match parts[0] {
            "add" if parts.len() == 1 => Some(" <path>".to_string()),
            "get" if parts.len() == 1 => Some(" <cid> [output]".to_string()),
            "cat" if parts.len() == 1 => Some(" <cid>".to_string()),
            "ls" if parts.len() == 1 => Some(" <cid>".to_string()),
            "stats" if parts.len() == 1 => Some(" [storage|semantic|logic]".to_string()),
            "semantic" if parts.len() == 1 => Some(" <search|stats> [args...]".to_string()),
            "logic" if parts.len() == 1 => Some(" <infer|prove|kb-stats> [args...]".to_string()),
            "search" | "find" if parts.len() == 1 => Some(" <query>".to_string()),
            "infer" | "query" if parts.len() == 1 => Some(" <goal>".to_string()),
            _ => {
                // Provide command completion hints
                let prefix = parts[0];
                self.commands
                    .iter()
                    .find(|cmd| cmd.starts_with(prefix) && cmd.len() > prefix.len())
                    .map(|cmd| cmd[prefix.len()..].to_string())
            }
        }
    }
}

impl Highlighter for CommandCompleter {}

impl Validator for CommandCompleter {
    fn validate(
        &self,
        ctx: &mut rustyline::validate::ValidationContext,
    ) -> rustyline::Result<rustyline::validate::ValidationResult> {
        let input = ctx.input();

        // Check for line continuation (backslash at end)
        if input.ends_with('\\') {
            return Ok(rustyline::validate::ValidationResult::Incomplete);
        }

        // Check for unclosed quotes
        let quote_count = input.chars().filter(|&c| c == '"').count();
        if quote_count % 2 != 0 {
            return Ok(rustyline::validate::ValidationResult::Incomplete);
        }

        // Check for unclosed parentheses (useful for logic queries)
        let open_parens = input.chars().filter(|&c| c == '(').count();
        let close_parens = input.chars().filter(|&c| c == ')').count();
        if open_parens > close_parens {
            return Ok(rustyline::validate::ValidationResult::Incomplete);
        }

        Ok(rustyline::validate::ValidationResult::Valid(None))
    }
}

impl Helper for CommandCompleter {}

/// Interactive shell configuration
#[derive(Debug, Clone)]
pub struct ShellConfig {
    /// Data directory
    pub data_dir: PathBuf,
    /// History file path
    pub history_file: PathBuf,
    /// Prompt string
    pub prompt: String,
    /// User-defined command aliases (alias -> command)
    pub aliases: HashMap<String, String>,
}

impl Default for ShellConfig {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));

        // Initialize built-in aliases
        let mut aliases = HashMap::new();

        // Common shortcuts
        aliases.insert("ll".to_string(), "ls".to_string());
        aliases.insert("list".to_string(), "ls".to_string());
        aliases.insert("show".to_string(), "cat".to_string());
        aliases.insert("view".to_string(), "cat".to_string());
        aliases.insert("download".to_string(), "get".to_string());
        aliases.insert("upload".to_string(), "add".to_string());
        aliases.insert("put".to_string(), "add".to_string());

        // Network shortcuts
        aliases.insert("connections".to_string(), "peers".to_string());
        aliases.insert("nodes".to_string(), "peers".to_string());
        aliases.insert("whoami".to_string(), "id".to_string());

        // Statistics shortcuts
        aliases.insert("status".to_string(), "stats".to_string());
        aliases.insert("statistics".to_string(), "stats".to_string());

        // Exit shortcuts (already handled in execute_command, but here for completeness)
        aliases.insert("logout".to_string(), "exit".to_string());
        aliases.insert("leave".to_string(), "exit".to_string());

        Self {
            data_dir: PathBuf::from(".ipfrs"),
            history_file: home.join(".ipfrs_history"),
            prompt: "ipfrs> ".to_string(),
            aliases,
        }
    }
}

/// Interactive shell session
pub struct Shell {
    config: ShellConfig,
    editor: Editor<CommandCompleter, rustyline::history::DefaultHistory>,
}

impl Shell {
    /// Create a new interactive shell
    pub fn new(config: ShellConfig) -> Result<Self> {
        let mut editor = Editor::new().context("Failed to create line editor")?;

        // Set up tab completion
        editor.set_helper(Some(CommandCompleter::new()));

        // Load history if it exists
        if config.history_file.exists() {
            if let Err(e) = editor.load_history(&config.history_file) {
                debug!("Failed to load history: {}", e);
            }
        }

        Ok(Self { config, editor })
    }

    /// Run the interactive shell
    pub async fn run(&mut self) -> Result<()> {
        print_header("IPFRS Interactive Shell");
        println!("Type 'help' for available commands, 'exit' or Ctrl+D to quit");
        println!("Use Tab for completion, Up/Down for history, Ctrl+R for search");
        println!("Multi-line input: end line with \\ or leave quotes/parentheses unclosed\n");

        loop {
            match self.editor.readline(&self.config.prompt) {
                Ok(line) => {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }

                    // Add to history
                    let _ = self.editor.add_history_entry(line);

                    // Parse and execute command
                    match self.execute_command(line).await {
                        Ok(should_continue) => {
                            if !should_continue {
                                break;
                            }
                        }
                        Err(e) => {
                            error(&format!("Error: {}", e));
                        }
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    println!("Goodbye!");
                    break;
                }
                Err(err) => {
                    error(&format!("Error reading line: {}", err));
                    break;
                }
            }
        }

        // Save history
        if let Err(e) = self.editor.save_history(&self.config.history_file) {
            debug!("Failed to save history: {}", e);
        }

        Ok(())
    }

    /// Execute a shell command
    /// Returns Ok(true) to continue, Ok(false) to exit
    async fn execute_command(&mut self, line: &str) -> Result<bool> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(true);
        }

        // Resolve alias
        let command = if let Some(resolved) = self.config.aliases.get(parts[0]) {
            resolved.as_str()
        } else {
            parts[0]
        };

        match command {
            "help" | "?" | "h" => {
                self.show_help();
                Ok(true)
            }
            "exit" | "quit" | "q" | "bye" => {
                println!("Goodbye!");
                Ok(false)
            }
            "clear" | "cls" | "clean" => {
                print!("\x1B[2J\x1B[1;1H");
                Ok(true)
            }
            "version" => {
                println!("IPFRS version {}", env!("CARGO_PKG_VERSION"));
                Ok(true)
            }
            "pwd" => {
                println!("{}", self.config.data_dir.display());
                Ok(true)
            }
            "info" => {
                self.show_info().await;
                Ok(true)
            }
            "stats" | "stat" => {
                self.show_stats(parts.get(1).copied()).await;
                Ok(true)
            }
            "ls" => {
                if parts.len() < 2 {
                    error("Usage: ls <cid>");
                } else {
                    self.list_directory(parts[1]).await;
                }
                Ok(true)
            }
            "cat" => {
                if parts.len() < 2 {
                    error("Usage: cat <cid>");
                } else {
                    self.cat_file(parts[1]).await;
                }
                Ok(true)
            }
            "add" => {
                if parts.len() < 2 {
                    error("Usage: add <path>");
                } else {
                    self.add_file(parts[1]).await;
                }
                Ok(true)
            }
            "get" => {
                if parts.len() < 2 {
                    error("Usage: get <cid> [output_path]");
                } else {
                    let output = parts.get(2).copied();
                    self.get_file(parts[1], output).await;
                }
                Ok(true)
            }
            "peers" | "peer" => {
                self.list_peers().await;
                Ok(true)
            }
            "id" => {
                self.show_id().await;
                Ok(true)
            }
            "semantic" | "search" | "find" => {
                if parts.len() < 2 {
                    error("Usage: semantic <search|stats> [args...] (or use: search/find <query>)");
                } else {
                    self.semantic_command(&parts[1..]).await;
                }
                Ok(true)
            }
            "logic" | "infer" | "query" => {
                if parts.len() < 2 {
                    error("Usage: logic <infer|prove|kb-stats> [args...] (or use: infer/query <goal>)");
                } else {
                    self.logic_command(&parts[1..]).await;
                }
                Ok(true)
            }
            "alias" => {
                if parts.len() < 2 {
                    // List all aliases
                    self.list_aliases();
                } else if parts.len() == 2 {
                    // Show specific alias
                    if let Some(resolved) = self.config.aliases.get(parts[1]) {
                        println!("'{}' is aliased to '{}'", parts[1], resolved);
                    } else {
                        error(&format!("No alias found for '{}'", parts[1]));
                    }
                } else if parts.len() >= 3 {
                    // Add new alias: alias <name> <command>
                    let alias_name = parts[1].to_string();
                    let alias_command = parts[2..].join(" ");
                    self.config
                        .aliases
                        .insert(alias_name.clone(), alias_command.clone());
                    success(&format!(
                        "Alias '{}' -> '{}' added",
                        alias_name, alias_command
                    ));
                }
                Ok(true)
            }
            "unalias" => {
                if parts.len() < 2 {
                    error("Usage: unalias <alias_name>");
                } else if self.config.aliases.remove(parts[1]).is_some() {
                    success(&format!("Alias '{}' removed", parts[1]));
                } else {
                    error(&format!("No alias found for '{}'", parts[1]));
                }
                Ok(true)
            }
            _ => {
                // Check if this might be a typo of a known command
                let suggestion = self.suggest_command(parts[0]);
                if let Some(suggested) = suggestion {
                    error(&format!(
                        "Unknown command: '{}'. Did you mean '{}'? Type 'help' for available commands.",
                        parts[0], suggested
                    ));
                } else {
                    error(&format!(
                        "Unknown command: '{}'. Type 'help' for available commands.",
                        parts[0]
                    ));
                }
                Ok(true)
            }
        }
    }

    /// Suggest a command based on similarity to known commands
    fn suggest_command(&self, input: &str) -> Option<String> {
        let commands = vec![
            "help", "exit", "quit", "clear", "version", "pwd", "info", "add", "get", "cat", "ls",
            "peers", "id", "stats", "semantic", "search", "logic", "infer", "alias", "unalias",
        ];

        // Simple suggestion based on prefix matching or common typos
        for cmd in &commands {
            if cmd.starts_with(input) && cmd.len() > input.len() {
                return Some(cmd.to_string());
            }
        }

        // Check for common typos (single character difference)
        for cmd in &commands {
            if levenshtein_distance(input, cmd) == 1 {
                return Some(cmd.to_string());
            }
        }

        None
    }

    /// List all defined aliases
    fn list_aliases(&self) {
        if self.config.aliases.is_empty() {
            println!("No aliases defined.");
            return;
        }

        println!("\n{}", "=".repeat(60));
        println!("Defined Aliases");
        println!("{}\n", "=".repeat(60));

        let mut aliases: Vec<_> = self.config.aliases.iter().collect();
        aliases.sort_by(|a, b| a.0.cmp(b.0));

        for (alias, command) in aliases {
            println!("  {} -> {}", alias, command);
        }
        println!();
    }

    /// Show help message
    fn show_help(&self) {
        println!("\n{}", "=".repeat(60));
        println!("IPFRS Interactive Shell - Available Commands");
        println!("{}\n", "=".repeat(60));

        println!("General:");
        println!("  help, ?, h              Show this help message");
        println!("  exit, quit, q, bye      Exit the shell");
        println!("  clear, cls, clean       Clear the screen");
        println!("  version                 Show version information");
        println!("  info                    Show node information");
        println!("  pwd                     Show current data directory");

        println!("\nFile Operations:");
        println!("  add <path>              Add a file to IPFRS");
        println!("  get <cid> [output]      Get a file from IPFRS");
        println!("  cat <cid>               Display file contents");
        println!("  ls <cid>                List directory contents");

        println!("\nNetwork:");
        println!("  id                      Show peer ID and addresses");
        println!("  peers, peer             List connected peers");

        println!("\nStatistics:");
        println!("  stats, stat             Show all statistics");
        println!("  stats storage           Show storage statistics");
        println!("  stats semantic          Show semantic search statistics");
        println!("  stats logic             Show logic programming statistics");

        println!("\nSemantic Search:");
        println!("  semantic search <query> Search similar content");
        println!("  search <query>          Alias for semantic search");
        println!("  find <query>            Alias for semantic search");
        println!("  semantic stats          Show semantic statistics");

        println!("\nLogic Programming:");
        println!("  logic infer <goal>      Run inference query");
        println!("  infer <goal>            Alias for logic infer");
        println!("  query <goal>            Alias for logic infer");
        println!("  logic prove <goal>      Generate proof");
        println!("  logic kb-stats          Show knowledge base statistics");

        println!("\nAlias Management:");
        println!("  alias                   List all aliases");
        println!("  alias <name>            Show specific alias");
        println!("  alias <name> <cmd>      Create new alias");
        println!("  unalias <name>          Remove an alias");

        println!("\nCommon Aliases:");
        println!("  ll, list → ls           download → get");
        println!("  show, view → cat        upload, put → add");
        println!("  whoami → id             status → stats");
        println!("  connections, nodes → peers");

        println!("\nTips:");
        println!("  • Use Tab for command completion");
        println!("  • Use Up/Down arrows for command history");
        println!("  • Create custom aliases with 'alias' command");
        println!("  • Typos will suggest similar commands");

        println!("\n{}", "=".repeat(60));
    }

    /// Show node information
    #[allow(dead_code)]
    async fn show_info(&self) {
        println!("\nIPFRS Node Information:");
        println!("  Data directory: {}", self.config.data_dir.display());
        println!("  Status: Running");
        success("Node is operational");
    }

    /// Show statistics
    #[allow(dead_code)]
    async fn show_stats(&self, category: Option<&str>) {
        match category {
            None => {
                println!("\nNode Statistics:");
                println!("  Storage: Available");
                println!("  Semantic: Available");
                println!("  Logic: Available");
                info!("Use 'stats <category>' for detailed statistics");
            }
            Some("storage") => {
                println!("\nStorage Statistics:");
                println!("  Blocks: N/A (connect to daemon)");
            }
            Some("semantic") => {
                println!("\nSemantic Statistics:");
                println!("  Indexed vectors: N/A (connect to daemon)");
            }
            Some("logic") => {
                println!("\nLogic Statistics:");
                println!("  Facts: N/A (connect to daemon)");
                println!("  Rules: N/A (connect to daemon)");
            }
            Some(cat) => {
                error(&format!(
                    "Unknown category: '{}'. Use storage, semantic, or logic.",
                    cat
                ));
            }
        }
    }

    /// List directory contents
    #[allow(dead_code)]
    async fn list_directory(&self, _cid: &str) {
        info!("Directory listing not yet implemented in shell");
        println!("Use 'ipfrs ls <cid>' from the command line");
    }

    /// Display file contents
    #[allow(dead_code)]
    async fn cat_file(&self, _cid: &str) {
        info!("Cat command not yet implemented in shell");
        println!("Use 'ipfrs cat <cid>' from the command line");
    }

    /// Add a file
    #[allow(dead_code)]
    async fn add_file(&self, _path: &str) {
        info!("Add command not yet implemented in shell");
        println!("Use 'ipfrs add <path>' from the command line");
    }

    /// Get a file
    #[allow(dead_code)]
    async fn get_file(&self, _cid: &str, _output: Option<&str>) {
        info!("Get command not yet implemented in shell");
        println!("Use 'ipfrs get <cid>' from the command line");
    }

    /// List peers
    #[allow(dead_code)]
    async fn list_peers(&self) {
        info!("Peers command not yet implemented in shell");
        println!("Use 'ipfrs swarm peers' from the command line");
    }

    /// Show peer ID
    #[allow(dead_code)]
    async fn show_id(&self) {
        info!("ID command not yet implemented in shell");
        println!("Use 'ipfrs id' from the command line");
    }

    /// Handle semantic commands
    #[allow(dead_code)]
    async fn semantic_command(&self, _args: &[&str]) {
        info!("Semantic commands not yet implemented in shell");
        println!("Use 'ipfrs semantic <command>' from the command line");
    }

    /// Handle logic commands
    #[allow(dead_code)]
    async fn logic_command(&self, _args: &[&str]) {
        info!("Logic commands not yet implemented in shell");
        println!("Use 'ipfrs logic <command>' from the command line");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_config_default() {
        let config = ShellConfig::default();
        assert_eq!(config.prompt, "ipfrs> ");
        assert_eq!(config.data_dir, PathBuf::from(".ipfrs"));
    }

    #[test]
    fn test_shell_creation() {
        let config = ShellConfig::default();
        let result = Shell::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_command_completer() {
        let completer = CommandCompleter::new();
        assert!(!completer.commands.is_empty());
        assert!(completer.commands.contains(&"help".to_string()));
        assert!(completer.commands.contains(&"exit".to_string()));
        assert!(completer.commands.contains(&"add".to_string()));
    }

    #[test]
    fn test_command_completer_aliases() {
        let completer = CommandCompleter::new();
        // Test that aliases are included
        assert!(completer.commands.contains(&"?".to_string()));
        assert!(completer.commands.contains(&"q".to_string()));
        assert!(completer.commands.contains(&"h".to_string()));
        assert!(completer.commands.contains(&"search".to_string()));
        assert!(completer.commands.contains(&"find".to_string()));
    }

    #[test]
    fn test_command_hint_logic() {
        let completer = CommandCompleter::new();

        // Test hint logic for various commands
        // Note: We can't easily test the hint() method due to rustyline's Context API,
        // but we can verify the command list is properly set up
        let commands = &completer.commands;

        // Verify command coverage
        assert!(commands.contains(&"add".to_string()));
        assert!(commands.contains(&"get".to_string()));
        assert!(commands.contains(&"cat".to_string()));
        assert!(commands.contains(&"ls".to_string()));
        assert!(commands.contains(&"stats".to_string()));
        assert!(commands.contains(&"semantic".to_string()));
        assert!(commands.contains(&"logic".to_string()));
    }

    #[test]
    fn test_multiline_validation_logic() {
        // Test the validation logic without using ValidationContext
        // (which has a complex internal API)

        // Test backslash detection
        assert!("test \\".ends_with('\\'));
        assert!(!"test".ends_with('\\'));

        // Test quote counting
        let quote_count_odd = "add \"file".chars().filter(|&c| c == '"').count();
        let quote_count_even = "add \"file\"".chars().filter(|&c| c == '"').count();
        assert_eq!(quote_count_odd % 2, 1); // Odd = unclosed
        assert_eq!(quote_count_even % 2, 0); // Even = closed

        // Test parentheses counting
        let input1 = "logic (foo";
        let open1 = input1.chars().filter(|&c| c == '(').count();
        let close1 = input1.chars().filter(|&c| c == ')').count();
        assert!(open1 > close1); // Unclosed

        let input2 = "logic (foo)";
        let open2 = input2.chars().filter(|&c| c == '(').count();
        let close2 = input2.chars().filter(|&c| c == ')').count();
        assert_eq!(open2, close2); // Balanced
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("cat", "cat"), 0);
        assert_eq!(levenshtein_distance("cat", "bat"), 1);
        assert_eq!(levenshtein_distance("cat", "ca"), 1);
        assert_eq!(levenshtein_distance("cat", "cats"), 1);
        assert_eq!(levenshtein_distance("help", "halp"), 1);
        assert_eq!(levenshtein_distance("add", "dad"), 2); // 'a'->'d' and 'd'->'a'
        assert_eq!(levenshtein_distance("exit", "exot"), 1); // 'i'->'o'
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_shell_config_aliases() {
        let config = ShellConfig::default();

        // Check that default aliases are set
        assert!(config.aliases.contains_key("ll"));
        assert_eq!(
            config
                .aliases
                .get("ll")
                .expect("test: alias 'll' should exist"),
            "ls"
        );

        assert!(config.aliases.contains_key("whoami"));
        assert_eq!(
            config
                .aliases
                .get("whoami")
                .expect("test: alias 'whoami' should exist"),
            "id"
        );

        assert!(config.aliases.contains_key("upload"));
        assert_eq!(
            config
                .aliases
                .get("upload")
                .expect("test: alias 'upload' should exist"),
            "add"
        );

        assert!(config.aliases.contains_key("download"));
        assert_eq!(
            config
                .aliases
                .get("download")
                .expect("test: alias 'download' should exist"),
            "get"
        );
    }

    #[test]
    fn test_command_completer_includes_aliases() {
        let completer = CommandCompleter::new();

        // Check that aliases are in the completion list
        assert!(completer.commands.contains(&"ll".to_string()));
        assert!(completer.commands.contains(&"whoami".to_string()));
        assert!(completer.commands.contains(&"alias".to_string()));
        assert!(completer.commands.contains(&"unalias".to_string()));
        assert!(completer.commands.contains(&"upload".to_string()));
        assert!(completer.commands.contains(&"download".to_string()));
    }
}
