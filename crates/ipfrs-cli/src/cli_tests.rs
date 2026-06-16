//! CLI parser tests - split from main.rs to maintain the 2000-line limit.
//!
//! This file is included from main.rs via `#[path]` in the `tests` module.

use super::*;
use clap::CommandFactory;

#[test]
fn test_cli_parsing() {
    // Test basic command parsing
    let cli = Cli::try_parse_from(["ipfrs", "version"]);
    assert!(cli.is_ok());
}

#[test]
fn test_init_command() {
    let cli = Cli::try_parse_from(["ipfrs", "init"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Init { data_dir } => {
                assert_eq!(data_dir, ".ipfrs");
            }
            _ => panic!("Expected Init command"),
        }
    }
}

#[test]
fn test_init_command_custom_dir() {
    let cli = Cli::try_parse_from(["ipfrs", "init", "--data-dir", "/tmp/ipfrs"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Init { data_dir } => {
                assert_eq!(data_dir, "/tmp/ipfrs");
            }
            _ => panic!("Expected Init command"),
        }
    }
}

#[test]
fn test_add_command() {
    let cli = Cli::try_parse_from(["ipfrs", "add", "test.txt"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Add { path, format } => {
                assert_eq!(path, "test.txt");
                assert_eq!(format, "text");
            }
            _ => panic!("Expected Add command"),
        }
    }
}

#[test]
fn test_add_command_json_format() {
    let cli = Cli::try_parse_from(["ipfrs", "add", "test.txt", "--format", "json"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Add { path, format } => {
                assert_eq!(path, "test.txt");
                assert_eq!(format, "json");
            }
            _ => panic!("Expected Add command"),
        }
    }
}

#[test]
fn test_get_command() {
    let cli = Cli::try_parse_from(["ipfrs", "get", "QmTest123"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Get { cid, output, timeout } => {
                assert_eq!(cid, "QmTest123");
                assert_eq!(output, None);
                assert_eq!(timeout, 30);
            }
            _ => panic!("Expected Get command"),
        }
    }
}

#[test]
fn test_get_command_with_output() {
    let cli = Cli::try_parse_from(["ipfrs", "get", "QmTest123", "-o", "output.txt"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Get { cid, output, timeout } => {
                assert_eq!(cid, "QmTest123");
                assert_eq!(output, Some("output.txt".to_string()));
                assert_eq!(timeout, 30);
            }
            _ => panic!("Expected Get command"),
        }
    }
}

#[test]
fn test_get_command_with_custom_timeout() {
    let cli = Cli::try_parse_from(["ipfrs", "get", "QmTest123", "--timeout", "60"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Get { cid, output, timeout } => {
                assert_eq!(cid, "QmTest123");
                assert_eq!(output, None);
                assert_eq!(timeout, 60);
            }
            _ => panic!("Expected Get command"),
        }
    }
}

#[test]
fn test_cat_command() {
    let cli = Cli::try_parse_from(["ipfrs", "cat", "QmTest123"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Cat { cid, timeout } => {
                assert_eq!(cid, "QmTest123");
                assert_eq!(timeout, 30);
            }
            _ => panic!("Expected Cat command"),
        }
    }
}

#[test]
fn test_cat_command_with_custom_timeout() {
    let cli = Cli::try_parse_from(["ipfrs", "cat", "QmTest123", "--timeout", "0"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Cat { cid, timeout } => {
                assert_eq!(cid, "QmTest123");
                assert_eq!(timeout, 0, "timeout=0 should disable the limit");
            }
            _ => panic!("Expected Cat command"),
        }
    }
}

#[test]
fn test_repo_compact_command() {
    let cli = Cli::try_parse_from(["ipfrs", "repo", "compact"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Repo {
                command: RepoCommands::Compact { force, format },
            } => {
                assert!(!force, "default force should be false");
                assert_eq!(format, "text");
            }
            _ => panic!("Expected Repo Compact command"),
        }
    }
}

#[test]
fn test_repo_compact_force_flag() {
    let cli = Cli::try_parse_from(["ipfrs", "repo", "compact", "--force"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Repo {
                command: RepoCommands::Compact { force, .. },
            } => {
                assert!(force, "--force should set force=true");
            }
            _ => panic!("Expected Repo Compact command"),
        }
    }
}

#[test]
fn test_ls_command() {
    let cli = Cli::try_parse_from(["ipfrs", "ls", "QmDir123"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Ls { cid, format } => {
                assert_eq!(cid, "QmDir123");
                assert_eq!(format, "text");
            }
            _ => panic!("Expected Ls command"),
        }
    }
}

#[test]
fn test_block_get_command() {
    let cli = Cli::try_parse_from(["ipfrs", "block", "get", "QmBlock123"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Block { command } => match command {
                BlockCommands::Get { cid } => {
                    assert_eq!(cid, "QmBlock123");
                }
                _ => panic!("Expected Block Get command"),
            },
            _ => panic!("Expected Block command"),
        }
    }
}

#[test]
fn test_block_put_command() {
    let cli = Cli::try_parse_from(["ipfrs", "block", "put", "data.bin"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Block { command } => match command {
                BlockCommands::Put { path, format } => {
                    assert_eq!(path, "data.bin");
                    assert_eq!(format, "text");
                }
                _ => panic!("Expected Block Put command"),
            },
            _ => panic!("Expected Block command"),
        }
    }
}

#[test]
fn test_block_stat_command() {
    let cli = Cli::try_parse_from(["ipfrs", "block", "stat", "QmBlock123"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Block { command } => match command {
                BlockCommands::Stat { cid, format } => {
                    assert_eq!(cid, "QmBlock123");
                    assert_eq!(format, "text");
                }
                _ => panic!("Expected Block Stat command"),
            },
            _ => panic!("Expected Block command"),
        }
    }
}

#[test]
fn test_block_rm_command() {
    let cli = Cli::try_parse_from(["ipfrs", "block", "rm", "QmBlock123"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Block { command } => match command {
                BlockCommands::Rm { cid, force } => {
                    assert_eq!(cid, "QmBlock123");
                    assert!(!force);
                }
                _ => panic!("Expected Block Rm command"),
            },
            _ => panic!("Expected Block command"),
        }
    }
}

#[test]
fn test_block_rm_command_force() {
    let cli = Cli::try_parse_from(["ipfrs", "block", "rm", "QmBlock123", "--force"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Block { command } => match command {
                BlockCommands::Rm { cid, force } => {
                    assert_eq!(cid, "QmBlock123");
                    assert!(force);
                }
                _ => panic!("Expected Block Rm command"),
            },
            _ => panic!("Expected Block command"),
        }
    }
}

#[test]
fn test_ping_command() {
    let cli = Cli::try_parse_from(["ipfrs", "ping", "12D3KooWTest"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Ping { peer_id, count } => {
                assert_eq!(peer_id, "12D3KooWTest");
                assert_eq!(count, 5);
            }
            _ => panic!("Expected Ping command"),
        }
    }
}

#[test]
fn test_ping_command_custom_count() {
    let cli = Cli::try_parse_from(["ipfrs", "ping", "12D3KooWTest", "-c", "10"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Ping { peer_id, count } => {
                assert_eq!(peer_id, "12D3KooWTest");
                assert_eq!(count, 10);
            }
            _ => panic!("Expected Ping command"),
        }
    }
}

#[test]
fn test_id_command() {
    let cli = Cli::try_parse_from(["ipfrs", "id"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Id { format } => {
                assert_eq!(format, "text");
            }
            _ => panic!("Expected Id command"),
        }
    }
}

#[test]
fn test_id_command_json_format() {
    let cli = Cli::try_parse_from(["ipfrs", "id", "--format", "json"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Id { format } => {
                assert_eq!(format, "json");
            }
            _ => panic!("Expected Id command"),
        }
    }
}

#[test]
fn test_version_command() {
    let cli = Cli::try_parse_from(["ipfrs", "version"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Version => {}
            _ => panic!("Expected Version command"),
        }
    }
}

#[test]
fn test_shell_command() {
    let cli = Cli::try_parse_from(["ipfrs", "shell"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Shell { data_dir } => {
                assert_eq!(data_dir, ".ipfrs");
            }
            _ => panic!("Expected Shell command"),
        }
    }
}

#[test]
fn test_verbose_flag() {
    let cli = Cli::try_parse_from(["ipfrs", "--verbose", "version"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        assert!(cli.verbose);
    }
}

#[test]
fn test_no_color_flag() {
    let cli = Cli::try_parse_from(["ipfrs", "--no-color", "version"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        assert!(cli.no_color);
    }
}

#[test]
fn test_config_flag() {
    let cli = Cli::try_parse_from(["ipfrs", "--config", "/tmp/config.toml", "version"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        assert_eq!(cli.config, Some("/tmp/config.toml".to_string()));
    }
}

#[test]
fn test_daemon_run_command() {
    let cli = Cli::try_parse_from(["ipfrs", "daemon", "run"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Daemon { command } => match command {
                Some(DaemonCommands::Run { data_dir }) => {
                    assert_eq!(data_dir, ".ipfrs");
                }
                _ => panic!("Expected Daemon Run command"),
            },
            _ => panic!("Expected Daemon command"),
        }
    }
}

#[test]
fn test_daemon_start_command() {
    let cli = Cli::try_parse_from(["ipfrs", "daemon", "start"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Daemon { command } => match command {
                Some(DaemonCommands::Start {
                    data_dir,
                    pid_file,
                    log_file,
                }) => {
                    assert_eq!(data_dir, ".ipfrs");
                    assert_eq!(pid_file, ".ipfrs/daemon.pid");
                    assert_eq!(log_file, ".ipfrs/daemon.log");
                }
                _ => panic!("Expected Daemon Start command"),
            },
            _ => panic!("Expected Daemon command"),
        }
    }
}

#[test]
fn test_daemon_stop_command() {
    let cli = Cli::try_parse_from(["ipfrs", "daemon", "stop"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Daemon { command } => match command {
                Some(DaemonCommands::Stop { pid_file }) => {
                    assert_eq!(pid_file, ".ipfrs/daemon.pid");
                }
                _ => panic!("Expected Daemon Stop command"),
            },
            _ => panic!("Expected Daemon command"),
        }
    }
}

#[test]
fn test_completions_bash() {
    let cli = Cli::try_parse_from(["ipfrs", "completions", "bash"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Completions { shell } => {
                assert!(matches!(shell, CompletionShell::Bash));
            }
            _ => panic!("Expected Completions command"),
        }
    }
}

#[test]
fn test_completions_zsh() {
    let cli = Cli::try_parse_from(["ipfrs", "completions", "zsh"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Completions { shell } => {
                assert!(matches!(shell, CompletionShell::Zsh));
            }
            _ => panic!("Expected Completions command"),
        }
    }
}

#[test]
fn test_tensor_add_command() {
    let cli = Cli::try_parse_from(["ipfrs", "tensor", "add", "model.safetensors"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Tensor { command } => match command {
                TensorCommands::Add { path, format } => {
                    assert_eq!(path, "model.safetensors");
                    assert_eq!(format, "text");
                }
                _ => panic!("Expected Tensor Add command"),
            },
            _ => panic!("Expected Tensor command"),
        }
    }
}

#[test]
fn test_tensor_get_command() {
    let cli = Cli::try_parse_from(["ipfrs", "tensor", "get", "QmTensor123"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Tensor { command } => match command {
                TensorCommands::Get { cid, output } => {
                    assert_eq!(cid, "QmTensor123");
                    assert_eq!(output, None);
                }
                _ => panic!("Expected Tensor Get command"),
            },
            _ => panic!("Expected Tensor command"),
        }
    }
}

#[test]
fn test_tensor_info_command() {
    let cli = Cli::try_parse_from(["ipfrs", "tensor", "info", "QmTensor123"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Tensor { command } => match command {
                TensorCommands::Info { cid, format } => {
                    assert_eq!(cid, "QmTensor123");
                    assert_eq!(format, "text");
                }
                _ => panic!("Expected Tensor Info command"),
            },
            _ => panic!("Expected Tensor command"),
        }
    }
}

#[test]
fn test_invalid_command() {
    let cli = Cli::try_parse_from(["ipfrs", "invalid_command"]);
    assert!(cli.is_err());
}

#[test]
fn test_missing_required_argument() {
    let cli = Cli::try_parse_from(["ipfrs", "add"]);
    assert!(cli.is_err());
}

#[test]
fn test_cli_help_generation() {
    let mut cmd = Cli::command();
    let help = cmd.render_help();
    let help_str = help.to_string();
    assert!(help_str.contains("ipfrs"));
    assert!(help_str.contains("IPFRS"));
}

#[test]
fn test_cli_has_all_major_commands() {
    let cmd = Cli::command();
    let subcommands: Vec<_> = cmd.get_subcommands().map(|c| c.get_name()).collect();

    // Check for essential commands
    assert!(subcommands.contains(&"init"));
    assert!(subcommands.contains(&"add"));
    assert!(subcommands.contains(&"get"));
    assert!(subcommands.contains(&"cat"));
    assert!(subcommands.contains(&"ls"));
    assert!(subcommands.contains(&"block"));
    assert!(subcommands.contains(&"daemon"));
    assert!(subcommands.contains(&"version"));
    assert!(subcommands.contains(&"shell"));
    assert!(subcommands.contains(&"plugin"));
}

#[test]
fn test_quiet_flag() {
    let cli = Cli::try_parse_from(["ipfrs", "--quiet", "version"]).expect("test: CLI parse should succeed");
    assert!(cli.quiet);
}

#[test]
fn test_quiet_flag_short() {
    let cli = Cli::try_parse_from(["ipfrs", "-q", "version"]).expect("test: CLI parse should succeed");
    assert!(cli.quiet);
}

#[test]
fn test_quiet_and_verbose_together() {
    // These flags are independent - both can be set
    let cli = Cli::try_parse_from(["ipfrs", "-q", "-v", "version"]).expect("test: CLI parse should succeed");
    assert!(cli.quiet);
    assert!(cli.verbose);
}

#[test]
fn test_exit_codes_defined() {
    // Ensure exit codes are accessible
    assert_eq!(exit_codes::SUCCESS, 0);
    assert_eq!(exit_codes::ERROR, 1);
    assert_eq!(exit_codes::USAGE_ERROR, 2);
    assert_eq!(exit_codes::NOT_FOUND, 3);
    assert_eq!(exit_codes::PERMISSION_DENIED, 4);
    assert_eq!(exit_codes::NETWORK_ERROR, 5);
    assert_eq!(exit_codes::IO_ERROR, 6);
    assert_eq!(exit_codes::TIMEOUT, 7);
    assert_eq!(exit_codes::CONFIG_ERROR, 8);
}

#[test]
fn test_plugin_list_command() {
    let cli = Cli::try_parse_from(["ipfrs", "plugin", "list"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Plugin { command } => match command {
                PluginCommands::List => {}
                _ => panic!("Expected PluginCommands::List"),
            },
            _ => panic!("Expected Plugin command"),
        }
    }
}

#[test]
fn test_plugin_info_command() {
    let cli = Cli::try_parse_from(["ipfrs", "plugin", "info", "test-plugin"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Plugin { command } => match command {
                PluginCommands::Info { name } => {
                    assert_eq!(name, "test-plugin");
                }
                _ => panic!("Expected PluginCommands::Info"),
            },
            _ => panic!("Expected Plugin command"),
        }
    }
}

#[test]
fn test_plugin_run_command() {
    let cli = Cli::try_parse_from(["ipfrs", "plugin", "run", "my-plugin", "--arg1", "value"]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Plugin { command } => match command {
                PluginCommands::Run { name, args } => {
                    assert_eq!(name, "my-plugin");
                    assert_eq!(args, vec!["--arg1", "value"]);
                }
                _ => panic!("Expected PluginCommands::Run"),
            },
            _ => panic!("Expected Plugin command"),
        }
    }
}

#[test]
fn test_plugin_run_with_multiple_args() {
    let cli = Cli::try_parse_from([
        "ipfrs",
        "plugin",
        "run",
        "my-plugin",
        "arg1",
        "arg2",
        "--flag",
        "-v",
    ]);
    assert!(cli.is_ok());
    if let Ok(cli) = cli {
        match cli.command {
            Commands::Plugin { command } => match command {
                PluginCommands::Run { name, args } => {
                    assert_eq!(name, "my-plugin");
                    assert_eq!(args, vec!["arg1", "arg2", "--flag", "-v"]);
                }
                _ => panic!("Expected PluginCommands::Run"),
            },
            _ => panic!("Expected Plugin command"),
        }
    }
}
