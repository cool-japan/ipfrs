//! Terminal User Interface (TUI) for IPFRS
//!
//! Provides an interactive dashboard for monitoring IPFRS node status,
//! network activity, storage statistics, and more.

use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline, Tabs, Wrap},
    Frame, Terminal,
};
use std::io;
use std::time::{Duration, Instant};

/// TUI application state
pub struct App {
    /// Current tab index
    current_tab: usize,
    /// Whether the app should quit
    should_quit: bool,
    /// Node statistics
    stats: NodeStats,
    /// Network activity history (for sparkline)
    network_history: Vec<u64>,
    /// Last update time
    last_update: Instant,
}

/// Node statistics
#[derive(Debug, Clone, Default)]
struct NodeStats {
    /// Peer count
    peer_count: usize,
    /// Total blocks stored
    block_count: u64,
    /// Total storage size in bytes
    storage_size: u64,
    /// Bandwidth in/out (bytes per second)
    bandwidth_in: u64,
    bandwidth_out: u64,
    /// Uptime in seconds
    uptime: u64,
    /// Number of pinned items
    pinned_count: usize,
    /// DHT routing table size
    dht_size: usize,
}

impl Default for App {
    fn default() -> Self {
        Self {
            current_tab: 0,
            should_quit: false,
            stats: NodeStats::default(),
            network_history: vec![0; 60],
            last_update: Instant::now(),
        }
    }
}

impl App {
    /// Create a new TUI app
    pub fn new() -> Self {
        Self::default()
    }

    /// Handle key events
    fn handle_key_event(&mut self, key: event::KeyEvent) {
        match (key.code, key.modifiers) {
            (KeyCode::Char('q'), _) | (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            (KeyCode::Right | KeyCode::Tab, _) => {
                self.current_tab = (self.current_tab + 1) % 4;
            }
            (KeyCode::Left, _) => {
                self.current_tab = if self.current_tab > 0 {
                    self.current_tab - 1
                } else {
                    3
                };
            }
            (KeyCode::Char('1'), _) => self.current_tab = 0,
            (KeyCode::Char('2'), _) => self.current_tab = 1,
            (KeyCode::Char('3'), _) => self.current_tab = 2,
            (KeyCode::Char('4'), _) => self.current_tab = 3,
            _ => {}
        }
    }

    /// Update statistics (mock data for now)
    fn update_stats(&mut self) {
        // In a real implementation, this would fetch data from the IPFRS node
        // For now, we'll use mock data that changes over time

        let elapsed = self.last_update.elapsed();
        if elapsed >= Duration::from_secs(1) {
            self.stats.uptime += elapsed.as_secs();
            self.last_update = Instant::now();

            // Simulate changing stats
            use std::time::SystemTime;
            let seed = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("system time is after UNIX epoch")
                .as_secs();

            self.stats.peer_count = ((seed % 10) + 5) as usize;
            self.stats.bandwidth_in = (seed % 1000) * 1024;
            self.stats.bandwidth_out = (seed % 500) * 1024;

            // Update network history
            self.network_history.remove(0);
            self.network_history.push(self.stats.bandwidth_in / 1024);
        }
    }
}

/// Run the TUI application
pub async fn run_tui() -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = App::new();

    // Run the main loop
    let res = run_app(&mut terminal, &mut app).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    res
}

/// Main application loop
async fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) -> Result<()>
where
    <B as Backend>::Error: Send + Sync + 'static,
{
    loop {
        // Update stats
        app.update_stats();

        // Draw UI
        terminal.draw(|f| ui(f, app))?;

        // Handle events with timeout
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                app.handle_key_event(key);
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

/// Draw the UI
fn ui(f: &mut Frame, app: &App) {
    let size = f.area();

    // Create main layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(size);

    // Draw tabs
    draw_tabs(f, chunks[0], app);

    // Draw content based on current tab
    match app.current_tab {
        0 => draw_overview(f, chunks[1], app),
        1 => draw_network(f, chunks[1], app),
        2 => draw_storage(f, chunks[1], app),
        3 => draw_help(f, chunks[1]),
        _ => {}
    }

    // Draw footer
    draw_footer(f, chunks[2], app);
}

/// Draw the tab bar
fn draw_tabs(f: &mut Frame, area: Rect, app: &App) {
    let titles = vec!["Overview", "Network", "Storage", "Help"];
    let tabs = Tabs::new(titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" IPFRS Dashboard "),
        )
        .select(app.current_tab)
        .style(Style::default().fg(Color::White))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );
    f.render_widget(tabs, area);
}

/// Draw the overview tab
fn draw_overview(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(area);

    // Peer gauge
    let peer_ratio = app.stats.peer_count as f64 / 50.0;
    let peer_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Peers "))
        .gauge_style(Style::default().fg(Color::Green))
        .percent((peer_ratio * 100.0).min(100.0) as u16)
        .label(format!("{} / 50", app.stats.peer_count));
    f.render_widget(peer_gauge, chunks[0]);

    // Storage gauge
    let storage_ratio = app.stats.storage_size as f64 / (10_u64 * 1024 * 1024 * 1024) as f64;
    let storage_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Storage "))
        .gauge_style(Style::default().fg(Color::Blue))
        .percent((storage_ratio * 100.0).min(100.0) as u16)
        .label(format_bytes(app.stats.storage_size));
    f.render_widget(storage_gauge, chunks[1]);

    // Bandwidth in
    let bw_in = Paragraph::new(format!(
        "Incoming: {} /s",
        format_bytes(app.stats.bandwidth_in)
    ))
    .block(Block::default().borders(Borders::ALL).title(" Bandwidth "));
    f.render_widget(bw_in, chunks[2]);

    // Bandwidth out
    let bw_out = Paragraph::new(format!(
        "Outgoing: {} /s",
        format_bytes(app.stats.bandwidth_out)
    ));
    f.render_widget(bw_out, chunks[3]);

    // Summary
    let uptime_hours = app.stats.uptime / 3600;
    let uptime_mins = (app.stats.uptime % 3600) / 60;
    let summary = [
        format!("Uptime: {}h {}m", uptime_hours, uptime_mins),
        format!("Blocks: {}", app.stats.block_count),
        format!("Pinned: {}", app.stats.pinned_count),
        format!("DHT Size: {}", app.stats.dht_size),
    ];
    let summary_widget = Paragraph::new(summary.join("\n"))
        .block(Block::default().borders(Borders::ALL).title(" Node Info "))
        .wrap(Wrap { trim: true });
    f.render_widget(summary_widget, chunks[4]);
}

/// Draw the network tab
fn draw_network(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Min(0)])
        .split(area);

    // Network activity sparkline
    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Network Activity (KB/s) "),
        )
        .data(&app.network_history)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(sparkline, chunks[0]);

    // Connected peers list (placeholder)
    let peers: Vec<ListItem> = vec![
        ListItem::new("QmPeer1... - /ip4/192.168.1.100/tcp/4001"),
        ListItem::new("QmPeer2... - /ip4/10.0.0.50/udp/4001/quic-v1"),
        ListItem::new("QmPeer3... - /ip6/::1/tcp/4001"),
    ];
    let peer_list = List::new(peers).block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" Connected Peers ({}) ", app.stats.peer_count)),
    );
    f.render_widget(peer_list, chunks[1]);
}

/// Draw the storage tab
fn draw_storage(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Length(5),
            Constraint::Min(0),
        ])
        .split(area);

    // Storage breakdown
    let storage_info = [
        format!("Total Size: {}", format_bytes(app.stats.storage_size)),
        format!("Block Count: {}", app.stats.block_count),
        format!("Pinned Items: {}", app.stats.pinned_count),
    ];
    let storage_widget = Paragraph::new(storage_info.join("\n"))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Storage Info "),
        )
        .wrap(Wrap { trim: true });
    f.render_widget(storage_widget, chunks[0]);

    // Recent blocks (placeholder)
    let blocks: Vec<ListItem> = vec![
        ListItem::new("QmHash1... - 1.2 MB - 2 mins ago"),
        ListItem::new("QmHash2... - 534 KB - 5 mins ago"),
        ListItem::new("QmHash3... - 2.1 GB - 10 mins ago"),
    ];
    let block_list = List::new(blocks).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Recent Blocks "),
    );
    f.render_widget(block_list, chunks[1]);

    // Cache stats
    let cache_info = [
        "Cache Hit Rate: 87.3%",
        "Cache Size: 100 MB / 256 MB",
        "Evictions: 1,234",
    ];
    let cache_widget = Paragraph::new(cache_info.join("\n"))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Cache Stats "),
        )
        .wrap(Wrap { trim: true });
    f.render_widget(cache_widget, chunks[2]);
}

/// Draw the help tab
fn draw_help(f: &mut Frame, area: Rect) {
    let help_text = vec![
        Line::from(vec![
            Span::styled(
                "q",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" - Quit"),
        ]),
        Line::from(vec![
            Span::styled(
                "Tab / ←/→",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" - Switch tabs"),
        ]),
        Line::from(vec![
            Span::styled(
                "1-4",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" - Jump to tab"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Overview", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" - Node statistics and gauges"),
        ]),
        Line::from(vec![
            Span::styled("Network", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" - Peer connections and activity"),
        ]),
        Line::from(vec![
            Span::styled("Storage", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" - Block storage and cache stats"),
        ]),
        Line::from(""),
        Line::from("Press Ctrl+C or q to exit the dashboard."),
    ];

    let help = Paragraph::new(help_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Help & Keyboard Shortcuts "),
        )
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: true });
    f.render_widget(help, area);
}

/// Draw the footer
fn draw_footer(f: &mut Frame, area: Rect, app: &App) {
    let footer_text = format!(
        " IPFRS v0.2.0 | Peers: {} | Blocks: {} | Press 'q' to quit ",
        app.stats.peer_count, app.stats.block_count
    );
    let footer = Paragraph::new(footer_text)
        .style(Style::default().fg(Color::White).bg(Color::DarkGray))
        .alignment(Alignment::Center);
    f.render_widget(footer, area);
}

/// Format bytes to human-readable string
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];

    if bytes == 0 {
        return "0 B".to_string();
    }

    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_app_creation() {
        let app = App::new();
        assert_eq!(app.current_tab, 0);
        assert!(!app.should_quit);
        assert_eq!(app.stats.peer_count, 0);
    }

    #[test]
    fn test_tab_navigation() {
        let mut app = App::new();

        // Test right navigation
        app.handle_key_event(event::KeyEvent::from(KeyCode::Right));
        assert_eq!(app.current_tab, 1);

        app.handle_key_event(event::KeyEvent::from(KeyCode::Right));
        assert_eq!(app.current_tab, 2);

        // Test wraparound
        app.current_tab = 3;
        app.handle_key_event(event::KeyEvent::from(KeyCode::Right));
        assert_eq!(app.current_tab, 0);

        // Test left navigation
        app.handle_key_event(event::KeyEvent::from(KeyCode::Left));
        assert_eq!(app.current_tab, 3);
    }

    #[test]
    fn test_quit_key() {
        let mut app = App::new();

        app.handle_key_event(event::KeyEvent::from(KeyCode::Char('q')));
        assert!(app.should_quit);
    }

    #[test]
    fn test_direct_tab_selection() {
        let mut app = App::new();

        app.handle_key_event(event::KeyEvent::from(KeyCode::Char('3')));
        assert_eq!(app.current_tab, 2);

        app.handle_key_event(event::KeyEvent::from(KeyCode::Char('1')));
        assert_eq!(app.current_tab, 0);
    }
}
