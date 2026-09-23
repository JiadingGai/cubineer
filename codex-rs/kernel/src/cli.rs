use clap::Args;
use clap::Subcommand;
use clap::ValueEnum;
use std::path::PathBuf;

#[derive(Debug, Args)]
pub struct KernelCli {
    #[command(subcommand)]
    pub command: KernelCommand,
}

#[derive(Debug, Subcommand)]
pub enum KernelCommand {
    /// Optimize a kernel using evaluator-scored native Codex subagents.
    Optimize(OptimizeArgs),
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Evaluator {
    Gpu,
    Simulated,
}

#[derive(Debug, Args)]
pub struct OptimizeArgs {
    #[arg(long, default_value = "kernelbench", value_parser = ["kernelbench", "kernelbench_veomni"])]
    pub dataset: String,
    #[arg(long)]
    pub dataset_root: PathBuf,
    #[arg(long)]
    pub cutlass_root: Option<PathBuf>,
    #[arg(long)]
    pub task: String,
    #[arg(long, default_value_t = 1)]
    pub level: u8,
    #[arg(long, default_value = "greedy", value_parser = ["greedy", "mcts"])]
    pub strategy: String,
    #[arg(long)]
    pub search_config: Option<PathBuf>,
    #[arg(long)]
    pub iterations: Option<usize>,
    #[arg(long)]
    pub candidates: Option<usize>,
    #[arg(long, default_value_t = 5)]
    pub parallel_sessions: usize,
    #[arg(long)]
    pub model: String,
    #[arg(long)]
    pub provider: String,
    #[arg(long)]
    pub reference_model: Option<String>,
    #[arg(long)]
    pub profile_model: Option<String>,
    #[arg(long)]
    pub memory_model: Option<String>,
    #[arg(long)]
    pub python: Option<PathBuf>,
    #[arg(long)]
    pub worker: Option<PathBuf>,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, value_enum, default_value = "gpu")]
    pub evaluator: Evaluator,
    #[arg(
        long,
        help = "Trusted synthetic hardware scenario; never candidate-controlled"
    )]
    pub scenario: Option<PathBuf>,
    #[arg(long, default_value = "proactive", value_parser = ["proactive", "reactive"])]
    pub profiling: String,
    #[arg(long)]
    pub ncu_full: bool,
    #[arg(long)]
    pub memory_off: bool,
    #[arg(long)]
    pub hint: Option<String>,
    #[arg(long, value_delimiter = ',', default_value = "0")]
    pub gpus: Vec<usize>,
    #[arg(long, default_value_t = 900)]
    pub timeout_seconds: u64,
    #[arg(long)]
    pub strict_config: bool,
}

impl OptimizeArgs {
    pub fn backend(&self) -> &'static str {
        match self.evaluator {
            Evaluator::Gpu => "gpu",
            Evaluator::Simulated => "simulated",
        }
    }
}
