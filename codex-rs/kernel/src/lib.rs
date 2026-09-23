mod cli;
mod controller;
mod domain;
mod host;
mod memory;
mod roles;
mod worker;

pub use cli::KernelCli;
pub use cli::KernelCommand;

pub async fn run(
    cli: KernelCli,
    paths: codex_arg0::Arg0DispatchPaths,
    overrides: codex_utils_cli::CliConfigOverrides,
) -> anyhow::Result<()> {
    match cli.command {
        KernelCommand::Optimize(args) => controller::run(args, paths, overrides).await,
    }
}
