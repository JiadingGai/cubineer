use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    kernel: codex_kernel::KernelCli,
    #[command(flatten)]
    config: codex_utils_cli::CliConfigOverrides,
}

fn main() -> anyhow::Result<()> {
    codex_arg0::arg0_dispatch_or_else(|paths| async move {
        let cli = Cli::parse();
        codex_kernel::run(cli.kernel, paths, cli.config).await
    })
}
