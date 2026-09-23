use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    kernel: codex_kernel::KernelCli,
    #[command(flatten)]
    config: codex_utils_cli::CliConfigOverrides,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    codex_arg0::arg0_dispatch_or_else(move |paths| async move {
        codex_kernel::run(cli.kernel, paths, cli.config).await
    })
}
