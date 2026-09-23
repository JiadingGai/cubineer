use super::*;

impl Host {
    pub async fn start(
        args: &OptimizeArgs,
        paths: Arg0DispatchPaths,
        overrides: codex_utils_cli::CliConfigOverrides,
    ) -> Result<Self> {
        let mut overrides = overrides.parse_overrides().map_err(anyhow::Error::msg)?;
        overrides.extend([
            ("model".into(), args.model.clone().into()),
            ("model_provider".into(), args.provider.clone().into()),
            ("features.memories".into(), false.into()),
        ]);
        let config = ConfigBuilder::default()
            .cli_overrides(overrides.clone())
            .harness_overrides(ConfigOverrides {
                cwd: Some(args.output.clone()),
                ..Default::default()
            })
            .strict_config(args.strict_config)
            .build()
            .await?;
        let parallel_sessions = args
            .parallel_sessions
            .min(config.agent_max_threads.unwrap_or(6));
        ensure!(
            parallel_sessions > 0,
            "parallel session limit must be positive"
        );
        let runtime_paths = ExecServerRuntimePaths::from_optional_paths(
            paths.codex_self_exe.clone(),
            paths.codex_linux_sandbox_exe.clone(),
        )?;
        #[cfg(target_os = "macos")]
        let runtime_paths = runtime_paths.with_allowed_symlinked_codex_home(
            codex_config::allowed_symlinked_codex_home(
                &config.config_layer_stack,
                &config.codex_home,
            ),
        );
        let environment_manager = EnvironmentManager::from_codex_home(
            config.codex_home.clone(),
            Some(runtime_paths),
            config.http_client_factory(),
        )
        .await?;
        let state = codex_core::init_state_db(&config).await;
        let client = InProcessAppServerClient::start(InProcessClientStartArgs {
            arg0_paths: paths,
            config: Arc::new(config),
            cli_overrides: overrides,
            loader_overrides: LoaderOverrides::default(),
            strict_config: args.strict_config,
            cloud_config_bundle: CloudConfigBundleLoader::default(),
            feedback: CodexFeedback::new(),
            log_db: None,
            state_db: state.clone(),
            environment_manager: Arc::new(environment_manager),
            config_warnings: Vec::new(),
            session_source: SessionSource::Exec,
            enable_codex_api_key_env: true,
            client_name: "cubineer".into(),
            client_version: env!("CARGO_PKG_VERSION").into(),
            experimental_api: true,
            mcp_server_openai_form_elicitation: false,
            opt_out_notification_methods: Vec::new(),
            channel_capacity: 256,
        })
        .await?;
        let started = async {
            let result = rpc(
                &client,
                "thread/start",
                json!({
                    "model": args.model, "modelProvider": args.provider, "cwd": args.output,
                    "allowProviderModelFallback": false,
                }),
            )
            .await?;
            ensure!(
                result["model"] == args.model && result["modelProvider"] == args.provider,
                "resolved root provider/model differs from requested configuration"
            );
            let root = result["thread"]["id"]
                .as_str()
                .context("missing root thread ID")?
                .to_owned();
            Ok::<_, anyhow::Error>(root)
        }
        .await;
        let root = match started {
            Ok(root) => root,
            Err(error) => {
                let _ = client.shutdown().await;
                return Err(error);
            }
        };
        Ok(Self {
            client,
            state,
            root,
            provider: args.provider.clone(),
            model: args.model.clone(),
            timeout: Duration::from_secs(args.timeout_seconds),
            usage: BTreeMap::new(),
            parallel_sessions,
        })
    }
}
