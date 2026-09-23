use super::*;

pub(super) fn analyze(tool: &Tool, metrics: &Value) -> Result<Option<Output>> {
    let name = tool.name.as_str();
    let get = |key: &str, default| number(metrics, key, default);
    let mut output = match name {
        "occupancy_limiter" => {
            let occupancy = get("sm__warps_active.avg.pct_of_peak_sustained_active", 100.0);
            if occupancy >= 50.0 {
                return Ok(None);
            }
            let limits = [
                ("registers", get("launch__occupancy_limit_registers", 100.0)),
                (
                    "shared_memory",
                    get("launch__occupancy_limit_shared_mem", 100.0),
                ),
                ("block_count", get("launch__occupancy_limit_blocks", 100.0)),
            ];
            let limiter = limits
                .iter()
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .context("occupancy limiters")?
                .0;
            let registers = metric(metrics, "launch__registers_per_thread", json!(32));
            let mut output = Output::base(&format!("{name}:{limiter}"))?;
            output.severity = if occupancy < 30.0 { "high" } else { "medium" }.into();
            output.title = format!("Low Occupancy (Limited by {})", title(limiter));
            output.summary = format!("Warp occupancy at {occupancy:.1}% - limited by {limiter}");
            output.observed(json!({"Warp occupancy": format!("{occupancy:.1}%"), "Registers per thread": display(&registers), "Occupancy limit (registers)": format!("{}%", display(&metric(metrics, "launch__occupancy_limit_registers", json!(100)))), "Occupancy limit (shared mem)": format!("{}%", display(&metric(metrics, "launch__occupancy_limit_shared_mem", json!(100)))), "Occupancy limit (blocks)": format!("{}%", display(&metric(metrics, "launch__occupancy_limit_blocks", json!(100))))}));
            if limiter == "registers" {
                output.recommendations[0] = format!(
                    "Current register usage: {:.0} per thread (target: <64)",
                    get("launch__registers_per_thread", 32.0)
                );
            }
            return Ok(Some(output));
        }
        "warp_stall_analyzer" => {
            let reasons = [
                "memory_dependency",
                "short_scoreboard",
                "long_scoreboard",
                "barrier",
                "branch_resolving",
            ];
            let values: Vec<_> = reasons
                .iter()
                .map(|reason| {
                    get(
                        &format!("smsp__warp_issue_stalled_{reason}_per_warp_active.pct"),
                        0.0,
                    )
                })
                .collect();
            let index = (0..values.len())
                .min_by(|a, b| values[*b].total_cmp(&values[*a]))
                .context("stall metrics")?;
            let value = values[index];
            if value < 40.0 {
                return Ok(None);
            }
            let reason = reasons[index];
            let mut output = Output::base(&format!("{name}:{reason}"))?;
            output.severity = if value >= 60.0 { "high" } else { "medium" }.into();
            output.title = format!("High Warp Stalls ({})", title(reason));
            output.summary = format!("{value:.1}% of warps stalled on {reason}");
            let labels = [
                "Memory dependency stalls",
                "Short scoreboard stalls",
                "Long scoreboard stalls",
                "Barrier stalls",
                "Branch resolving stalls",
            ];
            output.metrics_observed = labels
                .into_iter()
                .zip(values)
                .map(|(label, value)| (label.into(), format!("{value:.1}%").into()))
                .collect();
            output
                .metrics_observed
                .insert("Dominant stall".into(), reason.into());
            return Ok(Some(output));
        }
        _ => Output::base(name)?,
    };
    match name {
        "tensor_core_underutilization" => {
            let value = get(
                "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
                100.0,
            );
            if value >= 10.0 {
                return Ok(None);
            }
            output.summary = format!(
                "Only {value:.1}% tensor core utilization (essentially not using Tensor Cores)"
            );
            output.observed(json!({"tensor_core_utilization": format!("{value:.1}%")}));
        }
        "memory_coalescing" => {
            let l1 = get("l1tex__t_sector_hit_rate.pct", 100.0);
            let l2 = get("lts__t_sector_hit_rate.pct", 100.0);
            let stalls = get(
                "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
                0.0,
            );
            let critical = l1 < 30.0 && stalls > 40.0;
            if !(critical || l1 < 70.0 && stalls > 20.0) {
                return Ok(None);
            }
            output.severity = if critical { "high" } else { "medium" }.into();
            output.summary =
                format!("L1 hit rate: {l1:.1}% (target: >70.0%), memory stalls: {stalls:.1}%");
            output.observed(json!({"L1 cache hit rate": format!("{l1:.1}%"), "L2 cache hit rate": format!("{l2:.1}%"), "Memory dependency stalls": format!("{stalls:.1}%")}));
        }
        "high_dram_throughput" => {
            let mut value = get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0.0);
            if value == 0.0 {
                value = get(
                    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                    0.0,
                );
            }
            if value < 80.0 {
                return Ok(None);
            }
            output.severity = if value >= 95.0 { "high" } else { "medium" }.into();
            output.title = format!(
                "Memory-Bound Kernel{}",
                if value >= 95.0 { " (Saturated)" } else { "" }
            );
            output.summary =
                format!("DRAM throughput at {value:.1}% of peak - kernel is memory-bound");
            output.observed(json!({"DRAM throughput": format!("{value:.1}% of peak")}));
        }
        "high_register_usage" => {
            let value = get("launch__registers_per_thread", 32.0);
            if value < 96.0 {
                return Ok(None);
            }
            output.severity = if value >= 128.0 { "high" } else { "medium" }.into();
            output.title = format!(
                "High Register Usage{}",
                if value >= 128.0 {
                    " (Likely Spilling)"
                } else {
                    ""
                }
            );
            output.summary =
                format!("{value:.0} registers per thread - may cause register spilling");
            output.observed(json!({"Registers per thread": format!("{value:.0}")}));
        }
        "barrier_stall_detector" => {
            let value = get("smsp__warp_issue_stalled_barrier_per_warp_active.pct", 0.0);
            if value < 30.0 {
                return Ok(None);
            }
            output.summary = format!("{value:.1}% of warps stalled at __syncthreads()");
            output.observed(json!({"Barrier stalls": format!("{value:.1}%")}));
        }
        _ => bail!("unsupported analyzer {name}"),
    }
    Ok(Some(output))
}
