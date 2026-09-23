use super::*;

pub(super) fn tensor(sass: &Value) -> Result<Option<Output>> {
    if sass.as_object().is_none_or(serde_json::Map::is_empty) {
        return Ok(None);
    }
    let count = |kind: &str| sass[format!("{kind}_count")].as_u64().unwrap_or_default();
    let kind = ["wgmma", "hmma", "imma", "dmma", "bmma"]
        .into_iter()
        .find(|kind| count(kind) > 0)
        .unwrap_or(if sass["uses_cublas"] == true {
            "cublas"
        } else {
            "none"
        });
    let mut output = Output::base(&format!("wgmma_instruction_detector:{kind}"))?;
    let value = count(kind);
    match kind {
        "wgmma" => {
            output.summary = format!("Kernel uses {value} WGMMA (Hopper tensor core) instructions");
            output.observed(json!({"WGMMA count": value, "HMMA count": count("hmma"), "Uses TMA": if count("tma") > 0 { "Yes" } else { "No" }}));
            if count("tma") > 0 {
                output.recommendations[1] = "TMA is being used for efficient memory access".into();
            }
        }
        "hmma" => {
            output.summary = format!(
                "Kernel uses {value} HMMA instructions (Ampere-style) instead of WGMMA (Hopper)"
            );
            output.observed(json!({"HMMA count": value, "WGMMA count": 0}));
        }
        "imma" => {
            output.summary =
                format!("Kernel uses {value} IMMA instructions (INT8/INT4 tensor core ops)");
            output.observed(json!({"IMMA count": value, "WGMMA count": count("wgmma"), "HMMA count": count("hmma")}));
        }
        "dmma" => {
            output.summary =
                format!("Kernel uses {value} DMMA instructions (FP64 tensor core ops)");
            output.observed(json!({"DMMA count": value}));
        }
        "bmma" => {
            output.summary =
                format!("Kernel uses {value} BMMA instructions (INT1 binary tensor core ops)");
            output.observed(json!({"BMMA count": value}));
        }
        "cublas" => {
            let symbols: Vec<_> = sass["cublas_symbols"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            output.summary = format!(
                "Kernel links cuBLAS ({}). cuBLAS auto-uses tensor cores on modern GPUs.",
                symbols
                    .iter()
                    .take(2)
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            output.observed(json!({"cuBLAS symbols": symbols.iter().take(3).copied().collect::<Vec<_>>().join(", "), "WGMMA count": 0, "HMMA count": 0, "Note": "cuBLAS JIT-compiles TC kernels at runtime; SASS only shows the dispatcher"}));
        }
        "none" => {
            output.summary = "Kernel has 0 tensor core instructions (WGMMA/HMMA/IMMA/DMMA/BMMA) in compiled binary".into();
            output.observed(json!({"WGMMA count": 0, "HMMA count": 0, "IMMA count": 0, "DMMA count": 0, "FFMA (FP32 FMA) count": count("ffma"), "LDG/STG count": format!("{}/{}", count("ldg"), count("stg"))}));
        }
        _ => unreachable!(),
    }
    Ok(Some(output))
}

pub(super) fn spills(metrics: &Value, sass: &Value) -> Result<Option<Output>> {
    if sass.as_object().is_none_or(serde_json::Map::is_empty) {
        return Ok(None);
    }
    let stl = sass["stl_count"].as_u64().unwrap_or_default();
    let ldl = sass["ldl_count"].as_u64().unwrap_or_default();
    if stl + ldl == 0 {
        return Ok(None);
    }
    let get = |key: &str, default| -> Result<f64> {
        match metrics.get(key) {
            None => Ok(default),
            Some(value) => value
                .as_f64()
                .context(format!("nonnumeric spill metric {key}")),
        }
    };
    let scoreboard = get(
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        0.0,
    )?;
    let compute = get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0.0)?;
    let memory = get(
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        0.0,
    )?;
    let local = get("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", 0.0)?
        + get("l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum", 0.0)?;
    let total = local
        + get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", 0.0)?
        + get("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", 0.0)?;
    let ratio = if total > 0.0 { local / total } else { 0.0 };
    let occupancy = get("launch__occupancy_limit_registers", 1.0)?;
    let activity = get("smsp__issue_active.avg.pct_of_peak_sustained_active", 100.0)?;
    let eligible = get("smsp__warps_eligible.avg.per_cycle_active", 100.0)?;
    if !((scoreboard > 15.0 && compute < 50.0 && memory < 50.0)
        || (memory > 60.0 && ratio > 0.3)
        || (occupancy < 1.0 && (activity < 30.0 || eligible < 0.6)))
    {
        return Ok(None);
    }
    let mut output = Output::base("register_spill_detector")?;
    output.summary = format!("{stl} STL + {ldl} LDL spill instructions detected");
    output.observed(json!({"STL (Store to Local)": stl, "LDL (Load from Local)": ldl, "long_scoreboard": format!("{scoreboard:.1}%"), "compute_throughput": format!("{compute:.1}%"), "memory_throughput": format!("{memory:.1}%"), "local_traffic_ratio": format!("{:.1}%", ratio * 100.0)}));
    Ok(Some(output))
}
