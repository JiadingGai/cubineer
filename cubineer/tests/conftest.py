import pytest


@pytest.fixture
def hardware():
    return {
        "reference_seconds": 0.002,
        "candidate_seconds": 0.001,
        "kernel_names": ["vector_add"],
        "ncu_csv": "Kernel Name,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,l1tex__t_sector_hit_rate.pct,smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct\n,%,%,ns,%,%\nvector_add,12.0,80.0,1000000.0,10.0,60.0\n",
        "sass": "Function : vector_add\n/*0000*/ FFMA R1, R2, R3, R4;\n/*0010*/ HGMMA.64x64x16.F32 R4, desc[UR4], R0, !UPT;\n",
        "ncu_rules": "OPT Est. Speedup: 20%\n  Improve occupancy.\n\nOPT Est. Speedup: 70%\n  Coalesce global memory loads.\n",
    }
