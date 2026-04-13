package org;

/**
 * SimulationConfig.java
 *
 * Single source of truth for every tunable parameter in the simulation.
 * Change a value here and every module picks it up automatically.
 */
public class SimulationConfig {

    // ── CloudSim runtime ──────────────────────────────────────────────────────
    public static final int     NUM_USERS       = 1;
    public static final boolean TRACE_FLAG      = false;

    // ── Datacenter ────────────────────────────────────────────────────────────
    public static final String  DC_ARCH         = "x86";
    public static final String  DC_OS           = "Linux";
    public static final String  DC_VMM          = "Xen";
    public static final double  DC_TIME_ZONE    = 10.0;
    public static final double  DC_COST         = 3.0;
    public static final double  DC_COST_MEM     = 0.05;
    public static final double  DC_COST_STORAGE = 0.001;
    public static final double  DC_COST_BW      = 0.0;

    // ── Hosts ─────────────────────────────────────────────────────────────────
    public static final int     NUM_HOSTS       = 3;
    public static final int     HOST_MIPS       = 4000;
    public static final int     HOST_PES        = 4;
    public static final int     HOST_RAM        = 8192;
    public static final long    HOST_STORAGE    = 1_000_000;
    public static final int     HOST_BW         = 10_000;

    // ── VMs ───────────────────────────────────────────────────────────────────
    public static final int     NUM_VMS         = 4;

    /**
     * Heterogeneous MIPS per VM — 4× ratio between slowest and fastest.
     *
     *   VM 0 →  500 MIPS  (slow)
     *   VM 1 → 1000 MIPS  (medium)
     *   VM 2 → 1500 MIPS  (medium-fast)
     *   VM 3 → 2000 MIPS  (fast)
     *
     * This spread means a 3 000 MI task takes 6.0 s on VM 0 but only 1.5 s
     * on VM 3, making VM-selection decisions highly consequential and giving
     * the RL agent's speed-aware state a genuine advantage over homogeneous
     * configurations where all VMs look the same.
     *
     * Array length must equal NUM_VMS.
     */
    public static final int[]   VM_MIPS_VALUES  = {500, 1000, 1500, 2000};

    public static final int     VM_PES          = 1;
    public static final int     VM_RAM          = 1024;
    public static final long    VM_BW           = 1000;
    public static final long    VM_SIZE         = 10_000;
    public static final String  VM_VMM          = "Xen";

    // ── Cloudlets ─────────────────────────────────────────────────────────────
    public static final int     NUM_CLOUDLETS    = 40;
    public static final int     CL_PES           = 1;
    public static final long    CL_FILE_SIZE     = 300;
    public static final long    CL_OUTPUT_SIZE   = 300;

    // Length pattern: id%3==0 → SHORT, id%3==1 → MEDIUM, id%3==2 → LONG
    public static final long    CL_LENGTH_SHORT  = 1_000;
    public static final long    CL_LENGTH_MEDIUM = 2_000;
    public static final long    CL_LENGTH_LONG   = 3_000;

    // ── Energy model ──────────────────────────────────────────────────────────
    public static final double  IDLE_POWER       = 100.0;
    public static final double  MAX_POWER        = 200.0;

    private SimulationConfig() {}
}
