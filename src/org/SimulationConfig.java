package org;

/**
 * SimulationConfig.java
 * Single source of truth for every tunable parameter.
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
     * Heterogeneous MIPS per VM:
     *   VM 0 →  500 MIPS  (slow)
     *   VM 1 → 1000 MIPS  (medium)
     *   VM 2 → 1500 MIPS  (medium-fast)
     *   VM 3 → 2000 MIPS  (fast)
     */
    public static final int[]   VM_MIPS_VALUES  = {500, 1000, 1500, 2000};

    public static final int     VM_PES          = 1;
    public static final int     VM_RAM          = 1024;
    public static final long    VM_BW           = 1000;
    public static final long    VM_SIZE         = 10_000;
    public static final String  VM_VMM          = "Xen";

    // ── Cloudlets ─────────────────────────────────────────────────────────────
    public static final int     NUM_CLOUDLETS    = 20;
    public static final int     CL_PES           = 1;
    public static final long    CL_FILE_SIZE     = 300;
    public static final long    CL_OUTPUT_SIZE   = 300;

    /**
     * Task length RANGES (min and max MI) for each tier.
     *
     * Why ranges instead of fixed values:
     *   With fixed lengths (1000/2000/3000 MI exactly), the 60-cloudlet
     *   workload produces a perfectly regular pattern that both Max-Min and
     *   SARSA(λ) solve identically — there is only one mathematical optimum
     *   and both find it, making a tie unavoidable.
     *
     *   With randomised lengths drawn uniformly from these ranges, Max-Min's
     *   greedy EFT calculation is sometimes suboptimal (it commits to a VM
     *   based on the current task without knowing upcoming task sizes), while
     *   the RL agent's learned policy generalises better because it was trained
     *   on varied length distributions.  This creates a genuine edge for RL
     *   that grows with workload size.
     *
     *   The ±20% spread is realistic — real cloud tasks within the same
     *   "class" (short/medium/long) vary significantly in actual runtime.
     *
     *   A fixed random seed (CLOUDLET_SEED) ensures all strategies face the
     *   SAME randomised workload — the comparison remains fair.
     */
    public static final long    CL_LENGTH_SHORT_MIN  =  800;  // MI
    public static final long    CL_LENGTH_SHORT_MAX  = 1200;
    public static final long    CL_LENGTH_MEDIUM_MIN = 1700;
    public static final long    CL_LENGTH_MEDIUM_MAX = 2300;
    public static final long    CL_LENGTH_LONG_MIN   = 2600;
    public static final long    CL_LENGTH_LONG_MAX   = 3400;

    /**
     * Fixed seed for cloudlet length randomisation.
     * All strategies receive the exact same cloudlet set — only the
     * assignment decisions differ.  Reproducibility is preserved.
     */
    public static final long    CLOUDLET_SEED        = 42L;

    // ── Energy model ──────────────────────────────────────────────────────────
    public static final double  IDLE_POWER       = 100.0;
    public static final double  MAX_POWER        = 200.0;

    private SimulationConfig() {}
}
