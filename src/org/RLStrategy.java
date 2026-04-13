package org;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.Vm;
import java.util.*;

/**
 * RLStrategy v4 — Speed-aware delta state for heterogeneous VM fleets.
 *
 * ── Design rationale ───────────────────────────────────────────────────────
 *
 * With VM_MIPS_VALUES = {500, 1000, 1500, 2000}, two VMs can share the same
 * load delta (both at minimum finish time) yet differ by 4× in processing
 * speed.  Assigning a 3 000 MI task to VM 0 (500 MIPS) takes 6.0 s; to VM 3
 * (2000 MIPS) it takes 1.5 s.  A state that only encodes load delta cannot
 * distinguish these — the agent would treat them identically and sometimes
 * route large tasks to the slow VM, exactly the mistake that causes makespan
 * to grow.
 *
 * The fix is to add a static SPEED TIER character per VM alongside the
 * dynamic LOAD DELTA character.  The agent can then read the state and see
 * "ZF" (zero delta AND fastest) vs "ZS" (zero delta BUT slowest) and learn
 * to strongly prefer ZF VMs for large tasks.  This is the same information
 * that drives Max-Min's EFT rule, but the RL agent learns it from reward
 * signals across thousands of episodes rather than from a hard-coded formula.
 *
 * ── State encoding (2 chars per VM + task suffix) ──────────────────────────
 *
 *   Char 1 — LOAD DELTA (dynamic, updated every assignment step):
 *     'Z'  delta from min finish time < DELTA_ZERO  → EFT candidate
 *     'S'  delta from min finish time < DELTA_SMALL → slightly behind
 *     'L'  delta from min finish time ≥ DELTA_SMALL → heavily loaded
 *
 *   Char 2 — SPEED TIER (static, derived from VM MIPS at init time):
 *     'F'  MIPS > 120% of fleet average → fast
 *     'M'  MIPS within 80–120% of average → medium
 *     'S'  MIPS < 80% of fleet average → slow
 *
 *   Suffix — TASK SIZE:
 *     'T0' cloudletLength ≤ 1 000 MI  (short)
 *     'T1' cloudletLength ≤ 2 000 MI  (medium)
 *     'T2' cloudletLength >  2 000 MI  (long)
 *
 * With VM_MIPS_VALUES = {500, 1000, 1500, 2000}, average = 1 250 MIPS:
 *   VM 0 (500)  → S     VM 1 (1000) → M
 *   VM 2 (1500) → M     VM 3 (2000) → F
 *
 * Example state for finish times [0.0, 2.0, 1.0, 0.0], LONG task:
 *   deltas = [0.0, 2.0, 1.0, 0.0] → [Z, L, S, Z]
 *   speed  = [S,   M,   M,  F]
 *   key    = "ZSLMSMZFT2"
 *
 * ── State space ────────────────────────────────────────────────────────────
 *   (3 load × 3 speed)^4 VMs × 3 task sizes = 9^4 × 3 = 19 683 states.
 *   Fully tabular; saturates in approximately 2 000–3 000 training episodes
 *   at 40 cloudlets per episode.
 *
 * ── Reward function ────────────────────────────────────────────────────────
 *   − W_MAKESPAN  × Δmakespan          penalise critical-path growth
 *   − W_IMBALANCE × (maxFT − minFT)    penalise degree of imbalance
 *   + W_EFT_BONUS   if action == EFT VM  reward optimal placement
 *   + W_SPEED_BONUS × sizeTier          reward routing to fastest VM,
 *                                        scaled by task size (1/2/3)
 */
public class RLStrategy implements AssignmentStrategy {

    // ── Hyperparameters ───────────────────────────────────────────────────────

    private double alpha   = 0.3;
    private double gamma   = 0.90;
    private double epsilon = 1.0;

    private static final double EPSILON_MIN   = 0.01;
    /**
     * Per-episode decay.
     * 0.9992^5000 ≈ 0.018 — stays above EPSILON_MIN for ~4 800 episodes,
     * giving the 19 683-state table ample time to warm before the policy
     * becomes fully greedy.
     */
    private static final double EPSILON_DECAY = 0.9992;

    // ── Reward weights ────────────────────────────────────────────────────────

    private static final double W_MAKESPAN    = 10.0;
    private static final double W_IMBALANCE   =  2.0;
    private static final double W_EFT_BONUS   =  6.0;
    private static final double W_SPEED_BONUS =  2.0;

    // ── State thresholds ──────────────────────────────────────────────────────

    /** VMs within this many seconds of the minimum finish time are EFT candidates. */
    private static final double DELTA_ZERO  = 0.5;

    /** VMs within this many seconds are "slightly behind" the leader. */
    private static final double DELTA_SMALL = 2.5;

    /** MIPS below 80% of fleet average → slow tier. */
    private static final double SPEED_SLOW  = 0.80;

    /** MIPS above 120% of fleet average → fast tier. */
    private static final double SPEED_FAST  = 1.20;

    // ── Core data structures ──────────────────────────────────────────────────

    private final Map<String, double[]> qTable = new HashMap<>();
    private final Random random = new Random(42);

    /**
     * Pre-computed speed tier character per VM index.
     * Derived once from actual Vm.getMips() values on the first assign() call.
     * These are static for the lifetime of a simulation (MIPS never changes).
     */
    private char[] vmSpeedTier = null;

    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public void assign(List<Cloudlet> cloudlets, List<Vm> vms) {
        double[] vmFinishTimes = new double[vms.size()];

        // Compute speed tiers on first call (or if VM count changes)
        if (vmSpeedTier == null || vmSpeedTier.length != vms.size()) {
            vmSpeedTier = computeSpeedTiers(vms);
        }

        // Descending-length sort — same outer loop as Max-Min
        List<Cloudlet> sorted = new ArrayList<>(cloudlets);
        sorted.sort((a, b) -> Long.compare(b.getCloudletLength(), a.getCloudletLength()));

        String prevState  = null;
        int    prevAction = -1;
        double prevReward = 0.0;

        for (Cloudlet cl : sorted) {

            String state  = buildState(vmFinishTimes, cl, vms);
            int    action = chooseAction(state, vms.size());

            // Pre-action metrics for reward computation
            int    eftVm      = computeEftVm(vmFinishTimes, cl, vms);
            int    fastestVm  = computeFastestVm(vms);
            double execTime   = (double) cl.getCloudletLength() / vms.get(action).getMips();
            double oldMakespan = max(vmFinishTimes);

            // Apply assignment
            cl.setVmId(vms.get(action).getId());
            vmFinishTimes[action] += execTime;

            double newMakespan = max(vmFinishTimes);
            double minFinish   = min(vmFinishTimes);

            // ── Reward ────────────────────────────────────────────────────────
            double reward = 0.0;

            // Penalise makespan growth (primary objective)
            reward -= (newMakespan - oldMakespan) * W_MAKESPAN;

            // Penalise load imbalance (secondary objective — DI metric)
            reward -= (newMakespan - minFinish) * W_IMBALANCE;

            // EFT bonus: agent chose the VM that Max-Min's rule would pick
            if (action == eftVm) {
                reward += W_EFT_BONUS;
            }

            // Speed bonus: agent routed to the fastest VM in the fleet.
            // Scaled by task size so large tasks earn the biggest bonus for
            // landing on the fast VM — this is the key signal that teaches the
            // agent to prefer VM 3 (2000 MIPS) for LONG tasks even when
            // another VM has the same load delta.
            if (action == fastestVm) {
                int sizeTier = cl.getCloudletLength() > 2_000 ? 3
                        : cl.getCloudletLength() > 1_000 ? 2 : 1;
                reward += W_SPEED_BONUS * sizeTier;
            }

            // Deferred Q-update: update previous (state, action) with its reward
            // using the current state as the next-state. This is the correct
            // one-step-delayed update that avoids the off-by-one credit error.
            if (prevState != null) {
                updateQ(prevState, prevAction, prevReward, state, vms.size());
            }
            prevState  = state;
            prevAction = action;
            prevReward = reward;
        }

        // Terminal update: last cloudlet's (state, action, reward) with
        // maxNextQ = 0 since there is no successor state in the episode.
        if (prevState != null) {
            updateQ(prevState, prevAction, prevReward, null, vms.size());
        }

        // Decay epsilon once per episode (not per cloudlet)
        epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Speed tier computation
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Derives a static speed-tier character for each VM from its MIPS relative
     * to the fleet average.  Called once on the first assign() invocation.
     *
     * With VM_MIPS_VALUES = {500, 1000, 1500, 2000}, average = 1 250 MIPS:
     *   VM 0 (500 /1250 = 0.40) → 'S'   VM 1 (1000/1250 = 0.80) → 'M'
     *   VM 2 (1500/1250 = 1.20) → 'M'   VM 3 (2000/1250 = 1.60) → 'F'
     */
    private static char[] computeSpeedTiers(List<Vm> vms) {
        double avg = 0.0;
        for (Vm v : vms) avg += v.getMips();
        avg /= vms.size();

        char[] tiers = new char[vms.size()];
        for (int i = 0; i < vms.size(); i++) {
            double ratio = vms.get(i).getMips() / avg;
            tiers[i] = ratio > SPEED_FAST ? 'F'
                    : ratio < SPEED_SLOW ? 'S'
                      :                      'M';
        }
        return tiers;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // State construction
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Builds a two-character-per-VM state key plus a task-size suffix.
     *
     * Per VM:
     *   Char 1 (load delta): Z (EFT tier) / S (slightly behind) / L (overloaded)
     *   Char 2 (speed tier): F (fast) / M (medium) / S (slow)
     *
     * Example for 4 VMs with finish times [0.0, 3.0, 1.0, 0.0],
     * speeds [S, M, M, F], LONG task:
     *   deltas = [0.0, 3.0, 1.0, 0.0] → Z, L, S, Z
     *   key    = "ZSLMSMZF T2"
     *
     * The agent reads "ZF" (VM 3: unloaded AND fastest) and "ZS" (VM 0:
     * unloaded BUT slowest) as distinct states and learns to prefer ZF for
     * large tasks — the policy Max-Min implements by computing EFT explicitly.
     */
    private String buildState(double[] vmTimes, Cloudlet cl, List<Vm> vms) {
        double minTime = min(vmTimes);

        StringBuilder sb = new StringBuilder(vms.size() * 2 + 2);
        for (int i = 0; i < vms.size(); i++) {
            double delta = vmTimes[i] - minTime;
            sb.append(delta < DELTA_ZERO  ? 'Z'
                    : delta < DELTA_SMALL ? 'S'
                      :                       'L');
            sb.append(vmSpeedTier[i]);
        }

        long len = cl.getCloudletLength();
        sb.append('T').append(len > 2_000 ? 2 : len > 1_000 ? 1 : 0);

        return sb.toString();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Oracle helpers (reward shaping only — not used in action selection)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Returns the index of the VM with the minimum Earliest Finish Time.
     * Identical to Max-Min's greedy choice.  Used only in reward shaping —
     * not in action selection — so the agent still discovers the policy
     * through experience rather than having it hard-coded.
     */
    private int computeEftVm(double[] vmTimes, Cloudlet cl, List<Vm> vms) {
        int    best    = 0;
        double bestEft = Double.MAX_VALUE;
        for (int i = 0; i < vms.size(); i++) {
            double eft = vmTimes[i] + (double) cl.getCloudletLength() / vms.get(i).getMips();
            if (eft < bestEft) { bestEft = eft; best = i; }
        }
        return best;
    }

    /** Returns the index of the VM with the highest MIPS in the fleet. */
    private int computeFastestVm(List<Vm> vms) {
        int best = 0;
        for (int i = 1; i < vms.size(); i++) {
            if (vms.get(i).getMips() > vms.get(best).getMips()) best = i;
        }
        return best;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Q-Learning core
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * ε-greedy action selection.
     * With probability ε: explore (random VM).
     * Otherwise: exploit (VM with highest Q-value for current state).
     */
    private int chooseAction(String state, int numVMs) {
        qTable.putIfAbsent(state, new double[numVMs]);
        if (random.nextDouble() < epsilon) return random.nextInt(numVMs);
        double[] q    = qTable.get(state);
        int      best = 0;
        for (int i = 1; i < numVMs; i++) if (q[i] > q[best]) best = i;
        return best;
    }

    /**
     * Standard Bellman Q-update:
     *   Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
     *
     * When nextState is null (terminal transition), maxNextQ = 0.
     */
    private void updateQ(String s, int a, double r, String ns, int n) {
        qTable.putIfAbsent(s, new double[n]);
        double maxNext = 0.0;
        if (ns != null) {
            qTable.putIfAbsent(ns, new double[n]);
            for (double v : qTable.get(ns)) if (v > maxNext) maxNext = v;
        }
        double[] q = qTable.get(s);
        q[a] += alpha * (r + gamma * maxNext - q[a]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers & public accessors
    // ─────────────────────────────────────────────────────────────────────────

    private static double max(double[] a) {
        double m = a[0]; for (double v : a) if (v > m) m = v; return m;
    }

    private static double min(double[] a) {
        double m = a[0]; for (double v : a) if (v < m) m = v; return m;
    }

    /** Freeze (ε = 0) for final evaluation, or restore a custom value. */
    public void   setEpsilon(double e) { this.epsilon = e; }

    /** Current exploration rate — useful for training progress logs. */
    public double getEpsilon()         { return epsilon; }

    /** Number of distinct states the agent has encountered so far. */
    public int    getQTableSize()      { return qTable.size(); }
}
