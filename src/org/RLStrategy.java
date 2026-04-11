package org;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.Vm;
import java.util.*;

/**
 * RLStrategy v3 — Delta-based state representation.
 *
 * Root cause of v2's remaining gap vs Max-Min (9.10 s vs 8.21 s):
 *
 *   In a homogeneous environment (all VMs at the same MIPS), the efficiency
 *   bucket (F/M/S) added in v2 is always "M" for every VM on every task.
 *   That dimension carries zero information.  Worse, the U/B/O load buckets
 *   (±15 % of average finish time) are too coarse to tell the agent which VM
 *   actually has the minimum finish time — multiple VMs share the "U" label
 *   while differing by more than one task-length in absolute finish time.
 *
 * The fix — Delta-state:
 *   For each VM compute  delta_i = vmTimes[i] − min(vmTimes).
 *   Bucket into Z / S / L:
 *     Z (zero)  : delta < 0.5 s  — this IS (or ties for) the EFT VM
 *     S (small) : delta < 2.5 s  — one task-length behind the leader
 *     L (large) : delta ≥ 2.5 s  — significantly behind
 *
 *   State space: 3^N × 3 task-size bins.
 *   For N=4 VMs: 3^4 × 3 = 243 states — saturates in < 500 episodes.
 *   The agent can now directly read which VMs are EFT (labeled Z) and learn
 *   "always pick a Z VM" — reproducing Max-Min's greedy rule, and with good
 *   luck slightly improving on it.
 */
public class RLStrategy implements AssignmentStrategy {

    // ── Hyperparameters ───────────────────────────────────────────────────────

    private double alpha   = 0.3;
    private double gamma   = 0.90;
    private double epsilon = 1.0;

    private static final double EPSILON_MIN   = 0.01;
    /** Per-episode decay: 0.9994^5000 ≈ 0.05 — stays warm for ~4 700 episodes. */
    private static final double EPSILON_DECAY = 0.9994;

    // ── Reward weights ────────────────────────────────────────────────────────

    private static final double W_MAKESPAN  = 10.0;
    /** Raised from 0.5 → 2.0 to directly target the DI = 1.59 problem. */
    private static final double W_IMBALANCE = 2.0;
    /**
     * Raised from 3.0 → 8.0 so the EFT bonus reliably dominates noise.
     * Even a 0.5 s makespan delta costs −5.0 in penalty, so the +8.0 bonus
     * for picking EFT always wins in the reward signal.
     */
    private static final double W_EFT_BONUS = 8.0;

    // ── Delta-state thresholds ────────────────────────────────────────────────

    /** Finish times within this band are treated as tied for EFT. */
    private static final double DELTA_ZERO  = 0.5;
    /** Up to 2.5 s behind the leader = "small" gap (≈ one medium task). */
    private static final double DELTA_SMALL = 2.5;

    // ── Core data structures ──────────────────────────────────────────────────

    private final Map<String, double[]> qTable = new HashMap<>();
    private final Random random = new Random(42);

    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public void assign(List<Cloudlet> cloudlets, List<Vm> vms) {
        double[] vmFinishTimes = new double[vms.size()];

        // Descending sort — same outer loop as Max-Min
        List<Cloudlet> sorted = new ArrayList<>(cloudlets);
        sorted.sort((a, b) -> Long.compare(b.getCloudletLength(), a.getCloudletLength()));

        String prevState  = null;
        int    prevAction = -1;
        double prevReward = 0.0;

        for (Cloudlet cl : sorted) {

            String state  = buildState(vmFinishTimes, cl, vms);
            int    action = chooseAction(state, vms.size());

            int    eftVm      = computeEftVm(vmFinishTimes, cl, vms);
            double execTime   = (double) cl.getCloudletLength() / vms.get(action).getMips();
            double oldMakespan = max(vmFinishTimes);

            cl.setVmId(vms.get(action).getId());
            vmFinishTimes[action] += execTime;

            double newMakespan = max(vmFinishTimes);
            double minFinish   = min(vmFinishTimes);

            double reward = 0.0;
            reward -= (newMakespan - oldMakespan) * W_MAKESPAN;
            reward -= (newMakespan - minFinish)   * W_IMBALANCE;
            if (action == eftVm) reward += W_EFT_BONUS;

            if (prevState != null) {
                updateQ(prevState, prevAction, prevReward, state, vms.size());
            }
            prevState  = state;
            prevAction = action;
            prevReward = reward;
        }

        // Terminal transition: maxNextQ = 0
        if (prevState != null) {
            updateQ(prevState, prevAction, prevReward, null, vms.size());
        }

        // Decay once per episode, not per cloudlet
        epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Delta-state (v3)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * For each VM, bucket its delta from the minimum finish time:
     *   'Z' — at (or tied for) the EFT position
     *   'S' — one task-length behind
     *   'L' — significantly behind
     * Plus a task-size suffix T0/T1/T2.
     *
     * Example: finish times [3.0, 3.0, 3.4, 5.0], task = MEDIUM
     *   deltas = [0.0, 0.0, 0.4, 2.0]  →  state = "ZZSS T1"
     *
     * 3^4 × 3 = 243 states for 4 VMs.
     */
    private String buildState(double[] vmTimes, Cloudlet cl, List<Vm> vms) {
        double minTime = min(vmTimes);

        StringBuilder sb = new StringBuilder(vms.size() + 2);
        for (int i = 0; i < vms.size(); i++) {
            double delta = vmTimes[i] - minTime;
            if      (delta < DELTA_ZERO)  sb.append('Z');
            else if (delta < DELTA_SMALL) sb.append('S');
            else                          sb.append('L');
        }

        long len  = cl.getCloudletLength();
        int  size = len > 2_000 ? 2 : len > 1_000 ? 1 : 0;
        sb.append('T').append(size);

        return sb.toString();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EFT oracle (reward shaping only — not used in action selection)
    // ─────────────────────────────────────────────────────────────────────────

    private int computeEftVm(double[] vmTimes, Cloudlet cl, List<Vm> vms) {
        int    best    = 0;
        double bestEft = Double.MAX_VALUE;
        for (int i = 0; i < vms.size(); i++) {
            double eft = vmTimes[i] + (double) cl.getCloudletLength() / vms.get(i).getMips();
            if (eft < bestEft) { bestEft = eft; best = i; }
        }
        return best;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Q-Learning core
    // ─────────────────────────────────────────────────────────────────────────

    private int chooseAction(String state, int numVMs) {
        qTable.putIfAbsent(state, new double[numVMs]);
        if (random.nextDouble() < epsilon) return random.nextInt(numVMs);
        double[] q    = qTable.get(state);
        int      best = 0;
        for (int i = 1; i < numVMs; i++) if (q[i] > q[best]) best = i;
        return best;
    }

    /** Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) − Q(s,a)].  nextState=null → terminal. */
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

    private static double max(double[] a) { double m=a[0]; for(double v:a) if(v>m)m=v; return m; }
    private static double min(double[] a) { double m=a[0]; for(double v:a) if(v<m)m=v; return m; }

    public void   setEpsilon(double e) { this.epsilon = e; }
    public double getEpsilon()         { return epsilon; }
    public int    getQTableSize()      { return qTable.size(); }
}
