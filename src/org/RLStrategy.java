package org;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.Vm;
import java.util.*;

/**
 * RLStrategy — SARSA(λ) with speed-aware delta state.
 *
 * Changes from previous version:
 *   λ raised from 0.7 → 0.85
 *
 * Why:
 *   With λ=0.7 and γ=0.9, the trace decay per step = γλ = 0.63.
 *   After 10 steps: (0.63)^10 ≈ 0.006 — credit from step 1 is negligible
 *   by step 11.  At 80 cloudlets (80-step episodes) this meant the first
 *   ~30 assignment decisions received almost no credit correction from
 *   later rewards, causing the DI regression observed in results.
 *
 *   With λ=0.85 and γ=0.9, γλ = 0.765.
 *   After 10 steps: (0.765)^10 ≈ 0.065 — still meaningful.
 *   After 20 steps: (0.765)^20 ≈ 0.004 — fades out around step 20.
 *   This keeps credit propagation alive for the first ~20 decisions of
 *   every episode, which is sufficient for 40–80 cloudlet workloads.
 *
 * State space and reward function are identical to the previous version.
 * Only λ changes.
 */
public class RLStrategy implements AssignmentStrategy {

    // ── Hyperparameters ───────────────────────────────────────────────────────

    private double alpha   = 0.3;
    private double gamma   = 0.90;

    /**
     * Trace decay factor.
     * Raised from 0.7 → 0.85 to maintain credit propagation over the
     * longer episode lengths produced by 60–80 cloudlet workloads.
     *
     * λ=0   → reduces to TD(0), same as Q-Learning (no backward propagation)
     * λ=0.85→ credit meaningful for ~20 steps back (good for 40–80 tasks)
     * λ=1   → Monte Carlo, full episode credit (unstable in practice)
     */
    private double lambda  = 0.85;
    private double epsilon = 1.0;

    private static final double EPSILON_MIN   = 0.01;
    private static final double EPSILON_DECAY = 0.9992;

    // ── Reward weights ────────────────────────────────────────────────────────

    private static final double W_MAKESPAN    = 10.0;
    private static final double W_IMBALANCE   =  2.0;
    private static final double W_EFT_BONUS   =  6.0;
    private static final double W_SPEED_BONUS =  2.0;

    // ── State thresholds ──────────────────────────────────────────────────────

    private static final double DELTA_ZERO  = 0.5;
    private static final double DELTA_SMALL = 2.5;
    private static final double SPEED_SLOW  = 0.80;
    private static final double SPEED_FAST  = 1.20;

    // ── Core data structures ──────────────────────────────────────────────────

    private final Map<String, double[]> qTable = new HashMap<>();
    private final Map<String, double[]> eTrace = new HashMap<>();
    private final Random random = new Random(42);
    private char[] vmSpeedTier = null;

    // ─────────────────────────────────────────────────────────────────────────

    @Override
    public void assign(List<Cloudlet> cloudlets, List<Vm> vms) {
        double[] vmFinishTimes = new double[vms.size()];

        if (vmSpeedTier == null || vmSpeedTier.length != vms.size()) {
            vmSpeedTier = computeSpeedTiers(vms);
        }

        // Reset traces at episode start
        eTrace.clear();

        List<Cloudlet> sorted = new ArrayList<>(cloudlets);
        sorted.sort((a, b) -> Long.compare(b.getCloudletLength(), a.getCloudletLength()));

        // Choose first action before loop (SARSA requires s,a pair at step start)
        String currentState  = buildState(vmFinishTimes, sorted.get(0), vms);
        int    currentAction = chooseAction(currentState, vms.size());

        for (int i = 0; i < sorted.size(); i++) {
            Cloudlet cl = sorted.get(i);

            // Pre-action metrics
            int    eftVm      = computeEftVm(vmFinishTimes, cl, vms);
            int    fastestVm  = computeFastestVm(vms);
            double execTime   = (double) cl.getCloudletLength()
                    / vms.get(currentAction).getMips();
            double oldMakespan = max(vmFinishTimes);

            // Apply assignment
            cl.setVmId(vms.get(currentAction).getId());
            vmFinishTimes[currentAction] += execTime;

            double newMakespan = max(vmFinishTimes);
            double minFinish   = min(vmFinishTimes);

            // Reward
            double reward = 0.0;
            reward -= (newMakespan - oldMakespan) * W_MAKESPAN;
            reward -= (newMakespan - minFinish)   * W_IMBALANCE;
            if (currentAction == eftVm) reward += W_EFT_BONUS;
            if (currentAction == fastestVm) {
                int sizeTier = cl.getCloudletLength() > 2_000 ? 3
                        : cl.getCloudletLength() > 1_000 ? 2 : 1;
                reward += W_SPEED_BONUS * sizeTier;
            }

            // Observe next state and choose next action ON-POLICY
            boolean isTerminal = (i == sorted.size() - 1);
            String  nextState  = null;
            int     nextAction = 0;

            if (!isTerminal) {
                nextState  = buildState(vmFinishTimes, sorted.get(i + 1), vms);
                nextAction = chooseAction(nextState, vms.size());
            }

            // TD error: δ = r + γ·Q(s',a') − Q(s,a)
            qTable.putIfAbsent(currentState, new double[vms.size()]);
            double qCurrent = qTable.get(currentState)[currentAction];
            double qNext    = 0.0;
            if (!isTerminal) {
                qTable.putIfAbsent(nextState, new double[vms.size()]);
                qNext = qTable.get(nextState)[nextAction];
            }
            double tdError = reward + gamma * qNext - qCurrent;

            // Increment eligibility trace for current (state, action)
            eTrace.putIfAbsent(currentState, new double[vms.size()]);
            eTrace.get(currentState)[currentAction] += 1.0;

            // Update ALL visited (state, action) pairs using traces
            // Q(s,a)  += α · δ · e(s,a)
            // e(s,a)  *= γ · λ
            for (Map.Entry<String, double[]> entry : eTrace.entrySet()) {
                String   s      = entry.getKey();
                double[] traces = entry.getValue();
                qTable.putIfAbsent(s, new double[vms.size()]);
                double[] qVals  = qTable.get(s);
                for (int a = 0; a < vms.size(); a++) {
                    if (traces[a] > 1e-6) {
                        qVals[a]  += alpha * tdError * traces[a];
                        traces[a] *= gamma * lambda;
                    }
                }
            }

            currentState  = nextState;
            currentAction = nextAction;
        }

        epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);
    }

    // ── Speed tiers ───────────────────────────────────────────────────────────

    private static char[] computeSpeedTiers(List<Vm> vms) {
        double avg = 0.0;
        for (Vm v : vms) avg += v.getMips();
        avg /= vms.size();
        char[] t = new char[vms.size()];
        for (int i = 0; i < vms.size(); i++) {
            double r = vms.get(i).getMips() / avg;
            t[i] = r > SPEED_FAST ? 'F' : r < SPEED_SLOW ? 'S' : 'M';
        }
        return t;
    }

    // ── State construction ────────────────────────────────────────────────────

    /**
     * Two characters per VM (load delta + speed tier) plus task-size suffix.
     * State space: 9^4 × 3 = 19 683 states.
     *
     * Task size buckets use the MID-POINTS of the randomised ranges:
     *   T0: length < 1450 MI   (short tier centre ≈ 1000)
     *   T1: length < 2450 MI   (medium tier centre ≈ 2000)
     *   T2: length ≥ 2450 MI   (long tier centre ≈ 3000)
     *
     * Using range midpoints as bucket boundaries means a task drawn from
     * the SHORT range will almost always get T0, MEDIUM → T1, LONG → T2,
     * preserving the intended semantic even with randomised lengths.
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
        sb.append('T').append(len < 1_450 ? 0 : len < 2_450 ? 1 : 2);
        return sb.toString();
    }

    // ── Oracle helpers ────────────────────────────────────────────────────────

    private int computeEftVm(double[] vmTimes, Cloudlet cl, List<Vm> vms) {
        int best = 0; double bestEft = Double.MAX_VALUE;
        for (int i = 0; i < vms.size(); i++) {
            double eft = vmTimes[i] + (double) cl.getCloudletLength() / vms.get(i).getMips();
            if (eft < bestEft) { bestEft = eft; best = i; }
        }
        return best;
    }

    private int computeFastestVm(List<Vm> vms) {
        int best = 0;
        for (int i = 1; i < vms.size(); i++)
            if (vms.get(i).getMips() > vms.get(best).getMips()) best = i;
        return best;
    }

    // ── Q-Learning core ───────────────────────────────────────────────────────

    private int chooseAction(String state, int numVMs) {
        qTable.putIfAbsent(state, new double[numVMs]);
        if (random.nextDouble() < epsilon) return random.nextInt(numVMs);
        double[] q = qTable.get(state);
        int best = 0;
        for (int i = 1; i < numVMs; i++) if (q[i] > q[best]) best = i;
        return best;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static double max(double[] a) {
        double m = a[0]; for (double v : a) if (v > m) m = v; return m;
    }
    private static double min(double[] a) {
        double m = a[0]; for (double v : a) if (v < m) m = v; return m;
    }

    public void   setEpsilon(double e) { this.epsilon = e; }
    public double getEpsilon()         { return epsilon; }
    public int    getQTableSize()      { return qTable.size(); }
    public int    getTraceSize()       { return eTrace.size(); }
}
