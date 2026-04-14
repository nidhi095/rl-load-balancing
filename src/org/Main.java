package org;

import java.util.ArrayList;
import java.util.List;

public class Main {

    private static final int TRAINING_EPISODES = 5_000;
    private static final int LOG_INTERVAL      = 500;

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════╗");
        System.out.println("║    CloudSim Load Balancing – Strategy Comparison     ║");
        System.out.println("╚══════════════════════════════════════════════════════╝");
        System.out.printf("  VMs: %d   |   Cloudlets: %d%n%n",
                SimulationConfig.NUM_VMS, SimulationConfig.NUM_CLOUDLETS);

        List<Metrics> allResults = new ArrayList<>();

        // ── 1. Classical Heuristics ───────────────────────────────────────────
        System.out.println("[ Heuristics ]");
        runOne(new FCFSStrategy(),        "FCFS",         allResults);
        runOne(new RoundRobinStrategy(),  "Round Robin",  allResults);
        runOne(new LeastLoadedStrategy(), "Least Loaded", allResults);
        runOne(new MinMinStrategy(),      "Min-Min",      allResults);
        runOne(new MaxMinStrategy(),      "Max-Min",      allResults);

        // ── 2. SARSA(λ) Training ──────────────────────────────────────────────
        RLStrategy rl = new RLStrategy();

        System.out.printf("%n[ SARSA(λ) Training — %d episodes ]%n", TRAINING_EPISODES);
        System.out.printf("  %-8s  %-8s  %-10s  %-10s%n",
                "Episode", "Epsilon", "Q-States", "Traces");
        System.out.println("  " + "─".repeat(42));

        for (int ep = 1; ep <= TRAINING_EPISODES; ep++) {
            SimulationRunner.run(rl, "Training");

            if (ep == 1 || ep % LOG_INTERVAL == 0) {
                System.out.printf("  %-8d  %-8.4f  %-10d  %-10d%n",
                        ep,
                        rl.getEpsilon(),
                        rl.getQTableSize(),
                        rl.getTraceSize());
            }
        }

        System.out.println("  " + "─".repeat(42));
        System.out.printf("  Training complete.  ε=%.4f  Q-states=%d%n%n",
                rl.getEpsilon(), rl.getQTableSize());

        /*
         * What to look for in the training log:
         *
         *   Q-States should grow quickly (SARSA(λ) visits states faster than
         *   TD(0) because credit propagates backward, so early episodes explore
         *   more of the state space per episode).  It should plateau before
         *   episode 3 000 for the 19 683-state table.
         *
         *   Traces column shows active trace entries at the END of the last
         *   training episode.  After setEpsilon(0) this will be 0 since we
         *   call getTraceSize() after the episode completes and eTrace is
         *   cleared at the start of each episode.
         *
         *   If Q-States is still growing at episode 5 000, increase
         *   TRAINING_EPISODES to 8 000.
         */

        // ── 3. Final Evaluation (greedy policy, no exploration) ───────────────
        rl.setEpsilon(0.0);
        System.out.println("[ SARSA(λ) Final Evaluation ]");
        runOne(rl, "RL SARSA(λ)", allResults);

        // ── 4. Comparison Table ───────────────────────────────────────────────
        ResultPrinter.printComparisonTable(allResults);
    }

    private static void runOne(AssignmentStrategy strategy,
                               String name, List<Metrics> results) {
        ResultPrinter.printSectionHeader(name);
        try {
            SimulationResult result = SimulationRunner.run(strategy, name);
            Metrics metrics = MetricsCalculator.compute(result);
            ResultPrinter.printMetrics(metrics);
            results.add(metrics);
        } catch (Exception e) {
            System.err.println("  [ERROR] Strategy failed: " + name);
            e.printStackTrace();
        }
    }
}
