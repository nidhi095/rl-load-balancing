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

        // ── 1. Heuristics ─────────────────────────────────────────────────────
        System.out.println("[ Heuristics ]");
        runOne(new FCFSStrategy(),        "FCFS",         allResults);
        runOne(new RoundRobinStrategy(),  "Round Robin",  allResults);
        runOne(new LeastLoadedStrategy(), "Least Loaded", allResults);
        runOne(new MinMinStrategy(),      "Min-Min",      allResults);
        runOne(new MaxMinStrategy(),      "Max-Min",      allResults);

        // ── 2. RL Training ────────────────────────────────────────────────────
        RLStrategy rl = new RLStrategy();
        System.out.printf("%n[ RL Training — %d episodes ]%n", TRAINING_EPISODES);
        System.out.printf("  %-8s  %-8s  %-10s%n", "Episode", "Epsilon", "Q-States");
        System.out.println("  " + "─".repeat(30));

        for (int ep = 1; ep <= TRAINING_EPISODES; ep++) {
            SimulationRunner.run(rl, "Training");
            if (ep == 1 || ep % LOG_INTERVAL == 0) {
                System.out.printf("  %-8d  %-8.4f  %-10d%n",
                        ep, rl.getEpsilon(), rl.getQTableSize());
            }
        }

        System.out.println("  " + "─".repeat(30));
        System.out.printf("  Training complete.  Final ε=%.4f  Q-states=%d%n%n",
                rl.getEpsilon(), rl.getQTableSize());

        /*
         * Tip: if Q-States at episode 5 000 is still growing rapidly, the table
         * has not saturated — increase TRAINING_EPISODES to 8 000–10 000.
         * With the 243-state v3 table this should stop growing well before ep 1 000.
         */

        // ── 3. Final evaluation (greedy policy) ───────────────────────────────
        rl.setEpsilon(0.0);
        System.out.println("[ RL Final Evaluation ]");
        runOne(rl, "RL (Trained)", allResults);

        // ── 4. Comparison table ───────────────────────────────────────────────
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
