<h1>RL Load Balancing</h1>

<img width="1440" height="1312" alt="image" src="https://github.com/user-attachments/assets/4dd3d3b7-d06e-40d0-a69d-0bddf2f57175" />

<p>Description of each file:</p>
<ul>
  <li>SimulationConfig.java: Every constant in one place — change VM count, cloudlet lengths, energy model, etc. here only</li>
  <li>AssignmentStrategy.java: The interface. The only contract all heuristics must satisfy</li>
  <li>SimulationResult.java: Plain data object — completed cloudlets + VM list + label</li>
  <li>SimulationRunner.java: The engine. Direct refactor of your original file. Calls strategy.assign() as the only variable line</li>
  <li>FCFSStrategy.java: Arrival-order cycling</li>
  <li>RoundRobinStrategy.java: Explicit cyclic (distinguishable from FCFS when VM list is dynamic)</li>
  <li>LeastLoadedStrategy.java: Online greedy — no sorting, reacts to arrival order</li>
  <li>MinMinStrategy.java: Offline — sort shortest first, assign to min-load VM</li>
  <li>MaxMinStrategy.java: Offline — sort longest first, assign to min-load VM</li>
  <li>RLStrategy.java: Stub with full extension guide — falls back to least-loaded until you wire the model</li>
  <li>Metrics.java: Immutable value object for all 8 metrics</li>
  <li>MetricsCalculator.java: Pure computation, no printing, no CloudSim dependencies</li>
  <li>ResultPrinter.java: All System.out calls live here and nowhere else</li>  
  <li>Main.java: One runOne() call per strategy — the only file you touch to add a new algorithm</li>
</ul>
