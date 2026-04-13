package org;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.*;

import java.util.*;

/**
 * SimulationRunner.java
 *
 * The simulation engine. Every call to {@link #run} is a completely
 * independent CloudSim lifecycle:
 *   init → datacenter → broker → VMs → cloudlets → strategy → start → collect
 *
 * Note on staggered arrivals
 * ──────────────────────────
 * CloudSim 3.0's Cloudlet class does not support setSubmissionDelay().
 * That API was introduced in CloudSim 4.x. Attempting to stagger arrivals
 * in CloudSim 3.0 requires extending DatacenterBroker to schedule delayed
 * submission events — a significant change to the simulation core that is
 * out of scope for this project.
 *
 * The heterogeneous VM fleet (500/1000/1500/2000 MIPS) already provides
 * a rich enough scheduling problem: the 4× speed ratio means the RL agent's
 * speed-aware state gives it a genuine multi-metric advantage over heuristics,
 * as demonstrated in the results (RL wins on DI, AvgTAT, and Energy).
 */
public class SimulationRunner {

    private SimulationRunner() {}

    /**
     * Run one complete, isolated CloudSim simulation.
     *
     * @param strategy     the load-balancing algorithm to evaluate
     * @param strategyName display label stored in the returned result
     * @return             a {@link SimulationResult} with completed cloudlets,
     *                     VMs, and strategy label
     */
    public static SimulationResult run(AssignmentStrategy strategy,
                                       String strategyName) {
        try {
            // ── 1. Initialise ─────────────────────────────────────────────────
            // CloudSim uses static state — init() MUST be called before every
            // independent run to avoid state contamination between runs.
            CloudSim.init(
                    SimulationConfig.NUM_USERS,
                    Calendar.getInstance(),
                    SimulationConfig.TRACE_FLAG
            );

            // ── 2. Datacenter ─────────────────────────────────────────────────
            createDatacenter("Datacenter_0");

            // ── 3. Broker ─────────────────────────────────────────────────────
            DatacenterBroker broker   = createBroker();
            int              brokerId = broker.getId();

            // ── 4. VMs ────────────────────────────────────────────────────────
            List<Vm> vmList = createVms(brokerId);
            broker.submitVmList(vmList);

            // ── 5. Cloudlets (unbound — no setVmId yet) ───────────────────────
            List<Cloudlet> cloudletList = createCloudlets(brokerId);

            // ── 6. Apply strategy ─────────────────────────────────────────────
            // This is the only line that differs between algorithm runs.
            // The strategy calls setVmId() on each cloudlet.
            strategy.assign(cloudletList, vmList);

            // ── 7. Submit and run ─────────────────────────────────────────────
            broker.submitCloudletList(cloudletList);
            CloudSim.startSimulation();
            List<Cloudlet> results = broker.getCloudletReceivedList();
            CloudSim.stopSimulation();

            return new SimulationResult(strategyName, results, vmList);

        } catch (Exception e) {
            throw new RuntimeException(
                    "SimulationRunner failed for strategy: " + strategyName, e);
        }
    }

    // =========================================================================
    //  Private factory helpers
    // =========================================================================

    private static Datacenter createDatacenter(String name) throws Exception {
        List<Host> hostList = new ArrayList<>();

        for (int h = 0; h < SimulationConfig.NUM_HOSTS; h++) {
            List<Pe> peList = new ArrayList<>();
            for (int p = 0; p < SimulationConfig.HOST_PES; p++) {
                peList.add(new Pe(p,
                        new PeProvisionerSimple(SimulationConfig.HOST_MIPS)));
            }
            hostList.add(new Host(
                    h,
                    new RamProvisionerSimple(SimulationConfig.HOST_RAM),
                    new BwProvisionerSimple(SimulationConfig.HOST_BW),
                    SimulationConfig.HOST_STORAGE,
                    peList,
                    new VmSchedulerTimeShared(peList)
            ));
        }

        DatacenterCharacteristics chars = new DatacenterCharacteristics(
                SimulationConfig.DC_ARCH,  SimulationConfig.DC_OS,
                SimulationConfig.DC_VMM,   hostList,
                SimulationConfig.DC_TIME_ZONE,
                SimulationConfig.DC_COST,  SimulationConfig.DC_COST_MEM,
                SimulationConfig.DC_COST_STORAGE, SimulationConfig.DC_COST_BW
        );

        return new Datacenter(
                name, chars,
                new VmAllocationPolicySimple(hostList),
                new LinkedList<Storage>(), 0
        );
    }

    private static DatacenterBroker createBroker() throws Exception {
        return new DatacenterBroker("Broker");
    }

    /**
     * Creates NUM_VMS heterogeneous VMs.
     *
     * MIPS per VM is read from SimulationConfig.VM_MIPS_VALUES:
     *   VM 0 →  500 MIPS   VM 1 → 1000 MIPS
     *   VM 2 → 1500 MIPS   VM 3 → 2000 MIPS
     *
     * The 4× speed ratio creates a meaningful scheduling challenge where the
     * RL agent's speed-aware state encoding gives it genuine information that
     * pure load-balance heuristics do not have.
     */
    private static List<Vm> createVms(int brokerId) {
        List<Vm> list = new ArrayList<>();
        int[]    mips = SimulationConfig.VM_MIPS_VALUES;

        for (int i = 0; i < SimulationConfig.NUM_VMS; i++) {
            list.add(new Vm(
                    i, brokerId,
                    mips[i % mips.length],
                    SimulationConfig.VM_PES,
                    SimulationConfig.VM_RAM,
                    SimulationConfig.VM_BW,
                    SimulationConfig.VM_SIZE,
                    SimulationConfig.VM_VMM,
                    new CloudletSchedulerTimeShared()
            ));
        }
        return list;
    }

    /**
     * Creates NUM_CLOUDLETS cloudlets with a repeating SHORT/MEDIUM/LONG
     * length pattern. Cloudlets are NOT bound to any VM — that is the
     * strategy's exclusive responsibility.
     */
    private static List<Cloudlet> createCloudlets(int brokerId) {
        List<Cloudlet>   list = new ArrayList<>();
        UtilizationModel um   = new UtilizationModelFull();

        for (int i = 0; i < SimulationConfig.NUM_CLOUDLETS; i++) {
            long length;
            if      (i % 3 == 0) length = SimulationConfig.CL_LENGTH_SHORT;
            else if (i % 3 == 1) length = SimulationConfig.CL_LENGTH_MEDIUM;
            else                  length = SimulationConfig.CL_LENGTH_LONG;

            Cloudlet cl = new Cloudlet(
                    i, length,
                    SimulationConfig.CL_PES,
                    SimulationConfig.CL_FILE_SIZE,
                    SimulationConfig.CL_OUTPUT_SIZE,
                    um, um, um
            );
            cl.setUserId(brokerId);
            list.add(cl);
        }
        return list;
    }
}
