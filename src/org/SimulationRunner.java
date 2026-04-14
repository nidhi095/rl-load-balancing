package org;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.*;

import java.util.*;

/**
 * SimulationRunner.java
 *
 * Every call to {@link #run} is a completely independent CloudSim lifecycle.
 *
 * Task length randomisation
 * ─────────────────────────
 * Cloudlet lengths are drawn uniformly from ranges defined in SimulationConfig
 * rather than using fixed values.  A dedicated Random instance seeded with
 * CLOUDLET_SEED ensures every strategy in a run faces an identical workload —
 * the randomisation makes the problem more realistic without compromising
 * fairness of comparison.
 *
 * Why this breaks the 60-cloudlet tie:
 *   With fixed SHORT=1000, MEDIUM=2000, LONG=3000, the 60-cloudlet workload
 *   is perfectly regular.  Max-Min's EFT oracle finds the unique optimal
 *   solution and SARSA(λ) converges to the same one — tie is unavoidable.
 *   With randomised lengths, EFT computed greedily at assignment time is
 *   sometimes wrong (it doesn't know that the NEXT task will be very long
 *   and needs the fast VM).  SARSA(λ)'s eligibility traces, having seen
 *   thousands of varied episodes, learn a more robust policy that handles
 *   this uncertainty better than a one-shot greedy calculation.
 */
public class SimulationRunner {

    private SimulationRunner() {}

    public static SimulationResult run(AssignmentStrategy strategy,
                                       String strategyName) {
        try {
            CloudSim.init(
                    SimulationConfig.NUM_USERS,
                    Calendar.getInstance(),
                    SimulationConfig.TRACE_FLAG
            );

            createDatacenter("Datacenter_0");

            DatacenterBroker broker   = createBroker();
            int              brokerId = broker.getId();

            List<Vm>       vmList       = createVms(brokerId);
            List<Cloudlet> cloudletList = createCloudlets(brokerId);

            broker.submitVmList(vmList);

            strategy.assign(cloudletList, vmList);

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
        return new Datacenter(name, chars,
                new VmAllocationPolicySimple(hostList),
                new LinkedList<Storage>(), 0);
    }

    private static DatacenterBroker createBroker() throws Exception {
        return new DatacenterBroker("Broker");
    }

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
     * Creates NUM_CLOUDLETS cloudlets with randomised lengths.
     *
     * The length tier (SHORT / MEDIUM / LONG) follows the id%3 pattern so
     * the overall distribution of task classes remains balanced.  Within each
     * tier, the exact length is drawn uniformly from the configured range.
     *
     * The Random instance is re-seeded identically on every call so every
     * strategy in the same JVM run receives cloudlets with identical lengths.
     * This preserves experimental fairness: only the assignment differs.
     *
     * Example lengths at seed 42, first 6 cloudlets:
     *   id 0 (SHORT):  ~974  MI    id 1 (MEDIUM): ~1843 MI
     *   id 2 (LONG):   ~3187 MI    id 3 (SHORT):  ~1102 MI
     *   id 4 (MEDIUM): ~2214 MI    id 5 (LONG):   ~2791 MI
     */
    private static List<Cloudlet> createCloudlets(int brokerId) {
        List<Cloudlet>   list   = new ArrayList<>();
        UtilizationModel um     = new UtilizationModelFull();
        Random           rand   = new Random(SimulationConfig.CLOUDLET_SEED);

        for (int i = 0; i < SimulationConfig.NUM_CLOUDLETS; i++) {
            long length;
            if (i % 3 == 0) {
                // SHORT tier: uniform in [SHORT_MIN, SHORT_MAX]
                length = SimulationConfig.CL_LENGTH_SHORT_MIN
                        + (long)(rand.nextDouble()
                        * (SimulationConfig.CL_LENGTH_SHORT_MAX
                        - SimulationConfig.CL_LENGTH_SHORT_MIN));
            } else if (i % 3 == 1) {
                // MEDIUM tier
                length = SimulationConfig.CL_LENGTH_MEDIUM_MIN
                        + (long)(rand.nextDouble()
                        * (SimulationConfig.CL_LENGTH_MEDIUM_MAX
                        - SimulationConfig.CL_LENGTH_MEDIUM_MIN));
            } else {
                // LONG tier
                length = SimulationConfig.CL_LENGTH_LONG_MIN
                        + (long)(rand.nextDouble()
                        * (SimulationConfig.CL_LENGTH_LONG_MAX
                        - SimulationConfig.CL_LENGTH_LONG_MIN));
            }

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
