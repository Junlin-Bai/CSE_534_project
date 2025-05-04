import argparse
import os
import operator
import sys
import random
import json
from functools import reduce
import numpy as np
import netsquid as ns
from netsquid.util.simtools import sim_time
from netsquid.protocols.nodeprotocols import LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.qubits import qubitapi as qapi
from netsquid.qubits.operators import Z, X, Operator
from netsquid.util.datacollector import DataCollector
import netsquid.qubits.operators as ops
import netsquid.qubits.ketstates as ks

# Adjust the system path to include the parent directory for project-specific modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project-specific modules for network setup, logging, and protocols.
from utils.NetworkSetup import setup_network_parallel_chsh
from utils import Logging
from protocols.GenEntanglementConcurrent import GenEntanglementConcurrent
from protocols.MessageHandler import MessageHandler, MessageType
from protocols.EntanglementHandlerConcurrent import EntanglementHandlerConcurrent
from protocols.CHSH import  CHSHProtocol
from protocols.GHZ import GHZProtocol
from utils.ClassicalMessages import ClassicalMessage
import pydynaa as pd


class ExampleCHSH(LocalProtocol):
    """
    Protocol for generating entangled qubit pairs between two nodes (Alice and Bob)
    and measuring them using four predetermined CHSH settings:
    A0B0, A1B0, A0B1, and A1B1.

    The protocol assumes that the total number of entangled pairs generated is divisible by 4
    (or trims extra pairs) so that each measurement setting combination is used equally.
    """

    def __init__(
            self,
            network_nodes,
            num_runs,
            max_entangle_pairs,  # Use global N_4 variable.,
            node_distance,
            memory_depolar_rate,
            sample_size_rate,
            alpah,
            delta=0.1,
            skip_noise=False
    ):
        # Check that exactly 2 nodes (Alice and Bob) are provided.
        if len(network_nodes) != 2:
            raise ValueError("This protocol is designed for exactly 2 nodes (Alice and Bob) to do CHSH.")

        self.all_nodes = network_nodes
        self.num_runs = num_runs
        self.max_entangle_pairs = max_entangle_pairs
        self.skip_noise = skip_noise
        # Create a logger for recording protocol events.
        self.logger = Logging.Logger("CHSH_ExampleEntanglement", save_to_file=True, file_name="chsh.log",
                                     logging_enabled=False)
        self.null_logger = Logging.Logger("CHSH_ExampleEntanglement_null", logging_enabled=False)

        # Initialize the base protocol with a dictionary mapping node names to node objects.
        super().__init__(nodes={node.name: node for node in network_nodes}, name="CHSH_ExampleEntanglement")
        # Add a message handler subprotocol for classical communications.
        # Classical message handlers for Alice and Bob
        self.add_subprotocol(MessageHandler(
            node=self.all_nodes[0],
            name=f"message_handler_{self.all_nodes[0].name}",
            cc_ports=self.get_cc_ports(self.all_nodes[0])
        ))
        # --- GHZ entanglement setup ---
        # Add a message handler subprotocol for classical communications.
        genA_ghz = GenEntanglementConcurrent(
            total_pairs=self.max_entangle_pairs,
            entangle_node=self.all_nodes[1].name,
            node=self.all_nodes[0],
            name="genA_GHZ",
            is_source=True,
            logger=self.null_logger
        )
        self.add_subprotocol(genA_ghz)
        ehA_ghz = EntanglementHandlerConcurrent(
            node=self.all_nodes[0],
            name="ehA_GHZ",
            num_pairs=self.max_entangle_pairs,
            qubit_input_protocol=genA_ghz,
            cc_message_handler=self.subprotocols[f"message_handler_{self.all_nodes[0].name}"],
            entangle_node=self.all_nodes[1].name,
            memory_depolar_rate=memory_depolar_rate,
            node_distance=node_distance,
            is_top_layer=False,
            logger=self.null_logger
        )
        self.add_subprotocol(ehA_ghz)
        """
            entangle_node,
                     qubit_ready_protocols,
                     setting_list,
                     total_pairs,
                     sample_size_rate,
                     alpha,
                     cc_message_handler,
                     logger,
                     delta=0.1
            """
        # --- Setup  ---
        # generate settings for a and b
        N_each = int(self.max_entangle_pairs * sample_size_rate // 4)  # Integer division ensures equal distribution.

        settings_list = (
                [("A0", "B0")] * N_each +
                [("A1", "B0")] * N_each +
                [("A0", "B1")] * N_each +
                [("A1", "B1")] * N_each
        )
        # Randomize the order of settings to avoid systematic bias.
        random.shuffle(settings_list)
        a_setting, b_setting = [], []
        for (op_a, op_b) in settings_list:
            a_setting.append(op_a)
            b_setting.append(op_b)
        # --- Setup for Alice and Bob ---
        # generate ghz_settings for a and b
        length = len(settings_list)
        ghz_list = random.choices([("A0", "B0"), ("A1", "B1")], k=length)
        a_ghz, b_ghz = [], []
        for (opghz_a, opghz_b) in ghz_list:
            a_ghz.append(opghz_a)
            b_ghz.append(opghz_b)
        ghz_a = GHZProtocol(
            node=self.all_nodes[0],
            entangle_node=self.all_nodes[1].name,
            name=f"ghz_{{0}}->{{1}}".format(self.all_nodes[0].name, self.all_nodes[1].name),
            total_pairs=self.max_entangle_pairs,
            qubit_ready_protocols=ehA_ghz,
            setting_list=a_ghz,
            cc_message_handler=self.subprotocols[f"message_handler_{self.all_nodes[0].name}"],
            sample_size_rate=sample_size_rate,
            alpha=alpah,
            delta=delta,
            logger=self.logger
        )
        self.add_subprotocol(ghz_a)
        genA_ghz.entanglement_handler = ehA_ghz
        # --- Setup for Bob ---
        self.add_subprotocol(MessageHandler(
            node=self.all_nodes[1],
            name=f"message_handler_{self.all_nodes[1].name}",
            cc_ports=self.get_cc_ports(self.all_nodes[1])
        ))

        genB_ghz = GenEntanglementConcurrent(
            total_pairs=self.max_entangle_pairs,
            entangle_node=self.all_nodes[0].name,
            node=self.all_nodes[1],
            name="genB_GHZ",
            is_source=False,
            logger=self.null_logger
        )
        self.add_subprotocol(genB_ghz)
        ehB_ghz = EntanglementHandlerConcurrent(
            node=self.all_nodes[1],
            name="ehB_GHZ",
            num_pairs=self.max_entangle_pairs,
            qubit_input_protocol=genB_ghz,
            cc_message_handler=self.subprotocols[f"message_handler_{self.all_nodes[1].name}"],
            entangle_node=self.all_nodes[0].name,
            memory_depolar_rate=memory_depolar_rate,
            node_distance=node_distance,
            is_top_layer=False,
            logger=self.null_logger
        )
        self.add_subprotocol(ehB_ghz)
        ghz_b = GHZProtocol(
            node=self.all_nodes[1],
            entangle_node=self.all_nodes[0].name,
            name=f"ghz_{{0}}->{{1}}".format(self.all_nodes[1].name, self.all_nodes[0].name),
            total_pairs=self.max_entangle_pairs,
            qubit_ready_protocols=ehB_ghz,
            setting_list=b_ghz,
            cc_message_handler=self.subprotocols[f"message_handler_{self.all_nodes[1].name}"],
            sample_size_rate=sample_size_rate,
            alpha=alpah,
            delta=delta,
            logger=self.logger
        )
        self.add_subprotocol(ghz_b)
        genB_ghz.entanglement_handler = ehB_ghz


    def run(self):
        """
        Main protocol run loop.
        Generates entanglement, performs measurements based on CHSH settings,
        computes correlations, and calculates the CHSH S parameter.
            """
        self.start_subprotocols()

        # Run the protocol for the specified number of cycles.
        for run_idx in range(self.num_runs):
            print(f"Run {run_idx + 1}/{self.num_runs} starting...")
            start_t = ns.sim_time()  # Record the simulation start time.

            # Wait until entanglement handler subprotocol signal completion.
            p2 = self.subprotocols[f"ghz_{self.all_nodes[0].name}->{self.all_nodes[1].name}"]
            yield self.await_signal(p2, MessageType.GHZ_FINISHED)

            end_t = ns.sim_time()  # Record the simulation end time.
            duration = end_t - start_t
            # self.logger.info(f"Entanglement complete in {duration} ns. Gathering results...", color="purple")
            print(f"Entanglement complete in {duration} ns. Gathering results...")
            # Retrieve the results from each entanglement handler; each result is a dictionary mapping memory positions to fidelities.
            results_ghz = p2.get_signal_result(MessageType.GHZ_FINISHED)
            print(f"Total Unused pair: {len(results_ghz['entangled_pairs'])}")

            mem_poses = results_ghz['entangled_pairs'].keys()
            all_fids = []
            all_tel_fids = []
            for pos in mem_poses:
                q_a = self.all_nodes[0].subcomponents[f"{self.all_nodes[1].name}_qmemory"].pop(pos,
                                                                                          skip_noise=self.skip_noise)[0]
                q_b = self.all_nodes[1].subcomponents[f"{self.all_nodes[0].name}_qmemory"].pop(pos,
                                                                                                  skip_noise=self.skip_noise)[0]
                q_a_name = str(q_a.name).split("#")[-1].split("-")[0]
                q_b_name = str(q_b.name).split("#")[-1].split("-")[0]
                if q_a_name != q_b_name:
                    raise ValueError(f"Qubit names are not the same at {pos}:\n"
                                     f"\t{q_a_name}, {q_b_name}\n"
                                     f"\t{q_a.qstate}, {q_b.qstate}\n"
                                     f"\t{qapi.fidelity([q_a, q_b], ks.b00)}")
                if q_a.qstate != q_b.qstate:
                    raise ValueError(f"Qubit states are not the same: {q_a.qstate}, {q_b.qstate}")
                fid = qapi.fidelity([q_a, q_b], ks.b00)
                all_fids.append(fid)

                qubit = qapi.create_qubits(1)[0]
                qapi.operate(qubit, ops.H)
                qapi.operate(qubit, ops.S)
                # teleport the qubit
                tel_fid = self.test_teleportation(q_a, q_b, qubit)
                all_tel_fids.append(tel_fid)
            return_data = {
                "p_ghz": results_ghz["s_value"],
                "theta": results_ghz["theta"],
                "actual_fid": all_fids,
                "teleport_fid": all_tel_fids,
            }

            self.send_signal(Signals.SUCCESS, {"results": return_data,
                                               "run_index": run_idx})
            for subprotocol in self.subprotocols.values():
                subprotocol.reset()

    def get_cc_ports(self, node):
        """
        Return a dictionary that maps the name of the other node to its corresponding classical connection port.

        This enables classical communication between the nodes.
        """
        cc_ports = {}
        for n in self.all_nodes:
            if n != node:
                cc_ports[n.name] = node.get_conn_port(n.ID)
        return cc_ports
    @staticmethod
    def test_teleportation(qubit_a, qubit_b, teleport_qubit):
        """
        Test teleportation with two qubits
        :return:
        """
        # Store the initial state
        initial_state = teleport_qubit.qstate

        # Perform teleportation
        qapi.operate(qubits=[teleport_qubit, qubit_a], operator=ops.CNOT)
        qapi.operate(teleport_qubit, ops.H)
        m1, _ = qapi.measure(teleport_qubit)
        m2, _ = qapi.measure(qubit_a)
        if m1 == 1:
            qapi.operate(qubit_b, ops.Z)
        if m2 == 1:
            qapi.operate(qubit_b, ops.X)

        # Calculate fidelity
        # teleported_state = qapi.reduced_dm(qubit_b)
        # fidelity = qapi.fidelity(teleported_state, initial_state)
        fidelity = qapi.fidelity(qubit_b, ns.y0)

        return fidelity

def example_sim_run(nodes, num_runs,
                    memory_depolar_rate,
                    node_distance,
                    max_entangle_pairs,
                    sample_size_rate,
                    alpha,
                    delta=0.1,
                    skip_noise=False):
    """Example simulation setup for purification protocols.

    Returns
    -------
    :class:`~netsquid.examples.purify.FilteringExample`
        Example protocol to run.
    :class:`pandas.DataFrame`
        Dataframe of collected data.

    """
    chsh_example = ExampleCHSH(network_nodes=nodes,
                                 num_runs=num_runs,
                                 max_entangle_pairs=max_entangle_pairs,
                                 node_distance=node_distance,
                                 memory_depolar_rate=memory_depolar_rate,
                                 sample_size_rate=sample_size_rate,
                                 alpah=alpha,
                                 delta=delta,
                                 skip_noise=skip_noise)


    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=chsh_example,
                                     event_type=Signals.SUCCESS.value))
    return chsh_example, dc

def run_test_experiment():

    parser = argparse.ArgumentParser(description='CHSH experiment')
    parser.add_argument('--sample_rate', type=float, default=0.1,)
    parser.add_argument('--alpha', type=float, default=0.05,)
    parser.add_argument('--channel_rate', type=float, default=8000,)
    parser.add_argument('--memory_rate', type=int, default=0)
    parser.add_argument('--distance', type=float, default=0.5,)
    parser.add_argument('--runs', type=int, default=200,)
    parser.add_argument('--total_pair', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default="./results")

    args = parser.parse_args()


    sample_rate = args.sample_rate
    alpha = args.alpha
    channel_rate = args.channel_rate
    memory_rate = args.memory_rate
    distance = args.distance
    runs = args.runs
    total_pairs = args.total_pair
    save_dir = args.save_dir

    # test
    # sample_rate = 0.3
    # alpha = 0.05
    # channel_rate = 8000
    # memory_rate = 0
    # distance = 0.5
    # runs = 10
    # total_pairs = 1000


    network = setup_network_parallel_chsh(
        nodes_list=["Alice", "Bob"],
        network_name="AliceBobCHSH",
        memory_capacity=total_pairs,  # Use the N_4 variable for memory capacity.
        memory_depolar_rate=memory_rate,
        channel_depolar_rate=channel_rate,
        node_distance=distance  # Use the L variable for node distance.
    )

    sample_nodes = [node for node in network.nodes.values()]
    chsh_example, dc = example_sim_run(nodes=sample_nodes,
                                       num_runs=runs,
                                       memory_depolar_rate=memory_rate,
                                       node_distance=distance,
                                       max_entangle_pairs=total_pairs,
                                       sample_size_rate=sample_rate,
                                       alpha=alpha,
                                       delta=0.1,
                                       skip_noise=False)

    # ex1. fixed C=80,000, channel depolar rate = 8xxxhz, alaph 0.05, distance 1km
    # sample rate 0.1 -> 0.7, measure the 1 - sample rate fid, and teleport.
    # erros plot

    # ex2  fixed C=80,000, channel depolar rate = 8xxxhz, alaph 0.05, sample rate = TBD on experiment1
    # distance 0.5km -> 4km ? measure the 1 - sample rate fid, and teleport.

    # ex3 fixed C=80,000,  alaph 0.05, sample rate = TBD on experiment1, distance 1km
    # channel depolar rate 1000 -> 16000 hz, measure the 1 - sample rate fid, and teleport.

    # ex4 fixed C=80,000, channel depolar rate = 8xxxhz, alaph 0.05,
    # distance 0.5km -> 3km ? sample rate increasing based on distance, measure the 1 - sample rate fid, and teleport.

    # ex5. fixed C=80,000, channel depolar rate = 8xxxhz, distance 1km, sample_rate = TBD
    # alaph 0.20 -> 0.05


    chsh_example.start()
    ns.sim_run()

    print(dc.dataframe)
    os.makedirs(save_dir, exist_ok=True)
    dc.dataframe.to_json(os.path.join(save_dir, f"chsh_total_pairs-{total_pairs}_memory_{memory_rate}_channel_"
                                                f"{channel_rate}_distance_{distance}_sample_{sample_rate}_alpha_{alpha}.json"))

if __name__ == '__main__':
    run_test_experiment()