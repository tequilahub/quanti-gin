import matplotlib.pyplot as plt
import pandas as pd
from tequila.quantumchemistry import QuantumChemistryBase


def benchmarking_data_visualize_matplotlib(
    file_name,
    methods_to_plot=None,
    show_first_n_molecules=60,
    base_spacing=0.25,
    radius=1,
    radius_increase=0.25,
):
    """
    Visualize benchmarking results from CSV files.

    Loads benchmarking output and compares heuristic performance on molecular
    matching problems. Supports random, linear ("line"), and cyclic ("ring")
    datasets inferred from filename.

    Parameters
    ----------
    file_name : str
        Path to CSV file.

        Required columns:
        method, energy, runtime, ground state energy, energy gab

    methods_to_plot : list[str], optional
        Methods to include. If None, all are used.

    show_first_n_molecules : int, default=60
        Number of molecules in zoomed plots.

    base_spacing : float, default=0.25
        Atom spacing for linear datasets.

    radius : float, default=1
        Initial radius for ring datasets.

    radius_increase : float, default=0.25
        Radius increment per molecule.

    Returns
    -------
    None
        Produces matplotlib figures.

    Plots
    -----
    1. Energy distribution (boxplot)
    2. Runtime distribution (boxplot)
    3. Energy spread (max-min per molecule)
    4. Energy gap vs ground state
    5. Method energies vs ground state
    6. Zoomed comparison (first N molecules)
    7. Blossom vs 2-opt vs ground state (focused view)
    """

    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16

    # Load dataset
    data = pd.read_csv(file_name).dropna()

    methods = sorted(data["method"].unique())

    # Allow filtering
    if methods_to_plot is not None:
        data = data[data["method"].isin(methods_to_plot)]
        methods = methods_to_plot

    # 1 — ENERGY DISTRIBUTION
    fig, ax = plt.subplots()
    grouped = [data[data["method"] == m]["energy"] for m in methods]

    bp = ax.boxplot(
        grouped,
        labels=methods,
        showmeans=True,
        meanline=True,
        medianprops=dict(color="black"),
        meanprops=dict(color="red"),
        widths=0.6,
    )

    median_line = bp["medians"][0]
    mean_line = bp["means"][0]

    custom_lines = [
        plt.Line2D([], [], color=median_line.get_color(), label="Median"),
        plt.Line2D(
            [], [], color=mean_line.get_color(), linestyle="--", label="Mean (Average)"
        ),
    ]

    if "ring" in file_name:
        ax.legend(handles=custom_lines, loc="lower right")
    else:
        ax.legend(handles=custom_lines, loc="upper right")

    ax.set_title("Energy Distribution per Method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Energy (Eh)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2 — RUNTIME DISTRIBUTION
    fig, ax = plt.subplots()
    grouped = [data[data["method"] == m]["runtime"] for m in methods]

    bp = ax.boxplot(
        grouped,
        labels=methods,
        showmeans=True,
        meanline=True,
        medianprops=dict(color="black"),
        meanprops=dict(color="blue"),
        widths=0.6,
    )

    median_line = bp["medians"][0]
    mean_line = bp["means"][0]

    custom_lines = [
        plt.Line2D([], [], color=median_line.get_color(), label="Median"),
        plt.Line2D(
            [], [], color=mean_line.get_color(), linestyle="--", label="Mean (Average)"
        ),
    ]

    ax.legend(handles=custom_lines, loc="upper right")

    ax.set_title("Runtime Distribution per Method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Runtime (s)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3 — ENERGY SPREAD (min–max)
    data["mol_id"] = data.index // len(methods)

    spread = data.groupby("mol_id")["energy"].agg(["min", "max"]).reset_index()
    spread["spread"] = spread["max"] - spread["min"]

    if "line" in file_name:

        base_spacing = base_spacing + 0.05
        x_position = base_spacing + 0.05 * spread["mol_id"]

        fig, ax = plt.subplots()
        ax.plot(
            x_position,
            spread["spread"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color="tab:blue",
        )

        ax.set_title("Energy Gap between Best and Worst Calculated Energy(Eh)")
        ax.set_xlabel("Distance between the Atoms in Angstrom")
        ax.set_ylabel("Energy Gap (Eh)")
        plt.tight_layout()
        plt.show()

    elif "ring" in file_name:

        x_position = radius + radius_increase * spread["mol_id"]

        fig, ax = plt.subplots()
        ax.plot(
            x_position,
            spread["spread"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color="tab:blue",
        )

        ax.set_title("Energy Gap Between Best and Worst Calculated Energy(Eh)")
        ax.set_xlabel("Radius in Angstrom")
        ax.set_ylabel("Energy Gap (Eh)")
        plt.tight_layout()
        plt.show()

    else:

        fig, ax = plt.subplots()
        ax.plot(
            spread["mol_id"],
            spread["spread"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color="tab:blue",
        )

        ax.set_title("Energy Gap Between Best and Worst Calculated Energy(Eh)")
        ax.set_xlabel("Molecule ID")
        ax.set_ylabel("Energy Gap (Eh)")
        plt.tight_layout()
        plt.show()

    # 4 — ENERGY DIFFERENCE VS GROUND STATE
    fig, ax = plt.subplots()
    grouped = [data[data["method"] == m]["energy gab"] for m in methods]

    bp = ax.boxplot(
        grouped,
        labels=methods,
        showmeans=True,
        meanline=True,
        medianprops=dict(color="black"),
        meanprops=dict(color="green"),
        widths=0.6,
    )

    median_line = bp["medians"][0]
    mean_line = bp["means"][0]

    custom_lines = [
        plt.Line2D([], [], color=median_line.get_color(), label="Median"),
        plt.Line2D(
            [], [], color=mean_line.get_color(), linestyle="--", label="Mean (Average)"
        ),
    ]

    ax.legend(handles=custom_lines, loc="upper right")

    ax.set_title("Energy Gap (EH) to Ground State Energy(Eh)")
    ax.set_xlabel("Method")
    ax.set_ylabel("Energy (Eh) – Ground State(Eh)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 5 — METHOD ENERGIES VS GROUND STATE
    if methods_to_plot is None:

        methods_to_plot = [
            "blossom",
            "nearest_insertion",
            "nearest_neighbour",
            "simulated annealing",
            "2-opt",
            "genetic algorithm",
            "brute force",
            "blossom scaled",
        ]

    methods_to_linestyle = {
        "blossom": "dotted",
        "nearest_insertion": "dashed",
        "nearest_neighbour": ":",
        "simulated annealing": "dashdot",
        "2-opt": "--",
        "genetic algorithm": "-.",
        "brute force": "solid",
        "blossom scaled": "dashdot",
    }
    fig, ax = plt.subplots(figsize=(10, 6))

    if "line" in file_name:

        x_data = base_spacing + 0.05 * data["mol_id"]

        ax.plot(
            x_data,
            data["ground state energy"],
            color="black",
            linewidth=2,
            label="Ground State",
        )

        for method in methods_to_plot:
            linestyle = methods_to_linestyle[method]
            df = data[data["method"] == method].sort_values("mol_id")
            x_method = base_spacing + 0.05 * df["mol_id"]
            ax.plot(
                x_method,
                df["energy"],
                linewidth=1.5,
                alpha=0.8,
                linestyle=linestyle,
                label=method,
            )

        ax.set_title("Method Energies (Eh) vs Ground State Energy (Eh)")
        ax.set_xlabel("Distance between the Atoms in Angstrom")
        ax.set_ylabel("Energy (Eh)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    elif "ring" in file_name:

        x_data = radius + radius_increase * data["mol_id"]

        ax.plot(
            x_data,
            data["ground state energy"],
            color="black",
            linewidth=2,
            label="Ground State",
        )

        for method in methods_to_plot:
            linestyle = methods_to_linestyle[method]
            df = data[data["method"] == method].sort_values("mol_id")
            x_method = radius + radius_increase * df["mol_id"]
            ax.plot(
                x_method,
                df["energy"],
                linewidth=1.5,
                alpha=0.8,
                linestyle=linestyle,
                label=method,
            )

        ax.set_title("Method Energies (Eh) vs Ground State Energy (Eh)")
        ax.set_xlabel("Radius in Angstrom")
        ax.set_ylabel("Energy (Eh)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    else:
        ax.plot(
            data["mol_id"],
            data["ground state energy"],
            color="black",
            linewidth=2,
            label="Ground State",
        )

        for method in methods_to_plot:
            linestyle = methods_to_linestyle[method]
            df = data[data["method"] == method].sort_values("mol_id")
            ax.plot(
                df["mol_id"],
                df["energy"],
                linewidth=1.5,
                alpha=0.8,
                linestyle=linestyle,
                label=method,
            )

        ax.set_title("Method Energies (Eh) vs Ground State Energy (Eh)")
        ax.set_xlabel("Molecule ID")
        ax.set_ylabel("Energy (Eh)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    # 6 — ZOOMED FIRST N MOLECULES
    methods_to_linestyle = {
        "blossom": "dotted",
        "nearest_insertion": "dashed",
        "nearest_neighbour": ":",
        "simulated annealing": "dashdot",
        "2-opt": "--",
        "genetic algorithm": "-.",
        "brute force": "solid",
        "blossom scaled": "dashdot",
    }

    filtered = data[data["mol_id"] < show_first_n_molecules]

    fig, ax = plt.subplots(figsize=(10, 6))

    if "line" in file_name:
        x_data = base_spacing + 0.05 * filtered["mol_id"]
        x_label = "Distance between the Atoms in Angstrom"

    elif "ring" in file_name:
        x_data = radius + radius_increase * filtered["mol_id"]
        x_label = "Radius in Angstrom"

    else:
        x_data = filtered["mol_id"]
        x_label = "Molecule ID"

    ax.plot(
        x_data,
        filtered["ground state energy"],
        color="black",
        linewidth=2,
        label="Ground State",
    )

    for method in methods_to_plot:
        linestyle = methods_to_linestyle[method]
        df = filtered[filtered["method"] == method].sort_values("mol_id")

        if "line" in file_name:
            x_method = base_spacing + 0.05 * df["mol_id"]
        elif "ring" in file_name:
            x_method = radius + radius_increase * df["mol_id"]
        else:
            x_method = df["mol_id"]

        ax.plot(
            x_method,
            df["energy"],
            linewidth=1.5,
            alpha=0.8,
            linestyle=linestyle,
            label=method,
        )
    if "line" in file_name:
        ax.set_ylim(top=-1, bottom=filtered["ground state energy"].min() * 1.05)

    ax.set_title(f"Energy Comparison for First {show_first_n_molecules} Molecules")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Energy (Eh)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # 7 — ZOOMED FIRST N MOLECULES BLOSSOM VS 2-OPT

    fig, ax = plt.subplots(figsize=(10, 6))
    if "line" in file_name:
        x_data = base_spacing + 0.05 * filtered["mol_id"]
        x_label = "Distance between the Atoms in Angstrom"
    elif "ring" in file_name:
        x_data = radius + radius_increase * filtered["mol_id"]
        x_label = "Radius in Angstrom"
    else:
        x_data = filtered["mol_id"]
        x_label = "Molecule ID"
    # Ground state
    ax.plot(
        x_data,
        filtered["ground state energy"],
        color="black",
        linewidth=2,
        linestyle="-",
        label="Ground State",
    )
    ax.scatter(x_data, filtered["ground state energy"], color="black", s=30)

    # 2-opt
    df_2opt = filtered[filtered["method"] == "2-opt"]

    if "line" in file_name:
        x_2opt = base_spacing + 0.05 * df_2opt["mol_id"]

    elif "ring" in file_name:
        x_2opt = radius + radius_increase * df_2opt["mol_id"]

    else:
        x_2opt = df_2opt["mol_id"]

    ax.plot(
        x_2opt,
        df["energy"],
        color="purple",
        linewidth=1.5,
        linestyle="--",
        label="2-opt",
    )
    ax.scatter(x_2opt, df["energy"], color="purple", s=30)

    # Blossom
    df_blossom = filtered[filtered["method"] == "blossom"]

    if "line" in file_name:
        x_blossom = base_spacing + 0.05 * df_blossom["mol_id"]

    elif "ring" in file_name:
        x_blossom = radius + radius_increase * df_blossom["mol_id"]

    else:
        x_blossom = df_blossom["mol_id"]

    ax.plot(
        x_blossom,
        df["energy"],
        color="skyblue",
        linewidth=1.5,
        linestyle=":",
        label="blossom",
    )

    df_blossom_scaled = filtered[filtered["method"] == "blossom scaled"]

    if "line" in file_name:
        x_blossom_scaled = base_spacing + 0.05 * df_blossom_scaled["mol_id"]

    elif "ring" in file_name:
        x_blossom_scaled = radius + radius_increase * df_blossom_scaled["mol_id"]

    else:
        x_blossom_scaled = df_blossom_scaled["mol_id"]

    ax.plot(
        x_blossom_scaled,
        df["energy"],
        color="red",
        linewidth=1.5,
        linestyle=":",
        label="blossom scaled",
    )
    ax.scatter(x_blossom_scaled, df["energy"], color="red", s=30)

    ax.set_title(f"Energy Comparison for First {show_first_n_molecules} Molecules")
    ax.set_xlabel(x_label)
    if "line" in file_name:
        ax.set_ylim(top=-1, bottom=filtered["ground state energy"].min() * 1.05)

    ax.set_ylabel("Energy (Eh)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
