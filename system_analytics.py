# system_analytics.py
import json
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[INFO] matplotlib not available, plotting functions disabled")

# Optional pandas import
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[INFO] pandas not available, some analytics functions limited")

# Assuming user_memory.py is in the same directory or accessible in PYTHONPATH
import user_memory as UM_UserMemory # To use its load_user_memory function

# Define default paths (can be overridden by passing arguments to functions)
TRAIL_LOG_PATH_SA = Path("data/trail_log.json")
OCCURRENCE_LOG_PATH_SA = Path("data/symbol_occurrence_log.json")
CURRICULUM_METRICS_PATH_SA = Path("data/curriculum_metrics.json")
DATA_DIR_SA = Path("data") # Ensure this exists if functions create files here

def plot_node_activation_timeline(trail_log_path=TRAIL_LOG_PATH_SA):
    """
    Plots the cumulative count of chunks processed with a focus on
    Logic, Symbolic, or Ambiguous content types over time.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available. Cannot create plots.")
        return
        
    DATA_DIR_SA.mkdir(parents=True, exist_ok=True) # Ensure data directory exists
    if not trail_log_path.exists():
        print(f"❌ Trail log not found at {trail_log_path} for node activation plot.")
        return

    log_entries = []
    try:
        with open(trail_log_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
            # Assuming log is a list of entries directly (as per trail_log.py's _save_log)
            if isinstance(loaded_data, list):
                log_entries = [
                    entry for entry in loaded_data 
                    if "timestamp" in entry and "content_type_heuristic" in entry and "log_id" in entry and entry["log_id"].startswith("step_")
                ] # Filter for dynamic bridge entries
            else:
                print(f"❌ Trail log format at {trail_log_path} not recognized as a list of entries.")
                return
    except json.JSONDecodeError:
        print(f"❌ Error decoding trail log at {trail_log_path}.")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading trail log: {e}")
        return


    if not log_entries:
        print("🤷 No suitable DynamicBridge entries in trail log to plot node activation.")
        return

    log_entries.sort(key=lambda x: x["timestamp"])

    timestamps = []
    factual_counts_cumulative = []
    symbolic_counts_cumulative = []
    ambiguous_counts_cumulative = []

    current_factual = 0
    current_symbolic = 0
    current_ambiguous = 0

    for entry in log_entries:
        try:
            timestamps.append(datetime.fromisoformat(entry["timestamp"]))
            content_type = entry.get("content_type_heuristic", "ambiguous")
            if content_type == "factual":
                current_factual += 1
            elif content_type == "symbolic":
                current_symbolic += 1
            else: # ambiguous
                current_ambiguous += 1
            
            factual_counts_cumulative.append(current_factual)
            symbolic_counts_cumulative.append(current_symbolic)
            ambiguous_counts_cumulative.append(current_ambiguous)
        except ValueError:
            # print(f"Skipping entry with invalid timestamp: {entry.get('log_id', 'Unknown ID')}")
            continue # Skip entries that cause errors
            
    if not timestamps:
        print("🤷 No valid timestamped entries found to plot node activation.")
        return

    plt.figure(figsize=(12, 7))
    plt.plot(timestamps, factual_counts_cumulative, label="Logic Node Focus (Factual)", marker='o', linestyle='-')
    plt.plot(timestamps, symbolic_counts_cumulative, label="Symbolic Node Focus (Symbolic)", marker='x', linestyle='-')
    plt.plot(timestamps, ambiguous_counts_cumulative, label="Bridge Focus (Ambiguous)", marker='s', linestyle='--')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    plt.gcf().autofmt_xdate() 

    plt.title("Node Activation Timeline (Cumulative Focus)", fontsize=16)
    plt.xlabel("Processing Time", fontsize=12)
    plt.ylabel("Cumulative Count of Chunks Processed", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"Error displaying plot: {e}. Plot might not be shown in this environment.")


def plot_symbol_popularity_timeline(occurrence_log_path=OCCURRENCE_LOG_PATH_SA, top_n_symbols=7):
    """Plot symbol popularity over time"""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available. Cannot create plots.")
        return
    """
    Plots the cumulative occurrences of the top N most frequent symbols over time.
    """
    DATA_DIR_SA.mkdir(parents=True, exist_ok=True)
    if not occurrence_log_path.exists():
        print(f"❌ Symbol occurrence log not found at {occurrence_log_path}.")
        return
    
    entries = UM_UserMemory.load_user_memory(file_path=occurrence_log_path)
    if not entries:
        print("🤷 No symbol occurrences to plot popularity.")
        return

    df_data = []
    for entry in entries:
        try:
            # Ensure timestamp and symbol keys exist
            if "timestamp" in entry and "symbol" in entry:
                df_data.append({
                    "timestamp": datetime.fromisoformat(entry["timestamp"]),
                    "symbol": entry["symbol"]
                })
        except ValueError:
            # print(f"Skipping symbol occurrence entry with invalid timestamp or missing data.")
            continue 

    if not df_data:
        print("🤷 No valid timestamped symbol occurrences found for popularity plot.")
        return
        
    df = pd.DataFrame(df_data)
    if df.empty:
        print("🤷 DataFrame from occurrences is empty. Cannot plot popularity.")
        return

    top_symbols = df['symbol'].value_counts().nlargest(top_n_symbols).index.tolist()
    if not top_symbols:
        print(f"🤷 No symbols found to plot popularity for (top_n_symbols={top_n_symbols}).")
        return

    plt.figure(figsize=(12, 7))
    
    for symbol_token in top_symbols:
        symbol_df = df[df['symbol'] == symbol_token].copy()
        if symbol_df.empty:
            continue
        symbol_df.sort_values(by='timestamp', inplace=True)
        symbol_df['cumulative_occurrences'] = range(1, len(symbol_df) + 1)
        plt.plot(symbol_df['timestamp'], symbol_df['cumulative_occurrences'], label=f"{symbol_token}", marker='.', linestyle='-')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    plt.gcf().autofmt_xdate()

    plt.title(f"Top {len(top_symbols)} Symbol Popularity Over Time (Cumulative Occurrences)", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Cumulative Occurrences", fontsize=12)
    plt.legend(title="Symbols")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"Error displaying plot: {e}. Plot might not be shown in this environment.")


def plot_curriculum_metrics(metrics_path=CURRICULUM_METRICS_PATH_SA):
    """Plot curriculum metrics"""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available. Cannot create plots.")
        return
    """
    Plots the collected metrics for each curriculum phase as a grouped bar chart
    and prints a table.
    """
    DATA_DIR_SA.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        print(f"❌ Curriculum metrics file not found at {metrics_path}.")
        print("   Run autonomous_learner.py to generate curriculum_metrics.json.")
        return

    phase_metrics = {}
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            phase_metrics_loaded = json.load(f)
            # Ensure keys are integers for sorting and consistent access
            for k, v in phase_metrics_loaded.items():
                try:
                    phase_metrics[int(k)] = v
                except ValueError:
                    print(f"Warning: Invalid phase key '{k}' in metrics file. Skipping.")
    except json.JSONDecodeError:
        print(f"❌ Error decoding curriculum metrics at {metrics_path}.")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading curriculum metrics: {e}")
        return
            
    if not phase_metrics:
        print("🤷 No curriculum metrics data to display.")
        return

    phases = sorted(phase_metrics.keys())
    if not phases:
        print("🤷 No phases found in metrics data.")
        return

    metrics_to_plot = ["relevant_chunks_processed", "urls_visited", "new_symbols_generated", "meta_symbols_bound", "chunks_processed"]
    
    plot_data = defaultdict(list)
    for p in phases:
        for metric in metrics_to_plot:
            plot_data[metric].append(phase_metrics.get(p, {}).get(metric, 0))

    num_metrics = len(metrics_to_plot)
    num_phases = len(phases)
    
    # Create positions for the bars on X-axis
    bar_positions = range(num_phases) # [0, 1, 2, ...] for each phase group
    bar_width = 0.8 / num_metrics # Width of each individual bar

    plt.figure(figsize=(14, 8))
    ax = plt.subplot(111)
    for i, metric in enumerate(metrics_to_plot):
        # Calculate offset for each metric's bar within a phase group
        offsets = [pos + i * bar_width for pos in bar_positions]
        ax.bar(offsets, plot_data[metric], bar_width, label=metric.replace("_", " ").title())

    ax.set_xlabel("Learning Phase", fontsize=12)
    ax.set_ylabel("Metric Counts", fontsize=12)
    ax.set_title("Curriculum Phase Metrics", fontsize=16)
    # Set x-ticks to be in the middle of each group of bars
    ax.set_xticks([pos + bar_width * (num_metrics - 1) / 2 for pos in bar_positions])
    ax.set_xticklabels([f"Phase {p}" for p in phases])
    ax.legend(loc='upper left', bbox_to_anchor=(1,1)) # Place legend outside
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    try:
        plt.show()
    except Exception as e:
        print(f"Error displaying plot: {e}. Plot might not be shown in this environment.")


    print("\n📊 Curriculum Metrics Table:")
    header_parts = ["Phase"] + [m.replace("_", " ").title() for m in metrics_to_plot]
    col_widths = [max(len(part), 10) for part in header_parts] # Min width 10
    
    header_str = "| " + " | ".join([f"{part:<{col_widths[idx]}}" for idx, part in enumerate(header_parts)]) + " |"
    print(header_str)
    separator_str = "|" + "|".join(["-" * (w + 2) for w in col_widths]) + "|"
    print(separator_str)
    
    for p in phases:
        row_parts = [str(p)] + [str(phase_metrics.get(p, {}).get(metric, 0)) for metric in metrics_to_plot]
        row_str = "| " + " | ".join([f"{part:<{col_widths[idx]}}" for idx, part in enumerate(row_parts)]) + " |"
        print(row_str)

if __name__ == "__main__":
    print("📊 Running System Analytics Plots...")
    
    # Ensure data directory exists for functions that might try to write to it if it's missing
    DATA_DIR_SA.mkdir(parents=True, exist_ok=True)

    print("\n--- 1. Node Activation Timeline ---")
    plot_node_activation_timeline()

    print("\n--- 2. Symbol Popularity Timeline ---")
    # Create a dummy symbol_occurrence_log.json if it doesn't exist for testing
    if not OCCURRENCE_LOG_PATH_SA.exists() or OCCURRENCE_LOG_PATH_SA.stat().st_size == 0:
        print(f"Creating dummy symbol occurrence log for testing: {OCCURRENCE_LOG_PATH_SA}")
        dummy_occurrences = {"entries": [
            {"timestamp": "2025-01-01T10:00:00", "symbol": "💡", "context_text": "text1", "emotion_in_context":"curiosity"},
            {"timestamp": "2025-01-01T10:05:00", "symbol": "🔥", "context_text": "text2", "emotion_in_context":"passion"},
            {"timestamp": "2025-01-01T10:10:00", "symbol": "💡", "context_text": "text3", "emotion_in_context":"excitement"},
            {"timestamp": "2025-01-01T10:15:00", "symbol": "💻", "context_text": "text4", "emotion_in_context":"focus"},
            {"timestamp": "2025-01-01T10:20:00", "symbol": "💡", "context_text": "text5", "emotion_in_context":"wonder"},
            {"timestamp": "2025-01-01T10:25:00", "symbol": "🔥", "context_text": "text6", "emotion_in_context":"anger"},
        ]}
        with open(OCCURRENCE_LOG_PATH_SA, "w", encoding="utf-8") as f:
            json.dump(dummy_occurrences, f, indent=2)
    plot_symbol_popularity_timeline()

    print("\n--- 3. Curriculum Metrics ---")
    # Create a dummy curriculum_metrics.json if it doesn't exist for testing
    if not CURRICULUM_METRICS_PATH_SA.exists() or CURRICULUM_METRICS_PATH_SA.stat().st_size == 0:
        print(f"Creating dummy curriculum metrics for testing: {CURRICULUM_METRICS_PATH_SA}")
        dummy_metrics = {
            1: {"relevant_chunks_processed": 25, "urls_visited": 12, "new_symbols_generated": 2, "meta_symbols_bound": 0, "chunks_processed": 100},
            2: {"relevant_chunks_processed": 18, "urls_visited": 8, "new_symbols_generated": 5, "meta_symbols_bound": 1, "chunks_processed": 80}
        }
        with open(CURRICULUM_METRICS_PATH_SA, "w", encoding="utf-8") as f:
            json.dump(dummy_metrics, f, indent=2)
    plot_curriculum_metrics()
    
    print("\n✅ System Analytics Plots test run complete.")